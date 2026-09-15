// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration with the dedicated morsel scan builder.

use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::channel::oneshot;
use futures::future::BoxFuture;
use futures::future::try_join_all;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::dynamic::DynamicComparison;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::LayoutReaderContext;
use vortex_layout::LayoutReaderRef;
use vortex_layout::LayoutRef;
use vortex_layout::layouts::row_idx::RowIdx;
use vortex_layout::segments::SegmentSource;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use crate::MorselScan;
use crate::build::ExecPlan;
use crate::build::build_plan_with_row_offset;
use crate::driver::StreamCancellation;
use crate::driver::morsels;
use crate::io::IoService;
use crate::nodes::ConjunctMode;
use crate::source::SegmentSourceDriver;
use crate::stats::ScanStats;
use crate::stats::scan_diagnostics_enabled;

type PlanCacheKey = (Expression, Option<Expression>, ConjunctMode, u64);
/// One independently awaitable output unit returned by [`PushMorselScanExecutor`].
pub type MorselOutputTask = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;
/// Resolves to the final executor counters after an internally driven scan finishes.
pub type ScanStatsCompletion = BoxFuture<'static, VortexResult<ScanStats>>;
#[derive(Clone, Copy)]
struct ScanShape {
    morsels_in_row_range: u64,
    morsels_after_selection: u64,
    ranges_after_selection: u64,
    morsels_after_limit: u64,
    ranges_after_limit: u64,
}

impl ScanShape {
    fn apply(self, stats: &mut ScanStats, morsels_after_pruning: u64, ranges_after_pruning: u64) {
        stats.morsels_in_row_range = self.morsels_in_row_range;
        stats.morsels_after_selection = self.morsels_after_selection;
        stats.ranges_after_selection = self.ranges_after_selection;
        stats.morsels_after_limit = self.morsels_after_limit;
        stats.ranges_after_limit = self.ranges_after_limit;
        stats.morsels_after_pruning = morsels_after_pruning;
        stats.ranges_after_pruning = ranges_after_pruning;
    }
}

/// Morsels kept visible to background I/O ahead of the active workers in shared scans, so the
/// file driver sees enough adjacent segments to coalesce and cold reads overlap execution.
const SHARED_LOOKAHEAD_MORSELS: usize = 16;

/// Production SQL defaults for grouped-I/O frontier scheduling. Resident morsels expose their
/// own first frontier with no down lookahead. Up to two later predicate groups are admitted to
/// keep storage busy, while projection remains demand-gated. The controlled external bundling
/// experiment overrides only the per-thread down lookahead for its paired morsels.
const FRONTIER_LOOKAHEAD_PER_THREAD: usize = 0;
const FRONTIER_SPECULATIVE_PREDICATES_RIGHT: usize = 2;
const FRONTIER_REFILL_RANGES: usize = 32;

// This experiment admits projection I/O within the existing right bound. Keep it opt-in because
// selective scans can pay for reads that predicate-only admission would avoid.
static EXTERNAL_PROJECTION_PREFETCH: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("VORTEX_MORSEL_FRONTIER_PROJECTION_PREFETCH").as_deref() == Ok("1")
});

/// Adjacent natural morsels owned by one externally driven PushFrontier scheduler.
///
/// Two ranges are enough to turn independently timed predicate requests into one storage-visible
/// wave while keeping work stealing and retained output bounded tightly per engine thread.
const EXTERNAL_FRONTIER_BUNDLE_MORSELS: usize = 2;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum ExecutorIoPolicy {
    #[default]
    EagerLookahead,
    Frontier,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExecutorDriver {
    Internal,
    External,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExecutorIoSettings {
    eager_lookahead: bool,
    lookahead_morsels: usize,
    frontier_lookahead_per_thread: Option<usize>,
    speculative_predicate_frontiers: usize,
    frontier_refill_ranges: usize,
}

impl ExecutorIoPolicy {
    const fn from_frontier_io(frontier_io: bool) -> Self {
        if frontier_io {
            Self::Frontier
        } else {
            Self::EagerLookahead
        }
    }

    const fn settings(self, driver: ExecutorDriver) -> ExecutorIoSettings {
        match (self, driver) {
            (Self::EagerLookahead, ExecutorDriver::Internal) => ExecutorIoSettings {
                eager_lookahead: true,
                lookahead_morsels: SHARED_LOOKAHEAD_MORSELS,
                frontier_lookahead_per_thread: None,
                speculative_predicate_frontiers: 0,
                frontier_refill_ranges: FRONTIER_REFILL_RANGES,
            },
            (Self::EagerLookahead, ExecutorDriver::External) => ExecutorIoSettings {
                eager_lookahead: true,
                lookahead_morsels: 0,
                frontier_lookahead_per_thread: None,
                speculative_predicate_frontiers: 0,
                frontier_refill_ranges: FRONTIER_REFILL_RANGES,
            },
            (Self::Frontier, _) => ExecutorIoSettings {
                eager_lookahead: false,
                lookahead_morsels: 0,
                frontier_lookahead_per_thread: Some(FRONTIER_LOOKAHEAD_PER_THREAD),
                speculative_predicate_frontiers: FRONTIER_SPECULATIVE_PREDICATES_RIGHT,
                frontier_refill_ranges: FRONTIER_REFILL_RANGES,
            },
        }
    }

    const fn prefetch_projection(self, driver: ExecutorDriver, requested: bool) -> bool {
        matches!(
            (self, driver, requested),
            (Self::Frontier, ExecutorDriver::External, true)
        )
    }

    fn configure(self, scan: MorselScan, driver: ExecutorDriver) -> MorselScan {
        let settings = self.settings(driver);
        let scan = scan
            .with_lookahead_morsels(settings.lookahead_morsels)
            .with_eager_lookahead(settings.eager_lookahead);
        match settings.frontier_lookahead_per_thread {
            Some(frontiers) => {
                let scan = scan.with_frontier_lookahead_per_thread(frontiers);
                let scan = if self.prefetch_projection(driver, *EXTERNAL_PROJECTION_PREFETCH) {
                    scan.with_speculative_frontiers(settings.speculative_predicate_frontiers)
                } else {
                    scan.with_speculative_predicate_frontiers(
                        settings.speculative_predicate_frontiers,
                    )
                };
                scan.with_frontier_refill_ranges(settings.frontier_refill_ranges)
            }
            None => scan,
        }
    }
}

/// Keep stats evaluation bounded while exposing enough adjacent morsels for stats reads to batch.
const PRUNING_LOOKAHEAD_MORSELS: usize = 16;

/// Bridge small zone-map holes so requests from one physical segment remain visible together.
/// The scan's exact predicate still filters these false-positive rows.
const MAX_PRUNING_GAP_ROWS: u64 = 8 * 1024;

/// Push-morsel execution backend over a raw layout and segment source.
pub struct PushMorselScanExecutor {
    layout: LayoutRef,
    segments: Arc<dyn SegmentSource>,
    target_rows: u64,
    conjunct_mode: ConjunctMode,
    threads: usize,
    io_policy: ExecutorIoPolicy,
    external_driver: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    external_frontier_bundle_parallelism: Option<usize>,
    external_frontier_bundle_min_waves: usize,
    plan_cache: Mutex<HashMap<PlanCacheKey, Arc<ExecPlan>>>,
}

impl PushMorselScanExecutor {
    /// Create an executor over a raw layout and its segment source.
    pub fn new(layout: LayoutRef, segments: Arc<dyn SegmentSource>) -> Self {
        Self {
            layout,
            segments,
            target_rows: 128 * 1024,
            conjunct_mode: ConjunctMode::Cascade,
            threads: 4,
            io_policy: ExecutorIoPolicy::default(),
            external_driver: None,
            external_frontier_bundle_parallelism: None,
            external_frontier_bundle_min_waves: 1,
            plan_cache: Mutex::default(),
        }
    }

    /// Set the target number of rows per morsel.
    pub fn with_target_rows(mut self, target_rows: u64) -> Self {
        self.target_rows = target_rows;
        self
    }

    /// Set the conjunct evaluation policy.
    pub fn with_conjunct_mode(mut self, conjunct_mode: ConjunctMode) -> Self {
        self.conjunct_mode = conjunct_mode;
        self
    }

    /// Set the number of affinity workers used by one shared scan run.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// Select grouped-I/O frontier scheduling for this executor.
    ///
    /// The frontier production defaults admit at most two later predicate groups without
    /// speculating projection or adding down-frontier lookahead. Disabling it preserves the
    /// established eager-lookahead push policy.
    pub fn with_frontier_io(mut self, frontier_io: bool) -> Self {
        self.io_policy = ExecutorIoPolicy::from_frontier_io(frontier_io);
        self
    }

    /// Run each returned morsel future on the thread that polls it.
    pub fn with_external_threads(mut self, driver: Arc<dyn Fn() -> bool + Send + Sync>) -> Self {
        self.external_driver = Some(driver);
        self
    }

    /// Enable the experimental adjacent-morsel bundle for an externally driven frontier scan.
    #[doc(hidden)]
    pub fn with_external_frontier_bundle_parallelism(mut self, parallelism: usize) -> Self {
        self.external_frontier_bundle_parallelism = Some(parallelism.max(1));
        self
    }

    /// Require this many paired-output waves per external worker before bundling.
    #[doc(hidden)]
    pub fn with_external_frontier_bundle_min_waves(mut self, min_waves: usize) -> Self {
        self.external_frontier_bundle_min_waves = min_waves.max(1);
        self
    }

    /// Return the natural full-file row boundaries for this projection and filter.
    pub fn full_file_splits(
        &self,
        projection: &BoundExpression,
        filter: Option<&BoundExpression>,
    ) -> VortexResult<Vec<u64>> {
        let plan = self.plan(projection, filter, 0)?;
        let mut boundaries = Vec::with_capacity(plan.natural_splits().len() + 1);
        boundaries.push(0);
        boundaries.extend(
            plan.natural_splits()
                .iter()
                .copied()
                .filter(|boundary| *boundary != 0),
        );
        Ok(boundaries)
    }

    /// Build independently awaitable output tasks without constructing a layout reader.
    ///
    /// `limit` is honoured exactly on unfiltered scans: morsels past it are never read and the
    /// last one is capped. A filtered scan cannot know where the limit falls, so it returns every
    /// matching row and the caller trims. Dropping every returned future stops the scan.
    #[expect(clippy::too_many_arguments)]
    pub fn build(
        &self,
        session: VortexSession,
        projection: BoundExpression,
        filter: Option<BoundExpression>,
        row_range: Option<Range<u64>>,
        selection: vortex_scan::selection::Selection,
        limit: Option<u64>,
        row_offset: u64,
    ) -> VortexResult<Vec<MorselOutputTask>> {
        let (outputs, completion) = self.build_inner(
            session, projection, filter, row_range, selection, limit, row_offset, false,
        )?;
        debug_assert!(completion.is_none());
        Ok(outputs)
    }

    /// Build output tasks plus one future that resolves to final internal executor statistics.
    ///
    /// The completion future resolves only after the internally driven scan has finished. It must
    /// be retained and polled by the consumer; dropping all output tasks still cancels the scan.
    /// Externally driven scans have no single scan-wide completion point and are rejected.
    #[expect(clippy::too_many_arguments)]
    pub fn build_with_stats(
        &self,
        session: VortexSession,
        projection: BoundExpression,
        filter: Option<BoundExpression>,
        row_range: Option<Range<u64>>,
        selection: vortex_scan::selection::Selection,
        limit: Option<u64>,
        row_offset: u64,
    ) -> VortexResult<(Vec<MorselOutputTask>, ScanStatsCompletion)> {
        if self.external_driver.is_some() {
            return Err(vortex_err!(
                "scan-wide statistics are unavailable for externally driven scans"
            ));
        }
        let (outputs, completion) = self.build_inner(
            session, projection, filter, row_range, selection, limit, row_offset, true,
        )?;
        Ok((
            outputs,
            completion.ok_or_else(|| vortex_err!("missing scan statistics completion future"))?,
        ))
    }

    #[expect(clippy::too_many_arguments)]
    fn build_inner(
        &self,
        session: VortexSession,
        projection: BoundExpression,
        filter: Option<BoundExpression>,
        row_range: Option<Range<u64>>,
        selection: vortex_scan::selection::Selection,
        limit: Option<u64>,
        row_offset: u64,
        collect_stats: bool,
    ) -> VortexResult<(Vec<MorselOutputTask>, Option<ScanStatsCompletion>)> {
        if limit == Some(0) && !collect_stats {
            return Ok((Vec::new(), None));
        }

        let plan = self.plan(&projection, filter.as_ref(), row_offset)?;
        let full_range = row_range.unwrap_or_else(|| 0..plan.row_count());
        let natural_morsels = morsels(&plan, self.target_rows);
        let natural_morsel_count = natural_morsels.len();
        let morsels_in_row_range = if collect_stats {
            count_intersecting_morsels(&natural_morsels, &full_range)
        } else {
            0
        };
        let mut morsels = selected_morsels(natural_morsels, &full_range, &selection);
        let mut scan_shape = collect_stats.then(|| ScanShape {
            morsels_in_row_range,
            morsels_after_selection: count_nonempty_morsels(&morsels),
            ranges_after_selection: count_ranges(&morsels),
            morsels_after_limit: 0,
            ranges_after_limit: 0,
        });
        if limit == Some(0) {
            let completion = scan_shape.map(|shape| {
                let mut stats = ScanStats::default();
                shape.apply(&mut stats, 0, 0);
                Box::pin(async move { Ok(stats) }) as ScanStatsCompletion
            });
            return Ok((Vec::new(), completion));
        }
        // Without a filter every selected row is an output row, so the morsels past the limit
        // can be dropped before any I/O and the last one capped exactly. A filtered scan cannot
        // know where the limit falls.
        let mut row_caps = None;
        if let Some(limit) = limit
            && filter.is_none()
        {
            let mut remaining = limit;
            let mut caps = Vec::with_capacity(morsels.len());
            for morsel in &morsels {
                if remaining == 0 {
                    break;
                }
                let rows = morsel
                    .selected_ranges
                    .iter()
                    .map(|range| range.end - range.start)
                    .sum::<u64>();
                caps.push(usize::try_from(rows.min(remaining)).unwrap_or(usize::MAX));
                remaining = remaining.saturating_sub(rows);
            }
            morsels.truncate(caps.len());
            row_caps = Some(caps);
        }
        if let Some(shape) = &mut scan_shape {
            shape.morsels_after_limit = count_nonempty_morsels(&morsels);
            shape.ranges_after_limit = count_ranges(&morsels);
        }

        // Build a fresh pruning reader for this scan. Its zone-map state is shared only by the
        // morsels in this invocation; nothing survives into a later scan. Dynamic filters are
        // intentionally excluded because their bounds can change after this one-shot prepass.
        let pruner = filter
            .as_ref()
            .map(static_conjuncts)
            .transpose()?
            .filter(|conjuncts| !conjuncts.is_empty())
            .map(|conjuncts| {
                Ok::<_, vortex_error::VortexError>(StaticPruner {
                    reader: self.layout.new_reader(
                        "morsel-pruning".into(),
                        Arc::clone(&self.segments),
                        &session,
                        &LayoutReaderContext::new(),
                    )?,
                    conjuncts: conjuncts.into(),
                })
            })
            .transpose()?;

        if let Some(driver) = &self.external_driver {
            let bundle_parallelism = external_frontier_bundle_parallelism(
                self.external_frontier_bundle_parallelism,
                self.io_policy,
                filter.is_some(),
                limit.is_some(),
            );
            if let Some(parallelism) = bundle_parallelism {
                return build_external_frontier_bundle_outputs(
                    session,
                    plan,
                    Arc::clone(&self.segments),
                    morsels,
                    Arc::clone(driver),
                    pruner,
                    self.io_policy,
                    parallelism,
                    self.external_frontier_bundle_min_waves,
                    natural_morsel_count,
                )
                .map(|outputs| (outputs, None));
            }
            if scan_diagnostics_enabled() {
                let parallelism = self
                    .external_frontier_bundle_parallelism
                    .unwrap_or_default();
                let candidate_bundle_count =
                    morsels.len().div_ceil(EXTERNAL_FRONTIER_BUNDLE_MORSELS);
                let bundle_threshold =
                    parallelism.saturating_mul(self.external_frontier_bundle_min_waves);
                let fallback_reason = if self.external_frontier_bundle_parallelism.is_none() {
                    "not_configured"
                } else if self.io_policy != ExecutorIoPolicy::Frontier {
                    "non_frontier"
                } else if filter.is_none() {
                    "no_filter"
                } else if limit.is_some() {
                    "limit"
                } else {
                    "unknown"
                };
                tracing::trace!(
                    target: "vortex_morsel_push::external_frontier_bundle",
                    configured_parallelism = parallelism,
                    min_bundle_waves = self.external_frontier_bundle_min_waves,
                    bundle_threshold,
                    natural_morsels = natural_morsel_count,
                    selected_morsels = morsels.len(),
                    candidate_bundle_count,
                    selected_bundle_size = 1,
                    resulting_bundles = morsels.len(),
                    paired_bundles = 0,
                    fallback_reason,
                    "planning external frontier morsel bundles"
                );
            }
            return build_external_outputs(
                session,
                plan,
                Arc::clone(&self.segments),
                morsels,
                row_caps,
                Arc::clone(driver),
                pruner,
                self.io_policy,
            )
            .map(|outputs| (outputs, None));
        }

        let (mut stats_sender, stats_completion) = if collect_stats {
            let (sender, receiver) = oneshot::channel();
            let completion = Box::pin(async move {
                receiver
                    .await
                    .map_err(|_| vortex_err!("shared morsel scan coordinator stopped"))?
            }) as ScanStatsCompletion;
            (Some(sender), Some(completion))
        } else {
            (None, None)
        };

        // Each output future carries a guard; when the last guard drops, whether because its
        // future was consumed or discarded, there is nobody left to deliver to and the scan is
        // cancelled.
        let cancellation = StreamCancellation::new();
        let undelivered = Arc::new(AtomicUsize::new(morsels.len()));
        let mut senders = Vec::with_capacity(morsels.len());
        let mut outputs = Vec::with_capacity(morsels.len());
        for _ in 0..morsels.len() {
            let (sender, receiver) = oneshot::channel();
            senders.push(sender);
            let guard = DeliveryGuard {
                undelivered: Arc::clone(&undelivered),
                cancellation: Arc::clone(&cancellation),
            };
            outputs.push(Box::pin(async move {
                let _guard = guard;
                receiver
                    .await
                    .map_err(|_| vortex_err!("shared morsel scan coordinator stopped"))?
            })
                as BoxFuture<'static, VortexResult<Option<ArrayRef>>>);
        }

        let driver = SegmentSourceDriver::new(Arc::clone(&self.segments));
        let handle = session.handle();
        let coordinator_handle = handle.clone();
        let driver_handle = handle.clone();
        let max_threads = self.threads;
        let io_policy = self.io_policy;
        handle
            .spawn(async move {
                let morsels = match prune_morsels(pruner.as_ref(), morsels).await {
                    Ok(morsels) => morsels,
                    Err(err) => {
                        let message = err.to_string();
                        fail_senders(senders, &message);
                        if let Some(sender) = stats_sender.take() {
                            drop(sender.send(Err(vortex_err!(
                                "shared morsel scan pruning failed: {message}"
                            ))));
                        }
                        return;
                    }
                };
                let morsels_after_pruning = count_nonempty_morsels(&morsels);
                let ranges_after_pruning = count_ranges(&morsels);

                let mut ranges = Vec::new();
                let mut targets = Vec::new();
                let mut groups = Vec::with_capacity(morsels.len());
                for (morsel_index, (morsel, sender)) in morsels.into_iter().zip(senders).enumerate()
                {
                    if morsel.selected_ranges.is_empty() {
                        drop(sender.send(Ok(None)));
                        continue;
                    }
                    let group = Arc::new(OutputGroup::new(
                        morsel.selected_ranges.len(),
                        plan.output_dtype().clone(),
                        sender,
                        row_caps.as_ref().map(|caps| caps[morsel_index]),
                    ));
                    for (local_index, range) in morsel.selected_ranges.into_iter().enumerate() {
                        ranges.push(range);
                        targets.push(CompletionTarget {
                            group: Arc::clone(&group),
                            local_index,
                        });
                    }
                    groups.push(group);
                }

                if ranges.is_empty() {
                    if let (Some(sender), Some(scan_shape)) = (stats_sender.take(), scan_shape) {
                        let mut stats = ScanStats::default();
                        scan_shape.apply(&mut stats, morsels_after_pruning, ranges_after_pruning);
                        drop(sender.send(Ok(stats)));
                    }
                    return;
                }
                let threads = ranges.len().min(max_threads);
                let result = coordinator_handle
                    .spawn_blocking(move || {
                        let scan = MorselScan::new(plan, session)
                            .with_threads(threads)
                            .with_morsels(ranges)
                            .with_sparse_morsels(true)
                            .with_cancellation(cancellation)
                            .with_completion_sink(move |index, batch| {
                                targets[index].complete(batch);
                            });
                        let scan = io_policy.configure(scan, ExecutorDriver::Internal);
                        driver
                            .connect(scan, &driver_handle)?
                            .run()
                            .map(|(_, stats)| stats)
                    })
                    .await;
                match result {
                    Ok(mut stats) => {
                        if let (Some(sender), Some(scan_shape)) = (stats_sender.take(), scan_shape)
                        {
                            scan_shape.apply(
                                &mut stats,
                                morsels_after_pruning,
                                ranges_after_pruning,
                            );
                            drop(sender.send(Ok(stats)));
                        }
                    }
                    Err(err) => {
                        let message = err.to_string();
                        for group in groups {
                            group.fail(&message);
                        }
                        if let Some(sender) = stats_sender.take() {
                            drop(
                                sender
                                    .send(Err(vortex_err!("shared morsel scan failed: {message}"))),
                            );
                        }
                    }
                }
            })
            .detach();

        Ok((outputs, stats_completion))
    }

    fn plan(
        &self,
        projection: &BoundExpression,
        filter: Option<&BoundExpression>,
        row_offset: u64,
    ) -> VortexResult<Arc<ExecPlan>> {
        let projection = unbind(projection)?;
        let filter = filter.map(unbind).transpose()?;
        let plan_key = (
            projection.clone(),
            filter.clone(),
            self.conjunct_mode,
            row_offset,
        );
        let mut cache = self.plan_cache.lock();
        match cache.get(&plan_key) {
            Some(plan) => Ok(Arc::clone(plan)),
            None => {
                let plan = Arc::new(build_plan_with_row_offset(
                    &self.layout,
                    &projection,
                    filter.as_ref(),
                    self.conjunct_mode,
                    row_offset,
                )?);
                cache.insert(plan_key, Arc::clone(&plan));
                Ok(plan)
            }
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn build_external_outputs(
    session: VortexSession,
    plan: Arc<ExecPlan>,
    segments: Arc<dyn SegmentSource>,
    morsels: Vec<SelectedMorsel>,
    row_caps: Option<Vec<usize>>,
    driver: Arc<dyn Fn() -> bool + Send + Sync>,
    pruner: Option<StaticPruner>,
    io_policy: ExecutorIoPolicy,
) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<ArrayRef>>>>> {
    // One I/O service, and therefore one demand stream, spans every morsel of this file so
    // reads dedupe across them. The engine's threads advance the runtime the driver runs on.
    let source_id = segments.diagnostic_instance_id();
    let source = SegmentSourceDriver::new(segments);
    let (io, demand) = IoService::new();
    io.link_source(source_id);
    io.set_background_reads(source.prefers_background_reads());
    io.set_probe(Some(source.nowait_probe()));
    session
        .handle()
        .spawn(source.drive(demand, io.completions()))
        .detach();
    let mut outputs = Vec::with_capacity(morsels.len());
    for (morsel_index, morsel) in morsels.into_iter().enumerate() {
        let row_cap = row_caps.as_ref().map(|caps| caps[morsel_index]);
        let plan = Arc::clone(&plan);
        let io = Arc::clone(&io);
        let driver = Arc::clone(&driver);
        let session = session.clone();
        let pruner = pruner.clone();
        outputs.push(Box::pin(async move {
            let ranges = prune_morsel(pruner.as_ref(), morsel).await?.selected_ranges;
            if ranges.is_empty() {
                return Ok(None);
            }
            let scan = MorselScan::new_with_shared_io(plan, session, ranges, io)
                .with_external_driver(driver)
                .with_share_decodes(false)
                .with_sparse_morsels(true);
            let (batches, _) = io_policy
                .configure(scan, ExecutorDriver::External)
                .run_on_current_thread()?;
            match (combine_batches(batches)?, row_cap) {
                (Some(array), Some(cap)) if array.len() > cap => Ok(Some(array.slice(0..cap)?)),
                (batch, _) => Ok(batch),
            }
        })
            as BoxFuture<'static, VortexResult<Option<ArrayRef>>>);
    }
    Ok(outputs)
}

fn external_frontier_bundle_parallelism(
    configured_parallelism: Option<usize>,
    io_policy: ExecutorIoPolicy,
    has_filter: bool,
    has_limit: bool,
) -> Option<usize> {
    configured_parallelism
        .filter(|_| io_policy == ExecutorIoPolicy::Frontier && has_filter && !has_limit)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExternalFrontierBundlePlan {
    bundle_size: usize,
    candidate_bundle_count: usize,
    threshold: usize,
}

fn external_frontier_bundle_plan(
    morsel_count: usize,
    parallelism: usize,
    min_waves: usize,
) -> ExternalFrontierBundlePlan {
    let candidate_bundle_count = morsel_count.div_ceil(EXTERNAL_FRONTIER_BUNDLE_MORSELS);
    let threshold = parallelism.max(1).saturating_mul(min_waves.max(1));
    let bundle_size = if morsel_count >= EXTERNAL_FRONTIER_BUNDLE_MORSELS
        && candidate_bundle_count >= threshold
    {
        EXTERNAL_FRONTIER_BUNDLE_MORSELS
    } else {
        1
    };
    ExternalFrontierBundlePlan {
        bundle_size,
        candidate_bundle_count,
        threshold,
    }
}

struct ExternalFrontierBundle {
    coverage: Range<u64>,
    first: SelectedMorsel,
    second: Option<SelectedMorsel>,
}

impl ExternalFrontierBundle {
    fn into_morsels(self) -> Vec<SelectedMorsel> {
        let mut morsels = Vec::with_capacity(EXTERNAL_FRONTIER_BUNDLE_MORSELS);
        morsels.push(self.first);
        morsels.extend(self.second);
        morsels
    }
}

fn bundle_adjacent_contiguous_morsels(morsels: Vec<SelectedMorsel>) -> Vec<ExternalFrontierBundle> {
    let mut morsels = morsels.into_iter().peekable();
    let mut bundles = Vec::new();
    while let Some(first) = morsels.next() {
        let first_coverage = contiguous_selected_coverage(&first);
        let pair_next = first_coverage.as_ref().is_some_and(|first_coverage| {
            morsels.peek().is_some_and(|second| {
                contiguous_selected_coverage(second)
                    .is_some_and(|second_coverage| first_coverage.end == second_coverage.start)
            })
        });
        let second = if pair_next { morsels.next() } else { None };
        let coverage = match (
            &first_coverage,
            second.as_ref().and_then(contiguous_selected_coverage),
        ) {
            (Some(first), Some(second)) => first.start..second.end,
            (Some(first), None) => first.clone(),
            (None, _) => selected_coverage(&first),
        };
        bundles.push(ExternalFrontierBundle {
            coverage,
            first,
            second,
        });
    }
    bundles
}

fn contiguous_selected_coverage(morsel: &SelectedMorsel) -> Option<Range<u64>> {
    match morsel.selected_ranges.as_slice() {
        [range] => Some(range.clone()),
        _ => None,
    }
}

fn selected_coverage(morsel: &SelectedMorsel) -> Range<u64> {
    match (
        morsel.selected_ranges.first(),
        morsel.selected_ranges.last(),
    ) {
        (Some(first), Some(last)) => first.start..last.end,
        _ => 0..0,
    }
}

fn bundled_frontier_down(pruned: &[SelectedMorsel]) -> usize {
    let [first, second] = pruned else {
        return 0;
    };
    let ([first], [second]) = (
        first.selected_ranges.as_slice(),
        second.selected_ranges.as_slice(),
    ) else {
        return 0;
    };
    usize::from(first.end == second.start)
}

#[expect(clippy::too_many_arguments)]
fn build_external_frontier_bundle_outputs(
    session: VortexSession,
    plan: Arc<ExecPlan>,
    segments: Arc<dyn SegmentSource>,
    morsels: Vec<SelectedMorsel>,
    driver: Arc<dyn Fn() -> bool + Send + Sync>,
    pruner: Option<StaticPruner>,
    io_policy: ExecutorIoPolicy,
    parallelism: usize,
    min_bundle_waves: usize,
    natural_morsel_count: usize,
) -> VortexResult<Vec<BoxFuture<'static, VortexResult<Option<ArrayRef>>>>> {
    let bundle_plan = external_frontier_bundle_plan(morsels.len(), parallelism, min_bundle_waves);
    if bundle_plan.bundle_size == 1 {
        if scan_diagnostics_enabled() {
            tracing::trace!(
                target: "vortex_morsel_push::external_frontier_bundle",
                configured_parallelism = parallelism,
                min_bundle_waves,
                bundle_threshold = bundle_plan.threshold,
                natural_morsels = natural_morsel_count,
                selected_morsels = morsels.len(),
                candidate_bundle_count = bundle_plan.candidate_bundle_count,
                selected_bundle_size = 1,
                resulting_bundles = morsels.len(),
                paired_bundles = 0,
                fallback_reason = "min_bundle_waves",
                "planning external frontier morsel bundles"
            );
        }
        return build_external_outputs(
            session, plan, segments, morsels, None, driver, pruner, io_policy,
        );
    }

    let source_id = segments.diagnostic_instance_id();
    let source = SegmentSourceDriver::new(segments);
    let (io, demand) = IoService::new();
    io.link_source(source_id);
    io.set_background_reads(source.prefers_background_reads());
    io.set_probe(Some(source.nowait_probe()));
    session
        .handle()
        .spawn(source.drive(demand, io.completions()))
        .detach();
    let bundles = bundle_adjacent_contiguous_morsels(morsels);
    if scan_diagnostics_enabled() {
        let paired_bundles = bundles
            .iter()
            .filter(|bundle| bundle.second.is_some())
            .count();
        let fallback_reason = if paired_bundles == 0 {
            "no_adjacent_contiguous_pair"
        } else if paired_bundles < bundles.len() {
            "odd_tail_or_selection_gap"
        } else {
            "none"
        };
        tracing::trace!(
            target: "vortex_morsel_push::external_frontier_bundle",
            configured_parallelism = parallelism,
            min_bundle_waves,
            bundle_threshold = bundle_plan.threshold,
            natural_morsels = natural_morsel_count,
            selected_morsels = bundles.len() + paired_bundles,
            candidate_bundle_count = bundle_plan.candidate_bundle_count,
            selected_bundle_size = EXTERNAL_FRONTIER_BUNDLE_MORSELS,
            resulting_bundles = bundles.len(),
            paired_bundles,
            fallback_reason,
            "planning external frontier morsel bundles"
        );
    }
    let mut outputs = Vec::with_capacity(bundles.len());
    for bundle in bundles {
        debug_assert!(bundle.coverage.start <= bundle.coverage.end);
        let plan = Arc::clone(&plan);
        let io = Arc::clone(&io);
        let driver = Arc::clone(&driver);
        let session = session.clone();
        let pruner = pruner.clone();
        outputs.push(Box::pin(async move {
            let pruned = prune_morsels(pruner.as_ref(), bundle.into_morsels()).await?;
            let down = bundled_frontier_down(&pruned);
            let ranges = pruned
                .into_iter()
                .flat_map(|morsel| morsel.selected_ranges)
                .collect::<Vec<_>>();
            if ranges.is_empty() {
                return Ok(None);
            }
            let scan = MorselScan::new_with_shared_io(plan, session, ranges, io)
                .with_external_driver(driver)
                .with_share_decodes(false)
                .with_sparse_morsels(true);
            let scan = io_policy.configure(scan, ExecutorDriver::External);
            let scan = if down == 0 {
                scan
            } else {
                scan.with_frontier_lookahead_per_thread(down)
            };
            let (batches, _) = scan.run_on_current_thread()?;
            combine_batches(batches)
        })
            as BoxFuture<'static, VortexResult<Option<ArrayRef>>>);
    }
    Ok(outputs)
}

#[derive(Clone)]
struct StaticPruner {
    reader: LayoutReaderRef,
    conjuncts: Arc<[BoundExpression]>,
}

async fn prune_morsels(
    pruner: Option<&StaticPruner>,
    morsels: Vec<SelectedMorsel>,
) -> VortexResult<Vec<SelectedMorsel>> {
    let mut pruned = Vec::with_capacity(morsels.len());
    let mut morsels = morsels.into_iter();
    loop {
        let pending = morsels
            .by_ref()
            .take(PRUNING_LOOKAHEAD_MORSELS)
            .map(|morsel| prune_morsel(pruner, morsel))
            .collect::<Vec<_>>();
        if pending.is_empty() {
            return Ok(pruned);
        }
        pruned.extend(try_join_all(pending).await?);
    }
}

async fn prune_morsel(
    pruner: Option<&StaticPruner>,
    morsel: SelectedMorsel,
) -> VortexResult<SelectedMorsel> {
    let Some(pruner) = pruner else {
        return Ok(morsel);
    };

    let mut selected_ranges = Vec::new();
    for range in morsel.selected_ranges {
        let len = usize::try_from(range.end - range.start).unwrap_or(usize::MAX);
        // Construct every independent stats future before awaiting any of them. Besides avoiding
        // an artificial conjunct-by-conjunct dependency, this exposes all auxiliary segments to
        // the file source together so adjacent stats reads can coalesce just as they do in V1.
        let futures = pruner
            .conjuncts
            .iter()
            .map(|conjunct| {
                pruner
                    .reader
                    .pruning_evaluation(&range, conjunct, Mask::new_true(len))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        let mask = Mask::intersect_owned(try_join_all(futures).await?);
        let ranges = coalesce_ranges(mask_ranges(&range, &mask), MAX_PRUNING_GAP_ROWS);
        selected_ranges.extend(ranges);
    }
    Ok(SelectedMorsel { selected_ranges })
}

fn static_conjuncts(filter: &BoundExpression) -> VortexResult<Vec<BoundExpression>> {
    let mut conjuncts = Vec::new();
    let mut pending = vec![filter];
    while let Some(expr) = pending.pop() {
        let is_and = expr
            .as_scalar()
            .and_then(|scalar_fn| scalar_fn.as_opt::<Binary>())
            .is_some_and(|operator| *operator == Operator::And);
        if is_and {
            pending.extend(expr.children().iter().rev());
        } else if !expr.contains::<DynamicComparison>()? && !expr.contains::<RowIdx>()? {
            conjuncts.push(expr.clone());
        }
    }
    Ok(conjuncts)
}

fn fail_senders(senders: Vec<oneshot::Sender<VortexResult<Option<ArrayRef>>>>, message: &str) {
    for sender in senders {
        drop(sender.send(Err(vortex_err!(
            "shared morsel scan pruning failed: {message}"
        ))));
    }
}

fn combine_batches(mut batches: Vec<ArrayRef>) -> VortexResult<Option<ArrayRef>> {
    match batches.len() {
        0 => Ok(None),
        1 => Ok(batches.pop()),
        _ => {
            let dtype = batches[0].dtype().clone();
            ChunkedArray::try_new(batches, dtype).map(|array| Some(array.into_array()))
        }
    }
}

struct CompletionTarget {
    group: Arc<OutputGroup>,
    local_index: usize,
}

impl CompletionTarget {
    fn complete(&self, batch: VortexResult<Option<ArrayRef>>) {
        match batch {
            Ok(batch) => self.group.complete(self.local_index, batch),
            Err(err) => self.group.fail(&err.to_string()),
        }
    }
}

struct OutputGroup {
    remaining: AtomicUsize,
    dtype: DType,
    batches: Mutex<Vec<(usize, ArrayRef)>>,
    sender: Mutex<Option<oneshot::Sender<VortexResult<Option<ArrayRef>>>>>,
    /// Exact output rows for this morsel under an unfiltered limit.
    row_cap: Option<usize>,
}

impl OutputGroup {
    fn new(
        remaining: usize,
        dtype: DType,
        sender: oneshot::Sender<VortexResult<Option<ArrayRef>>>,
        row_cap: Option<usize>,
    ) -> Self {
        Self {
            remaining: AtomicUsize::new(remaining),
            dtype,
            batches: Mutex::new(Vec::new()),
            sender: Mutex::new(Some(sender)),
            row_cap,
        }
    }

    fn complete(&self, index: usize, batch: Option<ArrayRef>) {
        if let Some(batch) = batch {
            self.batches.lock().push((index, batch));
        }
        if self.remaining.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        let mut batches = std::mem::take(&mut *self.batches.lock());
        batches.sort_unstable_by_key(|(index, _)| *index);
        let result = match batches.len() {
            0 => Ok(None),
            1 => Ok(batches.pop().map(|(_, batch)| batch)),
            _ => ChunkedArray::try_new(
                batches.into_iter().map(|(_, batch)| batch),
                self.dtype.clone(),
            )
            .map(|array| Some(array.into_array())),
        };
        let result = match (result, self.row_cap) {
            (Ok(Some(array)), Some(cap)) if array.len() > cap => array.slice(0..cap).map(Some),
            (result, _) => result,
        };
        if let Some(sender) = self.sender.lock().take() {
            drop(sender.send(result));
        }
    }

    fn fail(&self, message: &str) {
        if let Some(sender) = self.sender.lock().take() {
            drop(sender.send(Err(vortex_err!("shared morsel scan failed: {message}"))));
        }
    }
}

/// Cancels the scan when the last output future is consumed or dropped.
struct DeliveryGuard {
    undelivered: Arc<AtomicUsize>,
    cancellation: Arc<StreamCancellation>,
}

impl Drop for DeliveryGuard {
    fn drop(&mut self) {
        if self.undelivered.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.cancellation.cancel();
        }
    }
}

fn mask_ranges(range: &Range<u64>, mask: &Mask) -> Vec<Range<u64>> {
    match mask.slices() {
        AllOr::All => vec![range.clone()],
        AllOr::None => Vec::new(),
        AllOr::Some(slices) => slices
            .iter()
            .map(|&(start, end)| range.start + start as u64..range.start + end as u64)
            .collect(),
    }
}

fn coalesce_ranges(ranges: Vec<Range<u64>>, max_gap: u64) -> Vec<Range<u64>> {
    let mut coalesced = Vec::<Range<u64>>::with_capacity(ranges.len());
    for range in ranges {
        match coalesced.last_mut() {
            Some(previous) if range.start.saturating_sub(previous.end) <= max_gap => {
                previous.end = previous.end.max(range.end);
            }
            _ => coalesced.push(range),
        }
    }
    coalesced
}

fn unbind(expr: &BoundExpression) -> VortexResult<Expression> {
    let Some(scalar_fn) = expr.as_scalar() else {
        return Ok(Expression::Root);
    };
    Expression::try_new(
        scalar_fn.clone(),
        expr.children()
            .iter()
            .map(unbind)
            .collect::<VortexResult<Vec<_>>>()?,
    )
}

struct SelectedMorsel {
    selected_ranges: Vec<Range<u64>>,
}

fn count_intersecting_morsels(morsels: &[Range<u64>], row_range: &Range<u64>) -> u64 {
    u64::try_from(
        morsels
            .iter()
            .filter(|range| range.start < row_range.end && row_range.start < range.end)
            .count(),
    )
    .unwrap_or(u64::MAX)
}

fn count_nonempty_morsels(morsels: &[SelectedMorsel]) -> u64 {
    u64::try_from(
        morsels
            .iter()
            .filter(|morsel| !morsel.selected_ranges.is_empty())
            .count(),
    )
    .unwrap_or(u64::MAX)
}

fn count_ranges(morsels: &[SelectedMorsel]) -> u64 {
    morsels
        .iter()
        .map(|morsel| u64::try_from(morsel.selected_ranges.len()).unwrap_or(u64::MAX))
        .sum()
}

fn selected_morsels(
    morsels: Vec<Range<u64>>,
    row_range: &Range<u64>,
    selection: &vortex_scan::selection::Selection,
) -> Vec<SelectedMorsel> {
    morsels
        .into_iter()
        .filter_map(|range| {
            let start = range.start.max(row_range.start);
            let end = range.end.min(row_range.end);
            (start < end).then_some(start..end)
        })
        .filter_map(|range| {
            let mask = selection.row_mask(&range);
            let selection_mask = mask.mask().clone();
            let selected_ranges = mask_ranges(&range, &selection_mask);
            (!selected_ranges.is_empty()).then_some(SelectedMorsel { selected_ranges })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::EXTERNAL_FRONTIER_BUNDLE_MORSELS;
    use super::ExecutorDriver;
    use super::ExecutorIoPolicy;
    use super::ExecutorIoSettings;
    use super::ExternalFrontierBundlePlan;
    use super::FRONTIER_LOOKAHEAD_PER_THREAD;
    use super::FRONTIER_REFILL_RANGES;
    use super::FRONTIER_SPECULATIVE_PREDICATES_RIGHT;
    use super::SHARED_LOOKAHEAD_MORSELS;
    use super::SelectedMorsel;
    use super::bundle_adjacent_contiguous_morsels;
    use super::bundled_frontier_down;
    use super::coalesce_ranges;
    use super::external_frontier_bundle_parallelism;
    use super::external_frontier_bundle_plan;

    #[rstest::rstest]
    fn projection_prefetch_requires_external_frontier_opt_in(
        #[values(ExecutorIoPolicy::EagerLookahead, ExecutorIoPolicy::Frontier)]
        policy: ExecutorIoPolicy,
        #[values(ExecutorDriver::Internal, ExecutorDriver::External)] driver: ExecutorDriver,
    ) {
        assert!(!policy.prefetch_projection(driver, false));
        assert_eq!(
            policy.prefetch_projection(driver, true),
            policy == ExecutorIoPolicy::Frontier && driver == ExecutorDriver::External
        );
    }

    #[test]
    fn production_frontier_policy_is_explicit_and_predicate_only() {
        assert_eq!(
            ExecutorIoPolicy::from_frontier_io(false),
            ExecutorIoPolicy::EagerLookahead
        );
        assert_eq!(
            ExecutorIoPolicy::from_frontier_io(true),
            ExecutorIoPolicy::Frontier
        );
        assert_eq!(
            ExecutorIoPolicy::Frontier.settings(ExecutorDriver::Internal),
            ExecutorIoSettings {
                eager_lookahead: false,
                lookahead_morsels: 0,
                frontier_lookahead_per_thread: Some(FRONTIER_LOOKAHEAD_PER_THREAD),
                speculative_predicate_frontiers: FRONTIER_SPECULATIVE_PREDICATES_RIGHT,
                frontier_refill_ranges: FRONTIER_REFILL_RANGES,
            }
        );
        assert_eq!(
            (
                FRONTIER_LOOKAHEAD_PER_THREAD,
                FRONTIER_SPECULATIVE_PREDICATES_RIGHT,
                FRONTIER_REFILL_RANGES,
            ),
            (0, 2, 32)
        );
        assert_eq!(
            ExecutorIoPolicy::Frontier.settings(ExecutorDriver::External),
            ExecutorIoPolicy::Frontier.settings(ExecutorDriver::Internal)
        );
    }

    #[test]
    fn external_eager_policy_preserves_legacy_zero_morsel_window() {
        assert_eq!(
            ExecutorIoPolicy::EagerLookahead.settings(ExecutorDriver::External),
            ExecutorIoSettings {
                eager_lookahead: true,
                lookahead_morsels: 0,
                frontier_lookahead_per_thread: None,
                speculative_predicate_frontiers: 0,
                frontier_refill_ranges: FRONTIER_REFILL_RANGES,
            }
        );
        assert_eq!(
            ExecutorIoPolicy::EagerLookahead.settings(ExecutorDriver::Internal),
            ExecutorIoSettings {
                eager_lookahead: true,
                lookahead_morsels: SHARED_LOOKAHEAD_MORSELS,
                frontier_lookahead_per_thread: None,
                speculative_predicate_frontiers: 0,
                frontier_refill_ranges: FRONTIER_REFILL_RANGES,
            }
        );
    }

    #[test]
    fn external_frontier_pairing_requires_eight_full_worker_waves() {
        assert_eq!(
            external_frontier_bundle_plan(222, 14, 8),
            ExternalFrontierBundlePlan {
                bundle_size: 1,
                candidate_bundle_count: 111,
                threshold: 112,
            }
        );
        assert_eq!(
            external_frontier_bundle_plan(223, 14, 8),
            ExternalFrontierBundlePlan {
                bundle_size: EXTERNAL_FRONTIER_BUNDLE_MORSELS,
                candidate_bundle_count: 112,
                threshold: 112,
            }
        );
        assert_eq!(
            external_frontier_bundle_plan(224, 14, 8),
            ExternalFrontierBundlePlan {
                bundle_size: EXTERNAL_FRONTIER_BUNDLE_MORSELS,
                candidate_bundle_count: 112,
                threshold: 112,
            }
        );
        assert_eq!(external_frontier_bundle_plan(223, 15, 8).bundle_size, 1);
        assert_eq!(
            external_frontier_bundle_plan(15, 1, 8).bundle_size,
            EXTERNAL_FRONTIER_BUNDLE_MORSELS
        );
        assert_eq!(external_frontier_bundle_plan(15, 2, 8).bundle_size, 1);
    }

    #[test]
    fn external_frontier_pairing_handles_one_worker_and_overflow() {
        assert_eq!(external_frontier_bundle_plan(1, 1, 1).bundle_size, 1);
        assert_eq!(
            external_frontier_bundle_plan(2, 1, 1).bundle_size,
            EXTERNAL_FRONTIER_BUNDLE_MORSELS
        );
        assert_eq!(
            external_frontier_bundle_plan(usize::MAX, usize::MAX, 2),
            ExternalFrontierBundlePlan {
                bundle_size: 1,
                candidate_bundle_count: usize::MAX.div_ceil(2),
                threshold: usize::MAX,
            }
        );
    }

    #[test]
    fn external_frontier_pairing_is_backend_filter_and_limit_isolated() {
        assert_eq!(
            external_frontier_bundle_parallelism(None, ExecutorIoPolicy::Frontier, true, false),
            None
        );
        assert_eq!(
            external_frontier_bundle_parallelism(
                Some(14),
                ExecutorIoPolicy::EagerLookahead,
                true,
                false
            ),
            None
        );
        assert_eq!(
            external_frontier_bundle_parallelism(
                Some(14),
                ExecutorIoPolicy::Frontier,
                false,
                false
            ),
            None
        );
        assert_eq!(
            external_frontier_bundle_parallelism(Some(14), ExecutorIoPolicy::Frontier, true, true),
            None
        );
        assert_eq!(
            external_frontier_bundle_parallelism(Some(14), ExecutorIoPolicy::Frontier, true, false),
            Some(14)
        );
    }

    #[test]
    fn adjacent_pairs_are_stable_disjoint_and_keep_an_odd_tail() {
        let morsels = (0..5)
            .map(|index| selected_one(index * 10, (index + 1) * 10))
            .collect::<Vec<_>>();
        let bundles = bundle_adjacent_contiguous_morsels(morsels);
        let coverages = bundles
            .iter()
            .map(|bundle| bundle.coverage.clone())
            .collect::<Vec<_>>();
        assert_eq!(coverages, [0..20, 20..40, 40..50]);
        assert!(
            coverages
                .iter()
                .zip(coverages.iter().skip(1))
                .all(|(left, right)| left.end <= right.start)
        );
        assert!(bundles[0].second.is_some());
        assert!(bundles[1].second.is_some());
        assert!(bundles[2].second.is_none());
    }

    #[test]
    fn selection_gap_breaks_an_external_frontier_pair() {
        let bundles = bundle_adjacent_contiguous_morsels(vec![
            selected_one(0, 10),
            selected_one(20, 30),
            selected_one(30, 40),
        ]);
        assert_eq!(
            bundles
                .iter()
                .map(|bundle| bundle.coverage.clone())
                .collect::<Vec<_>>(),
            [0..10, 20..40]
        );
    }

    #[test]
    fn down_one_requires_one_surviving_range_from_each_paired_morsel() {
        assert_eq!(
            bundled_frontier_down(&[selected_one(0, 10), selected_one(10, 20)]),
            1
        );
        assert_eq!(
            bundled_frontier_down(&[selected_one(0, 10), selected([])]),
            0
        );
        assert_eq!(
            bundled_frontier_down(&[selected([0..4, 6..10]), selected_one(10, 20)]),
            0
        );
        assert_eq!(
            bundled_frontier_down(&[selected_one(0, 9), selected_one(10, 20)]),
            0
        );
        assert_eq!(
            bundled_frontier_down(&[selected_one(0, 10), selected_one(11, 20)]),
            0
        );
        assert_eq!(bundled_frontier_down(&[selected_one(0, 10)]), 0);
    }

    #[test]
    fn noncontiguous_preprune_selection_never_pairs() {
        let bundles =
            bundle_adjacent_contiguous_morsels(vec![selected([0..4, 6..10]), selected_one(10, 20)]);
        assert_eq!(bundles.len(), 2);
        assert!(bundles.iter().all(|bundle| bundle.second.is_none()));
    }

    fn selected(selected_ranges: impl IntoIterator<Item = Range<u64>>) -> SelectedMorsel {
        SelectedMorsel {
            selected_ranges: selected_ranges.into_iter().collect(),
        }
    }

    fn selected_one(start: u64, end: u64) -> SelectedMorsel {
        selected(std::iter::once(start..end))
    }

    #[test]
    fn coalesces_only_small_gaps() {
        assert_eq!(
            coalesce_ranges(vec![0..10, 18..20, 30..40, 40..50], 8),
            vec![0..20, 30..50]
        );
    }

    #[test]
    fn coalesces_empty_ranges() {
        assert!(coalesce_ranges(Vec::new(), 8).is_empty());
    }
}
