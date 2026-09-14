// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::channel::oneshot;
use futures::future;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::stream::BoxStream;
use futures::stream::Fuse;
use futures::stream::SelectAll;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_io::CoalesceConfig;
use vortex_io::ReadAtNowait;
use vortex_io::ReadAtRequest;
use vortex_io::ReadAtStream;
use vortex_io::VortexReadAt;
use vortex_io::runtime::Handle;
use vortex_io::runtime::JoinOutcome;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_metrics::Counter;
use vortex_metrics::Gauge;
use vortex_metrics::Histogram;
use vortex_metrics::Label;
use vortex_metrics::MetricBuilder;
use vortex_metrics::MetricsRegistry;

use crate::SegmentSpec;
use crate::read::CoalesceGapBudget;
use crate::read::IoRequest;
use crate::read::IoRequestStream;
use crate::read::ReadRequest;
use crate::read::RequestId;

static NEXT_SOURCE_INSTANCE: AtomicU64 = AtomicU64::new(0);
static SCAN_DIAGNOSTICS_ENABLED: OnceLock<bool> = OnceLock::new();
const MAX_DIAGNOSTIC_BATCH_ITEMS: usize = 64;

fn scan_diagnostics_enabled() -> bool {
    *SCAN_DIAGNOSTICS_ENABLED.get_or_init(|| {
        std::env::var_os("VORTEX_SCAN_DIAGNOSTICS").is_some_and(|value| value != "0")
    })
}

#[derive(Clone)]
struct FileSourceDiagnostics {
    source_id: u64,
    read_at_id: Option<u64>,
    next_call_id: Arc<AtomicU64>,
    file_path: Arc<str>,
    partition: Arc<str>,
}

impl FileSourceDiagnostics {
    fn new(read_at_id: Option<u64>, file_path: Arc<str>, partition: Arc<str>) -> Self {
        Self {
            source_id: NEXT_SOURCE_INSTANCE.fetch_add(1, Ordering::Relaxed),
            read_at_id,
            next_call_id: Arc::new(AtomicU64::new(0)),
            file_path,
            partition,
        }
    }
}

struct DiagnosticRequestMapping {
    segment_id: SegmentId,
    request_id: Option<RequestId>,
    offset: Option<u64>,
    length: Option<usize>,
    outcome: &'static str,
}

struct DiagnosticPhysicalBatch {
    total_ranges: usize,
    child_offsets: Vec<u64>,
    child_lengths: Vec<usize>,
    logical_child_indexes: Vec<usize>,
    logical_request_ids: Vec<RequestId>,
    total_logical: usize,
}

impl DiagnosticPhysicalBatch {
    fn new(requests: &[IoRequest]) -> Self {
        let recorded = requests.len().min(MAX_DIAGNOSTIC_BATCH_ITEMS);
        let child_offsets = requests
            .iter()
            .take(recorded)
            .map(IoRequest::offset)
            .collect::<Vec<_>>();
        let child_lengths = requests
            .iter()
            .take(recorded)
            .map(IoRequest::len)
            .collect::<Vec<_>>();
        let mut logical_child_indexes = Vec::new();
        let mut logical_request_ids = Vec::new();
        let total_logical = requests
            .iter()
            .map(|request| request.request_ids().len())
            .sum::<usize>();
        for (child_index, request) in requests.iter().take(recorded).enumerate() {
            for request_id in request.request_ids() {
                if logical_request_ids.len() == MAX_DIAGNOSTIC_BATCH_ITEMS {
                    break;
                }
                logical_child_indexes.push(child_index);
                logical_request_ids.push(request_id);
            }
        }
        Self {
            total_ranges: requests.len(),
            child_offsets,
            child_lengths,
            logical_child_indexes,
            logical_request_ids,
            total_logical,
        }
    }
}

#[derive(Debug)]
/// Events sent from segment futures to the coalescing read driver.
pub enum ReadEvent {
    /// A segment read has been registered.
    Request(ReadRequest),
    /// A complete batch of segment reads has been registered for background execution.
    BackgroundRequests(Vec<ReadRequest>),
    /// A registered read is eligible to run as background work.
    Polled(RequestId),
    /// A demanded read should run before queued background reads.
    Promoted(RequestId),
    /// A registered read future was dropped before completion.
    Dropped(RequestId),
}

/// A [`SegmentSource`] for file-like IO.
/// ## Coalescing and Pre-fetching
///
/// It is important to understand the semantics of the read futures returned by a [`FileSegmentSource`].
/// Under the hood, each instance is backed by a stream that services read requests by
/// applying coalescing and concurrency constraints.
///
/// Each read future has four states:
/// * `registered` - the read future has been created, but not yet polled.
/// * `requested` - the read is eligible for background I/O or has been demanded.
/// * `in-flight` - the read request has been sent to the underlying storage system.
/// * `resolved` - the read future has completed and resolved a result.
///
/// When a read request is `registered`, it will not itself trigger any I/O, but is eligible to
/// be coalesced with other requests.
///
/// If a read future is dropped, it will be canceled if possible. This depends on the current
/// state of the request, as well as whether the underlying storage system supports cancellation.
///
/// I/O requests will be processed in the order they are `registered`, however coalescing may mean
/// other registered requests are lumped together into a single I/O operation.
/// A cloneable handle to the background read driver, shared by every in-flight [`ReadFuture`].
///
/// [`Shared`] fans a single completion out to all readers with correct waker bookkeeping, so a
/// reader is always woken when the driver finishes — even if another reader polled the driver more
/// recently and was then dropped. Its output is `()`; a driver panic is carried out of band in
/// [`DriverPanic`] so it can be re-raised on the reader side.
type SharedDriver = Shared<BoxFuture<'static, ()>>;

/// Slot holding the driver's panic payload, if it panicked while driving reads. The first reader to
/// observe completion takes the payload and re-raises it; later readers report a graceful error.
type DriverPanic = Arc<Mutex<Option<Box<dyn Any + Send>>>>;

fn validate_read_result(
    request: &IoRequest,
    result: VortexResult<BufferHandle>,
) -> VortexResult<BufferHandle> {
    result.and_then(|buffer| {
        if request.len() != buffer.len() {
            return Err(vortex_err!(
                "FileSegmentSource: expected buffer of length {} but received {}. {:?}",
                request.len(),
                buffer.len(),
                request
            ));
        }
        Ok(buffer)
    })
}

type IoBatchStream = Fuse<BoxStream<'static, Vec<IoRequest>>>;

enum ReadRangeResultsState {
    Reading(ReadAtStream),
    Missing,
}

/// Matches streamed range results back to their logical requests.
struct ReadRangeResults {
    state: ReadRangeResultsState,
    remaining: Vec<Option<IoRequest>>,
}

impl ReadRangeResults {
    fn new(results: ReadAtStream, requests: Vec<IoRequest>) -> Self {
        Self {
            state: ReadRangeResultsState::Reading(results),
            remaining: requests.into_iter().map(Some).collect(),
        }
    }
}

impl Stream for ReadRangeResults {
    type Item = (IoRequest, VortexResult<BufferHandle>);

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match &mut self.state {
                ReadRangeResultsState::Reading(results) => match results.poll_next_unpin(cx) {
                    Poll::Ready(Some((request, result))) => {
                        let Some(position) = self.remaining.iter().position(|req| {
                            req.as_ref().is_some_and(|req| {
                                req.offset() == request.offset
                                    && req.len() == request.length
                                    && req.alignment() == request.alignment
                            })
                        }) else {
                            tracing::warn!(?request, "reader returned an unknown range");
                            continue;
                        };
                        let req = self.remaining[position]
                            .take()
                            .vortex_expect("matched request is present");
                        return Poll::Ready(Some((req, result)));
                    }
                    Poll::Ready(None) => self.state = ReadRangeResultsState::Missing,
                    Poll::Pending => return Poll::Pending,
                },
                ReadRangeResultsState::Missing => {
                    let Some(req) = self.remaining.iter_mut().find_map(Option::take) else {
                        return Poll::Ready(None);
                    };
                    let error = vortex_err!(
                        "FileSegmentSource: read_ranges ended before resolving request. {:?}",
                        req
                    );
                    return Poll::Ready(Some((req, Err(error))));
                }
            }
        }
    }
}

/// Drives request batches while keeping the reader's concurrency slots occupied.
struct ReadDriver<R> {
    reader: Arc<R>,
    batches: IoBatchStream,
    pending: VecDeque<IoRequest>,
    reads: SelectAll<ReadRangeResults>,
    num_active: usize,
    batches_done: bool,
    concurrency: usize,
    metrics: RequestMetrics,
    diagnostics: Option<FileSourceDiagnostics>,
}

impl<R: VortexReadAt> ReadDriver<R> {
    fn new(
        reader: R,
        batches: BoxStream<'static, Vec<IoRequest>>,
        concurrency: usize,
        metrics: RequestMetrics,
        diagnostics: Option<FileSourceDiagnostics>,
    ) -> Self {
        Self {
            reader: Arc::new(reader),
            batches: batches.fuse(),
            pending: VecDeque::new(),
            reads: SelectAll::new(),
            num_active: 0,
            batches_done: false,
            concurrency,
            metrics,
            diagnostics,
        }
    }

    fn submit_pending(&mut self) {
        while self.num_active < self.concurrency && !self.pending.is_empty() {
            let batch_len = (self.concurrency - self.num_active).min(self.pending.len());
            let reqs = self.pending.drain(..batch_len).collect::<Vec<_>>();
            self.num_active += batch_len;
            self.metrics.read_ranges_calls.add(1);
            self.metrics.read_ranges_num_ranges.update(batch_len as f64);
            if batch_len > 1 {
                self.metrics.read_ranges_multi.add(1);
            }
            if let Some(in_flight_max) = &self.metrics.read_ranges_in_flight_max {
                in_flight_max.set_max(self.num_active as f64);
            }
            for req in &reqs {
                if let Some(request_size) = &self.metrics.read_ranges_request_size {
                    request_size.update(req.len() as f64);
                }
            }
            if let Some(diagnostics) = &self.diagnostics {
                let call_id = diagnostics.next_call_id.fetch_add(1, Ordering::Relaxed);
                let trace = DiagnosticPhysicalBatch::new(&reqs);
                let fill_stop = if self.num_active == self.concurrency {
                    "slots"
                } else {
                    "pending"
                };
                let span = tracing::trace_span!(
                    target: "vortex_file::read_ranges",
                    "read_ranges",
                    source_id = diagnostics.source_id,
                    read_at_id = diagnostics.read_at_id.unwrap_or_default(),
                    has_read_at_id = diagnostics.read_at_id.is_some(),
                    call_id,
                    t_submit_ns = vortex_io::diagnostic_timestamp_ns(),
                    num_ranges = trace.total_ranges,
                    recorded_ranges = trace.child_offsets.len(),
                    ranges_truncated = trace.total_ranges > trace.child_offsets.len(),
                    num_active = self.num_active,
                    concurrency = self.concurrency,
                    fill_stop,
                    file_path = diagnostics.file_path.as_ref(),
                    partition = diagnostics.partition.as_ref(),
                    child_offsets = ?trace.child_offsets,
                    child_lengths = ?trace.child_lengths,
                    total_logical = trace.total_logical,
                    recorded_logical = trace.logical_request_ids.len(),
                    logical_child_indexes = ?trace.logical_child_indexes,
                    logical_request_ids = ?trace.logical_request_ids,
                    logical_mapping_truncated = trace.total_logical > MAX_DIAGNOSTIC_BATCH_ITEMS,
                );
                let requests = reqs
                    .iter()
                    .map(|req| ReadAtRequest::new(req.offset(), req.len(), req.alignment()))
                    .collect::<Vec<_>>()
                    .into();
                let mut results = {
                    let _entered = span.enter();
                    self.reader.read_ranges(requests)
                };
                let poll_span = span.clone();
                let results = futures::stream::poll_fn(move |cx| {
                    let _entered = poll_span.enter();
                    results.poll_next_unpin(cx)
                })
                .boxed();
                self.reads.push(ReadRangeResults::new(results, reqs));
            } else {
                let requests = reqs
                    .iter()
                    .map(|req| ReadAtRequest::new(req.offset(), req.len(), req.alignment()))
                    .collect::<Vec<_>>()
                    .into();
                let results = self.reader.read_ranges(requests);
                self.reads.push(ReadRangeResults::new(results, reqs));
            }
        }
    }
}

impl<R: VortexReadAt> Stream for ReadDriver<R> {
    type Item = ();

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();

        // Keep unsubmitted work in IoRequestStream, where promotion can still reorder it. The
        // production stream emits one physical request at a time, so this loop removes exactly
        // the number that can be submitted now and combines them into one read_ranges call.
        if !this.batches_done && this.num_active < this.concurrency && this.pending.is_empty() {
            let available = this.concurrency - this.num_active;
            while this.pending.len() < available {
                match this.batches.poll_next_unpin(cx) {
                    Poll::Ready(Some(batch)) => this.pending.extend(batch),
                    Poll::Ready(None) => {
                        this.batches_done = true;
                        break;
                    }
                    Poll::Pending => break,
                }
            }
        }

        this.submit_pending();

        if this.batches_done && this.num_active == 0 {
            return Poll::Ready(None);
        }

        match this.reads.poll_next_unpin(cx) {
            Poll::Ready(Some((req, result))) => {
                this.num_active -= 1;
                let result = validate_read_result(&req, result);
                req.resolve(result);
                Poll::Ready(Some(()))
            }
            Poll::Ready(None) if this.num_active == 0 => Poll::Pending,
            Poll::Ready(None) => {
                vortex_panic!("read result streams ended with active requests")
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

pub struct FileSegmentSource {
    segments: Arc<[SegmentSpec]>,
    /// Reader retained for inline non-blocking segment probes.
    reader: Arc<dyn VortexReadAt>,
    /// A queue for sending read request events to the I/O stream.
    events: mpsc::UnboundedSender<ReadEvent>,
    /// Background request driver, joined by readers to surface a driver panic.
    driver: SharedDriver,
    /// Panic payload captured if the driver panicked while driving reads.
    driver_panic: DriverPanic,
    /// The next read request ID.
    next_id: Arc<AtomicUsize>,
    diagnostics: Option<FileSourceDiagnostics>,
}

impl FileSegmentSource {
    /// Open a file-backed segment source over `reader`.
    ///
    /// The returned source spawns a background driver on `handle` that coalesces and executes
    /// random-access read requests.
    pub fn open<R: VortexReadAt + Clone>(
        segments: Arc<[SegmentSpec]>,
        reader: R,
        handle: Handle,
        metrics: RequestMetrics,
    ) -> Self {
        Self::open_with_coalesce_gap_budget(segments, reader, handle, metrics, false)
    }

    /// Open a source with an optional aggregate budget for empty bytes introduced by coalescing.
    pub(crate) fn open_with_coalesce_gap_budget<R: VortexReadAt + Clone>(
        segments: Arc<[SegmentSpec]>,
        reader: R,
        handle: Handle,
        metrics: RequestMetrics,
        bounded_coalescing_gap: bool,
    ) -> Self {
        let (send, recv) = mpsc::unbounded();
        let nowait_reader: Arc<dyn VortexReadAt> = Arc::new(reader.clone());
        let diagnostics = metrics.diagnostics.as_ref().map(|labels| {
            FileSourceDiagnostics::new(
                reader.diagnostic_instance_id(),
                Arc::clone(&labels.file_path),
                Arc::clone(&labels.partition),
            )
        });

        let max_alignment = segments
            .iter()
            .map(|segment| segment.alignment)
            .max()
            .unwrap_or_else(Alignment::none);
        let coalesce_config = effective_coalesce_config(reader.coalesce_config(), max_alignment);
        let concurrency = reader.concurrency();
        if concurrency == 0 {
            vortex_panic!(
                "VortexReadAt::concurrency returned 0 (uri={:?}); this would stall I/O",
                reader.uri()
            );
        }
        let coalesce_gap_budget =
            bounded_coalescing_gap.then(|| CoalesceGapBudget::for_reader_concurrency(concurrency));

        let stream = IoRequestStream::new(
            StreamExt::boxed(recv),
            coalesce_config,
            coalesce_gap_budget,
            max_alignment,
            1,
            metrics.clone(),
        )
        .boxed();

        let drive_fut = ReadDriver::new(reader, stream, concurrency, metrics, diagnostics.clone())
            .collect::<()>();

        // Spawn the driver so the runtime makes I/O progress independently of any reader. Readers
        // join it (below) only to surface a panic raised while driving reads.
        let mut task = handle.spawn(drive_fut);
        let driver_panic: DriverPanic = Arc::new(Mutex::new(None));
        let driver = {
            let driver_panic = Arc::clone(&driver_panic);
            async move {
                // Poll for the terminal outcome without re-raising: a benign abort (runtime
                // teardown) resolves to `()` so readers report a graceful error, while a panic is
                // stashed for the first reader to re-raise.
                if let JoinOutcome::Panicked(panic) = future::poll_fn(|cx| task.poll_join(cx)).await
                {
                    *driver_panic.lock() = Some(panic);
                }
            }
            .boxed()
            .shared()
        };

        Self {
            segments,
            reader: nowait_reader,
            events: send,
            driver,
            driver_panic,
            next_id: Arc::new(AtomicUsize::new(0)),
            diagnostics,
        }
    }

    fn prepare_request(&self, id: SegmentId) -> VortexResult<(ReadRequest, SegmentFuture)> {
        let spec = *match self.segments.get(*id as usize) {
            Some(spec) => spec,
            None => {
                return Err(vortex_err!("Missing segment: {}", id));
            }
        };

        let SegmentSpec {
            offset,
            length,
            alignment,
        } = spec;

        let (send, recv) = oneshot::channel();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = ReadRequest {
            id,
            offset,
            length: length as usize,
            alignment,
            callback: send,
        };

        let future = ReadFuture {
            id,
            recv,
            promoted: false,
            finished: false,
            events: self.events.clone(),
            driver: self.driver.clone(),
            driver_panic: Arc::clone(&self.driver_panic),
        }
        .boxed();

        Ok((request, future))
    }

    fn trace_request_map(&self, total: usize, mappings: &[DiagnosticRequestMapping]) {
        let Some(diagnostics) = &self.diagnostics else {
            return;
        };
        debug_assert!(mappings.len() <= MAX_DIAGNOSTIC_BATCH_ITEMS);
        let segment_ids = mappings
            .iter()
            .map(|mapping| mapping.segment_id)
            .collect::<Vec<_>>();
        let request_ids = mappings
            .iter()
            .map(|mapping| mapping.request_id)
            .collect::<Vec<_>>();
        let offsets = mappings
            .iter()
            .map(|mapping| mapping.offset)
            .collect::<Vec<_>>();
        let lengths = mappings
            .iter()
            .map(|mapping| mapping.length)
            .collect::<Vec<_>>();
        let outcomes = mappings
            .iter()
            .map(|mapping| mapping.outcome)
            .collect::<Vec<_>>();
        tracing::trace!(
            target: "vortex_file::request_map",
            source_id = diagnostics.source_id,
            read_at_id = diagnostics.read_at_id.unwrap_or_default(),
            has_read_at_id = diagnostics.read_at_id.is_some(),
            t_map_ns = vortex_io::diagnostic_timestamp_ns(),
            total,
            recorded = mappings.len(),
            truncated = total > mappings.len(),
            file_path = diagnostics.file_path.as_ref(),
            partition = diagnostics.partition.as_ref(),
            segment_ids = ?segment_ids,
            request_ids = ?request_ids,
            offsets = ?offsets,
            lengths = ?lengths,
            outcomes = ?outcomes,
            "mapping segments to logical file requests"
        );
    }

    fn request_with_priority(&self, id: SegmentId, background: bool) -> SegmentFuture {
        // We eagerly register the read request here assuming the behaviour of [`FileSegmentSource`], where
        // coalescing becomes effective prior to the future being polled.
        let (request, future) = match self.prepare_request(id) {
            Ok(request) => request,
            Err(err) => {
                if self.diagnostics.is_some() {
                    self.trace_request_map(
                        1,
                        &[DiagnosticRequestMapping {
                            segment_id: id,
                            request_id: None,
                            offset: None,
                            length: None,
                            outcome: "missing",
                        }],
                    );
                }
                return future::ready(Err(err)).boxed();
            }
        };
        let request_id = request.id;
        let request_offset = request.offset;
        let request_length = request.length;

        if let Err(e) = self.events.unbounded_send(ReadEvent::Request(request)) {
            if self.diagnostics.is_some() {
                self.trace_request_map(
                    1,
                    &[DiagnosticRequestMapping {
                        segment_id: id,
                        request_id: Some(request_id),
                        offset: Some(request_offset),
                        length: Some(request_length),
                        outcome: "channel_closed",
                    }],
                );
            }
            return future::ready(Err(vortex_err!("Failed to submit read request: {e}"))).boxed();
        }
        if background && let Err(e) = self.events.unbounded_send(ReadEvent::Polled(request_id)) {
            if self.diagnostics.is_some() {
                self.trace_request_map(
                    1,
                    &[DiagnosticRequestMapping {
                        segment_id: id,
                        request_id: Some(request_id),
                        offset: Some(request_offset),
                        length: Some(request_length),
                        outcome: "priority_channel_closed",
                    }],
                );
            }
            return future::ready(Err(vortex_err!("Failed to submit background read: {e}")))
                .boxed();
        }

        if self.diagnostics.is_some() {
            self.trace_request_map(
                1,
                &[DiagnosticRequestMapping {
                    segment_id: id,
                    request_id: Some(request_id),
                    offset: Some(request_offset),
                    length: Some(request_length),
                    outcome: "queued",
                }],
            );
        }

        future
    }
}

fn effective_coalesce_config(
    configured: Option<CoalesceConfig>,
    max_alignment: Alignment,
) -> Option<CoalesceConfig> {
    configured.map(|mut config| {
        // Aligning the coalesced start down can add up to (alignment - 1) bytes. Increase max_size
        // to keep the reader's effective payload window consistent.
        let extra = (*max_alignment as u64).saturating_sub(1);
        config.max_size = config.max_size.saturating_add(extra);
        config
    })
}

impl SegmentSource for FileSegmentSource {
    fn diagnostic_instance_id(&self) -> Option<u64> {
        self.diagnostics
            .as_ref()
            .map(|diagnostics| diagnostics.source_id)
    }

    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.request_with_priority(id, false)
    }

    fn request_background(&self, id: SegmentId) -> SegmentFuture {
        self.request_with_priority(id, true)
    }

    fn request_background_batch(&self, ids: &[SegmentId]) -> Vec<SegmentFuture> {
        let mut requests = Vec::with_capacity(ids.len());
        let mut futures = Vec::with_capacity(ids.len());
        let mut mappings = self
            .diagnostics
            .as_ref()
            .map(|_| Vec::with_capacity(ids.len().min(MAX_DIAGNOSTIC_BATCH_ITEMS)));
        for &id in ids {
            match self.prepare_request(id) {
                Ok((request, future)) => {
                    if let Some(mappings) = &mut mappings
                        && mappings.len() < MAX_DIAGNOSTIC_BATCH_ITEMS
                    {
                        mappings.push(DiagnosticRequestMapping {
                            segment_id: id,
                            request_id: Some(request.id),
                            offset: Some(request.offset),
                            length: Some(request.length),
                            outcome: "queued",
                        });
                    }
                    requests.push(request);
                    futures.push(future);
                }
                Err(err) => {
                    if let Some(mappings) = &mut mappings
                        && mappings.len() < MAX_DIAGNOSTIC_BATCH_ITEMS
                    {
                        mappings.push(DiagnosticRequestMapping {
                            segment_id: id,
                            request_id: None,
                            offset: None,
                            length: None,
                            outcome: "missing",
                        });
                    }
                    futures.push(future::ready(Err(err)).boxed());
                }
            }
        }

        let submitted = if !requests.is_empty() {
            // One channel event preserves the scheduler's batch boundary: the driver cannot observe
            // an eligible member until every request in the batch is in its spatial index.
            self.events
                .unbounded_send(ReadEvent::BackgroundRequests(requests))
                .is_ok()
        } else {
            true
        };
        if let Some(mut mappings) = mappings {
            if !submitted {
                for mapping in &mut mappings {
                    if mapping.request_id.is_some() {
                        mapping.outcome = "channel_closed";
                    }
                }
            }
            self.trace_request_map(ids.len(), &mappings);
        }
        futures
    }

    fn request_nowait(&self, id: SegmentId) -> VortexResult<ReadAtNowait> {
        let spec = self
            .segments
            .get(*id as usize)
            .ok_or_else(|| vortex_err!("Missing segment: {}", id))?;
        self.reader
            .read_at_nowait(spec.offset, spec.length as usize, spec.alignment)
    }

    fn prefers_background_reads(&self) -> bool {
        true
    }
}

/// A future that resolves a read request from a [`FileSegmentSource`].
///
/// See the documentation for [`FileSegmentSource`] for details on coalescing and pre-fetching.
/// If dropped, the read request will be canceled where possible.
struct ReadFuture {
    id: usize,
    recv: oneshot::Receiver<VortexResult<BufferHandle>>,
    promoted: bool,
    finished: bool,
    events: mpsc::UnboundedSender<ReadEvent>,
    driver: SharedDriver,
    driver_panic: DriverPanic,
}

impl Future for ReadFuture {
    type Output = VortexResult<BufferHandle>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.recv.poll_unpin(cx) {
            // note: we are skipping promotion and dropped events for this if the future is ready on
            //       the first poll, that means this request was completed before it was polled,
            //       as part of a coalesced request.
            Poll::Ready(Ok(result)) => {
                self.finished = true;
                Poll::Ready(result)
            }
            // The request's sender was dropped, so the driver has finished. Join it so a panic
            // raised while driving reads is re-raised here rather than surfacing as a generic
            // error. Only report the dropped error once the driver has finished.
            Poll::Ready(Err(e)) => match self.driver.poll_unpin(cx) {
                Poll::Ready(()) => {
                    self.finished = true;
                    // Re-raise the driver panic on the first reader to observe it; later readers
                    // fall through to the graceful dropped error.
                    if let Some(panic) = self.driver_panic.lock().take() {
                        std::panic::resume_unwind(panic);
                    }
                    Poll::Ready(Err(vortex_err!("ReadRequest dropped by runtime: {e}")))
                }
                Poll::Pending => Poll::Pending,
            },
            Poll::Pending if !self.promoted => {
                self.promoted = true;
                match self.events.unbounded_send(ReadEvent::Promoted(self.id)) {
                    Ok(()) => Poll::Pending,
                    Err(e) => Poll::Ready(Err(vortex_err!("ReadRequest dropped by runtime: {e}"))),
                }
            }
            _ => Poll::Pending,
        }
    }
}

impl Drop for ReadFuture {
    fn drop(&mut self) {
        // Completed requests have already left driver state.
        if self.finished {
            return;
        }

        // Best-effort cancellation signal to the I/O stream.
        drop(self.events.unbounded_send(ReadEvent::Dropped(self.id)));
    }
}

/// Metrics emitted by the file segment request driver.
#[derive(Clone)]
pub struct RequestMetrics {
    /// Number of individual segment requests observed by the driver.
    pub individual_requests: Counter,
    /// Number of physical reads after coalescing.
    pub coalesced_requests: Counter,
    /// Distribution of how many segment requests were merged into each physical read.
    pub num_requests_coalesced: Histogram,
    /// Number of calls made to [`VortexReadAt::read_ranges`].
    pub read_ranges_calls: Counter,
    /// Number of `read_ranges` calls containing more than one physical range.
    pub read_ranges_multi: Counter,
    /// Distribution of physical range counts submitted per `read_ranges` call.
    pub read_ranges_num_ranges: Histogram,
    /// Distribution of physical request sizes submitted through `read_ranges`.
    read_ranges_request_size: Option<Histogram>,
    /// Maximum number of ranges in flight in this source driver.
    read_ranges_in_flight_max: Option<Gauge>,
    diagnostics: Option<RequestDiagnosticLabels>,
}

#[derive(Clone)]
struct RequestDiagnosticLabels {
    file_path: Arc<str>,
    partition: Arc<str>,
}

impl RequestMetrics {
    /// Create request metrics in `metrics_registry` with shared labels.
    pub fn new(metrics_registry: &dyn MetricsRegistry, labels: Vec<Label>) -> Self {
        Self::new_with_diagnostics(metrics_registry, labels, scan_diagnostics_enabled())
    }

    fn new_with_diagnostics(
        metrics_registry: &dyn MetricsRegistry,
        labels: Vec<Label>,
        diagnostics: bool,
    ) -> Self {
        let file_path = diagnostics.then(|| {
            labels
                .iter()
                .find(|label| label.key() == "file_path")
                .map_or_else(|| Arc::from(""), |label| Arc::from(label.value()))
        });
        let partition = diagnostics.then(|| {
            labels
                .iter()
                .find(|label| label.key() == "partition")
                .map_or_else(|| Arc::from(""), |label| Arc::from(label.value()))
        });
        Self {
            individual_requests: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("io.requests.individual"),
            coalesced_requests: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("io.requests.coalesced"),
            num_requests_coalesced: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .histogram("io.requests.coalesced.num_coalesced"),
            read_ranges_calls: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("io.read_ranges.calls"),
            read_ranges_multi: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("io.read_ranges.multi_range_calls"),
            read_ranges_num_ranges: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .histogram("io.read_ranges.num_ranges"),
            read_ranges_request_size: diagnostics.then(|| {
                MetricBuilder::new(metrics_registry)
                    .add_labels(labels.clone())
                    .histogram("io.read_ranges.request_size")
            }),
            read_ranges_in_flight_max: diagnostics.then(|| {
                MetricBuilder::new(metrics_registry)
                    .add_labels(labels)
                    .gauge("io.read_ranges.in_flight_max")
            }),
            diagnostics: file_path.zip(partition).map(|(file_path, partition)| {
                RequestDiagnosticLabels {
                    file_path,
                    partition,
                }
            }),
        }
    }
}

/// A [`SegmentSource`] that resolves segments synchronously from an
/// in-memory [`ByteBuffer`].
///
/// Resolves segments synchronously, bypassing the async I/O pipeline.
pub(crate) struct BufferSegmentSource {
    buffer: ByteBuffer,
    segments: Arc<[SegmentSpec]>,
}

impl BufferSegmentSource {
    /// Create a new `BufferSegmentSource` from a buffer and its segment map.
    pub fn new(buffer: ByteBuffer, segments: Arc<[SegmentSpec]>) -> Self {
        Self { buffer, segments }
    }
}

impl SegmentSource for BufferSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let spec = match self.segments.get(*id as usize) {
            Some(spec) => spec,
            None => {
                return future::ready(Err(vortex_err!("Missing segment: {}", id))).boxed();
            }
        };

        let start = spec.offset as usize;
        let end = start + spec.length as usize;
        if end > self.buffer.len() {
            return future::ready(Err(vortex_err!(
                "Segment {} range {}..{} out of bounds for buffer of length {}",
                *id,
                start,
                end,
                self.buffer.len()
            )))
            .boxed();
        }

        let slice = self.buffer.slice(start..end).aligned(spec.alignment);
        future::ready(Ok(BufferHandle::new_host(slice))).boxed()
    }
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;

    use futures::future::BoxFuture;
    use parking_lot::Mutex as TestMutex;
    use tracing::field::Field;
    use tracing::field::Visit;
    use tracing::span::Attributes;
    use tracing::span::Id;
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::Context;
    use tracing_subscriber::layer::SubscriberExt;
    use vortex_error::vortex_bail;
    use vortex_io::runtime::tokio::TokioRuntime;
    use vortex_layout::segments::SegmentSource;
    use vortex_metrics::DefaultMetricsRegistry;

    use super::*;

    fn io_request(id: RequestId, offset: u64, length: usize) -> IoRequest {
        let (callback, _receiver) = oneshot::channel();
        IoRequest::new_single(ReadRequest {
            id,
            offset,
            length,
            alignment: Alignment::none(),
            callback,
        })
    }

    #[test]
    fn coalesce_config_preserves_reader_window_and_allows_alignment() {
        let configured = CoalesceConfig::new(64 << 10, 16 << 20);

        let unchanged = effective_coalesce_config(Some(configured), Alignment::none())
            .vortex_expect("configured reader keeps coalescing enabled");
        assert_eq!(unchanged.distance, configured.distance);
        assert_eq!(unchanged.max_size, configured.max_size);

        let aligned = effective_coalesce_config(Some(configured), Alignment::new(8))
            .vortex_expect("configured reader keeps coalescing enabled");
        assert_eq!(aligned.distance, configured.distance);
        assert_eq!(aligned.max_size, configured.max_size + 7);

        assert!(effective_coalesce_config(None, Alignment::none()).is_none());
    }

    #[tokio::test]
    async fn read_range_results_matches_results_and_reports_missing_requests() {
        let requests = vec![io_request(0, 0, 4), io_request(1, 4, 4)];
        let unknown = ReadAtRequest::new(8, 4, Alignment::none());
        let returned = ReadAtRequest::new(4, 4, Alignment::none());
        let buffer = BufferHandle::new_host(ByteBuffer::from(vec![0; 4]));
        let results =
            futures::stream::iter([(unknown, Ok(buffer.clone())), (returned, Ok(buffer))]).boxed();

        let resolved = ReadRangeResults::new(results, requests)
            .collect::<Vec<_>>()
            .await;

        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].0.offset(), 4);
        assert!(resolved[0].1.is_ok());
        assert_eq!(resolved[1].0.offset(), 0);
        assert!(resolved[1].1.is_err());
    }

    #[tokio::test]
    async fn read_range_results_preserves_repeated_range_identity() {
        let requests = vec![io_request(7, 4, 4), io_request(8, 4, 4)];
        let returned = ReadAtRequest::new(4, 4, Alignment::none());
        let results = futures::stream::iter([
            (
                returned,
                Ok(BufferHandle::new_host(ByteBuffer::from(vec![1; 4]))),
            ),
            (
                returned,
                Ok(BufferHandle::new_host(ByteBuffer::from(vec![2; 4]))),
            ),
        ])
        .boxed();

        let resolved = ReadRangeResults::new(results, requests)
            .collect::<Vec<_>>()
            .await;
        assert_eq!(resolved[0].0.request_ids(), [7]);
        assert_eq!(resolved[1].0.request_ids(), [8]);
        assert!(resolved.iter().all(|(_, result)| result.is_ok()));
    }

    #[test]
    fn diagnostic_physical_mapping_is_bounded_and_keeps_duplicate_ranges_distinct() {
        let requests = (0..65).map(|id| io_request(id, 4, 4)).collect::<Vec<_>>();
        let trace = DiagnosticPhysicalBatch::new(&requests);

        assert_eq!(trace.total_ranges, 65);
        assert_eq!(trace.child_offsets.len(), MAX_DIAGNOSTIC_BATCH_ITEMS);
        assert_eq!(trace.child_lengths.len(), MAX_DIAGNOSTIC_BATCH_ITEMS);
        assert_eq!(
            trace.logical_child_indexes.len(),
            MAX_DIAGNOSTIC_BATCH_ITEMS
        );
        assert_eq!(trace.logical_request_ids.len(), MAX_DIAGNOSTIC_BATCH_ITEMS);
        assert_eq!(trace.logical_child_indexes[..2], [0, 1]);
        assert_eq!(trace.logical_request_ids[..2], [0, 1]);
        assert_eq!(trace.total_logical, 65);
    }

    #[derive(Default)]
    struct CapturedReadRanges {
        span_id: Option<Id>,
        source_id: Option<u64>,
        read_at_id: Option<u64>,
        call_id: Option<u64>,
        total_logical: Option<u64>,
        recorded_logical: Option<u64>,
        logical_child_indexes: Option<String>,
        logical_request_ids: Option<String>,
    }

    impl Visit for CapturedReadRanges {
        fn record_u64(&mut self, field: &Field, value: u64) {
            match field.name() {
                "source_id" => self.source_id = Some(value),
                "read_at_id" => self.read_at_id = Some(value),
                "call_id" => self.call_id = Some(value),
                "total_logical" => self.total_logical = Some(value),
                "recorded_logical" => self.recorded_logical = Some(value),
                _ => {}
            }
        }

        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            match field.name() {
                "logical_child_indexes" => {
                    self.logical_child_indexes = Some(format!("{value:?}"));
                }
                "logical_request_ids" => {
                    self.logical_request_ids = Some(format!("{value:?}"));
                }
                _ => {}
            }
        }
    }

    #[derive(Clone)]
    struct ReadRangesCaptureLayer {
        spans: Arc<TestMutex<Vec<CapturedReadRanges>>>,
    }

    impl<S: tracing::Subscriber> Layer<S> for ReadRangesCaptureLayer {
        fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, _ctx: Context<'_, S>) {
            if attrs.metadata().target() != "vortex_file::read_ranges" {
                return;
            }
            let mut captured = CapturedReadRanges {
                span_id: Some(id.clone()),
                ..CapturedReadRanges::default()
            };
            attrs.record(&mut captured);
            self.spans.lock().push(captured);
        }
    }

    #[derive(Clone)]
    struct SpanRecordingReadAt {
        synchronous_spans: Arc<TestMutex<Vec<Option<Id>>>>,
        poll_spans: Arc<TestMutex<Vec<Option<Id>>>>,
    }

    impl VortexReadAt for SpanRecordingReadAt {
        fn diagnostic_instance_id(&self) -> Option<u64> {
            Some(77)
        }

        fn concurrency(&self) -> usize {
            1
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            async { Ok(8) }.boxed()
        }

        fn read_at(
            &self,
            _offset: u64,
            _length: usize,
            _alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            async { vortex_bail!("test must use read_ranges") }.boxed()
        }

        fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
            self.synchronous_spans
                .lock()
                .push(tracing::Span::current().id());
            let poll_spans = Arc::clone(&self.poll_spans);
            let request_count = requests.len();
            futures::stream::iter(0..request_count)
                .map(move |index| {
                    poll_spans.lock().push(tracing::Span::current().id());
                    let request = requests[index];
                    let buffer = BufferHandle::new_host(ByteBuffer::from(vec![0; request.length]));
                    (request, Ok(buffer))
                })
                .boxed()
        }
    }

    #[test]
    fn repeated_ranges_have_distinct_call_spans_and_joinable_children() {
        let captured = Arc::new(TestMutex::new(Vec::new()));
        let synchronous_spans = Arc::new(TestMutex::new(Vec::new()));
        let poll_spans = Arc::new(TestMutex::new(Vec::new()));
        let reader = SpanRecordingReadAt {
            synchronous_spans: Arc::clone(&synchronous_spans),
            poll_spans: Arc::clone(&poll_spans),
        };
        let metrics = RequestMetrics::new_with_diagnostics(
            &DefaultMetricsRegistry::default(),
            Vec::new(),
            true,
        );
        let diagnostics = FileSourceDiagnostics::new(
            reader.diagnostic_instance_id(),
            Arc::from("fixture"),
            Arc::from("0"),
        );
        let subscriber = tracing_subscriber::registry().with(ReadRangesCaptureLayer {
            spans: Arc::clone(&captured),
        });

        tracing::subscriber::with_default(subscriber, || {
            futures::executor::block_on(async {
                let mut driver = ReadDriver::new(
                    reader,
                    futures::stream::pending().boxed(),
                    1,
                    metrics,
                    Some(diagnostics),
                );
                for request_id in [17, 18] {
                    driver.pending.push_back(io_request(request_id, 4, 4));
                    driver.submit_pending();
                    let (_request, result) = driver
                        .reads
                        .next()
                        .await
                        .vortex_expect("one physical result must arrive");
                    result.vortex_expect("fixture read must succeed");
                    driver.num_active -= 1;
                }
            });
        });

        let captured = captured.lock();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].source_id, captured[1].source_id);
        assert_eq!(captured[0].read_at_id, Some(77));
        assert_eq!(
            [captured[0].call_id, captured[1].call_id],
            [Some(0), Some(1)]
        );
        assert_eq!(captured[0].total_logical, Some(1));
        assert_eq!(captured[0].recorded_logical, Some(1));
        assert_eq!(captured[0].logical_child_indexes.as_deref(), Some("[0]"));
        assert_eq!(captured[1].logical_child_indexes.as_deref(), Some("[0]"));
        assert_eq!(captured[0].logical_request_ids.as_deref(), Some("[17]"));
        assert_eq!(captured[1].logical_request_ids.as_deref(), Some("[18]"));

        let span_ids = captured
            .iter()
            .map(|span| span.span_id.clone())
            .collect::<Vec<_>>();
        assert_ne!(span_ids[0], span_ids[1]);
        assert_eq!(*synchronous_spans.lock(), span_ids);
        assert_eq!(*poll_spans.lock(), span_ids);
    }

    #[derive(Clone)]
    struct PanickingReadAt;

    impl VortexReadAt for PanickingReadAt {
        fn concurrency(&self) -> usize {
            1
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            async { Ok(4) }.boxed()
        }

        fn read_at(
            &self,
            _offset: u64,
            _length: usize,
            _alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            async {
                panic!("read-at panic");
            }
            .boxed()
        }
    }

    fn panicking_source() -> FileSegmentSource {
        let segments: Arc<[SegmentSpec]> = Arc::from([SegmentSpec {
            offset: 0,
            length: 4,
            alignment: Alignment::none(),
        }]);
        let metrics = DefaultMetricsRegistry::default();
        FileSegmentSource::open(
            segments,
            PanickingReadAt,
            TokioRuntime::current(),
            RequestMetrics::new(&metrics, vec![]),
        )
    }

    fn panic_message(payload: &(dyn Any + Send)) -> &str {
        payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<non-string panic>")
    }

    #[tokio::test]
    #[should_panic(expected = "read-at panic")]
    async fn file_segment_source_propagates_read_driver_panic() {
        let source = panicking_source();
        let _result = source.request(SegmentId::from(0)).await;
    }

    // A read-driver panic must propagate on *every* run rather than sometimes surfacing as a
    // generic "dropped by runtime" error, which is nondeterministic.
    #[tokio::test]
    async fn file_segment_source_read_driver_panic_propagates_deterministically() {
        for i in 0..100 {
            let source = panicking_source();
            let outcome = AssertUnwindSafe(source.request(SegmentId::from(0)))
                .catch_unwind()
                .await;
            assert!(
                outcome.is_err(),
                "read-driver panic was not propagated on iteration {i}; \
                 the request resolved instead of panicking"
            );
        }
    }

    // A driver panic must surface to every concurrent reader sharing one driver: exactly one
    // reader re-raises the original panic, the rest report a graceful error, and none hang. This
    // also covers the fan-out invariant — `Shared` wakes every reader on completion, so a reader
    // dropped mid-flight can never swallow another reader's wake-up.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn file_segment_source_panic_propagates_to_all_concurrent_readers() {
        let source = Arc::new(panicking_source());
        let handle = TokioRuntime::current();
        let reader_count = 8;

        let readers: Vec<_> = (0..reader_count)
            .map(|_| {
                let source = Arc::clone(&source);
                handle.spawn(async move {
                    AssertUnwindSafe(source.request(SegmentId::from(0)))
                        .catch_unwind()
                        .await
                })
            })
            .collect();

        let joined =
            tokio::time::timeout(std::time::Duration::from_secs(5), future::join_all(readers))
                .await
                .expect("a reader hung instead of observing the driver panic");

        let mut original_panics = 0;
        for reader in joined {
            match reader {
                // The first reader to observe completion re-raises the original panic.
                Err(payload) => {
                    assert!(
                        panic_message(&*payload).contains("read-at panic"),
                        "got: {:?}",
                        panic_message(&*payload)
                    );
                    original_panics += 1;
                }
                // Every other reader reports a graceful dropped-by-runtime error.
                Ok(result) => assert!(result.is_err(), "expected a dropped-by-runtime error"),
            }
        }

        assert_eq!(
            original_panics, 1,
            "exactly one reader should re-raise the original driver panic"
        );
    }

    #[derive(Clone)]
    struct ReadRangesOnly {
        calls: Arc<AtomicUsize>,
        diagnostic_id: Option<u64>,
    }

    impl VortexReadAt for ReadRangesOnly {
        fn diagnostic_instance_id(&self) -> Option<u64> {
            self.diagnostic_id
        }

        fn concurrency(&self) -> usize {
            4
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            async { Ok(16) }.boxed()
        }

        fn read_at(
            &self,
            _offset: u64,
            _length: usize,
            _alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            async { panic!("read_at should not be called") }.boxed()
        }

        fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let results = requests
                .iter()
                .copied()
                .map(|request| {
                    let buffer = BufferHandle::new_host(
                        ByteBuffer::from(vec![0; request.length]).aligned(request.alignment),
                    );
                    (request, Ok(buffer))
                })
                .collect::<Vec<_>>();
            futures::stream::iter(results).boxed()
        }
    }

    #[tokio::test]
    async fn read_driver_batches_ready_requests() -> VortexResult<()> {
        let calls = Arc::new(AtomicUsize::new(0));
        let segments: Arc<[SegmentSpec]> = (0..4)
            .map(|i| SegmentSpec {
                offset: i * 4,
                length: 4,
                alignment: Alignment::none(),
            })
            .collect();
        let metrics = DefaultMetricsRegistry::default();
        let request_metrics = RequestMetrics::new_with_diagnostics(&metrics, vec![], true);
        let source = FileSegmentSource::open(
            segments,
            ReadRangesOnly {
                calls: Arc::clone(&calls),
                diagnostic_id: Some(77),
            },
            TokioRuntime::current(),
            request_metrics.clone(),
        );
        assert!(source.diagnostic_instance_id().is_some());
        assert_eq!(
            source
                .diagnostics
                .as_ref()
                .and_then(|diagnostics| diagnostics.read_at_id),
            Some(77)
        );

        let results = future::join_all((0..4).map(|i| source.request(SegmentId::from(i)))).await;

        for result in results {
            assert_eq!(result?.len(), 4);
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert_eq!(request_metrics.read_ranges_calls.value(), 1);
        assert_eq!(request_metrics.read_ranges_multi.value(), 1);
        assert_eq!(request_metrics.read_ranges_num_ranges.count(), 1);
        assert_eq!(request_metrics.read_ranges_num_ranges.total(), 4.0);
        let request_size = request_metrics
            .read_ranges_request_size
            .as_ref()
            .vortex_expect("diagnostic request-size histogram must be registered");
        assert_eq!(request_size.count(), 4);
        assert_eq!(request_size.total(), 16.0);
        assert_eq!(
            request_metrics
                .read_ranges_in_flight_max
                .as_ref()
                .vortex_expect("diagnostic in-flight gauge must be registered")
                .value(),
            4.0
        );
        Ok(())
    }

    #[test]
    fn request_diagnostics_are_opt_in_and_fixed_cardinality() {
        let registry = DefaultMetricsRegistry::default();
        let _metrics = RequestMetrics::new_with_diagnostics(&registry, vec![], false);
        assert_eq!(registry.snapshot().len(), 6);

        let registry = DefaultMetricsRegistry::default();
        let _metrics = RequestMetrics::new_with_diagnostics(&registry, vec![], true);
        assert_eq!(registry.snapshot().len(), 8);
    }

    #[derive(Clone)]
    struct ControlledReadRanges {
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
        batch_sizes: Arc<Mutex<Vec<usize>>>,
        permits: Arc<tokio::sync::Semaphore>,
    }

    impl VortexReadAt for ControlledReadRanges {
        fn concurrency(&self) -> usize {
            4
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            async { Ok(24) }.boxed()
        }

        fn read_at(
            &self,
            _offset: u64,
            _length: usize,
            _alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            async { panic!("read_at should not be called") }.boxed()
        }

        fn read_ranges(&self, requests: Arc<[ReadAtRequest]>) -> ReadAtStream {
            self.batch_sizes.lock().push(requests.len());
            let active = self.active.fetch_add(requests.len(), Ordering::SeqCst) + requests.len();
            self.max_active.fetch_max(active, Ordering::SeqCst);

            let reads = requests
                .iter()
                .copied()
                .map(|request| {
                    let active = Arc::clone(&self.active);
                    let permits = Arc::clone(&self.permits);
                    async move {
                        let Ok(permit) = permits.acquire_owned().await else {
                            vortex_panic!("test semaphore unexpectedly closed");
                        };
                        permit.forget();
                        active.fetch_sub(1, Ordering::SeqCst);
                        let buffer = BufferHandle::new_host(
                            ByteBuffer::from(vec![0; request.length]).aligned(request.alignment),
                        );
                        (request, Ok(buffer))
                    }
                })
                .collect::<Vec<_>>();
            futures::stream::iter(reads).buffer_unordered(4).boxed()
        }
    }

    #[tokio::test]
    async fn read_driver_refills_global_concurrency_across_batches() -> VortexResult<()> {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let batch_sizes = Arc::new(Mutex::new(Vec::new()));
        let permits = Arc::new(tokio::sync::Semaphore::new(0));
        let segments: Arc<[SegmentSpec]> = (0..6)
            .map(|i| SegmentSpec {
                offset: i * 4,
                length: 4,
                alignment: Alignment::none(),
            })
            .collect();
        let metrics = DefaultMetricsRegistry::default();
        let source = FileSegmentSource::open(
            segments,
            ControlledReadRanges {
                active: Arc::clone(&active),
                max_active: Arc::clone(&max_active),
                batch_sizes: Arc::clone(&batch_sizes),
                permits: Arc::clone(&permits),
            },
            TokioRuntime::current(),
            RequestMetrics::new(&metrics, vec![]),
        );
        let reads = TokioRuntime::current().spawn(async move {
            future::join_all((0..6).map(|i| source.request(SegmentId::from(i)))).await
        });

        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), async {
                while active.load(Ordering::SeqCst) != 4 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .is_ok()
        );

        permits.add_permits(1);
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), async {
                while batch_sizes.lock().len() < 2 || active.load(Ordering::SeqCst) != 4 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .is_ok()
        );
        assert_eq!(batch_sizes.lock().as_slice(), [4, 1]);
        assert_eq!(max_active.load(Ordering::SeqCst), 4);

        permits.add_permits(5);
        for result in reads.await {
            assert_eq!(result?.len(), 4);
        }
        assert_eq!(max_active.load(Ordering::SeqCst), 4);
        Ok(())
    }

    #[tokio::test]
    async fn read_driver_keeps_slots_full_while_a_straggler_is_in_flight() -> VortexResult<()> {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let batch_sizes = Arc::new(Mutex::new(Vec::new()));
        let permits = Arc::new(tokio::sync::Semaphore::new(0));
        let segments: Arc<[SegmentSpec]> = (0..8)
            .map(|i| SegmentSpec {
                offset: i * 4,
                length: 4,
                alignment: Alignment::none(),
            })
            .collect();
        let metrics = DefaultMetricsRegistry::default();
        let source = FileSegmentSource::open(
            segments,
            ControlledReadRanges {
                active: Arc::clone(&active),
                max_active: Arc::clone(&max_active),
                batch_sizes: Arc::clone(&batch_sizes),
                permits: Arc::clone(&permits),
            },
            TokioRuntime::current(),
            RequestMetrics::new(&metrics, vec![]),
        );
        let reads = TokioRuntime::current().spawn(async move {
            future::join_all((0..8).map(|i| source.request(SegmentId::from(i)))).await
        });

        wait_for_active_reads(&active, 4).await;

        // Complete three reads while leaving one original read blocked as a straggler. Each freed
        // slot must be refilled before the next completion; a batch-barrier implementation would
        // instead fall from four active reads to one and submit no replacement work.
        for expected_calls in 2..=4 {
            permits.add_permits(1);
            assert!(
                tokio::time::timeout(std::time::Duration::from_secs(1), async {
                    while batch_sizes.lock().len() < expected_calls
                        || active.load(Ordering::SeqCst) != 4
                    {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .is_ok()
            );
        }

        assert_eq!(batch_sizes.lock().as_slice(), [4, 1, 1, 1]);
        assert_eq!(active.load(Ordering::SeqCst), 4);
        assert_eq!(max_active.load(Ordering::SeqCst), 4);

        permits.add_permits(5);
        for result in reads.await {
            assert_eq!(result?.len(), 4);
        }
        Ok(())
    }

    async fn wait_for_active_reads(active: &AtomicUsize, expected: usize) {
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), async {
                while active.load(Ordering::SeqCst) != expected {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .is_ok()
        );
    }

    #[derive(Clone)]
    struct SlowErrReadAt;

    impl VortexReadAt for SlowErrReadAt {
        fn concurrency(&self) -> usize {
            4
        }

        fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
            async { Ok(1024) }.boxed()
        }

        fn read_at(
            &self,
            offset: u64,
            _length: usize,
            _alignment: Alignment,
        ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
            async move {
                // Stagger completions so some reads finish while others are still in flight.
                for _ in 0..(offset as usize % 5 + 1) {
                    tokio::task::yield_now().await;
                }
                vortex_bail!("slow read done")
            }
            .boxed()
        }
    }

    // Many segment reads driven on separate tasks must all make progress and complete.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn read_driver_concurrent_reads_make_progress() {
        let n = 16u32;
        let segments: Arc<[SegmentSpec]> = (0..n)
            .map(|i| SegmentSpec {
                offset: u64::from(i) * 4,
                length: 4,
                alignment: Alignment::none(),
            })
            .collect();
        let metrics = DefaultMetricsRegistry::default();
        let source = Arc::new(FileSegmentSource::open(
            segments,
            SlowErrReadAt,
            TokioRuntime::current(),
            RequestMetrics::new(&metrics, vec![]),
        ));

        let handle = TokioRuntime::current();
        let tasks: Vec<_> = (0..n)
            .map(|i| {
                let source = Arc::clone(&source);
                handle.spawn(async move { source.request(SegmentId::from(i)).await })
            })
            .collect();

        let joined =
            tokio::time::timeout(std::time::Duration::from_secs(5), future::join_all(tasks)).await;

        assert!(joined.is_ok(), "concurrent reads stalled before completing");
    }
}
