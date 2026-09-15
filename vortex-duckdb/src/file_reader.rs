// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use futures::FutureExt;
#[cfg(any(unix, windows))]
use object_store::ObjectStore;
#[cfg(any(unix, windows))]
use object_store::ObjectStoreExt;
use object_store::registry::ObjectStoreRegistry;
use url::Url;
use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute as _;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayExt as _;
use vortex::cloud::Registry;
use vortex::dtype::DType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;
use vortex::expr::BoundExpression;
use vortex::expr::Expression;
use vortex::file::VortexFile;
use vortex::file::multi::open_cached;
use vortex::file::multi::parse_uri_or_path;
use vortex::io::VortexReadAt;
use vortex::io::compat::Compat;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
#[cfg(any(unix, windows))]
use vortex::io::object_store::ObjectStoreReadAt;
#[cfg(any(unix, windows))]
use vortex::io::object_store::object_path_from_literal;
use vortex::io::runtime::BlockingRuntime as _;
use vortex::layout::LayoutReaderRef;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::mask::Mask;
use vortex_morsel_scan::MorselScanBuilder;
use vortex_morsel_scan::ScanBackend;
use vortex_morsel_scan::ScanExecutorOptions;
use vortex_morsel_scan::scan_backend_from_env;
use vortex_utils::parallelism::get_available_parallelism;

use crate::RUNTIME;
use crate::SESSION;
use crate::column_statistics::ColumnStatistics;
use crate::column_statistics::ColumnStatisticsAggregate;
use crate::duckdb::BindResultRef;
use crate::duckdb::DataChunkRef;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;
use crate::projection::Filter;
use crate::projection::extract_schema_from_dtype;
use crate::table_function::BindState;
use crate::table_function::GlobalState;
use crate::table_function::LocalState;
use crate::table_function::Split;
use crate::table_function::convert_result;

// See src/table_function.rs for definition of table function state machine.
// Duckdb's inner state machine, called by table function state machine, is
// file reader state machine. It opens Vortex files, reads their contents
// and populates output data chunks with data from these files.
//
// Definitions:
// "global lock" - lock over all threads. Only one thread may access some
//   section.
// "file-local lock" - multiple threads may access different File's in
//   parallel, but only one thread can access a single File at a time.
//
// First there's bind phase in planning, called by one thread on first
// file expanded from glob, to get scan schema and estimate overall scan
// cardinality.
//
// `reader_open` -> `reader_bind` -> `reader_get_statistics`
//
// Then there's query runtime phase, called for all files in scan:
//
// `reader_open` -> `reader_initialize` ->
// `reader_try_initialize_scan` -> `reader_scan`
//
// `reader_get_progress_in_file` is called during `reader_scan` calls from a
// separate thread.

static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);
fn resolve_filesystem(url: &Url) -> VortexResult<(FileSystemRef, String)> {
    // Compat makes us use tokio which is very bad for local reads on
    // high-core machines because reads go into blocking pool
    if url.scheme() == "file" {
        return Ok((
            Arc::new(ObjectStoreFileSystem::local(RUNTIME.handle())),
            url.path().to_string(),
        ));
    }

    let (object_store, path) = REGISTRY.resolve(url)?;

    Ok((
        Arc::new(ObjectStoreFileSystem::new(
            Arc::new(Compat::new(object_store)),
            RUNTIME.handle(),
        )),
        path.to_string(),
    ))
}

async fn open_scan_file(url: &Url) -> VortexResult<(Arc<dyn VortexReadAt>, String)> {
    #[cfg(any(unix, windows))]
    if url.scheme() == "file" {
        let path = url.path().to_string();
        let object_path = object_path_from_literal(&path);
        let store = Arc::new(object_store::local::LocalFileSystem::new()) as Arc<dyn ObjectStore>;
        let metadata = store.head(&object_path).await?;
        // Pin the first matching file payload for the scan, avoiding a new descriptor per range.
        let read = ObjectStoreReadAt::new(store, object_path, RUNTIME.handle())
            .into_file_payload_promoting(metadata, *SCAN_DIAGNOSTICS_ENABLED)?;
        return Ok((Arc::new(read), path));
    }
    let (fs, path) = resolve_filesystem(url)?;
    Ok((fs.open_read(&path).await?, path))
}

/// Advance the current-thread runtime by running every task that is ready right now.
///
/// Morsel workers on DuckDB threads call this while they wait for segment I/O; the runtime has
/// no threads of its own, so the I/O driver task only progresses when a waiting thread ticks it.
/// Returns whether any task ran so the caller can park briefly instead of spinning.
fn drive_runtime_once() -> bool {
    let mut ran = false;
    while RUNTIME.try_tick() {
        ran = true;
    }
    ran
}

const Q6_FRONTIER_BUNDLE_ENV: &str = "VORTEX_DUCKDB_Q6_FRONTIER_BUNDLE";
const FRONTIER_BUNDLE_MIN_WAVES_ENV: &str = "VORTEX_DUCKDB_FRONTIER_BUNDLE_MIN_WAVES";
const Q6_FRONTIER_BUNDLE_SIZE: &str = "2";
const DEFAULT_FRONTIER_BUNDLE_MIN_WAVES: usize = 8;

static SCAN_DIAGNOSTICS_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("VORTEX_SCAN_DIAGNOSTICS").is_some_and(|value| value != "0"));

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FrontierBundleConfig {
    min_waves: usize,
}

fn frontier_bundle_config(
    enabled: bool,
    min_waves_value: Option<&str>,
) -> VortexResult<Option<FrontierBundleConfig>> {
    if !enabled {
        return Ok(None);
    }
    let min_waves = match min_waves_value {
        None => DEFAULT_FRONTIER_BUNDLE_MIN_WAVES,
        Some(value) => value.parse::<usize>().map_err(|err| {
            vortex_err!("{FRONTIER_BUNDLE_MIN_WAVES_ENV} must be a positive integer: {err}")
        })?,
    };
    if min_waves == 0 {
        return Err(vortex_err!(
            "{FRONTIER_BUNDLE_MIN_WAVES_ENV} must be a positive integer"
        ));
    }
    Ok(Some(FrontierBundleConfig { min_waves }))
}

fn frontier_bundle_parallelism_for_execution(
    config: Option<FrontierBundleConfig>,
    execution_threads: usize,
) -> Option<usize> {
    config.map(|_| execution_threads)
}

fn q6_frontier_bundle_enabled(backend: ScanBackend, opt_in: Option<&str>) -> VortexResult<bool> {
    if backend != ScanBackend::PushFrontier {
        return Ok(false);
    }
    let Some(bundle_size) = opt_in else {
        return Ok(false);
    };
    if bundle_size != Q6_FRONTIER_BUNDLE_SIZE {
        return Err(vortex_err!(
            "{Q6_FRONTIER_BUNDLE_ENV} must be {Q6_FRONTIER_BUNDLE_SIZE}"
        ));
    }
    Ok(true)
}

fn q6_frontier_bundle_config_from_env(
    backend: ScanBackend,
) -> VortexResult<Option<FrontierBundleConfig>> {
    if backend != ScanBackend::PushFrontier {
        return Ok(None);
    }
    let opt_in = std::env::var(Q6_FRONTIER_BUNDLE_ENV);
    let opt_in = match opt_in {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(err) => return Err(vortex_err!("{Q6_FRONTIER_BUNDLE_ENV} is invalid: {err}")),
    };
    let enabled = q6_frontier_bundle_enabled(backend, opt_in.as_deref())?;
    if !enabled {
        // The benchmark override is deliberately irrelevant unless the B2 experiment itself is
        // enabled, including for V1 and legacy Push.
        return Ok(None);
    }
    let min_waves_value = match std::env::var(FRONTIER_BUNDLE_MIN_WAVES_ENV) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(err) => {
            return Err(vortex_err!(
                "{FRONTIER_BUNDLE_MIN_WAVES_ENV} is invalid: {err}"
            ));
        }
    };
    frontier_bundle_config(true, min_waves_value.as_deref())
}

pub struct OpenFileReader {
    file: VortexFile,
    reader: Option<LayoutReaderRef>,
    backend: ScanBackend,
    morsel_options: ScanExecutorOptions,
    frontier_bundle_config: Option<FrontierBundleConfig>,
    /// File splits stored in inverse order
    pub splits: Vec<Split>,
    pub cache: ConversionCache,
    total_splits: usize,
}

impl OpenFileReader {
    async fn open(file_path: String) -> VortexResult<Self> {
        let backend = scan_backend_from_env()?;
        let host_parallelism = get_available_parallelism().unwrap_or(1);
        let frontier_bundle_config = q6_frontier_bundle_config_from_env(backend)?;
        let url = parse_uri_or_path(&file_path)?;
        let (file, path) = open_scan_file(&url).await?;
        let file = open_cached(&SESSION, file, &path, None, &|options| options).await?;
        let reader = (backend == ScanBackend::V1)
            .then(|| file.layout_reader())
            .transpose()?;
        let morsel_options = ScanExecutorOptions::default()
            .with_threads(host_parallelism)
            .with_external_threads(drive_runtime_once);
        Ok(OpenFileReader {
            file,
            reader,
            backend,
            morsel_options,
            frontier_bundle_config,
            cache: ConversionCache::default(),
            splits: vec![],
            total_splits: 0,
        })
    }

    fn can_skip(&self, filter: &Filter) -> VortexResult<bool> {
        let Some(filter) = &filter.filter else {
            return Ok(false);
        };
        if let Some(reader) = &self.reader {
            let row_count = reader.row_count();
            let row_range = 0..row_count;
            let mask = Mask::new_true(usize::try_from(row_count).unwrap_or(usize::MAX));
            let evaluation = reader.pruning_evaluation(&row_range, filter, mask)?;
            match evaluation.now_or_never() {
                Some(mask) => mask.map(|mask| mask.all_false()),
                None => Ok(false),
            }
        } else {
            self.file.can_prune(&unbind(filter)?)
        }
    }
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

/// Called once per file while initializing the scan under file-local lock.
/// Files are opened lazily.
pub fn reader_open(file_path: &str) -> VortexResult<OpenFileReader> {
    RUNTIME.block_on(OpenFileReader::open(file_path.to_owned()))
}

/// Called once per scan with first file without locks. Populates "result"
/// with first file schema which is the scan schema. Unlike Parquet, we don't
/// support schema evolution, so if any file schema doesn't match first schema,
/// we break.
pub fn reader_bind(file: &OpenFileReader, result: &mut BindResultRef) -> VortexResult<BindState> {
    let dtype = file.file.dtype().clone();
    let columns = extract_schema_from_dtype(&dtype)?;

    for column in &columns {
        result.add_result_column(&column.name, &column.logical_type);
    }

    Ok(BindState {
        dtype,
        first_file_row_count: file.file.row_count(),
        filters: vec![],
        columns,
        has_non_optional_filter: AtomicBool::new(false),
        aggregates: vec![],
    })
}

/// Called once per file by one thread under file-local lock. Determines
/// whether the opened file should be skipped. If this function returns false,
/// duckdb closes the file and doesn't call reader_try_initialize_scan on it.
pub fn reader_initialize(file: &mut OpenFileReader, global: &GlobalState) -> VortexResult<bool> {
    if file.can_skip(&global.filter)? {
        return Ok(true);
    }

    // Getting splits is non-trivial work so we prefer doing it here under file
    // lock and not in reader_try_initialize_scan under global lock.
    let ordered = global.file_row_number_column_pos.is_some();
    let filter = &global.filter;
    let mut splits = match file.backend {
        ScanBackend::V1 => {
            let reader = file
                .reader
                .as_ref()
                .vortex_expect("V1 file is missing its layout reader");
            let mut builder = ScanBuilder::new(SESSION.clone(), Arc::clone(reader))
                .with_projection(global.projection.clone())
                .with_ordered(ordered)
                .with_some_filter(filter.filter.clone())
                .with_selection(filter.row_selection.clone());
            if let Some(row_range) = filter.row_range.as_ref() {
                builder = builder.with_row_range(row_range.clone());
            }
            builder.build()?
        }
        ScanBackend::Push | ScanBackend::PushFrontier => {
            let configured_parallelism = frontier_bundle_parallelism_for_execution(
                file.frontier_bundle_config,
                global.execution_threads,
            );
            if *SCAN_DIAGNOSTICS_ENABLED {
                tracing::trace!(
                    target: "vortex_duckdb::frontier_bundle",
                    backend = ?file.backend,
                    configured = file.frontier_bundle_config.is_some(),
                    configured_parallelism = configured_parallelism.unwrap_or_default(),
                    min_bundle_waves = file
                        .frontier_bundle_config
                        .map_or(0, |config| config.min_waves),
                    has_filter = filter.filter.is_some(),
                    has_non_optional_table_filter = filter.has_non_optional_filter,
                    non_optional_table_filter_count = filter.non_optional_table_filter_count,
                    optional_table_filter_count = filter.optional_table_filter_count,
                    converted_table_filter_count = filter.converted_table_filter_count,
                    additional_filter_count = filter.additional_filter_count,
                    "configuring DuckDB external frontier bundle policy"
                );
            }
            let mut morsel_options = file.morsel_options.clone();
            if let Some(config) = file.frontier_bundle_config {
                // DuckDB does not currently propagate SQL LIMIT into MorselScanBuilder. This
                // opt-in is therefore an intentionally controlled no-LIMIT Q6 experiment, not a
                // generally safe production scheduling policy.
                morsel_options = morsel_options
                    .with_external_frontier_bundle_parallelism(
                        configured_parallelism.vortex_expect(
                            "configured frontier bundle is missing execution parallelism",
                        ),
                    )
                    .with_external_frontier_bundle_min_waves(config.min_waves);
            }
            let mut builder = MorselScanBuilder::new(
                SESSION.clone(),
                file.backend,
                Arc::clone(file.file.footer().layout()),
                file.file.segment_source(),
                &morsel_options,
            )?
            .with_projection(global.projection.clone())
            .with_ordered(ordered)
            .with_some_filter(filter.filter.clone())
            .with_selection(filter.row_selection.clone());
            if let Some(row_range) = filter.row_range.as_ref() {
                builder = builder.with_row_range(row_range.clone());
            }
            builder.build()?
        }
    };

    // threads take last element of file.splits so we need to reverse
    splits.reverse();
    file.total_splits = splits.len();
    file.splits = splits;
    Ok(false)
}

/// Called by all threads under global lock. If this function returns true,
/// thread calls reader_scan on this file. If this function returns false,
/// duckdb thinks file is exhausted, closes the file, and the first thread to
/// get "false" switches to next file.
pub fn reader_try_initialize_scan(file: &mut OpenFileReader, local: &mut LocalState) -> bool {
    let Some(split) = file.splits.pop() else {
        return false;
    };
    local.split = Some(split);
    true
}

/// Called by all threads operating on a file without locks. If this function
/// returns false, duckdb closes the file, and first thread to get "false"
/// switches to next file.
pub fn reader_scan(
    file: &OpenFileReader,
    global: &GlobalState,
    local: &mut LocalState,
    chunk: &mut DataChunkRef,
) -> VortexResult<bool> {
    if !global.aggregates.is_empty() {
        return reader_scan_aggregate(global, local);
    }

    if local.exporter.is_none() {
        let Some(split) = local.split.take() else {
            return Ok(false);
        };
        let Some(array) = RUNTIME.block_on(split)? else {
            // split is filtered
            return Ok(true);
        };
        let mut ctx = SESSION.create_execution_ctx();
        let array = convert_result(array, &mut ctx)?;
        local.exporter = Some(new_scan_exporter(file.backend, &array, &file.cache, ctx)?);
    }
    let exporter = local.exporter.as_mut().vortex_expect("no exporter");

    let has_more_data = exporter.export(chunk, global.file_row_number_column_pos)?;
    if !has_more_data {
        local.exporter = None;
    }
    Ok(true)
}

fn new_scan_exporter(
    backend: ScanBackend,
    array: &StructArray,
    file_cache: &ConversionCache,
    ctx: ExecutionCtx,
) -> VortexResult<ArrayExporter> {
    // Frontier splits do not share decoded arrays. Keep conversion reuse within this output
    // bundle; the exporters own their converted buffers after construction.
    let bundle_cache = (backend == ScanBackend::PushFrontier).then(ConversionCache::default);
    ArrayExporter::try_new(array, bundle_cache.as_ref().unwrap_or(file_cache), ctx)
}

fn reader_scan_aggregate(global: &GlobalState, local: &mut LocalState) -> VortexResult<bool> {
    let Some(split) = local.split.take() else {
        return Ok(false);
    };
    let Some(array) = RUNTIME.block_on(split)? else {
        // split is filtered
        return Ok(true);
    };

    let mut ctx = SESSION.create_execution_ctx();
    let array = convert_result(array, &mut ctx)?;

    for (position, partial) in local.partials.iter_mut() {
        partial.accumulate(array.unmasked_field(*position), &mut ctx)?;
    }

    if global.has_count_star {
        let len = array.len() as u64;
        global.row_count.fetch_add(len, Ordering::Relaxed);
    }

    Ok(true)
}

/// Called by one thread in plan phase without locks only on the first file
/// after calling reader_open and reader_bind on it.
pub fn reader_get_statistics(
    file: &OpenFileReader,
    bind: &BindState,
    column: &str,
) -> Option<ColumnStatistics> {
    if !bind.aggregates.is_empty() {
        return None;
    }

    let stats_sets = file.file.file_stats()?.stats_sets();

    let DType::Struct(fields, _) = file.file.dtype() else {
        return None;
    };
    let index = fields.find(column)?;
    let dtype = fields.field_by_index(index)?;

    let stats = ColumnStatisticsAggregate::new(stats_sets.get(index)?);
    match ColumnStatistics::try_from(&stats, dtype) {
        Ok(stats) => Some(stats),
        Err(e) => vortex_panic!(e),
    }
}

/// Called from a separate thread (not related to threads for Vortex
/// table function) under global lock.
pub fn reader_get_progress_in_file(file: &OpenFileReader) -> f64 {
    let total = file.total_splits;
    let left = file.splits.len();
    let denom = total + (total == 0) as usize;
    100.0 * (total - left) as f64 / denom as f64
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute as _;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::buffer::buffer;
    use vortex::error::VortexResult;
    use vortex_morsel_scan::ScanBackend;

    use super::DEFAULT_FRONTIER_BUNDLE_MIN_WAVES;
    use super::FrontierBundleConfig;
    use super::frontier_bundle_config;
    use super::frontier_bundle_parallelism_for_execution;
    use super::new_scan_exporter;
    use super::q6_frontier_bundle_enabled;
    use crate::SESSION;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::exporter::ConversionCache;

    #[cfg(unix)]
    mod local_reader_tests {
        use std::sync::Arc;

        use futures::StreamExt;
        use vortex::buffer::Alignment;
        use vortex::error::VortexResult;
        use vortex::file::multi::parse_uri_or_path;
        use vortex::io::ReadAtRequest;
        use vortex::io::runtime::BlockingRuntime as _;

        use super::super::open_scan_file;
        use crate::RUNTIME;

        #[test]
        fn reads_keep_the_open_file_after_path_replacement() -> VortexResult<()> {
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("scan.vortex");
            std::fs::write(&path, b"original file")?;
            let url = parse_uri_or_path(&path.to_string_lossy())?;
            RUNTIME.block_on(async {
                let (read, _) = open_scan_file(&url).await?;
                let first = read.read_at(0, 8, Alignment::new(1)).await?;
                assert_eq!(first.to_host().await.as_slice(), b"original");

                std::fs::rename(&path, directory.path().join("previous.vortex"))?;
                std::fs::write(&path, b"replacement with a different size")?;

                assert_eq!(read.size().await?, 13);
                let requests: Arc<[ReadAtRequest]> = Arc::from([
                    ReadAtRequest::new(0, 8, Alignment::new(1)),
                    ReadAtRequest::new(9, 4, Alignment::new(1)),
                ]);
                let mut results = read.read_ranges(requests).collect::<Vec<_>>().await;
                results.sort_unstable_by_key(|(request, _)| request.offset);
                assert_eq!(results.len(), 2);
                for ((_, result), expected) in
                    results.into_iter().zip([b"original".as_slice(), b"file"])
                {
                    assert_eq!(result?.to_host().await.as_slice(), expected);
                }
                assert!(read.read_at(12, 2, Alignment::new(1)).await.is_err());
                Ok(())
            })
        }

        #[test]
        fn rejects_replacement_before_the_first_read() -> VortexResult<()> {
            let directory = tempfile::tempdir()?;
            let path = directory.path().join("scan.vortex");
            std::fs::write(&path, b"original file")?;
            let url = parse_uri_or_path(&path.to_string_lossy())?;
            RUNTIME.block_on(async {
                let (read, _) = open_scan_file(&url).await?;
                std::fs::rename(&path, directory.path().join("previous.vortex"))?;
                std::fs::write(&path, b"replacement with a different size")?;
                assert!(read.read_at(0, 8, Alignment::new(1)).await.is_err());
                Ok(())
            })
        }
    }

    #[rstest::rstest]
    #[case::v1(ScanBackend::V1, 1)]
    #[case::push(ScanBackend::Push, 1)]
    #[case::frontier(ScanBackend::PushFrontier, 0)]
    fn scan_exporter_cache_lifetime(
        #[case] backend: ScanBackend,
        #[case] retained_per_split: usize,
    ) -> VortexResult<()> {
        let cache = ConversionCache::default();
        for split in 1..=3 {
            let values = VarBinViewArray::from_iter_str(["a", "b"]).into_array();
            let field = DictArray::try_new(buffer![1u8, 0, 1].into_array(), values)?.into_array();
            let array = StructArray::from_fields(&[("field", field)])?;
            let mut exporter =
                new_scan_exporter(backend, &array, &cache, SESSION.create_execution_ctx())?;
            drop(array);
            assert_eq!(cache.dict_cache.len(), split * retained_per_split);

            let mut chunk = DataChunk::new([LogicalType::varchar()]);
            assert!(exporter.export(&mut chunk, None)?);
            assert_eq!(
                String::try_from(&*chunk)?,
                "Chunk - [1 Columns]\n- DICTIONARY VARCHAR: 3 = [ b, a, b]\n"
            );
            assert!(!exporter.export(&mut chunk, None)?);
        }
        Ok(())
    }

    #[test]
    fn q6_frontier_bundle_controls_default_off_and_preserve_push() -> VortexResult<()> {
        assert!(!q6_frontier_bundle_enabled(
            ScanBackend::PushFrontier,
            None
        )?);
        assert!(!q6_frontier_bundle_enabled(ScanBackend::Push, Some("2"))?);
        Ok(())
    }

    #[test]
    fn q6_frontier_bundle_requires_only_the_fixed_size() -> VortexResult<()> {
        assert!(q6_frontier_bundle_enabled(
            ScanBackend::PushFrontier,
            Some("2")
        )?);
        assert!(q6_frontier_bundle_enabled(ScanBackend::PushFrontier, Some("3")).is_err());
        Ok(())
    }

    #[test]
    fn frontier_bundle_min_waves_is_strict_and_defaults_to_eight() -> VortexResult<()> {
        assert_eq!(frontier_bundle_config(false, Some("invalid"))?, None);
        assert!(frontier_bundle_config(true, Some("invalid")).is_err());
        assert!(frontier_bundle_config(true, Some("0")).is_err());
        assert_eq!(
            frontier_bundle_config(true, None)?,
            Some(FrontierBundleConfig {
                min_waves: DEFAULT_FRONTIER_BUNDLE_MIN_WAVES,
            })
        );
        assert_eq!(
            frontier_bundle_config(true, Some("1"))?,
            Some(FrontierBundleConfig { min_waves: 1 })
        );
        assert_eq!(
            frontier_bundle_config(true, Some("8"))?,
            Some(FrontierBundleConfig { min_waves: 8 })
        );
        Ok(())
    }

    #[test]
    fn file_initialization_uses_each_execution_thread_snapshot() {
        let config = Some(FrontierBundleConfig { min_waves: 8 });
        assert_eq!(
            frontier_bundle_parallelism_for_execution(config, 1),
            Some(1)
        );
        assert_eq!(
            frontier_bundle_parallelism_for_execution(config, 14),
            Some(14)
        );
        assert_eq!(
            frontier_bundle_parallelism_for_execution(config, 2),
            Some(2)
        );
        assert_eq!(frontier_bundle_parallelism_for_execution(None, 14), None);
    }
}
