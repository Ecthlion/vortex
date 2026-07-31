// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bindings for scanning many Vortex files as a single logical table.
//!
//! Where [`crate::file`] opens exactly one file, this module resolves directories, glob patterns
//! and explicit lists of paths into a [`MultiLayoutDataSource`] so that a directory of Vortex
//! files can be read the same way a directory of Parquet files is.

use std::path::Path;
use std::sync::Arc;

use arrow_schema::Schema;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use itertools::Itertools;
use object_store::ObjectStore;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use pyo3_object_store::PyObjectStore;
use vortex::array::ArrayRef;
use vortex::array::iter::ArrayIterator;
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::dtype::DType;
use vortex::dtype::FieldNames;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::multi::MultiFileDataSource;
use vortex::file::multi::parse_uri_or_path;
use vortex::io::filesystem::FileSystemRef;
use vortex::io::object_store::ObjectStoreFileSystem;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::multi::MultiLayoutDataSource;
use vortex::scan::DataSource;
use vortex::scan::ScanRequest;
use vortex_arrow::ArrowSessionExt;

use crate::arrow::FromPyArrow;
use crate::arrow::IntoPyArrow;
use crate::arrow::ToPyArrow;
use crate::current_runtime;
use crate::dataset::PyVortexDataset;
use crate::dtype::PyDType;
use crate::error::PyVortexResult;
use crate::expr::PyExpr;
use crate::file::PyIntoProjection;
use crate::install_module;
use crate::iter::PyArrayIterator;
use crate::object_store::resolve::ResolvedStore;
use crate::object_store::resolve::resolve_store;
use crate::session::session;

/// The file extension appended when a source resolves to a directory.
const VORTEX_EXTENSION: &str = "vortex";

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "files")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.files", &m)?;

    m.add_function(wrap_pyfunction!(open_files, &m)?)?;
    m.add_class::<PyVortexFiles>()?;

    Ok(())
}

/// Open many Vortex files as a single logical table.
#[pyfunction]
#[pyo3(signature = (paths, *, store = None))]
pub fn open_files(
    py: Python,
    paths: PyIntoGlobs,
    store: Option<PyObjectStore>,
) -> PyVortexResult<PyVortexFiles> {
    let globs = paths.0;
    let store = store.map(|store| store.into_inner());

    let source = py.detach(move || {
        let session = session();
        let mut builder = MultiFileDataSource::new(session.clone());
        for glob in &globs {
            let (glob, fs) = resolve_glob(glob, store.clone())?;
            builder = builder.with_glob(glob, fs);
        }
        current_runtime().block_on(builder.build())
    })?;

    Ok(PyVortexFiles {
        dtype: source.dtype().clone(),
        source,
    })
}

/// Resolve a user-supplied directory, glob pattern or path into a glob and its filesystem.
///
/// Returning `None` for the filesystem defers to [`MultiFileDataSource`], which creates a local
/// filesystem for the glob.
fn resolve_glob(
    source: &str,
    store: Option<Arc<dyn ObjectStore>>,
) -> VortexResult<(String, Option<FileSystemRef>)> {
    // An explicit store makes the source store-relative, so it is used verbatim.
    if let Some(store) = store {
        let fs = ObjectStoreFileSystem::new(store, current_runtime().handle());
        return Ok((add_directory_suffix(source, Local::No), Some(Arc::new(fs))));
    }

    let url = parse_uri_or_path(source)?;
    if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| vortex_err!("invalid file URL: {source}"))?;
        // Keep the un-encoded path so that names containing spaces or other characters that
        // `Url` percent-encodes still resolve.
        let path = path
            .to_str()
            .ok_or_else(|| vortex_err!("path is not valid UTF-8: {source}"))?;
        return Ok((add_directory_suffix(path, Local::Yes), None));
    }

    let mut base_url = url.clone();
    base_url.set_path("");
    let ResolvedStore::ObjectStore(store, _) = resolve_store(base_url.as_str(), None)? else {
        vortex_bail!("expected an object store for URL: {source}");
    };
    let fs = ObjectStoreFileSystem::new(store, current_runtime().handle());
    Ok((
        add_directory_suffix(url.path(), Local::No),
        Some(Arc::new(fs)),
    ))
}

/// Whether a source names a path on the local filesystem, and so can be probed with `is_dir`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Local {
    Yes,
    No,
}

/// Expand a directory into a glob over the Vortex files it contains.
///
/// A source is treated as a directory when it ends in `/`, or when it names an existing local
/// directory. Sources that already contain glob characters are left alone, as is anything that
/// looks like a single file. Remote sources are never probed for existence, so a remote directory
/// has to be spelled with a trailing `/` (or an explicit glob) to be expanded.
fn add_directory_suffix(source: &str, local: Local) -> String {
    if source.contains(['*', '?', '[']) {
        return source.to_string();
    }
    if let Some(prefix) = source.strip_suffix('/') {
        return format!("{prefix}/*.{VORTEX_EXTENSION}");
    }
    if local == Local::Yes && Path::new(source).is_dir() {
        return format!("{source}/*.{VORTEX_EXTENSION}");
    }
    source.to_string()
}

/// Many Vortex files scanned as a single logical table.
#[pyclass(name = "VortexFiles", module = "vortex", frozen)]
pub struct PyVortexFiles {
    source: MultiLayoutDataSource,
    dtype: DType,
}

#[pymethods]
impl PyVortexFiles {
    /// The number of files backing this table.
    #[getter]
    fn file_count(&self) -> usize {
        self.source.children().len()
    }

    /// The dtype shared by every file.
    #[getter]
    #[pyo3(name = "dtype")]
    fn dtype_(slf: Bound<Self>) -> PyResult<Bound<PyDType>> {
        PyDType::init(slf.py(), slf.get().dtype.clone())
    }

    /// The Arrow schema shared by every file.
    fn schema(slf: Bound<Self>) -> PyVortexResult<Py<PyAny>> {
        let schema = Arc::new(session().arrow().to_arrow_schema(&slf.get().dtype)?);
        Ok(schema.to_pyarrow(slf.py())?)
    }

    /// The number of rows matching the given filter across all files.
    #[pyo3(signature = (*, expr = None))]
    fn count_rows(slf: Bound<Self>, expr: Option<PyExpr>) -> PyVortexResult<u64> {
        let source = slf.get().source.clone();
        let filter = expr.map(|e| e.into_inner());

        Ok(slf.py().detach(move || {
            let iter = array_iter(
                &source,
                MultiScanOptions {
                    projection: select(FieldNames::empty(), root()),
                    filter,
                    ordered: false,
                    ..Default::default()
                },
            )?;
            iter.map_ok(|array| array.len() as u64)
                .process_results(|iter| iter.sum::<u64>())
        })?)
    }

    /// Scan every file, returning a :class:`vortex.ArrayIterator` over the chunks.
    #[pyo3(signature = (projection = None, *, expr = None, limit = None, ordered = true))]
    fn scan(
        slf: Bound<Self>,
        projection: Option<PyIntoProjection>,
        expr: Option<PyExpr>,
        limit: Option<u64>,
        ordered: bool,
    ) -> PyVortexResult<PyArrayIterator> {
        let source = slf.get().source.clone();
        let projection = projection.map_or_else(root, |p| p.into_inner());
        let filter = expr.map(|e| e.into_inner());

        slf.py().detach(move || {
            let iter = array_iter(
                &source,
                MultiScanOptions {
                    projection,
                    filter,
                    limit,
                    ordered,
                    ..Default::default()
                },
            )?;
            Ok(PyArrayIterator::new(Box::new(iter)))
        })
    }

    /// Scan every file as a :class:`pyarrow.RecordBatchReader`.
    #[pyo3(signature = (projection = None, *, expr = None, limit = None, schema = None, ordered = true))]
    fn to_arrow(
        slf: Bound<Self>,
        projection: Option<PyIntoProjection>,
        expr: Option<PyExpr>,
        limit: Option<u64>,
        schema: Option<&Bound<PyAny>>,
        ordered: bool,
    ) -> PyVortexResult<Py<PyAny>> {
        let source = slf.get().source.clone();
        let projection = projection.map_or_else(root, |p| p.into_inner());
        let filter = expr.map(|e| e.into_inner());
        let schema = schema
            .map(|schema| Schema::from_pyarrow(&schema.as_borrowed()))
            .transpose()?
            .map(Arc::new);

        let reader = slf.py().detach(move || {
            let iter = array_iter(
                &source,
                MultiScanOptions {
                    projection,
                    filter,
                    limit,
                    ordered,
                    ..Default::default()
                },
            )?;
            let schema = match schema {
                Some(schema) => schema,
                None => Arc::new(session().arrow().to_arrow_schema(iter.dtype())?),
            };
            VortexResult::Ok(crate::iter::record_batch_reader(Box::new(iter), schema))
        })?;

        Ok(reader.into_pyarrow(slf.py())?)
    }

    /// Scan these files using the :class:`pyarrow.dataset.Dataset` API.
    fn to_dataset(&self) -> PyVortexResult<PyVortexDataset> {
        Ok(PyVortexDataset::try_new_multi(self.source.clone())?)
    }
}

/// Options for scanning a [`MultiLayoutDataSource`] into chunks with [`array_iter`].
pub(crate) struct MultiScanOptions {
    /// The projection expression, defaulting to all columns.
    pub(crate) projection: Expression,
    /// The predicate used to filter rows.
    pub(crate) filter: Option<Expression>,
    /// The maximum number of rows to read after filtering.
    pub(crate) limit: Option<u64>,
    /// Restrict the scan to a single partition (one file), as used by dataset fragments.
    pub(crate) partition: Option<u64>,
    /// Slice chunks larger than this many rows.
    pub(crate) batch_size: Option<usize>,
    /// Whether chunks are yielded in file order, or interleaved as files finish.
    pub(crate) ordered: bool,
}

impl Default for MultiScanOptions {
    fn default() -> Self {
        Self {
            projection: root(),
            filter: None,
            limit: None,
            partition: None,
            batch_size: None,
            ordered: true,
        }
    }
}

/// Build a blocking iterator over the chunks of every file in the data source.
///
/// Partitions are flattened in order when `ordered`, and otherwise interleaved so that files are
/// read concurrently.
pub(crate) fn array_iter(
    source: &MultiLayoutDataSource,
    options: MultiScanOptions,
) -> VortexResult<Box<dyn ArrayIterator + Send>> {
    let source = source.clone();
    let ordered = options.ordered;
    // The limit is pushed into each partition, which bounds the work per file, but it must also
    // be applied across the concatenated partitions to bound the total row count.
    let request = ScanRequest {
        projection: options.projection,
        filter: options.filter,
        limit: options.limit,
        ordered,
        partition_range: options.partition.map(|i| i..i + 1),
        ..Default::default()
    };

    let runtime = current_runtime();
    let scan = runtime.block_on(async move { source.scan(request).await })?;
    let dtype = scan.dtype().clone();

    let partitions = scan
        .partitions()
        .map(|partition| partition.and_then(|partition| partition.execute()));
    let chunks = if ordered {
        partitions.try_flatten().boxed()
    } else {
        partitions.try_flatten_unordered(None).boxed()
    };
    let chunks = truncate(chunks, options.limit);
    let chunks = split_larger(chunks, options.batch_size);

    let stream = ArrayStreamAdapter::new(dtype.clone(), chunks);
    Ok(Box::new(ArrayIteratorAdapter::new(
        dtype,
        runtime.block_on_stream(stream),
    )))
}

/// Slice chunks larger than `batch_size` rows into consecutive chunks of at most that many.
fn split_larger(
    chunks: stream::BoxStream<'static, VortexResult<ArrayRef>>,
    batch_size: Option<usize>,
) -> stream::BoxStream<'static, VortexResult<ArrayRef>> {
    let Some(batch_size) = batch_size.map(|b| b.max(1)) else {
        return chunks;
    };

    chunks
        .map_ok(move |chunk| {
            let len = chunk.len();
            stream::iter(
                (0..len)
                    .step_by(batch_size)
                    .map(move |start| chunk.slice(start..(start + batch_size).min(len))),
            )
        })
        .try_flatten()
        .boxed()
}

/// Stop the stream once `limit` rows have been yielded, slicing the chunk that crosses the limit.
fn truncate(
    chunks: stream::BoxStream<'static, VortexResult<ArrayRef>>,
    limit: Option<u64>,
) -> stream::BoxStream<'static, VortexResult<ArrayRef>> {
    let Some(limit) = limit else {
        return chunks;
    };
    // Chunk lengths are `usize`, so track the outstanding row count in the same units. A limit
    // beyond `usize::MAX` cannot be reached by an in-memory scan, so saturating is enough.
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);

    chunks
        .scan(limit, |remaining, chunk| {
            let item = match chunk {
                Ok(chunk) => {
                    if *remaining == 0 {
                        return futures::future::ready(None);
                    }
                    let take = (*remaining).min(chunk.len());
                    *remaining -= take;
                    if take < chunk.len() {
                        chunk.slice(0..take)
                    } else {
                        Ok(chunk)
                    }
                }
                Err(e) => Err(e),
            };
            futures::future::ready(Some(item))
        })
        .boxed()
}

/// A list of directories, glob patterns or file paths.
pub struct PyIntoGlobs(Vec<String>);

impl<'py> FromPyObject<'_, 'py> for PyIntoGlobs {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(single) = ob.cast::<PyString>() {
            return Ok(PyIntoGlobs(vec![single.to_str()?.to_string()]));
        }

        if let Ok(globs) = ob.extract::<Vec<String>>() {
            if globs.is_empty() {
                return Err(PyTypeError::new_err(
                    "paths must contain at least one directory, glob or file path",
                ));
            }
            return Ok(PyIntoGlobs(globs));
        }

        Err(PyTypeError::new_err(
            "paths must be a string or a list of strings",
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::Local;
    use super::add_directory_suffix;

    // Paths are chosen so they cannot exist locally, keeping the `Local::Yes` cases decided by
    // the trailing slash rather than by the filesystem.
    #[rstest]
    #[case::glob_untouched("/vortex-test/*.vortex", "/vortex-test/*.vortex")]
    #[case::question_mark_untouched("/vortex-test/part-?.bin", "/vortex-test/part-?.bin")]
    #[case::char_class_untouched("/vortex-test/part-[01]", "/vortex-test/part-[01]")]
    #[case::trailing_slash("/vortex-test/", "/vortex-test/*.vortex")]
    #[case::file_untouched("/vortex-test/table.vortex", "/vortex-test/table.vortex")]
    fn directory_suffix(#[case] source: &str, #[case] expected: &str) {
        for local in [Local::Yes, Local::No] {
            assert_eq!(add_directory_suffix(source, local), expected, "{local:?}");
        }
    }

    #[test]
    fn local_directory_needs_no_trailing_slash() {
        let dir = std::env::temp_dir();
        let dir = dir
            .to_str()
            .expect("temp dir is UTF-8")
            .trim_end_matches('/');

        assert_eq!(
            add_directory_suffix(dir, Local::Yes),
            format!("{dir}/*.vortex")
        );
        // A remote source is not probed, so the same string stays an exact path.
        assert_eq!(add_directory_suffix(dir, Local::No), dir);
    }
}
