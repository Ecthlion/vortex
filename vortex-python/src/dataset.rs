// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::RecordBatchReader;
use arrow_schema::SchemaRef;
use itertools::Itertools;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::iter::ArrayIterator;
use vortex::array::iter::ArrayIteratorExt;
use vortex::dtype::FieldName;
use vortex::dtype::FieldNames;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::root;
use vortex::expr::select;
use vortex::expr::stats::Precision;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::io::runtime::BlockingRuntime;
use vortex::layout::scan::multi::MultiLayoutDataSource;
use vortex::layout::scan::split_by::SplitBy;
use vortex::scan::DataSource;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_arrow::ArrowSessionExt;

use crate::arrays::PyArrayRef;
use crate::arrow::IntoPyArrow;
use crate::arrow::ToPyArrow;
use crate::current_runtime;
use crate::error::PyVortexResult;
use crate::expr::PyExpr;
use crate::files::MultiScanOptions;
use crate::files::array_iter;
use crate::install_module;
use crate::object_store::resolve::ResolvedStore;
use crate::object_store::resolve::resolve_store;
use crate::session::session;

pub(crate) fn init(py: Python, parent: &Bound<PyModule>) -> PyResult<()> {
    let m = PyModule::new(py, "dataset")?;
    parent.add_submodule(&m)?;
    install_module("vortex._lib.dataset", &m)?;

    m.add_class::<PyVortexDataset>()?;

    m.add_function(wrap_pyfunction!(dataset_from_url, &m)?)?;

    Ok(())
}

pub fn read_array_from_reader(
    vortex_file: &VortexFile,
    projection: Expression,
    filter: Option<Expression>,
    indices: Option<ArrayRef>,
    row_range: Option<(u64, u64)>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let projection = projection
        .optimize_recursive(vortex_file.dtype())?
        .bind(vortex_file.dtype())?;
    let filter = filter
        .map(|filter| {
            filter
                .optimize_recursive(vortex_file.dtype())?
                .bind(vortex_file.dtype())
        })
        .transpose()?;
    let mut scan = vortex_file.scan()?.with_projection(projection);

    if let Some(filter) = filter {
        scan = scan.with_filter(filter);
    }

    if let Some(indices) = indices {
        let primitive = indices.execute::<PrimitiveArray>(ctx)?;
        let indices = primitive.into_buffer();
        scan = scan.with_row_indices(StrictSortedBuffer::try_new(indices)?);
    }

    if let Some((l, r)) = row_range {
        scan = scan.with_row_range(l..r);
    }

    let runtime = current_runtime();
    scan.into_array_iter(&runtime)?.read_all()
}

fn projection_from_python(columns: Option<Vec<Bound<PyAny>>>) -> PyResult<Expression> {
    fn field_from_pyany(field: &Bound<PyAny>) -> PyResult<FieldName> {
        if field.clone().is_instance_of::<PyString>() {
            Ok(FieldName::from(field.cast::<PyString>()?.to_str()?))
        } else {
            Err(PyTypeError::new_err(format!(
                "projection: expected list of strings or None, but found: {field}.",
            )))
        }
    }

    Ok(match columns {
        None => root(),
        Some(columns) => {
            let fields: Vec<_> = columns
                .iter()
                .map(field_from_pyany)
                .collect::<PyResult<_>>()?;
            select(FieldNames::from(fields), root())
        }
    })
}

fn filter_from_python(row_filter: Option<&Bound<PyExpr>>) -> Option<Expression> {
    row_filter.map(|x| x.borrow().inner().clone())
}

#[pyclass(name = "VortexDataset", module = "dataset")]
pub struct PyVortexDataset {
    source: DatasetSource,
    schema: SchemaRef,
}

/// The rows behind a [`PyVortexDataset`]: a single Vortex file, or many files scanned as one.
enum DatasetSource {
    File(VortexFile),
    Multi(MultiLayoutDataSource),
}

impl PyVortexDataset {
    pub fn try_new(vxf: VortexFile) -> VortexResult<Self> {
        let schema = Arc::new(session().arrow().to_arrow_schema(vxf.dtype())?);
        Ok(Self {
            source: DatasetSource::File(vxf),
            schema,
        })
    }

    /// Create a dataset over many files scanned as a single logical table.
    ///
    /// Each file becomes one partition, exposed to Python as one dataset fragment.
    pub fn try_new_multi(source: MultiLayoutDataSource) -> VortexResult<Self> {
        let schema = Arc::new(session().arrow().to_arrow_schema(source.dtype())?);
        Ok(Self {
            source: DatasetSource::Multi(source),
            schema,
        })
    }

    pub async fn from_url(
        url: &str,
        store: Option<Arc<dyn object_store::ObjectStore>>,
    ) -> VortexResult<Self> {
        let session = session();
        let vxf = match resolve_store(url, store)? {
            ResolvedStore::ObjectStore(store, path) => {
                session
                    .open_options()
                    .open_object_store(&store, path)
                    .await?
            }
            ResolvedStore::Path(path) => session.open_options().open_path(path).await?,
        };
        PyVortexDataset::try_new(vxf)
    }

    pub(crate) fn to_array_inner<'py>(
        &self,
        py: Python<'py>,
        columns: Option<Vec<Bound<'py, PyAny>>>,
        row_filter: Option<&Bound<'py, PyExpr>>,
        indices: Option<PyArrayRef>,
        row_range: Option<(u64, u64)>,
        partition: Option<u64>,
    ) -> PyVortexResult<PyArrayRef> {
        let projection = projection_from_python(columns)?;
        let filter = filter_from_python(row_filter);
        let indices = indices.map(|i| i.into_inner());

        let array = match &self.source {
            DatasetSource::File(vxf) => {
                let vxf = vxf.clone();
                py.detach(move || {
                    check_file_arguments(partition)?;
                    let session = session();
                    let mut ctx = session.create_execution_ctx();
                    read_array_from_reader(&vxf, projection, filter, indices, row_range, &mut ctx)
                })?
            }
            DatasetSource::Multi(source) => {
                let source = source.clone();
                py.detach(move || {
                    check_multi_arguments(indices.is_some(), row_range)?;
                    array_iter(
                        &source,
                        MultiScanOptions {
                            projection,
                            filter,
                            partition,
                            ..Default::default()
                        },
                    )?
                    .read_all()
                })?
            }
        };
        Ok(PyArrayRef::from(array))
    }
}

/// Reject arguments that only apply to multi-file datasets.
fn check_file_arguments(partition: Option<u64>) -> VortexResult<()> {
    if partition.is_some() {
        vortex_bail!("partition is only supported for multi-file datasets");
    }
    Ok(())
}

/// Reject arguments that multi-file datasets do not support.
fn check_multi_arguments(indices: bool, row_range: Option<(u64, u64)>) -> VortexResult<()> {
    if indices {
        vortex_bail!("indices are not supported for multi-file datasets");
    }
    if row_range.is_some() {
        vortex_bail!(
            "row_range is not supported for multi-file datasets; scan a partition instead"
        );
    }
    Ok(())
}

#[pymethods]
impl PyVortexDataset {
    fn schema(self_: PyRef<Self>) -> PyResult<Py<PyAny>> {
        Arc::clone(&self_.schema).to_pyarrow(self_.py())
    }

    #[pyo3(signature = (*, columns = None, row_filter = None, indices = None, row_range = None, partition = None))]
    pub fn to_array<'py>(
        self_: PyRef<'py, Self>,
        columns: Option<Vec<Bound<'py, PyAny>>>,
        row_filter: Option<&Bound<'py, PyExpr>>,
        indices: Option<PyArrayRef>,
        row_range: Option<(u64, u64)>,
        partition: Option<u64>,
    ) -> PyVortexResult<PyArrayRef> {
        self_.to_array_inner(
            self_.py(),
            columns,
            row_filter,
            indices,
            row_range,
            partition,
        )
    }

    #[pyo3(signature = (*, columns = None, row_filter = None, split_by = None, row_range = None, partition = None))]
    pub fn to_record_batch_reader(
        self_: PyRef<Self>,
        columns: Option<Vec<Bound<'_, PyAny>>>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        split_by: Option<usize>,
        row_range: Option<(u64, u64)>,
        partition: Option<u64>,
    ) -> PyVortexResult<Py<PyAny>> {
        let projection = projection_from_python(columns)?;
        let filter = filter_from_python(row_filter);

        let reader = match &self_.source {
            DatasetSource::File(vxf) => {
                let vxf = vxf.clone();
                self_.py().detach(move || {
                    check_file_arguments(partition)?;
                    let projection = projection
                        .optimize_recursive(vxf.dtype())?
                        .bind(vxf.dtype())?;
                    let filter = filter
                        .map(|filter| filter.optimize_recursive(vxf.dtype())?.bind(vxf.dtype()))
                        .transpose()?;
                    let mut scan = vxf
                        .scan()?
                        .with_projection(projection)
                        .with_some_filter(filter)
                        .with_split_by(split_by.map(SplitBy::RowCount).unwrap_or_default());
                    if let Some((l, r)) = row_range {
                        scan = scan.with_row_range(l..r);
                    }

                    let schema = Arc::new(session().arrow().to_arrow_schema(&scan.dtype()?)?);
                    let runtime = current_runtime();
                    let reader: Box<dyn RecordBatchReader + Send> =
                        Box::new(scan.into_record_batch_reader(schema, &runtime)?);
                    VortexResult::Ok(reader)
                })?
            }
            DatasetSource::Multi(source) => {
                let source = source.clone();
                self_.py().detach(move || {
                    check_multi_arguments(false, row_range)?;
                    let iter = array_iter(
                        &source,
                        MultiScanOptions {
                            projection,
                            filter,
                            partition,
                            batch_size: split_by,
                            ..Default::default()
                        },
                    )?;
                    let schema = Arc::new(session().arrow().to_arrow_schema(iter.dtype())?);
                    VortexResult::Ok(crate::iter::record_batch_reader(Box::new(iter), schema))
                })?
            }
        };

        Ok(reader.into_pyarrow(self_.py())?)
    }

    /// The number of rows matching the filter.
    #[pyo3(signature = (*, row_filter = None, split_by = None, row_range = None, partition = None))]
    pub fn count_rows(
        self_: PyRef<Self>,
        row_filter: Option<&Bound<'_, PyExpr>>,
        split_by: Option<usize>,
        row_range: Option<(u64, u64)>,
        partition: Option<u64>,
    ) -> PyVortexResult<usize> {
        match &self_.source {
            DatasetSource::File(vxf) => {
                check_file_arguments(partition)?;
                if row_filter.is_none() {
                    let row_count = match row_range {
                        Some(range) => range.1 - range.0,
                        None => vxf.row_count(),
                    };
                    return row_count
                        .try_into()
                        .map_err(|e| PyValueError::new_err(e).into());
                }

                let vxf = vxf.clone();
                let filter = filter_from_python(row_filter);
                let n_rows: usize = self_.py().detach(move || {
                    let projection = select(FieldNames::empty(), root())
                        .optimize_recursive(vxf.dtype())?
                        .bind(vxf.dtype())?;
                    let filter = filter
                        .map(|filter| filter.optimize_recursive(vxf.dtype())?.bind(vxf.dtype()))
                        .transpose()?;
                    let mut scan = vxf
                        .scan()?
                        .with_projection(projection)
                        .with_some_filter(filter)
                        .with_split_by(split_by.map(SplitBy::RowCount).unwrap_or_default());
                    if let Some((l, r)) = row_range {
                        scan = scan.with_row_range(l..r);
                    }

                    let runtime = current_runtime();
                    scan.into_array_iter(&runtime)?
                        .map_ok(|array| array.len())
                        .process_results(|iter| iter.sum())
                })?;

                Ok(n_rows)
            }
            DatasetSource::Multi(source) => {
                // Row counts of deferred files are unknown until they open, so only a filter-free
                // count over already-opened files avoids a scan.
                if row_filter.is_none()
                    && partition.is_none()
                    && row_range.is_none()
                    && let Precision::Exact(n) = source.row_count()
                {
                    return n.try_into().map_err(|e| PyValueError::new_err(e).into());
                }

                let source = source.clone();
                let filter = filter_from_python(row_filter);
                let n_rows: usize = self_.py().detach(move || {
                    check_multi_arguments(false, row_range)?;
                    array_iter(
                        &source,
                        MultiScanOptions {
                            projection: select(FieldNames::empty(), root()),
                            filter,
                            partition,
                            ordered: false,
                            ..Default::default()
                        },
                    )?
                    .map_ok(|array| array.len())
                    .process_results(|iter| iter.sum())
                })?;

                Ok(n_rows)
            }
        }
    }

    /// The natural splits of this Dataset, used to build one fragment per split.
    ///
    /// Only single-file datasets have row-range splits; multi-file datasets are split by
    /// partition instead, see [`Self::partition_count`].
    #[pyo3(signature = (*))]
    pub fn splits(&self) -> PyVortexResult<Vec<(u64, u64)>> {
        match &self.source {
            DatasetSource::File(vxf) => Ok(vxf
                .splits()?
                .into_iter()
                .map(|x| (x.start, x.end))
                .collect()),
            DatasetSource::Multi(_) => Err(vortex_err!(
                "multi-file datasets split by partition, not by row range; see partition_count"
            )
            .into()),
        }
    }

    /// The number of partitions (files) of a multi-file dataset, or `None` for a single file.
    #[pyo3(signature = (*))]
    pub fn partition_count(&self) -> Option<usize> {
        match &self.source {
            DatasetSource::File(_) => None,
            DatasetSource::Multi(source) => Some(source.children().len()),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (url, *, store = None))]
pub fn dataset_from_url(
    py: Python,
    url: &str,
    store: Option<Bound<PyAny>>,
) -> PyVortexResult<PyVortexDataset> {
    let store_arc = if let Some(store_obj) = store {
        let py_store: pyo3_object_store::PyObjectStore = store_obj.extract()?;
        Some(py_store.into_inner())
    } else {
        None
    };

    Ok(py.detach(move || current_runtime().block_on(PyVortexDataset::from_url(url, store_arc)))?)
}
