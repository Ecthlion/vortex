// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DuckDB benchmark client that loads Vortex as an out-of-tree extension.

use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_char;
use std::ffi::c_void;
use std::path::Path;
use std::path::PathBuf;
use std::ptr;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use tracing::trace;
use vortex_bench::Benchmark;
use vortex_bench::Format;
use vortex_bench::IdempotentPath;
use vortex_bench::generate_duckdb_registration_sql;
use vortex_bench::runner::BenchmarkQueryResult;

type DuckDatabase = *mut c_void;
type DuckConnection = *mut c_void;
type DuckConfig = *mut c_void;

#[repr(C)]
struct DuckResult {
    deprecated_column_count: u64,
    deprecated_row_count: u64,
    deprecated_rows_changed: u64,
    deprecated_columns: *mut c_void,
    deprecated_error_message: *mut c_char,
    internal_data: *mut c_void,
}

#[repr(C)]
struct DuckString {
    data: *mut c_char,
    size: u64,
}

unsafe extern "C" {
    fn duckdb_create_config(config: *mut DuckConfig) -> u32;
    fn duckdb_set_config(config: DuckConfig, name: *const c_char, value: *const c_char) -> u32;
    fn duckdb_destroy_config(config: *mut DuckConfig);
    fn duckdb_open_ext(
        path: *const c_char,
        database: *mut DuckDatabase,
        config: DuckConfig,
        error: *mut *mut c_char,
    ) -> u32;
    fn duckdb_close(database: *mut DuckDatabase);
    fn duckdb_connect(database: DuckDatabase, connection: *mut DuckConnection) -> u32;
    fn duckdb_disconnect(connection: *mut DuckConnection);
    fn duckdb_query(
        connection: DuckConnection,
        query: *const c_char,
        result: *mut DuckResult,
    ) -> u32;
    fn duckdb_result_error(result: *mut DuckResult) -> *const c_char;
    fn duckdb_destroy_result(result: *mut DuckResult);
    fn duckdb_column_count(result: *mut DuckResult) -> u64;
    fn duckdb_column_name(result: *mut DuckResult, column: u64) -> *const c_char;
    fn duckdb_column_logical_type(result: *mut DuckResult, column: u64) -> *mut c_void;
    fn duckdb_destroy_logical_type(logical_type: *mut *mut c_void);
    fn duckdb_row_count(result: *mut DuckResult) -> u64;
    fn duckdb_rows_changed(result: *mut DuckResult) -> u64;
    fn duckdb_value_is_null(result: *mut DuckResult, column: u64, row: u64) -> bool;
    fn duckdb_value_string(result: *mut DuckResult, column: u64, row: u64) -> DuckString;
    fn duckdb_free(ptr: *mut c_void);
}

unsafe extern "C-unwind" {
    fn duckdb_vx_logical_type_stringify(logical_type: *mut c_void) -> *mut c_char;
}

/// DuckDB context for benchmarks.
pub struct DuckClient {
    db: Option<DuckDatabase>,
    connection: Option<DuckConnection>,
    pub db_path: PathBuf,
    pub threads: Option<usize>,
    init_sql: Vec<String>,
}

impl DuckClient {
    pub fn new(
        benchmark: &dyn Benchmark,
        format: Format,
        delete_database: bool,
        threads: Option<usize>,
    ) -> Result<Self> {
        let data_url = benchmark.data_url();
        let base_path = if data_url.scheme() == "file" {
            data_url
                .to_file_path()
                .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", data_url))?
        } else {
            format!("{name}/{format}/", name = benchmark.dataset_name()).to_data_path()
        };
        let dir = base_path.join(format.name());
        let db_path = dir.join("duckdb.db");

        if format != Format::OnDiskDuckDB {
            std::fs::create_dir_all(&dir)?;
        } else if data_url.scheme() != "file" {
            anyhow::bail!("DuckDB format requires local data prepared by data-gen");
        } else if !db_path.exists() {
            anyhow::bail!(
                "prepared DuckDB database is missing at {}",
                db_path.display()
            );
        }
        if delete_database && db_path.exists() && format != Format::OnDiskDuckDB {
            std::fs::remove_file(&db_path)?;
        }

        let (db, connection) = Self::open_and_setup_database(Some(&db_path), threads)?;
        Ok(Self {
            db: Some(db),
            connection: Some(connection),
            db_path,
            threads,
            init_sql: Vec::new(),
        })
    }

    fn query(&self, query: &str) -> Result<DuckQueryResult> {
        let query = CString::new(query).context("query contains a NUL byte")?;
        let mut result: DuckResult = unsafe { std::mem::zeroed() };
        let status = unsafe {
            duckdb_query(
                self.connection.context("DuckDB connection is closed")?,
                query.as_ptr(),
                &raw mut result,
            )
        };
        if status != 0 {
            let error = result_error(&mut result);
            unsafe { duckdb_destroy_result(&raw mut result) };
            anyhow::bail!("failed to execute query: {error}");
        }
        Ok(DuckQueryResult {
            result,
            prepared_display: None,
        })
    }

    fn open_and_setup_database(
        path: Option<&Path>,
        threads: Option<usize>,
    ) -> Result<(DuckDatabase, DuckConnection)> {
        let mut config = ptr::null_mut();
        if unsafe { duckdb_create_config(&raw mut config) } != 0 {
            anyhow::bail!("failed to create DuckDB config");
        }
        let option = CString::new("allow_unsigned_extensions")?;
        let enabled = CString::new("true")?;
        if unsafe { duckdb_set_config(config, option.as_ptr(), enabled.as_ptr()) } != 0 {
            unsafe { duckdb_destroy_config(&raw mut config) };
            anyhow::bail!("failed to enable unsigned DuckDB extensions");
        }

        let path = path
            .map(|path| CString::new(path.to_string_lossy().as_bytes()))
            .transpose()?;
        let mut db = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            duckdb_open_ext(
                path.as_ref().map_or(ptr::null(), |path| path.as_ptr()),
                &raw mut db,
                config,
                &raw mut error,
            )
        };
        unsafe { duckdb_destroy_config(&raw mut config) };
        if status != 0 {
            let message = if error.is_null() {
                "unknown DuckDB open error".to_string()
            } else {
                let message = unsafe { CStr::from_ptr(error) }
                    .to_string_lossy()
                    .into_owned();
                unsafe { duckdb_free(error.cast()) };
                message
            };
            anyhow::bail!("failed to open DuckDB: {message}");
        }

        let mut connection = ptr::null_mut();
        if unsafe { duckdb_connect(db, &raw mut connection) } != 0 {
            unsafe { duckdb_close(&raw mut db) };
            anyhow::bail!("failed to connect to DuckDB");
        }
        if let Ok(extension) = std::env::var("VORTEX_DUCKDB_EXTENSION") {
            query_raw(
                connection,
                &format!("LOAD '{}'", extension.replace('\'', "''")),
            )?;
        } else {
            // Link the workspace extension into this benchmark so local runs exercise the
            // implementation under test. An explicitly supplied loadable extension still wins.
            unsafe { vortex_duckdb::initialize_extension_from_raw(db) };
        }
        if let Some(thread_count) = threads {
            query_raw(connection, &format!("SET threads = {thread_count}"))?;
        }
        query_raw(connection, "SET parquet_metadata_cache = true")?;
        Ok((db, connection))
    }

    pub fn set_init_sql(&mut self, statements: Vec<String>) -> Result<()> {
        for statement in &statements {
            self.query(statement)?;
        }
        self.init_sql = statements;
        Ok(())
    }

    pub fn reopen(&mut self) -> Result<()> {
        self.close();
        let (db, connection) = Self::open_and_setup_database(Some(&self.db_path), self.threads)?;
        self.db = Some(db);
        self.connection = Some(connection);
        for statement in &self.init_sql {
            self.query(statement)?;
        }
        Ok(())
    }

    pub fn new_in_memory() -> Result<Self> {
        let db_path = PathBuf::from(":memory:");
        let (db, connection) = Self::open_and_setup_database(None, None)?;
        Ok(Self {
            db: Some(db),
            connection: Some(connection),
            db_path,
            threads: None,
            init_sql: Vec::new(),
        })
    }

    pub fn execute_query(&self, query: &str) -> Result<(usize, Option<Duration>)> {
        trace!("execute duckdb query: {query}");
        let start = Instant::now();
        let result = self.query(query)?;
        Ok((result.row_count(), Some(start.elapsed())))
    }

    pub fn register_tables<B: Benchmark + ?Sized>(
        &self,
        benchmark: &B,
        file_format: Format,
    ) -> Result<()> {
        if file_format == Format::OnDiskDuckDB {
            return Ok(());
        }
        let object_type = match file_format {
            Format::Parquet
            | Format::OnDiskVortex
            | Format::VortexCompact
            | Format::VortexSpatialNative => "VIEW",
            Format::OnDiskDuckDB => "TABLE",
            format => anyhow::bail!("Format {format} isn't supported for DuckDB"),
        };
        let format_url = benchmark.format_path(file_format, benchmark.data_url())?;
        let base_dir = format_url
            .as_str()
            .strip_prefix("file://")
            .unwrap_or(format_url.as_str())
            .trim_end_matches('/');
        for statement in
            generate_duckdb_registration_sql(benchmark, base_dir, file_format, object_type)
        {
            self.query(&statement)?;
        }
        Ok(())
    }

    pub fn execute_query_result(&self, query: &str) -> Result<(Option<Duration>, DuckQueryResult)> {
        trace!("execute duckdb query: {query}");
        let start = Instant::now();
        let result = self.query(query)?;
        Ok((Some(start.elapsed()), result))
    }

    fn close(&mut self) {
        if let Some(mut connection) = self.connection.take() {
            unsafe { duckdb_disconnect(&raw mut connection) };
        }
        if let Some(mut db) = self.db.take() {
            unsafe { duckdb_close(&raw mut db) };
        }
    }
}

impl Drop for DuckClient {
    fn drop(&mut self) {
        self.close();
    }
}

fn query_raw(connection: DuckConnection, query: &str) -> Result<()> {
    let query = CString::new(query)?;
    let mut result: DuckResult = unsafe { std::mem::zeroed() };
    let status = unsafe { duckdb_query(connection, query.as_ptr(), &raw mut result) };
    if status != 0 {
        let error = result_error(&mut result);
        unsafe { duckdb_destroy_result(&raw mut result) };
        anyhow::bail!("failed to execute query: {error}");
    }
    unsafe { duckdb_destroy_result(&raw mut result) };
    Ok(())
}

fn result_error(result: &mut DuckResult) -> String {
    unsafe {
        let error = duckdb_result_error(result);
        if error.is_null() {
            "unknown DuckDB error".to_string()
        } else {
            CStr::from_ptr(error).to_string_lossy().into_owned()
        }
    }
}

pub struct DuckQueryResult {
    result: DuckResult,
    prepared_display: Option<String>,
}

impl DuckQueryResult {
    fn result_row_count(&self) -> usize {
        let result = (&raw const self.result).cast_mut();
        let changed = unsafe { duckdb_rows_changed(result) };
        usize::try_from(if changed == 0 {
            unsafe { duckdb_row_count(result) }
        } else {
            changed
        })
        .unwrap_or(0)
    }

    /// Render the materialized result in a deterministic, schema-aware, row-order-independent
    /// text format. Duplicate rows remain distinct entries in the sorted row multiset.
    ///
    /// DuckDB's legacy value accessor is intentionally used only by opt-in inspection and EXPLAIN
    /// modes. Normal benchmark execution never pays the per-cell conversion cost.
    pub fn deterministic_display(&self) -> Result<String> {
        use std::fmt::Write as _;

        let result = (&raw const self.result).cast_mut();
        let column_count = unsafe { duckdb_column_count(result) };
        let row_count = unsafe { duckdb_row_count(result) };
        let mut output = String::new();
        writeln!(
            output,
            "duckdb-result-v1 columns={column_count} rows={row_count}"
        )?;

        for column in 0..column_count {
            let name_ptr = unsafe { duckdb_column_name(result, column) };
            if name_ptr.is_null() {
                anyhow::bail!("DuckDB returned no name for result column {column}");
            }
            let name = unsafe { CStr::from_ptr(name_ptr) }
                .to_str()
                .with_context(|| format!("result column {column} name is not UTF-8"))?;

            let mut logical_type = unsafe { duckdb_column_logical_type(result, column) };
            if logical_type.is_null() {
                anyhow::bail!("DuckDB returned no logical type for result column {column}");
            }
            let type_ptr = unsafe { duckdb_vx_logical_type_stringify(logical_type) };
            let logical_type_result = take_duckdb_c_string(type_ptr).with_context(|| {
                format!("failed to stringify logical type for result column {column}")
            });
            unsafe { duckdb_destroy_logical_type(&raw mut logical_type) };
            let logical_type = logical_type_result?;

            writeln!(
                output,
                "column={column} name={} logical_type={}",
                quoted(name),
                quoted(&logical_type)
            )?;
        }

        let mut rows = Vec::with_capacity(usize::try_from(row_count)?);
        for row in 0..row_count {
            let mut encoded_row = String::new();
            for column in 0..column_count {
                if unsafe { duckdb_value_is_null(result, column, row) } {
                    writeln!(encoded_row, "value={column} null")?;
                    continue;
                }
                let value = take_duckdb_string(unsafe { duckdb_value_string(result, column, row) })
                    .with_context(|| {
                        format!("failed to read result value at row {row}, column {column}")
                    })?;
                writeln!(encoded_row, "value={column} text={}", quoted(&value))?;
            }
            rows.push(encoded_row);
        }
        rows.sort_unstable();
        for (row, encoded_row) in rows.into_iter().enumerate() {
            writeln!(output, "row={row}")?;
            output.push_str(&encoded_row);
        }

        Ok(output)
    }

    /// Materialize and retain the deterministic display so EXPLAIN validation and display use the
    /// exact same conversion result.
    pub fn prepare_deterministic_display(&mut self) -> Result<&str> {
        if self.prepared_display.is_none() {
            self.prepared_display = Some(self.deterministic_display()?);
        }
        self.prepared_display
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("prepared DuckDB display is missing"))
    }
}

fn quoted(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    escaped.extend(value.escape_default());
    escaped.push('"');
    escaped
}

fn take_duckdb_c_string(value: *mut c_char) -> Result<String> {
    if value.is_null() {
        anyhow::bail!("DuckDB returned a null string pointer");
    }
    let result = unsafe { CStr::from_ptr(value) }
        .to_str()
        .context("DuckDB returned non-UTF-8 text")
        .map(str::to_owned);
    unsafe { duckdb_free(value.cast()) };
    result
}

fn take_duckdb_string(value: DuckString) -> Result<String> {
    if value.data.is_null() {
        anyhow::bail!("DuckDB returned a null string pointer");
    }
    let result = (|| {
        let size = usize::try_from(value.size).context("DuckDB string is too large")?;
        std::str::from_utf8(unsafe { std::slice::from_raw_parts(value.data.cast::<u8>(), size) })
            .context("DuckDB returned non-UTF-8 text")
            .map(str::to_owned)
    })();
    unsafe { duckdb_free(value.data.cast()) };
    result
}

impl Drop for DuckQueryResult {
    fn drop(&mut self) {
        unsafe { duckdb_destroy_result(&raw mut self.result) };
    }
}

impl BenchmarkQueryResult for DuckQueryResult {
    fn row_count(&self) -> usize {
        self.result_row_count()
    }

    fn display(mut self) -> String {
        self.prepared_display.take().unwrap_or_else(|| {
            self.deterministic_display()
                .unwrap_or_else(|error| format!("failed to display DuckDB result: {error:#}"))
        })
    }
}
