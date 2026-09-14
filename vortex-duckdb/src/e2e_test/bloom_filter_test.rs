// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests for the bloom filters DuckDB hash joins push into the Vortex scan.

use anyhow::Result;
use anyhow::anyhow;
use num_traits::AsPrimitive;
use tempfile::NamedTempFile;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::runtime::BlockingRuntime;

use crate::RUNTIME;
use crate::SESSION;
use crate::cpp;
use crate::duckdb::Connection;
use crate::duckdb::Database;

/// Rows in the probe-side file.
const PROBE_ROWS: i64 = 200_000;
/// One in this many probe rows joins.
const BUILD_SELECTIVITY: i64 = 1_000;
/// Keys are spread out so that the build side is too wide for DuckDB to build a perfect hash
/// table, which it prefers over a bloom filter.
const KEY_STRIDE: i64 = 20;

fn database_connection() -> Connection {
    let db = Database::open_in_memory().unwrap();
    db.register_vortex_scan_replacement().unwrap();
    crate::initialize(&db).unwrap();
    db.connect().unwrap()
}

fn write_vortex_file(keys: impl IntoArray) -> NamedTempFile {
    RUNTIME.block_on(async {
        let temp_file = tempfile::Builder::new()
            .suffix(".vortex")
            .tempfile()
            .unwrap();

        let payload = PrimitiveArray::from_iter(0..PROBE_ROWS).into_array();
        let array =
            StructArray::from_fields(&[("key", keys.into_array()), ("payload", payload)]).unwrap();

        let mut file = async_fs::File::create(&temp_file).await.unwrap();
        SESSION
            .write_options()
            .write(&mut file, array.into_array().to_array_stream())
            .await
            .unwrap();
        temp_file
    })
}

/// The probe side of the join, whose keys run `0, 20, 40, ...`.
fn write_probe_file() -> NamedTempFile {
    write_vortex_file(PrimitiveArray::from_iter(
        (0..PROBE_ROWS).map(|i| i * KEY_STRIDE),
    ))
}

/// Creates the build side, which a filter narrows down to one key in [`BUILD_SELECTIVITY`]. That
/// filter is what makes DuckDB consider a bloom filter at all.
fn create_build_table(connection: &Connection) -> Result<()> {
    connection.query("CREATE TABLE build (key BIGINT, tag VARCHAR)")?;
    connection.query(&format!(
        "INSERT INTO build SELECT i * {KEY_STRIDE}, \
         CASE WHEN i % {BUILD_SELECTIVITY} = 0 THEN 'hit' ELSE 'miss' END \
         FROM range(0, {PROBE_ROWS}) t(i)"
    ))?;
    Ok(())
}

fn join_query(file: &NamedTempFile) -> String {
    let path = file.path().to_string_lossy();
    format!(
        "SELECT count(*), sum(p.payload), sum(p.key) \
         FROM '{path}' p JOIN build b ON p.key = b.key WHERE b.tag = 'hit'"
    )
}

/// Runs a query returning a single row of three `BIGINT` aggregates.
fn aggregates(connection: &Connection, query: &str) -> Result<Vec<i64>> {
    let result = connection.query(query)?;
    let mut chunk = result
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("query returned no rows"))?;
    Ok((0..3)
        .map(|column| chunk.get_vector_mut(column).as_slice_with_len::<i64>(1)[0])
        .collect())
}

fn explain_analyze_json(connection: &Connection, query: &str) -> Result<String> {
    let result = connection.query(&format!("EXPLAIN (ANALYZE, FORMAT JSON) {query}"))?;
    let mut plan = String::new();
    for mut chunk in result {
        let len: usize = chunk.len().as_();
        let vector = chunk.get_vector_mut(1);
        for value in unsafe { vector.as_slice_mut::<cpp::duckdb_string_t>(len) } {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    cpp::duckdb_string_t_data(&raw mut *value).cast::<u8>(),
                    cpp::duckdb_string_t_length(*value).as_(),
                )
            };
            plan.push_str(&String::from_utf8_lossy(bytes));
        }
    }
    Ok(plan)
}

/// The rows the Vortex scan emitted, read out of a profiled plan.
fn vortex_scan_cardinality(plan: &str) -> Result<i64> {
    const CARDINALITY: &str = r#""operator_cardinality":"#;

    let scan = plan
        .find(r#""Function": "Vortex Scan""#)
        .ok_or_else(|| anyhow!("no Vortex scan in plan:\n{plan}"))?;
    let cardinality = plan[scan..]
        .find(CARDINALITY)
        .ok_or_else(|| anyhow!("no cardinality for the Vortex scan in plan:\n{plan}"))?;

    Ok(plan[scan + cardinality + CARDINALITY.len()..]
        .split(',')
        .next()
        .unwrap_or_default()
        .trim()
        .parse()?)
}

/// The join's bloom filter has to reach the Vortex scan and remove rows there, so that they are
/// never decoded, exported to DuckDB, or probed against the hash table.
#[test]
fn bloom_filter_removes_rows_inside_the_scan() -> Result<()> {
    let file = write_probe_file();
    let connection = database_connection();
    create_build_table(&connection)?;

    let plan = explain_analyze_json(&connection, &join_query(&file))?;
    assert!(
        plan.contains("IN BF"),
        "DuckDB did not push a bloom filter into the scan:\n{plan}"
    );

    let matching = PROBE_ROWS / BUILD_SELECTIVITY;
    let scanned = vortex_scan_cardinality(&plan)?;
    assert!(
        scanned < matching * 10,
        "the scan emitted {scanned} of {PROBE_ROWS} rows, so the bloom filter was not applied; \
         only {matching} of them join"
    );
    Ok(())
}

/// A row the filter removes must be a row that could not have joined, so the answer has to be
/// the one DuckDB computes for the same data with join filter pushdown turned off.
#[test]
fn bloom_filter_preserves_join_results() -> Result<()> {
    let file = write_probe_file();
    let connection = database_connection();
    create_build_table(&connection)?;

    let filtered = aggregates(&connection, &join_query(&file))?;

    connection.query("SET disabled_optimizers='join_filter_pushdown'")?;
    let unfiltered = aggregates(&connection, &join_query(&file))?;

    assert_eq!(filtered, unfiltered);
    assert_eq!(filtered[0], PROBE_ROWS / BUILD_SELECTIVITY);
    Ok(())
}

/// NULL keys cannot join on equality. DuckDB probes the filter with the hash it assigns to NULL,
/// and the scan has to reach the same answer.
#[test]
fn bloom_filter_handles_null_keys() -> Result<()> {
    let file = write_vortex_file(PrimitiveArray::from_option_iter(
        (0..PROBE_ROWS).map(|i| (i % 7 != 0).then_some(i * KEY_STRIDE)),
    ));
    let connection = database_connection();
    create_build_table(&connection)?;

    let filtered = aggregates(&connection, &join_query(&file))?;

    connection.query("SET disabled_optimizers='join_filter_pushdown'")?;
    let unfiltered = aggregates(&connection, &join_query(&file))?;

    assert_eq!(filtered, unfiltered);
    Ok(())
}

/// String keys hash through DuckDB's byte hash rather than its numeric one.
#[test]
fn bloom_filter_preserves_join_results_for_string_keys() -> Result<()> {
    let file = write_probe_file();
    let path = file.path().to_string_lossy().to_string();
    let connection = database_connection();
    create_build_table(&connection)?;

    let query = format!(
        "SELECT count(*), sum(p.payload), sum(p.key) \
         FROM '{path}' p JOIN build b ON p.key::VARCHAR = b.key::VARCHAR WHERE b.tag = 'hit'"
    );
    let filtered = aggregates(&connection, &query)?;

    connection.query("SET disabled_optimizers='join_filter_pushdown'")?;
    let unfiltered = aggregates(&connection, &query)?;

    assert_eq!(filtered, unfiltered);
    assert_eq!(filtered[0], PROBE_ROWS / BUILD_SELECTIVITY);
    Ok(())
}
