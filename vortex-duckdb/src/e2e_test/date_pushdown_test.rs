// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pushdown tests for `DATE` columns compared against `TIMESTAMP` bounds.
//!
//! DuckDB widens a `DATE` column to `TIMESTAMP` whenever the other side is one, which is what
//! `date '1993-07-01' + interval '3' month` produces. The scan only sees a column reference
//! once that cast has been folded into the literal, so every filter here has to both push and
//! keep counting what DuckDB itself counts.

use num_traits::AsPrimitive;
use rstest::rstest;
use tempfile::NamedTempFile;

use crate::cpp::duckdb_string_t;
use crate::duckdb::Connection;
use crate::duckdb::Database;

/// Two years of consecutive dates, spanning the bounds used below.
const ROWS: &str = "SELECT DATE '1993-01-01' + INTERVAL (i) DAY AS d FROM range(0, 730) t(i)";

fn database_connection() -> Connection {
    let db = Database::open_in_memory().unwrap();
    db.register_vortex_scan_replacement().unwrap();
    crate::initialize(&db).unwrap();
    db.connect().unwrap()
}

/// A connection holding [`ROWS`] both as a vortex file and as a native `dates` table, so the
/// same predicate can be run against each.
fn date_fixture() -> (Connection, NamedTempFile) {
    let conn = database_connection();
    let file = NamedTempFile::with_suffix(".vortex").unwrap();
    let path = file.path().to_string_lossy().to_string();

    conn.query(&format!("COPY ({ROWS}) TO '{path}' (FORMAT VORTEX);"))
        .unwrap();
    conn.query(&format!("CREATE TABLE dates AS {ROWS};"))
        .unwrap();
    (conn, file)
}

/// Read back the single `i64` of a one-row, one-column query.
fn query_i64(conn: &Connection, query: &str) -> i64 {
    let result = conn.query(query).unwrap();
    let chunk = result.into_iter().next().unwrap();
    chunk
        .get_vector(0)
        .as_slice_with_len::<i64>(chunk.len().as_())[0]
}

/// The `EXPLAIN` physical plan of `query` as one string.
fn explain_plan(conn: &Connection, query: &str) -> String {
    let explain = conn.query(&format!("EXPLAIN {query}")).unwrap();
    let mut plan = String::new();
    for mut chunk in explain {
        let len = chunk.len().as_();
        let vec = chunk.get_vector_mut(1);
        for value in unsafe { vec.as_slice_mut::<duckdb_string_t>(len) } {
            let slice: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    crate::cpp::duckdb_string_t_data(&raw mut *value) as _,
                    crate::cpp::duckdb_string_t_length(*value) as usize,
                )
            };
            plan.push_str(&String::from_utf8_lossy(slice));
        }
    }
    plan
}

/// Count the rows of the vortex file matching `filter`, and of the native table for comparison.
fn counts(conn: &Connection, path: &str, filter: &str) -> (i64, i64) {
    (
        query_i64(
            conn,
            &format!("SELECT count(*) FROM '{path}' WHERE {filter}"),
        ),
        query_i64(conn, &format!("SELECT count(*) FROM dates WHERE {filter}")),
    )
}

/// Every bound TPC-H states as `date + interval` reaches the scan, and still counts what DuckDB
/// counts natively. Q4, Q15 and Q20 each lose their upper bound without the fold.
#[rstest]
#[case::q4_range("d >= DATE '1993-07-01' AND d < DATE '1993-07-01' + INTERVAL '3' MONTH")]
#[case::q15_range("d >= DATE '1993-01-01' AND d < DATE '1993-01-01' + INTERVAL '3' MONTH")]
#[case::q20_range("d >= DATE '1993-01-01' AND d < DATE '1993-01-01' + INTERVAL '1' YEAR")]
#[case::upper_only("d < DATE '1993-07-01' + INTERVAL '3' MONTH")]
#[case::lower_only("d >= DATE '1993-07-01' + INTERVAL '3' MONTH")]
#[case::midnight_lt("d < TIMESTAMP '1993-07-01 00:00:00'")]
#[case::midnight_lte("d <= TIMESTAMP '1993-07-01 00:00:00'")]
#[case::midnight_gt("d > TIMESTAMP '1993-07-01 00:00:00'")]
#[case::midnight_gte("d >= TIMESTAMP '1993-07-01 00:00:00'")]
#[case::midnight_eq("d = TIMESTAMP '1993-07-01 00:00:00'")]
#[case::reversed("TIMESTAMP '1993-07-01 00:00:00' > d")]
fn date_timestamp_bound_pushes(#[case] filter: &str) {
    let (conn, file) = date_fixture();
    let path = file.path().to_string_lossy().to_string();

    let (vortex, native) = counts(&conn, &path, filter);
    assert_eq!(vortex, native, "`{filter}` disagrees with DuckDB");
    assert!(
        vortex > 0,
        "`{filter}` matches nothing, so it proves little"
    );

    let plan = explain_plan(
        &conn,
        &format!("SELECT count(*) FROM '{path}' WHERE {filter}"),
    );
    assert!(
        !plan.contains("FILTER"),
        "`{filter}` was not pushed:\n{plan}"
    );
}

/// A bound strictly inside a day has no exact `DATE` equivalent for `=` and `<>`, and rounds to
/// the day for the inequalities. Whether or not it pushes, the count must not move.
#[rstest]
#[case::lt("d < TIMESTAMP '1993-07-01 12:00:00'")]
#[case::lte("d <= TIMESTAMP '1993-07-01 12:00:00'")]
#[case::gt("d > TIMESTAMP '1993-07-01 12:00:00'")]
#[case::gte("d >= TIMESTAMP '1993-07-01 12:00:00'")]
#[case::eq("d = TIMESTAMP '1993-07-01 12:00:00'")]
#[case::not_eq("d <> TIMESTAMP '1993-07-01 12:00:00'")]
fn bound_inside_a_day_keeps_its_meaning(#[case] filter: &str) {
    let (conn, file) = date_fixture();
    let path = file.path().to_string_lossy().to_string();

    let (vortex, native) = counts(&conn, &path, filter);
    assert_eq!(vortex, native, "`{filter}` disagrees with DuckDB");
}

/// `TIMESTAMP WITH TIME ZONE` bounds depend on the session timezone, so they are deliberately
/// left for DuckDB. They must stay correct, and stay above the scan.
#[rstest]
#[case("Europe/London")]
#[case("America/New_York")]
fn timestamptz_bound_is_left_to_duckdb(#[case] timezone: &str) {
    let (conn, file) = date_fixture();
    let path = file.path().to_string_lossy().to_string();
    conn.query(&format!("SET TimeZone = '{timezone}';"))
        .unwrap();
    let filter = "d < TIMESTAMPTZ '1993-07-01 00:00:00'";

    let (vortex, native) = counts(&conn, &path, filter);
    assert_eq!(vortex, native, "`{filter}` disagrees with DuckDB");

    let plan = explain_plan(
        &conn,
        &format!("SELECT count(*) FROM '{path}' WHERE {filter}"),
    );
    assert!(
        plan.contains("FILTER"),
        "a timezone-dependent bound must not be folded to a DATE:\n{plan}"
    );
}
