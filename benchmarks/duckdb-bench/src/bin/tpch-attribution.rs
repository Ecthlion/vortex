// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Persistent DuckDB driver for real-file versus materialized-memory TPC-H attribution.

use std::fs;
use std::io;
use std::io::BufRead;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use clap::Parser;
use duckdb_bench::DuckClient;
use vortex_bench::runner::BenchmarkQueryResult;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    sql: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 14)]
    threads: usize,
    #[arg(long)]
    materialize: bool,
    #[arg(long)]
    parquet: bool,
}

fn literal(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

fn capture(client: &DuckClient, sql: &str, path: &Path) -> anyhow::Result<()> {
    let (_, result) = client.execute_query_result(sql)?;
    fs::write(path, result.deterministic_display()?)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.output)?;
    let client = DuckClient::new_in_memory()?;
    client.execute_query(&format!("SET threads = {}", args.threads))?;
    client.execute_query("SET memory_limit = '48GB'")?;
    // A memory baseline must fail rather than silently spill its tables or operators to disk.
    client.execute_query("SET temp_directory = ''")?;

    let start = Instant::now();
    for table in [
        "region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem",
    ] {
        let (reader, extension) = if args.parquet {
            ("read_parquet", "parquet")
        } else {
            ("read_vortex", "vortex")
        };
        let path = args.data.join(format!("{table}.{extension}"));
        anyhow::ensure!(path.is_file(), "missing real input: {}", path.display());
        let source = format!("{reader}({})", literal(&path.to_string_lossy()));
        let sql = if args.materialize {
            format!("CREATE TABLE {table} AS SELECT * FROM {source}")
        } else {
            format!("CREATE VIEW {table} AS SELECT * FROM {source}")
        };
        let table_start = Instant::now();
        client.execute_query(&sql)?;
        eprintln!("loaded {table} in {} ms", table_start.elapsed().as_millis());
    }
    let load_ns = start.elapsed().as_nanos();
    capture(
        &client,
        "SELECT count(*) AS lineitem_rows FROM lineitem",
        &args.output.join("data-count.stdout"),
    )?;
    capture(
        &client,
        "SELECT database_name, path FROM duckdb_databases() WHERE NOT internal",
        &args.output.join("databases.stdout"),
    )?;
    capture(
        &client,
        "SELECT name, value FROM duckdb_settings() WHERE name IN ('threads', 'memory_limit', 'temp_directory') ORDER BY name",
        &args.output.join("settings.stdout"),
    )?;
    capture(
        &client,
        "SELECT * FROM duckdb_memory()",
        &args.output.join("memory-after-load.stdout"),
    )?;
    println!("READY\t{load_ns}");
    io::stdout().flush()?;

    // Each command is action<TAB>query-number<TAB>artifact-prefix. A controller can interleave
    // separate persistent clients without timing their one-time materialization or result output.
    for command in io::stdin().lock().lines() {
        let command = command?;
        if command == "quit" {
            break;
        }
        let mut fields = command.splitn(3, '\t');
        let action = fields.next().context("missing action")?;
        let query: usize = fields.next().context("missing query")?.parse()?;
        anyhow::ensure!((1..=22).contains(&query), "query must be Q1-Q22");
        let prefix = fields.next().context("missing artifact prefix")?;
        let sql = fs::read_to_string(args.sql.join(format!("q{query}.sql")))?;
        // The native-table differential exposes a Q22 correctness failure with common_subplan.
        // Apply the same explicit workaround to every arm and retain normal settings otherwise.
        client.execute_query(if query == 22 {
            "SET disabled_optimizers = 'common_subplan'"
        } else {
            "SET disabled_optimizers = ''"
        })?;
        if action == "plan" {
            capture(
                &client,
                &format!("EXPLAIN {sql}"),
                Path::new(&format!("{prefix}.plan")),
            )?;
            println!("PLAN\t{query}");
        } else {
            anyhow::ensure!(matches!(action, "run" | "profile"), "unknown action");
            if action == "profile" {
                client.execute_query("SET profiling_mode = 'detailed'")?;
                client.execute_query("PRAGMA enable_profiling = 'json'")?;
                client.execute_query(&format!(
                    "SET profiling_output = {}",
                    literal(&format!("{prefix}.profile.json"))
                ))?;
            }
            let (elapsed, result) = client.execute_query_result(&sql)?;
            let ns = elapsed.context("missing query duration")?.as_nanos();
            let rows = result.row_count();
            let display = result.deterministic_display()?;
            drop(result);
            if action == "profile" {
                client.execute_query("PRAGMA disable_profiling")?;
            }
            fs::write(
                format!("{prefix}.stdout"),
                format!("=== Q{query} [vortex] result ===\n{display}"),
            )?;
            println!("DONE\t{query}\t{ns}\t{rows}");
        }
        io::stdout().flush()?;
    }
    capture(
        &client,
        "SELECT * FROM duckdb_memory()",
        &args.output.join("memory-at-end.stdout"),
    )?;
    Ok(())
}
