// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Persistent DuckDB driver for real-file ClickBench scan comparisons.

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
    parquet: bool,
    #[arg(long)]
    diagnostics: bool,
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
    if args.diagnostics {
        vortex_bench::setup_logging_and_tracing(false, false)?;
    }
    fs::create_dir_all(&args.output)?;
    let client = DuckClient::new_in_memory()?;
    client.execute_query(&format!("SET threads = {}", args.threads))?;
    client.execute_query("SET memory_limit = '48GB'")?;
    // Keep spill from silently changing the storage comparison.
    client.execute_query("SET temp_directory = ''")?;

    let start = Instant::now();
    let (reader, extension) = if args.parquet {
        ("read_parquet", "parquet")
    } else {
        ("read_vortex", "vortex")
    };
    let mut paths = fs::read_dir(&args.data)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|path| path.extension().is_some_and(|ext| ext == extension));
    paths.sort();
    anyhow::ensure!(paths.len() == 100, "expected all 100 ClickBench shards");
    let files = paths
        .iter()
        .map(|path| literal(&path.to_string_lossy()))
        .collect::<Vec<_>>()
        .join(",");
    client.execute_query(&format!(
        "CREATE VIEW hits AS SELECT * FROM {reader}([{files}])"
    ))?;
    client.execute_query("CREATE MACRO octet_length(a) AS strlen(a)")?;
    let queries = fs::read_to_string(&args.sql)?;
    let queries = queries
        .split(';')
        .map(str::trim)
        .filter(|sql| !sql.is_empty())
        .collect::<Vec<_>>();
    anyhow::ensure!(queries.len() == 43, "expected ClickBench Q0-Q42");
    let load_ns = start.elapsed().as_nanos();
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
    // separate persistent clients without timing their one-time setup or result output.
    for command in io::stdin().lock().lines() {
        let command = command?;
        if command == "quit" {
            break;
        }
        let mut fields = command.splitn(3, '\t');
        let action = fields.next().context("missing action")?;
        let query: usize = fields.next().context("missing query")?.parse()?;
        anyhow::ensure!(query < queries.len(), "query must be Q0-Q42");
        let prefix = fields.next().context("missing artifact prefix")?;
        let sql = queries[query];
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
            let (elapsed, result) = client.execute_query_result(sql)?;
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
