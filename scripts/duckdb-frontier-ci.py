#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Paired real-file DuckDB V1/frontier SQL comparison, independent of historical baselines."""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import selectors
import statistics
import subprocess
import time

ARMS = ("v1", "w8", "prefetch")
REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append(path, value):
    with path.open("a") as sink:
        sink.write(json.dumps(value, sort_keys=True) + "\n")


def decode(text):
    text = re.sub(r"\\u\{([0-9a-fA-F]+)\}", lambda m: f"\\U{int(m[1], 16):08x}", text)
    return ast.literal_eval(text)


def result(path):
    lines = path.read_text().splitlines()
    if lines[0].startswith("=== Q"):
        lines = lines[1:]
    match = re.fullmatch(r"duckdb-result-v1 columns=(\d+) rows=(\d+)", lines[0])
    if not match:
        raise ValueError(f"Invalid result header: {path}")
    columns, count = map(int, match.groups())
    header = lines[: columns + 1]
    rows = []
    for line in lines[columns + 1 :]:
        if line.startswith("row="):
            rows.append([])
        else:
            value = line.split(" ", 1)[1]
            rows[-1].append(None if value == "null" else decode(value.removeprefix("text=")))
    if len(rows) != count or any(len(row) != columns for row in rows):
        raise ValueError(f"Truncated result: {path}")
    floating = {i for i, line in enumerate(header[1:]) if re.search(r'logical_type="(DOUBLE|FLOAT)"', line)}
    return header, sorted(rows, key=lambda row: tuple((v is not None, v or "") for v in row)), floating


def assert_results_equal(expected, actual):
    header, rows, floating = result(expected)
    other_header, other_rows, _ = result(actual)
    if header != other_header:
        raise ValueError(f"Schema/count mismatch: {expected} vs {actual}")
    for left, right in zip(rows, other_rows, strict=True):
        for column, (x, y) in enumerate(zip(left, right, strict=True)):
            if x == y:
                continue
            if column in floating and x is not None and y is not None:
                if math.isclose(float(x), float(y), rel_tol=1e-12, abs_tol=1e-10):
                    continue
            raise ValueError(f"Value mismatch column {column}: {expected} vs {actual}: {x!r} != {y!r}")


def arm_order(block):
    orders = [ARMS[i:] + ARMS[:i] for i in range(len(ARMS))]
    orders += [tuple(reversed(order)) for order in orders]
    return orders[block % len(orders)]


class Client:
    def __init__(self, args, root, arm, sql, parquet=False):
        self.root = root
        root.mkdir(parents=True)
        env = {key: value for key, value in os.environ.items() if not key.startswith("VORTEX_") and key != "RUST_LOG"}
        env.update(RUST_LOG="warn", VORTEX_SCAN_BACKEND="v1" if arm == "v1" else "push-frontier",
                   VORTEX_MORSEL_FRONTIER_PROJECTION_PREFETCH="1" if arm == "prefetch" else "0")
        if arm != "v1":
            env.update(VORTEX_DUCKDB_Q6_FRONTIER_BUNDLE="2", VORTEX_DUCKDB_FRONTIER_BUNDLE_MIN_WAVES="8")
        data = args.data / ("parquet" if parquet else "vortex-file-compressed")
        command = [str(args.binary), "--data", str(data), "--sql", str(sql), "--output", str(root),
                   "--threads", str(args.threads)] + (["--parquet"] if parquet else [])
        self.log = (root / "stderr.log").open("w")
        self.proc = subprocess.Popen(command, env=env, cwd=REPO, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        (root / "command.json").write_text(json.dumps({"command": command, "environment": {
            key: value for key, value in env.items() if key.startswith("VORTEX_") or key == "RUST_LOG"
        }}, indent=2))
        try:
            if not self.line().startswith("READY\t"):
                raise ValueError(f"Driver startup failed: {root}")
            if "value=1 null" not in (root / "databases.stdout").read_text():
                raise ValueError("Expected an in-memory catalog containing real-file views")
            if args.suite == "tpch" and result(root / "data-count.stdout")[1] != [["59986052"]]:
                raise ValueError("Expected real SF10 lineitem: 59,986,052 rows")
        except BaseException:
            self.close()
            raise

    def line(self):
        if not self.selector.select(timeout=600):
            raise TimeoutError(f"Driver timeout: {self.root}")
        line = self.proc.stdout.readline().rstrip("\n")
        if not line:
            raise RuntimeError(f"Driver exited unexpectedly; inspect {self.root / 'stderr.log'}")
        return line

    def execute(self, action, query, prefix):
        self.proc.stdin.write(f"{action}\t{query}\t{prefix}\n")
        self.proc.stdin.flush()
        response = self.line().split("\t")
        if action == "plan":
            if response != ["PLAN", str(query)]:
                raise ValueError(response)
            return None
        if len(response) != 4 or response[:2] != ["DONE", str(query)]:
            raise ValueError(response)
        output = Path(f"{prefix}.stdout")
        return {"q": query, "query_ns": int(response[2]), "rows": int(response[3]),
                "output": str(output), "sha256": sha256(output), "finished_ns": time.time_ns()}

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=20)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        finally:
            self.selector.close()
            self.log.close()
        if self.proc.returncode:
            raise RuntimeError(f"Driver failed with {self.proc.returncode}: {self.root}")


def summarize(output, queries, rounds, suite, threads):
    records = [json.loads(line) for line in (output / "measured.jsonl").read_text().splitlines()]
    if len(records) != rounds * len(queries) * len(ARMS):
        raise ValueError("Incomplete timing matrix")
    medians = {q: {arm: statistics.median(row["query_ns"] / 1e6 for row in records
                                       if row["q"] == q and row["arm"] == arm) for arm in ARMS} for q in queries}
    totals = {arm: sum(value[arm] for value in medians.values()) for arm in ARMS}
    rng = random.Random(0)
    intervals = {}
    for arm in ARMS[1:]:
        ratios = []
        for block in range(rounds):
            times = {a: sum(row["query_ns"] for row in records if row["block"] == block and row["arm"] == a) for a in ("v1", arm)}
            ratios.append(times[arm] / times["v1"])
        bootstrap = sorted(math.exp(statistics.mean(math.log(rng.choice(ratios)) for _ in ratios)) for _ in range(4000))
        intervals[arm] = {"paired_total_ratios": ratios, "ratio_ci95": [bootstrap[100], bootstrap[3899]] if rounds >= 2 else None}
    scope = "full comparison" if rounds >= 6 and len(queries) == (43 if suite == "clickbench" else 22) else "harness smoke only"
    summary = {"scope": scope, "rounds": rounds, "threads": threads, "queries": queries, "sum_medians_ms": totals, "medians_ms": medians,
               "paired_bootstrap": intervals, "note": "Exploratory paired-round intervals; no per-query multiple-comparison correction."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    body = f"# DuckDB {output.name}: V1 vs frontier\n\n{scope}: {rounds} rounds, {len(queries)} queries; real-file scans, same binary, {threads} threads.\n\n"
    body += "Positive percentages below mean less elapsed time. Totals are sums of per-query medians.\n\n"
    body += "| Query | V1 ms | W8 ms | Prefetch ms | W8 vs V1 | Prefetch vs V1 | Prefetch vs W8 |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for q, value in [*medians.items(), ("Total", totals)]:
        v, w, p = (value[a] for a in ARMS)
        body += f"| {q} | {v:.2f} | {w:.2f} | {p:.2f} | {100*(1-w/v):+.2f}% | {100*(1-p/v):+.2f}% | {100*(1-p/w):+.2f}% |\n"
    body += "\nPaired-round 95% bootstrap intervals for total-time ratios versus V1 (below 1 favors frontier):\n\n"
    for arm, value in intervals.items():
        if value["ratio_ci95"] is None:
            body += f"- {arm}: not estimated from a single smoke-test round.\n"
            continue
        low, high = value["ratio_ci95"]
        body += f"- {arm}: [{low:.4f}, {high:.4f}]\n"
    body += "\nCorrectness uses an independent Parquet reference. ClickBench tie-breaker SQL is validated separately; canonical SQL is timed unchanged. TPC-H Q22 disables common_subplan in every arm, including the reference, because the default-plan correctness bug remains unresolved. Profiles are separate diagnostic runs; operator timings are summed worker time, not recoverable wall time. No spills are allowed.\n"
    (output / "SUMMARY.md").write_text(body)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("plans", "raw", "profiles", "validation"):
        (output / name).mkdir()
    files = sorted(args.data.glob("*/*.vortex")) + sorted(args.data.glob("*/*.parquet"))
    expected_files = 200 if args.suite == "clickbench" else 16
    if len(files) != expected_files:
        raise ValueError(f"Expected {expected_files} real input files, got {len(files)}")
    canonical = REPO / "vortex-bench/sql" / ("clickbench_queries.sql" if args.suite == "clickbench" else "tpch")
    validation = canonical.with_name("clickbench_correctness_queries.sql") if args.suite == "clickbench" else canonical
    queries = list(range(43)) if args.suite == "clickbench" else list(range(1, 23))
    if args.queries:
        selected = [int(q) for q in args.queries.split(",")]
        if len(selected) != len(set(selected)) or not set(selected) <= set(queries):
            raise ValueError("Invalid or duplicate query IDs")
        queries = selected
    changed = set()
    if args.suite == "clickbench":
        timed_sql = [s.strip() for s in canonical.read_text().split(";") if s.strip()]
        checked_sql = [s.strip() for s in validation.read_text().split(";") if s.strip()]
        if len(timed_sql) != 43 or len(checked_sql) != 43:
            raise ValueError("Expected all 43 ClickBench statements")
        changed = {q for q in queries if timed_sql[q] != checked_sql[q]}
    tracked = files + [args.binary, Path(__file__).resolve(), REPO / "Cargo.lock"]
    tracked += [canonical, validation] if canonical.is_file() else sorted(canonical.glob("*.sql"))
    tracked += list((REPO / "target/release_debug").glob("**/libduckdb.*"))
    before = {str(p): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in tracked}
    (output / "manifest.json").write_text(json.dumps({"files": before, "suite": args.suite,
        "queries": queries, "rounds": args.rounds, "threads": args.threads,
        "validation_only_tie_breakers": sorted(changed), "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}, indent=2))
    refs = {}
    reference = Client(args, output / "reference", "v1", validation, parquet=True)
    try:
        for q in queries:
            refs[q] = reference.execute("run", q, output / "reference" / f"q{q:02}")
            if args.suite == "clickbench" and q == 0 and result(Path(refs[q]["output"]))[1] != [["99997497"]]:
                raise ValueError("Expected all 99,997,497 ClickBench rows")
    finally:
        reference.close()
    for arm in ARMS:
        client = Client(args, output / "validation" / arm, arm, validation)
        try:
            for q in queries:
                row = client.execute("run", q, output / "validation" / arm / f"q{q:02}")
                assert_results_equal(Path(refs[q]["output"]), Path(row["output"]))
                append(output / "checks.jsonl", {"arm": arm, "phase": "validation", **row})
        finally:
            client.close()
    print("Independent Parquet correctness passed", flush=True)
    clients = {}
    try:
        for arm in ARMS:
            clients[arm] = Client(args, output / "clients" / arm, arm, canonical)
        for q in queries:
            for arm, client in clients.items():
                client.execute("plan", q, output / "plans" / f"q{q:02}-{arm}")
            if len({sha256(output / "plans" / f"q{q:02}-{arm}.plan") for arm in ARMS}) != 1:
                raise ValueError(f"V1/frontier plan mismatch Q{q}")
        for block in range(args.rounds):
            for q in queries if block % 2 == 0 else reversed(queries):
                for arm in arm_order(block):
                    for phase in ("warm", "measured"):
                        row = clients[arm].execute("run", q, output / "raw" / f"b{block}-q{q:02}-{arm}-{phase}")
                        if row["rows"] != refs[q]["rows"]:
                            raise ValueError(f"Row count mismatch Q{q}")
                        if q not in changed:
                            assert_results_equal(Path(refs[q]["output"]), Path(row["output"]))
                        row.update(arm=arm, block=block, phase=phase, full_value_check=q not in changed)
                        append(output / "checks.jsonl", row)
                        if phase == "measured":
                            append(output / "measured.jsonl", row)
                print(f"round {block + 1}/{args.rounds} Q{q} complete", flush=True)
            if sha256(args.binary) != before[str(args.binary)]["sha256"]:
                raise ValueError("Benchmark binary changed")
        for q in queries:
            for arm, client in clients.items():
                row = client.execute("profile", q, output / "profiles" / f"q{q:02}-{arm}")
                if row["rows"] != refs[q]["rows"]:
                    raise ValueError(f"Profile row count mismatch Q{q}")
                if q not in changed:
                    assert_results_equal(Path(refs[q]["output"]), Path(row["output"]))
                profile = json.loads((output / "profiles" / f"q{q:02}-{arm}.profile.json").read_text())
                if profile["system_peak_temp_dir_size"]:
                    raise ValueError(f"Unexpected spill Q{q}")
    finally:
        for client in clients.values():
            client.close()
    if any(sha256(Path(path)) != info["sha256"] for path, info in before.items()):
        raise ValueError("Input/binary/SQL identity changed during campaign")
    summarize(output, queries, args.rounds, args.suite, args.threads)
    (output / "COMPLETE").write_text("All correctness, plan, cardinality, identity and spill gates passed.\n")
    print((output / "SUMMARY.md").read_text(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("tpch", "clickbench"), required=True)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing parquet/ and vortex-file-compressed/")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=14)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--queries", help="Optional subset for harness smoke tests; CI leaves this unset")
    args = parser.parse_args()
    if args.rounds < 1 or args.threads < 1:
        parser.error("threads and rounds must be positive")
    args.binary = args.binary.resolve()
    args.data = args.data.resolve()
    run(args)


if __name__ == "__main__":
    main()
