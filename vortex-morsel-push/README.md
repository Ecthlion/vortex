# vortex-morsel-push

A morsel-driven push executor for Vortex layouts. A worker activates sources for a row range;
sources produce batches and push them through compiled physical pipelines.

```text
activate source → read/decode → push batch → downstream stages → output
```

`build_plan` binds expressions, identifies sources, and fuses eligible operator chains. Within
a pipeline, each batch passes directly to the next stage on the same worker. Operators with
multiple inputs retain and align batches at pipeline boundaries. Bottom-level `Chunked<Flat…>`
layouts compile into one flat source over a shared sequence of segment descriptors, removing
that chunk boundary from the pipeline. Each segment retains its own read ticket and decode lease.
The source pushes segment overlaps in row order as downstream credits arrive; its full row range
does not change morsel or batch sizes.

`next_plan` registers the reads a morsel can need. Execution begins with `push_start` on its
sources. `push_input` and `push_end` deliver batches and input completion. A stage waiting for
I/O retains its state and resumes through `push_resume` when its exact tickets complete.
`push_credit` tells a producer that downstream capacity is available again.

Predicates produce authoritative selections that activate later predicates and projected
columns. Optional demand hints can defer speculative I/O; correctness depends on the selections,
not on whether hints arrive. Selections can arrive in fragments: each segment becomes eligible
once its overlapping rows are known, allowing an ordered prefix to run while later selections
are pending. All-false selections avoid unnecessary decode work.

The executor does not poll storage futures. `SegmentSourceDriver` answers the scan's `IoDemand`
stream on a separate runtime task or thread. `MorselScan::into_stream` provides ordered output,
bounded capacity, and cancellation; `run` collects the output. Leased shared cells retain decoded
chunks until the last overlapping morsel retires. Raw segment cells carry exact planned-use
counts; their final use drops ready bytes or cancels the outstanding source future. The
dedicated-thread, owned-I/O variant is shut down and joined before the scan leaves a query run.
Shared external drivers, including DuckDB's detached runtime task, instead terminate when demand
and service ownership close; they are not synchronously joined by the subscan.

The prototype supports flat, chunked, and non-nullable struct layouts, plus transparent zoned
and legacy-statistics wrappers. Unsupported layouts are build errors. Import provenance is in
[UPSTREAM.md](UPSTREAM.md); the [executor primer](../docs/developer-guide/internals/scan-execution-models/morsel-executor-primer.md)
explains the current contracts. The grouped cursor implementation process and bug checklist are in
[IO_FRONTIER_IMPLEMENTATION.md](IO_FRONTIER_IMPLEMENTATION.md); scheduling evidence and the
decoded-sharing follow-up are recorded in [IO_FRONTIERS.md](IO_FRONTIERS.md). The current branch
state and context-free continuation instructions are in
[IO_FRONTIER_HANDOVER.md](IO_FRONTIER_HANDOVER.md).

## Evaluation

Both evaluators run push pipelines and validate output against V1 before timing:

```bash
cargo run --release -p vortex-morsel-push --features _test-harness --bin morsel-push-eval
cargo run --release -p vortex-morsel-push --features _test-harness --bin tpch-push-eval -- 1
```

The default comparison keeps query semantics identical: both paths receive the same projection
and filter, scan the full row range with the default selection, preserve row order, use the same
session, layout, and segment source, and consume timed output immediately. The V1 rows use the
current `LayoutReader` `ScanBuilder` with its default split policy and per-worker concurrency. The
evaluator's default push rows use the grouped-I/O frontier scheduler;
`MorselConfig::frontier_defaults` changes only the path selector from `None` to `Some(0)`, adding no
extra range-frontier lookahead. Explicit policy matrices retain a clearly labelled `push current`
row when comparing the two push schedulers.

SQL integrations select this executor with `VORTEX_SCAN_BACKEND=push`. That label retains the
established eager-lookahead policy. `VORTEX_SCAN_BACKEND=push-frontier` selects the grouped-I/O
frontier scheduler with the current production-integration policy: zero additional down-frontier
lookahead, up to two predicate-only right groups for the current range, and row-frontier refills of
32 ranges. Projection remains demand-gated. DuckDB's adjacent-morsel bundle-size-two/W8 experiment
is a separate opt-in. It computes eligibility from selected pre-static-prune morsels after
row-range/Selection and applicable executor-limit truncation, then pairs contiguous selected
morsels before static pruning. The pair remains bundled after pruning; the second morsel receives
one predicate-only down frontier only when both members retain exactly one range and remain exactly
adjacent, otherwise down lookahead becomes zero. It is default-off because its final SF10 resource
gate failed. Its current `MorselScan` limit guard is not a SQL LIMIT guarantee because DuckDB does
not yet propagate that limit; see
[IO_FRONTIER_HANDOVER.md](IO_FRONTIER_HANDOVER.md).
