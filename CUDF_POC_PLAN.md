# cuDF NDS-H Vortex POC

**Goal:** [Benchmark-only upstream POC](https://github.com/NVIDIA/cudf/issues/23877#issuecomment-5457730105)
comparing Vortex with Parquet: **≥2× for both end-to-end projected reads and queries
across Q1/Q5/Q6/Q9/Q10 at SF1/SF10, warm and cold**. The goal is **not met**.

[Setup](benchmarks/cudf-ndsh/README.md) · [Validation](benchmarks/cudf-ndsh/VALIDATION.md) · [Progress](benchmarks/cudf-ndsh/PROGRESS.md)

## Current implementation

- **Benchmark integration:** default-OFF Q1/Q5/Q6/Q9/Q10 comparisons in
  `cpp/benchmarks/ndsh/`, with one include hook in `cpp/benchmarks/CMakeLists.txt`.
  Both formats use the original pinned cuDF generator, identical logical fixtures,
  scan projections, post-read predicates, and existing cuDF query operations.
  Native Parquet-pushdown benchmarks remain separate.
- **Vortex I/O:** CPU writing in 16,777,216-row CUDA-flat blocks; reads use pooled
  cacheable pinned-host staging → HtoD → GPU decode → Arrow Device imports → one
  final owning cuDF materialization. The CUDA memory pool retains up to 8 GiB.
- **Cache controls:** warm reads use the OS page cache. Cold callbacks sync and evict
  each input file, then require zero resident pages. Cold Vortex data uses `O_DIRECT`,
  metadata is buffered, and Parquet uses its native reader. Coldness is defined at
  the OS page-cache level.
- **Timing:** CPU wall time covers complete reads, selected query work, destruction,
  and device completion. Fixture writing, correctness checks, and eviction are untimed.

Exact projection/value checks and independent CPU references cover generated results,
with synthetic cases for nonempty results, boundaries, joins, nulls, and empty inputs.
The original generator can yield degenerate Q6/Q10 results and low-SF supplier joins;
match counts identify which runs provide meaningful full-query performance evidence.

The optional [generator-fixes.patch](benchmarks/cudf-ndsh/generator-fixes.patch) supplies
four generator fixes and a regression target. Selecting it requires regenerating both
formats' fixtures and labeling the dataset and measurements separately.

## Status and next steps

The rebased Vortex Release archive and all five query executables plus the adapter-test
executable built and linked against pinned Release cuDF. **Current-source runtime
validation and measurements are pending.** Recorded checks and historical measurements
are in [Validation](benchmarks/cudf-ndsh/VALIDATION.md).

Next: validate the adapter and queries, collect fresh SF1 then SF10 warm/cold read/query
baselines, and profile bottlenecks. Scale to SF100 after these matrices are stable,
with separate Vortex/RMM memory accounting. Publish the Vortex prerequisites and update
the immutable pin for the upstream POC.
