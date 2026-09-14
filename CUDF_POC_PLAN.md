# cuDF NDS-H Vortex POC

**Goal:** [Benchmark-only upstream POC](https://github.com/NVIDIA/cudf/issues/23877#issuecomment-5457730105)
comparing Vortex with Parquet: **≥2× for both end-to-end projected reads and queries
across Q1/Q5/Q6/Q9/Q10 at SF1/SF10, warm and cold**. The goal is **not met**.

[Setup](benchmarks/cudf-ndsh/README.md) · [Validation](benchmarks/cudf-ndsh/VALIDATION.md) · [Progress](benchmarks/cudf-ndsh/PROGRESS.md)

## Scope

Default-OFF local-file comparisons use matched fixtures and projections, Vortex GPU
decompression, and existing cuDF query operations. Independent CPU references and
synthetic cases check correctness outside timing. See the README for the
[I/O and timing contract](benchmarks/cudf-ndsh/README.md#io-and-timing-contract) and
[dataset choices](benchmarks/cudf-ndsh/README.md#dataset-and-correctness).

## Next steps

**The clean Release recipe and current-source runtime measurements remain unvalidated.**
Earlier isolated-build and runtime evidence is recorded in
[Validation](benchmarks/cudf-ndsh/VALIDATION.md).

1. Build the tracked recipe, then validate adapter and query correctness.
2. Collect fresh SF1, then SF10 warm/cold read/query baselines. Label the generator
   choice and match counts; use nondegenerate results for full-query performance claims.
3. Profile bottlenecks and optimize. Scale to SF100 once these matrices are stable,
   accounting for Vortex and RMM memory separately.
4. Publish validated revisions and prerequisites, then prepare the upstream POC.
