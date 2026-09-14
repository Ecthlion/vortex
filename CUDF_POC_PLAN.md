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

**Release SF1 passes all 56 states at `91142e2c18`** with CUDA 13.0.88 / GCC 14.3.0.
Vortex is faster in all 28 format pairs; 24 reach 2×. Q6/Q10 have empty results with
the original generator. Timings and caveats are in
[Validation](benchmarks/cudf-ndsh/VALIDATION.md).

1. Run current-source memcheck validation when requested.
2. Collect SF10 warm/cold read/query baselines. Label the generator choice and match
   counts; use nondegenerate results for full-query performance claims.
3. Profile bottlenecks and optimize. Scale to SF100 once these matrices are stable,
   accounting for Vortex and RMM memory separately.
4. Publish validated revisions and prerequisites, then prepare the upstream POC.
