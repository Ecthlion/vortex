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

**Chunked I/O at `98b3d12746` improves Q1/Q6 SF1 latency by 14–54%**; all eight
warm/cold read/query comparisons exceed 2× Parquet. The last complete five-query
baseline is `91142e2c18` (56 passing states, 24/28 comparisons ≥2×). Q6/Q10 have
empty results with the original generator. See
[Validation](benchmarks/cudf-ndsh/VALIDATION.md).

1. Finish Q5/Q9/Q10 SF1 comparisons with the improved I/O path and validate memcheck
   when requested.
2. Collect SF10 warm/cold read/query baselines. Label the generator choice and match
   counts; use nondegenerate results for full-query performance claims.
3. Profile bottlenecks and optimize. Scale to SF100 once these matrices are stable,
   accounting for Vortex and RMM memory separately.
4. Publish validated revisions and prerequisites, then prepare the upstream POC.
