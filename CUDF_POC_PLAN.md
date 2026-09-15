# cuDF NDS-H Vortex POC

**Goal:** [Benchmark-only upstream POC](https://github.com/NVIDIA/cudf/issues/23877#issuecomment-5457730105)
comparing Vortex with Parquet: **≥2× for both end-to-end projected reads and queries
across Q1/Q5/Q6/Q9/Q10 at SF1/SF10, warm and cold**. The full goal is **not yet
established**: optimized SF1 meets the timing threshold, but Q6/Q10 results are empty
and current-source SF10 remains pending.

[Setup](benchmarks/cudf-ndsh/README.md) · [Validation](benchmarks/cudf-ndsh/VALIDATION.md) · [Progress](benchmarks/cudf-ndsh/PROGRESS.md)

## Scope

Default-OFF local-file comparisons use matched fixtures and projections, Vortex GPU
decompression, and existing cuDF query operations. Independent CPU references and
synthetic cases check correctness outside timing. See the README for the
[I/O and timing contract](benchmarks/cudf-ndsh/README.md#io-and-timing-contract) and
[dataset choices](benchmarks/cudf-ndsh/README.md#dataset-and-correctness).

## Next steps

**Optimized SF1 is complete at `b414dd8307`** (chunked-I/O source `98b3d12746`):
all 56 warm/cold read/query states pass, with all 28 labeled Parquet/Vortex pairs
≥2× (**2.24–5.06×**). Smoke and 15 adapter tests pass again, with no skips.
See [Validation](benchmarks/cudf-ndsh/VALIDATION.md#current-release-sf1-run) for the
current timings, sampling caveats, and provenance.

1. Run current-source memcheck validation when requested.
2. Collect SF10 warm/cold read/query baselines. Label the generator choice and match
   counts; use nondegenerate results for full-query performance claims.
3. Collect post-change profiles, then optimize measured bottlenecks. Scale to SF100
   once these matrices are stable, accounting for Vortex and RMM memory separately.
4. Publish validated revisions and prerequisites, then prepare the upstream POC.
