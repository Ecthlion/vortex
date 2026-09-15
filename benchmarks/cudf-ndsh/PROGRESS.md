# Resume: cuDF NDS-H Vortex POC

Checkpoint: 2026-09-15 · branch `ad/cudf-ndsh-build-support`.
[Plan](../../CUDF_POC_PLAN.md) · [Setup](README.md) · [Validation](VALIDATION.md)

## Current state

Q1/Q5/Q6/Q9/Q10 have default-OFF matched Parquet/Vortex read/query comparisons,
GPU decompression, and independent correctness checks. The README defines the
[I/O and timing contract](README.md#io-and-timing-contract),
[dataset choices](README.md#dataset-and-correctness), and
[tracked Release build/run recipe](README.md#build-from-a-clean-checkout).

**Optimized SF1 is complete at clean `b414dd8307`** (docs-only changes beyond
chunked-I/O source `98b3d12746`). All five queries, including all three Q9 engines,
pass all **56 warm/cold read/query states**; all **28 labeled format pairs are ≥2×**
(2.24–5.06×). The seven-target incremental build succeeded in 12.7 s; smoke and
15 adapter tests pass again, with no skips. The 33 focused CUDA tests from
`98b3d12746` remain relevant but were not rerun today.

Toolchain checked: Release, NVCC 13.0.88 / GCC 14.3.0 on GH200, driver 595.71.05.
The original generator still produces empty Q6/Q10 results, so their full-query
timings do not establish nonempty-query performance. The full SF1/SF10 goal remains
open. [Validation](VALIDATION.md#current-release-sf1-run) records current
timings, sampling caveats, and provenance; earlier measurements are historical.

## Next steps

1. Collect post-change profiles of the improved read path; the full SF1 matrix is done.
2. Run current-source memcheck validation when requested, then collect SF10 baselines.
   Run GPU benchmarks sequentially and report generator choice and match counts.
   Use nondegenerate results for full-query performance claims; collect generator-fixed
   comparisons separately if selected.
3. Profile full reads and queries using the
   [profiling safety guard](VALIDATION.md#profile-evidence-and-safety), then optimize
   the measured bottlenecks. Scale to SF100 once matrices are stable, with separate
   Vortex/RMM memory accounting.
4. Publish validated revisions and prerequisites, then prepare the upstream POC.

## Worktree cautions

- Edit the harness directly in `benchmarks/cudf-ndsh/src/vortex_ndsh/`, `tests/`,
  and `vortex.cmake`. These sources and the Vortex library use the same checkout.
- `upstream.patch` contains only the cuDF-side integration, against
  `5339497a1a17d799687cbf189fb113411fb015ca`. Its editable checkout is ignored
  `build/cudf-ndsh-src`; never modify `/home/ubuntu/cudf`.
- After editing the cuDF integration, export with
  `git --no-pager -C build/cudf-ndsh-src diff -- cpp/benchmarks > benchmarks/cudf-ndsh/upstream.patch`.
- Use `reproduce.py` with its own work directory rather than rebuilding
  `build/cudf-ndsh-build`, whose dependencies have advanced and whose objects are
  mixed-era. Historical isolated-build artifacts remain under
  `build/cudf-ndsh-sf1-rebased/`; they are not inputs to the new recipe.
- Current SF1 results: `build/cudf-ndsh-repro-cuda130-release-v2/results/20260915T054834.964684Z/`.
  After the successful incremental build, `build.json` was refreshed at clean
  `b414dd8307` with actual identity and binary hashes; `run` requires the recorded
  clean revision. `recipe.json` still describes the initial configure. Use a fresh
  work directory for `reproduce.py build` after source/configuration changes.
- Historical optimization results and preserved baseline binaries:
  `build/cudf-ndsh-perf-20260914/`; earlier full-matrix provenance is linked in
  [Validation](VALIDATION.md#historical-sf1-measurements).
- All required sources are tracked. Benchmark JSON, logs, binaries, Nsight reports,
  and SQLite exports are ignored.
  Old profiles contain sensitive environment metadata; follow the profiling safety guard.
