# Resume: cuDF NDS-H Vortex POC

Checkpoint: 2026-09-14 · branch `ad/cudf-ndsh-build-support`.
[Plan](../../CUDF_POC_PLAN.md) · [Setup](README.md) · [Validation](VALIDATION.md)

## Implemented

- Default-OFF Q1/Q5/Q6/Q9/Q10 projected-read and full-query comparisons. The main
  patch lives in `cpp/benchmarks/ndsh/` plus one parent CMake include hook.
- Both formats use the original pinned cuDF generator, identical logical fixtures,
  scan projections, post-read predicates, and existing cuDF query operations.
  Native Parquet-pushdown benchmarks remain separate.
- Vortex writes 16,777,216-row CUDA-flat blocks. Reads use cacheable pinned staging,
  GPU decompression, Arrow Device imports, and a final owning cuDF materialization;
  CUDA pool retention is 8 GiB.
- Warm/cold controls verify OS-page-cache eviction before each timed cold callback;
  cold Vortex data reads use `O_DIRECT`. Complete reads/queries, destruction, and
  device completion are timed; setup, checks, and eviction are untimed.
- Exact projection/value checks and independent CPU references, plus synthetic
  nonempty, boundary, join, null, and empty-result cases. Q6 covers NULL zero-match
  sums; Q9 covers duplicate `partsupp` join multiplicity.
- Optional `generator-fixes.patch` supplies four independent fixes and the
  `NDSH_DATA_GENERATOR_TEST` regression target. Select and label this dataset
  separately, regenerating both formats' fixtures.

## Build and evidence

[`reproduce.py`](reproduce.py) and the source dependency lock capture the Release
build/run workflow; commands are in [README.md](README.md#build-from-a-clean-checkout).
cuDF selects the toolchain from the caller's environment/CMake definitions; Vortex
forwards that selection into Cargo. Compiler versions and flags are recorded with
the build. **The clean recipe and current-source runtime measurements are not yet
validated.** See [Validation](VALIDATION.md#current-build-status) for compiler limitations.

Earlier source checkpoint `03fd6013e2687a11e35033ec7307ac909ee88f9d` built a Vortex
Release archive and an isolated consumer: **75/75** objects, **4/4** archives and
**6/6** linked executables against an existing pinned Release cuDF library. That is
historical isolated-build evidence, not validation of the clean recipe.

[VALIDATION.md](VALIDATION.md) records build provenance, earlier offline/compile
checks, and historical runtime evidence. Historical timings describe earlier
configurations rather than current baselines. The ≥2× goal for reads and queries
at SF1/SF10, warm/cold, remains open.

## Next steps

1. Build the tracked recipe from a clean committed checkout, then run adapter and
   query correctness/memcheck validation.
2. Collect fresh SF1, then SF10 warm/cold read/query baselines.
   Run GPU benchmarks sequentially and report generator choice and match counts.
   The original generator can yield degenerate Q6/Q10 results and low-SF supplier
   joins; use nondegenerate results for full-query performance claims.
3. Profile full reads and queries using the
   [profiling safety guard](VALIDATION.md#profile-evidence-and-safety), then optimize
   the measured bottlenecks. Scale to SF100 once matrices are stable, with separate
   Vortex/RMM memory accounting.
4. Publish Vortex prerequisites, update the immutable pin, and prepare the upstream POC.

## Worktree cautions

- Main deliverable: `benchmarks/cudf-ndsh/upstream.patch`, against cuDF
  `5339497a1a17d799687cbf189fb113411fb015ca`. Editable source is ignored
  `build/cudf-ndsh-src`. Apply to a fresh pinned checkout elsewhere; never modify
  `/home/ubuntu/cudf`.
- Export source changes with
  `git --no-pager -C build/cudf-ndsh-src diff -- cpp/benchmarks > benchmarks/cudf-ndsh/upstream.patch`.
  New files need `git add -N` in that checkout before export.
- Use `reproduce.py` with its own work directory rather than rebuilding
  `build/cudf-ndsh-build`, whose dependencies have advanced and whose objects are
  mixed-era. Historical isolated-build artifacts remain under
  `build/cudf-ndsh-sf1-rebased/`; they are not inputs to the new recipe.
- Benchmark JSON, logs, binaries, Nsight reports, and SQLite exports are ignored.
  Old profiles contain sensitive environment metadata; follow the profiling safety guard.
