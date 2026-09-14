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

Rebased onto `56c550cff25c663f86c3f6b5f2e96c2f8062098b`; build source checkpoint:
`03fd6013e2687a11e35033ec7307ac909ee88f9d`.

The fresh Vortex Release archive built successfully. An isolated consumer compiled
**75/75** objects, created **4/4** archives, and linked **6/6** executables against
pinned Release cuDF, with clean compiler/linker diagnostics and resolved runtime
dependencies. **Current-source runtime validation and measurements are pending.**

Ready executables under `build/cudf-ndsh-sf1-rebased/consumer/bin/`:

- `NDSH_Q01_NVBENCH`, `NDSH_Q05_NVBENCH`, `NDSH_Q06_NVBENCH`
- `NDSH_Q09_NVBENCH`, `NDSH_Q10_NVBENCH`, `NDSH_VORTEX_IO_TEST`

[VALIDATION.md](VALIDATION.md) records build provenance, earlier offline/compile
checks, and historical runtime evidence. Historical timings describe earlier
configurations rather than current baselines. The ≥2× goal for reads and queries
at SF1/SF10, warm/cold, remains open.

## Next steps

1. Run adapter and query correctness/memcheck validation using the fresh executables.
2. Regenerate paired fixtures and collect SF1, then SF10 warm/cold read/query baselines.
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
- The isolated consumer uses RMM `543cecf2cde1ba0fe4097920db8918079e8acea0`, matching
  the existing pinned Release `libcudf.so`. Shared dependencies in
  `build/cudf-ndsh-build` have advanced, and an interrupted build left mixed-era
  objects. Preserve the isolated consumer; avoid broad rebuilds or mixing headers
  and libraries from different dependency revisions.
- Fresh Vortex archive:
  `build/cudf-ndsh-build/_deps/vortex-build/ffi/vortex-artifacts/libvortex_ffi.a`.
  Build records and the matching FlatBuffers compiler are under
  `build/cudf-ndsh-sf1-rebased/`.
- Benchmark JSON, logs, binaries, Nsight reports, and SQLite exports are ignored.
  Old profiles contain sensitive environment metadata; follow the profiling safety guard.
