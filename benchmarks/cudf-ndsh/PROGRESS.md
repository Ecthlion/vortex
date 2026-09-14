# Resume: cuDF NDS-H Vortex POC

Checkpoint: 2026-09-14 · branch `ad/cudf-ndsh-build-support`.
[Plan](../../CUDF_POC_PLAN.md) · [Setup](README.md) · [Validation](VALIDATION.md)

## Current state

Q1/Q5/Q6/Q9/Q10 have default-OFF matched Parquet/Vortex read/query comparisons,
GPU decompression, and independent correctness checks. The README defines the
[I/O and timing contract](README.md#io-and-timing-contract),
[dataset choices](README.md#dataset-and-correctness), and
[tracked Release build/run recipe](README.md#build-from-a-clean-checkout).

**The clean recipe and current-source runtime measurements are not yet validated.**
[Validation](VALIDATION.md) records compiler limitations and historical build/runtime
evidence, not current baselines. The ≥2× goal for reads and queries at SF1/SF10,
warm/cold, remains open.

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
- Benchmark JSON, logs, binaries, Nsight reports, and SQLite exports are ignored.
  Old profiles contain sensitive environment metadata; follow the profiling safety guard.
