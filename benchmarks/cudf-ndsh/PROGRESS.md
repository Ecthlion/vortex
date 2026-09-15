# Resume: cuDF NDS-H Vortex POC

Checkpoint: 2026-09-15 · branch `ad/cudf-ndsh-build-support`.
[Plan](../../CUDF_POC_PLAN.md) · [Setup](README.md) · [Validation](VALIDATION.md)

## Current state

Q1/Q5/Q6/Q9/Q10 have default-OFF matched Parquet/Vortex read/query comparisons,
GPU decompression, and independent correctness checks. The README defines the
[I/O and timing contract](README.md#io-and-timing-contract),
[dataset choices](README.md#dataset-and-correctness), and
[tracked Release build/run recipe](README.md#build-from-a-clean-checkout).

**Chunked-I/O optimization `98b3d12746`: Q1/Q6 pass all 16 SF1 states**, plus 15
adapter tests and 33 focused CUDA tests. Splitting large reads into 4 MiB chunks
reduces Vortex latency by 14–54%; all eight Q1/Q6 comparisons now exceed 2× Parquet.
Toolchain: Release, CUDA 13.0.88 / GCC 14.3.0 on GH200.

The last complete five-query baseline is `91142e2c18`: all 56 states pass, with
24/28 comparisons reaching 2×. Q5/Q9/Q10 await measurements with the new I/O path.
The original generator produces empty Q6/Q10 results; those full-query timings
are not representative of nonempty queries. [Validation](VALIDATION.md) records
results, profiles, and provenance.

## Next steps

1. Finish Q5/Q9/Q10 SF1 comparisons and profile the improved read path. The last
   remaining-query build was interrupted; inspect its process/log state before resuming.
2. Run current-source memcheck validation when requested, then collect SF10 baselines.
   Run GPU benchmarks sequentially and report generator choice and match counts. Use nondegenerate results for full-query
   performance claims; collect generator-fixed comparisons separately if selected.
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
- Complete baseline results: `build/cudf-ndsh-repro-cuda130-release-v2/results/20260914T173000.589488Z/`.
  The build directory has since been rebuilt incrementally; its old `build.json` no longer
  describes all binaries. Use a fresh work directory for `reproduce.py build`.
- Optimization results and preserved baseline binaries: `build/cudf-ndsh-perf-20260914/`.
- Benchmark JSON, logs, binaries, Nsight reports, and SQLite exports are ignored.
  Old profiles contain sensitive environment metadata; follow the profiling safety guard.
