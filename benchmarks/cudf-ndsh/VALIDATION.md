# Recorded benchmark results

**Historical evidence, not validation of the current branch.** The latest recorded
full-matrix run used clean Vortex `b414dd8307ee25d595382c9d1e8bf6fb15520624`
(docs-only changes beyond chunked-I/O source `98b3d12746`) and cuDF
`5339497a1a17d799687cbf189fb113411fb015ca`. The subsequent simplification and
one-flag CMake changes have not been validated by these runs.

For current usage, scope, and safety rules, see the [README](README.md).

## Recorded build and toolchain

The 2026-09-15 checkpoint was on branch `ad/cudf-ndsh-build-support`. The run used
NVIDIA GH200, driver **595.71.05**, NVCC **13.0.88**, GCC **14.3.0**, and Release
(`-O3 -DNDEBUG`). At `b414dd8307`, all seven targets built incrementally in **12.7 s**;
the original pinned Release `libcudf.so` hash was unchanged. Prepared cuDF/dependency
revisions and the selected toolchain were checked.

Afterward, `build.json` recorded the actual source identity, toolchain, and binary
hashes, with `build_method=incremental` and `incremental_from` retaining the prior
record. There was no fresh configure: `recipe.json` still described the initial one.
Provenance updates cannot substitute for rebuilding; use a fresh work directory
after source/configuration changes.

Included fixes were `1ab26426d8` (NVCC realpath before Cargo tool lookup),
`a9d705465e` (explicit NVTX/KvikIO links), and `91142e2c18` (`cudaSetDevice` before
the smoke test's stream check).

NVCC **13.0.88** worked with the full SDK described in the README. The recipe's
12.8 minimum is not a compatibility guarantee: tested NVCC **12.8.93** had a
parameter-pack emission bug, and **13.1.115** a private
`cudf::ast::literal::ast_scalar` access bug.
Both failures reproduced with host GCC 13 and 14. Earlier isolated 12.8 access probes
did not validate a full build. Probe records are under ignored
`build/cudf-ndsh-repro-cuda128-release-v2/compiler-compat-probes-20260914/`.

## Recorded Release SF1 run

The complete Q1/Q5/Q6/Q9/Q10 matrix covered Parquet/Vortex × read/query × warm/cold,
including Q9's three amount engines. Recorded command:

```sh
python3 -B benchmarks/cudf-ndsh/reproduce.py run \
  --work-dir=/home/ubuntu/vortex/build/cudf-ndsh-repro-cuda130-release-v2 \
  --scale-factor=1 --timeout=1200 --min-samples=100 --sample-timeout=5
```

Artifacts relative to ignored `build/cudf-ndsh-repro-cuda130-release-v2/`:

- Incremental build: `logs/20260915T054809.795368Z/`.
- Successful run: `logs/20260915T054834.964399Z/`.
- Results: `results/20260915T054834.964684Z/sf1-q{1,5,6,9,10}.json`, with
  refreshed `build.json` and `run.json` in the same directory.

| Check                 | Recorded result                       |
| --------------------- | ------------------------------------- |
| Incremental build     | 7 targets passed in 12.7 s            |
| Smoke / adapter tests | Passed / 15 passed; no skips          |
| SF1 matrix            | 56 unique passing states; no skips    |
| Sampling              | 100–1,040 samples/state; 20,893 total |

The 33 focused CUDA tests had passed for `98b3d12746`; they were not rerun at this
checkpoint. SF10, memcheck, and post-optimization profiles were still pending.

### Dataset and match counts

This run used the **original pinned generator**, not `generator-fixes.patch`.
Exact projected values and independent CPU references passed for both formats:

| Query            | Matched rows | Result                 |
| ---------------- | -----------: | ---------------------- |
| Q1               |    4,497,687 | 4 groups               |
| Q5               |        5,104 | 5 countries            |
| Q6               |            0 | SUM/revenue `NULL`     |
| Q9 (all engines) |       64,353 | 175 nation/year groups |
| Q10              |            0 | 0 customers            |

Q6/Q10 are degenerate: these timings establish execution/read behavior and
empty-result correctness, **not meaningful nonempty full-query performance**.
The full SF1/SF10 goal is not established despite meeting the SF1 timing threshold.

### CPU-wall timings

Means in **ms**, from JSON `nv/cold/time/cpu/mean` × 1,000; query includes reads.
Cache labels refer to the harness axis, not NVBench's sampling label. The
[README timing contract](README.md#io-and-timing-contract) defines cold as verified
OS-page-cache eviction, with `O_DIRECT` Vortex data reads versus native Parquet.

| Query / engine | Cache | Parquet read | Vortex read | Parquet query | Vortex query |
| -------------- | ----- | -----------: | ----------: | ------------: | -----------: |
| Q1             | warm  |       14.032 |       3.904 |        18.835 |        8.422 |
| Q1             | cold  |       18.872 |       4.420 |        23.812 |        8.955 |
| Q5             | warm  |       17.613 |       5.998 |        20.494 |        8.898 |
| Q5             | cold  |       22.494 |       7.323 |        25.421 |       10.145 |
| Q6             | warm  |        7.890 |       1.772 |         8.395 |        2.279 |
| Q6             | cold  |       11.096 |       2.191 |        11.648 |        2.733 |
| Q9 binaryop    | warm  |       26.837 |       6.027 |        30.012 |        9.122 |
| Q9 binaryop    | cold  |       32.766 |       7.726 |        36.170 |       10.724 |
| Q9 AST         | warm  |       26.896 |       6.047 |        29.955 |        8.932 |
| Q9 AST         | cold  |       32.864 |       7.700 |        35.998 |       10.665 |
| Q9 transform   | warm  |       26.915 |       6.034 |        29.954 |        9.012 |
| Q9 transform   | cold  |       32.867 |       7.690 |        36.080 |       10.723 |
| Q10            | warm  |       26.084 |       8.161 |        27.399 |        9.284 |
| Q10            | cold  |       30.815 |       9.122 |        32.265 |       10.643 |

Q9 reads repeat under three engine labels, not three distinct read implementations.
All 28 labeled format pairs reached ≥2× (**2.24–5.06×**, using unrounded means).
CPU relative standard deviation was median **1.25%**, maximum **3.01%**, with no
states above 5%. There were **35 sampling-limit warnings** (5-second limit,
0.5% GPU-noise target); all affected states reported PASS. These were neither
runner failures nor 1,200-second command timeouts.

## Profile evidence

Before chunking, Nsight Q6 warm read spent 3.698 ms in one 47.98 MB `pread`, versus
0.093 ms executing decode kernels. The focused `98b3d12746` experiment reduced
Q1/Q6 Vortex latency by **14–54%**: 16 states passed, all eight comparisons exceeded
2×, and 15 adapter / 33 focused CUDA tests passed. Q6 remained empty. With
`--min-samples 100 --timeout 5`, Vortex CPU relative SD was 1.5–6.2% afterward,
versus 8.0–13.8% for before-run cold states. Instrumented observations are not
benchmark timings. Follow the [profiling safety rules](README.md#profiling-safety);
old profiles contain sensitive environment metadata.

Ignored `build/cudf-ndsh-perf-20260914/` holds paired
`{baseline,chunked}-sf1-q{1,6}.json`, `baseline-build.json`,
`chunked-provenance.json`, and `focused-tests-summary.json`.

## Archived measurements

Earlier SF1/SF10 tables, compiler/build journals, sanitizer results, detailed
profile observations, and artifact paths are retained in the committed record:

```sh
git --no-pager show 90723345ee:benchmarks/cudf-ndsh/VALIDATION.md
```

These are not current baselines: they used older source/dataset configurations.
`build/cudf-ndsh-build` has mixed-era objects; `build/cudf-ndsh-sf1-rebased` reused a
cuDF library rather than a fully pinned clean build. Neither is an input to the
current recipe. New measurements need a fresh build, paired fixtures, generator
labels, and match counts.

## Remaining validation

- Validate the current source, including targeted tests and memcheck, when requested.
- Collect nondegenerate SF1/SF10 comparisons with generator labels and match counts.
- Collect current read/query profiles and separate Vortex/RMM memory accounting,
  then optimize measured bottlenecks; extend to SF100 only after the smaller
  matrices are stable.
- Publish validated revisions and prerequisites, then prepare the upstream POC.
