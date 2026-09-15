# Validation

Checkpoint: 2026-09-15. Full-matrix build/run at clean
`b414dd8307ee25d595382c9d1e8bf6fb15520624` (docs-only changes beyond chunked-I/O
source `98b3d12746`). Pinned cuDF:
`5339497a1a17d799687cbf189fb113411fb015ca`, Release (`-O3 -DNDEBUG`).
Hardware/toolchain: NVIDIA GH200, driver **595.71.05**, NVCC **13.0.88**, GCC **14.3.0**.
[Current state](PROGRESS.md) · [Setup and commands](README.md)

## Current build status

At clean `b414dd8307`, **all seven targets built incrementally in 12.7 s**. The
original pinned Release `libcudf.so` hash is unchanged. Prepared cuDF/dependency
revisions were checked, and the selected toolchain matches NVCC 13.0.88 / GCC 14.3.0
on GH200 with driver 595.71.05.

After that successful build, `build.json` was refreshed with the actual source
identity, toolchain, and binary hashes: `build_method=incremental`, with
`incremental_from` retaining the prior record. `recipe.json` still describes the
initial configure, not a fresh configure at this checkpoint. The tracked `run`
requires the recorded clean source revision and matching binary/library hashes.
Use a fresh work directory for the [build recipe](README.md#build-from-a-clean-checkout)
when changing source, compiler settings, or CMake definitions; provenance updates
alone are not a substitute for a build.

Included build/runtime fixes: `1ab26426d8` resolves the NVCC realpath before Cargo
tool lookup; `a9d705465e` declares explicit NVTX/KvikIO links; `91142e2c18` initializes
the smoke test with `cudaSetDevice` before its stream check.

NVCC **13.0.88** is validated with the full SDK described in the README. The recipe's
12.8 minimum does not imply compiler compatibility: tested NVCC **12.8.93** has a
parameter-pack emission bug, and **13.1.115** a private
`cudf::ast::literal::ast_scalar` access bug; both were reproduced with host GCC 13
and 14. Earlier 12.8 success on isolated access probes did not validate a complete
build. Compatibility probes are under ignored
`build/cudf-ndsh-repro-cuda128-release-v2/compiler-compat-probes-20260914/`.

## Current Release SF1 run

The **complete optimized five-query matrix** covers Q1/Q5/Q6/Q9/Q10,
Parquet/Vortex × read/full-query × warm/cold, including all three Q9 amount engines.
Successful command from the Vortex root:

```sh
python3 -B benchmarks/cudf-ndsh/reproduce.py run \
  --work-dir=/home/ubuntu/vortex/build/cudf-ndsh-repro-cuda130-release-v2 \
  --scale-factor=1 --timeout=1200 --min-samples=100 --sample-timeout=5
```

Artifacts below are relative to ignored `build/cudf-ndsh-repro-cuda130-release-v2/`:

- Incremental build: `logs/20260915T054809.795368Z/`.
- Successful run: `logs/20260915T054834.964399Z/`.
- Results: `results/20260915T054834.964684Z/sf1-q{1,5,6,9,10}.json`, with
  refreshed `build.json` and `run.json` in the same directory.

Recorded checks from this run (not rerun for this documentation update):

| Check                                              | Result                                    |
| -------------------------------------------------- | ----------------------------------------- |
| Incremental build                                  | All 7 targets passed in 12.7 s            |
| Build smoke / adapter tests                        | Passed again / 15 passed; no skips        |
| SF1 read/query × format × cache × Q9 engine matrix | 56 unique, valid passing states; no skips |
| Sampling                                           | 100–1,040 samples/state; 20,893 total     |

The **33 focused CUDA tests passed for `98b3d12746`** and remain relevant to this
unchanged source; they were **not rerun today**. Current-source memcheck, SF10, and
post-change profiles remain pending. Sanitizer results below are historical only.

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

Match counts are unchanged. Q6/Q10 are degenerate: their timings establish
execution/read behavior and empty-result correctness, **not meaningful nonempty
full-query performance**. Despite the SF1 timing threshold being met, the full
SF1/SF10 goal is **not yet established**.

### CPU-wall timings

CPU wall means in **ms**, taken from JSON `nv/cold/time/cpu/mean` × 1,000; query
includes reads. Cache labels refer to the harness axis, not NVBench's sampling
label. See the [timing contract](#timing-contract): cold means verified **OS page-cache
eviction, not all caches**, with `O_DIRECT` Vortex data reads versus native Parquet.

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

Q9 read is repeated under three engine labels; these are not three distinct read
implementations. **All 28 labeled Parquet/Vortex pairs reach ≥2×**, with speedups
ranging from **2.24× to 5.06×**. Ratios use unrounded CPU-wall means.

CPU relative standard deviation has median **1.25%**, maximum **3.01%**;
**no states exceed 5%**. There were **35 NVBench sampling-limit warnings** at the
5-second limit with a 0.5% GPU-noise target. All warned states still report **PASS**;
these are sampling warnings, not runner failures or the 1,200-second command timeout.

## Historical SF1 measurements

The pre-optimization full matrix at `91142e2c18` passed all 56 states, with 24/28
labeled pairs ≥2× (1.74–3.55×). It used `--min-samples=3 --sample-timeout=30`:
20–1,376 samples/state, 31,226 total; median CPU relative SD 1.30%, eight states >5%,
and two sampling-limit warnings. Offline harness / CUDA CMake checks passed 38 / 8.
Results and build/run records remain under
`build/cudf-ndsh-repro-cuda130-release-v2/results/20260914T173000.589488Z/`.
The initial build and completion logs are `logs/20260914T171459.140647Z/` and
`logs/20260914T172951.677938Z/` in that work directory; the prior build record's
`resumed_from` identifies recovery after an interrupted runner. These are historical,
not the current binary identity or timing baseline.

The focused experiment for `98b3d12746` reduced Q1/Q6 Vortex latency by **14–54%**:
all 16 states passed, with all eight comparisons >2×. Direct NVBench runs used
`--min-samples 100 --timeout 5`; Vortex after-run CPU relative SD was 1.5–6.2%,
versus 8.0–13.8% for before-run cold states. Q6 remained empty. Targeted Release
consumers built, and **15 adapter tests plus 33 focused CUDA tests passed**, with no
ignored tests. Rust filters were `pinned::tests` and `pooled_read_at::file::tests`,
using `cargo test --locked --offline -p vortex-cuda --lib --features _test-harness`,
the CMake-selected Release toolchain/target, and `--test-threads=1`.

The pre-change Nsight Q6 warm read spent 3.698 ms in one 47.98 MB `pread`, versus
0.093 ms executing decode kernels. These are instrumented observations, not benchmark
timings. The optimization splits large reads into 4 MiB chunks, at most 32 concurrent
host reads per file, transferring completed chunks into one GPU allocation.
Fixtures, encodings, projections, cuDF operations, and the timing/cache contract
are unchanged. No post-change profile has been collected.

Focused artifacts under ignored `build/cudf-ndsh-perf-20260914/`:

- `baseline-sf1-q{1,6}.json` and `chunked-sf1-q{1,6}.json`.
- `baseline-build.json`, `baseline-binaries/`, and `chunked-provenance.json`
  (source/binary hashes; benchmarks ran before the source commit).
- Baseline profiles: `baseline-q6-{warm,cold}.sqlite`, `baseline-q1-warm.sqlite`;
  range-scoped analysis: `derived-final-read-analysis.json`.
- Logs: `logs/20260914T203030.285386Z/` (before) and
  `logs/20260914T204642.079773Z/` (after and adapter checks).
- `focused-tests-summary.json` and `focused-tests-sanitized-{build,pinned,pooled}.json`
  record exact test commands and results.

## Historical isolated build

The rebased Vortex Release archive and all six consumer executables built and linked
against an existing pinned Release cuDF library. This did not perform a clean,
fully pinned cuDF build. Historical build records do not establish which
RAPIDS-CMake revision they used; they are separate from the current pinned recipe.

| Check                                              | Result                           |
| -------------------------------------------------- | -------------------------------- |
| CUDA-enabled Vortex Release archive                | Built in 265 s                   |
| Isolated consumer compilation                      | 75/75 passed, no warnings/errors |
| Consumer static archives                           | 4/4 created                      |
| Five query executables and adapter-test executable | 6/6 linked, clean diagnostics    |
| ELF dependency inspection in a clean environment   | All six resolved                 |

Source checkpoint: `03fd6013e2687a11e35033ec7307ac909ee88f9d`. The isolated consumer
uses RMM `543cecf2cde1ba0fe4097920db8918079e8acea0`, matching the existing pinned
Release `libcudf.so`. The Vortex build used FlatBuffers 25.12.19. Archive SHA256:
`71cc02d96317acb32ca4d4730566e9e12a9f3b5b93cd7b47766cfe8db5371b2f`.

Records under ignored `build/cudf-ndsh-sf1-rebased/`:

- `vortex-build-command.json`, `vortex-build-result.json`, `vortex-build.log`.
- `consumer/commands.json`, `consumer/results.json`, `consumer/provenance.json`.
- `consumer/link-results.json`, `consumer/link-provenance.json`,
  `consumer/link-summary.json`, `consumer/runtime-dependencies.json`.
- Ready executables: `consumer/bin/NDSH_Q{01,05,06,09,10}_NVBENCH` and
  `consumer/bin/NDSH_VORTEX_IO_TEST`.

## Historical source checks

These checks preceded the rebase and do not validate the current source:

| Check                                                           | Result                    |
| --------------------------------------------------------------- | ------------------------- |
| Offline integration tests                                       | 17 passed                 |
| clang-format, all five query files                              | Passed                    |
| Ruff lint / format                                              | Passed                    |
| Main patch: forward/cached and reverse apply checks             | Passed                    |
| Optional generator patch: forward apply to pristine pinned cuDF | Passed                    |
| Both patch application orders                                   | Identical trees           |
| Compile-only Q1/Q5/Q6/Q9/Q10, Vortex ON/OFF                     | 10/10 passed, no warnings |
| Separate generator regression test                              | Compiled                  |

Compile records are under ignored `build/cudf-ndsh-minimal-compile/`:
`summary.md`, `results.json`, `ninja-compdb.json`, `commands.txt`, `commands.json`,
`generator-test-summary.md`, and `generator-test-command.json`.

### Earlier checks

The following results describe historical source configurations:

| Check                                                       | Result    |
| ----------------------------------------------------------- | --------- |
| `cargo nextest run -p vortex-cuda pinned`                   | 5 passed  |
| `cargo +nightly fmt --all`                                  | Passed    |
| `cargo clippy --all-targets --all-features`                 | Passed    |
| clang-format, 8 edited C++ files                            | Passed    |
| `python3 -B benchmarks/cudf-ndsh/test_build_integration.py` | 14 passed |
| `.venv/bin/ruff check` / `format --check`, integration test | Passed    |
| Refreshed patch: forward/cached and reverse apply checks    | Passed    |

| Check                                                       | Result                                 |
| ----------------------------------------------------------- | -------------------------------------- |
| Q1/Q5/Q6/Q9/Q10 matched projected read/query, SF0.01        | All 28 states passed                   |
| Pinned Release SF1 / selected SF10 states                   | 28 / 20 passed                         |
| Same-fixture CPU references and independent synthetic cases | Passed                                 |
| Compute Sanitizer, all 28 SF0.01 states                     | 0 errors, no skips                     |
| Pinned adapter / CUDA FFI tests                             | 15 / 20 passed; adapter memcheck clean |
| Generator tests, including order-date/supplier regressions  | 8 passed; memcheck clean               |
| NDS-H / FFI CMake integration                               | 8 / 13 passed                          |
| Original Q10 Parquet-pushdown/write benchmark, SF0.01       | Passed                                 |

SF1 covered all Q9 amount engines; the earlier 20-state SF10 check used binary-op.
Supplemental cuDF 26.08 Debug runs provided execution evidence; performance tables
below use pinned Release cuDF.

## Timing contract

Both formats use identical logical fixtures, scan projections, post-read predicates,
and existing cuDF query operations. The default dataset uses the original pinned
cuDF generator. Details: [README.md](README.md).

- Compare CPU wall means. Timed work includes complete reads, import/copies/final
  materialization, query execution where selected, destruction, and device completion.
  Fixture writing, validation, and cache eviction are untimed.
- Before every timed cold callback, each input file gets `fdatasync` +
  `POSIX_FADV_DONTNEED`, followed by required `mincore` residency == 0. Cold Vortex
  data reads use `O_DIRECT`; metadata is buffered, and Parquet uses its native reader.
  This defines coldness at the **OS page-cache level**.
- Vortex reads local files through pinned-host staging → HtoD → GPU decode.
  RMM statistics cover cuDF allocations; Vortex memory requires separate accounting.

## Historical timings

**These measurements describe earlier source/dataset configurations, not current
baselines.** The current SF1 baseline is recorded [above](#current-release-sf1-run).
Future baselines require regenerated paired fixtures and labels for the generator
and match counts. All values are CPU wall means in **ms**; query means
include reads. Filenames containing `rebuilt` refer to those historical builds.

### SF10 warm

| Query               | Parquet read | Vortex read | Parquet query | Vortex query |
| ------------------- | -----------: | ----------: | ------------: | -----------: |
| Q1                  |       69.955 |      34.139 |       101.245 |       65.529 |
| Q5 (prior warm run) |       53.956 |      22.541 |        58.808 |       27.388 |
| Q6                  |       38.466 |      18.568 |        40.432 |       20.598 |
| Q9 binaryop         |       74.063 |      24.907 |        81.133 |       32.197 |
| Q9 AST              |       73.632 |      25.100 |        81.005 |       32.216 |
| Q9 transform        |       73.807 |      25.079 |        80.927 |       32.238 |
| Q10 (noisy)         |       67.160 |      29.154 |        75.112 |       37.495 |

### SF10 cold

| Query        | Parquet read | Vortex read | Parquet query | Vortex query |
| ------------ | -----------: | ----------: | ------------: | -----------: |
| Q1           |      141.331 |      62.248 |       174.044 |       93.190 |
| Q5           |      113.263 |      40.002 |       120.876 |       45.064 |
| Q6           |       81.915 |      30.455 |        85.390 |       32.310 |
| Q9 binaryop  |      142.833 |      48.408 |       153.286 |       55.584 |
| Q9 AST       |            — |           — |       152.012 |       55.646 |
| Q9 transform |            — |           — |       152.408 |       55.706 |
| Q10          |      121.270 |      49.929 |       131.099 |       58.386 |

Q9 cold has one reported read comparison; the other rows report query engines only.
Artifacts under ignored `build/cudf-ndsh-build/`:

- Q1 warm/cold: `sf10-q1-rebuilt-warm-cold-pinned-release.json`.
- Q5: `sf10-q5-cold-pinned-release.json`.
- Q6: `sf10-q6-cold-pinned-release.json`.
- Q9: `sf10-q9-cold-pinned-release.json`.
- Q10: `sf10-q10-cold-pinned-release.json`.

### SF1 Q1 warm/cold

Artifact: `build/cudf-ndsh-build/sf1-q1-rebuilt-warm-cold-pinned-release.json` (ignored).

| Cache | Parquet read | Vortex read | Parquet query | Vortex query |
| ----- | -----------: | ----------: | ------------: | -----------: |
| Warm  |       14.011 |       6.540 |        18.373 |       10.876 |
| Cold  |       18.771 |       6.561 |        23.383 |       10.926 |

Vortex cold read/query noise is high: 10.7% / 6.7%. These historical measurements
fall short of the ≥2× target across all read/query and warm/cold combinations.

## Profile evidence and safety

Historical full-read SQLite evidence records ~36.31 ms timed Vortex read: file reads
extend to 22.6 ms, first decode starts at 22.9 ms, HtoD transfers 1.62 GiB in 7.89 ms,
decode takes 7.25 ms, and final materialization 1.94 ms. These figures describe the
historical read configuration.

**Prefix profiler launches with `env -u ANTHROPIC_API_KEY`** (before `nsys`) and
otherwise use a sanitized environment. Existing old profiles contain sensitive
environment metadata: **do not inspect or publish that metadata**, or share raw
profiles containing it.

## Correctness scope

Exact projected names/types/values and independent CPU references cover generated
results, including zero matches. The original pinned generator can yield degenerate
Q6/Q10 results and low-SF supplier joins. Match counts identify degenerate queries;
use nondegenerate results for full-query performance claims. Synthetic cases cover
nonempty results, boundaries, joins, nulls, and empty inputs. Generated CPU oracles
target non-null schemas.

- Q1 counts/quantity sums are exact; floating checks use `1e-10` relative tolerance
  with an absolute floor of `1e-10`.
- Q6 checks that zero-match `SUM` is NULL and reports revenue as the string `"NULL"`.
  Cases cover boundary, sliced, float32, no-match, and empty inputs.
- Q9 follows the benchmark's unrounded `SUM(amount)` and preserves duplicate
  `partsupp` join multiplicity. Handwritten cases cover different costs for matching
  duplicates, unmatched duplicates, and empty inputs. The duplicate case expects
  seven matches and profits of 190/170/130 for ALPHA-1994/ALPHA-1996/ZULU-1995.

The optional [generator-fixes.patch](generator-fixes.patch) supplies four independent
fixes: discount/quantity RNG correlation, order year/month RNG correlation, price
alignment, and fractional supplier scale factors. It applies independently to pinned
cuDF and adds `NDSH_DATA_GENERATOR_TEST`. Its changes affect all NDS-H consumers,
including Vortex OFF. Selecting this dataset requires regenerated paired fixtures
and separately labeled baselines.

All required sources are tracked. Runtime benchmark artifacts, including JSON,
logs, binaries, Nsight reports, and SQLite exports, are ignored.
