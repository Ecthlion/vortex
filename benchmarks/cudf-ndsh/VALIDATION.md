# Validation

Checkpoint: 2026-09-15. Chunked-I/O source: `98b3d12746`.
Full-matrix baseline source: `91142e2c18`. Pinned cuDF:
`5339497a1a17d799687cbf189fb113411fb015ca`, Release (`-O3 -DNDEBUG`).
Hardware/toolchain: NVIDIA GH200, driver **595.71.05**, NVCC **13.0.88**, GCC **14.3.0**.
[Current state](PROGRESS.md) · [Setup and commands](README.md)

## Chunked-I/O optimization (SF1)

Nsight identified large host reads as the main bottleneck: the final Q6 warm read
spent 3.698 ms in one 47.98 MB `pread`, versus 0.093 ms executing decode kernels.
These are instrumented observations, not benchmark timings.

`98b3d12746` splits large reads into 4 MiB chunks on the existing I/O runtime, with
at most 32 concurrent host reads per open file. Completed chunks transfer directly
into one GPU allocation. Fixtures, encodings, projections, and cuDF operations are
unchanged.

The following uninstrumented CPU-wall means are in **ms**. Before and after use
`--min-samples 100 --timeout 5`; Parquet is from the after run. Queries include reads.

| Query | Cache | Workload | Vortex before | Vortex after | Parquet | Parquet / after |
| ----- | ----- | -------- | ------------: | -----------: | ------: | --------------: |
| Q1    | warm  | read     |         6.298 |        4.141 |  14.062 |           3.40× |
| Q1    | cold  | read     |         6.070 |        4.466 |  18.887 |           4.23× |
| Q1    | warm  | query    |        10.686 |        8.618 |  18.876 |           2.19× |
| Q1    | cold  | query    |        10.412 |        8.938 |  23.809 |           2.66× |
| Q6    | warm  | read     |         4.036 |        1.875 |   7.937 |           4.23× |
| Q6    | cold  | read     |         3.017 |        2.141 |  11.113 |           5.19× |
| Q6    | warm  | query    |         4.568 |        2.435 |   8.429 |           3.46× |
| Q6    | cold  | query    |         3.809 |        2.728 |  11.625 |           4.26× |

All 16 states passed correctness checks. Q1 has 4,497,687 matches/four groups;
**Q6 still has zero matches and SUM NULL**, so its full-query timings do not establish
nonempty-query performance. Vortex after-run CPU relative SD ranges from 1.5–6.2%;
before-run cold measurements were noisier (8.0–13.8%). Five-second sampling limits
can emit timeout warnings while still producing valid results.

Validation: targeted Release consumers built; **15 adapter tests and 33 focused
CUDA tests passed**, with no ignored tests. Rust filters were `pinned::tests` and
`pooled_read_at::file::tests`, using `cargo test --locked --offline -p vortex-cuda
--lib --features _test-harness`, the CMake-selected Release toolchain/target, and
`--test-threads=1`.

Artifacts under ignored `build/cudf-ndsh-perf-20260914/`:

- `baseline-sf1-q{1,6}.json` and `chunked-sf1-q{1,6}.json`.
- `baseline-build.json`, `baseline-binaries/`, and `chunked-provenance.json`
  (source/binary hashes; benchmarks ran before the source commit).
- Baseline profile files: `baseline-q6-{warm,cold}.sqlite`, `baseline-q1-warm.sqlite`;
  range-scoped analysis: `derived-final-read-analysis.json`.
- Commands/logs: `logs/20260914T203030.285386Z/` (before) and
  `logs/20260914T204642.079773Z/` (after and adapter checks).
- `focused-tests-summary.json` and `focused-tests-sanitized-{build,pinned,pooled}.json`
  record exact test commands and results.

Q5/Q9/Q10 measurements with this change, SF10, and current memcheck remain pending.
The remaining-query rebuild was interrupted; inspect it before resuming. The
original work directory now contains incrementally rebuilt artifacts, so its old
`build.json` no longer describes all binaries. Use a fresh work directory for the
tracked clean-build recipe; preserved baseline binaries remain in the experiment directory.

## Current build status

At baseline `91142e2c18`, the [build recipe](README.md#build-from-a-clean-checkout)
produced fresh Release cuDF and CUDA-enabled Vortex, plus **all seven executables**. After the Python runner
was interrupted, its CMake child finished. An incremental target build passed
(**0.5 s, exit 0**); source and toolchain checks then allowed recovery of the binary
hash record through `reproduce.py` helpers. `resumed_from` identifies the original logs.

Three in-branch build/runtime fixes are included: `1ab26426d8` resolves the NVCC
realpath before Cargo tool lookup; `a9d705465e` declares explicit NVTX/KvikIO links;
`91142e2c18` initializes the smoke test with `cudaSetDevice` before its stream check.

NVCC **13.0.88** is validated with the full SDK described in the README. The recipe's
12.8 minimum does not imply compiler compatibility: tested NVCC **12.8.93** has a
parameter-pack emission bug, and **13.1.115** a private
`cudf::ast::literal::ast_scalar` access bug; both were reproduced with host GCC 13
and 14. Earlier 12.8 success on isolated access probes did not validate a complete
build. Compatibility probes are under ignored
`build/cudf-ndsh-repro-cuda128-release-v2/compiler-compat-probes-20260914/`.

## Current Release SF1 run

This is the last **complete five-query baseline**, before the chunked-I/O change
above. Successful command from the Vortex root:

```sh
python3 -B benchmarks/cudf-ndsh/reproduce.py run \
  --work-dir=/home/ubuntu/vortex/build/cudf-ndsh-repro-cuda130-release-v2 \
  --scale-factor=1 --timeout=1200 --min-samples=3 --sample-timeout=30
```

Artifacts below are relative to ignored `build/cudf-ndsh-repro-cuda130-release-v2/`:

- Initial build: `logs/20260914T171459.140647Z/`.
- Completion/provenance recovery: `logs/20260914T172951.677938Z/`.
- Successful run: `logs/20260914T173000.589177Z/`.
- Results: `results/20260914T173000.589488Z/sf1-q{1,5,6,9,10}.json`, with
  `build.json` (hashes/toolchain/`resumed_from`) and `run.json` in the same directory.

Recorded checks at this checkpoint (not rerun for this documentation update):

| Check                                              | Result                                    |
| -------------------------------------------------- | ----------------------------------------- |
| Fresh Release cuDF + Vortex / executables          | Built / 7 of 7 built                      |
| Build smoke / adapter tests                        | Passed / 15 passed                        |
| Offline harness / CUDA CMake tests                 | 38 passed / 8 passed                      |
| SF1 read/query × format × cache × Q9 engine matrix | 56 unique, valid passing states; no skips |
| Sampling                                           | 20–1,376 samples/state; 31,226 total      |

No runtime hard errors occurred. **No current-source memcheck was run**; sanitizer
results below are historical only.

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

Q6/Q10 are degenerate: their timings establish execution/read behavior, **not
meaningful nonempty full-query performance**.

### CPU-wall timings

CPU wall means in **ms**, taken from JSON `nv/cold/time/cpu/mean` × 1,000; query
includes reads. Cache labels refer to the harness axis, not NVBench's sampling
label. See the [timing contract](#timing-contract): cold means verified **OS page-cache
eviction, not all caches**, with `O_DIRECT` Vortex data reads versus native Parquet.

| Query / engine | Cache | Parquet read | Vortex read | Parquet query | Vortex query |
| -------------- | ----- | -----------: | ----------: | ------------: | -----------: |
| Q1             | warm  |       14.068 |       6.348 |        18.760 |       10.800 |
| Q1             | cold  |       18.808 |       5.593 |        23.676 |       10.167 |
| Q5             | warm  |       17.512 |       8.423 |        20.514 |       11.242 |
| Q5             | cold  |       22.463 |       9.167 |        25.540 |       12.088 |
| Q6             | warm  |        7.914 |       4.105 |         8.390 |        4.604 |
| Q6             | cold  |       11.074 |       3.241 |        11.658 |        3.807 |
| Q9 binaryop    | warm  |       27.020 |       8.510 |        30.128 |       11.536 |
| Q9 AST         | warm  |       26.999 |       8.491 |        29.984 |       11.453 |
| Q9 transform   | warm  |       27.007 |       8.503 |        30.020 |       11.516 |
| Q9 binaryop    | cold  |       32.746 |       9.322 |        36.044 |       12.125 |
| Q9 AST         | cold  |       32.865 |       9.248 |        35.877 |       12.160 |
| Q9 transform   | cold  |       32.749 |       9.279 |        36.090 |       12.268 |
| Q10            | warm  |       26.096 |      11.065 |        27.366 |       12.393 |
| Q10            | cold  |       30.724 |      10.872 |        32.125 |       11.882 |

Q9 read is repeated under three engine labels; these are not three distinct read
implementations. Across the 28 labeled Parquet/Vortex comparisons, speedups range
from **1.74× to 3.55×**, with **24/28 ≥2×**. The four below target are warm Q1 query
(1.74×), Q5 query (1.82×), Q6 read (1.93×) and Q6 query (1.82×).

Median CPU relative standard deviation is **1.30%**; 8 states exceed 5%, including
3 above 10%: Vortex Q1 cold read **11.52%**, Q6 cold read **11.45%**, and Q6 cold
query **10.39%**. Two NVBench sampling-timeout warnings still emitted valid passes:
Parquet Q9 binaryop warm read (**1,109 samples, 0.59% CPU SD**) and Parquet Q9 AST
warm query (**1,000 samples, 0.75% CPU SD**). These hit the 30-second sampling limit
above NVBench's 0.50% GPU-noise threshold, not the runner's 1,200-second command limit.

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

These checks preceded the rebase and are not validation of `91142e2c18`:

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

All benchmark JSON, logs, Nsight reports, and SQLite exports are ignored.
