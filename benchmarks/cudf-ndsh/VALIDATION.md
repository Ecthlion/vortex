# Validation

Checkpoint: 2026-09-14. Pinned cuDF:
`5339497a1a17d799687cbf189fb113411fb015ca`, Release (`-O3 -DNDEBUG`).
[Current state](PROGRESS.md) · [Setup and commands](README.md)

## Current build status

The [clean build/run recipe](README.md#build-from-a-clean-checkout) pins source inputs
and records the caller-selected toolchain. **It has not been executed end to end.**
Current-source runtime validation, memcheck and performance measurements remain pending.
The RAPIDS-CMake pin defines the recipe going forward; historical build records do not
establish which RAPIDS-CMake revision they used.

CUDA 12.8+ is the recipe's version floor, not a claim that every compiler release is
validated. Recorded NVCC 13.1/GCC 14 probes fail in unmodified cuDF join code with a
private `cudf::ast::literal::ast_scalar` access error at `std::bool_constant<true>`.
A standalone standard-C++ reproducer has the same failure. NVCC 12.8 compiled the
isolated probes; its complete cuDF/Vortex build is still unvalidated. This compiler
issue needs a separate upstream resolution. Probe sources/logs are under ignored
`build/cudf-ndsh-build/access-repro/`.

## Earlier isolated build

The rebased Vortex Release archive and all six consumer executables built and linked
against an existing pinned Release cuDF library. This did not perform a clean,
fully pinned cuDF build.

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

## Recorded source checks

These checks preceded the rebase and this documentation update:

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
baselines.** Fresh baselines require regenerated paired fixtures and labels for the
generator and match counts. All values are CPU wall means in **ms**; query means
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
