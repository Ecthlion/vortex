# cuDF NDS-H Vortex benchmarks

A default-OFF Q1/Q5/Q6/Q9/Q10 harness compares local Parquet and Vortex files using
matched logical data, projections, and existing cuDF query operations. Native
Parquet-pushdown/output benchmarks remain separate. This is benchmark integration,
not an installed client-facing Vortex/cuDF API.

The [upstream POC goal](https://github.com/NVIDIA/cudf/issues/23877#issuecomment-5457730105)
is ≥2× for end-to-end projected reads and full queries at SF1/SF10, warm and cold.
The full goal is not established: the recorded SF1 run has empty Q6/Q10 results,
and current-source SF10 validation is pending. [Recorded results](VALIDATION.md)
predate the recent simplification and build-option changes; they do not validate
the current branch.

## Build from a clean checkout

[`reproduce.py`](reproduce.py) fetches pinned cuDF sources, applies
[`upstream.patch`](upstream.patch), and builds Release cuDF, CUDA-enabled Vortex,
the smoke/adapter tests, and five query executables. It uses the original generator.

Requirements:

- Linux, CMake 4+, Ninja, Python 3.11+, Git, curl, libclang, and Rustup with the
  [repository toolchain](../../rust-toolchain.toml).
- A compatible host compiler/driver and a **full CUDA SDK**: CUDART, NVRTC, cuRAND,
  nvJitLink, cuFile, NVTX, and profiler development headers. In conda, include
  `cuda-nvtx-dev` and `cuda-profiler-api`; an NVCC-only installation is insufficient.
- The recorded working combination is NVCC **13.0.88**, GCC **14.3.0**, and GH200
  driver **595.71.05**. The recipe accepts CUDA 12.8+, but tested 12.8 and 13.1
  compilers failed full cuDF builds; see [toolchain evidence](VALIDATION.md#recorded-build-and-toolchain).

From the Vortex root, with source changes committed; this example selects Hopper:

```sh
python3 benchmarks/cudf-ndsh/reproduce.py build \
  --cmake-arg=-DCMAKE_CUDA_ARCHITECTURES=90

python3 benchmarks/cudf-ndsh/reproduce.py run --scale-factor 1
```

The runner preserves compiler/environment settings such as `CC`, `CXX`, `CUDACXX`,
`CUDAHOSTCXX`, `CMAKE_PREFIX_PATH`, and `LIBCLANG_PATH`. Pass additional CMake
settings with repeated `--cmake-arg=-DNAME=VALUE`, including `CMAKE_CUDA_COMPILER`,
`CMAKE_CUDA_HOST_COMPILER`, and `CUDAToolkit_ROOT`. Use absolute paths for file-valued
settings. Vortex forwards the selected toolkit, architectures, and explicit CUDA
host compiler to Cargo; cuDF owns nvCOMP selection.

### Vortex source selection

For patched cuDF benchmark builds (`BUILD_BENCHMARKS=ON`), the single Vortex-specific
switch is `-DCUDF_WITH_VORTEX=ON`, replacing `CUDF_NDSH_WITH_VORTEX`. Without a local
override, CPM downloads Vortex at configure time, pinned to
`90723345eeed838405341da9d44e02c94f10e6be`. Unset/`OFF` does not fetch or configure Vortex.

Optional `-DFETCHCONTENT_SOURCE_DIR_VORTEX=/absolute/path` selects a local checkout
instead. The runner supplies this override automatically so the harness and Vortex
library come from the checkout being recorded.

**The default download requires publishing that pin first.** At the last remote
inspection, the branch still lacked the current harness. Use the local override
until the pin is available remotely.

[`build-lock.json`](build-lock.json) pins cuDF/RMM, RAPIDS-CMake, CPM, and C++ source
dependencies; `Cargo.lock` and the Vortex checkout specify Rust dependencies and
SDK downloads. System tools/libraries come from the caller's environment and are
recorded with the build. Vortex's **25.12.19** `flatc` is built separately from cuDF's
**24.3.25**.

### Work directories and results

The default work directory is `build/cudf-ndsh-repro`. Use a new `--work-dir` after
changing source, compiler settings, or CMake definitions. Downloads require network
access and a full cuDF build needs substantial disk space. Put fixtures on disk,
not tmpfs, for cold-I/O measurements. Commands default to `--timeout=1200`,
`--jobs=2`, and `--cargo-jobs=4`; a timeout does not trigger an automatic retry.

`run` checks the recorded clean source revision and binary/library hashes, reuses
the build environment, and gives the recorded cuDF library precedence on the library
search path. It runs smoke/adapter checks, then queries **sequentially on device 0**.
Each query covers Parquet/Vortex × read/query × warm/cold; Q9 covers all three amount
engines. Missing, skipped, duplicate, or untimed states fail the run. Use
`--scale-factor 10` for SF10 or `--queries 6` for a focused run.

Commands, selected environment, and tool versions go under `<work-dir>/logs/`;
benchmark JSON and provenance go under `<work-dir>/results/`; fixtures use
`<work-dir>/tmp/`. These artifacts are ignored. Do not relabel old binaries as a new
source revision or reuse mixed-era build directories in place of a fresh build.

## I/O and timing contract

- Both formats use matched full-table fixtures, the same projected columns, and
  cuDF post-read filters. Fixture generation and correctness checks are untimed.
- Writes use 16,777,216-row cuDF chunks → compact host Arrow → CPU-compressed
  CUDA-flat blocks. Explicit row blocks disable byte coalescing and outer layout
  dictionaries; file blocks and output batches are row-based.
- Reads use pooled cacheable pinned-host staging → HtoD → GPU decode → retained Arrow
  Device imports → one final owning cuDF materialization. Large reads use 4 MiB
  chunks, up to 32 concurrent host reads per file, and one destination GPU allocation.
- The adapter supports device 0, local files, flat typed columns, and ordered
  top-level projections. It retains up to 8 GiB in CUDA's default memory pool
  between synchronized reads. Peak read memory includes retained Vortex batches
  and the owning result; **RMM statistics cover cuDF allocations only**.
- Warm states use buffered reads and a read-only warmup. Before each timed cold
  iteration, selected-format files receive `fdatasync` + `POSIX_FADV_DONTNEED`,
  followed by required `mincore` residency == 0. Cold Vortex data uses `O_DIRECT`,
  metadata stays buffered, and Parquet uses its native reader. This measures
  **OS-page-cache coldness**, not coldness of all caches.
- Compare CPU wall means, including complete reads/import/materialization, query
  work where selected, owner destruction, and device-wide synchronization covering
  producer cleanup. Writing, correctness checks, and eviction are outside timing.

## Dataset and correctness

The original pinned generator can produce empty Q6/Q10 results and low-SF supplier
joins. Report generator choice and match counts: empty results establish correctness
and execution behavior, **not meaningful nonempty full-query performance**.

Exact projected names/types/values and independent CPU references check generated
non-null inputs. Synthetic cases cover nonempty results, boundaries, slices, joins,
nulls, and empty inputs. Q1 counts/quantity sums remain exact; floating comparisons
use `1e-10` relative tolerance with a `1e-10` absolute floor. Q6 includes float32
predicate boundaries and requires zero-match `SUM` to be NULL, reported as `"NULL"`.
Q9 uses unrounded `SUM(amount)` and preserves duplicate `partsupp` join multiplicity.
Handwritten cases cover different costs for matching duplicates, unmatched duplicates,
and empty inputs. The duplicate case expects seven matches and profits of
190/170/130 for ALPHA-1994/ALPHA-1996/ZULU-1995.

Optional [`generator-fixes.patch`](generator-fixes.patch) independently fixes
four generator issues: discount/quantity RNG correlation, order year/month RNG
correlation, price alignment, and fractional supplier scale factors. It adds
`NDSH_DATA_GENERATOR_TEST` and affects all NDS-H consumers, including Vortex OFF.
The runner does not apply it. Selecting it requires regenerated paired fixtures
and separately labeled baselines.

## Maintaining the integration

- [`src/vortex_ndsh/`](src/vortex_ndsh/) owns the adapter, local benchmarks, CPU
  references, and fixture/cache helpers. Each `qNN.inc` is included once, after its cuDF
  query definitions through the target-private `CUDF_NDSH_QUERY_EXTENSION`, reusing
  the query implementation rather than copying it.
- [`vortex.cmake`](vortex.cmake) owns build/benchmark wiring;
  [`tests/`](tests/) contains the adapter and smoke executables. The cuDF patch owns
  query reader/consumer callbacks, named-table generation, and the opt-in CPM loader.
- Edit cuDF integration in ignored `build/cudf-ndsh-src`, based on
  `5339497a1a17d799687cbf189fb113411fb015ca`; do not modify `/home/ubuntu/cudf`.
  From the Vortex root, re-export the patch from that checkout, including added files:

    ```sh
    git -C build/cudf-ndsh-src add -N cpp/cmake/thirdparty/get_vortex.cmake
    git --no-pager -C build/cudf-ndsh-src diff -- \
      cpp/CMakeLists.txt cpp/cmake/thirdparty/get_vortex.cmake cpp/benchmarks \
      > benchmarks/cudf-ndsh/upstream.patch
    ```

### Optional checks

```sh
python3 -B -m unittest discover -s benchmarks/cudf-ndsh -v
uvx ruff check benchmarks/cudf-ndsh/*.py
uvx ruff format --check benchmarks/cudf-ndsh/*.py
```

Offline Python tests cover build wiring, reproduction safeguards, and selected
benchmark-policy guards. They do not replace real cuDF builds or GPU validation.

### Profiling safety

Prefix profiler launches with `env -u ANTHROPIC_API_KEY` and otherwise use a sanitized
environment. Old profiles contain sensitive environment metadata: **do not inspect
or publish that metadata, or share raw profiles containing it**. Historical profile
observations are recorded in [Validation](VALIDATION.md#profile-evidence).
