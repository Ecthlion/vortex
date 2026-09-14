# cuDF NDS-H Vortex POC

[Plan](../../CUDF_POC_PLAN.md) · [Validation](VALIDATION.md) · [Resume here](PROGRESS.md)

A **default-OFF Q1/Q5/Q6/Q9/Q10 read/query harness** compares local Parquet/Vortex
files using the same logical data, scan projections, and cuDF post-read filters.
Native Parquet-pushdown/output benchmarks remain separate.

## Source layout

- [`upstream.patch`](upstream.patch) changes only cuDF's five query files, named-table
  generation helpers, and one opt-in CMake hook. The queries retain their original
  operations; reader and result-consumer callbacks allow matched comparisons.
- [`src/vortex_ndsh/`](src/vortex_ndsh/) owns the Vortex adapter, local benchmarks,
  CPU references, and fixture/cache helpers. Each `qNN.inc` is included once, after
  its cuDF query definitions, through the target-private `CUDF_NDSH_QUERY_EXTENSION`.
  It reuses those definitions rather than copying the query implementation.
- [`vortex.cmake`](vortex.cmake) builds the harness and Vortex library from the same
  checkout. [`tests/`](tests/) contains the adapter and build-smoke executables.

Current-source build/runtime validation and performance measurements are pending;
see [Validation](VALIDATION.md) and [Progress](PROGRESS.md).

## I/O and timing contract

- Both formats use existing cuDF query operations after reading the projected columns.
- Write: 16,777,216-row cuDF chunks → compact host Arrow → CPU-compressed CUDA-flat
  blocks. Explicit row blocks disable byte coalescing and outer layout dictionaries.
- Read: pooled cacheable pinned-host staging → HtoD → GPU decode → retained Arrow
  Device imports → one final owning cuDF materialization. The adapter uses device 0,
  local files, flat typed columns, and ordered top-level projections, with up to 8 GiB
  retained in CUDA's default memory pool between synchronized reads. Peak read memory
  includes retained Vortex batches and the owning result; chunking is row-based.
- `cache=warm` uses buffered reads with the existing OS page cache. `cache=cold`
  measures **OS-page-cache coldness**: before every manual cold callback's timed
  portion, each selected-format file receives `fdatasync` + `POSIX_FADV_DONTNEED`,
  followed by required `mincore` residency == 0. Cold Vortex data uses `O_DIRECT`,
  metadata stays buffered, and Parquet uses its native reader.
- Compare CPU wall time, including complete reads/import/materialization, query work
  where selected, destruction, and device-wide synchronization covering producer
  cleanup. Writing, correctness checks, and eviction are untimed. **RMM memory
  statistics cover cuDF allocations only.**

## Dataset and correctness

Comparisons default to the **original pinned cuDF data generator**. It can produce
empty/degenerate Q6/Q10 results and low-SF supplier joins. Exact projection/value and
independent CPU result checks cover zero matches as well as synthetic nonempty,
boundary, null, and empty cases. Match counts identify degenerate queries, which are
**not meaningful full-query performance evidence**.

Q6 checks that zero-match `SUM` is NULL and reports revenue as the string `"NULL"`;
its tests include boundary, sliced, float32, no-match, and empty inputs. Q9 follows the
benchmark's unrounded `SUM(amount)` and preserves duplicate `partsupp` join
multiplicity, with matching/unmatched duplicate and empty-input cases.

The optional [generator-fixes.patch](generator-fixes.patch) supplies four generator
fixes and the `NDSH_DATA_GENERATOR_TEST` regression target. It applies independently
to pinned cuDF and affects all NDS-H consumers, including Vortex OFF. When selecting
this dataset, regenerate both formats' fixtures, label them as generator-fixed, and
collect separately labeled baselines.

## Build from a clean checkout

[`reproduce.py`](reproduce.py) wraps cuDF's CMake build: it fetches pinned sources,
applies `upstream.patch`, and builds Release cuDF, CUDA-enabled Vortex, the smoke/adapter
tests and all five query executables. It uses the original generator.

Start in a working **Linux cuDF development environment** with an NVIDIA CUDA toolkit
**12.8 or newer** including profiler headers (`cuda-profiler-api` in conda), a compatible
host compiler/driver, CMake 4+, Ninja, Python 3.11+,
Git and curl. Vortex also requires libclang and Rustup with the toolchain in
[`rust-toolchain.toml`](../../rust-toolchain.toml). See
[Validation](VALIDATION.md#current-build-status) for current compatibility evidence.

From the Vortex root, with source changes committed; this example selects Hopper:

```sh
python3 benchmarks/cudf-ndsh/reproduce.py build \
  --cmake-arg=-DCMAKE_CUDA_ARCHITECTURES=90

python3 benchmarks/cudf-ndsh/reproduce.py run --scale-factor 1
```

The runner preserves build settings such as `CC`, `CXX`, `CUDACXX`, `CUDAHOSTCXX`,
`CMAKE_PREFIX_PATH` and `LIBCLANG_PATH`. Additional CMake definitions can be passed
with repeated `--cmake-arg=-DNAME=VALUE`, including `CMAKE_CUDA_COMPILER` and
`CMAKE_CUDA_HOST_COMPILER`. The runner selects this checkout through
`FETCHCONTENT_SOURCE_DIR_VORTEX`; direct cuDF CMake users set that path alongside
`CUDF_NDSH_WITH_VORTEX=ON`. Use absolute paths for file-valued definitions. Vortex's
CMake-to-Cargo bridge forwards the selected toolkit, architectures and explicit CUDA
host compiler. cuDF owns nvCOMP selection for the chosen environment.

[`build-lock.json`](build-lock.json) pins cuDF/RMM, RAPIDS-CMake, CPM and the C++ source
dependencies. System tools and libraries come from the caller's environment; their
selected compiler paths, versions and flags are recorded with the build. Vortex's
Rust dependencies and build-time SDK downloads are specified by the checkout and
`Cargo.lock`. The runner builds Vortex's **25.12.19** `flatc` separately from cuDF's
**24.3.25**.

The default work directory is `build/cudf-ndsh-repro`. Use a new `--work-dir` when
changing source, compiler settings or CMake definitions. Public downloads need network
access. Allow disk space for a full cuDF build and put the work directory on disk,
not tmpfs, for cold-I/O measurements. Commands are bounded by `--timeout` (1,200 seconds
each); `--jobs` defaults to 2 and `--cargo-jobs` to 4. If a build times out, inspect its
log before explicitly rerunning with a larger limit.

## Run and collect results

`run` reuses the recorded build environment, giving the freshly built cuDF library
precedence on the library search path. It verifies the clean source revision and
binary/library hashes, runs smoke and adapter checks, then runs Q1/Q5/Q6/Q9/Q10
**sequentially on device 0**.
Each query covers Parquet/Vortex × read/full-query × warm/cold; Q9 covers all three existing
amount engines. Correctness checks and fixture generation remain outside timing.
Use `--scale-factor 10` for SF10, or `--queries 6` for a focused run.

Commands, selected environment and tool versions go under `<work-dir>/logs/`;
benchmark JSON and build provenance go under `<work-dir>/results/`. Temporary fixtures
use `<work-dir>/tmp/`, on the chosen filesystem. Missing, skipped or untimed states fail
the run. Build artifacts and results are ignored; the harness sources, cuDF patch,
recipe and dependency lock are versioned. **The clean recipe has not yet been
executed end to end**; recorded build evidence is in [Validation](VALIDATION.md).

Prefix profiler launches with `env -u ANTHROPIC_API_KEY` and follow the
[profile metadata safety rules](VALIDATION.md#profile-evidence-and-safety).

## Optional checks

```sh
python3 -B -m unittest discover -s benchmarks/cudf-ndsh -v
python3 -B -m unittest discover -s vortex-ffi/cmake/tests -v
uvx ruff check benchmarks/cudf-ndsh/*.py
uvx ruff format --check benchmarks/cudf-ndsh/*.py
```

Results and artifacts are recorded in [Validation](VALIDATION.md).
