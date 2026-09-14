# cuDF NDS-H Vortex POC

[Plan](../../CUDF_POC_PLAN.md) · [Validation](VALIDATION.md) · [Resume here](PROGRESS.md)

`upstream.patch` adds a **default-OFF Q1/Q5/Q6/Q9/Q10 read/query harness** for local
Parquet/Vortex comparisons, with `write_vortex` / `read_vortex` adapters. Both formats
use the same logical full-table data, identical scan projections, and cuDF post-read
filters. Native Parquet-pushdown/output benchmarks are separate.

Build integration lives in `cpp/benchmarks/ndsh/` with one include hook in
`cpp/benchmarks/CMakeLists.txt`. Current-source runtime validation and performance
measurements are pending; see [Validation](VALIDATION.md) for recorded checks and
[Progress](PROGRESS.md) for next steps.

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

[`reproduce.py`](reproduce.py) is the build/run entry point. It fetches pinned cuDF
and dependencies, applies `upstream.patch`, and builds Release cuDF, CUDA-enabled
Vortex, the smoke/adapter tests, and all five query executables through CMake.
It uses the original generator and this Vortex checkout as the local source.

This recipe targets **Linux AArch64/SBSA with a Hopper GPU (SM90)**. Prerequisites:

- CUDA SDK **13.1.2** (NVCC 13.1.115, runtime 13.1.80), including NVRTC, cuFile and
  NVML development files, plus a compatible NVIDIA driver. The default SDK path is
  `/usr/local/cuda-13.1`.
- Clang/libclang **18.1.3**, defaulting to `/usr/bin/clang++` and
  `/usr/lib/llvm-18/lib/libclang.so`.
- Rustup with the toolchain in [`rust-toolchain.toml`](../../rust-toolchain.toml),
  Git, curl, glibc ≥2.28, and micromamba **2.6.2** for the explicit environment lock.
- Network access for public source/package downloads and disk space for a full cuDF
  build. Put the work directory on disk, not tmpfs, for cold-I/O measurements.

From the Vortex root, with source changes committed:

```sh
micromamba create -y -p build/cudf-ndsh-env \
  --file benchmarks/cudf-ndsh/environment-linux-aarch64.lock

build/cudf-ndsh-env/bin/python benchmarks/cudf-ndsh/reproduce.py build \
  --toolchain build/cudf-ndsh-env

build/cudf-ndsh-env/bin/python benchmarks/cudf-ndsh/reproduce.py run \
  --toolchain build/cudf-ndsh-env --scale-factor 1
```

The Conda lock supplies GCC/G++ **14.3.0**, CMake, Ninja, Python and host libraries;
use a dedicated prefix without additional packages. [`build-lock.json`](build-lock.json)
pins cuDF/RMM, RAPIDS-CMake, CPM and other C++ inputs. Vortex's Rust dependencies
and build-time SDK downloads are specified by the checkout and `Cargo.lock`.
The runner builds Vortex's **25.12.19** `flatc` separately from cuDF's **24.3.25**.
[`nvcc131-cudf-hook.cmake`](nvcc131-cudf-hook.cmake) applies the compiler workaround
only to cuDF's CUDA sources.

The default work directory is `build/cudf-ndsh-repro`. Change `--work-dir` for a new
source revision; an existing directory must belong to the same recipe. SDK locations
can be selected with `--cuda-root`, `--clangxx` and `--libclang`. Commands are bounded
by `--timeout` (1,200 seconds each); `--jobs` defaults to 2 and `--cargo-jobs` to 4.
If a build times out, inspect its log before explicitly rerunning with a larger limit.

## Run and collect results

`run` checks the recorded source revision and binary/library hashes, runs smoke and
adapter checks, then runs Q1/Q5/Q6/Q9/Q10 **sequentially on device 0**. Each query
covers Parquet/Vortex × read/full-query × warm/cold; Q9 covers all three existing
amount engines. Correctness checks and fixture generation remain outside timing.
Use `--scale-factor 10` for SF10, or `--queries 6` for a focused run.

Commands, selected environment and tool versions go under `<work-dir>/logs/`;
benchmark JSON and build provenance go under `<work-dir>/results/`. Temporary fixtures
use `<work-dir>/tmp/`, on the chosen filesystem. Missing, skipped
or untimed states fail the run. Build artifacts and results are ignored; the recipe,
locks and benchmark source patch are versioned. **The clean recipe has not yet been
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
