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

## Apply and build

From the Vortex root, for a **fresh** cuDF checkout:

```sh
git clone https://github.com/NVIDIA/cudf.git build/cudf-ndsh-src
git -C build/cudf-ndsh-src checkout --detach 5339497a1a17d799687cbf189fb113411fb015ca
git -C build/cudf-ndsh-src apply --check ../../benchmarks/cudf-ndsh/upstream.patch
git -C build/cudf-ndsh-src apply ../../benchmarks/cudf-ndsh/upstream.patch
```

The development checkout is already patched; never modify `/home/ubuntu/cudf`.
Build instructions are in patched
[`cpp/benchmarks/ndsh/VORTEX.md`](../../build/cudf-ndsh-src/cpp/benchmarks/ndsh/VORTEX.md).
The pinned Release build tree is `build/cudf-ndsh-build`. Use cuDF headers and libraries
from the same pinned revision.

**Local Vortex sources are required:** use a complete checkout with CUDA-layout edition
registration, device decimal slicing, bitmap alignment/padding, dictionary export, and
`vx_cuda_scan_path_arrow_device_stream_projected`. The retained base pin
`bffdca1109e99e6957ea2fc18f4a7809c88e0a0c` predates these prerequisites; CMake requires
`FETCHCONTENT_SOURCE_DIR_VORTEX` until a fixed immutable revision is published.

## Run after building the current source

Benchmarks are named `ndsh_q{1,5,6,9,10}_local`. The default scale-factor axis includes
SF10; select axes explicitly for comparable runs. Example SF10 Q10 matrix:

```sh
build/cudf-ndsh-build/benchmarks/NDSH_Q10_NVBENCH \
  --benchmark ndsh_q10_local --axis scale_factor=10 \
  --axis 'format=[parquet,vortex]' --axis 'workload=[read,q10]' \
  --axis 'cache=[warm,cold]' --min-samples 3 --timeout 30 \
  --json build/cudf-ndsh-build/sf10-q10-final-source.json
```

Use `scale_factor=1` for SF1. Q1/Q5/Q6/Q9 use executables `NDSH_Q01_NVBENCH`,
`NDSH_Q05_NVBENCH`, `NDSH_Q06_NVBENCH`, and `NDSH_Q09_NVBENCH`, respectively, with
matching benchmark/workload names. Q9 also needs explicit
`--axis 'engine=[binaryop,ast,transform]'`. Run GPU benchmarks sequentially. Prefix
profiler launches with `env -u ANTHROPIC_API_KEY` and follow the
[profile metadata safety rules](VALIDATION.md#profile-evidence-and-safety).
JSON, logs, binaries, Nsight reports, and SQLite exports are ignored, not committed.

## Optional checks

```sh
python3 -B -m unittest discover -s benchmarks/cudf-ndsh -v
python3 -B -m unittest discover -s vortex-ffi/cmake/tests -v
ruff check benchmarks/cudf-ndsh/test_build_integration.py
ruff format --check benchmarks/cudf-ndsh/test_build_integration.py
```

Results and artifacts are recorded in [Validation](VALIDATION.md).
