# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Build pinned cuDF/Vortex from source and run the matched NDS-H matrix."""

import argparse
import datetime
import hashlib
import itertools
import json
import math
import os
import platform
import shlex
import shutil
import signal
import subprocess
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
QUERIES = (1, 5, 6, 9, 10)
TARGETS = ("NDSH_VORTEX_BUILD_SMOKE", "NDSH_VORTEX_IO_TEST", *(f"NDSH_Q{q:02}_NVBENCH" for q in QUERIES))
RECIPE_FILES = (
    "reproduce.py",
    "build-lock.json",
    "environment-linux-aarch64.lock",
    "nvcc131-cudf-hook.cmake",
    "upstream.patch",
)


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, value: dict | list):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def timestamp() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def git_output(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "--no-pager", "--no-optional-locks", "-C", str(path), *args], text=True, timeout=30
    ).strip()


def locked_environment(prefix: Path):
    """Require the explicit package set, rather than an ambient RAPIDS/Python environment."""
    expected = {}
    for line in (HERE / "environment-linux-aarch64.lock").read_text().splitlines():
        if line.startswith("https://"):
            package, sha256 = line.rsplit("/", 1)[1].split("#")
            name = package.removesuffix(".conda").removesuffix(".tar.bz2")
            expected[f"{name}.json"] = sha256
    actual = {p.name: json.loads(p.read_text()).get("sha256") for p in (prefix / "conda-meta").glob("*.json")}
    if not expected or actual != expected:
        raise RuntimeError("Toolchain prefix differs from environment-linux-aarch64.lock; create a fresh prefix")
    if any((prefix / "lib/python3.12/site-packages").glob("*.dist-info")):
        raise RuntimeError("Use a fresh toolchain prefix without pip-installed packages")


def environment(args: argparse.Namespace) -> dict[str, str]:
    prefix, cuda = args.toolchain, args.cuda_root
    return {
        "HOME": str(Path.home()),
        "PATH": os.pathsep.join(map(str, (prefix / "bin", cuda / "bin", Path.home() / ".cargo/bin", "/usr/bin", "/bin"))),
        "LD_LIBRARY_PATH": os.pathsep.join(map(str, (prefix / "lib", cuda / "lib64"))),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_EDITOR": "true",
        "GIT_TERMINAL_PROMPT": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(args.work_dir / "tmp"),
        "CARGO_BUILD_JOBS": str(args.cargo_jobs),
        "NVCC_CCBIN": str(prefix / "bin/aarch64-conda-linux-gnu-g++"),
        "LIBCLANG_PATH": str(args.libclang),
        "FLATC": str(args.work_dir / "flatc-build/flatc"),
        "PKG_CONFIG_PATH": str(prefix / "lib/pkgconfig"),
    }


class Runner:
    def __init__(self, work: Path, env: dict[str, str], timeout: int) -> None:
        self.work, self.env, self.timeout = work, env, timeout
        (work / "tmp").mkdir(exist_ok=True)
        self.logs = work / "logs" / timestamp()
        self.logs.mkdir(parents=True)
        self.commands = []
        save(self.logs / "environment.json", env)

    def run(self, label: str, argv: list[str | Path | int], cwd: Path | None = None) -> str:
        argv = list(map(str, argv))
        cwd = Path(cwd or self.work)
        log = self.logs / f"{len(self.commands):03}-{label}.log"
        print(f"[{label}] {shlex.join(argv)}\n  log: {log}", flush=True)
        start = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.Popen(
                argv, cwd=cwd, env=self.env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                code = process.wait(timeout=self.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise RuntimeError(f"{label} interrupted or exceeded {self.timeout}s; see {log}") from error
            finally:
                self.commands.append(
                    {
                        "argv": argv,
                        "cwd": str(cwd),
                        "exit_code": process.returncode,
                        "seconds": time.monotonic() - start,
                        "log": str(log),
                    }
                )
                save(self.logs / "commands.json", self.commands)
        output = log.read_text(errors="replace")
        if code:
            raise RuntimeError(f"{label} failed ({code}); see {log}\n{output[-6000:]}")
        print(f"[{label}] completed in {time.monotonic() - start:.1f}s", flush=True)
        return output

    def download(self, name: str, source: dict) -> Path:
        path = self.work / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            self.run(name, ["curl", "--fail", "--location", "--output", temporary, source["url"]])
            if digest(temporary) != source["sha256"]:
                raise RuntimeError(f"Checksum mismatch: {name}")
            temporary.replace(path)
        if digest(path) != source["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {name}")
        return path

    def checkout(self, name: str, source: dict, *, patched: bool = False) -> Path:
        path = self.work / name
        if not path.exists():
            self.run(f"{name}-init", ["git", "init", "--quiet", path])
            self.run(f"{name}-remote", ["git", "-C", path, "remote", "add", "origin", source["repository"]])
            self.run(f"{name}-fetch", ["git", "-C", path, "fetch", "--depth=1", "origin", source["commit"]])
            self.run(f"{name}-checkout", ["git", "-C", path, "checkout", "--detach", "FETCH_HEAD"])
        if git_output(path, "rev-parse", "HEAD") != source["commit"]:
            raise RuntimeError(f"Unexpected revision in {path}; use a new work directory")
        if not patched and git_output(path, "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError(f"Modified dependency checkout: {path}")
        return path


def identity(args: argparse.Namespace) -> dict:
    if git_output(ROOT, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Commit source changes before a reproducible build/run")
    return {
        "vortex_revision": git_output(ROOT, "rev-parse", "HEAD"),
        "inputs": {name: digest(HERE / name) for name in RECIPE_FILES},
        "cargo_lock": digest(ROOT / "Cargo.lock"),
        "toolchain": str(args.toolchain),
        "cuda_root": str(args.cuda_root),
        "clangxx": str(args.clangxx),
        "libclang": str(args.libclang),
        "external_tools": {
            str(path): digest(path)
            for path in (args.cuda_root / "version.json", args.clangxx, args.libclang / "libclang.so")
        },
    }


def initialize_work(work: Path, recipe: dict):
    marker = work / "recipe.json"
    if marker.exists():
        if json.loads(marker.read_text()) != recipe:
            raise RuntimeError("Build inputs changed; select a new --work-dir")
    else:
        if work.exists() and any(work.iterdir()):
            raise RuntimeError("Refusing an existing nonempty work directory without recipe.json")
        work.mkdir(parents=True, exist_ok=True)
        save(marker, recipe)


def configure_command(args: argparse.Namespace, lock: dict, nvcomp: Path) -> list[str | Path]:
    work, prefix = args.work_dir, args.toolchain
    flags = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_C_COMPILER": prefix / "bin/aarch64-conda-linux-gnu-gcc",
        "CMAKE_CXX_COMPILER": prefix / "bin/aarch64-conda-linux-gnu-g++",
        "CMAKE_CUDA_HOST_COMPILER": prefix / "bin/aarch64-conda-linux-gnu-g++",
        "CMAKE_CUDA_COMPILER": args.cuda_root / "bin/nvcc",
        "CUDAToolkit_ROOT": args.cuda_root,
        "CMAKE_MAKE_PROGRAM": prefix / "bin/ninja",
        "CMAKE_PREFIX_PATH": prefix,
        "Python_EXECUTABLE": prefix / "bin/python",
        "Python3_EXECUTABLE": prefix / "bin/python",
        "CMAKE_CUDA_ARCHITECTURES": lock["cuda_architectures"],
        "CMAKE_C_FLAGS": "",
        "CMAKE_CXX_FLAGS": "",
        "CMAKE_CUDA_FLAGS": "",
        "CMAKE_C_FLAGS_RELEASE": "-O3 -DNDEBUG",
        "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG",
        "CMAKE_CUDA_FLAGS_RELEASE": "-O3 -DNDEBUG",
        "CMAKE_FIND_USE_PACKAGE_REGISTRY": "OFF",
        "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY": "OFF",
        "CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH": "OFF",
        "BUILD_TESTS": "OFF",
        "BUILD_BENCHMARKS": "ON",
        "BUILD_SHARED_LIBS": "ON",
        "CUDF_BUILD_TESTUTIL": "ON",
        "CUDF_BUILD_STREAMS_TEST_UTIL": "ON",
        "CUDF_BUILD_STATIC_DEPS": "ON",
        "CUDF_KVIKIO_REMOTE_IO": "ON",
        "CUDF_USE_PER_THREAD_DEFAULT_STREAM": "OFF",
        "CUDF_LTO_ARCHITECTURE": "75",
        "CUDA_ENABLE_LINEINFO": "OFF",
        "CUDA_WARNINGS_AS_ERRORS": "ON",
        "USE_NVTX": "ON",
        "RMM_NVTX": "OFF",
        "CUDF_NDSH_WITH_VORTEX": "ON",
        "FETCHCONTENT_SOURCE_DIR_VORTEX": ROOT,
        "FETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE": work / "rapids-cmake",
        "RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE": HERE / "build-lock.json",
        "CMAKE_PROJECT_CUDF_INCLUDE": HERE / "nvcc131-cudf-hook.cmake",
        "nvcomp_DIR": nvcomp / "lib/cmake/nvcomp",
        "CPM_DOWNLOAD_LOCATION": work / "CPM.cmake",
        "CPM_USE_LOCAL_PACKAGES": "OFF",
        "CPM_LOCAL_PACKAGES_ONLY": "OFF",
        "CPM_DOWNLOAD_ALL": "OFF",
        "CPM_dlpack_SOURCE": work / "dlpack",
        "CPM_xxhash_SOURCE": work / "xxhash",
        # Keep NVBench's hashed URL and PATCH_COMMAND together in its own declaration.
        "CPM_DOWNLOAD_nlohmann_json": "ON",
        # CURL comes from the Conda lock; a missing package must not trigger a source fallback.
        "CPM_DOWNLOAD_CURL": "OFF",
        "CMAKE_REQUIRE_FIND_PACKAGE_CURL": "ON",
        "CURL_NO_CURL_CMAKE": "ON",
        "CURL_INCLUDE_DIR": prefix / "include",
        "CURL_LIBRARY": prefix / "lib/libcurl.so",
    }
    flags.update({f"CPM_DOWNLOAD_{name}": "ON" for name in lock["packages"]})
    return [
        prefix / "bin/cmake",
        "--fresh",
        "-S",
        work / "cudf/cpp",
        "-B",
        work / "cudf-build",
        "-G",
        "Ninja",
        *(f"-D{name}={value}" for name, value in flags.items()),
    ]


def build(args: argparse.Namespace, lock: dict, runner: Runner, recipe: dict):
    # A failed rebuild must not leave the previous success marker usable by `run`.
    (runner.work / "build.json").unlink(missing_ok=True)
    for name in ("include/curl/curl.h", "include/curl/curlver.h", "lib/libcurl.so", "lib/pkgconfig/libcurl.pc"):
        if not (args.toolchain / name).is_file():
            raise RuntimeError(f"Missing locked toolchain file: {name}")
    nvcc = runner.run("nvcc-version", [args.cuda_root / "bin/nvcc", "--version"])
    if f"V{lock['nvcc_version']}" not in nvcc:
        raise RuntimeError(f"Expected NVCC {lock['nvcc_version']}")
    toolkit = json.loads((args.cuda_root / "version.json").read_text())
    if toolkit["cuda"]["version"] != lock["cuda_version"]:
        raise RuntimeError(f"Expected CUDA toolkit {lock['cuda_version']}")
    if toolkit["cuda_cudart"]["version"] != lock["cuda_runtime_version"]:
        raise RuntimeError(f"Expected CUDA runtime {lock['cuda_runtime_version']}")
    save(runner.logs / "cuda-version.json", toolkit)
    runner.run(
        "python-packages",
        [
            args.toolchain / "bin/python",
            "-c",
            "from importlib.metadata import distributions; "
            "assert not list(distributions()), 'Use a toolchain prefix without additional Python packages'",
        ],
    )
    clang = runner.run("clang-version", [args.clangxx, "--version"])
    if f"version {lock['clang_version']}" not in clang:
        raise RuntimeError(f"Expected Clang {lock['clang_version']}")
    runner.run("rust-version", ["rustc", "--version"], ROOT)
    runner.run("cargo-version", ["cargo", "--version"], ROOT)
    cudf = runner.checkout("cudf", lock["cudf"], patched=True)
    runner.checkout("rapids-cmake", lock["rapids_cmake"])
    flatc = runner.checkout("flatc", lock["flatc"])
    # DOWNLOAD_ONLY calls bypass FetchContent overrides; supply pinned local sources instead.
    for name in ("dlpack", "xxhash"):
        package = lock["packages"][name]
        runner.checkout(name, {"repository": package["git_url"], "commit": package["git_tag"]})
    runner.download("CPM.cmake", lock["cpm"])
    source_record = runner.work / "cudf-source.json"
    if not source_record.exists():
        if git_output(cudf, "status", "--porcelain"):
            raise RuntimeError("cuDF checkout is not pristine")
        runner.run("patch-check", ["git", "-C", cudf, "apply", "--check", HERE / "upstream.patch"])
        runner.run("patch", ["git", "-C", cudf, "apply", HERE / "upstream.patch"])
        runner.run("patch-new-files", ["git", "-C", cudf, "add", "-N", "--", "cpp/benchmarks"])
        save(
            source_record,
            {
                "diff": git_output(cudf, "diff", "--binary", "HEAD"),
                "status": git_output(cudf, "status", "--porcelain", "--untracked-files=all"),
            },
        )
    recorded = json.loads(source_record.read_text())
    if recorded != {
        "diff": git_output(cudf, "diff", "--binary", "HEAD"),
        "status": git_output(cudf, "status", "--porcelain", "--untracked-files=all"),
    }:
        raise RuntimeError("Prepared cuDF source changed; use a new work directory")
    cmake = args.toolchain / "bin/cmake"
    runner.run(
        "flatc-configure",
        [
            cmake,
            "-S",
            flatc,
            "-B",
            runner.work / "flatc-build",
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CXX_COMPILER={args.clangxx}",
            "-DFLATBUFFERS_BUILD_FLATC=ON",
            "-DFLATBUFFERS_BUILD_FLATLIB=OFF",
            "-DFLATBUFFERS_BUILD_TESTS=OFF",
            "-DFLATBUFFERS_INSTALL=OFF",
        ],
    )
    runner.run(
        "flatc-build",
        [cmake, "--build", runner.work / "flatc-build", "--target", "flatc", "--parallel", args.jobs],
    )
    if lock["flatc"]["version"] not in runner.run("flatc-version", [runner.env["FLATC"], "--version"]):
        raise RuntimeError("Unexpected Vortex flatc version")
    archive = runner.download("nvcomp.tar.xz", lock["nvcomp"])
    nvcomp = runner.work / lock["nvcomp"]["directory"]
    if not nvcomp.exists():
        with tarfile.open(archive) as source:
            source.extractall(runner.work, filter="data")
    # cuCollections otherwise fetches its bootstrap from RAPIDS-CMake's moving main branch.
    cuco_bootstrap = runner.work / "cudf-build/_deps/cuco-build/CUCO_RAPIDS.cmake"
    cuco_bootstrap.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(runner.work / "rapids-cmake/RAPIDS.cmake", cuco_bootstrap)
    runner.run("cudf-configure", configure_command(args, lock, nvcomp))
    runner.run(
        "cudf-build",
        [cmake, "--build", runner.work / "cudf-build", "--target", *TARGETS, "--parallel", args.jobs],
    )
    if identity(args) != recipe:
        raise RuntimeError("Vortex source changed during the build")
    save(
        runner.work / "build.json",
        {
            "recipe": recipe,
            "logs": str(runner.logs),
            "generator": "original",
            "binaries": {name: digest(runner.work / "cudf-build/benchmarks" / name) for name in TARGETS},
            "libcudf": digest(runner.work / "cudf-build/libcudf.so"),
        },
    )


def benchmark_command(binary: Path, query: int, args: argparse.Namespace, output: Path) -> list[str | Path]:
    command = [
        binary,
        "--benchmark",
        f"ndsh_q{query}_local",
        "--devices",
        "0",
        "--axis",
        f"scale_factor={args.scale_factor:.17g}",
        "--axis",
        "format=[parquet,vortex]",
        "--axis",
        f"workload=[read,q{query}]",
        "--axis",
        "cache=[warm,cold]",
        "--min-samples",
        str(args.min_samples),
        "--timeout",
        str(args.sample_timeout),
        "--json",
        output,
    ]
    if query == 9:
        command += ["--axis", "engine=[binaryop,ast,transform]"]
    return command


def validate_results(data: dict, query: int, scale_factor: float):
    if [bench["name"] for bench in data["benchmarks"]] != [f"ndsh_q{query}_local"]:
        raise RuntimeError(f"Expected only ndsh_q{query}_local results")
    states = data["benchmarks"][0]["states"]
    engines = ("binaryop", "ast", "transform") if query == 9 else (None,)
    expected = set(itertools.product(("parquet", "vortex"), ("read", f"q{query}"), ("warm", "cold"), engines))
    actual = []
    for state in states:
        axes = {axis["name"]: axis["value"] for axis in state["axis_values"]}
        actual.append((axes["format"], axes["workload"], axes["cache"], axes.get("engine")))
        means = [
            float(item["value"])
            for summary in state["summaries"]
            if summary["tag"] == "nv/cold/time/cpu/mean"
            for item in summary["data"]
            if item["name"] == "value"
        ]
        if (
            state.get("is_skipped")
            or state["device"] != 0
            or float(axes["scale_factor"]) != scale_factor
            or len(means) != 1
            or not math.isfinite(means[0])
            or means[0] <= 0
        ):
            raise RuntimeError(f"Skipped, mismatched, or untimed state: {state['name']}")
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError(f"Incomplete Q{query} read/query × format × cache matrix")


def benchmark(args: argparse.Namespace, runner: Runner, recipe: dict):
    record = json.loads((runner.work / "build.json").read_text())
    binaries = runner.work / "cudf-build/benchmarks"
    if (
        record["recipe"] != recipe
        or set(record["binaries"]) != set(TARGETS)
        or any(digest(binaries / name) != sha for name, sha in record["binaries"].items())
    ):
        raise RuntimeError("Build record or binaries changed; rebuild before benchmarking")
    if digest(runner.work / "cudf-build/libcudf.so") != record["libcudf"]:
        raise RuntimeError("libcudf changed since the recorded build")
    results = runner.work / "results" / timestamp()
    results.mkdir(parents=True)
    save(results / "build.json", record)
    save(
        results / "run.json",
        {
            "scale_factor": args.scale_factor,
            "queries": args.queries,
            "min_samples": args.min_samples,
            "sample_timeout": args.sample_timeout,
            "logs": str(runner.logs),
        },
    )
    runner.run("gpu", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,utilization.gpu", "--format=csv"])
    for name in TARGETS[:2]:
        runner.run(name, [binaries / name], results)
    for query in args.queries:
        output = results / f"sf{args.scale_factor:g}-q{query}.json"
        runner.run(
            f"sf{args.scale_factor:g}-q{query}",
            benchmark_command(binaries / f"NDSH_Q{query:02}_NVBENCH", query, args, output),
            results,
        )
        validate_results(json.loads(output.read_text()), query, args.scale_factor)
        print(f"Results: {output}", flush=True)
    print(f"Completed SF{args.scale_factor:g} matrix: {results}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "run"))
    parser.add_argument("--work-dir", type=Path, default=ROOT / "build/cudf-ndsh-repro")
    parser.add_argument("--toolchain", type=Path, required=True, help="Prefix created from environment-linux-aarch64.lock")
    parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda-13.1"))
    parser.add_argument("--clangxx", type=Path, default=Path("/usr/bin/clang++"))
    parser.add_argument("--libclang", type=Path, default=Path("/usr/lib/llvm-18/lib"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cargo-jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1200, help="Maximum seconds per command; no automatic retries")
    parser.add_argument("--scale-factor", type=float, default=1)
    parser.add_argument("--queries", type=int, nargs="+", choices=QUERIES, default=list(QUERIES))
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--sample-timeout", type=int, default=30, help="NVBench timeout per state")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        parser.error("This lock targets Linux AArch64/SBSA")
    if (
        not math.isfinite(args.scale_factor)
        or args.scale_factor <= 0
        or min(args.jobs, args.cargo_jobs, args.timeout, args.min_samples, args.sample_timeout) <= 0
    ):
        parser.error("Scale factor and command limits must be positive and finite")
    for name in ("work_dir", "toolchain", "cuda_root", "clangxx", "libclang"):
        # Preserve compiler symlink names: clang++ and clang select different driver modes.
        setattr(args, name, Path(os.path.abspath(getattr(args, name))))
    locked_environment(args.toolchain)
    recipe = identity(args)
    initialize_work(args.work_dir, recipe)
    runner = Runner(args.work_dir, environment(args), args.timeout)
    lock = json.loads((HERE / "build-lock.json").read_text())
    if args.action == "build":
        build(args, lock, runner, recipe)
    else:
        benchmark(args, runner, recipe)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error
