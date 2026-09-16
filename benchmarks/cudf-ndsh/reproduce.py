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
import re
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
QUERIES = (1, 5, 6, 9, 10)
TARGETS = ("NDSH_VORTEX_BUILD_SMOKE", "NDSH_VORTEX_IO_TEST", *(f"NDSH_Q{q:02}_NVBENCH" for q in QUERIES))
COMPILERS = ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER", "CMAKE_CUDA_HOST_COMPILER")
BUILD_ENVIRONMENT = (
    "PATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "CC",
    "CXX",
    "CUDACXX",
    "CUDAHOSTCXX",
    "NVCC_CCBIN",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
    "CUDAFLAGS",
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "CMAKE_PREFIX_PATH",
    "CMAKE_TOOLCHAIN_FILE",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDAToolkit_ROOT",
    "CUDA_PATH",
    "CUDA_HOME",
    "LIBCLANG_PATH",
    "BINDGEN_EXTRA_CLANG_ARGS",
    "PKG_CONFIG_PATH",
    "PKG_CONFIG_LIBDIR",
    "PKG_CONFIG_SYSROOT_DIR",
    "CONDA_PREFIX",
    "VIRTUAL_ENV",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "RUSTUP_TOOLCHAIN",
    "RUSTC_WRAPPER",
    "RUSTFLAGS",
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


def environment(args: argparse.Namespace) -> dict[str, str]:
    # Preserve the caller's build setup without collecting unrelated credentials.
    return {
        "PATH": os.defpath,
        **{name: os.environ[name] for name in BUILD_ENVIRONMENT if name in os.environ},
        "HOME": str(Path.home()),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_EDITOR": "true",
        "GIT_TERMINAL_PROMPT": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(args.work_dir / "tmp"),
        "CARGO_BUILD_JOBS": str(args.cargo_jobs),
        "FLATC": str(args.work_dir / "flatc-build/flatc"),
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
        """Return combined output on success, retaining logs on failure.

        Timeout or interruption kills the entire subprocess group, not just its leader.
        """
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
        """Verify fresh and cached downloads against SHA-256; reject corrupt cached files."""
        path = self.work / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            self.run(name, ["curl", "--fail", "--location", "--output", temporary, source["url"]])
            if digest(temporary) != source["sha256"]:
                raise RuntimeError(f"Checksum mismatch: {name}")
            temporary.replace(path)
        elif digest(path) != source["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {name}")
        return path

    def checkout(self, name: str, source: dict, *, patched: bool = False) -> Path:
        """Require the pinned HEAD, leaving existing checkouts untouched.

        ``patched=True`` permits local changes; the caller must validate their contents.
        """
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


def identity() -> dict:
    if git_output(ROOT, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Commit source changes before a reproducible build/run")
    return {"vortex_revision": git_output(ROOT, "rev-parse", "HEAD")}


def source_state(path: Path) -> dict[str, str]:
    """Capture tracked changes against HEAD and status; untracked contents are not recorded."""
    return {
        "diff": git_output(path, "diff", "--binary", "HEAD"),
        "status": git_output(path, "status", "--porcelain", "--untracked-files=all"),
    }


def initialize_work(work: Path, recipe: dict):
    """Claim an empty directory, or reuse one only if its recorded recipe matches exactly."""
    marker = work / "recipe.json"
    if marker.exists():
        if json.loads(marker.read_text()) != recipe:
            raise RuntimeError("Build inputs changed; select a new --work-dir")
    else:
        if work.exists() and any(work.iterdir()):
            raise RuntimeError("Refusing an existing nonempty work directory without recipe.json")
        work.mkdir(parents=True, exist_ok=True)
        save(marker, recipe)


def configure_command(args: argparse.Namespace, lock: dict) -> list[str | Path]:
    """Keep caller toolchain settings, but give recipe-owned definitions final precedence."""
    work = args.work_dir
    flags = {
        "CMAKE_BUILD_TYPE": "Release",
        "BUILD_TESTS": "OFF",
        "BUILD_BENCHMARKS": "ON",
        "BUILD_SHARED_LIBS": "ON",
        "CUDF_WITH_VORTEX": "ON",
        "FETCHCONTENT_SOURCE_DIR_VORTEX": ROOT,
        "FETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE": work / "rapids-cmake",
        "RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE": HERE / "build-lock.json",
        "CPM_DOWNLOAD_LOCATION": work / "CPM.cmake",
        "CPM_dlpack_SOURCE": work / "dlpack",
        "CPM_xxhash_SOURCE": work / "xxhash",
        # Keep NVBench's hashed URL and PATCH_COMMAND together in its own declaration.
        "CPM_DOWNLOAD_nlohmann_json": "ON",
    }
    flags.update({f"CPM_DOWNLOAD_{name}": "ON" for name in lock["packages"]})
    return [
        "cmake",
        "--fresh",
        "-S",
        work / "cudf/cpp",
        "-B",
        work / "cudf-build",
        "-G",
        "Ninja",
        *args.cmake_arg,
        *(f"-D{name}={value}" for name, value in flags.items()),
    ]


def compiler_arguments(arguments: list[str]) -> list[str]:
    # flatc is a native build tool; use the caller's host compiler selection too.
    names = {
        "CMAKE_C_COMPILER",
        "CMAKE_CXX_COMPILER",
        "CMAKE_C_COMPILER_ARG1",
        "CMAKE_CXX_COMPILER_ARG1",
        "CMAKE_TOOLCHAIN_FILE",
        "CMAKE_SYSROOT",
    }
    return [arg for arg in arguments if arg[2:].split("=", 1)[0].split(":", 1)[0] in names]


def record_toolchain(runner: Runner) -> dict:
    """Record configured compiler identities before enforcing the CUDA >= 12.8 requirement."""
    cache = {}
    for line in (runner.work / "cudf-build/CMakeCache.txt").read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    names = (
        *COMPILERS,
        *(f"{name}_ARG1" for name in COMPILERS),
        "CMAKE_CUDA_ARCHITECTURES",
        "CMAKE_TOOLCHAIN_FILE",
        "CMAKE_SYSROOT",
        "CUDAToolkit_BIN_DIR",
        "CMAKE_C_FLAGS",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CUDA_FLAGS",
        "CMAKE_EXE_LINKER_FLAGS",
        "CMAKE_C_FLAGS_RELEASE",
        "CMAKE_CXX_FLAGS_RELEASE",
        "CMAKE_CUDA_FLAGS_RELEASE",
        "nvcomp_DIR",
    )
    selected = {name: cache[name] for name in names if cache.get(name)}
    tools = {}
    for name in COMPILERS:
        if selected.get(name):
            compiler = Path(selected[name])
            arguments = shlex.split(cache.get(f"{name}_ARG1", ""))
            tools[name] = {
                "version": runner.run(name.lower(), [compiler, *arguments, "--version"]),
                "sha256": digest(compiler),
            }
    result = {"cache": selected, "tools": tools}
    save(runner.logs / "toolchain.json", result)
    version = re.search(r"release\s+(\d+)\.(\d+)", tools["CMAKE_CUDA_COMPILER"]["version"])
    if not version or tuple(map(int, version.groups())) < (12, 8):
        raise RuntimeError("This benchmark recipe requires an NVIDIA CUDA toolkit >= 12.8")
    return result


def build(args: argparse.Namespace, lock: dict, runner: Runner, recipe: dict):
    # A failed rebuild must not leave the previous success marker usable by `run`.
    (runner.work / "build.json").unlink(missing_ok=True)
    runner.run("cmake-version", ["cmake", "--version"])
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
    if source_record.exists():
        if json.loads(source_record.read_text()) != source_state(cudf):
            raise RuntimeError("Prepared cuDF source changed; use a new work directory")
    else:
        if git_output(cudf, "status", "--porcelain"):
            raise RuntimeError("cuDF checkout is not pristine")
        # Index new files too, so source_state() records their contents rather than only their names.
        runner.run("patch", ["git", "-C", cudf, "apply", "--index", HERE / "upstream.patch"])
        save(source_record, source_state(cudf))
    runner.run(
        "flatc-configure",
        [
            "cmake",
            "-S",
            flatc,
            "-B",
            runner.work / "flatc-build",
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            *compiler_arguments(args.cmake_arg),
            "-DFLATBUFFERS_BUILD_FLATC=ON",
            "-DFLATBUFFERS_BUILD_FLATLIB=OFF",
            "-DFLATBUFFERS_BUILD_TESTS=OFF",
            "-DFLATBUFFERS_INSTALL=OFF",
        ],
    )
    runner.run(
        "flatc-build",
        ["cmake", "--build", runner.work / "flatc-build", "--target", "flatc", "--parallel", args.jobs],
    )
    if lock["flatc"]["version"] not in runner.run("flatc-version", [runner.env["FLATC"], "--version"]):
        raise RuntimeError("Unexpected Vortex flatc version")

    # cuCollections otherwise fetches its bootstrap from RAPIDS-CMake's moving main branch.
    cuco_bootstrap = runner.work / "cudf-build/_deps/cuco-build/CUCO_RAPIDS.cmake"
    cuco_bootstrap.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(runner.work / "rapids-cmake/RAPIDS.cmake", cuco_bootstrap)
    runner.run("cudf-configure", configure_command(args, lock))
    toolchain = record_toolchain(runner)
    runner.run(
        "cudf-build",
        ["cmake", "--build", runner.work / "cudf-build", "--target", *TARGETS, "--parallel", args.jobs],
    )
    if identity() != recipe:
        raise RuntimeError("Vortex source changed during the build")
    save(
        runner.work / "build.json",
        {
            "recipe": recipe,
            "logs": str(runner.logs),
            "generator": "original",
            "environment": runner.env,
            "cmake_args": args.cmake_arg,
            "toolchain": toolchain,
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
    """Require each read/query × format × cache state exactly once, including Q9 engines.

    Every state must run on device 0 at the requested scale and have one finite,
    positive NVBench cold CPU mean; skipped states are rejected.
    """
    if [bench["name"] for bench in data["benchmarks"]] != [f"ndsh_q{query}_local"]:
        raise RuntimeError(f"Expected only ndsh_q{query}_local results")
    states = data["benchmarks"][0]["states"]
    engines = ("binaryop", "ast", "transform") if query == 9 else (None,)
    remaining = set(itertools.product(("parquet", "vortex"), ("read", f"q{query}"), ("warm", "cold"), engines))
    for state in states:
        axes = {axis["name"]: axis["value"] for axis in state["axis_values"]}
        key = (axes["format"], axes["workload"], axes["cache"], axes.get("engine"))
        if key not in remaining:
            raise RuntimeError(f"Duplicate or unexpected Q{query} state: {state['name']}")
        remaining.remove(key)
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
    if remaining:
        raise RuntimeError(f"Incomplete Q{query} read/query × format × cache matrix")


def benchmark(args: argparse.Namespace, recipe: dict):
    """Verify recorded build hashes before GPU calls, then reuse the build's environment."""
    work = args.work_dir
    record = json.loads((work / "build.json").read_text())
    binaries = work / "cudf-build/benchmarks"
    if (
        record["recipe"] != recipe
        or set(record["binaries"]) != set(TARGETS)
        or any(digest(binaries / name) != sha for name, sha in record["binaries"].items())
    ):
        raise RuntimeError("Build record or binaries changed; rebuild before benchmarking")
    if digest(work / "cudf-build/libcudf.so") != record["libcudf"]:
        raise RuntimeError("libcudf changed since the recorded build")
    env = record["environment"].copy()
    # ELF RUNPATH follows LD_LIBRARY_PATH; the library we verified must take precedence.
    paths = [str(work / "cudf-build")]
    if env.get("LD_LIBRARY_PATH"):
        paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
    runner = Runner(work, env, args.timeout)
    results = work / "results" / timestamp()
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
        runner.run(output.stem, benchmark_command(binaries / f"NDSH_Q{query:02}_NVBENCH", query, args, output), results)
        validate_results(json.loads(output.read_text()), query, args.scale_factor)
        print(f"Results: {output}", flush=True)
    print(f"Completed SF{args.scale_factor:g} matrix: {results}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "run"))
    parser.add_argument("--work-dir", type=Path, default=ROOT / "build/cudf-ndsh-repro")
    parser.add_argument(
        "--cmake-arg",
        action="append",
        default=[],
        help="Build-only CMake definition, e.g. --cmake-arg=-DCMAKE_CUDA_ARCHITECTURES=90 (repeatable)",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cargo-jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1200, help="Maximum seconds per command; no automatic retries")
    parser.add_argument("--scale-factor", type=float, default=1)
    parser.add_argument("--queries", type=int, nargs="+", choices=QUERIES, default=list(QUERIES))
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--sample-timeout", type=int, default=30, help="NVBench timeout per state")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("cuDF/Vortex GPU benchmarks require Linux")
    if any(not re.fullmatch(r"-D[A-Za-z_][A-Za-z_0-9-]*(?::[A-Za-z]+)?=.*", arg) for arg in args.cmake_arg):
        parser.error("Pass each CMake definition as --cmake-arg=-DNAME=VALUE")
    if args.action == "run" and args.cmake_arg:
        parser.error("run uses the recorded build; --cmake-arg is build-only")
    if (
        not math.isfinite(args.scale_factor)
        or args.scale_factor <= 0
        or min(args.jobs, args.cargo_jobs, args.timeout, args.min_samples, args.sample_timeout) <= 0
    ):
        parser.error("Scale factor and command limits must be positive and finite")
    args.work_dir = args.work_dir.resolve()
    recipe = identity()
    if args.action == "build":
        env = environment(args)
        initialize_work(args.work_dir, {"source": recipe, "cmake_args": args.cmake_arg, "environment": env})
        runner = Runner(args.work_dir, env, args.timeout)
        lock = json.loads((HERE / "build-lock.json").read_text())
        build(args, lock, runner, recipe)
    else:
        benchmark(args, recipe)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error
