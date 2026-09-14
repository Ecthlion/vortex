# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Offline stdlib tests for the reproduction runner; no toolchain or GPU required."""

import hashlib
import importlib.util
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("ndsh_reproduce", HERE / "reproduce.py")
assert SPEC is not None and SPEC.loader is not None
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)
LOCK = json.loads((HERE / "build-lock.json").read_text(encoding="utf-8"))
QUERIES = (1, 5, 6, 9, 10)


def result_data(query: int) -> dict:
    states = []
    engines = ("binaryop", "ast", "transform") if query == 9 else (None,)
    for fmt, workload, cache, engine in itertools.product(
        ("parquet", "vortex"),
        ("read", f"q{query}"),
        ("warm", "cold"),
        engines,
    ):
        axes = dict(scale_factor="1", format=fmt, workload=workload, cache=cache)
        if engine is not None:
            axes["engine"] = engine
        states.append(
            {
                "name": str(axes),
                "device": 0,
                "axis_values": [{"name": k, "value": v} for k, v in axes.items()],
                "summaries": [
                    {
                        "tag": "nv/cold/time/cpu/mean",
                        "data": [{"name": "value", "value": 0.25}],
                    }
                ],
            }
        )
    return {"benchmarks": [{"name": f"ndsh_q{query}_local", "states": states}]}


class ReproduceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ndsh-reproduce-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = SimpleNamespace(
            work_dir=self.root / "work",
            cmake_arg=[],
            cargo_jobs=4,
            jobs=2,
            scale_factor=1.0,
            queries=list(QUERIES),
            min_samples=3,
            sample_timeout=30,
            timeout=60,
        )
        self.recipe = {"vortex_revision": "original"}
        self.runner = Mock(work=self.args.work_dir, logs=self.root)

    def write(self, path: Path, text: str = "fixture"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_sf1_commands_and_complete_matrices(self):
        for query in QUERIES:
            with self.subTest(query=query):
                binary = self.root / f"q{query}"
                output = self.root / "result.json"
                command = list(map(str, reproduce.benchmark_command(binary, query, self.args, output)))
                axes = {command[i + 1] for i, value in enumerate(command) if value == "--axis"}
                expected = {
                    "scale_factor=1",
                    "format=[parquet,vortex]",
                    "cache=[warm,cold]",
                    f"workload=[read,q{query}]",
                }
                if query == 9:
                    expected.add("engine=[binaryop,ast,transform]")
                self.assertEqual(axes, expected)
                self.assertEqual(command[:3], [str(binary), "--benchmark", f"ndsh_q{query}_local"])
                for flag, value in (
                    ("--devices", "0"),
                    ("--json", str(output)),
                    ("--min-samples", "3"),
                    ("--timeout", "30"),
                ):
                    self.assertEqual(command[command.index(flag) + 1], value)
                data = result_data(query)
                reproduce.validate_results(data, query, 1.0)
                data["benchmarks"][0]["states"].reverse()
                reproduce.validate_results(data, query, 1.0)

    def test_rejects_incomplete_skipped_and_untimed_matrices(self):
        for query, defect in itertools.product(
            QUERIES,
            (
                "missing",
                "duplicate",
                "extra",
                "format",
                "workload",
                "cache",
                "engine",
                "skipped",
                "untimed",
                "wrong_timer",
                "wrong_scale",
                "wrong_name",
                "wrong_device",
                0,
                -1,
                "nan",
                "inf",
            ),
        ):
            with self.subTest(query=query, defect=defect):
                data = result_data(query)
                states = data["benchmarks"][0]["states"]
                state = states[0]
                if defect == "missing":
                    states.pop()
                elif defect == "duplicate":
                    states[-1] = state
                elif defect == "extra":
                    states.append(state)
                elif defect in ("format", "workload", "cache", "engine"):
                    state["axis_values"] = [axis for axis in state["axis_values"] if axis["name"] != defect]
                    state["axis_values"].append({"name": defect, "value": "unexpected"})
                elif defect == "skipped":
                    state["is_skipped"] = True
                elif defect == "untimed":
                    state["summaries"] = []
                elif defect == "wrong_timer":
                    state["summaries"][0]["tag"] = "nv/cold/time/gpu/mean"
                elif defect == "wrong_scale":
                    scale_axis = next(axis for axis in state["axis_values"] if axis["name"] == "scale_factor")
                    scale_axis["value"] = "10"
                elif defect == "wrong_name":
                    data["benchmarks"][0]["name"] = f"ndsh_q{query}_other"
                elif defect == "wrong_device":
                    state["device"] = 1
                else:
                    state["summaries"][0]["data"][0]["value"] = defect
                with self.assertRaises(RuntimeError):
                    reproduce.validate_results(data, query, 1.0)

    def write_build_record(self, env: dict[str, str] | None = None):
        build = self.args.work_dir / "cudf-build"
        sha = hashlib.sha256(b"fixture").hexdigest()
        for name in reproduce.TARGETS:
            self.write(build / "benchmarks" / name)
        self.write(build / "libcudf.so")
        self.write(
            self.args.work_dir / "build.json",
            json.dumps(
                {
                    "recipe": self.recipe,
                    "binaries": dict.fromkeys(reproduce.TARGETS, sha),
                    "libcudf": sha,
                    "environment": env or {},
                }
            ),
        )

    def test_stale_binary_and_library_hashes_fail_before_gpu_calls(self):
        self.write_build_record()
        build = self.args.work_dir / "cudf-build"
        for path in (build / "benchmarks/NDSH_Q09_NVBENCH", build / "libcudf.so"):
            with self.subTest(path=path), patch.object(reproduce, "Runner") as runner:
                self.write(path, "stale")
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    reproduce.benchmark(self.args, self.recipe)
                runner.assert_not_called()
                self.assertFalse((self.args.work_dir / "results").exists())
                self.write(path)

    def test_identity_requires_a_clean_revision(self):
        with patch.object(reproduce, "git_output", side_effect=["", "revision"]) as git:
            self.assertEqual(reproduce.identity(), {"vortex_revision": "revision"})
        self.assertEqual(
            git.call_args_list,
            [
                call(reproduce.ROOT, "status", "--porcelain", "--untracked-files=all"),
                call(reproduce.ROOT, "rev-parse", "HEAD"),
            ],
        )
        for status in (" M tracked", "?? untracked"):
            with self.subTest(status=status), patch.object(reproduce, "git_output", return_value=status):
                with self.assertRaisesRegex(RuntimeError, "Commit source changes"):
                    reproduce.identity()

    def test_work_directory_ownership_and_recipe(self):
        work = self.args.work_dir
        self.write(work / "unowned", "keep me")
        with self.assertRaisesRegex(RuntimeError, "nonempty"):
            reproduce.initialize_work(work, self.recipe)
        self.assertEqual((work / "unowned").read_text(), "keep me")
        self.assertFalse((work / "recipe.json").exists())
        owned = self.root / "owned"
        reproduce.initialize_work(owned, self.recipe)
        reproduce.initialize_work(owned, self.recipe)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            reproduce.initialize_work(owned, {"vortex_revision": "changed"})
        self.assertEqual(json.loads((owned / "recipe.json").read_text()), self.recipe)

    def test_release_configuration_preserves_source_pins_without_toolchain_defaults(self):
        work = self.args.work_dir
        command = list(map(str, reproduce.configure_command(self.args, LOCK)))
        flags = dict(value[2:].split("=", 1) for value in command if value.startswith("-D"))
        self.assertEqual(command[:2], ["cmake", "--fresh"])
        expected = {
            "CMAKE_BUILD_TYPE": "Release",
            "BUILD_TESTS": "OFF",
            "BUILD_BENCHMARKS": "ON",
            "BUILD_SHARED_LIBS": "ON",
            "CUDF_NDSH_WITH_VORTEX": "ON",
            "FETCHCONTENT_SOURCE_DIR_VORTEX": reproduce.ROOT,
            "FETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE": work / "rapids-cmake",
            "RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE": HERE / "build-lock.json",
            "CPM_DOWNLOAD_LOCATION": work / "CPM.cmake",
            "CPM_dlpack_SOURCE": work / "dlpack",
            "CPM_xxhash_SOURCE": work / "xxhash",
            "CPM_DOWNLOAD_nlohmann_json": "ON",
        }
        expected.update({f"CPM_DOWNLOAD_{name}": "ON" for name in LOCK["packages"]})
        # Exact flags exclude invented compiler settings, project hooks, and nvcomp overrides.
        self.assertEqual(flags, {key: str(value) for key, value in expected.items()})
        for package in LOCK["packages"].values():
            self.assertRegex(
                package.get("git_tag", package.get("url_hash", "")),
                r"^(?:[0-9a-f]{40}|SHA256=[0-9a-f]{64})$",
            )
        self.assertNotIn("nlohmann_json", LOCK["packages"])

    def test_configure_forwards_caller_cuda_compilers(self):
        self.args.cmake_arg = [
            f"-DCMAKE_CUDA_COMPILER:FILEPATH={self.root / 'cuda/bin/nvcc'}",
            f"-DCMAKE_CUDA_HOST_COMPILER={self.root / 'host/bin/g++'}",
            "-DCMAKE_CUDA_ARCHITECTURES=90",
        ]
        command = list(map(str, reproduce.configure_command(self.args, LOCK)))
        flags = dict(value[2:].split("=", 1) for value in command if value.startswith("-D"))
        self.assertEqual(flags["CMAKE_CUDA_COMPILER:FILEPATH"], str(self.root / "cuda/bin/nvcc"))
        self.assertEqual(flags["CMAKE_CUDA_HOST_COMPILER"], str(self.root / "host/bin/g++"))
        self.assertEqual(flags["CMAKE_CUDA_ARCHITECTURES"], "90")
        for argument in self.args.cmake_arg:
            self.assertEqual(command.count(argument), 1)

    def test_flatc_compiler_arguments_include_only_host_compilers_and_toolchain(self):
        for suffix in ("", ":FILEPATH"):
            with self.subTest(suffix=suffix):
                host = [
                    f"-D{name}{suffix}={self.root / name.lower()}"
                    for name in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_TOOLCHAIN_FILE", "CMAKE_SYSROOT")
                ]
                host += [
                    "-DCMAKE_C_COMPILER_ARG1=--driver-mode=gcc",
                    "-DCMAKE_CXX_COMPILER_ARG1:STRING=--driver-mode=g++",
                ]
                arguments = [
                    *host,
                    f"-DCMAKE_CUDA_COMPILER{suffix}={self.root / 'nvcc'}",
                    f"-DCMAKE_CUDA_HOST_COMPILER{suffix}={self.root / 'cuda-host'}",
                    "-DCMAKE_CUDA_COMPILER_ARG1=--allow-unsupported-compiler",
                    "-DCMAKE_CUDA_HOST_COMPILER_ARG1:STRING=--driver-mode=g++",
                    f"-DCUDAToolkit_ROOT={self.root / 'cuda'}",
                    "-DCMAKE_CUDA_ARCHITECTURES=90",
                    "-DCMAKE_CXX_FLAGS=-O2",
                    "-DCMAKE_BUILD_TYPE=Release",
                    f"-Dnvcomp_DIR={self.root / 'nvcomp'}",
                ]
                self.assertEqual(reproduce.compiler_arguments(arguments), host)

    def test_failed_build_clears_record_and_checks_out_download_only_package_pins(self):
        record = self.args.work_dir / "build.json"
        self.write(record, "stale success")
        versions = {
            "cmake-version": "cmake version 4.1.2",
            "rust-version": "rustc 1.90.0",
            "cargo-version": "cargo 1.90.0",
        }

        def run(label: str, *_args: object) -> str:
            self.assertFalse(record.exists())
            return versions[label]

        self.runner.run.side_effect = run
        # Stop before archives or configuration; all external operations are mocked.
        self.runner.download.side_effect = RuntimeError("stop after checkouts")
        with self.assertRaisesRegex(RuntimeError, "stop after checkouts"):
            reproduce.build(self.args, LOCK, self.runner, self.recipe)
        self.assertFalse(record.exists())
        self.runner.download.assert_called_once_with("CPM.cmake", LOCK["cpm"])
        for name in ("dlpack", "xxhash"):
            package = LOCK["packages"][name]
            self.runner.checkout.assert_any_call(
                name,
                {
                    "repository": package["git_url"],
                    "commit": package["git_tag"],
                },
            )

    def test_environment_preserves_caller_build_settings_without_secrets_or_shared_cache(self):
        preserved = {
            name: f"caller-{name}"
            for name in (
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
        }
        excluded = dict.fromkeys(
            (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN",
                "GITHUB_TOKEN",
                "CPM_SOURCE_CACHE",
                "CCACHE_DIR",
                "SCCACHE_DIR",
                "PYTHONPATH",
                "PYTHONHOME",
            ),
            "ambient",
        )
        ambient = {
            **preserved,
            **excluded,
            "HOME": str(self.root),
            "FLATC": "ambient-flatc",
            "TMPDIR": "ambient-tmp",
            "CARGO_BUILD_JOBS": "999",
            "PYTHONNOUSERSITE": "0",
        }
        with patch.dict(os.environ, ambient, clear=True):
            env = reproduce.environment(self.args)
        for name, value in preserved.items():
            self.assertEqual(env[name], value, name)
        for name in excluded:
            self.assertNotIn(name, env)
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["FLATC"], str(self.args.work_dir / "flatc-build/flatc"))
        self.assertEqual(env["TMPDIR"], str(self.args.work_dir / "tmp"))
        self.assertEqual(env["CARGO_BUILD_JOBS"], "4")

    def test_record_toolchain_uses_cmake_cache_and_requires_cuda_12_8(self):
        compilers = {
            name: self.root / name.lower()
            for name in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER", "CMAKE_CUDA_HOST_COMPILER")
        }
        for name, path in compilers.items():
            self.write(path, name)
        cache = {
            **{name: str(path) for name, path in compilers.items()},
            "CMAKE_CUDA_ARCHITECTURES": "90",
            "CMAKE_TOOLCHAIN_FILE": str(self.root / "toolchain.cmake"),
            "CMAKE_SYSROOT": str(self.root / "sysroot"),
            "CUDAToolkit_BIN_DIR": str(self.root / "cuda/bin"),
            "CMAKE_CXX_FLAGS": "-O2 -DVALUE=a=b",
            "CMAKE_CUDA_FLAGS_RELEASE": "-O3 -DNDEBUG",
            "nvcomp_DIR": str(self.root / "native-nvcomp/lib/cmake/nvcomp"),
        }
        self.write(
            self.args.work_dir / "cudf-build/CMakeCache.txt",
            "\n".join(
                [
                    "# CMake cache fixture",
                    "//Compiler settings",
                    "",
                    *(f"{name}:FILEPATH={path}" for name, path in compilers.items()),
                    *(f"{name}:STRING={value}" for name, value in cache.items() if name not in compilers),
                    "CMAKE_C_FLAGS:STRING=",
                    "UNRELATED:STRING=ignored",
                ]
            ),
        )
        for version, accepted in (
            ("12.8", True),
            ("12.9", True),
            ("13.0", True),
            ("13.1", True),
            ("13.9", True),
            ("12.7", False),
            ("12.0", False),
            ("11.9", False),
            ("unknown", False),
        ):
            with self.subTest(version=version):
                versions = {name.lower(): f"{name} fixture version\n" for name in compilers}
                versions["cmake_cuda_compiler"] = f"Cuda compilation tools, release {version}, V{version}.0\n"
                self.runner.run.reset_mock()
                self.runner.run.side_effect = lambda label, _argv, versions=versions: versions[label]
                expected = {
                    "cache": cache,
                    "tools": {
                        name: {"version": versions[name.lower()], "sha256": hashlib.sha256(name.encode()).hexdigest()}
                        for name in compilers
                    },
                }
                if accepted:
                    self.assertEqual(reproduce.record_toolchain(self.runner), expected)
                else:
                    with self.assertRaisesRegex(RuntimeError, r">= 12\.8"):
                        reproduce.record_toolchain(self.runner)
                self.assertEqual(json.loads((self.runner.logs / "toolchain.json").read_text()), expected)
                self.assertEqual(
                    self.runner.run.call_args_list,
                    [call(name.lower(), [path, "--version"]) for name, path in compilers.items()],
                )

    def test_record_toolchain_passes_mandatory_compiler_arguments_to_version(self):
        wrapper = self.root / "compiler-wrapper"
        self.write(wrapper)
        arguments = {
            name: f'"compiler tools/{name.lower()}" --config \'config with spaces\''
            for name in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER", "CMAKE_CUDA_HOST_COMPILER")
        }
        self.write(
            self.args.work_dir / "cudf-build/CMakeCache.txt",
            "\n".join(
                line
                for name, arg1 in arguments.items()
                for line in (f"{name}:FILEPATH={wrapper}", f"{name}_ARG1:STRING={arg1}")
            ),
        )
        self.runner.run.side_effect = lambda label, _argv: (
            "Cuda compilation tools, release 12.8\n" if label == "cmake_cuda_compiler" else "compiler fixture version\n"
        )
        record = reproduce.record_toolchain(self.runner)
        for name, arg1 in arguments.items():
            self.assertEqual(record["cache"][f"{name}_ARG1"], arg1)
        self.assertEqual(
            self.runner.run.call_args_list,
            [
                call(
                    name.lower(),
                    [wrapper, f"compiler tools/{name.lower()}", "--config", "config with spaces", "--version"],
                )
                for name in arguments
            ],
        )

    def test_benchmark_reuses_build_environment(self):
        work = self.args.work_dir
        library_dir = str(work / "cudf-build")
        self.args.queries = []
        for library_path, runtime_path in (
            (None, library_dir),
            ("recorded-lib", os.pathsep.join((library_dir, "recorded-lib"))),
        ):
            with self.subTest(library_path=library_path):
                env = {"PATH": "recorded-bin"}
                if library_path is not None:
                    env["LD_LIBRARY_PATH"] = library_path
                self.write_build_record(env)
                source = (work / "build.json").read_text()
                result = str(library_path)
                with (
                    patch.object(reproduce, "timestamp", return_value=result),
                    patch.object(reproduce, "environment", autospec=True) as environment,
                    patch.object(reproduce, "Runner", return_value=self.runner) as runner,
                ):
                    reproduce.benchmark(self.args, self.recipe)
                environment.assert_not_called()
                runner.assert_called_once_with(work, {**env, "LD_LIBRARY_PATH": runtime_path}, 60)
                self.assertEqual((work / "build.json").read_text(), source)
                self.assertEqual(
                    json.loads((work / "results" / result / "build.json").read_text()), json.loads(source)
                )

    def test_main_run_dispatches_with_source_identity(self):
        work = self.args.work_dir.resolve()
        with (
            patch("sys.argv", ["reproduce.py", "run", "--work-dir", str(work), "--timeout", "60"]),
            patch.object(reproduce.platform, "system", return_value="Linux"),
            patch.object(reproduce, "identity", autospec=True, return_value=self.recipe) as identity,
            patch.object(reproduce, "benchmark", autospec=True) as benchmark,
        ):
            reproduce.main()
        identity.assert_called_once_with()
        run_args = benchmark.call_args.args[0]
        self.assertEqual(run_args.work_dir, work)
        self.assertEqual(run_args.cmake_arg, [])
        self.assertEqual(run_args.timeout, 60)
        benchmark.assert_called_once_with(run_args, self.recipe)


if __name__ == "__main__":
    unittest.main()
