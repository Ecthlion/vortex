# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Offline stdlib tests for the reproduction runner; no toolchain or GPU required."""

import hashlib
import itertools
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import reproduce

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "build-lock.json").read_text(encoding="utf-8"))
QUERIES = (1, 5, 6, 9, 10)
COMPILERS = ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "CMAKE_CUDA_COMPILER", "CMAKE_CUDA_HOST_COMPILER")


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
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="ndsh-reproduce-")))
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
                data["benchmarks"][0]["states"].reverse()
                reproduce.validate_results(data, query, 1.0)

    def test_rejects_incomplete_skipped_and_untimed_matrices(self):
        for query in (1, 9):
            data = result_data(query)
            benchmark = data["benchmarks"][0]
            states = benchmark["states"]
            state = states[0]
            summary = state["summaries"][0]
            scale_axis = next(axis for axis in state["axis_values"] if axis["name"] == "scale_factor")
            defects: list[tuple[str | int, dict, dict]] = [
                ("missing", benchmark, {"states": states[:-1]}),
                ("duplicate", benchmark, {"states": [*states[:-1], state]}),
                ("skipped", state, {"is_skipped": True}),
                ("untimed", state, {"summaries": []}),
                ("wrong_timer", summary, {"tag": "nv/cold/time/gpu/mean"}),
                ("wrong_scale", scale_axis, {"value": "10"}),
                ("wrong_name", benchmark, {"name": f"ndsh_q{query}_other"}),
                ("wrong_device", state, {"device": 1}),
            ]
            for name in ("format", "workload", "cache", "engine"):
                axes = [axis for axis in state["axis_values"] if axis["name"] != name]
                defects.append((name, state, {"axis_values": [*axes, {"name": name, "value": "unexpected"}]}))
            for value in (0, -1, "nan", "inf"):
                defects.append((value, summary["data"][0], {"value": value}))
            for defect, target, changes in defects:
                with self.subTest(query=query, defect=defect), patch.dict(target, changes):
                    with self.assertRaises(RuntimeError):
                        reproduce.validate_results(data, query, 1.0)

    def write_build_record(self, env: dict[str, str] | None = None):
        build = self.args.work_dir / "cudf-build"
        sha = hashlib.sha256(b"fixture").hexdigest()
        for name in reproduce.TARGETS:
            self.write(build / "benchmarks" / name)
        self.write(build / "libcudf.so")
        record = {
            "recipe": self.recipe,
            "binaries": dict.fromkeys(reproduce.TARGETS, sha),
            "libcudf": sha,
            "environment": env or {},
        }
        self.write(self.args.work_dir / "build.json", json.dumps(record))

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

    def test_download_checks_fresh_and_cached_hashes(self):
        name = "fixture.cmake"
        path = self.args.work_dir / name
        temporary = path.with_suffix(".cmake.part")
        source = {"url": "https://example.invalid/fixture", "sha256": hashlib.sha256(b"fixture").hexdigest()}
        self.runner.run.side_effect = lambda _label, argv: self.write(argv[4])
        for phase in ("fresh", "cached"):
            with self.subTest(phase=phase):
                self.assertEqual(reproduce.Runner.download(self.runner, name, source), path)
        self.runner.run.assert_called_once_with(
            name, ["curl", "--fail", "--location", "--output", temporary, source["url"]]
        )
        self.assertEqual(path.read_text(), "fixture")
        self.assertFalse(temporary.exists())
        self.runner.run.reset_mock()
        self.write(path, "corrupt")
        with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
            reproduce.Runner.download(self.runner, name, source)
        self.runner.run.assert_not_called()
        path.unlink()
        self.runner.run.side_effect = lambda _label, argv: self.write(argv[4], "corrupt")
        with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
            reproduce.Runner.download(self.runner, name, source)
        self.assertFalse(path.exists())
        self.assertEqual(temporary.read_text(), "corrupt")

    def test_identity_requires_a_clean_revision(self):
        with patch.object(reproduce, "git_output", side_effect=["", "revision"]) as git:
            self.assertEqual(reproduce.identity(), {"vortex_revision": "revision"})
        git.assert_any_call(reproduce.ROOT, "status", "--porcelain", "--untracked-files=all")
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

    def test_failed_build_clears_success_and_tracks_indexed_source_pins(self):
        work = self.args.work_dir
        record = work / "build.json"
        self.write(record, "stale success")
        self.runner.checkout.side_effect = lambda name, *_args, **_kwargs: work / name
        original = {"diff": "indexed patch including added loader", "status": "A  loader.cmake"}

        def run(label, *_args):
            self.assertFalse(record.exists())
            if label == "flatc-configure":
                raise RuntimeError("stop before configure")
            return ""

        self.runner.run.side_effect = run
        with (
            patch.object(reproduce, "git_output", return_value=""),
            patch.object(reproduce, "source_state", return_value=original) as source_state,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop before configure"):
                reproduce.build(self.args, LOCK, self.runner, self.recipe)
            self.assertFalse(record.exists())
            self.runner.download.assert_called_once_with("CPM.cmake", LOCK["cpm"])
            for name in ("dlpack", "xxhash"):
                package = LOCK["packages"][name]
                self.runner.checkout.assert_any_call(
                    name, {"repository": package["git_url"], "commit": package["git_tag"]}
                )
            self.runner.run.assert_any_call(
                "patch", ["git", "-C", work / "cudf", "apply", "--index", HERE / "upstream.patch"]
            )
            self.assertEqual(json.loads((work / "cudf-source.json").read_text()), original)
            source_state.return_value = {**original, "diff": "changed loader contents"}
            with self.assertRaisesRegex(RuntimeError, "Prepared cuDF source changed"):
                reproduce.build(self.args, LOCK, self.runner, self.recipe)

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
            "CUDF_WITH_VORTEX": "ON",
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

    def test_configure_forwards_compilers_but_flatc_uses_only_host_settings(self):
        host = [
            "-DCMAKE_C_COMPILER=caller-cc",
            "-DCMAKE_CXX_COMPILER:FILEPATH=caller-cxx",
            "-DCMAKE_C_COMPILER_ARG1=--driver-mode=gcc",
            "-DCMAKE_CXX_COMPILER_ARG1:STRING=--driver-mode=g++",
            "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=toolchain.cmake",
            "-DCMAKE_SYSROOT=sysroot",
        ]
        self.args.cmake_arg = [
            *host,
            "-DCMAKE_CUDA_COMPILER:FILEPATH=caller-nvcc",
            "-DCMAKE_CUDA_HOST_COMPILER=caller-cuda-host",
            "-DCMAKE_CUDA_COMPILER_ARG1=--allow-unsupported-compiler",
            "-DCMAKE_CUDA_HOST_COMPILER_ARG1:STRING=--driver-mode=g++",
            "-DCMAKE_CUDA_ARCHITECTURES=90",
            "-DCMAKE_CXX_FLAGS=-O2",
        ]
        self.assertEqual(reproduce.compiler_arguments(self.args.cmake_arg), host)
        command = list(map(str, reproduce.configure_command(self.args, LOCK)))
        for argument in self.args.cmake_arg:
            self.assertEqual(command.count(argument), 1)

    def test_environment_preserves_caller_build_settings_without_secrets_or_shared_cache(self):
        preserved = {
            "PATH": "caller-bin",
            "LD_LIBRARY_PATH": "caller-lib",
            "CPATH": "caller-include",
            "CC": "caller-cc",
            "CXX": "caller-cxx",
            "CUDACXX": "caller-nvcc",
            "CUDAHOSTCXX": "caller-cuda-host",
            "NVCC_CCBIN": "caller-cuda-host",
            "CXXFLAGS": "-O2",
            "CMAKE_TOOLCHAIN_FILE": "toolchain.cmake",
            "CUDA_VISIBLE_DEVICES": "1",
            "RUSTFLAGS": "-Ctarget-cpu=native",
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

    def test_record_toolchain_preserves_compilers_and_arg1_and_requires_cuda_12_8(self):
        compilers = {name: self.root / name.lower() for name in COMPILERS}
        for name, path in compilers.items():
            self.write(path, name)
        cache = {
            **{name: str(path) for name, path in compilers.items()},
            **{
                f"{name}_ARG1": f'"compiler tools/{name.lower()}" --config \'config with spaces\''
                for name in COMPILERS
            },
            "CMAKE_CUDA_ARCHITECTURES": "90",
            "CMAKE_CXX_FLAGS": "-O2 -DVALUE=a=b",
            "nvcomp_DIR": str(self.root / "native-nvcomp/lib/cmake/nvcomp"),
        }
        self.write(
            self.args.work_dir / "cudf-build/CMakeCache.txt",
            "\n".join(f"{name}:STRING={value}" for name, value in cache.items()),
        )
        versions = {name.lower(): f"{name} fixture version\n" for name in compilers}
        self.runner.run.side_effect = lambda label, _argv: versions[label]
        for version, accepted in (("12.8", True), ("13.0", True), ("12.7", False), ("unknown", False)):
            with self.subTest(version=version):
                versions["cmake_cuda_compiler"] = f"Cuda compilation tools, release {version}, V{version}.0\n"
                self.runner.run.reset_mock()
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
                    [
                        call(
                            name.lower(),
                            [path, f"compiler tools/{name.lower()}", "--config", "config with spaces", "--version"],
                        )
                        for name, path in compilers.items()
                    ],
                )

    def test_main_run_uses_source_identity_and_recorded_environment(self):
        work = self.args.work_dir.resolve()
        env = {"PATH": "recorded-bin", "LD_LIBRARY_PATH": "recorded-lib"}
        self.write_build_record(env)
        source = (work / "build.json").read_text()
        argv = ["reproduce.py", "run", "--work-dir", str(work), "--queries", "1", "--timeout", "60"]

        def run(label, command, *_args):
            if label == "sf1-q1":
                self.write(command[-1], json.dumps(result_data(1)))

        self.runner.run.side_effect = run
        with (
            patch("sys.argv", argv),
            patch.object(reproduce.platform, "system", return_value="Linux"),
            patch.object(reproduce, "identity", return_value=self.recipe) as identity,
            patch.object(reproduce, "timestamp", return_value="run"),
            patch.object(reproduce, "Runner", return_value=self.runner) as runner,
        ):
            reproduce.main()
        identity.assert_called_once_with()
        runtime_path = os.pathsep.join((str(work / "cudf-build"), "recorded-lib"))
        runner.assert_called_once_with(work, {**env, "LD_LIBRARY_PATH": runtime_path}, 60)
        self.assertEqual((work / "build.json").read_text(), source)
        self.assertEqual(json.loads((work / "results/run/build.json").read_text()), json.loads(source))
        self.assertEqual(json.loads((work / "results/run/run.json").read_text())["queries"], [1])


if __name__ == "__main__":
    unittest.main()
