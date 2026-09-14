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
from unittest.mock import Mock, patch

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
        axes = dict(
            scale_factor="1",
            format=fmt,
            workload=workload,
            cache=cache,
        )
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
    return {
        "benchmarks": [
            {
                "name": f"ndsh_q{query}_local",
                "states": states,
            }
        ]
    }


class ReproduceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ndsh-reproduce-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = SimpleNamespace(
            work_dir=self.root / "work",
            toolchain=self.root / "toolchain",
            cuda_root=self.root / "cuda",
            clangxx=self.root / "clang++",
            libclang=self.root / "llvm/lib",
            cargo_jobs=4,
            jobs=2,
            scale_factor=1.0,
            queries=list(QUERIES),
            min_samples=3,
            sample_timeout=30,
        )
        self.recipe = {"inputs": {"reproduce.py": "original"}}
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
                reproduce.validate_results(result_data(query), query, 1.0)

    def test_rejects_incomplete_skipped_and_untimed_matrices(self):
        for query, defect in itertools.product(
            QUERIES,
            (
                "missing",
                "duplicate",
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

    def test_stale_binary_and_library_hashes_fail_before_gpu_calls(self):
        build = self.args.work_dir / "cudf-build"
        binaries = {name: hashlib.sha256(b"fixture").hexdigest() for name in reproduce.TARGETS}
        record = {
            "recipe": self.recipe,
            "binaries": binaries,
            "libcudf": hashlib.sha256(b"fixture").hexdigest(),
        }
        self.write(self.args.work_dir / "build.json", json.dumps(record))
        for path in [
            *(build / "benchmarks" / name for name in binaries),
            build / "libcudf.so",
        ]:
            self.write(path)
        for path in (build / "benchmarks/NDSH_Q09_NVBENCH", build / "libcudf.so"):
            with self.subTest(path=path):
                self.write(path, "stale")
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    reproduce.benchmark(self.args, self.runner, self.recipe)
                self.assertEqual(self.runner.mock_calls, [])
                self.assertFalse((self.args.work_dir / "results").exists())
                self.write(path)

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
            reproduce.initialize_work(owned, {"inputs": {"reproduce.py": "changed"}})
        self.assertEqual(json.loads((owned / "recipe.json").read_text()), self.recipe)

    def test_release_toolchain_and_package_pins(self):
        prefix, work, cuda = self.args.toolchain, self.args.work_dir, self.args.cuda_root
        nvcomp = work / LOCK["nvcomp"]["directory"]
        command = list(map(str, reproduce.configure_command(self.args, LOCK, nvcomp)))
        flags = dict(value[2:].split("=", 1) for value in command if value.startswith("-D"))
        self.assertEqual(command[0], str(prefix / "bin/cmake"))
        self.assertIn("--fresh", command)
        expected = {
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_CUDA_ARCHITECTURES": LOCK["cuda_architectures"],
            "CMAKE_C_COMPILER": prefix / "bin/aarch64-conda-linux-gnu-gcc",
            "CMAKE_CXX_COMPILER": prefix / "bin/aarch64-conda-linux-gnu-g++",
            "CMAKE_CUDA_HOST_COMPILER": prefix / "bin/aarch64-conda-linux-gnu-g++",
            "CMAKE_CUDA_COMPILER": cuda / "bin/nvcc",
            "CUDAToolkit_ROOT": cuda,
            "CMAKE_MAKE_PROGRAM": prefix / "bin/ninja",
            "CMAKE_PREFIX_PATH": prefix,
            "Python_EXECUTABLE": prefix / "bin/python",
            "Python3_EXECUTABLE": prefix / "bin/python",
            "RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE": HERE / "build-lock.json",
            "nvcomp_DIR": nvcomp / "lib/cmake/nvcomp",
            "CPM_dlpack_SOURCE": work / "dlpack",
            "CPM_xxhash_SOURCE": work / "xxhash",
            "CPM_DOWNLOAD_CURL": "OFF",
            "CMAKE_REQUIRE_FIND_PACKAGE_CURL": "ON",
            "CURL_NO_CURL_CMAKE": "ON",
            "CURL_INCLUDE_DIR": prefix / "include",
            "CURL_LIBRARY": prefix / "lib/libcurl.so",
            "CPM_DOWNLOAD_nlohmann_json": "ON",
        }
        for key, value in expected.items():
            self.assertEqual(flags.get(key), str(value), key)
        for language in ("C", "CXX", "CUDA"):
            self.assertEqual(flags[f"CMAKE_{language}_FLAGS"], "")
            self.assertEqual(flags[f"CMAKE_{language}_FLAGS_RELEASE"], "-O3 -DNDEBUG")
        for name, package in LOCK["packages"].items():
            self.assertEqual(flags.get(f"CPM_DOWNLOAD_{name}"), "ON", name)
            self.assertRegex(
                package.get("git_tag", package.get("url_hash", "")),
                r"^(?:[0-9a-f]{40}|SHA256=[0-9a-f]{64})$",
            )
        self.assertNotIn("nlohmann_json", LOCK["packages"])
        self.assertEqual({key for key in flags if "nlohmann" in key.lower()}, {"CPM_DOWNLOAD_nlohmann_json"})

    def test_build_checks_out_download_only_packages_at_locked_commits(self):
        for name in (
            "include/curl/curl.h",
            "include/curl/curlver.h",
            "lib/libcurl.so",
            "lib/pkgconfig/libcurl.pc",
        ):
            self.write(self.args.toolchain / name)
        self.write(
            self.args.cuda_root / "version.json",
            json.dumps(
                {
                    "cuda": {"version": LOCK["cuda_version"]},
                    "cuda_cudart": {"version": LOCK["cuda_runtime_version"]},
                }
            ),
        )
        self.write(self.args.work_dir / "build.json", "stale success")
        versions = {
            "nvcc-version": f"V{LOCK['nvcc_version']}",
            "clang-version": f"version {LOCK['clang_version']}",
        }
        self.runner.run.side_effect = lambda label, *args: versions.get(label, "")
        # Stop before archives or configuration; all external operations are mocked.
        self.runner.download.side_effect = RuntimeError("stop after checkouts")
        with self.assertRaisesRegex(RuntimeError, "stop after checkouts"):
            reproduce.build(self.args, LOCK, self.runner, self.recipe)
        self.assertFalse((self.args.work_dir / "build.json").exists())
        for name in ("dlpack", "xxhash"):
            package = LOCK["packages"][name]
            self.runner.checkout.assert_any_call(
                name,
                {
                    "repository": package["git_url"],
                    "commit": package["git_tag"],
                },
            )

    def test_environment_ignores_ambient_flags_caches_and_secrets(self):
        with patch.dict(os.environ, {"HOME": str(self.root)}, clear=True):
            clean = reproduce.environment(self.args)
            hostile = dict.fromkeys(
                (
                    "PATH",
                    "LD_LIBRARY_PATH",
                    "CC",
                    "CXX",
                    "CUDACXX",
                    "CFLAGS",
                    "CXXFLAGS",
                    "LDFLAGS",
                    "CMAKE_PREFIX_PATH",
                    "CPM_SOURCE_CACHE",
                    "CCACHE_DIR",
                    "SCCACHE_DIR",
                    "RUSTC_WRAPPER",
                    "RUSTFLAGS",
                    "NVCC_PREPEND_FLAGS",
                    "AWS_SECRET_ACCESS_KEY",
                    "GITHUB_TOKEN",
                    "PYTHONPATH",
                    "PYTHONHOME",
                    "PYTHONWARNINGS",
                    "PYTHONOPTIMIZE",
                    "PYTHONNOUSERSITE",
                    "CARGO_BUILD_JOBS",
                ),
                "ambient",
            )
            with patch.dict(os.environ, hostile):
                self.assertEqual(reproduce.environment(self.args), clean)
        self.assertEqual(clean["PYTHONNOUSERSITE"], "1")
        self.assertEqual(clean["NVCC_CCBIN"], str(self.args.toolchain / "bin/aarch64-conda-linux-gnu-g++"))
        self.assertEqual(clean["FLATC"], str(self.args.work_dir / "flatc-build/flatc"))
        self.assertEqual(clean["CARGO_BUILD_JOBS"], "4")

    def test_conda_lock_requires_exact_package_hashes_and_set(self):
        metadata = self.args.toolchain / "conda-meta"
        for line in (HERE / "environment-linux-aarch64.lock").read_text().splitlines():
            if line.startswith("https://"):
                filename, sha = line.rsplit("/", 1)[1].split("#")
                self.write(
                    metadata / f"{filename.removesuffix('.conda')}.json",
                    json.dumps({"sha256": sha}),
                )
        reproduce.locked_environment(self.args.toolchain)
        package = next(metadata.glob("*.json"))
        original = package.read_text()
        for defect in ("hash", "missing", "extra"):
            with self.subTest(defect=defect):
                if defect == "hash":
                    self.write(package, json.dumps({"sha256": "0" * 64}))
                elif defect == "missing":
                    package.unlink()
                else:
                    self.write(metadata / "unlocked.json", original)
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    reproduce.locked_environment(self.args.toolchain)
                self.write(package, original)


if __name__ == "__main__":
    unittest.main()
