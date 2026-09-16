# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Offline build wiring and timing/ownership guards.

Full cuDF builds and GPU cases cover C++ includes, projections, and query results;
source checks here focus on contracts those cannot establish, such as timing boundaries.
Run with python3 -B benchmarks/cudf-ndsh/test_build_integration.py.
"""

import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path("benchmarks/cudf-ndsh")
PATCH = ROOT / HARNESS / "upstream.patch"
MODULE = HARNESS / "vortex.cmake"
SOURCES = HARNESS / "src/vortex_ndsh"
SMOKE = HARNESS / "tests/vortex_build_smoke.cpp"
IO = SOURCES / "vortex_io.cpp"
IO_TEST = HARNESS / "tests/vortex_io_test.cpp"
BENCHMARKS = Path("cpp/benchmarks")
CUDF_CMAKE = Path("cpp/CMakeLists.txt")
LOADER = Path("cpp/cmake/thirdparty/get_vortex.cmake")
NDSH = BENCHMARKS / "ndsh"
QUERIES = (1, 5, 6, 9, 10)


def patch_postimages():
    # Inspect exported hunk postimages, never the ignored development checkout.
    sources = {}
    for section in PATCH.read_text(encoding="utf-8").split("diff --git ")[1:]:
        _, separator, post_image = section.partition("\n+++ b/")
        if separator:
            path, _, hunks = post_image.partition("\n")
            sources[Path(path)] = "\n".join(
                line[1:] for line in hunks.splitlines() if line.startswith(("+", " "))
            )
    return sources


def patch_hook(sources):
    option = re.search(r"^option\(CUDF_WITH_VORTEX\b[^\n]*\)$", sources[CUDF_CMAKE], re.MULTILINE)
    hook = re.search(
        r"^if\(CUDF_WITH_VORTEX\)\n.*?^endif\(\)",
        sources[BENCHMARKS / "CMakeLists.txt"],
        re.MULTILINE | re.DOTALL,
    )
    if option is None or hook is None:
        raise AssertionError("Exported patch is missing the CUDF_WITH_VORTEX option or bootstrap hook")
    return option.group(0), hook.group(0)


PARENT = r"""
function(assert_equal actual expected)
  if(NOT "${actual}" STREQUAL "${expected}")
    message(FATAL_ERROR "Expected '${expected}', got '${actual}' (${ARGN})")
  endif()
endfunction()

function(CPMAddPackage)
  file(APPEND "${CMAKE_BINARY_DIR}/cpm-vortex.txt" "vortex\n")
  cmake_parse_arguments(PARSE_ARGV 0 package "" "NAME;GIT_REPOSITORY;GIT_TAG;DOWNLOAD_ONLY" "")
  assert_equal("${package_NAME}" vortex)
  assert_equal("${package_GIT_REPOSITORY}" "https://github.com/vortex-data/vortex.git")
  assert_equal("${package_GIT_TAG}" "90723345eeed838405341da9d44e02c94f10e6be")
  assert_equal("${package_DOWNLOAD_ONLY}" TRUE)
  set(vortex_SOURCE_DIR "${CMAKE_SOURCE_DIR}/fake-vortex" PARENT_SCOPE)
endfunction()

set(parent_options VORTEX_ENABLE_CUDA VORTEX_BUILD_TESTS VORTEX_BUILD_EXAMPLES VORTEX_WARNINGS_AS_ERRORS)
set(parent_values OFF ON ON ON)
foreach(option value IN ZIP_LISTS parent_options parent_values)
  set(${option} ${value} CACHE BOOL "Parent option")
endforeach()
set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "Parent architecture")
set(CMAKE_CXX_FLAGS "-DPARENT_CXX_FLAG=1" CACHE STRING "Parent flags" FORCE)
set(CMAKE_CUDA_FLAGS "--parent-cuda-flag" CACHE STRING "Parent flags" FORCE)

add_library(cudf::cudf INTERFACE IMPORTED)
set(dependencies nanoarrow::nanoarrow CUDA::cudart nvtx3::nvtx3-cpp kvikio::kvikio)
set(markers NANOARROW_FAKE_LINK=1 CUDA_FAKE_LINK=1 NVTX_FAKE_LINK=1 KVIKIO_FAKE_LINK=1)
foreach(target marker IN ZIP_LISTS dependencies markers)
  add_library(${target} INTERFACE IMPORTED)
  target_compile_definitions(${target} INTERFACE ${marker})
endforeach()
add_library(ndsh_utilities INTERFACE)
set(benchmarks NDSH_Q01_NVBENCH NDSH_Q06_NVBENCH NDSH_Q05_NVBENCH
               NDSH_Q09_NVBENCH NDSH_Q10_NVBENCH NDSH_HELPER)
foreach(target IN LISTS benchmarks)
  add_executable(${target} dummy.cpp)
  target_link_libraries(${target} PRIVATE cudf::cudf ndsh_utilities)
  if(CUDF_WITH_VORTEX AND NOT target STREQUAL "NDSH_HELPER")
    target_compile_definitions(${target} PRIVATE EXPECT_VORTEX=1)
  endif()
endforeach()
add_executable(unrelated dummy.cpp)

include(ndsh-hook.cmake)

foreach(option value IN ZIP_LISTS parent_options parent_values)
  get_property(cached CACHE ${option} PROPERTY VALUE)
  assert_equal("${${option}}" "${value}" "parent ${option}")
  assert_equal("${cached}" "${value}" "cache ${option}")
endforeach()

if(CUDF_WITH_VORTEX)
  foreach(target IN ITEMS NDSH_VORTEX_IO NDSH_VORTEX_BUILD_SMOKE NDSH_VORTEX_IO_TEST)
    get_target_property(excluded ${target} EXCLUDE_FROM_ALL)
    assert_equal("${excluded}" TRUE "${target} must require an explicit target or dependent")
  endforeach()
endif()
"""

FAKE_VORTEX = """
project(fake_vortex LANGUAGES CXX)
option(VORTEX_ENABLE_CUDA "Fake CUDA default" OFF)
option(VORTEX_BUILD_TESTS "Fake test default" ON)
option(VORTEX_BUILD_EXAMPLES "Fake example default" ON)
option(VORTEX_WARNINGS_AS_ERRORS "Fake warning default" ON)
assert_equal("${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/fake-vortex/lang/cpp")
assert_equal("${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_BINARY_DIR}/cpp/benchmarks/_deps/vortex-build")
assert_equal("${VORTEX_ENABLE_CUDA}" ON)
assert_equal("${VORTEX_BUILD_TESTS}" OFF)
assert_equal("${VORTEX_BUILD_EXAMPLES}" OFF)
assert_equal("${VORTEX_WARNINGS_AS_ERRORS}" OFF)
assert_equal("${CMAKE_CUDA_ARCHITECTURES}" 90)
assert_equal("${CMAKE_CXX_FLAGS}" "-DPARENT_CXX_FLAG=1")
assert_equal("${CMAKE_CUDA_FLAGS}" "--parent-cuda-flag")
add_library(vortex_cpp_static INTERFACE)
add_library(Vortex::cpp_static ALIAS vortex_cpp_static)
target_compile_definitions(vortex_cpp_static INTERFACE VORTEX_FAKE_LINK=1)
get_target_property(system vortex_cpp_static SYSTEM)
if(NOT system)
  message(FATAL_ERROR "Vortex subdirectory lost SYSTEM isolation")
endif()
# A default build must not build unrelated ALL targets from the Vortex subdirectory.
add_custom_target(vortex_unwanted ALL COMMAND "${CMAKE_COMMAND}" -E false)
"""

DUMMY = """
#ifndef PARENT_CXX_FLAG
#error Parent compiler flags were lost
#endif
#if defined(VORTEX_FAKE_LINK) || defined(NANOARROW_FAKE_LINK) || defined(CUDA_FAKE_LINK) || defined(NVTX_FAKE_LINK)
#error Private adapter dependency usage requirements leaked
#endif
#if defined(CUDF_WITH_VORTEX) || defined(CUDF_NDSH_WITH_VORTEX)
#error Unexpected benchmark Vortex definition
#endif
#if defined(EXPECT_VORTEX) != defined(KVIKIO_FAKE_LINK) || defined(EXPECT_VORTEX) != defined(CUDF_NDSH_QUERY_EXTENSION)
#error Query dependency or extension missing or leaked
#endif
#ifdef EXPECT_VORTEX
#include CUDF_NDSH_QUERY_EXTENSION
#else
int main() { return 0; }
#endif
"""

IO_STUB = """
#ifndef PARENT_CXX_FLAG
#error Parent compiler flags were lost
#endif
#if !defined(VORTEX_FAKE_LINK) || !defined(NANOARROW_FAKE_LINK) || !defined(CUDA_FAKE_LINK) || !defined(NVTX_FAKE_LINK)
#error Missing private adapter dependency usage requirements
#endif
#if defined(CUDF_NDSH_QUERY_EXTENSION) || defined(CUDF_WITH_VORTEX) || defined(CUDF_NDSH_WITH_VORTEX) || defined(EXPECT_VORTEX) || defined(KVIKIO_FAKE_LINK)
#error Benchmark definitions leaked into the adapter
#endif
int ndsh_vortex_io_stub() { return 0; }
"""


class BuildIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for tool in ("cmake", "c++"):
            if shutil.which(tool) is None:
                raise unittest.SkipTest(f"{tool} is required")
        version = subprocess.run(
            ["cmake", "--version"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.split()[2]
        if tuple(map(int, version.split(".")[:2])) < (3, 25):
            raise unittest.SkipTest("CMake >= 3.25 is required for block() and add_subdirectory(SYSTEM)")

    def setUp(self):
        build = ROOT / "build"
        build.mkdir(exist_ok=True)
        self.source = Path(self.enterContext(tempfile.TemporaryDirectory(prefix="ndsh-unittest-", dir=build)))
        self.binary = self.source / "out"
        self.fake = self.source / "fake-vortex"
        # Compile only wiring stubs. Real smoke/I/O sources must stay excluded from default builds.
        for path in (MODULE, SMOKE, IO_TEST):
            self.write(self.fake / path, (ROOT / path).read_text(encoding="utf-8"))
        self.write(self.fake / IO, IO_STUB)
        for query in QUERIES:
            self.write(
                self.fake / SOURCES / f"q{query:02}.inc",
                '#include "utilities.hpp"\n'
                "int ndsh_vortex_io_stub();\n"
                "int main() { return ndsh_vortex_io_stub(); }\n",
            )
        self.write(NDSH / "utilities.hpp", "// Requires the query's private upstream include directory.\n")
        sources = patch_postimages()
        option, hook = patch_hook(sources)
        self.write(
            "CMakeLists.txt",
            "cmake_minimum_required(VERSION 3.25)\n"
            "project(ndsh_integration_test LANGUAGES CXX)\n"
            f"{option}\n"
            "add_subdirectory(cpp/benchmarks)\n",
        )
        self.write(BENCHMARKS / "CMakeLists.txt", PARENT)
        self.write(BENCHMARKS / "ndsh-hook.cmake", hook)
        self.write(LOADER, sources[LOADER])
        self.write(BENCHMARKS / "dummy.cpp", DUMMY)
        self.write("fake-vortex/CMakeLists.txt", 'message(FATAL_ERROR "Expected lang/cpp entry point")')
        self.write("fake-vortex/lang/cpp/CMakeLists.txt", FAKE_VORTEX)

    def write(self, path, content):
        destination = self.source / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(textwrap.dedent(content), encoding="utf-8")

    def run_command(self, *command, succeeds=True):
        result = subprocess.run(
            command, cwd=self.source, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60
        )
        self.assertEqual(
            result.returncode == 0, succeeds, f"{command} exited {result.returncode}\n{result.stdout}"
        )
        return result.stdout

    def configure(self, *options, override=None, succeeds=True):
        return self.run_command(
            "cmake",
            "-S",
            str(self.source),
            "-B",
            str(self.binary),
            # Exercise the Linux gates without enabling CUDA, even on a non-Linux host.
            "-DCMAKE_SYSTEM_NAME=Linux",
            "-DCMAKE_BUILD_TYPE=Debug",
            *options,
            *(() if override is None else (f"-DFETCHCONTENT_SOURCE_DIR_VORTEX={override}",)),
            succeeds=succeeds,
        )

    def build(self, *targets):
        options = ("--target", *targets) if targets else ()
        self.run_command("cmake", "--build", str(self.binary), "--parallel", "2", *options)

    def assert_download(self, expected):
        record = self.binary / "cpm-vortex.txt"
        self.assertEqual(
            record.read_text(encoding="utf-8") if record.exists() else "",
            "vortex\n" if expected else "",
        )

    def test_disabled_by_default_and_explicitly(self):
        self.write(self.fake / MODULE, 'message(FATAL_ERROR "Disabled Vortex module was included")')
        for setting in (None, "OFF"):
            with self.subTest(setting=setting):
                self.binary = self.source / (setting or "default")
                self.configure(*(() if setting is None else (f"-DCUDF_WITH_VORTEX={setting}",)))
                cache = (self.binary / "CMakeCache.txt").read_text(encoding="utf-8")
                self.assertIn("CUDF_WITH_VORTEX:BOOL=OFF", cache)
                self.assert_download(False)
                self.build()

    def test_enabled_private_links_and_excluded_targets(self):
        for index, override in enumerate((None, self.fake)):
            with self.subTest(override=override):
                self.binary = self.source / f"enabled-{index}"
                self.configure("-DCUDF_WITH_VORTEX=ON", override=override)
                self.assert_download(not override)
        archive = self.binary / BENCHMARKS / "libNDSH_VORTEX_IO.a"
        self.build("unrelated", "NDSH_HELPER")
        self.assertFalse(archive.exists(), "Unselected targets built the adapter")
        # Both source routes use the same stubs; build the local override once.
        self.build()
        self.assertTrue(archive.is_file(), "Default build did not build the adapter through queries")
        for query in QUERIES:
            target = f"NDSH_Q{query:02}_NVBENCH"
            self.assertTrue((self.binary / BENCHMARKS / target).is_file(), f"Default build omitted {target}")
        for target in ("NDSH_VORTEX_BUILD_SMOKE", "NDSH_VORTEX_IO_TEST"):
            self.assertFalse(
                (self.binary / BENCHMARKS / target).exists(), f"Explicit target {target} was built implicitly"
            )

    def test_source_validation_is_enabled_only(self):
        (self.fake / MODULE).unlink()
        self.configure("-DCUDF_WITH_VORTEX=OFF", "-DCMAKE_SYSTEM_NAME=Generic", override=self.fake)
        self.assert_download(False)
        self.binary = self.source / "invalid-source"
        output = self.configure("-DCUDF_WITH_VORTEX=ON", override=self.fake, succeeds=False)
        self.assert_download(False)
        message = " ".join(output.split())
        self.assertIn("CUDF_WITH_VORTEX requires Vortex sources containing", message)
        self.assertIn("benchmarks/cudf-ndsh/vortex.cmake", message)
        self.assertIn("-DFETCHCONTENT_SOURCE_DIR_VORTEX=/path/to/vortex", message)
        self.assertIn(str(self.fake), output)

    def test_enabled_requires_linux_before_download(self):
        output = self.configure("-DCUDF_WITH_VORTEX=ON", "-DCMAKE_SYSTEM_NAME=Generic", succeeds=False)
        self.assertIn("CUDF_WITH_VORTEX requires Linux and a CUDA toolkit", " ".join(output.split()))
        self.assert_download(False)


class BenchmarkSourceTests(unittest.TestCase):
    def source(self, path, function=None):
        source = (ROOT / path).read_text(encoding="utf-8")
        if function is not None:
            # These free functions end at column zero; nested scopes stay indented.
            source = self.require_match(
                source, rf"(?ms)\b{re.escape(function)}\([^;{{]*\)\s*\{{(.*?)^\}}"
            ).group(1)
        return " ".join(source.split())

    def require_match(self, source, pattern):
        match = re.search(pattern, source)
        self.assertIsNotNone(match, f"Missing source pattern: {pattern}")
        return match

    def test_shared_local_benchmark_phase_ordering(self):
        self.assertEqual(
            self.source(SOURCES / "local_io.hpp", "warm_local_inputs"),
            "if (!cold) { auto inputs = read_all(); CUDF_CUDA_TRY(cudaDeviceSynchronize()); }",
        )
        self.require_match(
            self.source(SOURCES / "local_io.hpp", "exec_local_benchmark"),
            r'state.add_element_count\(files.rows, "Rows"\); state.exec\(\s*'
            r"nvbench::exec_tag::sync \| nvbench::exec_tag::timer, "
            r"\[&\]\(nvbench::launch&, auto& timer\) \{ "
            r"if \(cold\) \{ evict_file_pages\(files.paths\(use_vortex\)\); \} "
            r"timer.start\(\); run\(\); CUDF_CUDA_TRY\(cudaDeviceSynchronize\(\)\); timer.stop\(\); \}\);",
        )

    def test_local_benchmarks_wire_warmup_and_timed_owners(self):
        sync = r"CUDF_CUDA_TRY\(cudaStreamSynchronize\(stream.get\(\)\)\);"
        for query in QUERIES:
            with self.subTest(query=query):
                local = self.source(SOURCES / f"q{query:02}.inc", f"ndsh_q{query}_local")
                execution = self.require_match(
                    local,
                    r"ndsh::exec_local_benchmark\(\s*state, files.tables, use_vortex, cold, \[&\] \{ (.*) \}\);",
                )
                setup, timed = local[: execution.start()], execution.group(1)
                self.assertIn(f"ndsh::local_options{{state, {query}}}", setup)
                self.require_match(setup, r"ndsh::read_local_file\([^;]+, cold\)")
                self.assertNotRegex(local, r"state\.exec\(|timer\.(?:start|stop)\(|evict_file_pages\(")
                if query in (1, 6):
                    warmup = "ndsh::warm_local_inputs(cold, read);"
                else:
                    read_all = (
                        "load_data(read)"
                        if query == 9
                        else f"ndsh::read_local_tables(q{query}_tables, q{query}_projections, read)"
                    )
                    warmup = f"ndsh::warm_local_inputs(cold, [&] {{ return {read_all}; }});"
                self.assertIn(warmup, setup)
                if query == 1:
                    self.require_match(
                        setup.split(warmup, 1)[1],
                        r"if \(!read_only\) \{ auto result = execute_q1\(.*?; "
                        r"ndsh::check_q1_result\(files.reference, \*result, stream\); "
                        r"CUDF_CUDA_TRY\(cudaDeviceSynchronize\(\)\); \}",
                    )
                nvtx = (
                    re.escape(f'cudf::benchmark::scoped_range timed_range{{"ndsh_q{query}_local_timed"}}; ')
                    if query in (1, 10)
                    else ""
                )
                if query in (1, 6):
                    owners = rf"auto result = read_only \? read\(\) : execute_q{query}\(.*?\); {sync}"
                else:
                    owners = (
                        r"if \(read_only\) \{ auto inputs = "
                        + re.escape(read_all)
                        + rf"; {sync} \}} else \{{ auto result = execute_q{query}\([^;]+; {sync} \}}"
                    )
                self.require_match(timed, f"^{nvtx}{owners}$")

    def test_per_file_eviction_verifies_no_resident_pages(self):
        self.require_match(
            self.source(SOURCES / "local_io.hpp", "evict_file_pages"),
            r"for \([^)]*: paths\) \{ kvikio::drop_file_page_cache\(path\); \} "
            r"for \([^)]*: paths\) \{ auto const resident_pages = kvikio::get_page_cache_info\(path\).first; "
            r"CUDF_EXPECTS\(resident_pages == 0,",
        )

    def test_adapter_owners_release_after_consumer_completion(self):
        source = self.source(IO)
        self.assertIn("nanoarrow::device::UniqueDeviceArrayStream value;", source)
        batch = self.require_match(source, r"struct device_batch \{(.*?)\};").group(1)
        self.assertLess(
            batch.index("nanoarrow::device::UniqueDeviceArray value;"),
            batch.index("std::optional<cudf::unique_table_view_t> view;"),
        )
        self.assertIn("device_batch(device_batch&&) noexcept = default;", batch)
        self.assertIn("device_batch& operator=(device_batch&&) = delete;", batch)
        self.assertNotIn("~device_batch", batch)
        self.assertIn("std::vector<device_batch> batches; stream_drain drain{impl_->stream};", source)
        completion = source.index('nvtx3::scoped_range range{"vortex.consumer_sync"}; drain.wait();')
        release = source.index('nvtx3::scoped_range range{"vortex.release_batches"}; batches.clear();')
        self.assertLess(source.index('"vortex.materialize"'), completion)
        self.assertLess(completion, release)


if __name__ == "__main__":
    unittest.main(verbosity=2)
