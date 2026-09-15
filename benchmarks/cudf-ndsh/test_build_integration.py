# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Offline build wiring and benchmark-policy guards.

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

function(assert_properties targets expected)
  foreach(target IN LISTS targets)
    foreach(property IN LISTS ARGN)
      get_target_property(actual ${target} ${property})
      if(actual STREQUAL "actual-NOTFOUND")
        set(actual "")
      endif()
      assert_equal("${actual}" "${expected}" "${target}.${property}")
    endforeach()
  endforeach()
endfunction()

function(CPMAddPackage)
  file(APPEND "${CMAKE_BINARY_DIR}/cpm-vortex.txt" "vortex\n")
  cmake_parse_arguments(PARSE_ARGV 0 package "" "NAME;GIT_REPOSITORY;GIT_TAG;DOWNLOAD_ONLY" "")
  assert_equal("${package_UNPARSED_ARGUMENTS}" "")
  assert_equal("${package_KEYWORDS_MISSING_VALUES}" "")
  assert_equal("${package_NAME}" vortex)
  assert_equal("${package_GIT_REPOSITORY}" "https://github.com/vortex-data/vortex.git")
  assert_equal("${package_GIT_TAG}" "90723345eeed838405341da9d44e02c94f10e6be")
  assert_equal("${package_DOWNLOAD_ONLY}" TRUE)
  set(vortex_SOURCE_DIR "${CMAKE_SOURCE_DIR}/fake-vortex" PARENT_SCOPE)
endfunction()

set(vortex_SOURCE_DIR "parent-vortex-source")
set(vortex_module "parent-vortex-module")
set(VORTEX_ENABLE_CUDA OFF CACHE BOOL "Parent CUDA policy")
set(VORTEX_BUILD_TESTS ON CACHE BOOL "Parent test policy")
set(VORTEX_BUILD_EXAMPLES ON CACHE BOOL "Parent example policy")
set(VORTEX_WARNINGS_AS_ERRORS ON CACHE BOOL "Parent warning policy")
set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "Parent architecture")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON CACHE BOOL "Parent warning policy")
set(CMAKE_CXX_FLAGS "-DPARENT_CXX_FLAG=1" CACHE STRING "Parent flags" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-DPARENT_DEBUG_FLAG=1" CACHE STRING "Parent flags" FORCE)
set(CMAKE_CUDA_FLAGS "--parent-cuda-flag" CACHE STRING "Parent flags" FORCE)
set(CMAKE_CUDA_FLAGS_DEBUG "--parent-cuda-debug-flag" CACHE STRING "Parent flags" FORCE)
set(CUDAToolkit_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/parent-toolkit" CACHE PATH "Parent toolkit")
set(nvcomp_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/parent-nvcomp" CACHE PATH "Parent nvCOMP")
get_cmake_property(parent_variables VARIABLES)
list(FILTER parent_variables INCLUDE REGEX
     "^(VORTEX_|vortex_|CUDAToolkit_|nvcomp_|CMAKE_.*(FLAGS|ARCHITECTURES|STANDARD|WARNING_AS_ERROR|COMPILER))")
foreach(variable IN LISTS parent_variables)
  set(before_${variable} "${${variable}}")
  foreach(property IN ITEMS VALUE TYPE HELPSTRING)
    get_property(before_${variable}_${property} CACHE ${variable} PROPERTY ${property})
  endforeach()
endforeach()

add_library(cudf INTERFACE)
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
endforeach()
add_executable(unrelated dummy.cpp)

include(ndsh-hook.cmake)

foreach(variable IN LISTS parent_variables)
  assert_equal("${${variable}}" "${before_${variable}}" "parent ${variable}")
  foreach(property IN ITEMS VALUE TYPE HELPSTRING)
    get_property(after CACHE ${variable} PROPERTY ${property})
    assert_equal("${after}" "${before_${variable}_${property}}" "cache ${variable}.${property}")
  endforeach()
endforeach()

foreach(target IN LISTS benchmarks)
  set(links "cudf::cudf;ndsh_utilities")
  set(definitions "")
  if(CUDF_WITH_VORTEX AND target MATCHES "^NDSH_Q(0[1569]|10)_NVBENCH$")
    list(APPEND links NDSH_VORTEX_IO kvikio::kvikio)
    set(definitions "CUDF_NDSH_QUERY_EXTENSION=\"vortex_ndsh/q${CMAKE_MATCH_1}.inc\"")
  endif()
  assert_properties(${target} "${links}" LINK_LIBRARIES)
  assert_properties(${target} "${definitions}" COMPILE_DEFINITIONS)
  assert_properties(${target} "" INTERFACE_LINK_LIBRARIES INTERFACE_COMPILE_DEFINITIONS
                    INTERFACE_INCLUDE_DIRECTORIES)
  if(definitions)
    target_compile_definitions(${target} PRIVATE EXPECT_VORTEX=1)
  else()
    assert_properties(${target} "" INCLUDE_DIRECTORIES)
  endif()
endforeach()
assert_properties("cudf;cudf::cudf;ndsh_utilities;unrelated" ""
                  LINK_LIBRARIES INTERFACE_LINK_LIBRARIES INCLUDE_DIRECTORIES
                  INTERFACE_INCLUDE_DIRECTORIES COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
foreach(target marker IN ZIP_LISTS dependencies markers)
  assert_properties(${target} "" LINK_LIBRARIES INTERFACE_LINK_LIBRARIES COMPILE_DEFINITIONS)
  assert_properties(${target} "${marker}" INTERFACE_COMPILE_DEFINITIONS)
endforeach()

set(integration_targets NDSH_VORTEX_IO NDSH_VORTEX_BUILD_SMOKE NDSH_VORTEX_IO_TEST)
if(CUDF_WITH_VORTEX)
  assert_properties("${integration_targets}" TRUE EXCLUDE_FROM_ALL)
  assert_properties("${integration_targets}" "" COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
  assert_properties(NDSH_VORTEX_IO STATIC_LIBRARY TYPE)
  set(private_links Vortex::cpp_static nanoarrow::nanoarrow CUDA::cudart nvtx3::nvtx3-cpp)
  assert_properties(NDSH_VORTEX_IO "cudf::cudf;${private_links}" LINK_LIBRARIES)
  list(TRANSFORM private_links REPLACE "(.+)" "$<LINK_ONLY:\\1>")
  assert_properties(NDSH_VORTEX_IO "cudf::cudf;${private_links}" INTERFACE_LINK_LIBRARIES)
  assert_properties(NDSH_VORTEX_IO cxx_std_20 COMPILE_FEATURES INTERFACE_COMPILE_FEATURES)
  set(harness "${CMAKE_SOURCE_DIR}/fake-vortex/benchmarks/cudf-ndsh")
  assert_properties(NDSH_VORTEX_IO "${harness}/src/vortex_ndsh/vortex_io.cpp" SOURCES)
  assert_properties(NDSH_VORTEX_BUILD_SMOKE "${harness}/tests/vortex_build_smoke.cpp" SOURCES)
  assert_properties(NDSH_VORTEX_IO_TEST "${harness}/tests/vortex_io_test.cpp" SOURCES)
  assert_properties(NDSH_VORTEX_BUILD_SMOKE "cudf::cudf;Vortex::cpp_static;CUDA::cudart" LINK_LIBRARIES)
  assert_properties(NDSH_VORTEX_BUILD_SMOKE cxx_std_20 COMPILE_FEATURES)
  assert_properties(NDSH_VORTEX_IO_TEST
                    "NDSH_VORTEX_IO;nanoarrow::nanoarrow;CUDA::cudart;nvtx3::nvtx3-cpp" LINK_LIBRARIES)
  assert_properties("NDSH_VORTEX_BUILD_SMOKE;NDSH_VORTEX_IO_TEST" "" INTERFACE_LINK_LIBRARIES)
else()
  foreach(target IN ITEMS Vortex::cpp_static vortex_unwanted ${integration_targets})
    if(TARGET ${target})
      message(FATAL_ERROR "Disabled integration created Vortex target: ${target}")
    endif()
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
                # Queries pull in the adapter through ALL, but not standalone validation targets.
                self.build()
                self.assertTrue(archive.is_file(), "Default build did not build the adapter through queries")
                for query in QUERIES:
                    target = f"NDSH_Q{query:02}_NVBENCH"
                    self.assertTrue(
                        (self.binary / BENCHMARKS / target).is_file(), f"Default build omitted {target}"
                    )
                for target in ("NDSH_VORTEX_BUILD_SMOKE", "NDSH_VORTEX_IO_TEST"):
                    self.assertFalse(
                        (self.binary / BENCHMARKS / target).exists(),
                        f"Explicit target {target} was built implicitly",
                    )

    def test_source_validation_is_enabled_only(self):
        (self.fake / MODULE).unlink()
        not_directory = self.source / "not-a-checkout"
        self.write(not_directory, "not a directory\n")
        directory_module = self.source / "directory-module"
        (directory_module / MODULE).mkdir(parents=True)
        overrides = (None, "", self.fake, self.source / "missing-vortex", not_directory, directory_module)
        for setting in ("OFF", "ON"):
            enabled = setting == "ON"
            for index, override in enumerate(overrides):
                with self.subTest(setting=setting, override=override):
                    self.binary = self.source / f"checkout-{setting}-{index}"
                    output = self.configure(
                        f"-DCUDF_WITH_VORTEX={setting}",
                        # Disabled integration must ignore even an unsupported host.
                        f"-DCMAKE_SYSTEM_NAME={'Linux' if enabled else 'Generic'}",
                        override=override,
                        succeeds=not enabled,
                    )
                    self.assert_download(enabled and not override)
                    if enabled:
                        message = " ".join(output.split())
                        self.assertIn("CUDF_WITH_VORTEX requires Vortex sources containing", message)
                        self.assertIn("benchmarks/cudf-ndsh/vortex.cmake", message)
                        self.assertIn("-DFETCHCONTENT_SOURCE_DIR_VORTEX=/path/to/vortex", message)
                        self.assertIn(str(override or self.fake), output)

    def test_enabled_requires_linux_before_download(self):
        for index, override in enumerate((None, self.fake)):
            with self.subTest(override=override):
                self.binary = self.source / f"unsupported-{index}"
                output = self.configure(
                    "-DCUDF_WITH_VORTEX=ON",
                    "-DCMAKE_SYSTEM_NAME=Generic",
                    override=override,
                    succeeds=False,
                )
                self.assertIn("CUDF_WITH_VORTEX requires Linux and a CUDA toolkit", " ".join(output.split()))
                self.assert_download(False)


class BenchmarkSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = patch_postimages()
        paths = [ROOT / MODULE, *(ROOT / SOURCES).glob("*"), *(ROOT / HARNESS / "tests").glob("*.cpp")]
        for path in paths:
            if path.is_file():
                cls.sources[path.relative_to(ROOT)] = path.read_text(encoding="utf-8")

    def source(self, path, function=None):
        self.assertIn(path, self.sources, f"Patch or tracked harness is missing {path}")
        source = self.sources[path]
        if function is not None:
            # These free functions end at column zero; nested scopes stay indented.
            source = self.require_match(
                source, rf"(?ms)\b{re.escape(function)}\([^;{{]*\)\s*\{{(.*?)^\}}"
            ).group(1)
        return " ".join(source.split())

    def query_source(self, query):
        return self.source(NDSH / f"q{query:02}.cpp") + " " + self.source(SOURCES / f"q{query:02}.inc")

    def require_match(self, source, pattern):
        match = re.search(pattern, source)
        self.assertIsNotNone(match, f"Missing source pattern: {pattern}")
        return match

    def test_module_does_not_fetch_or_discover_a_toolchain(self):
        module = self.source(MODULE)
        self.assertNotRegex(module, r"FetchContent|CPMAddPackage|GIT_|file\(STRINGS|vortex_cuda\.h")
        self.assertNotRegex(module, r"find_package\(|enable_language\(|CMAKE_CUDA_COMPILER|nvcomp")

    def test_patch_contains_only_bootstrap_and_generic_seams(self):
        cmake_paths = {CUDF_CMAKE, BENCHMARKS / "CMakeLists.txt", LOADER}
        expected = cmake_paths | {
            NDSH / "utilities.cpp",
            NDSH / "utilities.hpp",
            *(NDSH / f"q{query:02}.cpp" for query in QUERIES),
        }
        paths = {
            Path(path)
            for path in re.findall(r"^diff --git a/\S+ b/(\S+)$", PATCH.read_text(encoding="utf-8"), re.MULTILINE)
        }
        self.assertEqual(paths, expected)
        for path in expected - cmake_paths:
            with self.subTest(path=path):
                source = self.source(path)
                self.assertNotRegex(source.lower(), r"vortex|ndsh_q\d+_local|check_q\d+_result|_reference\.hpp")

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
                self.assertIn('cold = ndsh::use_cold_cache(state.get_string("cache"))', setup)
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
                        r'if \(workload == "q1"\) \{ auto result = execute_q1\(.*?; '
                        r"ndsh::check_q1_result\(files.reference, \*result, stream\); "
                        r"CUDF_CUDA_TRY\(cudaDeviceSynchronize\(\)\); \}",
                    )
                nvtx = (
                    re.escape(f'cudf::benchmark::scoped_range timed_range{{"ndsh_q{query}_local_timed"}}; ')
                    if query in (1, 10)
                    else ""
                )
                if query in (1, 6):
                    owners = rf'auto result = workload == "read" \? read\(\) : execute_q{query}\(.*?\); {sync}'
                else:
                    owners = (
                        r'if \(workload == "read"\) \{ auto inputs = '
                        + re.escape(read_all)
                        + rf"; {sync} \}} else \{{ auto result = execute_q{query}\([^;]+; {sync} \}}"
                    )
                self.require_match(timed, f"^{nvtx}{owners}$")

    def test_read_only_tables_keep_projection_order_and_owners(self):
        source = self.source(SOURCES / "local_io.hpp")
        self.assertIn("std::vector<std::unique_ptr<table_with_names>> read_local_tables(", source)
        self.assertIn("std::unique_ptr<cudf::ast::operation> const no_predicate;", source)
        self.assertIn(
            "for (auto const& name : names) { "
            "tables.push_back(read(name, projections.at(name), no_predicate)); } return tables;",
            source,
        )

    def test_fixture_caches_are_lazy_and_construct_in_place(self):
        self.assertIn(
            "template <typename Files> Files const& local_fixture(double scale_factor) { "
            "static std::map<double, Files> fixtures; "
            "return fixtures.try_emplace(scale_factor, scale_factor).first->second; }",
            self.source(SOURCES / "local_io.hpp"),
        )
        for query in QUERIES:
            with self.subTest(query=query):
                source = self.query_source(query)
                self.assertIn(
                    f'auto const& files = ndsh::local_fixture<q{query}_files>(state.get_float64("scale_factor"));',
                    source,
                )
                setup = self.require_match(
                    source,
                    rf"explicit q{query}_files\(double scale_factor\) \{{ (.*?)"
                    r"(?:ndsh::vortex_io io\{|ndsh::make_reference_files<)",
                ).group(1)
                validation = f"check_q{query}_{'cases' if query in (9, 10) else 'boundaries'}();"
                self.assertIn(validation, setup)
                if query == 6:
                    self.assertLess(setup.index(validation), setup.index("check_q6_reference_boundaries();"))

    def test_generated_tables_are_checked_and_drained_before_release(self):
        self.require_match(
            self.source(SOURCES / "local_io.hpp", "make_reference_files"),
            r"^cuda::stream_ref const stream = cudf::get_default_stream\(\); vortex_io io\{stream.get\(\)\}; "
            r"\{ Builder builder; for_each_generated_table\(\s*"
            r"scale_factor, names, \[&\]\([^)]*generated\) \{ .*?"
            r"tables.write\(name, generated, io\); auto const projected = generated.select\(columns\); "
            r"builder.add_table\(name, projected, stream\); "
            r"check_file_projections\(tables, name, projected, columns, io\); "
            r"CUDF_CUDA_TRY\(cudaDeviceSynchronize\(\)\); \}\); reference = builder.finish\(\); \} "
            r"for \(bool use_vortex : \{false, true\}\) \{ validate\(\s*\[&\]\([^)]*\) \{ "
            r"return read_local_file\(tables.path\(name, use_vortex\), use_vortex, io, columns\); \}, stream\); \} "
            r"CUDF_CUDA_TRY\(cudaDeviceSynchronize\(\)\);$",
        )
        for query in (5, 9, 10):
            with self.subTest(query=query):
                self.require_match(
                    self.query_source(query),
                    rf"ndsh::make_reference_files<ndsh::q{query}_reference_builder>\(\s*"
                    rf"scale_factor, tables, reference, q{query}_tables, q{query}_projections, "
                    rf"{'false' if query == 5 else 'true'}, \[&\]\(auto&& read, cuda::stream_ref stream\) \{{",
                )

    def test_per_file_eviction_verifies_no_resident_pages(self):
        source = self.source(SOURCES / "local_io.hpp")
        self.assertIn('CUDF_EXPECTS(cache == "warm" || cache == "cold",', source)
        self.assertIn('return cache == "cold";', source)
        self.require_match(
            source,
            r"for \([^)]*: paths\) \{ kvikio::drop_file_page_cache\(path\); \} "
            r"for \([^)]*: paths\) \{ auto const resident_pages = kvikio::get_page_cache_info\(path\).first; "
            r"CUDF_EXPECTS\(resident_pages == 0,",
        )

    def test_device_import_decodes_dictionaries_and_opts_into_direct_io(self):
        local = self.source(SOURCES / "local_io.hpp")
        self.assertIn("bool direct_io = false", self.source(IO.with_suffix(".hpp")))
        self.assertIn("bool direct_io = false", local)
        self.assertIn("if (!use_vortex) { return read_parquet(cudf::io::source_info{path}, columns); }", local)
        self.require_match(local, r"io.read_vortex\(path, [^,]+, columns, direct_io\)")
        self.assertIn(
            "options.flags = VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES | (direct_io ? VX_CUDA_SCAN_FLAG_DIRECT_IO : 0);",
            self.source(IO),
        )
        self.assertIn("cudf::from_arrow_device(", self.source(IO))

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

    def test_generator_fixes_are_separate(self):
        cmake = BENCHMARKS / "CMakeLists.txt"
        self.assertNotIn("NDSH_DATA_GENERATOR_TEST", self.source(cmake))
        fixes = PATCH.with_name("generator-fixes.patch").read_text(encoding="utf-8")
        generator = cmake.parent / "common/ndsh_data_generator"
        paths = [Path(path) for path in re.findall(r"^diff --git a/\S+ b/(\S+)$", fixes, re.MULTILINE)]
        self.assertIn(generator / "ndsh_data_generator_test.cpp", paths)
        self.assertTrue(all(path == cmake or path.parent == generator for path in paths))
        self.assertIn("NDSH_DATA_GENERATOR_TEST", fixes)
        self.assertNotIn("q6_reference.hpp", fixes)
        self.assertNotIn("vortex", fixes.lower())

    def test_query_execution_is_not_retuned(self):
        for query in QUERIES:
            with self.subTest(query=query):
                source = self.query_source(query)
                self.assertNotRegex(source, r"std::(?:async|future)")
                if query != 9:
                    # Write while the original inputs/intermediates are still in scope.
                    self.assertIn("return consume(", source)
                    self.assertIn(f'[](auto const& result) {{ result->to_parquet("q{query}.parquet"); }}', source)
        # Native MEAN requests live in unchanged hunk gaps; do not replace or remove them.
        q1_patch = PATCH.read_text(encoding="utf-8").split("diff --git a/cpp/benchmarks/ndsh/q01.cpp ", 1)[1]
        q1_patch = q1_patch.split("diff --git ", 1)[0]
        self.assertNotRegex(q1_patch, r"(?m)^[+-].*cudf::aggregation::Kind::MEAN")

    def test_empty_fixtures_remain_checked_and_reported(self):
        shared = self.source(SOURCES / "local_io.hpp") + " " + self.source(SOURCES / "reference_io.hpp")
        self.assertEqual(
            self.source(SOURCES / "local_io.hpp", "check_file_projections"),
            "for (bool use_vortex : {false, true}) { "
            "auto input = read_local_file(files.path(name, use_vortex), use_vortex, io, columns); "
            "check_projection(expected, *input, columns); }",
        )
        for query in QUERIES:
            with self.subTest(query=query):
                source = self.query_source(query)
                self.assertIn(f"check_q{query}_result(", source)
                if query in (1, 6):
                    self.assertIn("check_projection(", source)
                self.assertIn(f'"ndsh/q{query}/matched_rows"', source)
                source += " " + self.source(SOURCES / f"q{query}_reference.hpp") + " " + shared
                self.assertNotIn("fixture has no", source)
                self.assertNotRegex(source, r"CUDF_EXPECTS\((?:reference\.)?matched > 0")
        self.assertIn('summary.set_string("value", "NULL")', self.query_source(6))

    def test_no_query_specific_kernels(self):
        for path, source in self.sources.items():
            with self.subTest(path=path):
                if path.suffix in (".cpp", ".hpp", ".inc", ".cu", ".cuh", ".cmake", ".txt"):
                    self.assertNotRegex(source, r"\b__global__\b|<<<")


if __name__ == "__main__":
    unittest.main(verbosity=2)
