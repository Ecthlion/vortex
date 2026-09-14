# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Offline CMake/source regression tests; run with python3 -B path/to/test_build_integration.py."""

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


def patch_hook():
    cmake = patch_postimages()[BENCHMARKS / "CMakeLists.txt"]
    hook = re.search(r"^option\(CUDF_NDSH_WITH_VORTEX\b.*?^endif\(\)", cmake, re.MULTILINE | re.DOTALL)
    if hook is None:
        raise AssertionError("Exported patch is missing the CUDF_NDSH_WITH_VORTEX bootstrap hook")
    return hook.group(0)


PARENT = r"""
function(assert_equal actual expected)
  if(NOT "${actual}" STREQUAL "${expected}")
    message(FATAL_ERROR "Expected '${expected}', got '${actual}' (${ARGN})")
  endif()
endfunction()

function(assert_property target property expected)
  get_target_property(actual ${target} ${property})
  if(actual STREQUAL "actual-NOTFOUND")
    set(actual "")
  endif()
  assert_equal("${actual}" "${expected}" "${target}.${property}")
endfunction()

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
     "^(VORTEX_|CUDAToolkit_|nvcomp_|CMAKE_.*(FLAGS|ARCHITECTURES|STANDARD|WARNING_AS_ERROR|COMPILER))")
foreach(variable IN LISTS parent_variables)
  set(before_${variable} "${${variable}}")
  foreach(property IN ITEMS VALUE TYPE HELPSTRING)
    get_property(before_${variable}_${property} CACHE ${variable} PROPERTY ${property})
  endforeach()
endforeach()

add_library(cudf INTERFACE)
add_library(cudf::cudf INTERFACE IMPORTED)
add_library(nanoarrow::nanoarrow INTERFACE IMPORTED)
add_library(CUDA::cudart INTERFACE IMPORTED)
target_compile_definitions(nanoarrow::nanoarrow INTERFACE NANOARROW_FAKE_LINK=1)
target_compile_definitions(CUDA::cudart INTERFACE CUDA_FAKE_LINK=1)
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
  set(includes "")
  if(CUDF_NDSH_WITH_VORTEX AND target MATCHES "^NDSH_Q(0[1569]|10)_NVBENCH$")
    list(APPEND links NDSH_VORTEX_IO)
    set(definitions "CUDF_NDSH_QUERY_EXTENSION=\"vortex_ndsh/q${CMAKE_MATCH_1}.inc\"")
    set(includes "${CMAKE_CURRENT_SOURCE_DIR}/ndsh")
  endif()
  assert_property(${target} LINK_LIBRARIES "${links}")
  assert_property(${target} COMPILE_DEFINITIONS "${definitions}")
  assert_property(${target} INCLUDE_DIRECTORIES "${includes}")
  assert_property(${target} INTERFACE_LINK_LIBRARIES "")
  assert_property(${target} INTERFACE_COMPILE_DEFINITIONS "")
  assert_property(${target} INTERFACE_INCLUDE_DIRECTORIES "")
  if(definitions)
    target_compile_definitions(${target} PRIVATE EXPECT_VORTEX=1)
  endif()
endforeach()
foreach(target IN ITEMS cudf cudf::cudf ndsh_utilities unrelated)
  foreach(property IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES INCLUDE_DIRECTORIES
                            INTERFACE_INCLUDE_DIRECTORIES COMPILE_DEFINITIONS INTERFACE_COMPILE_DEFINITIONS)
    assert_property(${target} ${property} "")
  endforeach()
endforeach()

foreach(target IN ITEMS nanoarrow::nanoarrow CUDA::cudart)
  foreach(property IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES COMPILE_DEFINITIONS)
    assert_property(${target} ${property} "")
  endforeach()
endforeach()
assert_property(nanoarrow::nanoarrow INTERFACE_COMPILE_DEFINITIONS NANOARROW_FAKE_LINK=1)
assert_property(CUDA::cudart INTERFACE_COMPILE_DEFINITIONS CUDA_FAKE_LINK=1)

set(integration_targets NDSH_VORTEX_IO NDSH_VORTEX_BUILD_SMOKE NDSH_VORTEX_IO_TEST)
if(CUDF_NDSH_WITH_VORTEX)
  foreach(target IN ITEMS Vortex::cpp_static ${integration_targets})
    if(NOT TARGET ${target})
      message(FATAL_ERROR "Missing enabled Vortex target: ${target}")
    endif()
  endforeach()
  foreach(target IN LISTS integration_targets)
    assert_property(${target} EXCLUDE_FROM_ALL TRUE)
    assert_property(${target} COMPILE_DEFINITIONS "")
    assert_property(${target} INTERFACE_COMPILE_DEFINITIONS "")
  endforeach()
  assert_property(NDSH_VORTEX_IO TYPE STATIC_LIBRARY)
  assert_property(NDSH_VORTEX_IO LINK_LIBRARIES "cudf::cudf;Vortex::cpp_static;nanoarrow::nanoarrow;CUDA::cudart")
  assert_property(NDSH_VORTEX_IO INTERFACE_LINK_LIBRARIES
                  "cudf::cudf;$<LINK_ONLY:Vortex::cpp_static>;$<LINK_ONLY:nanoarrow::nanoarrow>;$<LINK_ONLY:CUDA::cudart>")
  assert_property(NDSH_VORTEX_IO COMPILE_FEATURES cxx_std_20)
  assert_property(NDSH_VORTEX_IO INTERFACE_COMPILE_FEATURES cxx_std_20)
  set(harness "${FETCHCONTENT_SOURCE_DIR_VORTEX}/benchmarks/cudf-ndsh")
  assert_property(NDSH_VORTEX_IO SOURCES "${harness}/src/vortex_ndsh/vortex_io.cpp")
  assert_property(NDSH_VORTEX_BUILD_SMOKE SOURCES "${harness}/tests/vortex_build_smoke.cpp")
  assert_property(NDSH_VORTEX_IO_TEST SOURCES "${harness}/tests/vortex_io_test.cpp")
  assert_property(NDSH_VORTEX_IO INCLUDE_DIRECTORIES "${harness}/src")
  assert_property(NDSH_VORTEX_IO INTERFACE_INCLUDE_DIRECTORIES "${harness}/src")
  assert_property(NDSH_VORTEX_BUILD_SMOKE LINK_LIBRARIES "cudf::cudf;Vortex::cpp_static")
  assert_property(NDSH_VORTEX_BUILD_SMOKE INTERFACE_LINK_LIBRARIES "")
  assert_property(NDSH_VORTEX_BUILD_SMOKE COMPILE_FEATURES cxx_std_20)
  assert_property(NDSH_VORTEX_IO_TEST LINK_LIBRARIES "NDSH_VORTEX_IO;nanoarrow::nanoarrow;CUDA::cudart")
  assert_property(NDSH_VORTEX_IO_TEST INTERFACE_LINK_LIBRARIES "")
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
assert_equal("${CMAKE_CURRENT_SOURCE_DIR}" "${FETCHCONTENT_SOURCE_DIR_VORTEX}/lang/cpp")
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
#if defined(VORTEX_FAKE_LINK) || defined(NANOARROW_FAKE_LINK) || defined(CUDA_FAKE_LINK)
#error Private adapter dependency usage requirements leaked
#endif
#ifdef CUDF_NDSH_WITH_VORTEX
#error Obsolete benchmark Vortex definition
#endif
#ifdef EXPECT_VORTEX
#ifndef CUDF_NDSH_QUERY_EXTENSION
#error Missing private benchmark extension definition
#endif
#include CUDF_NDSH_QUERY_EXTENSION
#else
#ifdef CUDF_NDSH_QUERY_EXTENSION
#error Benchmark extension definition leaked
#endif
int main() { return 0; }
#endif
"""

IO_STUB = """
#ifndef PARENT_CXX_FLAG
#error Parent compiler flags were lost
#endif
#if !defined(VORTEX_FAKE_LINK) || !defined(NANOARROW_FAKE_LINK) || !defined(CUDA_FAKE_LINK)
#error Missing private adapter dependency usage requirements
#endif
#if defined(CUDF_NDSH_QUERY_EXTENSION) || defined(CUDF_NDSH_WITH_VORTEX) || defined(EXPECT_VORTEX)
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
        temporary = tempfile.TemporaryDirectory(prefix="ndsh-unittest-", dir=build)
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name)
        self.binary = self.source / "out"
        self.fake = self.source / "fake-vortex"
        harness = self.fake / HARNESS
        harness.mkdir(parents=True)
        shutil.copy2(ROOT / MODULE, self.fake / MODULE)
        for directory in ("src", "tests"):
            shutil.copytree(ROOT / HARNESS / directory, harness / directory)
        for path in (MODULE, SMOKE, IO, IO.with_suffix(".hpp"), IO_TEST):
            self.assertTrue((self.fake / path).is_file(), f"Tracked harness is missing {path}")
        # Compile tests cover wiring with stubs; the real sources need separate GPU validation.
        # Leave smoke/I/O test sources intact: neither may build implicitly without CUDA/Rust.
        self.write(self.fake / IO, IO_STUB)
        for query in QUERIES:
            extension = self.fake / SOURCES / f"q{query:02}.inc"
            self.assertTrue(extension.is_file(), f"Tracked harness is missing {extension.name}")
            self.write(
                extension,
                '#include "utilities.hpp"\n'
                "int ndsh_vortex_io_stub();\n"
                "int main() { return ndsh_vortex_io_stub(); }\n",
            )
        self.write(NDSH / "utilities.hpp", "// Requires the query's private upstream include directory.\n")
        self.write(
            "CMakeLists.txt",
            "cmake_minimum_required(VERSION 3.25)\n"
            "project(ndsh_integration_test LANGUAGES CXX)\n"
            "add_subdirectory(cpp/benchmarks)\n",
        )
        self.write(BENCHMARKS / "CMakeLists.txt", PARENT)
        self.write(BENCHMARKS / "ndsh-hook.cmake", patch_hook())
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
        if succeeds:
            self.assertEqual(result.returncode, 0, f"{command}\n{result.stdout}")
        else:
            self.assertNotEqual(result.returncode, 0, f"{command} unexpectedly succeeded\n{result.stdout}")
        return result.stdout

    def configure(self, *options, local_source=True, succeeds=True):
        source_override = (f"-DFETCHCONTENT_SOURCE_DIR_VORTEX={self.fake}",) if local_source else ()
        return self.run_command(
            "cmake",
            "-S",
            str(self.source),
            "-B",
            str(self.binary),
            # Exercise the module's Linux gate without enabling CUDA, even on a non-Linux host.
            "-DCMAKE_SYSTEM_NAME=Linux",
            "-DCMAKE_BUILD_TYPE=Debug",
            *source_override,
            *options,
            succeeds=succeeds,
        )

    def build(self, *targets):
        options = ("--target", *targets) if targets else ()
        self.run_command("cmake", "--build", str(self.binary), "--parallel", "2", *options)

    def test_disabled_by_default_and_explicitly(self):
        self.write(self.fake / MODULE, 'message(FATAL_ERROR "Disabled Vortex module was included")')
        for setting in (None, "OFF"):
            with self.subTest(setting=setting):
                self.binary = self.source / (setting or "default")
                self.configure(*(() if setting is None else (f"-DCUDF_NDSH_WITH_VORTEX={setting}",)))
                cache = (self.binary / "CMakeCache.txt").read_text(encoding="utf-8")
                self.assertIn("CUDF_NDSH_WITH_VORTEX:BOOL=OFF", cache)
                self.build()

    def test_enabled_private_links_and_excluded_targets(self):
        self.configure("-DCUDF_NDSH_WITH_VORTEX=ON")
        archive = self.binary / BENCHMARKS / "libNDSH_VORTEX_IO.a"
        self.build("unrelated")
        self.assertFalse(archive.exists(), "Unrelated target built the adapter")
        self.build("NDSH_HELPER")
        self.assertFalse(archive.exists(), "Unselected benchmarks built the adapter")
        # cuDF queries participate in ALL and pull in the adapter, but not standalone validation targets.
        self.build()
        self.assertTrue(archive.is_file(), "Default build did not build the static adapter through queries")
        for query in QUERIES:
            target = f"NDSH_Q{query:02}_NVBENCH"
            self.assertTrue((self.binary / BENCHMARKS / target).is_file(), f"Default build omitted {target}")
        for target in ("NDSH_VORTEX_BUILD_SMOKE", "NDSH_VORTEX_IO_TEST"):
            self.assertFalse(
                (self.binary / BENCHMARKS / target).exists(), f"Explicit target {target} was built implicitly"
            )

    def test_disabled_ignores_unusable_local_source(self):
        (self.fake / MODULE).unlink()
        not_directory = self.source / "not-a-checkout"
        self.write(not_directory, "not a directory\n")
        overrides = (None, "", self.fake, self.source / "missing-vortex", not_directory)
        for setting in (None, "OFF"):
            for index, override in enumerate(overrides):
                with self.subTest(setting=setting, override=override):
                    self.binary = self.source / f"disabled-{setting}-{index}"
                    self.configure(
                        *(() if setting is None else (f"-DCUDF_NDSH_WITH_VORTEX={setting}",)),
                        *(() if override is None else (f"-DFETCHCONTENT_SOURCE_DIR_VORTEX={override}",)),
                        # The enabled-only Linux/CUDA requirement must not apply to OFF.
                        "-DCMAKE_SYSTEM_NAME=Generic",
                        local_source=False,
                    )

    def test_enabled_requires_local_module_offline(self):
        (self.fake / MODULE).unlink()
        not_directory = self.source / "not-a-checkout"
        self.write(not_directory, "not a directory\n")
        overrides = (None, "", self.fake, self.source / "missing-vortex", not_directory)
        for index, override in enumerate(overrides):
            with self.subTest(override=override):
                self.binary = self.source / f"missing-{index}"
                output = self.configure(
                    "-DCUDF_NDSH_WITH_VORTEX=ON",
                    *(() if override is None else (f"-DFETCHCONTENT_SOURCE_DIR_VORTEX={override}",)),
                    local_source=False,
                    succeeds=False,
                )
                message = " ".join(output.split())
                self.assertIn("CUDF_NDSH_WITH_VORTEX=ON requires a Vortex checkout", message)
                self.assertIn("benchmarks/cudf-ndsh/vortex.cmake", message)
                self.assertIn("-DFETCHCONTENT_SOURCE_DIR_VORTEX=/path/to/vortex", message)
                if override:
                    self.assertIn(str(override), output)

    def test_enabled_rejects_directory_in_place_of_module(self):
        (self.fake / MODULE).unlink()
        (self.fake / MODULE).mkdir()
        output = self.configure("-DCUDF_NDSH_WITH_VORTEX=ON", succeeds=False)
        self.assertIn("requires a Vortex checkout", " ".join(output.split()))

    def test_enabled_requires_linux(self):
        output = self.configure("-DCUDF_NDSH_WITH_VORTEX=ON", "-DCMAKE_SYSTEM_NAME=Generic", succeeds=False)
        self.assertIn("requires Linux and a CUDA toolkit", output)


class BenchmarkSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.patch_sources = patch_postimages()
        cls.sources = dict(cls.patch_sources)
        paths = [ROOT / MODULE, *(ROOT / SOURCES).glob("*"), *(ROOT / HARNESS / "tests").glob("*.cpp")]
        for path in paths:
            if path.is_file():
                cls.sources[path.relative_to(ROOT)] = path.read_text(encoding="utf-8")

    def source(self, path):
        self.assertIn(path, self.sources, f"Patch or tracked harness is missing {path}")
        return " ".join(self.sources[path].split())

    def query_source(self, query):
        return self.source(NDSH / f"q{query:02}.cpp") + " " + self.source(SOURCES / f"q{query:02}.inc")

    def require_match(self, source, pattern):
        match = re.search(pattern, source)
        self.assertIsNotNone(match, f"Missing source pattern: {pattern}")
        return match

    def test_bootstrap_uses_the_module_checkout(self):
        hook = patch_hook()
        self.assertRegex(hook, r'^option\(CUDF_NDSH_WITH_VORTEX "[^"]+" OFF\)\nif\(CUDF_NDSH_WITH_VORTEX\)')
        self.assertIn('include("${FETCHCONTENT_SOURCE_DIR_VORTEX}/benchmarks/cudf-ndsh/vortex.cmake")', hook)
        module = self.source(MODULE)
        self.assertIn('get_filename_component(vortex_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)', module)
        self.assertIn(
            'add_subdirectory("${vortex_root}/lang/cpp" '
            '"${CMAKE_CURRENT_BINARY_DIR}/_deps/vortex-build" EXCLUDE_FROM_ALL SYSTEM)',
            module,
        )
        self.assertNotRegex(module, r"FetchContent|GIT_|file\(STRINGS|vortex_cuda\.h")
        self.assertNotRegex(module, r"find_package\(|enable_language\(|CMAKE_CUDA_COMPILER|nvcomp")

    def test_patch_contains_only_generic_seams(self):
        expected = {
            BENCHMARKS / "CMakeLists.txt",
            NDSH / "utilities.cpp",
            NDSH / "utilities.hpp",
            *(NDSH / f"q{query:02}.cpp" for query in QUERIES),
        }
        paths = {
            Path(path)
            for path in re.findall(r"^diff --git a/\S+ b/(\S+)$", PATCH.read_text(encoding="utf-8"), re.MULTILINE)
        }
        self.assertEqual(paths, expected)
        for path in expected - {BENCHMARKS / "CMakeLists.txt"}:
            with self.subTest(path=path):
                source = self.source(path)
                self.assertNotRegex(source.lower(), r"vortex|ndsh_q\d+_local|check_q\d+_result|_reference\.hpp")
        for name in ("generate_lineitem", "for_each_generated_table"):
            self.assertIn(name, self.source(NDSH / "utilities.hpp"))
            self.assertIn(name, self.source(NDSH / "utilities.cpp"))

    def test_query_extension_hooks_and_shared_projections(self):
        for query in QUERIES:
            with self.subTest(query=query):
                path = NDSH / f"q{query:02}.cpp"
                source = self.source(path)
                self.assertTrue(
                    source.endswith(
                        "#ifdef CUDF_NDSH_QUERY_EXTENSION #include CUDF_NDSH_QUERY_EXTENSION #endif"
                    ),
                    f"{path} must end with the generic extension hook",
                )
                self.assertEqual(source.count("#include CUDF_NDSH_QUERY_EXTENSION"), 1)
                projection = f"q{query}_{'columns' if query in (1, 6) else 'projections'}"
                extension = self.source(SOURCES / f"q{query:02}.inc")
                self.assertIn(projection, source)
                self.assertIn(projection, extension)
                self.assertNotRegex(extension, rf"\bconst\s+{projection}\s*\{{")
                if query != 9:
                    self.assertIn(f"execute_q{query}(", source)
                    self.assertIn("return consume(", source)

    def test_harness_quoted_includes_resolve(self):
        for path, source in self.sources.items():
            if not path.is_relative_to(HARNESS) or path.suffix not in (".cpp", ".hpp", ".inc"):
                continue
            for header in re.findall(r'^#include "([^"]+)"', source, re.MULTILINE):
                with self.subTest(path=path, header=header):
                    candidates = (path.parent / header, SOURCES.parent / header)
                    if path.parent == SOURCES:
                        candidates += (NDSH / header,)
                    self.assertTrue(
                        any(candidate in self.sources for candidate in candidates),
                        f"{path}: {header} is not reachable through the configured include directories",
                    )

    def test_local_benchmark_names_and_axes(self):
        for query in QUERIES:
            with self.subTest(query=query):
                source = self.query_source(query)
                name = f"ndsh_q{query}_local"
                registration = self.require_match(source, rf"NVBENCH_BENCH\({name}\)([^;]+);").group(1)
                self.assertIn(f'.set_name("{name}")', registration)
                scales = self.require_match(registration, r'add_float64_axis\("scale_factor", \{([^}]+)\}\)')
                self.assertIn(10.0, [float(value) for value in scales.group(1).split(",")])
                axes = {
                    axis: set(re.findall(r'"([^"]+)"', values))
                    for axis, values in re.findall(r'add_string_axis\("([^"]+)", \{([^}]+)\}\)', registration)
                }
                self.assertEqual(axes.get("cache"), {"warm", "cold"})
                self.assertEqual(axes.get("format"), {"parquet", "vortex"})
                self.assertEqual(axes.get("workload"), {"read", f"q{query}"})
                if query == 9:
                    # This is the existing generic cuDF transform mode, not a custom query kernel.
                    self.assertEqual(axes.get("engine"), {"binaryop", "ast", "transform"})

    def test_cache_preparation_precedes_each_manual_timer(self):
        for query in QUERIES:
            with self.subTest(query=query):
                source = self.query_source(query)
                local = self.require_match(source, rf"void ndsh_q{query}_local\([^)]*\) \{{(.*)").group(1)
                self.require_match(local, r'cold = ndsh::use_cold_cache\(state.get_string\("cache"\)\)')
                self.require_match(local, r"ndsh::read_local_file\([^;]+, cold\)")
                self.require_match(local, r"if \(!cold\) \{ auto warmup =")
                execution = self.require_match(local, r"state.exec\((.*?)timer.stop\(\);").group(1)
                self.assertIn("nvbench::exec_tag::sync | nvbench::exec_tag::timer", execution)
                eviction = self.require_match(
                    execution, r"if \(cold\) \{ ndsh::evict_file_pages\((.*?)\); \} timer.start\(\);"
                )
                self.assertEqual(eviction.group(1), "files.tables.paths(use_vortex)")
                timed = execution.split("timer.start();", 1)[1]
                self.assertNotIn("evict_file_pages", timed)
                self.assertIn(f"execute_q{query}(", timed)
                self.assertIn("cudaDeviceSynchronize()", timed)

    def test_read_only_tables_keep_projection_order_and_owners(self):
        source = self.source(SOURCES / "local_io.hpp")
        self.assertIn("std::vector<std::unique_ptr<table_with_names>> read_local_tables(", source)
        self.assertIn("std::unique_ptr<cudf::ast::operation> const no_predicate;", source)
        self.require_match(
            source,
            r"for \(auto const& name : names\) \{ "
            r"tables.push_back\(read\(name, projections.at\(name\), no_predicate\)\); \} return tables;",
        )
        for query in (5, 10):
            with self.subTest(query=query):
                source = self.query_source(query)
                read = f"ndsh::read_local_tables(q{query}_tables, q{query}_projections, read)"
                self.assertIn(f"auto warmup = {read};", source)
                self.require_match(
                    source,
                    r'if \(workload == "read"\) \{ auto inputs = '
                    + re.escape(read)
                    + r"; CUDF_CUDA_TRY\(cudaStreamSynchronize\(stream.get\(\)\)\); \} else",
                )

    def test_fixture_caches_are_lazy_and_construct_in_place(self):
        for query in QUERIES:
            with self.subTest(query=query):
                source = self.query_source(query)
                setup = self.require_match(
                    source,
                    rf"static std::map<double, q{query}_files> fixtures; "
                    r"if \(auto const files = fixtures.find\(scale_factor\); files != fixtures.end\(\)\) "
                    r"\{ return files->second; \} (.*?) "
                    r"return fixtures.try_emplace\(scale_factor, scale_factor, io\).first->second;",
                ).group(1)
                validation = f"check_q{query}_{'cases' if query in (9, 10) else 'boundaries'}();"
                session = self.require_match(setup, r"ndsh::vortex_io io\{[^;]+\};$").start()
                self.assertLess(setup.index(validation), session)
                if query == 6:
                    self.assertLess(setup.index(validation), setup.index("check_q6_reference_boundaries();"))
                    self.assertLess(setup.index("check_q6_reference_boundaries();"), session)
                # No placeholder or temporary fixture: a failed emplace leaves the key absent.
                self.assertNotIn("fixtures[", setup)
                self.assertNotIn(f"std::make_unique<q{query}_files>", source)

    def test_fixture_projection_checks_keep_both_formats_and_cleanup(self):
        source = self.source(SOURCES / "local_io.hpp")
        self.require_match(
            source,
            r"inline void check_file_projections\([^)]*\) \{ "
            r"for \(bool use_vortex : \{false, true\}\) \{ "
            r"auto input = read_local_file\(files.path\(name, use_vortex\), use_vortex, io, columns\); "
            r"check_projection\(expected, \*input, columns\); \} \}",
        )
        for query in (5, 9, 10):
            with self.subTest(query=query):
                source = self.query_source(query)
                self.require_match(
                    source,
                    r"builder.add_table\(name, projected, stream\); "
                    r"ndsh::check_file_projections\(tables, name, projected, columns, io\); "
                    r"CUDF_CUDA_TRY\(cudaDeviceSynchronize\(\)\); \}\);",
                )

    def test_per_file_eviction_verifies_no_resident_pages(self):
        source = self.source(SOURCES / "local_io.hpp")
        self.require_match(source, r'CUDF_EXPECTS\(cache == "warm" \|\| cache == "cold",')
        self.assertIn('return cache == "cold";', source)
        self.require_match(
            source,
            r"for \([^)]*: paths\) \{ kvikio::drop_file_page_cache\(path\); \} "
            r"for \([^)]*: paths\) \{ auto const resident_pages = kvikio::get_page_cache_info\(path\).first; "
            r"CUDF_EXPECTS\(resident_pages == 0,",
        )

    def test_direct_io_is_opt_in_for_vortex_only(self):
        local = self.source(SOURCES / "local_io.hpp")
        self.assertIn("bool direct_io = false", self.source(IO.with_suffix(".hpp")))
        self.assertIn("bool direct_io = false", local)
        self.require_match(
            local, r"if \(!use_vortex\) \{ return read_parquet\(cudf::io::source_info\{path\}, columns\); \}"
        )
        self.require_match(local, r"io.read_vortex\(path, [^,]+, columns, direct_io\)")
        self.require_match(
            self.source(IO),
            r"options.flags = VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES "
            r"\| \(direct_io \? VX_CUDA_SCAN_FLAG_DIRECT_IO : 0\);",
        )

    def test_plain_dictionary_decoded_import_is_preserved(self):
        source = self.source(IO)
        self.require_match(source, r"options.flags = VX_CUDA_SCAN_FLAG_DECODE_DICTIONARIES\b")
        self.assertIn("vx_cuda_scan_path_arrow_device_stream_projected(", source)
        self.assertIn("cudf::from_arrow_device(", source)
        self.require_match(
            source, r"if \(column.type\(\).id\(\) == cudf::type_id::DICTIONARY32\) \{ throw std::runtime_error\("
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
        self.require_match(
            source, r"std::vector<device_batch> batches; stream_drain drain\{impl_->stream\};"
        )
        completion = self.require_match(
            source, r'nvtx3::scoped_range range\{"vortex.consumer_sync"\}; drain.wait\(\);'
        )
        release = self.require_match(
            source, r'nvtx3::scoped_range range\{"vortex.release_batches"\}; batches.clear\(\);'
        )
        self.assertLess(source.index('"vortex.materialize"'), completion.start())
        self.assertLess(completion.end(), release.start())

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
        for query in QUERIES:
            with self.subTest(query=query):
                source = self.query_source(query)
                self.assertIn(f"check_q{query}_result(", source)
                projection_check = "check_projection" if query in (1, 6) else "check_file_projections"
                self.assertIn(f"{projection_check}(", source)
                self.assertIn(f'"ndsh/q{query}/matched_rows"', source)
                self.assertNotIn("fixture has no", source)
                self.assertNotRegex(source, r"CUDF_EXPECTS\((?:reference\.)?matched > 0")
        q6 = self.query_source(6)
        self.assertIn('CUDF_EXPECTS(!value, "Q6 SUM over no matching rows must be null")', q6)
        self.assertIn('summary.set_string("value", "NULL")', q6)
        self.assertIn("check_q6_reference_boundaries();", q6)
        q9_reference = self.source(SOURCES / "q9_reference.hpp")
        self.assertIn("std::unordered_multimap<uint64_t, double> supply_costs_", q9_reference)
        self.assertIn("supply_costs_.equal_range(", q9_reference)
        self.assertIn("duplicate_expected.matched = 7;", self.query_source(9))

    def test_no_query_specific_kernels(self):
        for path, source in self.sources.items():
            with self.subTest(path=path):
                if path.suffix in (".cpp", ".hpp", ".inc", ".cu", ".cuh", ".cmake", ".txt"):
                    self.assertNotRegex(source, r"\b__global__\b|<<<")


if __name__ == "__main__":
    unittest.main(verbosity=2)
