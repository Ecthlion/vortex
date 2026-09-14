# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "CUDF_NDSH_WITH_VORTEX requires Linux and a CUDA toolkit")
endif()

# The harness and library come from this checkout. cuDF owns the compiler and toolkit configuration.
block()
  get_filename_component(vortex_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
  set(VORTEX_ENABLE_CUDA ON)
  set(VORTEX_BUILD_TESTS OFF)
  set(VORTEX_BUILD_EXAMPLES OFF)
  set(VORTEX_WARNINGS_AS_ERRORS OFF)
  add_subdirectory("${vortex_root}/lang/cpp"
                   "${CMAKE_CURRENT_BINARY_DIR}/_deps/vortex-build" EXCLUDE_FROM_ALL SYSTEM)

  add_library(NDSH_VORTEX_IO STATIC EXCLUDE_FROM_ALL
              "${CMAKE_CURRENT_LIST_DIR}/src/vortex_ndsh/vortex_io.cpp")
  target_compile_features(NDSH_VORTEX_IO PUBLIC cxx_std_20)
  target_include_directories(NDSH_VORTEX_IO PUBLIC "${CMAKE_CURRENT_LIST_DIR}/src")
  target_link_libraries(NDSH_VORTEX_IO
                        PUBLIC cudf::cudf
                        PRIVATE Vortex::cpp_static nanoarrow::nanoarrow CUDA::cudart nvtx3::nvtx3-cpp)

  foreach(query IN ITEMS 01 05 06 09 10)
    target_link_libraries(NDSH_Q${query}_NVBENCH PRIVATE NDSH_VORTEX_IO kvikio::kvikio)
    target_include_directories(NDSH_Q${query}_NVBENCH PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ndsh")
    target_compile_definitions(NDSH_Q${query}_NVBENCH
                               PRIVATE CUDF_NDSH_QUERY_EXTENSION="vortex_ndsh/q${query}.inc")
  endforeach()

  # Explicit targets: exercise real cuDF, C++ wrapper, and CUDA FFI symbols before building NDS-H.
  add_executable(NDSH_VORTEX_BUILD_SMOKE EXCLUDE_FROM_ALL
                 "${CMAKE_CURRENT_LIST_DIR}/tests/vortex_build_smoke.cpp")
  target_compile_features(NDSH_VORTEX_BUILD_SMOKE PRIVATE cxx_std_20)
  target_link_libraries(NDSH_VORTEX_BUILD_SMOKE PRIVATE cudf::cudf Vortex::cpp_static CUDA::cudart)

  add_executable(NDSH_VORTEX_IO_TEST EXCLUDE_FROM_ALL
                 "${CMAKE_CURRENT_LIST_DIR}/tests/vortex_io_test.cpp")
  target_link_libraries(NDSH_VORTEX_IO_TEST
                          PRIVATE NDSH_VORTEX_IO nanoarrow::nanoarrow CUDA::cudart nvtx3::nvtx3-cpp)
endblock()
