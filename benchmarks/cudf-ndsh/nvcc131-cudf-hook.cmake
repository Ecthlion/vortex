# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# NVCC 13.1 can substitute a private std::variant alternative into bool_constant<true>.
# Establish the dispatcher first, without changing access checks or dependency targets.
if(NOT PROJECT_NAME STREQUAL "CUDF" OR
   NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  return()
endif()

function(_ndsh_apply_nvcc131_workaround)
  if(NOT TARGET cudf)
    message(FATAL_ERROR "NDS-H NVCC workaround: root cudf target was not created")
  endif()

  get_target_property(_cudf_source_dir cudf SOURCE_DIR)
  if(NOT _cudf_source_dir STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "NDS-H NVCC workaround: cudf is not a root-project target")
  endif()

  target_compile_options(
    cudf PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--pre-include=cudf/detail/utilities/dispatchers.hpp>"
  )
  message(STATUS "NDS-H: applied NVCC 13.1 workaround to root cudf CUDA sources only")
endfunction()

cmake_language(DEFER CALL _ndsh_apply_nvcc131_workaround)
