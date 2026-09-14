/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

#include <vortex/session.hpp>
#include <vortex_cuda.h>

#include <exception>
#include <iostream>
#include <memory>
#include <string_view>

int main()
{
  try {
    // Driver-backed stream synchronization requires an initialized current context.
    CUDF_CUDA_TRY(cudaSetDevice(0));
    auto const stream = cudf::get_default_stream();
    stream.sync();
    vortex::Session host_session;
    vx_error* error = nullptr;
    auto session    = std::unique_ptr<vx_session, decltype(&vx_session_free)>{
      vx_cuda_session_new(&error), vx_session_free};
    auto owned_error = std::unique_ptr<vx_error, decltype(&vx_error_free)>{error, vx_error_free};
    if (!session) {
      std::cerr << "Vortex CUDA session creation failed";
      if (owned_error) {
        auto const message = vx_error_message(owned_error.get());
        std::cerr << ": " << std::string_view{message.ptr, message.len};
      }
      std::cerr << '\n';
      return 1;
    }
    stream.sync();
    std::cout << "cuDF + Vortex C++/CUDA FFI initialization succeeded\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "cuDF/Vortex build smoke failed: " << error.what() << '\n';
    return 1;
  }
}
