/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 */

// Included at file scope after q06.cpp: consumes execute_q6 and q6_columns,
// plus table_with_names and generate_lineitem from utilities.hpp.

#include "local_io.hpp"
#include "q6_reference.hpp"
#include "vortex_io.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>

#include <cmath>
#include <map>
#include <optional>

namespace {

template <typename Read>
std::unique_ptr<table_with_names> execute_q6(Read&& read)
{
  return ::execute_q6(read, true, [](auto& result) { return std::move(result); });
}

std::optional<double> revenue_value(table_with_names const& result)
{
  auto value = cudf::get_element(result.column("revenue"), 0);
  if (!value->is_valid()) { return std::nullopt; }
  return static_cast<cudf::numeric_scalar<double> const&>(*value).value();
}

void check_q6_result(ndsh::q6_reference_result const& expected, table_with_names const& result)
{
  auto const value = revenue_value(result);
  if (expected.matched == 0) {
    CUDF_EXPECTS(!value, "Q6 SUM over no matching rows must be null");
  } else {
    CUDF_EXPECTS(
      value && std::isfinite(*value) &&
        std::abs(*value - expected.revenue) <= 1e-10 * std::max(1.0, std::abs(expected.revenue)),
      "Q6 GPU revenue differs from CPU reference");
  }
}

void check_q6_reference_boundaries()
{
  using cudf::test::fixed_width_column_wrapper;
  cuda::stream_ref const stream = cudf::get_default_stream();
  // Exclude a matching sentinel; near-boundary doubles qualify only after the float32 cast.
  fixed_width_column_wrapper<double> price{{1000, 100, 100, 100, 100, 100, 100, 100, 100}};
  fixed_width_column_wrapper<double> discount{
    {0.06, 0.06, 0.05 - 1e-10, 0.07 + 1e-10, 0.06, 0.06, 0.049, 0.071, 0.06}};
  fixed_width_column_wrapper<cudf::timestamp_D, int32_t> shipdate{
    {8766, 8766, 8766, 9130, 8765, 9131, 8766, 8766, 8766}};
  fixed_width_column_wrapper<int8_t> quantity{{23, 23, 23, 23, 23, 23, 23, 23, 24}};
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
  auto const input = cudf::table_view{{price, discount, shipdate, quantity}};
  for (auto const& projected : cudf::slice(input, {1, 9, 4, 6, 1, 1}, stream)) {
    auto const expected =
      projected.num_rows() == 8 ? ndsh::q6_reference_result{3, 18.0} : ndsh::q6_reference_result{};
    auto const cpu = ndsh::q6_cpu_reference(projected, stream);
    CUDF_EXPECTS(
      cpu.matched == expected.matched && std::abs(cpu.revenue - expected.revenue) < 1e-12,
      "Q6 CPU reference boundary/slice regression");
    auto result = execute_q6([&](auto const& columns, auto const&) {
      return std::make_unique<table_with_names>(std::make_unique<cudf::table>(projected),
                                                columns);
    });
    check_q6_result(expected, *result);
  }
}

void check_q6_boundaries()
{
  cudf::test::fixed_width_column_wrapper<double> price{
    {100, 100, 100, 100, 100, 100, 100, 100, 100}};
  cudf::test::fixed_width_column_wrapper<double> discount{
    {0.06, 0.05, 0.07, 0.06, 0.049, 0.071, 0.06, 0.06, 0.06},
    {true, true, true, true, true, true, true, true, false}};
  // Epoch days: 1993-12-31, 1994-01-01, 1994-12-31, 1995-01-01.
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, int32_t> shipdate{
    {8765, 8766, 9130, 9131, 8766, 8766, 8766, 8766, 8766}};
  cudf::test::fixed_width_column_wrapper<double> quantity{{23, 23, 23, 23, 23, 23, 24, 23, 23}};
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
  auto result = execute_q6([&](auto const& columns, auto const&) {
    return std::make_unique<table_with_names>(
      std::make_unique<cudf::table>(cudf::table_view{{price, discount, shipdate, quantity}}),
      columns);
  });
  auto value = revenue_value(*result);
  CUDF_EXPECTS(value && std::abs(*value - 18.0) < 1e-8, "Q6 boundary/null regression");
}

struct q6_files {
  ndsh::local_table_files tables;
  ndsh::q6_reference_result reference;

  q6_files(double scale_factor, ndsh::vortex_io const& io)
  {
    auto generated = generate_lineitem(scale_factor);
    tables.write("lineitem", *generated, io);
    reference = ndsh::q6_cpu_reference(generated->select(q6_columns), cudf::get_default_stream());
    CUDF_EXPECTS(std::isfinite(reference.revenue), "Q6 CPU reference revenue must be finite");
    for (bool use_vortex : {false, true}) {
      auto input =
        ndsh::read_local_file(tables.path("lineitem", use_vortex), use_vortex, io, q6_columns);
      ndsh::check_projection(generated->select(q6_columns), *input, q6_columns);
      auto result = execute_q6([&](auto const&, auto const&) { return std::move(input); });
      check_q6_result(reference, *result);
    }
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
};

q6_files const& q6_fixture(double scale_factor)
{
  // Cache only files/host results, so all format/workload states use identical rows.
  static std::map<double, q6_files> fixtures;
  if (auto const files = fixtures.find(scale_factor); files != fixtures.end()) {
    return files->second;
  }
  check_q6_boundaries();
  check_q6_reference_boundaries();
  ndsh::vortex_io io{cuda::stream_ref{cudf::get_default_stream()}.get()};
  return fixtures.try_emplace(scale_factor, scale_factor, io).first->second;
}

void ndsh_q6_local(nvbench::state& state)
{
  auto const format   = state.get_string("format");
  auto const workload = state.get_string("workload");
  auto const cold     = ndsh::use_cold_cache(state.get_string("cache"));
  CUDF_EXPECTS(format == "parquet" || format == "vortex", "Unknown Q6 format");
  CUDF_EXPECTS(workload == "read" || workload == "q6", "Unknown Q6 workload");
  auto const use_vortex         = format == "vortex";
  auto const& files             = q6_fixture(state.get_float64("scale_factor"));
  cuda::stream_ref const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const memory = cudf::memory_stats_logger();
  ndsh::vortex_io io{stream.get()};
  auto read = [&] {
    return ndsh::read_local_file(
      files.tables.path("lineitem", use_vortex), use_vortex, io, q6_columns, cold);
  };
  if (!cold) {
    auto warmup = read();
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
  state.add_element_count(files.tables.rows, "Rows");
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
      if (cold) { ndsh::evict_file_pages(files.tables.paths(use_vortex)); }
      timer.start();
      {
        auto result = workload == "read"
                        ? read()
                        : execute_q6([&](auto const&, auto const&) { return read(); });
        CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
      }
      // Include cleanup queued on Vortex's independent producer streams, for both formats.
      CUDF_CUDA_TRY(cudaDeviceSynchronize());
      timer.stop();
    });
  state.add_buffer_size(files.tables.bytes(use_vortex), "file_size", "File size");
  state.add_buffer_size(memory.peak_memory_usage(), "rmm_peak", "RMM peak (excludes Vortex)");
  auto& summary = state.add_summary("ndsh/q6/revenue");
  summary.set_string("name", "Verified revenue");
  if (files.reference.matched == 0) {
    summary.set_string("value", "NULL");
  } else {
    summary.set_float64("value", files.reference.revenue);
  }
  auto& matched = state.add_summary("ndsh/q6/matched_rows");
  matched.set_string("name", "Q6 matched rows");
  matched.set_int64("value", files.reference.matched);
}

}  // namespace

NVBENCH_BENCH(ndsh_q6_local)
  .set_name("ndsh_q6_local")
  .add_float64_axis("scale_factor", {0.01, 0.1, 1, 10})
  .add_string_axis("format", {"parquet", "vortex"})
  .add_string_axis("workload", {"read", "q6"})
  .add_string_axis("cache", {"warm", "cold"});
