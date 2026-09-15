/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 */

// Included at file scope after q01.cpp: consumes execute_q1 and q1_columns,
// plus table_with_names, generate_lineitem, and apply_filter from utilities.hpp.

#include "local_io.hpp"
#include "q1_reference.hpp"
#include "vortex_io.hpp"

#include <benchmarks/common/nvtx_ranges.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>

#include <cstdint>
#include <map>

namespace {

template <typename Read>
std::unique_ptr<table_with_names> execute_q1(Read&& read, bool filter_shipdate)
{
  return ::execute_q1(read, filter_shipdate, [](auto& result) { return std::move(result); });
}

void check_q1_boundaries()
{
  cudf::test::strings_column_wrapper returnflag{"R", "A", "R", "A", "R", "X", "Z"};
  cudf::test::strings_column_wrapper linestatus{"O", "F", "F", "F", "O", "X", "Z"};
  cudf::test::fixed_width_column_wrapper<int8_t> quantity{{4, 2, 3, 6, 8, 50, 50}};
  cudf::test::fixed_width_column_wrapper<double> price{{100, 80, 40, 120, 200, 1000, 1000}};
  cudf::test::fixed_width_column_wrapper<double> discount{{0.25, 0.25, 0.5, 0.0, 0.5, 0.0, 0.0}};
  // Epoch days: 1998-09-01, inclusive cutoff 1998-09-02, 1998-09-03.
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, int32_t> shipdate{
    {10470, 10471, 10470, 10470, 10471, 10472, 10471}, {true, true, true, true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> orderkey{{1, 2, 3, 4, 5, 6, 7}};
  cudf::test::fixed_width_column_wrapper<double> tax{{0.1, 0.2, 0.1, 0.0, 0.2, 0.0, 0.0}};
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
  cudf::table_view const input{
    {returnflag, linestatus, quantity, price, discount, shipdate, orderkey, tax}};
  ndsh::q1_reference_result const expected{{{{"A", "F"}, {8, 200, 180, 192, 4, 100, 0.125, 2}},
                                            {{"R", "F"}, {3, 40, 20, 22, 3, 40, 0.5, 1}},
                                            {{"R", "O"}, {12, 300, 175, 202.5, 6, 150, 0.375, 2}}},
                                           5};
  cuda::stream_ref const stream = cudf::get_default_stream();
  auto const cpu        = ndsh::q1_cpu_reference(cudf::slice(input, {0, 6}).front(), stream);
  auto const sliced_cpu = ndsh::q1_cpu_reference(cudf::slice(input, {1, 6}).front(), stream);
  auto sliced_expected  = expected;
  sliced_expected.groups.at({"R", "O"}) = {8, 200, 100, 120, 8, 200, 0.5, 1};
  sliced_expected.matched               = 4;
  // Full/sliced input, rejected/null dates, and empty input; exercise both reader contracts.
  for (auto const& slice : cudf::slice(input, {0, 7, 1, 7, 5, 7, 0, 0})) {
    for (bool filter_shipdate : {true, false}) {
      auto result = execute_q1(
        [&](auto const& columns, auto const& predicate) {
          auto table =
            std::make_unique<table_with_names>(std::make_unique<cudf::table>(slice), columns);
          return filter_shipdate ? std::move(table) : apply_filter(table, *predicate);
        },
        filter_shipdate);
      ndsh::check_q1_result(slice.num_rows() == 7   ? expected
                            : slice.num_rows() == 6 ? sliced_expected
                                                    : ndsh::q1_reference_result{},
                            *result,
                            stream);
      if (slice.num_rows() >= 6) {
        ndsh::check_q1_result(slice.num_rows() == 7 ? cpu : sliced_cpu, *result, stream);
      }
    }
  }
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
}

struct q1_files {
  ndsh::local_table_files tables;
  ndsh::q1_reference_result reference;

  explicit q1_files(double scale_factor)
  {
    check_q1_boundaries();
    cuda::stream_ref const stream = cudf::get_default_stream();
    ndsh::vortex_io io{stream.get()};
    auto generated = generate_lineitem(scale_factor);
    CUDF_EXPECTS(generated->table().num_columns() == 16, "Q1 fixture requires full lineitem");
    tables.write("lineitem", *generated, io);
    reference = ndsh::q1_cpu_reference(generated->select(q1_columns), stream);

    for (bool use_vortex : {false, true}) {
      auto input =
        ndsh::read_local_file(tables.path("lineitem", use_vortex), use_vortex, io, q1_columns);
      ndsh::check_projection(generated->select(q1_columns), *input, q1_columns);
      auto result = execute_q1([&](auto const&, auto const&) { return std::move(input); }, true);
      ndsh::check_q1_result(reference, *result, stream);
    }
    generated.reset();
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
};

q1_files const& q1_fixture(double scale_factor)
{
  // Retain only files and host references, shared by every format/workload state for this SF.
  static std::map<double, q1_files> fixtures;
  return fixtures.try_emplace(scale_factor, scale_factor).first->second;
}

void ndsh_q1_local(nvbench::state& state)
{
  auto const format   = state.get_string("format");
  auto const workload = state.get_string("workload");
  auto const cold     = ndsh::use_cold_cache(state.get_string("cache"));
  CUDF_EXPECTS(format == "parquet" || format == "vortex", "Unknown Q1 format");
  CUDF_EXPECTS(workload == "read" || workload == "q1", "Unknown Q1 workload");
  auto const use_vortex         = format == "vortex";
  auto const& files             = q1_fixture(state.get_float64("scale_factor"));
  cuda::stream_ref const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const memory = cudf::memory_stats_logger();
  ndsh::vortex_io io{stream.get()};
  auto read = [&] {
    return ndsh::read_local_file(
      files.tables.path("lineitem", use_vortex), use_vortex, io, q1_columns, cold);
  };

  ndsh::warm_local_inputs(cold, read);
  if (workload == "q1") {
    auto result = execute_q1([&](auto const&, auto const&) { return read(); }, true);
    ndsh::check_q1_result(files.reference, *result, stream);
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
  ndsh::exec_local_benchmark(state, files.tables, use_vortex, cold, [&] {
    cudf::benchmark::scoped_range timed_range{"ndsh_q1_local_timed"};
    auto result = workload == "read"
                    ? read()
                    : execute_q1([&](auto const&, auto const&) { return read(); }, true);
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  });
  state.add_buffer_size(files.tables.bytes(use_vortex), "file_size", "File size");
  state.add_buffer_size(memory.peak_memory_usage(), "rmm_peak", "RMM peak (excludes Vortex)");
  auto& matched = state.add_summary("ndsh/q1/matched_rows");
  matched.set_string("name", "Q1 matched rows");
  matched.set_int64("value", files.reference.matched);
  auto& groups = state.add_summary("ndsh/q1/groups");
  groups.set_string("name", "Q1 groups");
  groups.set_int64("value", files.reference.groups.size());
}

}  // namespace

NVBENCH_BENCH(ndsh_q1_local)
  .set_name("ndsh_q1_local")
  .add_float64_axis("scale_factor", {0.01, 0.1, 1, 10})
  .add_string_axis("format", {"parquet", "vortex"})
  .add_string_axis("workload", {"read", "q1"})
  .add_string_axis("cache", {"warm", "cold"});
