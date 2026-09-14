/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 */

// Included at file scope after q09.cpp: consumes engine_type, engine_from_string,
// load_data, compute_profit, q9_tables, and q9_projections, plus table_with_names
// and for_each_generated_table from utilities.hpp.

#include "local_io.hpp"
#include "q9_reference.hpp"
#include "vortex_io.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>

#include <cmath>

namespace {

template <typename Read>
std::unique_ptr<table_with_names> execute_q9(
  engine_type engine,
  Read&& read,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto const data = load_data(std::forward<Read>(read));
  return compute_profit(engine, data, stream, mr);
}

void check_q9_cases()
{
  cudf::test::fixed_width_column_wrapper<int8_t> nation_key{{10, 20}};
  cudf::test::strings_column_wrapper nation_name{"ALPHA", "ZULU"};
  cudf::test::fixed_width_column_wrapper<int32_t> supplier_key{{101, 102, 103}};
  cudf::test::fixed_width_column_wrapper<int8_t> supplier_nation{{10, 20, 99}};
  // The extra rows duplicate a matching key, a supplier with no nation, and a missing supplier.
  cudf::test::fixed_width_column_wrapper<int32_t> partsupp_supplier{
    {101, 102, 102, 101, 103, 101, 103, 999, 999}};
  cudf::test::fixed_width_column_wrapper<int32_t> partsupp_part{{1, 3, 2, 2, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<double> supply_cost{{10, 5, 20, 30, 40, 20, 50, 60, 70}};
  cudf::test::fixed_width_column_wrapper<int32_t> order_key{{1, 2, 3, 4}};
  // Epoch days: 1994-01-01, 1995-01-01, 1994-12-31, 1996-01-01.
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, int32_t> order_date{
    {8766, 9131, 9130, 9496}};
  cudf::test::fixed_width_column_wrapper<int32_t> part_key{{1, 2, 3, 4}};
  cudf::test::strings_column_wrapper part_name{"forest green", "GREEN", "lightgreen", "blue"};
  cudf::test::fixed_width_column_wrapper<int32_t> line_supplier{
    {101, 101, 102, 101, 102, 101, 101, 999, 101, 103, 102, 101}};
  cudf::test::fixed_width_column_wrapper<int32_t> line_part{{1, 1, 3, 1, 2, 2, 3, 1, 1, 1, 1, 4}};
  cudf::test::fixed_width_column_wrapper<int32_t> line_order{
    {1, 1, 2, 4, 2, 3, 3, 1, 999, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<double> price{
    {100, 50, 200, 100, 500, 500, 500, 500, 500, 500, 500, 500}};
  cudf::test::fixed_width_column_wrapper<double> discount{
    {0.1, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  cudf::test::fixed_width_column_wrapper<int8_t> quantity{{2, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
  CUDF_CUDA_TRY(cudaDeviceSynchronize());

  std::map<std::string, cudf::table_view> const input{
    {"nation", cudf::table_view{{nation_key, nation_name}}},
    {"supplier", cudf::table_view{{supplier_key, supplier_nation}}},
    {"partsupp", cudf::table_view{{partsupp_supplier, partsupp_part, supply_cost}}},
    {"orders", cudf::table_view{{order_key, order_date}}},
    {"part", cudf::table_view{{part_key, part_name}}},
    {"lineitem",
     cudf::table_view{{line_supplier, line_part, line_order, price, discount, quantity}}}};
  ndsh::q9_reference_result expected;
  expected.sum_profit[{"ALPHA", 1996}] = 90.0;
  expected.sum_profit[{"ALPHA", 1994}] = 110.0;
  expected.sum_profit[{"ZULU", 1995}]  = 130.0;
  expected.matched                     = 4;
  ndsh::q9_reference_result duplicate_expected;
  duplicate_expected.sum_profit[{"ALPHA", 1996}] = 170.0;
  duplicate_expected.sum_profit[{"ALPHA", 1994}] = 190.0;
  duplicate_expected.sum_profit[{"ZULU", 1995}]  = 130.0;
  duplicate_expected.matched                     = 7;
  cuda::stream_ref const stream                  = cudf::get_default_stream();
  for (auto const& partsupp : cudf::slice(input.at("partsupp"), {0, 5, 0, 9}, stream)) {
    // Full input, only rows rejected by filters/joins, and no rows, with and without duplicates.
    for (auto const& lines : cudf::slice(input.at("lineitem"), {0, 12, 4, 12, 0, 0}, stream)) {
      auto const want       = lines.num_rows() == 12
                                ? (partsupp.num_rows() == 5 ? expected : duplicate_expected)
                                : ndsh::q9_reference_result{};
      auto tables           = input;
      tables.at("partsupp") = partsupp;
      tables.at("lineitem") = lines;
      ndsh::q9_reference_builder builder;
      for (auto const& name : {"nation", "supplier", "partsupp", "orders", "part", "lineitem"}) {
        builder.add_table(name, tables.at(name), stream);
      }
      auto const cpu = builder.finish();
      CUDF_EXPECTS(cpu.matched == want.matched && cpu.sum_profit.size() == want.sum_profit.size(),
                   "Q9 CPU reference row/group regression");
      for (auto const& [key, value] : want.sum_profit) {
        auto const actual = cpu.sum_profit.find(key);
        CUDF_EXPECTS(actual != cpu.sum_profit.end() &&
                       std::abs(actual->second - value) <= 1e-10 * std::max(1.0, std::abs(value)),
                     "Q9 CPU reference amount/year regression");
      }
      for (auto const engine : {engine_type::BINARYOP, engine_type::AST, engine_type::TRANSFORM}) {
        auto result =
          execute_q9(engine, [&](std::string const& name, std::vector<std::string> const& columns) {
            CUDF_EXPECTS(columns == q9_projections.at(name), "Q9 projection mismatch");
            return std::make_unique<table_with_names>(
              std::make_unique<cudf::table>(tables.at(name)), columns);
          });
        ndsh::check_q9_result(want, *result, stream);
      }
    }
  }
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
}

struct q9_files {
  ndsh::local_table_files tables;
  ndsh::q9_reference_result reference;

  q9_files(double scale_factor, ndsh::vortex_io const& io)
  {
    cuda::stream_ref const stream = cudf::get_default_stream();
    {
      ndsh::q9_reference_builder builder;
      for_each_generated_table(
        scale_factor, q9_tables, [&](std::string const& name, table_with_names const& generated) {
          auto const& columns = q9_projections.at(name);
          CUDF_EXPECTS(
            generated.table().num_columns() > static_cast<cudf::size_type>(columns.size()),
            "Q9 fixture requires full generated table: " + name);
          tables.write(name, generated, io);
          auto const projected = generated.select(columns);
          builder.add_table(name, projected, stream);
          ndsh::check_file_projections(tables, name, projected, columns, io);
          CUDF_CUDA_TRY(cudaDeviceSynchronize());
        });
      reference = builder.finish();
    }

    for (bool use_vortex : {false, true}) {
      for (auto const engine : {engine_type::BINARYOP, engine_type::AST, engine_type::TRANSFORM}) {
        auto result =
          execute_q9(engine, [&](std::string const& name, std::vector<std::string> const& columns) {
            return ndsh::read_local_file(tables.path(name, use_vortex), use_vortex, io, columns);
          });
        ndsh::check_q9_result(reference, *result, stream);
      }
    }
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
};

q9_files const& q9_fixture(double scale_factor)
{
  // Share only full-table files and the CPU result across format/workload/engine states at this SF.
  static std::map<double, q9_files> fixtures;
  if (auto const files = fixtures.find(scale_factor); files != fixtures.end()) {
    return files->second;
  }
  check_q9_cases();
  cuda::stream_ref const stream = cudf::get_default_stream();
  ndsh::vortex_io io{stream.get()};
  return fixtures.try_emplace(scale_factor, scale_factor, io).first->second;
}

void ndsh_q9_local(nvbench::state& state)
{
  auto const format   = state.get_string("format");
  auto const workload = state.get_string("workload");
  auto const engine   = engine_from_string(state.get_string("engine"));
  auto const cold     = ndsh::use_cold_cache(state.get_string("cache"));
  CUDF_EXPECTS(format == "parquet" || format == "vortex", "Unknown Q9 format");
  CUDF_EXPECTS(workload == "read" || workload == "q9", "Unknown Q9 workload");
  auto const use_vortex         = format == "vortex";
  auto const& files             = q9_fixture(state.get_float64("scale_factor"));
  cuda::stream_ref const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const memory = cudf::memory_stats_logger();
  ndsh::vortex_io io{stream.get()};
  auto read = [&](std::string const& name, std::vector<std::string> const& columns) {
    return ndsh::read_local_file(
      files.tables.path(name, use_vortex), use_vortex, io, columns, cold);
  };
  if (!cold) {
    auto warmup = load_data(read);
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
  state.add_element_count(files.tables.rows, "Rows");
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
      if (cold) { ndsh::evict_file_pages(files.tables.paths(use_vortex)); }
      timer.start();
      if (workload == "read") {
        auto inputs = load_data(read);
        CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
      } else {
        auto result = execute_q9(engine, read, stream, cudf::get_current_device_resource_ref());
        CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
      }
      // Include destruction on independent Vortex producer streams, for both formats.
      CUDF_CUDA_TRY(cudaDeviceSynchronize());
      timer.stop();
    });
  state.add_buffer_size(files.tables.bytes(use_vortex), "file_size", "Total file size");
  state.add_buffer_size(memory.peak_memory_usage(), "rmm_peak", "RMM peak (excludes Vortex)");
  auto& matched = state.add_summary("ndsh/q9/matched_rows");
  matched.set_string("name", "Q9 matched rows");
  matched.set_int64("value", files.reference.matched);
  auto& groups = state.add_summary("ndsh/q9/groups");
  groups.set_string("name", "Q9 nation/year groups");
  groups.set_int64("value", files.reference.sum_profit.size());
}

}  // namespace

NVBENCH_BENCH(ndsh_q9_local)
  .set_name("ndsh_q9_local")
  .add_float64_axis("scale_factor", {0.01, 0.1, 1, 10})
  .add_string_axis("format", {"parquet", "vortex"})
  .add_string_axis("workload", {"read", "q9"})
  .add_string_axis("engine", {"binaryop", "ast", "transform"})
  .add_string_axis("cache", {"warm", "cold"});
