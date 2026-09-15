/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 */

// Included at file scope after q10.cpp: consumes execute_q10, q10_tables, and
// q10_projections, plus table_with_names, for_each_generated_table, and apply_filter
// from utilities.hpp.

#include "local_io.hpp"
#include "q10_reference.hpp"
#include "vortex_io.hpp"

#include <benchmarks/common/nvtx_ranges.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>

#include <cmath>
#include <cstdint>

namespace {

template <typename Read>
std::unique_ptr<table_with_names> execute_q10(Read&& read, bool filter_predicates)
{
  return ::execute_q10(read, filter_predicates, [](auto& result) { return std::move(result); });
}

void check_q10_cases()
{
  cudf::test::fixed_width_column_wrapper<int8_t> nation_key{{10, 20}};
  cudf::test::strings_column_wrapper nation_name{"ALPHA", "ZULU"};
  cudf::test::fixed_width_column_wrapper<int32_t> customer_key{{1, 2, 3, 4}};
  cudf::test::strings_column_wrapper customer_name{"Alice", "Bob", "NoNation", "NoOrders"};
  cudf::test::fixed_width_column_wrapper<int8_t> customer_nation{{10, 20, 99, 10}};
  cudf::test::fixed_width_column_wrapper<double> account_balance{{100.0, -20.0, 0.0, 5.0}};
  cudf::test::strings_column_wrapper customer_address{"1 Main", "2 Main", "3 Main", "4 Main"};
  cudf::test::strings_column_wrapper customer_phone{"10-1", "20-2", "99-3", "10-4"};
  cudf::test::strings_column_wrapper customer_comment{"first", "second", "third", "fourth"};
  cudf::test::fixed_width_column_wrapper<int32_t> order_customer{{1, 1, 2, 2, 2, 3, 999, 2, 2}};
  cudf::test::fixed_width_column_wrapper<int32_t> order_key{
    {101, 102, 105, 103, 104, 106, 107, 109, 108}};
  // Epoch days include both date boundaries and a final null date.
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, int32_t> order_date{
    {8674, 8765, 8705, 8766, 8673, 8705, 8705, 8705, 8674},
    {true, true, true, true, true, true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<double> price{{100.0,
                                                        50.0,
                                                        200.0,
                                                        80.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0,
                                                        1000.0}};
  cudf::test::fixed_width_column_wrapper<double> discount{
    {0.1, 0.2, 0.25, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  cudf::test::fixed_width_column_wrapper<int32_t> line_order{
    {101, 101, 102, 105, 103, 104, 106, 107, 109, 109, 999, 108, 105}};
  cudf::test::strings_column_wrapper return_flag(
    {"R", "R", "R", "R", "R", "R", "R", "R", "r", "N", "R", "R", ""},
    {true, true, true, true, true, true, true, true, true, true, true, true, false});
  CUDF_CUDA_TRY(cudaDeviceSynchronize());

  std::map<std::string, cudf::table_view> const input{
    {"nation", cudf::table_view{{nation_name, nation_key}}},
    {"customer",
     cudf::table_view{{customer_key,
                       customer_name,
                       customer_nation,
                       account_balance,
                       customer_address,
                       customer_phone,
                       customer_comment}}},
    {"orders", cudf::table_view{{order_customer, order_key, order_date}}},
    {"lineitem", cudf::table_view{{price, discount, line_order, return_flag}}}};
  ndsh::q10_reference_result expected;
  expected.customers.emplace(
    1, ndsh::q10_customer_result{"Alice", 100.0, "ALPHA", "1 Main", "10-1", "first", 280.0});
  expected.customers.emplace(
    2, ndsh::q10_customer_result{"Bob", -20.0, "ZULU", "2 Main", "20-2", "second", 40.0});
  expected.matched              = 4;
  cuda::stream_ref const stream = cudf::get_default_stream();
  auto const order_cases        = cudf::slice(input.at("orders"), {0, 9, 3, 9, 0, 0}, stream);
  auto const cpu_order_cases    = cudf::slice(input.at("orders"), {0, 8, 3, 8, 0, 0}, stream);
  auto const cpu_lineitem       = cudf::slice(input.at("lineitem"), {0, 12}, stream).front();
  // Full input, only rows rejected by predicates/joins, and no orders.
  for (std::size_t case_index = 0; case_index < order_cases.size(); ++case_index) {
    auto const& orders     = order_cases[case_index];
    auto const& cpu_orders = cpu_order_cases[case_index];
    auto const want        = case_index == 0 ? expected : ndsh::q10_reference_result{};
    ndsh::q10_reference_builder builder;
    for (auto const& name : {"nation", "customer", "orders", "lineitem"}) {
      auto const source = std::string{name} == "orders"     ? cpu_orders
                          : std::string{name} == "lineitem" ? cpu_lineitem
                                                            : input.at(name);
      builder.add_table(name, source, stream);
    }
    auto const cpu = builder.finish();
    CUDF_EXPECTS(cpu.matched == want.matched && cpu.customers.size() == want.customers.size(),
                 "Q10 CPU reference row/customer regression");
    for (auto const& [key, expected_customer] : want.customers) {
      auto const customer = cpu.customers.find(key);
      CUDF_EXPECTS(customer != cpu.customers.end(), "Missing Q10 CPU reference customer");
      auto const& actual_customer = customer->second;
      CUDF_EXPECTS(actual_customer.name == expected_customer.name &&
                     actual_customer.account_balance == expected_customer.account_balance &&
                     actual_customer.nation == expected_customer.nation &&
                     actual_customer.address == expected_customer.address &&
                     actual_customer.phone == expected_customer.phone &&
                     actual_customer.comment == expected_customer.comment &&
                     std::abs(actual_customer.revenue - expected_customer.revenue) <=
                       1e-10 * std::max(1.0, std::abs(expected_customer.revenue)),
                   "Q10 CPU reference predicate/join/revenue regression");
    }
    for (bool post_read_filters : {true, false}) {
      auto result = execute_q10(
        [&](std::string const& name,
            std::vector<std::string> const& columns,
            std::unique_ptr<cudf::ast::operation> const& predicate) {
          CUDF_EXPECTS(columns == q10_projections.at(name), "Q10 projection mismatch");
          auto const source = name == "orders" ? orders : input.at(name);
          auto table =
            std::make_unique<table_with_names>(std::make_unique<cudf::table>(source), columns);
          return !post_read_filters && predicate ? apply_filter(table, *predicate)
                                                 : std::move(table);
        },
        post_read_filters);
      ndsh::check_q10_result(want, *result, stream);
    }
  }
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
}

struct q10_files {
  ndsh::local_table_files tables;
  ndsh::q10_reference_result reference;

  explicit q10_files(double scale_factor)
  {
    check_q10_cases();
    ndsh::make_reference_files<ndsh::q10_reference_builder>(
      scale_factor, tables, reference, q10_tables, q10_projections, true,
      [&](auto&& read, cuda::stream_ref stream) {
        auto result = execute_q10(read, true);
        ndsh::check_q10_result(reference, *result, stream);
      });
  }
};

void ndsh_q10_local(nvbench::state& state)
{
  auto const format   = state.get_string("format");
  auto const workload = state.get_string("workload");
  auto const cold     = ndsh::use_cold_cache(state.get_string("cache"));
  CUDF_EXPECTS(format == "parquet" || format == "vortex", "Unknown Q10 format");
  CUDF_EXPECTS(workload == "read" || workload == "q10", "Unknown Q10 workload");
  auto const use_vortex         = format == "vortex";
  auto const& files             = ndsh::local_fixture<q10_files>(state.get_float64("scale_factor"));
  cuda::stream_ref const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const memory = cudf::memory_stats_logger();
  ndsh::vortex_io io{stream.get()};
  auto read = [&](std::string const& name,
                  std::vector<std::string> const& columns,
                  std::unique_ptr<cudf::ast::operation> const&) {
    return ndsh::read_local_file(
      files.tables.path(name, use_vortex), use_vortex, io, columns, cold);
  };
  ndsh::warm_local_inputs(cold, [&] {
    return ndsh::read_local_tables(q10_tables, q10_projections, read);
  });
  ndsh::exec_local_benchmark(state, files.tables, use_vortex, cold, [&] {
    cudf::benchmark::scoped_range timed_range{"ndsh_q10_local_timed"};
    if (workload == "read") {
      auto inputs = ndsh::read_local_tables(q10_tables, q10_projections, read);
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    } else {
      auto result = execute_q10(read, true);
      CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    }
  });
  state.add_buffer_size(files.tables.bytes(use_vortex), "file_size", "Total file size");
  state.add_buffer_size(memory.peak_memory_usage(), "rmm_peak", "RMM peak (excludes Vortex)");
  auto& matched = state.add_summary("ndsh/q10/matched_rows");
  matched.set_string("name", "Q10 matched rows");
  matched.set_int64("value", files.reference.matched);
  auto& customers = state.add_summary("ndsh/q10/customers");
  customers.set_string("name", "Q10 customers");
  customers.set_int64("value", files.reference.customers.size());
}

}  // namespace

NVBENCH_BENCH(ndsh_q10_local)
  .set_name("ndsh_q10_local")
  .add_float64_axis("scale_factor", {0.01, 0.1, 1, 10})
  .add_string_axis("format", {"parquet", "vortex"})
  .add_string_axis("workload", {"read", "q10"})
  .add_string_axis("cache", {"warm", "cold"});
