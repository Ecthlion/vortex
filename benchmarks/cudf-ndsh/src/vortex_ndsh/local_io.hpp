/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "utilities.hpp"
#include "vortex_io.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include <kvikio/file_utils.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace ndsh {
class local_table_files {
 public:
  int64_t rows = 0;

  void write(std::string const& name, table_with_names const& table, vortex_io const& io)
  {
    CUDF_EXPECTS(!files_.contains(name), "Duplicate fixture table: " + name);
    auto const base = directory_.path() + name;
    table.to_parquet(base + ".parquet");
    io.write_vortex(base + ".vortex", table.table(), table.column_names());
    files_.emplace(name, std::make_pair(base + ".parquet", base + ".vortex"));
    rows += table.table().num_rows();
  }

  std::string const& path(std::string const& name, bool use_vortex) const
  {
    auto const& files = files_.at(name);
    return use_vortex ? files.second : files.first;
  }

  std::uintmax_t bytes(bool use_vortex) const
  {
    std::uintmax_t total = 0;
    for (auto const& [name, files] : files_) {
      total += std::filesystem::file_size(use_vortex ? files.second : files.first);
    }
    return total;
  }

  std::vector<std::string> paths(bool use_vortex) const
  {
    std::vector<std::string> result;
    result.reserve(files_.size());
    for (auto const& [name, files] : files_) {
      result.push_back(use_vortex ? files.second : files.first);
    }
    return result;
  }

 private:
  temp_directory directory_{"ndsh_local"};
  std::map<std::string, std::pair<std::string, std::string>> files_;
};

inline std::unique_ptr<table_with_names> read_local_file(std::string const& path,
                                                         bool use_vortex,
                                                         vortex_io const& io,
                                                         std::vector<std::string> const& columns,
                                                         bool direct_io = false)
{
  if (!use_vortex) { return read_parquet(cudf::io::source_info{path}, columns); }
  auto result = io.read_vortex(path, 16 << 20, columns, direct_io);
  std::vector<std::string> names;
  for (auto const& field : result.metadata.schema_info) {
    names.push_back(field.name);
  }
  return std::make_unique<table_with_names>(std::move(result.tbl), std::move(names));
}

template <typename Read>
std::vector<std::unique_ptr<table_with_names>> read_local_tables(
  std::vector<std::string> const& names,
  std::map<std::string, std::vector<std::string>> const& projections,
  Read&& read)
{
  std::vector<std::unique_ptr<table_with_names>> tables;
  tables.reserve(names.size());
  std::unique_ptr<cudf::ast::operation> const no_predicate;
  for (auto const& name : names) {
    tables.push_back(read(name, projections.at(name), no_predicate));
  }
  return tables;
}

inline bool use_cold_cache(std::string const& cache)
{
  CUDF_EXPECTS(cache == "warm" || cache == "cold", "Unknown cache mode");
  return cache == "cold";
}

inline void evict_file_pages(std::vector<std::string> const& paths)
{
  CUDF_EXPECTS(!paths.empty(), "Cold-cache benchmark requires at least one input file");
  for (auto const& path : paths) {
    kvikio::drop_file_page_cache(path);
  }
  for (auto const& path : paths) {
    auto const resident_pages = kvikio::get_page_cache_info(path).first;
    CUDF_EXPECTS(resident_pages == 0, "Input pages remain cached after eviction: " + path);
  }
}

template <typename ReadAll>
void warm_local_inputs(bool cold, ReadAll&& read_all)
{
  if (!cold) {
    auto inputs = read_all();
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
}

// The callback keeps owners alive through consumer-stream synchronization, then releases them.
// The final device sync includes cleanup on independent Vortex producer streams, for both formats.
template <typename Run>
void exec_local_benchmark(nvbench::state& state,
                          local_table_files const& files,
                          bool use_vortex,
                          bool cold,
                          Run&& run)
{
  static_assert(std::is_void_v<std::invoke_result_t<Run&>>,
                "The timed callback must release its owners before returning");
  state.add_element_count(files.rows, "Rows");
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch&, auto& timer) {
               if (cold) { evict_file_pages(files.paths(use_vortex)); }
               timer.start();
               run();
               CUDF_CUDA_TRY(cudaDeviceSynchronize());
               timer.stop();
             });
}

inline void check_projection(cudf::table_view expected,
                             table_with_names const& actual,
                             std::vector<std::string> const& columns)
{
  CUDF_EXPECTS(actual.column_names() == columns &&
                 actual.table().num_columns() == expected.num_columns() &&
                 actual.table().num_rows() == expected.num_rows(),
               "Projected schema/row count mismatch");
  for (cudf::size_type i = 0; i < expected.num_columns(); ++i) {
    auto const lhs = expected.column(i);
    auto const rhs = actual.table().column(i);
    CUDF_EXPECTS(lhs.type() == rhs.type(), "Projected type mismatch");
    if (lhs.is_empty()) { continue; }
    auto equal = cudf::binary_operation(
      lhs, rhs, cudf::binary_operator::NULL_EQUALS, cudf::data_type{cudf::type_id::BOOL8});
    auto all = cudf::reduce(equal->view(),
                            *cudf::make_all_aggregation<cudf::reduce_aggregation>(),
                            cudf::data_type{cudf::type_id::BOOL8});
    CUDF_EXPECTS(all->is_valid() && static_cast<cudf::numeric_scalar<bool> const&>(*all).value(),
                 "Projected values mismatch");
  }
}

inline void check_file_projections(local_table_files const& files,
                                    std::string const& name,
                                    cudf::table_view expected,
                                    std::vector<std::string> const& columns,
                                    vortex_io const& io)
{
  for (bool use_vortex : {false, true}) {
    auto input = read_local_file(files.path(name, use_vortex), use_vortex, io, columns);
    check_projection(expected, *input, columns);
  }
}
}  // namespace ndsh
