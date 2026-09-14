/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "reference_io.hpp"
#include "utilities.hpp"

#include <cudf/utilities/error.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ndsh {
/** Q1 metrics: sum_qty, sum_base_price, sum_disc_price, sum_charge, avg_qty, avg_price, avg_disc,
 * count_order. */
struct q1_reference_result {
  std::map<std::pair<std::string, std::string>, std::array<double, 8>> groups;
  cudf::size_type matched = 0;
};

/** CPU-only oracle for the eight projected, non-null generated Q1 columns; outside timed work. */
inline q1_reference_result q1_cpu_reference(cudf::table_view projected, cuda::stream_ref stream)
{
  using enum cudf::type_id;
  detail::reference_schema(
    projected, {STRING, STRING, INT8, FLOAT64, FLOAT64, TIMESTAMP_DAYS, INT32, FLOAT64});
  q1_reference_result result;
  for (cudf::size_type begin = 0; begin < projected.num_rows();) {
    auto const count = std::min<cudf::size_type>(1 << 20, projected.num_rows() - begin);
    auto values      = [&](auto type, int index) {
      return detail::reference_host_copy(
        projected.column(index).data<decltype(type)>() + begin, count, stream);
    };
    auto const flags    = detail::reference_host_strings(projected.column(0), begin, count, stream);
    auto const statuses = detail::reference_host_strings(projected.column(1), begin, count, stream);
    auto const quantity = values(int8_t{}, 2);
    auto const price    = values(double{}, 3);
    auto const discount = values(double{}, 4);
    auto const shipdate = values(cudf::timestamp_D{}, 5);
    auto const tax      = values(double{}, 7);
    for (cudf::size_type i = 0; i < count; ++i) {
      // 1998-09-02 is 10471 days after 1970-01-01; the boundary is inclusive.
      if (shipdate[i].time_since_epoch().count() > 10471) continue;
      auto& group             = result.groups[{flags[i], statuses[i]}];
      double const disc_price = price[i] * (1.0 - discount[i]);
      double const charge     = disc_price * (1.0 + tax[i]);
      // INT8 quantities over at most INT32 rows sum exactly in double.
      group[0] += quantity[i];
      group[1] += price[i];
      group[2] += disc_price;
      group[3] += charge;
      group[6] += discount[i];
      ++group[7];
      ++result.matched;
    }
    begin += count;
  }
  for (auto& [key, group] : result.groups) {
    group[4] = group[0] / group[7];
    group[5] = group[1] / group[7];
    group[6] /= group[7];
  }
  return result;
}

/** Validate named, sorted Q1 output, including empty tables; use 1e-10 relative float tolerance. */
inline void check_q1_result(q1_reference_result const& expected,
                            table_with_names const& actual,
                            cuda::stream_ref stream)
{
  std::vector<std::string> const names{"l_returnflag",
                                       "l_linestatus",
                                       "sum_qty",
                                       "sum_base_price",
                                       "sum_disc_price",
                                       "sum_charge",
                                       "avg_qty",
                                       "avg_price",
                                       "avg_disc",
                                       "count_order"};
  CUDF_EXPECTS(actual.column_names().size() == names.size() && actual.table().num_columns() == 10,
               "Expected exactly ten Q1 output columns");
  for (auto const& name : names) {
    CUDF_EXPECTS(std::count(actual.column_names().begin(), actual.column_names().end(), name) == 1,
                 "Missing or duplicate Q1 output column: " + name);
  }
  auto const table = actual.select(names);
  using enum cudf::type_id;
  detail::reference_schema(
    table, {STRING, STRING, INT64, FLOAT64, FLOAT64, FLOAT64, FLOAT64, FLOAT64, FLOAT64, INT32});
  CUDF_EXPECTS(static_cast<std::size_t>(table.num_rows()) == expected.groups.size(),
               "Q1 group count mismatch");
  auto group      = expected.groups.begin();
  int64_t matched = 0;
  for (cudf::size_type begin = 0; begin < table.num_rows();) {
    auto const count    = std::min<cudf::size_type>(1 << 20, table.num_rows() - begin);
    auto const first    = group;
    auto const flags    = detail::reference_host_strings(table.column(0), begin, count, stream);
    auto const statuses = detail::reference_host_strings(table.column(1), begin, count, stream);
    for (cudf::size_type i = 0; i < count; ++i, ++group) {
      CUDF_EXPECTS(group->first == std::make_pair(flags[i], statuses[i]),
                   "Q1 ordered keys mismatch");
    }
    for (int metric = 0; metric < 8; ++metric) {
      auto check = [&](auto type) {
        auto const values = detail::reference_host_copy(
          table.column(metric + 2).data<decltype(type)>() + begin, count, stream);
        auto reference = first;
        for (cudf::size_type i = 0; i < count; ++i, ++reference) {
          double const value = values[i], want = reference->second[metric];
          bool const exact = metric == 0 || metric == 7;
          CUDF_EXPECTS(std::isfinite(value) && std::isfinite(want) &&
                         (exact ? value == want
                                : std::abs(value - want) <= 1e-10 * std::max(1.0, std::abs(want))),
                       "Q1 metric mismatch: " + names[metric + 2]);
          if (metric == 7) matched += static_cast<int64_t>(values[i]);
        }
      };
      if (metric == 0) {
        check(int64_t{});
      } else if (metric == 7) {
        check(int32_t{});
      } else {
        check(double{});
      }
    }
    begin += count;
  }
  CUDF_EXPECTS(matched == expected.matched, "Q1 matched row count mismatch");
}
}  // namespace ndsh
