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
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ndsh {
struct q5_reference_result {
  std::map<std::string, double> revenue;
  int64_t matched = 0;
};

// One callback per table in dependency order; only filtered CPU dimensions survive callbacks.
class q5_reference_builder {
 public:
  void add_table(std::string const& name, cudf::table_view projected, cuda::stream_ref stream)
  {
    if (name == "part" || name == "partsupp") return;
    static constexpr std::array<char const*, 6> tables{
      "region", "nation", "supplier", "customer", "orders", "lineitem"};
    CUDF_EXPECTS(next_table_ < tables.size() && name == tables[next_table_],
                 "Q5 requires region, nation, supplier, customer, orders, lineitem in order");
    using enum cudf::type_id;
    if (name == "region") {
      detail::reference_schema(projected, {INT8, STRING});
    } else if (name == "nation") {
      detail::reference_schema(projected, {INT8, INT8, STRING});
    } else if (name == "supplier" || name == "customer") {
      detail::reference_schema(projected, {INT32, INT8});
    } else if (name == "orders") {
      detail::reference_schema(projected, {INT32, INT32, TIMESTAMP_DAYS});
    } else {
      detail::reference_schema(projected, {INT32, INT32, FLOAT64, FLOAT64});
    }
    // Check even discarded dimension rows, so filtering cannot conceal duplicate primary keys.
    std::unordered_set<int32_t> primary_keys;
    auto unique = [&](int32_t key) {
      CUDF_EXPECTS(primary_keys.insert(key).second, "Duplicate Q5 primary key in " + name);
    };
    for (cudf::size_type begin = 0; begin < projected.num_rows();) {
      auto const count = std::min<cudf::size_type>(1 << 20, projected.num_rows() - begin);
      auto values      = [&](auto type, int index) {
        return detail::reference_host_copy(
          projected.column(index).data<decltype(type)>() + begin, count, stream);
      };
      if (name == "region") {
        auto const keys = values(int8_t{}, 0);
        auto const names =
          detail::reference_host_strings(projected.column(1), begin, count, stream);
        for (cudf::size_type i = 0; i < count; ++i) {
          unique(keys[i]);
          if (names[i] == "ASIA") regions_.insert(keys[i]);
        }
      } else if (name == "nation") {
        auto const keys    = values(int8_t{}, 0);
        auto const regions = values(int8_t{}, 1);
        auto const names =
          detail::reference_host_strings(projected.column(2), begin, count, stream);
        for (cudf::size_type i = 0; i < count; ++i) {
          unique(keys[i]);
          if (regions_.contains(regions[i])) nations_.emplace(keys[i], names[i]);
        }
      } else if (name == "supplier" || name == "customer") {
        auto const keys    = values(int32_t{}, 0);
        auto const nations = values(int8_t{}, 1);
        auto& dimension    = name == "supplier" ? suppliers_ : customers_;
        for (cudf::size_type i = 0; i < count; ++i) {
          unique(keys[i]);
          if (nations_.contains(nations[i])) dimension.emplace(keys[i], nations[i]);
        }
      } else if (name == "orders") {
        auto const customers = values(int32_t{}, 0);
        auto const keys      = values(int32_t{}, 1);
        auto const dates     = values(cudf::timestamp_D{}, 2);
        for (cudf::size_type i = 0; i < count; ++i) {
          unique(keys[i]);
          auto const date = dates[i].time_since_epoch().count();
          // 1994-01-01 inclusive through 1995-01-01 exclusive, in epoch days.
          if (date < 8766 || date >= 9131) continue;
          auto const customer = customers_.find(customers[i]);
          if (customer != customers_.end()) orders_.emplace(keys[i], customer->second);
        }
      } else {
        auto const orders    = values(int32_t{}, 0);
        auto const suppliers = values(int32_t{}, 1);
        auto const prices    = values(double{}, 2);
        auto const discounts = values(double{}, 3);
        for (cudf::size_type i = 0; i < count; ++i) {
          auto const order = orders_.find(orders[i]);
          if (order == orders_.end()) continue;
          auto const supplier = suppliers_.find(suppliers[i]);
          // The supplier must match both its key and the customer's nation.
          if (supplier == suppliers_.end() || supplier->second != order->second) continue;
          ++result_.matched;
          result_.revenue[nations_.at(order->second)] += prices[i] * (1.0 - discounts[i]);
        }
      }
      begin += count;
    }
    if (name == "orders") customers_.clear();
    ++next_table_;
  }

  q5_reference_result finish() const
  {
    CUDF_EXPECTS(next_table_ == 6, "Incomplete Q5 reference inputs");
    return result_;
  }

 private:
  std::size_t next_table_ = 0;
  std::unordered_set<int8_t> regions_;
  std::unordered_map<int8_t, std::string> nations_;
  std::unordered_map<int32_t, int8_t> suppliers_, customers_, orders_;
  q5_reference_result result_;
};

inline void check_q5_result(q5_reference_result const& expected,
                            table_with_names const& actual,
                            cuda::stream_ref stream)
{
  std::vector<std::string> const names{"n_name", "revenue"};
  CUDF_EXPECTS(actual.column_names().size() == 2 && actual.table().num_columns() == 2,
               "Expected exactly two Q5 output columns");
  for (auto const& name : names) {
    CUDF_EXPECTS(std::count(actual.column_names().begin(), actual.column_names().end(), name) == 1,
                 "Missing or duplicate Q5 output column: " + name);
  }
  auto const table = actual.select(names);
  detail::reference_schema(table, {cudf::type_id::STRING, cudf::type_id::FLOAT64});
  CUDF_EXPECTS(static_cast<std::size_t>(table.num_rows()) == expected.revenue.size(),
               "Q5 group count mismatch");
  std::unordered_set<std::string> seen;
  double previous = std::numeric_limits<double>::infinity();
  for (cudf::size_type begin = 0; begin < table.num_rows();) {
    auto const count     = std::min<cudf::size_type>(1 << 20, table.num_rows() - begin);
    auto const countries = detail::reference_host_strings(table.column(0), begin, count, stream);
    auto const revenues =
      detail::reference_host_copy(table.column(1).data<double>() + begin, count, stream);
    for (cudf::size_type i = 0; i < count; ++i) {
      auto const group = expected.revenue.find(countries[i]);
      CUDF_EXPECTS(group != expected.revenue.end() && seen.insert(countries[i]).second,
                   "Unexpected or duplicate Q5 country: " + countries[i]);
      double const value = revenues[i], want = group->second;
      CUDF_EXPECTS(std::isfinite(value) && std::isfinite(want) &&
                     std::abs(value - want) <= 1e-10 * std::max(1.0, std::abs(want)),
                   "Q5 revenue mismatch: " + countries[i]);
      CUDF_EXPECTS(value <= previous, "Q5 revenues are not sorted descending");
      previous = value;
    }
    begin += count;
  }
}
}  // namespace ndsh
