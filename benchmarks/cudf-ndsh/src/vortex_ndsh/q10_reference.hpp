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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ndsh {
struct q10_customer_result {
  std::string name;
  double account_balance;
  std::string nation;
  std::string address;
  std::string phone;
  std::string comment;
  double revenue = 0;
};

struct q10_reference_result {
  std::map<int32_t, q10_customer_result> customers;
  int64_t matched = 0;
};

// One callback per projected table in generator order; all computation is on bounded host copies.
class q10_reference_builder {
 public:
  void add_table(std::string const& name, cudf::table_view projected, cuda::stream_ref stream)
  {
    static constexpr std::array<char const*, 4> tables{"nation", "customer", "orders", "lineitem"};
    CUDF_EXPECTS(next_table_ < tables.size() && name == tables[next_table_],
                 "Q10 requires nation, customer, orders, lineitem in order");
    using enum cudf::type_id;
    if (name == "nation") {
      detail::reference_schema(projected, {STRING, INT8});
    } else if (name == "customer") {
      detail::reference_schema(projected, {INT32, STRING, INT8, FLOAT64, STRING, STRING, STRING});
    } else if (name == "orders") {
      detail::reference_schema(projected, {INT32, INT32, TIMESTAMP_DAYS});
    } else {
      detail::reference_schema(projected, {FLOAT64, FLOAT64, INT32, STRING});
    }

    // Filtered dimensions must also reject duplicates among discarded rows.
    std::unordered_set<int32_t> primary_keys;
    for (cudf::size_type begin = 0; begin < projected.num_rows();) {
      auto const count = std::min<cudf::size_type>(1 << 20, projected.num_rows() - begin);
      auto values      = [&](auto type, int index) {
        return detail::reference_host_copy(
          projected.column(index).data<decltype(type)>() + begin, count, stream);
      };
      if (name == "nation") {
        auto const names =
          detail::reference_host_strings(projected.column(0), begin, count, stream);
        auto const keys = values(int8_t{}, 1);
        for (cudf::size_type i = 0; i < count; ++i) {
          CUDF_EXPECTS(nations_.try_emplace(keys[i], names[i]).second, "Duplicate Q10 nation key");
        }
      } else if (name == "customer") {
        auto const keys = values(int32_t{}, 0);
        auto const names =
          detail::reference_host_strings(projected.column(1), begin, count, stream);
        auto const nations  = values(int8_t{}, 2);
        auto const balances = values(double{}, 3);
        auto const addresses =
          detail::reference_host_strings(projected.column(4), begin, count, stream);
        auto const phones =
          detail::reference_host_strings(projected.column(5), begin, count, stream);
        auto const comments =
          detail::reference_host_strings(projected.column(6), begin, count, stream);
        for (cudf::size_type i = 0; i < count; ++i) {
          CUDF_EXPECTS(primary_keys.insert(keys[i]).second, "Duplicate Q10 customer key");
          auto const nation = nations_.find(nations[i]);
          if (nation == nations_.end()) continue;
          customers_.emplace(
            keys[i],
            q10_customer_result{
              names[i], balances[i], nation->second, addresses[i], phones[i], comments[i]});
        }
      } else if (name == "orders") {
        auto const customers = values(int32_t{}, 0);
        auto const keys      = values(int32_t{}, 1);
        auto const dates     = values(cudf::timestamp_D{}, 2);
        for (cudf::size_type i = 0; i < count; ++i) {
          CUDF_EXPECTS(primary_keys.insert(keys[i]).second, "Duplicate Q10 order key");
          auto const date = dates[i].time_since_epoch().count();
          // 1993-10-01 inclusive through 1994-01-01 exclusive, in epoch days.
          if (date < 8674 || date >= 8766 || !customers_.contains(customers[i])) continue;
          orders_.emplace(keys[i], customers[i]);
        }
      } else {
        auto const prices    = values(double{}, 0);
        auto const discounts = values(double{}, 1);
        auto const orders    = values(int32_t{}, 2);
        auto const flags =
          detail::reference_host_strings(projected.column(3), begin, count, stream);
        for (cudf::size_type i = 0; i < count; ++i) {
          if (flags[i] != "R") continue;
          auto const order = orders_.find(orders[i]);
          if (order == orders_.end()) continue;
          ++result_.matched;
          auto const customer = customers_.find(order->second);
          CUDF_EXPECTS(customer != customers_.end(), "Missing Q10 customer for qualifying order");
          auto [result, inserted] = result_.customers.try_emplace(order->second, customer->second);
          result->second.revenue += prices[i] * (1.0 - discounts[i]);
        }
      }
      begin += count;
    }
    ++next_table_;
  }

  q10_reference_result finish() const
  {
    CUDF_EXPECTS(next_table_ == 4, "Incomplete Q10 reference inputs");
    return result_;
  }

 private:
  std::size_t next_table_ = 0;
  std::unordered_map<int8_t, std::string> nations_;
  std::unordered_map<int32_t, q10_customer_result> customers_;
  std::unordered_map<int32_t, int32_t> orders_;
  q10_reference_result result_;
};

inline void check_q10_result(q10_reference_result const& expected,
                             table_with_names const& actual,
                             cuda::stream_ref stream)
{
  std::vector<std::string> const names{
    "c_custkey", "c_name", "c_acctbal", "c_phone", "n_name", "c_address", "c_comment", "revenue"};
  CUDF_EXPECTS(actual.column_names() == names && actual.table().num_columns() == 8,
               "Unexpected Q10 output columns");
  auto const table = actual.table();
  detail::reference_schema(table,
                           {cudf::type_id::INT32,
                            cudf::type_id::STRING,
                            cudf::type_id::FLOAT64,
                            cudf::type_id::STRING,
                            cudf::type_id::STRING,
                            cudf::type_id::STRING,
                            cudf::type_id::STRING,
                            cudf::type_id::FLOAT64});
  CUDF_EXPECTS(static_cast<std::size_t>(table.num_rows()) == expected.customers.size(),
               "Q10 customer count mismatch");
  std::unordered_set<int32_t> seen;
  double previous = std::numeric_limits<double>::infinity();
  for (cudf::size_type begin = 0; begin < table.num_rows();) {
    auto const count = std::min<cudf::size_type>(1 << 20, table.num_rows() - begin);
    auto const keys =
      detail::reference_host_copy(table.column(0).data<int32_t>() + begin, count, stream);
    auto const customer_names =
      detail::reference_host_strings(table.column(1), begin, count, stream);
    auto const balances =
      detail::reference_host_copy(table.column(2).data<double>() + begin, count, stream);
    auto const phones    = detail::reference_host_strings(table.column(3), begin, count, stream);
    auto const nations   = detail::reference_host_strings(table.column(4), begin, count, stream);
    auto const addresses = detail::reference_host_strings(table.column(5), begin, count, stream);
    auto const comments  = detail::reference_host_strings(table.column(6), begin, count, stream);
    auto const revenues =
      detail::reference_host_copy(table.column(7).data<double>() + begin, count, stream);
    for (cudf::size_type i = 0; i < count; ++i) {
      auto const customer = expected.customers.find(keys[i]);
      CUDF_EXPECTS(customer != expected.customers.end() && seen.insert(keys[i]).second,
                   "Unexpected or duplicate Q10 customer: " + std::to_string(keys[i]));
      auto const& want = customer->second;
      CUDF_EXPECTS(customer_names[i] == want.name && balances[i] == want.account_balance &&
                     nations[i] == want.nation && addresses[i] == want.address &&
                     phones[i] == want.phone && comments[i] == want.comment,
                   "Q10 customer attributes mismatch: " + std::to_string(keys[i]));
      double const value = revenues[i];
      CUDF_EXPECTS(detail::reference_equal(value, want.revenue),
                   "Q10 revenue mismatch: " + std::to_string(keys[i]));
      CUDF_EXPECTS(value <= previous, "Q10 revenues are not sorted descending");
      previous = value;
    }
    begin += count;
  }
  CUDF_EXPECTS(seen.size() == expected.customers.size(), "Missing Q10 output customers");
}
}  // namespace ndsh
