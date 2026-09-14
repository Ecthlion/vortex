/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace ndsh::detail {
inline void reference_schema(cudf::table_view table, std::initializer_list<cudf::type_id> types)
{
  CUDF_EXPECTS(static_cast<std::size_t>(table.num_columns()) == types.size(),
               "Unexpected reference column count");
  int i = 0;
  for (auto type : types) {
    auto column = table.column(i++);
    CUDF_EXPECTS(column.type().id() == type && column.null_count() == 0,
                 "Unexpected reference type or nulls");
  }
}

template <typename T>
auto reference_host_copy(T const* data, std::size_t count, cuda::stream_ref stream)
{
  return cudf::detail::make_host_vector(cudf::device_span<T const>{data, count}, stream);
}

inline auto reference_host_strings(cudf::column_view column,
                                   cudf::size_type begin,
                                   cudf::size_type count,
                                   cuda::stream_ref stream)
{
  cudf::strings_column_view strings{column};
  auto const offsets = strings.offsets();
  CUDF_EXPECTS(
    (offsets.type().id() == cudf::type_id::INT32 || offsets.type().id() == cudf::type_id::INT64) &&
      offsets.null_count() == 0,
    "Unexpected reference string offsets");
  auto copy = [&](auto offset_type) {
    auto const host_offsets = reference_host_copy(
      offsets.data<decltype(offset_type)>() + column.offset() + begin, count + 1, stream);
    std::vector<std::string> result(count);
    if (host_offsets.back() == host_offsets.front()) return result;
    // Copy only this batch's character interval, not the entire backing column.
    auto const chars = reference_host_copy(strings.chars_begin(stream) + host_offsets.front(),
                                           host_offsets.back() - host_offsets.front(),
                                           stream);
    for (cudf::size_type i = 0; i < count; ++i) {
      result[i].assign(chars.data() + (host_offsets[i] - host_offsets.front()),
                       host_offsets[i + 1] - host_offsets[i]);
    }
    return result;
  };
  return offsets.type().id() == cudf::type_id::INT32 ? copy(int32_t{}) : copy(int64_t{});
}
}  // namespace ndsh::detail
