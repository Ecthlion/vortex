/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vortex_ndsh/vortex_io.hpp"

#include <cudf/copying.hpp>
#include <cudf/interop.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#if __has_include(<rmm/mr/cuda_async_memory_resource.hpp>)
#include <rmm/mr/cuda_async_memory_resource.hpp>
#else
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#endif

#include <cuda_runtime_api.h>

#include <nanoarrow/nanoarrow.h>
#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

// Production helpers deliberately absent from the public header. Test the staging
// boundary and schema validation without exposing test-only API or depending on FFI headers.
namespace ndsh::detail {
cudf::unique_device_array_t stage_host_chunk(cudf::table_view chunk,
                                             cudaStream_t stream,
                                             rmm::device_async_resource_ref mr);
void check_flat_schema(ArrowSchema const& schema);
}  // namespace ndsh::detail

namespace {

void require(bool condition, std::string_view message)
{
  if (!condition) { throw std::runtime_error(std::string{message}); }
}

void check_cuda(cudaError_t status, char const* operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string{operation} + ": " + cudaGetErrorString(status));
  }
}

class temporary_directory {
 public:
  temporary_directory()
  {
    auto pattern = (std::filesystem::temp_directory_path() / "ndsh-vortex-io-XXXXXX").string();
    auto const created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      throw std::runtime_error(std::string{"mkdtemp: "} + std::strerror(errno));
    }
    path_ = created;
  }

  ~temporary_directory()
  {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
    if (error) { std::cerr << "Temporary directory cleanup failed: " << error.message() << '\n'; }
  }

  temporary_directory(temporary_directory const&)            = delete;
  temporary_directory& operator=(temporary_directory const&) = delete;

  std::string file(std::string const& name) const { return (path_ / name).string(); }

 private:
  std::filesystem::path path_;
};

class cuda_stream {
 public:
  cuda_stream()
  {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create stream");
  }

  ~cuda_stream()
  {
    auto const completed = cudaStreamSynchronize(stream_);
    auto const destroyed = cudaStreamDestroy(stream_);
    if (completed != cudaSuccess || destroyed != cudaSuccess) {
      std::cerr << "CUDA stream cleanup failed: " << cudaGetErrorString(completed) << ", "
                << cudaGetErrorString(destroyed) << '\n';
    }
  }

  cuda_stream(cuda_stream const&)            = delete;
  cuda_stream& operator=(cuda_stream const&) = delete;

  cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_{};
};

// Declare after host Arrow owners so even exception unwinding waits before releasing their buffers.
class host_buffer_fence {
 public:
  explicit host_buffer_fence(cudaStream_t stream) : stream_{stream} {}
  ~host_buffer_fence()
  {
    if (pending_) {
      auto const status = cudaStreamSynchronize(stream_);
      if (status != cudaSuccess) {
        std::cerr << "Host buffer fence failed: " << cudaGetErrorString(status) << '\n';
      }
    }
  }

  host_buffer_fence(host_buffer_fence const&)            = delete;
  host_buffer_fence& operator=(host_buffer_fence const&) = delete;

  void wait()
  {
    check_cuda(cudaStreamSynchronize(stream_), "synchronize Arrow transfer");
    pending_ = false;
  }

 private:
  cudaStream_t stream_;
  bool pending_{true};
};

// Current cuDF takes cuda::stream_ref; the older 26.08 library takes rmm::cuda_stream_view.
auto cudf_stream(cudaStream_t stream) { return decltype(cudf::get_default_stream()){stream}; }

struct column_spec {
  char const* name;
  ArrowType arrow_type;
  cudf::type_id cudf_type;
  int32_t precision{0};
  int32_t arrow_scale{0};
};

std::array const columns{
  column_spec{"bool flag", NANOARROW_TYPE_BOOL, cudf::type_id::BOOL8},
  column_spec{"i8", NANOARROW_TYPE_INT8, cudf::type_id::INT8},
  column_spec{"i16", NANOARROW_TYPE_INT16, cudf::type_id::INT16},
  column_spec{"i32", NANOARROW_TYPE_INT32, cudf::type_id::INT32},
  column_spec{"i64", NANOARROW_TYPE_INT64, cudf::type_id::INT64},
  column_spec{"u8", NANOARROW_TYPE_UINT8, cudf::type_id::UINT8},
  column_spec{"u16", NANOARROW_TYPE_UINT16, cudf::type_id::UINT16},
  column_spec{"u32", NANOARROW_TYPE_UINT32, cudf::type_id::UINT32},
  column_spec{"u64", NANOARROW_TYPE_UINT64, cudf::type_id::UINT64},
  column_spec{"f32", NANOARROW_TYPE_FLOAT, cudf::type_id::FLOAT32},
  column_spec{"f64", NANOARROW_TYPE_DOUBLE, cudf::type_id::FLOAT64},
  column_spec{"utf8.text", NANOARROW_TYPE_STRING, cudf::type_id::STRING},
  column_spec{"date_days", NANOARROW_TYPE_DATE32, cudf::type_id::TIMESTAMP_DAYS},
  column_spec{"decimal32", NANOARROW_TYPE_DECIMAL32, cudf::type_id::DECIMAL32, 9, 2},
  column_spec{"decimal64", NANOARROW_TYPE_DECIMAL64, cudf::type_id::DECIMAL64, 18, 4},
  column_spec{"decimal128", NANOARROW_TYPE_DECIMAL128, cudf::type_id::DECIMAL128, 38, 6},
  column_spec{
    "decimal_negative_scale", NANOARROW_TYPE_DECIMAL128, cudf::type_id::DECIMAL128, 38, -2},
  column_spec{"all_valid", NANOARROW_TYPE_INT32, cudf::type_id::INT32},
  column_spec{"all_null", NANOARROW_TYPE_INT32, cudf::type_id::INT32}};

std::vector<std::string> column_names(std::vector<std::size_t> const& selected = {})
{
  std::vector<std::string> names;
  auto const count = selected.empty() ? columns.size() : selected.size();
  for (std::size_t c = 0; c < count; ++c) {
    names.emplace_back(columns[selected.empty() ? c : selected[c]].name);
  }
  return names;
}

struct host_table {
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array;
};

int decimal_bitwidth(ArrowType type)
{
  switch (type) {
    case NANOARROW_TYPE_DECIMAL32: return 32;
    case NANOARROW_TYPE_DECIMAL64: return 64;
    case NANOARROW_TYPE_DECIMAL128: return 128;
    default: throw std::runtime_error("Not a supported decimal type");
  }
}

template <typename T>
int append_signed(ArrowArray* array, int64_t row)
{
  auto const value = row % 17 == 0   ? std::numeric_limits<T>::min()
                     : row % 17 == 1 ? std::numeric_limits<T>::max()
                                     : static_cast<T>(row % 201 - 100);
  return ArrowArrayAppendInt(array, value);
}

template <typename T>
int append_unsigned(ArrowArray* array, int64_t row)
{
  auto const value = row % 17 == 0 ? std::numeric_limits<T>::max() : static_cast<T>(row % 251);
  return ArrowArrayAppendUInt(array, value);
}

int append_fixture_value(ArrowArray* array, column_spec const& spec, int64_t row)
{
  switch (spec.arrow_type) {
    case NANOARROW_TYPE_BOOL: return ArrowArrayAppendInt(array, row % 2);
    case NANOARROW_TYPE_INT8: return append_signed<int8_t>(array, row);
    case NANOARROW_TYPE_INT16: return append_signed<int16_t>(array, row);
    case NANOARROW_TYPE_INT32: return append_signed<int32_t>(array, row);
    case NANOARROW_TYPE_INT64: return append_signed<int64_t>(array, row);
    case NANOARROW_TYPE_UINT8: return append_unsigned<uint8_t>(array, row);
    case NANOARROW_TYPE_UINT16: return append_unsigned<uint16_t>(array, row);
    case NANOARROW_TYPE_UINT32: return append_unsigned<uint32_t>(array, row);
    case NANOARROW_TYPE_UINT64: return append_unsigned<uint64_t>(array, row);
    case NANOARROW_TYPE_FLOAT:
    case NANOARROW_TYPE_DOUBLE: return ArrowArrayAppendDouble(array, (row % 1001 - 500) * 0.25);
    case NANOARROW_TYPE_DATE32: return ArrowArrayAppendInt(array, row % 40001 - 20000);
    case NANOARROW_TYPE_STRING: {
      // Repetition encourages dictionary compression without depending on the writer's heuristics.
      constexpr std::array<std::string_view, 8> strings{
        "",
        "repeated string long enough to make dictionary compression worthwhile",
        "repeated string long enough to make dictionary compression worthwhile",
        "caf\xc3\xa9",
        std::string_view{"a\0b", 3},
        "another repeated string long enough to make dictionary compression worthwhile",
        "",
        "\xe6\x97\xa5\xe6\x9c\xac"};
      auto const value = strings[static_cast<std::size_t>(row) % strings.size()];
      return ArrowArrayAppendString(array, {value.data(), static_cast<int64_t>(value.size())});
    }
    case NANOARROW_TYPE_DECIMAL32:
    case NANOARROW_TYPE_DECIMAL64:
    case NANOARROW_TYPE_DECIMAL128: {
      ArrowDecimal value;
      ArrowDecimalInit(&value, decimal_bitwidth(spec.arrow_type), spec.precision, spec.arrow_scale);
      if (spec.arrow_type == NANOARROW_TYPE_DECIMAL128 && row % 7 < 2) {
        NANOARROW_THROW_NOT_OK(ArrowDecimalSetDigits(
          &value, ArrowCharView(row % 7 == 0 ? "12345678901234567890123456789012345678"
                                            : "-12345678901234567890123456789012345678")));
      } else {
        ArrowDecimalSetInt(&value, (row % 10001 - 5000) * 101);
      }
      return ArrowArrayAppendDecimal(array, &value);
    }
    default: throw std::runtime_error("Unsupported fixture type");
  }
}

host_table make_fixture(int64_t first_row,
                        cudf::size_type rows,
                        std::vector<std::size_t> const& selected = {})
{
  auto const count = selected.empty() ? columns.size() : selected.size();
  host_table result;
  ArrowSchemaInit(result.schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(result.schema.get(), count));
  result.schema->flags &= ~ARROW_FLAG_NULLABLE;
  for (std::size_t c = 0; c < count; ++c) {
    auto const index = selected.empty() ? c : selected[c];
    auto const& spec = columns[index];
    auto child       = result.schema->children[c];
    if (spec.precision != 0) {
      NANOARROW_THROW_NOT_OK(
        ArrowSchemaSetTypeDecimal(child, spec.arrow_type, spec.precision, spec.arrow_scale));
    } else {
      NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(child, spec.arrow_type));
    }
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(child, spec.name));
    if (index == columns.size() - 2) { child->flags &= ~ARROW_FLAG_NULLABLE; }
  }

  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(result.array.get(), result.schema.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(result.array.get()));
  for (int64_t i = 0; i < rows; ++i) {
    auto const row = first_row + i;
    for (std::size_t c = 0; c < count; ++c) {
      auto const index = selected.empty() ? c : selected[c];
      bool const is_null =
        index == columns.size() - 1 ||
        (index != columns.size() - 2 && (row + static_cast<int64_t>(index) * 3) % 11 == 3);
      NANOARROW_THROW_NOT_OK(
        is_null ? ArrowArrayAppendNull(result.array->children[c], 1)
                : append_fixture_value(result.array->children[c], columns[index], row));
    }
    NANOARROW_THROW_NOT_OK(ArrowArrayFinishElement(result.array.get()));
  }
  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(result.array.get(), NANOARROW_VALIDATION_LEVEL_FULL, nullptr));
  return result;
}

std::unique_ptr<cudf::table> make_source(cudaStream_t stream, int64_t first, cudf::size_type rows)
{
  auto fixture = make_fixture(first, rows);
  host_buffer_fence fence{stream};
  auto source = cudf::from_arrow(fixture.schema.get(), fixture.array.get(), cudf_stream(stream));
  fence.wait();
  return source;
}

nanoarrow::UniqueArrayView array_view(ArrowSchema const* schema, ArrowArray const* array)
{
  nanoarrow::UniqueArrayView view;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewInitFromSchema(view.get(), schema, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayViewSetArray(view.get(), array, nullptr));
  NANOARROW_THROW_NOT_OK(
    ArrowArrayViewValidate(view.get(), NANOARROW_VALIDATION_LEVEL_FULL, nullptr));
  return view;
}

// nanoarrow 0.7/0.8 only provide IDENTICAL comparison. Rebuild logical values to normalize offsets,
// ignored null payloads, optional all-valid masks, and bitmap padding before comparing entire
// arrays. No data values are compared element by element.
nanoarrow::UniqueArray canonical_array(ArrowSchema const* schema, ArrowArray const* array)
{
  auto view = array_view(schema, array);
  require(view->storage_type == NANOARROW_TYPE_STRUCT && view->null_count == 0,
          "Expected a non-nullable Arrow table");

  nanoarrow::UniqueArray result;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(result.get(), schema, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(result.get()));
  for (int64_t row = 0; row < view->length; ++row) {
    for (int64_t c = 0; c < view->n_children; ++c) {
      auto const child = view->children[c];
      auto out         = result->children[c];
      require(child->dictionary == nullptr, "Expected decoded logical values, not a dictionary");
      require(child->length == view->length, "Arrow child length differs");
      if (ArrowArrayViewIsNull(child, row)) {
        NANOARROW_THROW_NOT_OK(ArrowArrayAppendNull(out, 1));
        continue;
      }
      int status = NANOARROW_OK;
      switch (child->storage_type) {
        case NANOARROW_TYPE_BOOL:
        case NANOARROW_TYPE_INT8:
        case NANOARROW_TYPE_INT16:
        case NANOARROW_TYPE_INT32:
        case NANOARROW_TYPE_INT64:
          status = ArrowArrayAppendInt(out, ArrowArrayViewGetIntUnsafe(child, row));
          break;
        case NANOARROW_TYPE_UINT8:
        case NANOARROW_TYPE_UINT16:
        case NANOARROW_TYPE_UINT32:
        case NANOARROW_TYPE_UINT64:
          status = ArrowArrayAppendUInt(out, ArrowArrayViewGetUIntUnsafe(child, row));
          break;
        case NANOARROW_TYPE_FLOAT:
        case NANOARROW_TYPE_DOUBLE:
          status = ArrowArrayAppendDouble(out, ArrowArrayViewGetDoubleUnsafe(child, row));
          break;
        case NANOARROW_TYPE_STRING:
          status = ArrowArrayAppendString(out, ArrowArrayViewGetStringUnsafe(child, row));
          break;
        case NANOARROW_TYPE_DECIMAL32:
        case NANOARROW_TYPE_DECIMAL64:
        case NANOARROW_TYPE_DECIMAL128: {
          ArrowDecimal value;
          ArrowSchemaView field{};
          NANOARROW_THROW_NOT_OK(ArrowSchemaViewInit(&field, schema->children[c], nullptr));
          ArrowDecimalInit(&value,
                           decimal_bitwidth(child->storage_type),
                           field.decimal_precision,
                           field.decimal_scale);
          ArrowArrayViewGetDecimalUnsafe(child, row, &value);
          status = ArrowArrayAppendDecimal(out, &value);
          break;
        }
        default: throw std::runtime_error("Unsupported Arrow comparison storage type");
      }
      NANOARROW_THROW_NOT_OK(status);
    }
    NANOARROW_THROW_NOT_OK(ArrowArrayFinishElement(result.get()));
  }
  if (auto const remainder = view->length % 8; remainder != 0) {
    auto const mask = static_cast<uint8_t>((1U << remainder) - 1);
    for (int64_t c = 0; c < view->n_children; ++c) {
      auto bitmap = ArrowArrayValidityBitmap(result->children[c]);
      if (bitmap->buffer.size_bytes != 0) {
        bitmap->buffer.data[bitmap->buffer.size_bytes - 1] &= mask;
      }
      if (view->children[c]->storage_type == NANOARROW_TYPE_BOOL) {
        auto values = ArrowArrayBuffer(result->children[c], 1);
        values->data[values->size_bytes - 1] &= mask;
      }
    }
  }
  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(result.get(), NANOARROW_VALIDATION_LEVEL_FULL, nullptr));
  return result;
}

void check_arrow_arrays(ArrowSchema const* schema,
                        ArrowArray const* array,
                        host_table const& expected)
{
  auto normalized_actual   = canonical_array(schema, array);
  auto normalized_expected = canonical_array(expected.schema.get(), expected.array.get());
  auto actual_view         = array_view(schema, normalized_actual.get());
  auto expected_view       = array_view(expected.schema.get(), normalized_expected.get());
  ArrowError reason{};
  int equal = 0;
  NANOARROW_THROW_NOT_OK(ArrowArrayViewCompare(
    actual_view.get(), expected_view.get(), NANOARROW_COMPARE_IDENTICAL, &equal, &reason));
  require(equal != 0, std::string{"Arrow arrays differ: "} + reason.message);
}

void check_table(cudf::io::table_with_metadata const& actual,
                 host_table const& expected,
                 cudaStream_t stream,
                 std::vector<std::size_t> const& selected = {})
{
  require(actual.tbl != nullptr, "Reader returned no owning table");
  require(actual.tbl->num_rows() == expected.array->length, "Row count differs");
  require(actual.metadata.num_rows_per_source ==
            std::vector<std::size_t>{static_cast<std::size_t>(expected.array->length)},
          "Row metadata differs");
  auto const count = selected.empty() ? columns.size() : selected.size();
  require(actual.tbl->num_columns() == static_cast<cudf::size_type>(count), "Column count differs");
  require(actual.metadata.schema_info.size() == count, "Metadata column count differs");
  std::vector<cudf::column_metadata> metadata;
  for (std::size_t c = 0; c < count; ++c) {
    auto const& spec = columns[selected.empty() ? c : selected[c]];
    auto const type  = actual.tbl->view().column(static_cast<cudf::size_type>(c)).type();
    require(type.id() == spec.cudf_type, std::string{spec.name} + ": dtype differs");
    if (spec.precision != 0) {
      require(type.scale() == -spec.arrow_scale,
              std::string{spec.name} + ": decimal scale differs");
    }
    require(actual.metadata.schema_info[c].name == spec.name,
            "Column name differs at index " + std::to_string(c));
    require(actual.tbl->view().column(static_cast<cudf::size_type>(c)).null_count() ==
              expected.array->children[c]->null_count,
            std::string{spec.name} + ": null count differs");
    metadata.emplace_back(spec.name);
  }

  auto schema = cudf::to_arrow_schema(actual.tbl->view(), {metadata.data(), metadata.size()});
  auto host   = cudf::to_arrow_host(actual.tbl->view(), cudf_stream(stream));
  host_buffer_fence fence{stream};
  fence.wait();
  require(host->device_type == ARROW_DEVICE_CPU, "to_arrow_host did not return host data");
  check_arrow_arrays(schema.get(), &host->array, expected);
}

cudf::io::table_with_metadata read_completed(ndsh::vortex_io const& io,
                                             std::string const& path,
                                             cudaStream_t stream,
                                             std::optional<std::size_t> batch_rows,
                                             std::vector<std::size_t> const& selected = {})
{
  auto result = batch_rows
                  ? io.read_vortex(path, *batch_rows,
                                   selected.empty() ? std::vector<std::string>{}
                                                    : column_names(selected))
                  : io.read_vortex(path);
  // Check before any test-side synchronization could hide an unfinished consumer stream.
  check_cuda(cudaStreamQuery(stream), "read_vortex must complete the consumer stream");
  return result;
}

void write_source(ndsh::vortex_io const& io,
                  std::string const& path,
                  cudaStream_t stream,
                  int64_t first,
                  cudf::size_type rows,
                  cudf::size_type slice_begin,
                  std::optional<cudf::size_type> chunk_rows)
{
  auto source = make_source(stream, first, rows + (slice_begin == 0 ? 0 : slice_begin + 4));
  auto input  = source->view();
  if (slice_begin != 0) {
    input = cudf::slice(input, {slice_begin, slice_begin + rows}, cudf_stream(stream)).front();
    require(input.column(0).offset() == slice_begin, "Test input is not actually sliced");
  }
  if (chunk_rows) {
    io.write_vortex(path, input, column_names(), *chunk_rows);
  } else {
    io.write_vortex(path, input, column_names());
  }
  check_cuda(cudaStreamQuery(stream), "write staging cleanup must complete the stream");
}

struct round_trip_case {
  char const* name;
  cudf::size_type rows;
  cudf::size_type slice_begin;
  std::optional<cudf::size_type> chunk_rows;
  std::optional<std::size_t> batch_rows;
};

void round_trip(temporary_directory const& directory, cudaStream_t stream, round_trip_case const& test)
{
  auto const& [name, rows, slice_begin, chunk_rows, batch_rows] = test;
  auto const path = directory.file(std::string{name} + ".vortex");
  {
    ndsh::vortex_io writer{stream};
    write_source(writer, path, stream, 0, rows, slice_begin, chunk_rows);
  }
  ndsh::vortex_io reader{stream};
  check_table(read_completed(reader, path, stream, batch_rows), make_fixture(slice_begin, rows), stream);
}

void test_staged_string_bytes(cudaStream_t stream)
{
  constexpr cudf::size_type parent_rows = 4096;
  auto source                           = make_source(stream, 0, parent_rows);
  std::vector<cudf::column_metadata> metadata;
  for (auto const& name : column_names()) {
    metadata.emplace_back(name);
  }
  auto schema = cudf::to_arrow_schema(source->view(), metadata);
  // Prefix and offset slices exercise both string-compaction conditions; keep the empty path too.
  std::array<std::array<cudf::size_type, 2>, 3> const ranges{{{0, 7}, {5, 12}, {5, 5}}};
  for (auto const& range : ranges) {
    auto input    = cudf::slice(source->view(), {range[0], range[1]}, cudf_stream(stream)).front();
    auto expected = make_fixture(range[0], range[1] - range[0]);
    auto host =
      ndsh::detail::stage_host_chunk(input, stream, cudf::get_current_device_resource_ref());
    check_cuda(cudaStreamQuery(stream), "host staging must complete the consumer stream");
    require(host->device_type == ARROW_DEVICE_CPU, "Staging returned non-host data");
    for (std::size_t c = 0; c < columns.size(); ++c) {
      if (columns[c].cudf_type != cudf::type_id::STRING) { continue; }
      // Both arrays are nanoarrow-owned. Inspect actual buffer sizes: ArrayView's
      // inferred size would miss an unnecessarily retained trailing parent payload.
      auto const* chars          = ArrowArrayBuffer(host->array.children[c], 2);
      auto const* expected_chars = ArrowArrayBuffer(expected.array->children[c], 2);
      require(chars->size_bytes == expected_chars->size_bytes,
              "String staging retained bytes outside the row chunk");
    }
    check_arrow_arrays(schema.get(), &host->array, expected);
  }
}

void test_async_owned_reads(temporary_directory const& directory, cudaStream_t stream)
{
  rmm::mr::cuda_async_memory_resource resource;
  // This fence runs after all table destructors, before the explicit resource dies.
  host_buffer_fence resource_fence{stream};
  {
    auto const first_path  = directory.file("owned-first.vortex");
    auto const second_path = directory.file("owned-second.vortex");
    std::vector<cudf::io::table_with_metadata> results;
    {
      ndsh::vortex_io io{stream, rmm::device_async_resource_ref{resource}};
      write_source(io, first_path, stream, 0, 37, 0, 37);
      write_source(io, second_path, stream, 10000, 37, 0, 13);
      // Exercise both owning-copy and concatenation paths, including boolean import scratch.
      results.push_back(read_completed(io, first_path, stream, 64));
      // A different file must not overwrite buffers held by the earlier returned table.
      results.push_back(read_completed(io, second_path, stream, 7));
    }
    require(std::filesystem::remove(first_path), "Failed to remove first owned-read file");
    require(std::filesystem::remove(second_path), "Failed to remove second owned-read file");
    check_table(results[0], make_fixture(0, 37), stream);
    check_table(results[1], make_fixture(10000, 37), stream);
  }
  resource_fence.wait();
}

template <typename Function>
void expect_error(char const* label, Function&& function, bool match_message = false)
{
  try {
    function();
  } catch (std::exception const& error) {
    require(!match_message || std::string_view{error.what()}.find(label) != std::string_view::npos,
            std::string{"Unexpected error: "} + error.what());
    return;
  }
  throw std::runtime_error(std::string{label} + ": expected an exception");
}

void test_errors(temporary_directory const& directory, cudaStream_t stream)
{
  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(schema.get(), 0));
  for (auto const flags : {int64_t{0}, int64_t{ARROW_FLAG_NULLABLE}}) {
    schema->flags = flags;
    // Rejection is schema-level, independent of whether any batches/rows exist.
    expect_error("read_vortex requires at least one column",
                 [&] { ndsh::detail::check_flat_schema(*schema.get()); }, true);
  }
  auto empty = make_fixture(0, 0);
  ndsh::detail::check_flat_schema(*empty.schema.get());

  auto source = make_source(stream, 0, 9);
  auto names  = column_names();
  ndsh::vortex_io io{stream};
  auto const zero_path = directory.file("zero-columns.vortex");
  expect_error("write_vortex requires at least one column",
               [&] { io.write_vortex(zero_path, cudf::table_view{}, {}); }, true);
  require(!std::filesystem::exists(zero_path), "Rejected zero-column write created a file");

  auto const path = directory.file("nul-column-name.vortex");
  for (auto const& name : {std::string{"a\0b", 3}, std::string{"\0a", 2}, std::string{"a\0", 2}}) {
    names.back() = name;
    expect_error("column names must not contain NUL bytes",
                 [&] { io.write_vortex(path, source->view(), names, 3); }, true);
    require(!std::filesystem::exists(path), "Rejected column name created a file");
  }
  names = column_names();
  io.write_vortex(path, source->view(), names, 3);
  expect_error("unknown CUDA scan column",
               [&] { static_cast<void>(io.read_vortex(path, 7, {"i32", "missing"})); }, true);
  std::vector<std::size_t> const selected{11, 16, 0, 8, 18, 12};
  auto projected = read_completed(io, path, stream, 7, selected);
  check_table(projected, make_fixture(0, 9, selected), stream, selected);
  for (std::size_t c = 0; c < selected.size(); ++c) {
    require(projected.metadata.schema_info[c].is_nullable ==
              source->view().column(static_cast<cudf::size_type>(selected[c])).nullable(),
            "Projected nullability metadata differs");
  }
  expect_error("duplicate CUDA scan column",
               [&] { static_cast<void>(io.read_vortex(path, 7, {"i32", "i32"})); }, true);

  for (auto const chunk_rows : {0, -1}) {
    auto const name = chunk_rows == 0 ? "invalid-zero.vortex" : "invalid-negative.vortex";
    expect_error(name, [&] {
      io.write_vortex(directory.file(name), source->view(), names, chunk_rows);
    });
  }
  for (auto const count : {names.size() - 1, names.size() + 1}) {
    auto invalid_names = names;
    invalid_names.resize(count, "extra");
    auto const name = count < names.size() ? "invalid-few.vortex" : "invalid-many.vortex";
    expect_error(name, [&] {
      io.write_vortex(directory.file(name), source->view(), invalid_names, 3);
    });
  }
  auto const missing = directory.file("does-not-exist.vortex");
  require(!std::filesystem::exists(missing), "Missing-path test unexpectedly has a file");
  expect_error("missing read path", [&] { static_cast<void>(io.read_vortex(missing, 3)); });

  auto const valid = directory.file("after-errors.vortex");
  io.write_vortex(valid, source->view(), names, 3);
  check_table(read_completed(io, valid, stream, 2), make_fixture(0, 9), stream);
}

}  // namespace

int main()
{
  std::string current_test = "initialization";
  try {
    check_cuda(cudaSetDevice(0), "select CUDA device 0");
    // Never replace the process's current resource: it and this stream outlive all contexts/tables.
    cuda_stream stream;
    temporary_directory directory;
    int passed = 0;
    auto run   = [&](char const* name, auto&& test) {
      current_test = name;
      test();
      check_cuda(cudaStreamSynchronize(stream.get()), "complete test allocations/deallocations");
      ++passed;
      std::cout << "[PASS] " << name << '\n';
    };
    run("bounded string host staging", [&] { test_staged_string_bytes(stream.get()); });
    run("explicit async resource, owning reads, and stream completion",
        [&] { test_async_owned_reads(directory, stream.get()); });
    for (auto const& test : std::array<round_trip_case, 5>{{
           {"defaults", 37, 0, std::nullopt, std::nullopt},
           {"sliced", 69, 5, 11, 7},
           {"word-boundary", 520, 0, 520, 507},
           {"empty", 0, 0, 7, 0},
           {"low-cardinality", 8193, 0, 2048, 0}}}) {
      run(test.name, [&] { round_trip(directory, stream.get(), test); });
    }
    run("zero columns, invalid names/arguments, missing path, projection, and context reuse",
        [&] { test_errors(directory, stream.get()); });
    std::cout << "vortex_io: " << passed << " tests passed on CUDA device 0 (non-default stream)\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << "[FAIL] " << current_test << ": " << error.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "[FAIL] " << current_test << ": unknown exception\n";
    return 1;
  }
}
