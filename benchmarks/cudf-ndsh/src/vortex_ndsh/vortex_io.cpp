/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vortex_io.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/interop.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtx3/nvtx3.hpp>

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.hpp>

// Use cuDF's nanoarrow definitions instead of redeclaring the Arrow C ABI structs.
#define USE_OWN_ARROW
using FFI_ArrowSchema      = ArrowSchema;
using FFI_ArrowArray       = ArrowArray;
using FFI_ArrowArrayStream = ArrowArrayStream;
#include <vortex_cuda.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace ndsh {
namespace {

using cudf_stream = decltype(cudf::get_default_stream());
using session_ptr = std::unique_ptr<vx_session, decltype(&vx_session_free)>;
using array_ptr   = std::unique_ptr<vx_array const, decltype(&vx_array_free)>;
using dtype_ptr   = std::unique_ptr<vx_dtype const, decltype(&vx_dtype_free)>;
using sink_ptr    = std::unique_ptr<vx_array_sink, decltype(&vx_array_sink_abort)>;

void check_cuda(cudaError_t status)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string{"Vortex I/O CUDA error: "} + cudaGetErrorString(status));
  }
}

void check_device()
{
  int device = -1;
  check_cuda(cudaGetDevice(&device));
  if (device != 0) {
    throw std::invalid_argument("Vortex benchmark I/O currently requires CUDA device 0");
  }
}

void retain_vortex_cuda_pool_memory()
{
  std::uint64_t release_threshold = 8ULL << 30;
  cudaMemPool_t pool{};
  check_cuda(cudaDeviceGetMemPool(&pool, 0));
  check_cuda(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &release_threshold));
}

vx_view path_view(std::string const& path)
{
  if (path.empty() || path.find('\0') != std::string::npos) {
    throw std::invalid_argument("Vortex I/O requires a nonempty local path without NUL bytes");
  }
  return {path.data(), path.size()};
}

void check_error(vx_error*& error, std::string const& operation)
{
  auto owned = std::unique_ptr<vx_error, decltype(&vx_error_free)>{std::exchange(error, nullptr),
                                                                   vx_error_free};
  if (owned) {
    auto const message = vx_error_message(owned.get());
    throw std::runtime_error(operation + ": " + std::string{message.ptr, message.len});
  }
}

// Declare after buffer owners to drain before unwinding releases them. Failed
// synchronization is fatal: producer memory may still be in use.
class stream_drain {
 public:
  explicit stream_drain(cudaStream_t stream) : stream_{stream} {}
  ~stream_drain()
  {
    if (pending_ && cudaStreamSynchronize(stream_) != cudaSuccess) { std::terminate(); }
  }
  void wait()
  {
    check_cuda(cudaStreamSynchronize(stream_));
    pending_ = false;
  }

 private:
  cudaStream_t stream_;
  bool pending_{true};
};

struct device_stream {
  nanoarrow::device::UniqueDeviceArrayStream value;
  void check(int status, std::string const& operation)
  {
    if (status != 0) {
      auto const* error = value->get_last_error ? value->get_last_error(value.get()) : nullptr;
      throw std::runtime_error(operation + ": " + (error ? error : "Arrow device stream failed"));
    }
  }
};

struct device_batch {
  nanoarrow::device::UniqueDeviceArray value;
  // Destroy the view and its deleter-owned import scratch before releasing Arrow buffers.
  std::optional<cudf::unique_table_view_t> view;

  device_batch()                        = default;
  device_batch(device_batch&&) noexcept = default;
  // Vector growth only needs move construction; memberwise assignment would release value first.
  device_batch& operator=(device_batch&&) = delete;
};

}  // namespace

// Implementation details with external linkage for focused tests, not public adapter API.
namespace detail {

cudf::unique_device_array_t stage_host_chunk(cudf::table_view chunk,
                                             cudaStream_t stream,
                                             rmm::device_async_resource_ref mr)
{
  cudf::unique_device_array_t host{nullptr, nullptr};
  {
    std::vector<std::unique_ptr<cudf::column>> owned_strings;
    std::vector<cudf::column_view> columns;
    columns.reserve(chunk.num_columns());
    owned_strings.reserve(chunk.num_columns());
    stream_drain drain{stream};
    for (auto const& column : chunk) {
      if (column.type().id() == cudf::type_id::STRING && column.size() != 0) {
        auto const strings = cudf::strings_column_view{column};
        if (strings.offset() != 0 || strings.size() != strings.offsets().size() - 1) {
          // Compact the slice: host export otherwise copies the parent's entire chars buffer.
          owned_strings.push_back(std::make_unique<cudf::column>(column, cudf_stream{stream}, mr));
          columns.push_back(owned_strings.back()->view());
          continue;
        }
      }
      columns.push_back(column);
    }
    host = cudf::to_arrow_host(cudf::table_view{columns}, cudf_stream{stream}, mr);
    drain.wait();
  }
  // Include async frees of the staging copies.
  check_cuda(cudaStreamSynchronize(stream));
  return host;
}

void check_flat_schema(ArrowSchema const& schema)
{
  if (!schema.format || std::string_view{schema.format} != "+s" || schema.n_children < 0 ||
      schema.n_children > std::numeric_limits<cudf::size_type>::max()) {
    throw std::runtime_error("read_vortex requires a table-shaped Arrow schema");
  }
  if (schema.n_children == 0) {
    throw std::runtime_error("read_vortex requires at least one column");
  }
  for (int64_t i = 0; i < schema.n_children; ++i) {
    auto const& field  = *schema.children[i];
    auto const& values = field.dictionary ? *field.dictionary : field;
    if (values.n_children != 0 || values.dictionary) {
      throw std::runtime_error("read_vortex currently supports flat columns only");
    }
  }
}

}  // namespace detail

namespace {

std::unique_ptr<cudf::table> empty_table(ArrowSchema const& schema,
                                         cudaStream_t stream,
                                         rmm::device_async_resource_ref mr)
{
  nanoarrow::UniqueArray empty;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(empty.get(), &schema, nullptr));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(empty.get()));
  NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuildingDefault(empty.get(), nullptr));
  std::unique_ptr<cudf::table> result;
  stream_drain drain{stream};
  result = cudf::from_arrow(&schema, empty.get(), cudf_stream{stream}, mr);
  drain.wait();
  return result;
}

}  // namespace

struct vortex_io::impl {
  session_ptr session{nullptr, vx_session_free};
  cudaStream_t stream;
  rmm::device_async_resource_ref mr;

  impl(cudaStream_t stream, rmm::device_async_resource_ref mr) : stream{stream}, mr{mr}
  {
    check_device();
    retain_vortex_cuda_pool_memory();
    // Validate the supplied consumer stream before creating Vortex's independent stream pool.
    unsigned int flags = 0;
    check_cuda(cudaStreamGetFlags(stream, &flags));
    vx_error* error = nullptr;
    session.reset(vx_cuda_session_new(&error));
    check_error(error, "create Vortex CUDA session");
    if (!session) { throw std::runtime_error("Vortex CUDA session creation returned null"); }
  }
};

vortex_io::vortex_io(cudaStream_t stream, rmm::device_async_resource_ref mr)
  : impl_{std::make_unique<impl>(stream, mr)}
{
}

vortex_io::~vortex_io() = default;

void vortex_io::write_vortex(std::string const& path,
                             cudf::table_view table,
                             std::vector<std::string> const& column_names,
                             cudf::size_type chunk_rows) const
{
  check_device();
  auto const file_path = path_view(path);
  if (chunk_rows <= 0) { throw std::invalid_argument("write_vortex chunk_rows must be positive"); }
  if (column_names.size() != static_cast<std::size_t>(table.num_columns())) {
    throw std::invalid_argument("write_vortex requires one name per column");
  }
  if (table.num_columns() == 0) {
    throw std::invalid_argument("write_vortex requires at least one column");
  }
  for (auto const& column : table) {
    auto const id = column.type().id();
    if (id == cudf::type_id::LIST || id == cudf::type_id::STRUCT ||
        id == cudf::type_id::DICTIONARY32 || id == cudf::type_id::EMPTY) {
      throw std::invalid_argument(
        "write_vortex supports flat, typed, non-dictionary input columns");
    }
  }
  std::vector<cudf::column_metadata> metadata;
  metadata.reserve(column_names.size());
  for (auto const& name : column_names) {
    if (name.find('\0') != std::string::npos) {
      throw std::invalid_argument("write_vortex column names must not contain NUL bytes");
    }
    metadata.emplace_back(name);
  }

  sink_ptr sink{nullptr, vx_array_sink_abort};
  vx_error* error        = nullptr;
  auto const operation   = "write_vortex(" + path + ")";
  cudf::size_type offset = 0;
  do {
    auto const end    = offset + std::min(chunk_rows, table.num_rows() - offset);
    auto const chunks = cudf::slice(table, {offset, end}, cudf_stream{impl_->stream});
    // Use the original schema for every slice, including slices with no nulls.
    auto schema = cudf::to_arrow_schema(table, metadata);
    auto host   = detail::stage_host_chunk(chunks.front(), impl_->stream, impl_->mr);
    if (host->device_type != ARROW_DEVICE_CPU) {
      throw std::runtime_error(operation + ": cuDF host export returned non-host data");
    }
    auto array = array_ptr{
      vx_array_from_arrow(impl_->session.get(), &host->array, schema.get(), false, &error),
      vx_array_free};
    check_error(error, operation);
    if (!array) { throw std::runtime_error(operation + ": Arrow import returned null"); }
    if (!sink) {
      auto dtype = dtype_ptr{vx_array_dtype(array.get()), vx_dtype_free};
      sink.reset(vx_cuda_array_sink_open_file_block_rows(impl_->session.get(),
                                                         file_path,
                                                         dtype.get(),
                                                         static_cast<std::size_t>(chunk_rows),
                                                         &error));
      check_error(error, operation);
      if (!sink) { throw std::runtime_error(operation + ": writer returned null"); }
    }
    vx_array_sink_push(sink.get(), array.get(), &error);
    check_error(error, operation);
    offset = end;
  } while (offset < table.num_rows());
  // close consumes the sink even on failure.
  vx_array_sink_close(sink.release(), &error);
  check_error(error, operation);
}

cudf::io::table_with_metadata vortex_io::read_vortex(std::string const& path,
                                                     std::size_t batch_rows,
                                                     std::vector<std::string> const& columns,
                                                     bool direct_io) const
{
  nvtx3::scoped_range read_range{"vortex.read"};
  check_device();
  auto const file_path = path_view(path);
  if (batch_rows > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::invalid_argument("read_vortex batch_rows exceeds the cuDF row limit");
  }
  auto const operation = "read_vortex(" + path + ")";
  device_stream input;
  vx_error* error = nullptr;
  vx_cuda_scan_options options{};
  options.flags = direct_io ? VX_CUDA_SCAN_FLAG_DIRECT_IO : 0;
  options.batch_rows = batch_rows;
  std::vector<vx_view> column_views;
  column_views.reserve(columns.size());
  for (auto const& name : columns) {
    column_views.push_back({name.data(), name.size()});
  }
  // FFI copies names during the call; neither the views nor their bytes escape.
  int status;
  {
    nvtx3::scoped_range range{"vortex.scan_open"};
    status = vx_cuda_scan_path_arrow_device_stream_projected(
      impl_->session.get(),
      file_path,
      &options,
      column_views.empty() ? nullptr : column_views.data(),
      column_views.size(),
      input.value.get(),
      &error);
  }
  check_error(error, operation);
  if (status != 0) { throw std::runtime_error(operation + ": Vortex scan failed"); }

  nanoarrow::UniqueSchema schema;
  input.check(input.value->get_schema(input.value.get(), schema.get()), operation);
  detail::check_flat_schema(*schema.get());
  cudf::io::table_with_metadata result;
  for (int64_t i = 0; i < schema->n_children; ++i) {
    cudf::io::column_name_info info;
    auto const& field = *schema->children[i];
    info.name         = field.name ? field.name : "";
    info.is_nullable  = (field.flags & ARROW_FLAG_NULLABLE) != 0;
    result.metadata.schema_info.push_back(std::move(info));
  }

  std::vector<device_batch> batches;
  stream_drain drain{impl_->stream};
  int64_t rows = 0;
  while (true) {
    batches.emplace_back();
    auto& batch = batches.back();
    {
      nvtx3::scoped_range range{"vortex.get_next"};
      input.check(input.value->get_next(input.value.get(), batch.value.get()), operation);
    }
    if (!batch.value->array.release) {
      batches.pop_back();
      break;
    }
    if (batch.value->device_type != ARROW_DEVICE_CUDA || batch.value->device_id != 0) {
      throw std::runtime_error("read_vortex expected an Arrow CUDA batch on device 0");
    }
    auto const count = batch.value->array.length;
    if (count < 0 || count > std::numeric_limits<cudf::size_type>::max() - rows) {
      throw std::overflow_error(operation + ": result exceeds the cuDF row limit");
    }
    rows += count;
    {
      nvtx3::scoped_range range{"vortex.arrow_device_import"};
      batch.view.emplace(cudf::from_arrow_device(
        schema.get(), batch.value.get(), cudf_stream{impl_->stream}, impl_->mr));
    }
    for (auto const& column : **batch.view) {
      if (column.type().id() == cudf::type_id::DICTIONARY32) {
        throw std::runtime_error(operation + ": decoded scan returned a dictionary column");
      }
    }
  }
  {
    nvtx3::scoped_range range{"vortex.materialize"};
    if (batches.empty()) {
      result.tbl = empty_table(*schema.get(), impl_->stream, impl_->mr);
    } else if (batches.size() == 1) {
      result.tbl = std::make_unique<cudf::table>(
        **batches.front().view, cudf_stream{impl_->stream}, impl_->mr);
    } else {
      std::vector<cudf::table_view> views;
      views.reserve(batches.size());
      for (auto const& batch : batches) {
        views.push_back(**batch.view);
      }
      result.tbl = cudf::concatenate(views, cudf_stream{impl_->stream}, impl_->mr);
    }
  }
  {
    nvtx3::scoped_range range{"vortex.consumer_sync"};
    drain.wait();
  }
  result.metadata.num_rows_per_source = {static_cast<std::size_t>(rows)};
  // Import scratch and Vortex buffers are released only after materialization completes.
  {
    nvtx3::scoped_range range{"vortex.release_batches"};
    batches.clear();
  }
  return result;
}

}  // namespace ndsh
