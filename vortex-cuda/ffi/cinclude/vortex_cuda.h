// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

// THIS FILE IS AUTO-GENERATED, DO NOT MAKE EDITS DIRECTLY

#include <stddef.h>
#include <stdint.h>

#include "vortex.h"

/* Link against the CUDA-enabled FFI library that provides both the base Vortex FFI and these CUDA
 * entry points. Do not pass Vortex handles between independently linked Rust FFI libraries. */

/* Definitions from the Arrow C Device data interface. Define USE_OWN_ARROW_DEVICE to skip them.
 * https://arrow.apache.org/docs/format/CDeviceDataInterface.html */
#if !defined(ARROW_C_DEVICE_DATA_INTERFACE) && !defined(USE_OWN_ARROW_DEVICE)
#define ARROW_C_DEVICE_DATA_INTERFACE

typedef int32_t ArrowDeviceType;
#define ARROW_DEVICE_CPU          1
#define ARROW_DEVICE_CUDA         2
#define ARROW_DEVICE_CUDA_HOST    3
#define ARROW_DEVICE_OPENCL       4
#define ARROW_DEVICE_VULKAN       7
#define ARROW_DEVICE_METAL        8
#define ARROW_DEVICE_VPI          9
#define ARROW_DEVICE_ROCM         10
#define ARROW_DEVICE_ROCM_HOST    11
#define ARROW_DEVICE_EXT_DEV      12
#define ARROW_DEVICE_CUDA_MANAGED 13
#define ARROW_DEVICE_ONEAPI       14
#define ARROW_DEVICE_WEBGPU       15
#define ARROW_DEVICE_HEXAGON      16

struct ArrowDeviceArray {
    struct ArrowArray array;
    int64_t device_id;
    ArrowDeviceType device_type;
    void *sync_event;
    int64_t reserved[3];
};
#endif

#if !defined(ARROW_C_DEVICE_STREAM_INTERFACE) && !defined(USE_OWN_ARROW_DEVICE)
#define ARROW_C_DEVICE_STREAM_INTERFACE
struct ArrowDeviceArrayStream {
    ArrowDeviceType device_type;
    int (*get_schema)(struct ArrowDeviceArrayStream *, struct ArrowSchema *out);
    int (*get_next)(struct ArrowDeviceArrayStream *, struct ArrowDeviceArray *out);
    const char *(*get_last_error)(struct ArrowDeviceArrayStream *);
    void (*release)(struct ArrowDeviceArrayStream *);
    void *private_data;
};
#endif

/**
 * Bypass the operating system page cache for pooled data-plane reads.
 * Footer and zone-map reads remain buffered. Supported only on Linux.
 */
#define VX_CUDA_SCAN_FLAG_DIRECT_IO (1u << 0)

/**
 * Options for scanning a CUDA-compatible Vortex file.
 *
 * Zero-initialize this struct to use buffered file I/O and layout-derived batch splitting.
 */
typedef struct vx_cuda_scan_options {
    /**
     * A bitwise combination of `VX_CUDA_SCAN_FLAG_*` values. Unknown bits are ignored.
     */
    uint32_t flags;
    /**
     * Maximum rows in each output batch. Zero uses layout-derived splitting.
     * Physical layout boundaries may produce shorter batches.
     */
    size_t batch_rows;
} vx_cuda_scan_options;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Create a CUDA Vortex session.
 *
 * Repeated [`vx_cuda_array_export_arrow_device`] calls reuse this CUDA state. Returns an owned
 * session handle, or null and an optional `vx_error` on failure.
 *
 * # Safety
 *
 * If `error_out` is non-null, it must be valid for writing one error pointer.
 */
vx_session *vx_cuda_session_new(vx_error **error_out);

/**
 * Open a Vortex file sink configured to produce CUDA-readable files.
 *
 * Push host-resident arrays and close or abort the returned sink with the standard
 * `vx_array_sink_*` functions. This function configures the on-disk encodings and layout; it does
 * not move arrays to the GPU during the write.
 *
 * # Safety
 *
 * `session`, `path`, and `dtype` must satisfy the same requirements as
 * `vx_array_sink_open_file`. If `error_out` is non-null, it must be valid for writing one error
 * pointer.
 */
vx_array_sink *vx_cuda_array_sink_open_file(const vx_session *session,
                                            vx_view path,
                                            const vx_dtype *dtype,
                                            vx_error **error_out);

/**
 * Open a CUDA-readable Vortex file sink with a fixed row block size.
 *
 * `block_rows` controls the row granularity of CUDA-flat data blocks. Passing zero uses the default
 * writer strategy: 8,192-row blocks may be coalesced into data blocks targeting 1 MiB. Any nonzero
 * value disables byte-size coalescing and outer layout dictionaries, so passing 8,192 is not
 * equivalent to passing zero.
 *
 * Write and scan sizing are independent; scan batches preserve on-disk layout boundaries.
 *
 * # Safety
 *
 * `session`, `path`, and `dtype` must satisfy the same requirements as
 * `vx_array_sink_open_file`. If `error_out` is non-null, it must be valid for writing one error
 * pointer.
 */
vx_array_sink *vx_cuda_array_sink_open_file_block_rows(const vx_session *session,
                                                       vx_view path,
                                                       const vx_dtype *dtype,
                                                       size_t block_rows,
                                                       vx_error **error_out);

/**
 * Scan a local Vortex file with buffered I/O and export an Arrow C Device stream.
 *
 * Footer and zone-map reads remain on the host. Data segments are staged through pinned host
 * buffers and transferred directly to the GPU.
 *
 * The file must use encodings and layouts supported by the CUDA execution path, such as files
 * written by [`vx_cuda_array_sink_open_file`]. Pinned staging buffers are reused across scans made
 * with the same CUDA session.
 *
 * Dictionaries, including nested children, are always decoded on CUDA to export plain Arrow
 * values with a stable batch schema. This may increase device memory use; device-resident
 * dictionaries require CUDA decoding support. The caller's session policy is unchanged.
 *
 * On success returns `0` and writes an owned [`ArrowDeviceArrayStream`] to `out_stream`. The
 * caller must release the stream and each array produced by it through their embedded Arrow
 * release callbacks.
 *
 * On error returns `1` and, when `error_out` is non-null, writes a `vx_error` (free with
 * `vx_error_free`).
 *
 * # Safety
 *
 * `session` must be a valid borrowed handle created by `vortex-ffi`. `path` must be valid for the
 * duration of this call and contain UTF-8. `out_stream` must be a valid writable pointer. If
 * `error_out` is non-null, it must be valid for writing one error pointer.
 */
int vx_cuda_scan_path_arrow_device_stream(const vx_session *session,
                                          vx_view path,
                                          struct ArrowDeviceArrayStream *out_stream,
                                          vx_error **error_out);

/**
 * Scan a local Vortex file and export an Arrow C Device stream with bounded row batches.
 *
 * `batch_rows` sets the maximum number of rows in each output batch. Physical layout boundaries
 * may produce shorter batches. Passing zero preserves the layout-derived splitting used by
 * [`vx_cuda_scan_path_arrow_device_stream`].
 *
 * Scan and write sizing are independent; scan batches preserve on-disk layout boundaries.
 *
 * # Safety
 *
 * `session` must be a valid borrowed handle created by `vortex-ffi`. `path` must be valid for the
 * duration of this call and contain UTF-8. `out_stream` must be a valid writable pointer. If
 * `error_out` is non-null, it must be valid for writing one error pointer.
 */
int vx_cuda_scan_path_arrow_device_stream_batch_rows(const vx_session *session,
                                                     vx_view path,
                                                     size_t batch_rows,
                                                     struct ArrowDeviceArrayStream *out_stream,
                                                     vx_error **error_out);

/**
 * Scan a local Vortex file with explicit options and export an Arrow C Device stream.
 *
 * This has the same ownership and file compatibility requirements as
 * [`vx_cuda_scan_path_arrow_device_stream`]. Pass a null `options` pointer or a zero-initialized
 * [`vx_cuda_scan_options`] to use buffered file I/O and layout-derived batch splitting.
 * Dictionaries are always decoded as described in [`vx_cuda_scan_path_arrow_device_stream`].
 *
 * # Safety
 *
 * `session` must be a valid borrowed handle created by `vortex-ffi`. `path` must be valid for the
 * duration of this call and contain UTF-8. `options`, when non-null, must point to a valid
 * [`vx_cuda_scan_options`]. `out_stream` must be a valid writable pointer. If `error_out` is
 * non-null, it must be valid for writing one error pointer.
 */
int vx_cuda_scan_path_arrow_device_stream_with_options(const vx_session *session,
                                                       vx_view path,
                                                       const struct vx_cuda_scan_options *options,
                                                       struct ArrowDeviceArrayStream *out_stream,
                                                       vx_error **error_out);

/**
 * Scan selected top-level columns of a local Vortex file as an Arrow C Device stream.
 *
 * This has the same options, ownership, and file compatibility requirements as
 * [`vx_cuda_scan_path_arrow_device_stream_with_options`]. Zero `ncolumns` selects all columns and
 * ignores `columns`. Otherwise, names are case-sensitive literal top-level field names (not field
 * paths), returned in the requested order. Unknown or duplicate names and non-struct file dtypes
 * are rejected. Projection is applied by the scan builder before reading or decoding column data.
 *
 * Names are copied during this call; the stream does not borrow them. The projected schema is
 * available even for a zero-row file. On error, `out_stream` is left unchanged.
 *
 * # Safety
 *
 * `session`, `path`, `options`, `out_stream`, and `error_out` must satisfy the requirements of
 * [`vx_cuda_scan_path_arrow_device_stream_with_options`]. For nonzero `ncolumns`, `columns` must
 * point to that many initialized, aligned [`vx_view`] values. Each name must point to `len`
 * readable bytes for this call, or be null with zero length. Names must contain UTF-8.
 */
int vx_cuda_scan_path_arrow_device_stream_projected(const vx_session *session,
                                                    vx_view path,
                                                    const struct vx_cuda_scan_options *options,
                                                    const vx_view *columns,
                                                    size_t ncolumns,
                                                    struct ArrowDeviceArrayStream *out_stream,
                                                    vx_error **error_out);

/**
 * Export a borrowed Vortex array for cuDF's Arrow Device import path.
 *
 * On success returns `0` and writes independently releasable `out_schema` and `out_array`; the
 * caller passes them to cuDF and releases both via their embedded Arrow callbacks after import. On
 * error returns `1` and, when `error_out` is non-null, writes a `vx_error` (free with
 * `vx_error_free`).
 *
 * `out_array` is exported on `ARROW_DEVICE_CUDA`; struct arrays become table-shaped schemas,
 * non-struct arrays a single column field.
 *
 * Export is stream-ordered; `out_array->sync_event` is valid until `out_array` is released.
 *
 * # Safety
 *
 * `session` and `array` must be valid borrowed handles created by `vortex-ffi`. `out_schema`
 * and `out_array` must be valid writable pointers. If `error_out` is non-null, it must be valid
 * for writing one error pointer.
 */
int vx_cuda_array_export_arrow_device(const vx_session *session,
                                      const vx_array *array,
                                      FFI_ArrowSchema *out_schema,
                                      struct ArrowDeviceArray *out_array,
                                      vx_error **error_out);

/**
 * Consume a Vortex partition and scan it as an Arrow C Device stream.
 *
 * This function takes ownership of `partition`. Callers must not free or reuse it after calling
 * this function, regardless of success or failure.
 *
 * On success returns `0` and writes an owned `ArrowDeviceArrayStream` to `out_stream`. The stream
 * owns the resulting scan iterator. The caller must release the stream through its embedded Arrow
 * `release` callback, and must release each produced `ArrowDeviceArray` through its embedded
 * `ArrowArray.release` callback.
 *
 * On error returns `1` and, when `error_out` is non-null, writes a `vx_error` (free with
 * `vx_error_free`).
 *
 * # Safety
 *
 * `session` must be a valid borrowed handle created by `vortex-ffi`. `partition` must be an owned
 * partition handle created by `vortex-ffi`. `out_stream` must be a valid writable pointer. If
 * `error_out` is non-null, it must be valid for writing one error pointer.
 */
int vx_cuda_partition_scan_arrow_device_stream(const vx_session *session,
                                               vx_partition *partition,
                                               struct ArrowDeviceArrayStream *out_stream,
                                               vx_error **error_out);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
