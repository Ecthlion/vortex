// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The host/guest ABI shared between [`crate::WasmKernel`] and the `vortex-wasm-guest` SDK.
//!
//! A kernel is the portable decoder for one array encoding, driven with the encoding's **real
//! serialized parts**. All multi-byte integers in the frames are little-endian; the authoritative
//! frame layouts are documented in `vortex-wasm-guest`'s `abi` module and
//! `docs/design/wasm-encodings.md`.

/// Version of the host/guest ABI implemented by this crate.
///
/// The host refuses to run a kernel whose recorded ABI version it does not understand.
pub const ABI_VERSION: u32 = 2;

/// Name of the host import module that the guest links against.
pub const HOST_MODULE: &str = "vortex_host";

/// Name of the guest's exported linear memory.
pub const MEMORY_EXPORT: &str = "memory";

/// Guest export: `vx_alloc(len: i32) -> i32`. Allocates `len` bytes, returns the offset.
pub const ALLOC_EXPORT: &str = "vx_alloc";

/// Guest export: `vx_children(input_ptr: i32, input_len: i32) -> i32`. Given the node header
/// frame (parent length, nullability, serialized child count, metadata), returns the offset of
/// the child-descriptor list — the dtype and length of each serialized child, so the host can
/// decode them. Negative values are error codes.
pub const CHILDREN_EXPORT: &str = "vx_children";

/// Guest export: `vx_decode(input_ptr: i32, input_len: i32) -> i32`. The input frame carries the
/// node's metadata, its raw buffers (already in guest memory), and its host-decoded children (as
/// Arrow C Data Interface struct pairs). Returns the offset of an
/// `(array_ptr: u32, schema_ptr: u32)` pair pointing at the decoded output's Arrow C structs.
/// Negative values are error codes.
pub const DECODE_EXPORT: &str = "vx_decode";

/// Host import: `vx_host_log(ptr: i32, len: i32)`. Logs a UTF-8 string from guest memory.
pub const HOST_LOG_IMPORT: &str = "vx_host_log";
