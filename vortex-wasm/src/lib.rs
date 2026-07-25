// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Embedded WebAssembly decoder kernels for the Vortex file format.
//!
//! A Vortex file can carry the *decoder* for an array encoding inside the file as a sandboxed
//! WebAssembly module. The data itself is written in the **normal serialized array format** by the
//! native encoding; the kernel is a portable fallback decoder for readers that were never compiled
//! against that encoding. At read time the kernels are merged into the session's encoding registry
//! ([`register_wasm_encodings`]) — a native decoder always supersedes a kernel — and unknown
//! encodings then deserialize through the sandboxed kernel transparently.
//!
//! - [`abi`] defines the host/guest ABI constants.
//! - [`convert`] moves arrays across the boundary in Vortex's own canonical layouts.
//! - [`WasmKernel`] is the `wasmtime`-backed runtime that drives the ABI.
//! - [`WasmEncodingPlugin`] adapts a kernel into a session-registered array encoding.
//!
//! See `docs/design/wasm-encodings.md` for the full design.

pub mod abi;
mod convert;
mod kernel;
mod plugin;

pub use kernel::ChildDescriptor;
pub use kernel::ChildMode;
pub use kernel::KernelOutput;
pub use kernel::WasmDecoder;
pub use kernel::WasmKernel;
pub use plugin::WasmEncodingPlugin;
pub use plugin::register_wasm_encodings;
