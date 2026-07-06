// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Guest SDK for writing Vortex WebAssembly decoder kernels, in Rust.
//!
//! A kernel is the portable decoder for one Vortex array encoding, embedded in a Vortex file and
//! run by `vortex-wasm` when a reader lacks the native encoding. It receives the encoding's
//! **real serialized parts** — the same metadata, buffers, and children a native
//! `VTable::deserialize` gets — so the kernel lives alongside the native encoding (e.g.
//! `encodings/fastlanes/wasm`) and reuses the same pure decode libraries (`fastlanes`, `fsst`).
//!
//! An encoding author builds a `cdylib` for `wasm32-unknown-unknown` that depends on this crate,
//! implements [`WasmEncoding`], and calls [`export_wasm_encoding!`].
//!
//! The SDK is **`#![no_std]` and dependency-free** (`core`/`alloc` only) to keep kernels small —
//! in particular it does not use `vortex-error`. Instead of linking `std` (whose dlmalloc
//! allocator and panic machinery dominate a kernel's size), the SDK provides the runtime itself: a
//! grow-only bump `#[global_allocator]` over linear memory plus a trap-on-panic handler, both
//! behind the default `runtime` feature (disable it when a dependency links `std`, or to bring
//! your own allocator). Decoded arrays cross the host/guest boundary as the
//! [Arrow C Data Interface](crate::arrow), which is plain byte layout, so no Arrow library is
//! needed; prost-encoded metadata is parsed with the tiny [`proto`] reader. Errors use the
//! minimal [`GuestError`].
//!
//! See `docs/design/wasm-encodings.md`.

#![no_std]

extern crate alloc;

pub mod abi;
pub mod arrow;
mod encoding;
mod error;
pub mod host;
pub mod node;
pub mod proto;
#[cfg(all(target_arch = "wasm32", feature = "runtime"))]
mod runtime;

#[doc(hidden)]
pub use encoding::__run_children;
#[doc(hidden)]
pub use encoding::__run_decode;
pub use encoding::WasmEncoding;
pub use error::GuestError;
pub use error::GuestResult;
