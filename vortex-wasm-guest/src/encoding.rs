// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`WasmEncoding`] trait an encoding author implements, and the glue that exports it.

use alloc::vec::Vec;

use crate::data::Decoded;
use crate::data::write;
use crate::error::GuestResult;
use crate::node::ChildSpec;
use crate::node::NodeHeader;
use crate::node::NodeView;
use crate::node::write_child_specs;

/// The wasm decoder for a single Vortex array encoding.
///
/// A kernel is the portable mirror of a native `VTable::deserialize`: it receives the encoding's
/// **real serialized parts** — metadata, raw buffers, and children. Because only the encoding
/// knows its children's dtypes, decode happens in two steps:
///
/// 1. [`children`](Self::children) — from the metadata, declare each serialized child's dtype and
///    length so the host can decode them (natively, or recursively through another kernel).
/// 2. [`decode`](Self::decode) — with buffers and decoded children in hand, produce the decoded
///    output as a [`Decoded`].
///
/// Wire it up with [`export_wasm_encoding!`](crate::export_wasm_encoding).
pub trait WasmEncoding {
    /// Declare the dtype and length of each serialized child, in child order.
    ///
    /// `header.n_children` is the actual number of serialized children — use it to detect
    /// optional trailing children such as a validity bitmap.
    fn children(header: &NodeHeader<'_>) -> GuestResult<Vec<ChildSpec>>;

    /// Decode the node into its canonical output.
    fn decode(node: &NodeView<'_>) -> GuestResult<Decoded>;
}

fn input_slice(in_ptr: i32, in_len: i32) -> &'static [u8] {
    if in_len <= 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(in_ptr as *const u8, in_len as usize) }
    }
}

/// Internal entry point invoked by [`export_wasm_encoding!`]. Not part of the stable API.
#[doc(hidden)]
pub fn __run_children<E: WasmEncoding>(in_ptr: i32, in_len: i32) -> i32 {
    match NodeHeader::parse(input_slice(in_ptr, in_len)).and_then(|h| E::children(&h)) {
        Ok(specs) => write_child_specs(&specs),
        Err(_) => -1,
    }
}

/// Internal entry point invoked by [`export_wasm_encoding!`]. Not part of the stable API.
#[doc(hidden)]
pub fn __run_decode<E: WasmEncoding>(in_ptr: i32, in_len: i32) -> i32 {
    match NodeView::parse(input_slice(in_ptr, in_len)).and_then(|node| E::decode(&node)) {
        Ok(decoded) => write(&decoded),
        Err(_) => -1,
    }
}

/// Export a [`WasmEncoding`] as a complete kernel: defines the `vx_alloc`, `vx_children`, and
/// `vx_decode` exports expected by the host ABI.
///
/// ```ignore
/// struct MyEncoding;
/// impl vortex_wasm_guest::WasmEncoding for MyEncoding { /* ... */ }
/// vortex_wasm_guest::export_wasm_encoding!(MyEncoding);
/// ```
#[macro_export]
macro_rules! export_wasm_encoding {
    ($ty:ty) => {
        /// ABI version export required by the host ABI.
        ///
        /// The host compares this against its own [`abi::ABI_VERSION`](crate::abi::ABI_VERSION)
        /// and refuses to run the kernel if they disagree, so a kernel built against an older SDK
        /// fails loudly instead of misreading frames.
        #[unsafe(no_mangle)]
        pub extern "C" fn vx_abi_version() -> i32 {
            $crate::abi::ABI_VERSION as i32
        }

        /// Guest allocator export required by the host ABI.
        #[unsafe(no_mangle)]
        pub extern "C" fn vx_alloc(len: i32) -> i32 {
            $crate::host::alloc(len.max(0) as usize) as i32
        }

        /// Child-descriptor export required by the host ABI.
        #[unsafe(no_mangle)]
        pub extern "C" fn vx_children(in_ptr: i32, in_len: i32) -> i32 {
            $crate::__run_children::<$ty>(in_ptr, in_len)
        }

        /// Decode entrypoint export required by the host ABI.
        #[unsafe(no_mangle)]
        pub extern "C" fn vx_decode(in_ptr: i32, in_len: i32) -> i32 {
            $crate::__run_decode::<$ty>(in_ptr, in_len)
        }
    };
}
