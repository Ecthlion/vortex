// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Guest allocator API.
//!
//! The ABI has no host callbacks during decode — the host pushes buffers and decoded children
//! into guest memory up front — so the only export the SDK needs beyond the entry points is the
//! allocator backing `vx_alloc`.

use core::alloc::Layout;

/// Allocate `len` bytes in linear memory, **8-byte aligned**, and return the offset.
///
/// Backs the `vx_alloc` export the host calls to place inputs, buffers, and child arrays into
/// guest memory, and is available to kernels for their own scratch/output buffers. The 8-byte
/// alignment is part of the ABI: every host-uploaded buffer lands aligned, so kernels can view
/// typed data in place (e.g. cast a packed buffer to `&[u32]`) instead of copying it out, and the
/// Arrow C structs' `int64` fields are naturally aligned.
///
/// The allocation is deliberately leaked — with the SDK's grow-only bump allocator (the `runtime`
/// feature) nothing is ever freed; the whole linear memory is reclaimed when the per-decode
/// instance is dropped.
pub fn alloc(len: usize) -> *mut u8 {
    let layout = Layout::from_size_align(len.max(1), 8).expect("allocation too large");
    // SAFETY: the layout has non-zero size.
    let ptr = unsafe { alloc::alloc::alloc(layout) };
    if ptr.is_null() {
        alloc::alloc::handle_alloc_error(layout);
    }
    ptr
}

/// Allocate `bytes.len()` bytes, copy `bytes` in, and return the offset.
pub fn alloc_bytes(bytes: &[u8]) -> u32 {
    let ptr = alloc(bytes.len().max(1));
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
    ptr as u32
}
