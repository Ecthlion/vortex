// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Guest allocator API.
//!
//! The v2 ABI has no host callbacks during decode — the host pushes buffers and decoded children
//! into guest memory up front — so the only export the SDK needs beyond the entry points is the
//! allocator backing `vx_alloc`.

use alloc::vec::Vec;

/// Allocate `len` bytes in linear memory and return the offset.
///
/// Backs the `vx_alloc` export the host calls to place inputs, buffers, and child arrays into
/// guest memory, and is available to kernels for their own scratch/output buffers. The allocation
/// is deliberately leaked — with the SDK's grow-only bump allocator (the `runtime` feature)
/// nothing is ever freed; the whole linear memory is reclaimed when the per-decode instance is
/// dropped.
pub fn alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len.max(1));
    let ptr = buf.as_mut_ptr();
    core::mem::forget(buf);
    ptr
}

/// Allocate `bytes.len()` bytes, copy `bytes` in, and return the offset.
pub fn alloc_bytes(bytes: &[u8]) -> u32 {
    let ptr = alloc(bytes.len().max(1));
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
    ptr as u32
}
