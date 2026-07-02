// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The guest-side runtime the SDK provides so kernels can be `#![no_std]`: a bump
//! `#[global_allocator]` over linear memory and a trap-on-panic `#[panic_handler]`.
//!
//! Only compiled for `wasm32` targets and only with the default `runtime` feature; a kernel that
//! wants `std` (or its own allocator/panic handler) disables the feature to avoid the duplicate
//! lang items.
//!
//! The allocator never frees: a kernel instance decodes exactly once and its whole linear memory is
//! dropped with the store afterwards, so `dealloc` is a no-op and there is no free-list bookkeeping
//! to ship in the binary. Allocation starts at `__heap_base` (set by the linker, after
//! data/stack) and grows memory on demand.

use core::alloc::GlobalAlloc;
use core::alloc::Layout;
use core::arch::wasm32;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering;

const PAGE: usize = 64 * 1024;

unsafe extern "C" {
    /// First byte after static data + shadow stack; provided by `wasm-ld`.
    static __heap_base: u8;
}

/// Grow-only bump allocator over the module's linear memory.
struct BumpAlloc;

/// Next free offset; 0 means "not yet initialized from `__heap_base`". Kernels are single-threaded,
/// the atomic is only for interior mutability.
static NEXT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for BumpAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let mut next = NEXT.load(Ordering::Relaxed);
        if next == 0 {
            next = &raw const __heap_base as usize;
        }

        let start = match next.checked_add(layout.align() - 1) {
            Some(v) => v & !(layout.align() - 1),
            None => return core::ptr::null_mut(),
        };
        let Some(end) = start.checked_add(layout.size()) else {
            return core::ptr::null_mut();
        };

        let have = wasm32::memory_size(0) * PAGE;
        if end > have {
            let delta = (end - have).div_ceil(PAGE);
            if wasm32::memory_grow(0, delta) == usize::MAX {
                return core::ptr::null_mut();
            }
        }

        NEXT.store(end, Ordering::Relaxed);
        start as *mut u8
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Grow-only: everything is reclaimed when the per-decode instance is dropped.
    }
}

#[global_allocator]
static ALLOCATOR: BumpAlloc = BumpAlloc;

/// Panics become wasm traps, which the host surfaces as decode errors.
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    wasm32::unreachable()
}
