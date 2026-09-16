// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Physical record widths and the structural kernels specialised for each of them.
//!
//! A [`Record`] is an opaque, fully initialised run of bytes with a compile-time size and a
//! natural alignment. The shared fixed-width kernels never inspect a record's contents: they only
//! move whole records, so a `u32`, an `f32`, and a 4-byte decimal all run the same [`Record4`]
//! kernels. Dispatching a runtime byte width to one of these types happens once, in
//! [`match_each_record_width!`](super::match_each_record_width), and everything downstream is
//! monomorphised on the record type.

use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::ByteBuffer;
use vortex_mask::MaskValues;

use super::take::take_values_scalar;
use crate::arrays::filter::filter_buffer;
use crate::dtype::UnsignedPType;

/// A fixed-width record with a family of structural kernels specialised for its width.
///
/// # Safety
///
/// Implementors must be exactly [`BYTES`](Record::BYTES) bytes with no padding, aligned to
/// [`ALIGNMENT`](Record::ALIGNMENT), and every bit pattern must be a valid value. This lets any
/// suitably aligned byte buffer be viewed as `[Self]` and back without copying, and lets the SIMD
/// kernels move records through same-width integer lanes.
pub(crate) unsafe trait Record: Copy + Send + Sync + 'static {
    /// The number of bytes each record occupies.
    const BYTES: usize;

    /// The alignment a byte buffer needs before it can be viewed as records of this type.
    ///
    /// This matches the natural alignment of the logical types stored at this width, so the
    /// canonical Primitive and Decimal buffers always satisfy it.
    const ALIGNMENT: Alignment;

    /// Gathers `indices` from `values`.
    ///
    /// # Panics
    ///
    /// Panics if any index is out of bounds for `values`.
    fn take<I: UnsignedPType>(
        values: &[Self],
        indices: &[I],
        allocator: &BufferAllocatorRef,
    ) -> Buffer<Self>;

    /// Keeps the records selected by `mask`, compacting in place when `values` is uniquely
    /// owned.
    fn filter(
        values: Buffer<Self>,
        mask: &MaskValues,
        allocator: &BufferAllocatorRef,
    ) -> Buffer<Self>;

    /// Views `bytes` as records without copying.
    ///
    /// Returns the buffer unchanged when its pointer is not aligned to [`ALIGNMENT`] or its
    /// length is not a whole number of records, so callers can fall back to a byte-wise kernel.
    ///
    /// [`ALIGNMENT`]: Record::ALIGNMENT
    fn view(bytes: ByteBuffer) -> Result<Buffer<Self>, ByteBuffer> {
        if Self::ALIGNMENT.is_ptr_aligned(bytes.as_ptr()) && bytes.len().is_multiple_of(Self::BYTES)
        {
            Ok(Buffer::from_byte_buffer_aligned(bytes, Self::ALIGNMENT))
        } else {
            Err(bytes)
        }
    }
}

/// Defines a record type of `$bytes` bytes aligned to `$align`, with its `take` and `filter`
/// kernels. Widths with an AVX2 gather lane name it with `gather = <lane>`.
macro_rules! record {
    // Widths with a gather lane ride the AVX2 gather; every other width still runs the scalar
    // loop compiled with AVX2 enabled so it keeps the wider vector moves.
    (@avx2_take $lane:ty) => {
        super::take::avx2::take_avx2::<Self, $lane, I>
    };
    (@avx2_take) => {
        super::take::avx2::take_scalar_avx2::<Self, I>
    };
    (
        $(#[$meta:meta])*
        $name:ident, $bytes:literal, align = $align:literal $(, gather = $lane:ty)?
    ) => {
        $(#[$meta])*
        #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
        #[repr(C, align($align))]
        pub(crate) struct $name([u8; $bytes]);

        const _: () = {
            assert!(usize::is_power_of_two($bytes), "record widths are powers of two");
            assert!(size_of::<$name>() == $bytes, "record must not carry padding");
            assert!(align_of::<$name>() == $align, "record alignment mismatch");
        };

        impl $name {
            #[cfg(test)]
            #[allow(dead_code, reason = "tests construct only some record widths")]
            pub(crate) const fn from_bytes(bytes: [u8; $bytes]) -> Self {
                Self(bytes)
            }
        }

        // SAFETY: A `#[repr(C)]` byte array newtype has no padding, and its size and alignment
        // are asserted above. Every bit pattern is a valid byte array.
        unsafe impl Record for $name {
            const BYTES: usize = $bytes;
            const ALIGNMENT: Alignment = Alignment::new($align);

            #[inline]
            fn take<I: UnsignedPType>(
                values: &[Self],
                indices: &[I],
                allocator: &BufferAllocatorRef,
            ) -> Buffer<Self> {
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                if *super::take::HAS_AVX2 {
                    // SAFETY: AVX2 was detected above, and `Record` guarantees an initialised
                    // representation the same size as any gather lane it names.
                    return unsafe { record!(@avx2_take $($lane)?)(values, indices, allocator) };
                }
                take_values_scalar(values, indices, allocator)
            }

            #[inline]
            fn filter(
                values: Buffer<Self>,
                mask: &MaskValues,
                allocator: &BufferAllocatorRef,
            ) -> Buffer<Self> {
                // `filter_buffer` picks between in-place compaction, cached indices/slices,
                // SIMD compress, byte-compress, and bitmap iteration based on the record width
                // and mask density.
                filter_buffer(values, mask, allocator)
            }
        }
    };
}

record!(
    /// A 1-byte record: `u8`, `i8`, and 1-byte decimals.
    Record1,
    1,
    align = 1
);
record!(
    /// A 2-byte record: `u16`, `i16`, `f16`, and 2-byte decimals.
    Record2,
    2,
    align = 2
);
record!(
    /// A 4-byte record: `u32`, `i32`, `f32`, and 4-byte decimals. Gathers through a `u32` lane.
    Record4,
    4,
    align = 4,
    gather = u32
);
record!(
    /// An 8-byte record: `u64`, `i64`, `f64`, and 8-byte decimals. Gathers through a `u64` lane.
    Record8,
    8,
    align = 8,
    gather = u64
);
record!(
    /// A 16-byte record: `i128` decimals and binary views.
    Record16,
    16,
    align = 16
);
record!(
    /// A 32-byte record: `i256` decimals.
    ///
    /// `i256` is a pair of 128-bit halves, so its buffers are only guaranteed 16-byte alignment.
    Record32,
    32,
    align = 16
);

#[cfg(test)]
#[allow(
    clippy::host_endian_bytes,
    reason = "records are compared against the in-memory representation of the source buffer"
)]
mod tests {
    use vortex_buffer::Alignment;
    use vortex_buffer::buffer;

    use super::*;
    use crate::dtype::i256;

    #[test]
    fn record_alignment_matches_canonical_buffers() {
        assert!(Alignment::of::<u16>().is_aligned_to(Record2::ALIGNMENT));
        assert!(Alignment::of::<u32>().is_aligned_to(Record4::ALIGNMENT));
        assert!(Alignment::of::<u64>().is_aligned_to(Record8::ALIGNMENT));
        assert!(Alignment::of::<i128>().is_aligned_to(Record16::ALIGNMENT));
        assert!(Alignment::of::<i256>().is_aligned_to(Record32::ALIGNMENT));
    }

    #[test]
    fn view_is_zero_copy_for_aligned_bytes() {
        let bytes = buffer![1u32, 2, 3].into_byte_buffer();
        let ptr = bytes.as_ptr();
        let records = Record4::view(bytes).expect("a u32 buffer is 4-byte aligned");
        assert_eq!(records.len(), 3);
        assert_eq!(records.as_ptr().cast::<u8>(), ptr);
        assert_eq!(records[1], Record4::from_bytes(2u32.to_ne_bytes()));
    }

    #[test]
    fn view_rejects_misaligned_bytes() {
        let bytes = buffer![0u8; 12].into_byte_buffer();
        let misaligned = bytes.slice_unaligned(1..9);
        assert!(Record4::view(misaligned).is_err());
    }

    #[test]
    fn view_rejects_partial_records() {
        let bytes = buffer![0u32; 3].into_byte_buffer();
        assert!(Record8::view(bytes).is_err());
    }
}
