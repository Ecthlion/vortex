// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::arrays::fixed_width::Record;
use crate::arrays::fixed_width::match_each_record_width;
use crate::dtype::UnsignedPType;

/// Takes `indices` from a buffer of `record_count` records that are each `byte_width` bytes.
///
/// Widths with a [`Record`] type view the buffer as records without copying and run that width's
/// kernel. Any other width, or a buffer that is not aligned to its record type, is copied byte
/// by byte.
pub(super) fn take_byte_records<I: UnsignedPType>(
    values: &ByteBuffer,
    byte_width: usize,
    record_count: usize,
    indices: &[I],
    allocator: &BufferAllocatorRef,
) -> VortexResult<ByteBuffer> {
    let alignment = values.alignment();

    match_each_record_width!(
        byte_width,
        |R| {
            if let Ok(records) = R::view(values.clone()) {
                debug_assert_eq!(records.len(), record_count);
                return Ok(R::take(records.as_slice(), indices, allocator)
                    .into_byte_buffer()
                    .aligned(alignment));
            }
        },
        _ => {}
    );

    let output_len = indices
        .len()
        .checked_mul(byte_width)
        .ok_or_else(|| vortex_err!("Fixed-width take output length overflows usize"))?;
    let mut result = BufferMut::<u8>::with_capacity_in(output_len, allocator.clone());
    for index in indices {
        let index = index.as_();
        assert!(
            index < record_count,
            "take index {index} out of bounds for length {record_count}"
        );
        let start = index * byte_width;
        result.extend_from_slice(&values[start..start + byte_width]);
    }
    Ok(result.freeze().into_byte_buffer().aligned(alignment))
}
