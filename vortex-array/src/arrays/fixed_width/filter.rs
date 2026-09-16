// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_mask::MaskValues;
use vortex_mask::MaskValuesRef;

use super::FixedWidthArray;
use super::Record;
use super::match_each_record_width;
use super::with_values;
use crate::array::Array;
use crate::arrays::filter::filter_validity;

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests;

pub(crate) fn filter<V: FixedWidthArray>(
    array: &Array<V>,
    mask: &MaskValuesRef,
    allocator: &BufferAllocatorRef,
) -> Array<V> {
    let array = array.as_view();
    let values = filter_records(
        V::values(array),
        V::byte_width(array),
        mask.as_ref(),
        allocator,
    );
    let validity = filter_validity(
        array
            .validity()
            .vortex_expect("validity is derivable for a valid fixed-width array"),
        mask,
    );
    with_values(array, values, mask.true_count(), validity)
        .vortex_expect("filtering fixed-width values preserves array invariants")
}

/// Filters a buffer of records that are each `byte_width` bytes.
///
/// Widths with a [`Record`] type view the buffer as records without copying and run that width's
/// kernel, which compacts in place when `values` is uniquely owned. Any other width, or a buffer
/// that is not aligned to its record type, is compacted byte by byte.
fn filter_records(
    values: ByteBuffer,
    byte_width: usize,
    mask: &MaskValues,
    allocator: &BufferAllocatorRef,
) -> ByteBuffer {
    let alignment = values.alignment();

    let values = match_each_record_width!(
        byte_width,
        |R| {
            match R::view(values) {
                Ok(records) => {
                    return R::filter(records, mask, allocator)
                        .into_byte_buffer()
                        .aligned(alignment);
                }
                Err(values) => values,
            }
        },
        _ => { values }
    );

    match values.try_into_mut() {
        Ok(mut values) => {
            let mut destination = 0;
            mask.bit_buffer().for_each_set_index(|index| {
                let source = index * byte_width;
                values.copy_within(source..source + byte_width, destination);
                destination += byte_width;
            });
            values.truncate(destination);
            values.freeze().into_byte_buffer().aligned(alignment)
        }
        Err(values) => {
            let mut filtered =
                BufferMut::with_capacity_in(mask.true_count() * byte_width, allocator.clone());
            mask.bit_buffer().for_each_set_index(|index| {
                let start = index * byte_width;
                filtered.extend_from_slice(&values[start..start + byte_width]);
            });
            filtered.freeze().into_byte_buffer().aligned(alignment)
        }
    }
}
