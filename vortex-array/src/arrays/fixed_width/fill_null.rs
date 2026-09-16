// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;

/// Overwrites every position set in `is_invalid` with `fill`.
///
/// This is generic over the value type rather than the encoding, because filling is the same
/// operation whatever the records mean. The allocation is reused when `values` is uniquely owned,
/// so a freshly decoded array is filled in place.
pub(crate) fn fill_invalid<T: Copy>(
    values: Buffer<T>,
    fill: T,
    is_invalid: &BitBuffer,
) -> Buffer<T> {
    let mut values = values.into_mut();
    for invalid_index in is_invalid.set_indices() {
        values[invalid_index] = fill;
    }
    values.freeze()
}
