// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use super::FixedWidthArray;
use super::with_values_handle;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;

/// Slices a fixed-width array to `range` by slicing its records buffer on record boundaries.
///
/// Slicing at a multiple of the record width preserves the buffer's alignment, so the result
/// keeps sharing the input's allocation rather than copying.
pub(crate) fn slice<V: FixedWidthArray>(
    array: ArrayView<'_, V>,
    range: Range<usize>,
) -> VortexResult<Option<ArrayRef>> {
    let byte_width = V::byte_width(array);
    let byte_range = range.start * byte_width..range.end * byte_width;
    let values = V::values_handle(array).slice(byte_range);
    let len = range.len();
    let validity = array.validity()?.slice(range)?;

    let sliced = with_values_handle(array, values, len, validity)?;
    Ok(Some(sliced.into_array()))
}
