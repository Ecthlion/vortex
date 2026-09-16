// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::FixedWidthArray;
use super::with_values_handle;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayView;
use crate::validity::Validity;

/// Intersects the validity of `array` with `validity`.
///
/// Masking only ever narrows validity, so the records buffer is reused untouched and the result
/// shares the input's allocation. The result's nullability follows the intersected validity, which
/// is how a non-nullable input becomes nullable.
pub(crate) fn mask_validity<V: FixedWidthArray>(
    array: ArrayView<'_, V>,
    validity: Validity,
) -> VortexResult<Array<V>> {
    let validity = array.validity()?.and(validity)?;
    with_values_handle(array, V::values_handle(array), array.len(), validity)
}

/// Masks a fixed-width array by a boolean array of valid positions.
pub(crate) fn mask<V: FixedWidthArray>(
    array: ArrayView<'_, V>,
    mask: &ArrayRef,
) -> VortexResult<Option<ArrayRef>> {
    let masked = mask_validity(array, Validity::Array(mask.clone()))?;
    Ok(Some(masked.into_array()))
}
