// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::FixedWidthArray;
use super::with_values_handle;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::validity::Validity;

/// Masks a fixed-width array by intersecting its validity with `mask`.
///
/// Masking only ever narrows validity, so the records buffer is reused untouched and the result
/// shares the input's allocation.
pub(crate) fn mask<V: FixedWidthArray>(
    array: ArrayView<'_, V>,
    mask: &ArrayRef,
) -> VortexResult<Option<ArrayRef>> {
    let validity = array.validity()?.and(Validity::Array(mask.clone()))?;
    let masked = with_values_handle(array, V::values_handle(array), array.len(), validity)?;
    Ok(Some(masked.into_array()))
}
