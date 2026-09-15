// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lossless conversions between current DBP arrays and version-specific storage.

use vortex_array::ArrayRepresentationView;
use vortex_array::ArrayView;
use vortex_error::VortexResult;

use super::decimal_byte_parts_v1_id;
use super::decimal_byte_parts_v2_id;
use super::v1::DecimalBytePartsV1;
use super::v1::DecimalBytePartsV1Data;
use super::v1::DecimalBytePartsV1View;
use super::v2::DecimalBytePartsV2View;
use crate::DecimalByteParts;
use crate::DecimalBytePartsArray;
use crate::DecimalBytePartsArraySlotsExt;
use crate::DecimalBytePartsData;

/// Reuse the single child when the current representation already has the v1 shape.
///
/// An array with lower parts cannot be downgraded by relabeling its MSP: its position gives it
/// a different numerical weight. This conversion does not execute or narrow those arrays.
pub(super) fn try_to_v1(
    array: ArrayView<'_, DecimalByteParts>,
) -> Option<DecimalBytePartsV1View<'_>> {
    if !array.lower_parts().is_empty() {
        return None;
    }
    Some(ArrayRepresentationView::new(
        decimal_byte_parts_v1_id(),
        array.array(),
        DecimalBytePartsV1Data::EMPTY,
    ))
}

/// The v1 child layout is a subset of v2; replace its data and validate using the current VTable.
pub(super) fn from_v1(array: DecimalBytePartsV1) -> VortexResult<DecimalBytePartsArray> {
    array
        .map_data(|_| DecimalBytePartsData)
        .into_array(DecimalByteParts)
}

pub(super) fn to_v2(array: ArrayView<'_, DecimalByteParts>) -> DecimalBytePartsV2View<'_> {
    ArrayRepresentationView::new(decimal_byte_parts_v2_id(), array.array(), array.data())
}
