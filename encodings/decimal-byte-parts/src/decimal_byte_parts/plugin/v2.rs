// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde for DBP values with unsigned lower parts.

use prost::Message as _;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayRepresentation;
use vortex_array::ArrayRepresentationView;
use vortex_array::ArraySerialization;
use vortex_array::ArraySlots;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use super::decimal_byte_parts_v2_id;
use crate::DecimalBytePartsData;
use crate::decimal_byte_parts::MAX_LOWER_PARTS;

pub(super) type DecimalBytePartsV2 = ArrayRepresentation<DecimalBytePartsData>;
pub(super) type DecimalBytePartsV2View<'a> = ArrayRepresentationView<'a, &'a DecimalBytePartsData>;

/// Metadata for decimal byte parts with per-child storage types.
#[derive(Clone, prost::Message)]
pub struct DecimalBytePartsV2Metadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub(super) zeroth_child_ptype: i32,
    #[prost(uint32, tag = "2")]
    pub(super) lower_part_count: u32,
    /// Unsigned storage types of the lower parts, most significant first.
    #[prost(enumeration = "PType", repeated, tag = "3")]
    pub(super) lower_part_ptypes: Vec<i32>,
}

pub(super) fn serialize(array: &DecimalBytePartsV2View<'_>) -> VortexResult<ArraySerialization> {
    let children = array
        .slots()
        .iter()
        .map(|slot| {
            slot.as_ref()
                .vortex_expect("v2 children are present")
                .clone()
        })
        .collect::<Vec<_>>();
    vortex_ensure!(
        (2..=MAX_LOWER_PARTS + 1).contains(&children.len()),
        "v2 requires lower parts"
    );
    let metadata = DecimalBytePartsV2Metadata {
        zeroth_child_ptype: PType::try_from(children[0].dtype())? as i32,
        lower_part_count: u32::try_from(children.len() - 1)
            .map_err(|_| vortex_err!("lower part count exceeds u32"))?,
        lower_part_ptypes: children[1..]
            .iter()
            .map(|part| PType::try_from(part.dtype()).map(|ptype| ptype as i32))
            .collect::<VortexResult<_>>()?,
    }
    .encode_to_vec();
    Ok(ArraySerialization::new(
        array.id(),
        metadata,
        vec![],
        children,
    ))
}

pub(super) fn deserialize(parts: ArrayDeserialization<'_>) -> VortexResult<DecimalBytePartsV2> {
    vortex_ensure!(
        parts.serialized_id == decimal_byte_parts_v2_id(),
        "expected the v2 format"
    );
    let metadata = DecimalBytePartsV2Metadata::decode(parts.metadata)?;
    vortex_ensure!(
        parts.dtype.as_decimal_opt().is_some(),
        "expected a decimal dtype"
    );
    let lower_part_count = usize::try_from(metadata.lower_part_count)
        .map_err(|_| vortex_err!("lower part count out of range"))?;
    vortex_ensure!(
        (1..=MAX_LOWER_PARTS).contains(&lower_part_count),
        "v2 must carry between one and {MAX_LOWER_PARTS} lower parts"
    );
    vortex_ensure!(
        metadata.lower_part_ptypes.len() == lower_part_count,
        "expected {lower_part_count} lower-part dtypes, got {}",
        metadata.lower_part_ptypes.len()
    );
    vortex_ensure!(
        parts.children.len() == 1 + lower_part_count,
        "expected {} children, got {}",
        1 + lower_part_count,
        parts.children.len()
    );

    let msp_ptype = PType::try_from(metadata.zeroth_child_ptype)?;
    vortex_ensure!(
        msp_ptype.is_signed_int(),
        "MSP must have a signed integer dtype"
    );
    let msp_dtype = DType::Primitive(msp_ptype, parts.dtype.nullability());
    let mut slots = ArraySlots::with_capacity(parts.children.len());
    slots.push(Some(parts.children.get(0, &msp_dtype, parts.len)?));
    for (idx, raw_ptype) in metadata.lower_part_ptypes.into_iter().enumerate() {
        let ptype = PType::try_from(raw_ptype)
            .map_err(|_| vortex_err!("invalid PType {raw_ptype} for lower part {idx}"))?;
        vortex_ensure!(
            ptype.is_unsigned_int(),
            "lower part {idx} must have an unsigned integer dtype, got {ptype}"
        );
        slots.push(Some(parts.children.get(
            1 + idx,
            &DType::Primitive(ptype, Nullability::NonNullable),
            parts.len,
        )?));
    }
    Ok(ArrayRepresentation::new(
        decimal_byte_parts_v2_id(),
        parts.dtype.clone(),
        parts.len,
        DecimalBytePartsData,
        slots,
    ))
}
