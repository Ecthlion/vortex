// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The frozen single-child DBP representation and its serde implementation.

use prost::Message as _;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayRef;
use vortex_array::ArrayRepresentation;
use vortex_array::ArrayRepresentationView;
use vortex_array::ArraySerialization;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::smallvec::smallvec;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::decimal_byte_parts_v1_id;

// Retain the old data shape. It needs no VTable, compute kernels, hashing, or validity methods.
#[derive(Debug, Default)]
pub(super) struct DecimalBytePartsV1Data {
    _lower_parts: Vec<ArrayRef>,
}

impl DecimalBytePartsV1Data {
    pub(super) const EMPTY: &'static Self = &Self {
        _lower_parts: Vec::new(),
    };
}

pub(super) type DecimalBytePartsV1 = ArrayRepresentation<DecimalBytePartsV1Data>;
pub(super) type DecimalBytePartsV1View<'a> =
    ArrayRepresentationView<'a, &'a DecimalBytePartsV1Data>;

#[derive(Clone, prost::Message)]
struct DecimalBytePartsMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    zeroth_child_ptype: i32,
    #[prost(uint32, tag = "2")]
    lower_part_count: u32,
}

pub(super) fn serialize(array: &DecimalBytePartsV1View<'_>) -> VortexResult<ArraySerialization> {
    let msp = array.slots()[0].as_ref().vortex_expect("v1 has an MSP");
    let metadata = DecimalBytePartsMetadata {
        zeroth_child_ptype: PType::try_from(msp.dtype())? as i32,
        lower_part_count: 0,
    }
    .encode_to_vec();
    Ok(ArraySerialization::new(
        array.id(),
        metadata,
        vec![],
        vec![msp.clone()],
    ))
}

pub(super) fn deserialize(parts: ArrayDeserialization<'_>) -> VortexResult<DecimalBytePartsV1> {
    vortex_ensure!(
        parts.serialized_id == decimal_byte_parts_v1_id(),
        "expected the v1 format"
    );
    let metadata = DecimalBytePartsMetadata::decode(parts.metadata)?;
    vortex_ensure!(
        parts.dtype.as_decimal_opt().is_some(),
        "expected a decimal dtype"
    );
    vortex_ensure!(
        metadata.lower_part_count == 0,
        "v1 must not carry lower parts"
    );
    vortex_ensure!(parts.children.len() == 1, "v1 must carry exactly one child");
    let ptype = PType::try_from(metadata.zeroth_child_ptype)?;
    vortex_ensure!(
        ptype.is_signed_int(),
        "MSP must have a signed integer dtype"
    );
    let encoded_dtype = DType::Primitive(ptype, parts.dtype.nullability());
    let msp = parts.children.get(0, &encoded_dtype, parts.len)?;
    Ok(ArrayRepresentation::new(
        decimal_byte_parts_v1_id(),
        parts.dtype.clone(),
        parts.len,
        DecimalBytePartsV1Data::default(),
        smallvec![Some(msp)],
    ))
}
