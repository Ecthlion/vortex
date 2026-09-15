// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Convert between DBP representations and dispatch to each version's serde implementation.

use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::DecimalByteParts;

mod convert;
#[cfg(test)]
mod tests;
mod v1;
mod v2;

pub use v2::DecimalBytePartsV2Metadata;

/// The frozen single-child decimal byte-parts format ID.
pub fn decimal_byte_parts_v1_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.decimal_byte_parts");
    *ID
}

/// The current in-memory DBP identity and the serialized format for arrays with lower parts.
pub fn decimal_byte_parts_v2_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.decimal_byte_parts_v2");
    *ID
}

/// Serde for the current DBP array through explicit version-specific representations.
///
/// Each version owns its data representation and serde functions. The plugin converts a current
/// array without lower parts into v1 before invoking the frozen serializer. Arrays with lower
/// parts use v2. Both decoders return storage without a compute VTable; the plugin upgrades and
/// validates that storage before returning a current [`DecimalByteParts`] array.
///
/// Register this plugin, or call [`crate::initialize`], to enable both formats. Direct registration
/// of [`DecimalByteParts`] does not support serde.
#[derive(Clone, Debug)]
pub struct DecimalBytePartsPlugin;

impl ArrayPlugin for DecimalBytePartsPlugin {
    fn id(&self) -> ArrayId {
        VTable::id(&DecimalByteParts)
    }

    fn serialized_ids(&self) -> Vec<ArrayId> {
        vec![decimal_byte_parts_v1_id(), decimal_byte_parts_v2_id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<DecimalByteParts>().ok_or_else(|| {
            vortex_err!(
                "DecimalByteParts plugin cannot serialize {}",
                array.encoding_id()
            )
        })?;
        let serialized = match convert::try_to_v1(view) {
            Some(legacy) => v1::serialize(&legacy)?,
            None => v2::serialize(&convert::to_v2(view))?,
        };
        Ok(Some(serialized))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let array = if parts.serialized_id == decimal_byte_parts_v1_id() {
            convert::from_v1(v1::deserialize(parts)?)?
        } else if parts.serialized_id == decimal_byte_parts_v2_id() {
            v2::deserialize(parts)?.into_array(DecimalByteParts)?
        } else {
            vortex_bail!(
                "DecimalByteParts plugin does not recognize serialized ID {}",
                parts.serialized_id
            )
        };
        Ok(array.into_array())
    }
}
