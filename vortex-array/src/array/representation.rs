// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Array storage for historical representations that have serde but no compute VTable.

use std::fmt::Debug;
use std::fmt::Formatter;

use vortex_error::VortexResult;

use crate::Array;
use crate::ArrayId;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::ArraySlots;
use crate::VTable;
use crate::array::ArrayInner;
use crate::dtype::DType;

/// A representation's data and child slots, without executable array behavior.
///
/// This wraps `ArrayInner<D, ()>`: the array's data and child slots, with no statistics allocation.
/// `D` need not have a [`VTable`]. A historical decoder can retain its data type while an explicit
/// conversion upgrades its data and attaches the current VTable.
///
/// Serializers borrow existing arrays through [`ArrayRepresentationView`] instead of constructing
/// owned storage. Decoders move their data and slots into this representation and then into the
/// executable array; only that final array allocates statistics.
///
/// Construction does not validate a format's invariants. Its decoder or conversion must do that.
/// [`Self::into_array`] always validates against the destination VTable before exposing an array.
pub struct ArrayRepresentation<D>(ArrayInner<D, ()>);

impl<D> ArrayRepresentation<D> {
    /// Store a representation's identity, logical metadata, data, and child slots.
    pub fn new(id: ArrayId, dtype: DType, len: usize, data: D, slots: ArraySlots) -> Self {
        Self(ArrayInner {
            len,
            encoding_id: id,
            dtype,
            slots,
            stats: (),
            data,
        })
    }

    /// The identity of this representation, which may name a historical serialized format.
    pub fn id(&self) -> ArrayId {
        self.0.encoding_id
    }

    /// The logical dtype of the represented values.
    pub fn dtype(&self) -> &DType {
        &self.0.dtype
    }

    /// The number of rows.
    pub fn len(&self) -> usize {
        self.0.len
    }

    /// Whether the representation has no rows.
    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    /// The version-specific data, which may own its own buffers.
    pub fn data(&self) -> &D {
        &self.0.data
    }

    /// The child slots for this representation.
    pub fn slots(&self) -> &[Option<ArrayRef>] {
        &self.0.slots
    }

    /// Borrow this representation's metadata, data, and slots without cloning them.
    pub fn as_view(&self) -> ArrayRepresentationView<'_, &D> {
        ArrayRepresentationView {
            id: self.id(),
            dtype: self.dtype(),
            len: self.len(),
            data: self.data(),
            slots: self.slots(),
        }
    }

    /// Convert the version-specific data while retaining logical metadata and child slots.
    ///
    /// This is useful when upgrading a format whose child layout has not changed. It does not
    /// validate the new representation or prove that the conversion preserves values.
    pub fn map_data<T>(self, f: impl FnOnce(D) -> T) -> ArrayRepresentation<T> {
        let inner = self.0;
        ArrayRepresentation(ArrayInner {
            len: inner.len,
            encoding_id: inner.encoding_id,
            dtype: inner.dtype,
            slots: inner.slots,
            stats: inner.stats,
            data: f(inner.data),
        })
    }

    /// Attach the current array VTable, replacing the historical representation's identity.
    ///
    /// # Errors
    ///
    /// Returns an error if the data, dtype, length, or slots fail the destination's validation.
    pub fn into_array<V: VTable<TypedArrayData = D>>(self, vtable: V) -> VortexResult<Array<V>> {
        let inner = self.0;
        Array::try_from_parts(
            ArrayParts::new(vtable, inner.dtype, inner.len, inner.data).with_slots(inner.slots),
        )
    }
}

impl<D: Debug> Debug for ArrayRepresentation<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayRepresentation")
            .field("id", &self.id())
            .field("dtype", self.dtype())
            .field("len", &self.len())
            .field("data", self.data())
            .field("slots", &self.slots())
            .finish()
    }
}

/// A version-specific view borrowing an array's logical metadata and child slots.
///
/// Constructing the view does not allocate or clone child references. `D` can borrow the current
/// array's data or contain a small version-specific adapter. It does not require a [`VTable`].
pub struct ArrayRepresentationView<'a, D> {
    id: ArrayId,
    dtype: &'a DType,
    len: usize,
    data: D,
    slots: &'a [Option<ArrayRef>],
}

impl<'a, D> ArrayRepresentationView<'a, D> {
    /// Borrow the metadata and slots stored in `array`'s `ArrayInner` for a serialized version.
    ///
    /// The caller supplies that version's identity and data adapter. This does not validate
    /// whether the original child layout is appropriate for the requested version.
    pub fn new(id: ArrayId, array: &'a ArrayRef, data: D) -> Self {
        Self {
            id,
            dtype: array.dtype(),
            len: array.len(),
            data,
            slots: array.slots(),
        }
    }

    /// The version represented by this view.
    pub fn id(&self) -> ArrayId {
        self.id
    }

    /// The borrowed logical dtype.
    pub fn dtype(&self) -> &'a DType {
        self.dtype
    }

    /// The number of rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view has no rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The version-specific data adapter.
    pub fn data(&self) -> &D {
        &self.data
    }

    /// The original child slots, borrowed directly from the array.
    pub fn slots(&self) -> &'a [Option<ArrayRef>] {
        self.slots
    }
}
