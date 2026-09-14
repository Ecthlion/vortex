// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ExecutionCtx;
use crate::ProbeAccess;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::scalar::Scalar;
use crate::vtable::NotSupported;

/// Element-level operations for an array encoding.
///
/// This trait is separated from [`VTable`] so encodings can organize scalar
/// access independently from traversal, serialization, and execution. The erased
/// [`ArrayRef`](crate::ArrayRef)
/// methods perform common checks before dispatching here.
pub trait OperationsVTable<V: VTable> {
    /// Encoding-specific state retained by repeated scalar access.
    ///
    /// Default construction should be cheap and avoid allocation or execution. Preparation
    /// belongs in [`Self::probe_scalar`]. State owns its preparation and may retain shared
    /// buffer or array handles. Request retained child probes through [`ProbeAccess`].
    /// Use `()` when no local state is needed.
    type ProbeState: Default + 'static;

    /// Read a non-null scalar, optionally retaining state for subsequent reads.
    ///
    /// Bounds and validity have been checked; the row is non-null. `ProbeAccess::Once` requests one-off access and
    /// never initializes a context; `ProbeAccess::Repeated` reuses local state and child probes for this source.
    /// The scalar must retain the source's logical dtype, including nullability.
    ///
    /// The default preserves the existing scalar path without adding caching.
    fn probe_scalar(
        array: ArrayView<'_, V>,
        index: usize,
        _probe: ProbeAccess<'_, Self::ProbeState>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // FIXME: Remove this default once all encodings have migrated to probe_scalar.
        Self::scalar_at(array, index, ctx)
    }

    /// Fetch the scalar at the given index.
    ///
    /// ## Preconditions
    ///
    /// Bounds-checking has already been performed by the time this function is called,
    /// and the index is guaranteed to be non-null. Implementations may assume `index < len`.
    ///
    /// ## Postconditions
    ///
    /// The returned [`Scalar`] must have the same logical dtype as the array's element dtype.
    // FIXME: Remove this hook once all encodings have migrated to probe_scalar.
    #[deprecated(
        note = "Implement `OperationsVTable::probe_scalar` instead, which is handed a \
        `ProbeAccess` so the encoding can retain preparation across lookups."
    )]
    fn scalar_at(
        array: ArrayView<'_, V>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar>;
}

impl<V: VTable> OperationsVTable<V> for NotSupported {
    type ProbeState = ();

    fn scalar_at(
        array: ArrayView<'_, V>,
        _index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        vortex_bail!(
            "Legacy scalar_at operation is not supported for {} arrays",
            array.encoding_id()
        )
    }
}
