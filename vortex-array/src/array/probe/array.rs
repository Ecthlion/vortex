// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::array::ArrayView;
use crate::array::VTable;

/// Everything an encoding's `probe_scalar` runs with.
///
/// Passed to [`OperationsVTable::probe_scalar`](crate::vtable::OperationsVTable::probe_scalar).
/// Holds the typed view of the array being read; it owns nothing, so building one per read is
/// free.
pub struct ProbeState<'a, V: VTable> {
    array: ArrayView<'a, V>,
}

impl<'a, V: VTable> ProbeState<'a, V> {
    /// State for a single read of `array`.
    #[inline]
    pub fn once(array: ArrayView<'a, V>) -> Self {
        Self { array }
    }

    /// The typed view of the array being read.
    #[inline]
    pub fn array(&self) -> ArrayView<'a, V> {
        self.array
    }
}
