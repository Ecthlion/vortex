// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! State passed to an encoding's
//! [`probe_scalar`](crate::vtable::OperationsVTable::probe_scalar).

mod array;
pub use array::*;
