// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row access over arrays.
//!
//! [`ArrayProbe`] reads a borrowed array once and retains nothing. [`RepeatedArrayProbe`] owns
//! its array and keeps encoding state, its validity probe and child probes between reads. Both
//! implement [`Probe`]. Encodings implement a single
//! [`probe_scalar`](crate::vtable::OperationsVTable::probe_scalar) that serves both through
//! [`ProbeState`].

mod array;
pub use array::*;
mod validity;
pub use validity::*;
