// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Random scalar access with optional, encoding-specific retained state.

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::array::VTable;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

/// Whether scalar access should retain preparation for subsequent lookups.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProbeUsage {
    /// Use temporary resources only, without initializing retained probe state.
    Once,
    /// Allow the encoding to retain preparation and decoded data between lookups.
    Repeated,
}

/// A borrowed scalar accessor that retains encoding-specific state until it is dropped.
///
/// `'a` is the borrow of the root array. Its [`ProbeState`] retains a child probe for each
/// requested slot, so one lifetime covers the whole tree and no array handles are cloned.
/// Anything an encoding builds itself, such as a decoded page or validity mask, is owned
/// by its state.
///
/// Construction never allocates or executes the array. Repeated access initializes the
/// encoding's context in a Box on the first in-bounds lookup. Child storage and decoding
/// may allocate separately, when first needed.
///
/// `Once` is a caching hint, not a restriction on how many times the probe may be called.
/// Returned scalars own their values and can outlive the probe. Probes are local to a thread.
pub struct ArrayProbe<'a> {
    array: &'a ArrayRef,
    state: Option<ProbeStorage<'a>>,
}

/// Scalar access to a source array, with optional retained preparation.
pub enum ProbeAccess<'a, 'p, S> {
    /// Read slots with fresh probes that retain no state.
    Once(&'a ArrayRef),
    /// Reuse the source's local state and child probes.
    Repeated(&'p mut ProbeState<'a, S>),
}

impl<'a, S> ProbeAccess<'a, '_, S> {
    /// Access retained local state, if repeated access was requested.
    pub fn state_mut(&mut self) -> Option<&mut S> {
        match self {
            Self::Once(_) => None,
            Self::Repeated(state) => Some(state.state_mut()),
        }
    }

    /// Access a child slot using this context's retention policy.
    pub fn slot(&mut self, slot: usize) -> VortexResult<ProbeSlot<'a, '_>> {
        match self {
            Self::Once(array) => {
                let child = array
                    .slots()
                    .get(slot)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| vortex_err!("Probe child slot {slot} is missing"))?;
                Ok(ProbeSlot::Once(child))
            }
            Self::Repeated(state) => Ok(ProbeSlot::Repeated(state.child(slot)?)),
        }
    }
}

/// A child accessor that either starts a fresh probe or borrows a retained probe.
pub enum ProbeSlot<'a, 'p> {
    /// A source read without retained preparation.
    Once(&'a ArrayRef),
    /// An existing repeated-access probe.
    Repeated(&'p mut ArrayProbe<'a>),
}

impl<'a> ProbeSlot<'a, '_> {
    /// The source array for this slot.
    pub fn array(&self) -> &'a ArrayRef {
        match self {
            Self::Once(array) => array,
            Self::Repeated(probe) => probe.array(),
        }
    }

    /// Read a scalar using fresh or retained preparation as appropriate.
    pub fn execute_scalar(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        match self {
            Self::Once(array) => array.probe(ProbeUsage::Once).scalar_at(index, ctx),
            Self::Repeated(probe) => probe.scalar_at(index, ctx),
        }
    }
}

/// Local encoding state and lazy child probes retained for one source array.
///
/// The framework initializes this context once for repeated access and passes it to
/// [`OperationsVTable::probe_scalar`](crate::vtable::OperationsVTable::probe_scalar).
/// One-off access uses [`ProbeAccess::Once`] instead. `S` is the encoding's associated state type;
/// it may borrow the source tree for `'a` and own any prepared resources.
pub struct ProbeState<'a, S> {
    state: S,
    children: ProbeChildren<'a>,
}

impl<'a, S: Default> ProbeState<'a, S> {
    pub(crate) fn new(array: &'a ArrayRef) -> Self {
        Self {
            state: S::default(),
            children: ProbeChildren {
                array,
                slots: Vec::new(),
            },
        }
    }
}

impl<'a, S> ProbeState<'a, S> {
    /// Access the encoding's retained local state.
    pub fn state_mut(&mut self) -> &mut S {
        &mut self.state
    }

    /// Get the retained probe for a source slot, creating it on the first request.
    ///
    /// Returns an error for an absent or out-of-bounds slot. Creating a child probe does
    /// not execute the child or initialize its encoding state.
    pub fn child(&mut self, slot: usize) -> VortexResult<&mut ArrayProbe<'a>> {
        self.children.child(slot)
    }

    /// Borrow local state and child access together, allowing disjoint mutable access.
    pub fn parts(&mut self) -> (&mut S, &mut ProbeChildren<'a>) {
        (&mut self.state, &mut self.children)
    }
}

/// Lazy child probes bound to the slots of one source array.
///
/// Obtain this through [`ProbeState::parts`] when retaining a mutable borrow of local state
/// while accessing children. Each slot has independent state, even if two slots reference
/// the same array. The slot table allocates on its first valid request; unrequested slots
/// remain empty. Dropping the parent context drops every created child probe.
pub struct ProbeChildren<'a> {
    array: &'a ArrayRef,
    slots: Vec<Option<ArrayProbe<'a>>>,
}

impl<'a> ProbeChildren<'a> {
    /// Get or create a repeated-access probe for the given source slot.
    ///
    /// Returns an error for an absent or out-of-bounds slot without allocating a slot table.
    pub fn child(&mut self, slot: usize) -> VortexResult<&mut ArrayProbe<'a>> {
        let child = self
            .array
            .slots()
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| vortex_err!("Probe child slot {slot} is missing"))?;
        if self.slots.is_empty() {
            self.slots.resize_with(self.array.slots().len(), || None);
        }
        Ok(self.slots[slot].get_or_insert_with(|| child.probe(ProbeUsage::Repeated)))
    }
}

impl ArrayRef {
    /// Create an accessor with the requested policy for retaining state between scalar lookups.
    ///
    /// ```
    /// use vortex_array::{IntoArray, ProbeUsage, VortexSessionExecute};
    /// use vortex_array::arrays::PrimitiveArray;
    ///
    /// let array = PrimitiveArray::from_iter([10i32, 20, 30]).into_array();
    /// let mut ctx = vortex_array::array_session().create_execution_ctx();
    /// let mut probe = array.probe(ProbeUsage::Repeated);
    /// assert_eq!(probe.scalar_at(2, &mut ctx)?, 30i32.into());
    /// assert_eq!(probe.scalar_at(0, &mut ctx)?, 10i32.into());
    /// # Ok::<(), vortex_error::VortexError>(())
    /// ```
    pub fn probe(&self, usage: ProbeUsage) -> ArrayProbe<'_> {
        ArrayProbe {
            array: self,
            state: match usage {
                ProbeUsage::Once => None,
                ProbeUsage::Repeated => Some(ProbeStorage::new()),
            },
        }
    }
}

impl<'a> ArrayProbe<'a> {
    /// The array this probe reads from.
    pub fn array(&self) -> &'a ArrayRef {
        self.array
    }

    /// Read a scalar, including its nullness, preparing and reusing state as appropriate.
    pub fn scalar_at(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        vortex_ensure!(index < self.array.len(), OutOfBounds: index, 0, self.array.len());
        let scalar =
            self.array
                .dyn_array()
                .probe_scalar(self.array, index, self.state.as_mut(), ctx)?;
        debug_assert_eq!(scalar.dtype(), self.array.dtype(), "Scalar dtype mismatch");
        Ok(scalar)
    }
}

pub(crate) struct ProbeStorage<'a> {
    // FIXME: Consider inline storage if benchmarks justify avoiding this allocation.
    probe: Option<Box<dyn RetainedProbe + 'a>>,
}

trait RetainedProbe {
    fn scalar_at(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar>;
}

struct EncodingProbe<'a, V: VTable> {
    array: ArrayView<'a, V>,
    state: ProbeState<'a, <V::OperationsVTable as OperationsVTable<V>>::ProbeState<'a>>,
}

impl<V: VTable> RetainedProbe for EncodingProbe<'_, V> {
    fn scalar_at(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        <V::OperationsVTable as OperationsVTable<V>>::probe_scalar(
            self.array,
            index,
            ProbeAccess::Repeated(&mut self.state),
            ctx,
        )
    }
}

impl<'a> ProbeStorage<'a> {
    fn new() -> Self {
        Self { probe: None }
    }

    fn get_or_init(
        &mut self,
        init: impl FnOnce() -> Box<dyn RetainedProbe + 'a>,
    ) -> &mut (dyn RetainedProbe + 'a) {
        self.probe.get_or_insert_with(init).as_mut()
    }

    pub(crate) fn scalar_at<V: VTable>(
        &mut self,
        array: ArrayView<'a, V>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        self.get_or_init(|| {
            Box::new(EncodingProbe {
                array,
                state: ProbeState::new(array.array()),
            })
        })
        .scalar_at(index, ctx)
    }
}

#[cfg(test)]
mod tests;
