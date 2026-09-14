// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Random scalar access with optional, encoding-specific retained state.

use std::any::Any;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::array::VTable;
use crate::scalar::Scalar;
use crate::validity::Validity;
use crate::vtable::OperationsVTable;

/// Whether scalar access should retain preparation for subsequent lookups.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProbeUsage {
    /// Use temporary resources only, without initializing retained probe state.
    Once,
    /// Allow the encoding to retain preparation and decoded data between lookups.
    Repeated,
}

/// A scalar accessor that owns its source and retains preparation between lookups.
///
/// Array handles share their buffers. Repeated access initializes encoding state and
/// child probes lazily; following the same slots reuses the same preparation.
/// `Once` retains no preparation. The probe can outlive the original array handle.
/// Probes are local to a thread.
pub struct ArrayProbe {
    array: ArrayRef,
    state: Option<ProbeStorage>,
}

/// Scalar access to a source array, with optional retained preparation.
pub enum ProbeAccess<'p, S> {
    /// Read slots with fresh probes that retain no state.
    Once(&'p ArrayRef),
    /// Reuse the source's local state and child probes.
    Repeated(ProbeCtx<'p, S>),
}

/// Access to retained preparation, independent of source ownership.
pub struct ProbeCtx<'p, S> {
    state: &'p mut S,
    children: &'p mut ProbeChildren,
    validity: &'p mut Option<ProbeValidity>,
}

impl<S> ProbeAccess<'_, S> {
    /// Access retained local state, if repeated access was requested.
    pub fn state_mut(&mut self) -> Option<&mut S> {
        match self {
            Self::Once(_) => None,
            Self::Repeated(probe) => Some(probe.state),
        }
    }

    /// Access validity using this context's retention policy.
    ///
    /// Array-backed validity is read lazily through an owned probe; repeated access retains it.
    pub fn validity(&mut self) -> VortexResult<ValidityProbe<'_>> {
        match self {
            Self::Once(array) => ValidityProbe::once(array),
            Self::Repeated(probe) => validity_access(&probe.children.array, probe.validity),
        }
    }

    /// Access a child slot using this context's retention policy.
    ///
    /// Returns `None` for an absent slot and an error for an out-of-bounds slot.
    pub fn slot(&mut self, slot: usize) -> VortexResult<Option<ProbeSlot<'_>>> {
        match self {
            Self::Once(array) => {
                let child = array
                    .slots()
                    .get(slot)
                    .ok_or_else(|| vortex_err!("Probe slot {slot} is out of bounds"))?;
                Ok(child.as_ref().map(ProbeSlot::Once))
            }
            Self::Repeated(probe) => Ok(probe.children.slot(slot)?.map(ProbeSlot::Repeated)),
        }
    }
}

/// A child accessor that either starts a fresh probe or borrows a retained probe.
pub enum ProbeSlot<'p> {
    /// A source read without retained preparation.
    Once(&'p ArrayRef),
    /// An existing repeated-access probe.
    Repeated(&'p mut ArrayProbe),
}

impl ProbeSlot<'_> {
    /// The source array for this slot.
    pub fn array(&self) -> &ArrayRef {
        match self {
            Self::Once(array) => array,
            Self::Repeated(probe) => probe.array(),
        }
    }

    /// Read a scalar using fresh or retained preparation as appropriate.
    pub fn execute_scalar(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        match self {
            Self::Once(array) => array.probe(ProbeUsage::Once).execute_scalar(index, ctx),
            Self::Repeated(probe) => probe.execute_scalar(index, ctx),
        }
    }
}

/// Local encoding state and lazy child probes retained for one source array.
///
/// The framework initializes this context once for repeated access and passes it to
/// [`OperationsVTable::probe_scalar`].
/// One-off access uses [`ProbeAccess::Once`] instead. `S` is the encoding's associated state type;
/// it owns its prepared resources, including shared buffer or array handles.
pub struct ProbeState<S> {
    state: S,
    children: ProbeChildren,
    validity: Option<ProbeValidity>,
}

impl<S: Default> ProbeState<S> {
    pub(crate) fn new(array: &ArrayRef) -> Self {
        Self {
            state: S::default(),
            children: ProbeChildren {
                array: array.clone(),
                slots: Vec::new(),
            },
            validity: None,
        }
    }
}

impl<S> ProbeState<S> {
    /// Borrow retained preparation and children for scalar execution.
    pub fn access(&mut self) -> ProbeAccess<'_, S> {
        ProbeAccess::Repeated(ProbeCtx {
            state: &mut self.state,
            children: &mut self.children,
            validity: &mut self.validity,
        })
    }

    /// Access the encoding's retained local state.
    pub fn state_mut(&mut self) -> &mut S {
        &mut self.state
    }

    /// Borrow local state and child access together, allowing disjoint mutable access.
    pub fn parts(&mut self) -> (&mut S, &mut ProbeChildren) {
        (&mut self.state, &mut self.children)
    }
}

/// Lazy child probes bound to the slots of one source array.
///
/// Obtain this through [`ProbeState::parts`] when retaining a mutable borrow of local state
/// while accessing children. Each slot has independent state, even if two slots reference
/// the same array. The slot table allocates on its first valid request; unrequested slots
/// remain empty. Dropping the parent context drops every created child probe.
pub struct ProbeChildren {
    array: ArrayRef,
    slots: Vec<Option<ArrayProbe>>,
}

impl ProbeChildren {
    /// Get or create a repeated-access probe for the given source slot.
    ///
    /// Returns `None` for an absent slot and an error for an out-of-bounds slot.
    /// Neither case allocates a slot table.
    pub fn slot(&mut self, slot: usize) -> VortexResult<Option<&mut ArrayProbe>> {
        let child = self
            .array
            .slots()
            .get(slot)
            .ok_or_else(|| vortex_err!("Probe slot {slot} is out of bounds"))?;
        let Some(child) = child else {
            return Ok(None);
        };
        if self.slots.is_empty() {
            self.slots.resize_with(self.array.slots().len(), || None);
        }
        Ok(Some(
            self.slots[slot].get_or_insert_with(|| child.probe(ProbeUsage::Repeated)),
        ))
    }
}

impl ArrayProbe {
    /// Own an array and choose whether to retain preparation.
    pub fn new(array: ArrayRef, usage: ProbeUsage) -> Self {
        Self {
            array,
            state: match usage {
                ProbeUsage::Once => None,
                ProbeUsage::Repeated => Some(ProbeStorage::new()),
            },
        }
    }

    /// The array this probe reads from.
    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    /// Read a scalar, including its nullness, preparing and reusing state as appropriate.
    pub fn execute_scalar(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        vortex_ensure!(index < self.array.len(), OutOfBounds: index, 0, self.array.len());
        if self.array.dtype().is_nullable() && !self.execute_is_valid(index, ctx)? {
            return Ok(Scalar::null(self.array.dtype().clone()));
        }
        let scalar =
            self.array
                .dyn_array()
                .probe_scalar(&self.array, index, self.state.as_mut(), ctx)?;
        debug_assert_eq!(scalar.dtype(), self.array.dtype(), "Scalar dtype mismatch");
        Ok(scalar)
    }

    /// Check bounds and read validity using this probe's retention policy.
    pub fn execute_is_valid(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        vortex_ensure!(index < self.array.len(), OutOfBounds: index, 0, self.array.len());
        if !self.array.dtype().is_nullable() {
            return Ok(true);
        }
        self.array
            .dyn_array()
            .probe_is_valid(&self.array, index, self.state.as_mut(), ctx)
    }
}

/// A validity accessor with the parent probe's retention policy.
pub struct ValidityProbe<'p> {
    len: usize,
    inner: ValidityAccess<'p>,
}

enum ValidityAccess<'p> {
    Once(Validity),
    Repeated(&'p mut ProbeValidity),
}

impl ValidityProbe<'_> {
    pub(super) fn once(array: &ArrayRef) -> VortexResult<Self> {
        Ok(Self {
            len: array.len(),
            inner: ValidityAccess::Once(array.validity()?),
        })
    }

    /// Read a non-null boolean scalar indicating whether the requested row is valid.
    pub fn execute_scalar(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        vortex_ensure!(index < self.len, OutOfBounds: index, 0, self.len);
        match &mut self.inner {
            ValidityAccess::Once(validity) => match validity {
                Validity::NonNullable | Validity::AllValid => Ok(true.into()),
                Validity::AllInvalid => Ok(false.into()),
                Validity::Array(array) => array.execute_scalar(index, ctx),
            },
            ValidityAccess::Repeated(validity) => validity.execute_scalar(index, ctx),
        }
    }
}

fn validity_access<'p>(
    array: &ArrayRef,
    slot: &'p mut Option<ProbeValidity>,
) -> VortexResult<ValidityProbe<'p>> {
    let validity = match slot {
        Some(validity) => validity,
        slot @ None => slot.insert(ProbeValidity::new(array.validity()?, ProbeUsage::Repeated)),
    };
    Ok(ValidityProbe {
        len: array.len(),
        inner: ValidityAccess::Repeated(validity),
    })
}

enum ProbeValidity {
    Constant(bool),
    Array(ArrayProbe),
}

impl ProbeValidity {
    fn new(validity: Validity, usage: ProbeUsage) -> Self {
        match validity {
            Validity::NonNullable | Validity::AllValid => Self::Constant(true),
            Validity::AllInvalid => Self::Constant(false),
            Validity::Array(array) => Self::Array(ArrayProbe::new(array, usage)),
        }
    }

    fn execute_scalar(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        match self {
            Self::Constant(valid) => Ok((*valid).into()),
            Self::Array(probe) => probe.execute_scalar(index, ctx),
        }
    }
}

pub(crate) struct ProbeStorage {
    // FIXME: Consider inline storage if benchmarks justify avoiding this allocation.
    probe: Option<Box<dyn Any>>,
}

impl ProbeStorage {
    fn new() -> Self {
        Self { probe: None }
    }

    fn get_or_init<S: Default + 'static>(
        &mut self,
        array: &ArrayRef,
    ) -> VortexResult<&mut ProbeState<S>> {
        self.probe
            .get_or_insert_with(|| Box::new(ProbeState::<S>::new(array)))
            .downcast_mut::<ProbeState<S>>()
            .ok_or_else(|| vortex_err!("Probe state type mismatch"))
    }

    pub(crate) fn validity<S: Default + 'static>(
        &mut self,
        array: &ArrayRef,
    ) -> VortexResult<ValidityProbe<'_>> {
        let state = self.get_or_init::<S>(array)?;
        validity_access(array, &mut state.validity)
    }

    pub(crate) fn scalar_at<V: VTable>(
        &mut self,
        array: ArrayView<'_, V>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let state = self.get_or_init::<<V::OperationsVTable as OperationsVTable<V>>::ProbeState>(
            array.array(),
        )?;
        <V::OperationsVTable as OperationsVTable<V>>::probe_scalar(
            array,
            index,
            state.access(),
            ctx,
        )
    }
}

#[cfg(test)]
mod tests;
