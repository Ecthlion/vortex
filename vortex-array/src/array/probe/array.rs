// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::probe::ProbeValidity;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

/// Row access shared by [`ArrayProbe`] and [`RepeatedArrayProbe`].
///
/// Both probe kinds expose the same reads; the trait lets code that recurses through child
/// slots treat them uniformly.
pub trait Probe {
    /// The array this probe reads from.
    fn array(&self) -> &ArrayRef;

    /// Read the scalar at `index`, including its nullness.
    fn execute_scalar(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar>;

    /// Whether the row at `index` is valid.
    fn execute_is_valid(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool>;

    /// Whether the row at `index` is null.
    fn execute_is_invalid(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        Ok(!self.execute_is_valid(index, ctx)?)
    }
}

/// A one-off row accessor over a borrowed array.
///
/// It is a bare reference: it retains nothing between reads and has no destructor, so building
/// one per read, including for every child slot of a nested array, costs nothing. Use
/// [`RepeatedArrayProbe`] when the same array is read many times.
///
/// This is split is a performance concern -- If we merge these there was a performance regression
/// for the one-off array probe.
#[derive(Clone, Copy)]
pub struct ArrayProbe<'a> {
    array: &'a ArrayRef,
}

impl<'a> ArrayProbe<'a> {
    /// Borrow an array for one-off reads.
    #[inline]
    pub fn new(array: &'a ArrayRef) -> Self {
        Self { array }
    }
}

impl Probe for ArrayProbe<'_> {
    #[inline]
    fn array(&self) -> &ArrayRef {
        self.array
    }

    // Always inlined: this is the per-element read path, and callers that read once per row
    // must get a single body with one dynamic call rather than an extra frame. Measured at
    // about 10% on the per-element benchmarks against plain `#[inline]`.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn execute_scalar(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        let array = self.array;
        if !self.execute_is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        check_dtype(
            array,
            array.dyn_array().probe_scalar_once(array, index, ctx),
        )
    }

    // Always inlined, see `execute_scalar`.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn execute_is_valid(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        let array = self.array;
        check_bounds(array, index)?;
        if !array.dtype().is_nullable() {
            return Ok(true);
        }
        // Matching the validity directly keeps this path free of any retained temporary.
        array.validity()?.execute_is_valid(index, ctx)
    }
}

/// A row accessor that owns its array and keeps preparation between reads.
///
/// The encoding's state, the validity probe and the probes over child slots are created on
/// first use and reused by every following read. Dropping the probe drops all of them. Array
/// handles share their buffers, so the probe can outlive the handle it was built from. Probes
/// are local to a thread.
pub struct RepeatedArrayProbe {
    array: ArrayRef,
    storage: ProbeStorage,
    /// Created on the first read of a nullable array. Boxed because the validity probe holds
    /// another `RepeatedArrayProbe`.
    validity: Option<Box<ProbeValidity>>,
}

impl RepeatedArrayProbe {
    /// Own an array for repeated reads.
    pub fn new(array: ArrayRef) -> Self {
        Self {
            array,
            storage: ProbeStorage::default(),
            validity: None,
        }
    }
}

impl Probe for RepeatedArrayProbe {
    fn array(&self) -> &ArrayRef {
        &self.array
    }

    fn execute_scalar(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        if !self.execute_is_valid(index, ctx)? {
            return Ok(Scalar::null(self.array.dtype().clone()));
        }
        let array = &self.array;
        let result = array
            .dyn_array()
            .probe_scalar_retained(array, index, &mut self.storage, ctx);
        check_dtype(array, result)
    }

    fn execute_is_valid(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        check_bounds(&self.array, index)?;
        if !self.array.dtype().is_nullable() {
            return Ok(true);
        }
        if self.validity.is_none() {
            self.validity = Some(Box::new(self.array.validity()?.probe()));
        }
        self.validity
            .as_mut()
            .ok_or_else(|| vortex_err!("validity probe was just initialized"))?
            .execute_is_valid(index, ctx)
    }
}

/// Pass an encoding's result through, checking its dtype in debug builds.
///
/// Unwrapping and re-wrapping the result here would cost an extra copy of the scalar on every
/// read.
#[inline]
fn check_dtype(array: &ArrayRef, result: VortexResult<Scalar>) -> VortexResult<Scalar> {
    result.inspect(|scalar| {
        debug_assert_eq!(scalar.dtype(), array.dtype(), "Scalar dtype mismatch");
    })
}

#[inline]
fn check_bounds(array: &ArrayRef, index: usize) -> VortexResult<()> {
    if index >= array.len() {
        return Err(out_of_bounds(index, array.len()));
    }
    Ok(())
}

/// Kept out of line so the error path, which captures a backtrace, does not count against the
/// hot probe bodies when the inliner sizes them.
#[cold]
#[inline(never)]
fn out_of_bounds(index: usize, len: usize) -> VortexError {
    vortex_err!(OutOfBounds: index, 0, len)
}

/// The encoding state type of `V`'s operations vtable.
pub type EncodingProbeState<V> =
    <<V as VTable>::OperationsVTable as OperationsVTable<V>>::ProbeState;

/// Everything an encoding's `probe_scalar` runs with: the typed view of the array being read
/// and, for a repeated read, a borrow of the state its [`RepeatedArrayProbe`] keeps.
///
/// Passed to [`OperationsVTable::probe_scalar`](crate::vtable::OperationsVTable::probe_scalar).
/// A one-off read gets [`ProbeState::once`], which holds only the view. A repeated read borrows
/// the [`RepeatedState`]: the encoding's own state and the lazily created child probes. Neither
/// owns anything, so building one per read is free.
///
/// Encodings never inspect the policy: [`ProbeState::slot`] hands out a probe over a child under
/// whichever policy is in force, [`ProbeState::child_scalar`] and [`ProbeState::child_is_valid`]
/// read one directly, and [`ProbeState::retained`] hands out the encoding state only when it is
/// kept.
pub struct ProbeState<'a, V: VTable> {
    array: ArrayView<'a, V>,
    retained: Option<&'a mut RepeatedState<EncodingProbeState<V>>>,
}

/// What a [`RepeatedArrayProbe`] keeps for its encoding between reads.
pub struct RepeatedState<S> {
    state: S,
    /// Probes over the source's child slots, allocated on first use. Slots that are never
    /// requested stay empty.
    slots: Vec<Option<RepeatedArrayProbe>>,
}

impl<'a, V: VTable> ProbeState<'a, V> {
    /// State for a single read of `array`. Encodings use this to run their `probe_scalar` path
    /// from `scalar_at`.
    #[inline]
    pub fn once(array: ArrayView<'a, V>) -> Self {
        Self {
            array,
            retained: None,
        }
    }

    /// State for a read through a [`RepeatedArrayProbe`], borrowing what it keeps.
    #[inline]
    pub(crate) fn repeated(
        array: ArrayView<'a, V>,
        retained: &'a mut RepeatedState<EncodingProbeState<V>>,
    ) -> Self {
        Self {
            array,
            retained: Some(retained),
        }
    }

    /// The typed view of the array being read.
    #[inline]
    pub fn array(&self) -> ArrayView<'a, V> {
        self.array
    }

    /// The encoding's retained state, or `None` for a one-off read.
    ///
    /// Encodings whose repeated algorithm differs from their one-off one branch on this.
    #[inline]
    pub fn retained(&mut self) -> Option<&mut EncodingProbeState<V>> {
        self.retained
            .as_deref_mut()
            .map(|repeated| &mut repeated.state)
    }

    /// A probe over the array's child in `slot`, under this state's policy.
    ///
    /// For a one-off read this is a borrowed [`ArrayProbe`] built for the call; for a repeated
    /// read it is the [`RepeatedArrayProbe`] kept in the slot table, created on first use.
    /// Errors for an out-of-bounds or absent slot.
    #[inline]
    pub fn slot(&mut self, slot: usize) -> VortexResult<impl Probe + '_> {
        let parent = self.array.array();
        Ok(match &mut self.retained {
            None => ChildProbe::Once(ArrayProbe::new(child_of(parent, slot)?)),
            Some(repeated) => ChildProbe::Repeated(repeated.child_probe(parent, slot)?),
        })
    }

    /// Read the scalar at `index` of the array's child in `slot`, including its nullness.
    ///
    /// Equivalent to [`Self::slot`] followed by [`Probe::execute_scalar`], without building
    /// the intermediate probe: use this for the per-row read of a child, and `slot` when a
    /// probe needs to be held or passed on.
    #[inline]
    pub fn child_scalar(
        &mut self,
        slot: usize,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let parent = self.array.array();
        match &mut self.retained {
            None => ArrayProbe::new(child_of(parent, slot)?).execute_scalar(index, ctx),
            Some(repeated) => repeated
                .child_probe(parent, slot)?
                .execute_scalar(index, ctx),
        }
    }

    /// Whether the row at `index` of the array's child in `slot` is valid.
    ///
    /// Same relationship to [`Self::slot`] as [`Self::child_scalar`].
    #[inline]
    pub fn child_is_valid(
        &mut self,
        slot: usize,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        let parent = self.array.array();
        match &mut self.retained {
            None => ArrayProbe::new(child_of(parent, slot)?).execute_is_valid(index, ctx),
            Some(repeated) => repeated
                .child_probe(parent, slot)?
                .execute_is_valid(index, ctx),
        }
    }
}

/// The probe [`ProbeState::slot`] hands out: a bare one-off probe or the retained one.
enum ChildProbe<'a> {
    Once(ArrayProbe<'a>),
    Repeated(&'a mut RepeatedArrayProbe),
}

impl Probe for ChildProbe<'_> {
    #[inline]
    fn array(&self) -> &ArrayRef {
        match self {
            Self::Once(probe) => probe.array(),
            Self::Repeated(probe) => probe.array(),
        }
    }

    // Always inlined for the same reason as `ArrayProbe::execute_scalar`: the one-off arm is
    // the per-element path through nested arrays.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn execute_scalar(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        match self {
            Self::Once(probe) => probe.execute_scalar(index, ctx),
            Self::Repeated(probe) => probe.execute_scalar(index, ctx),
        }
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn execute_is_valid(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        match self {
            Self::Once(probe) => probe.execute_is_valid(index, ctx),
            Self::Repeated(probe) => probe.execute_is_valid(index, ctx),
        }
    }
}

impl<S: Default> Default for RepeatedState<S> {
    fn default() -> Self {
        Self {
            state: S::default(),
            slots: Vec::new(),
        }
    }
}

impl<S> RepeatedState<S> {
    /// Get or create the retained probe over `parent`'s child in `slot`.
    fn child_probe(
        &mut self,
        parent: &ArrayRef,
        slot: usize,
    ) -> VortexResult<&mut RepeatedArrayProbe> {
        let child = child_of(parent, slot)?;
        if self.slots.is_empty() {
            self.slots.resize_with(parent.slots().len(), || None);
        }
        Ok(self.slots[slot].get_or_insert_with(|| RepeatedArrayProbe::new(child.clone())))
    }
}

/// The child of `parent` in `slot`, as an error if the slot is out of bounds or absent.
#[inline]
fn child_of(parent: &ArrayRef, slot: usize) -> VortexResult<&ArrayRef> {
    parent
        .slots()
        .get(slot)
        .ok_or_else(|| vortex_err!("Probe slot {slot} is out of bounds"))?
        .as_ref()
        .ok_or_else(|| vortex_err!("Probe slot {slot} is absent"))
}

/// Type-erased, lazily initialized [`RepeatedState`] owned by a [`RepeatedArrayProbe`].
#[derive(Default)]
pub(crate) struct ProbeStorage(Option<Box<dyn Any>>);

impl ProbeStorage {
    pub(crate) fn get_or_init<S: Default + 'static>(
        &mut self,
    ) -> VortexResult<&mut RepeatedState<S>> {
        self.0
            .get_or_insert_with(|| Box::new(RepeatedState::<S>::default()))
            .downcast_mut::<RepeatedState<S>>()
            .ok_or_else(|| vortex_err!("Probe state type mismatch"))
    }
}

#[cfg(test)]
mod tests {
    use std::mem::needs_drop;
    use std::mem::size_of;

    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::array::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::Struct;
    use crate::arrays::StructArray;

    fn nullable_ints() -> ArrayRef {
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array()
    }

    fn check_reads(probe: &mut dyn Probe, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        assert!(probe.execute_scalar(3, ctx).is_err());
        assert!(probe.execute_is_valid(3, ctx).is_err());
        assert!(probe.execute_scalar(1, ctx)?.is_null());
        assert!(!probe.execute_is_valid(1, ctx)?);
        assert_eq!(probe.execute_scalar(2, ctx)?, Scalar::from(Some(30i32)));
        assert_eq!(probe.execute_scalar(0, ctx)?, Scalar::from(Some(10i32)));
        Ok(())
    }

    #[test]
    fn once_probe_checks_bounds_and_nulls() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = nullable_ints();
        check_reads(&mut array.probe(), &mut ctx)
    }

    #[test]
    fn once_probe_is_a_bare_reference() {
        assert_eq!(size_of::<ArrayProbe<'_>>(), size_of::<&ArrayRef>());
        assert!(!needs_drop::<ArrayProbe<'_>>());
    }

    #[test]
    fn repeated_probe_initializes_lazily_and_outlives_handle() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = nullable_ints();
        let mut probe = RepeatedArrayProbe::new(array.clone());
        drop(array);
        assert!(probe.storage.0.is_none());
        assert!(probe.validity.is_none());

        check_reads(&mut probe, &mut ctx)?;

        assert!(probe.storage.0.is_some());
        assert!(probe.validity.is_some());
        Ok(())
    }

    #[test]
    fn once_state_reads_children_without_retaining() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = struct_of_two_fields()?;
        let typed = array
            .as_opt::<Struct>()
            .ok_or_else(|| vortex_err!("expected a struct"))?;
        let mut state = ProbeState::once(typed);

        assert!(state.slot(5).is_err());
        // Slot 0 is the struct's absent validity.
        assert!(state.slot(0).is_err());
        {
            let mut field = state.slot(2)?;
            assert_eq!(field.array().len(), 2);
            assert_eq!(field.execute_scalar(1, &mut ctx)?, Scalar::from(4i64));
            assert!(field.execute_is_valid(0, &mut ctx)?);
        }
        assert_eq!(state.child_scalar(2, 0, &mut ctx)?, Scalar::from(3i64));
        assert!(state.child_is_valid(1, 0, &mut ctx)?);
        assert!(state.retained().is_none());
        assert!(!needs_drop::<ProbeState<'_, Struct>>());
        Ok(())
    }

    #[test]
    fn repeated_state_creates_children_on_demand() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = struct_of_two_fields()?;
        let typed = array
            .as_opt::<Struct>()
            .ok_or_else(|| vortex_err!("expected a struct"))?;
        let mut repeated = RepeatedState::<()>::default();
        {
            let mut state = ProbeState::repeated(typed, &mut repeated);
            assert!(state.slot(5).is_err());
            assert!(state.slot(0).is_err());
            assert!(state.retained().is_some());
            let mut field = state.slot(2)?;
            assert_eq!(field.execute_scalar(1, &mut ctx)?, Scalar::from(4i64));
            assert_eq!(field.execute_scalar(0, &mut ctx)?, Scalar::from(3i64));
        }

        assert_eq!(repeated.slots.len(), array.slots().len());
        assert!(repeated.slots[1].is_none());
        let child = repeated.slots[2]
            .as_ref()
            .ok_or_else(|| vortex_err!("missing child probe"))?;
        assert_eq!(child.array().len(), 2);
        Ok(())
    }

    fn struct_of_two_fields() -> VortexResult<ArrayRef> {
        Ok(StructArray::from_fields(&[
            ("a", PrimitiveArray::from_iter([1i32, 2]).into_array()),
            ("b", PrimitiveArray::from_iter([3i64, 4]).into_array()),
        ])?
        .into_array())
    }

    #[test]
    fn storage_rejects_mismatched_state_type() -> VortexResult<()> {
        let mut storage = ProbeStorage::default();
        storage.get_or_init::<()>()?;
        assert!(storage.get_or_init::<u8>().is_err());
        Ok(())
    }
}
