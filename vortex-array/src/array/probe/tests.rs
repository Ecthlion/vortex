// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::Cell;
use std::rc::Rc;

use rstest::rstest;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::ProbeAccess;
use super::ProbeState;
use super::ProbeStorage;
use super::ProbeUsage;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::scalar::Scalar;

#[derive(Default)]
struct TrackedState {
    reads: usize,
    drops: Rc<Cell<usize>>,
}

impl Drop for TrackedState {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

#[test]
fn state_is_initialized_once_and_survives_moves() -> VortexResult<()> {
    let array = PrimitiveArray::from_iter([42i32]).into_array();
    let mut storage = ProbeStorage::new();
    assert!(storage.probe.is_none());
    let state = storage.get_or_init::<TrackedState>(&array)?;
    state.state.reads += 1;
    let drops = Rc::clone(&state.state.drops);
    let mut moved = (storage, ());
    let state = moved.0.get_or_init::<TrackedState>(&array)?;
    assert_eq!(state.state.reads, 1);
    assert!(Rc::ptr_eq(&state.state.drops, &drops));
    assert_eq!(drops.get(), 0);
    drop(moved);
    assert_eq!(drops.get(), 1);
    Ok(())
}

#[test]
fn children_are_lazy_reused_and_bound_to_their_source_slots() -> VortexResult<()> {
    let leaf = PrimitiveArray::from_iter([10i32, 20]).into_array();
    let array = StructArray::from_fields(&[("left", leaf.clone()), ("right", leaf)])?.into_array();
    let mut ctx = crate::array_session().create_execution_ctx();
    let mut probe = ProbeState::<usize>::new(&array);
    assert_eq!(probe.children.slots.capacity(), 0);
    // Slot zero is the absent struct validity; invalid requests must not allocate.
    assert!(probe.children.slot(0)?.is_none());
    assert!(probe.children.slot(99).is_err());
    assert_eq!(probe.children.slots.capacity(), 0);

    let first = std::ptr::from_mut(
        probe
            .children
            .slot(1)?
            .ok_or_else(|| vortex_err!("missing fixture slot"))?,
    );
    assert!(probe.children.slots[2].is_none());
    let (state, children) = probe.parts();
    let child = children
        .slot(1)?
        .ok_or_else(|| vortex_err!("missing fixture slot"))?;
    assert!(child.execute_scalar(2, &mut ctx).is_err());
    assert_eq!(child.execute_scalar(0, &mut ctx)?, 10i32.into());
    *state += 1;
    assert_eq!(*probe.state_mut(), 1);
    assert_eq!(
        std::ptr::from_mut(
            probe
                .children
                .slot(1)?
                .ok_or_else(|| vortex_err!("missing fixture slot"))?
        ),
        first
    );
    assert!(
        probe
            .children
            .slot(1)?
            .ok_or_else(|| vortex_err!("missing fixture slot"))?
            .state
            .as_ref()
            .is_some_and(|state| state.probe.is_some())
    );

    // Identical sources in different slots still get independent probe state.
    assert_ne!(
        std::ptr::from_mut(
            probe
                .children
                .slot(2)?
                .ok_or_else(|| vortex_err!("missing fixture slot"))?
        ),
        first
    );
    let source = array.slots()[1]
        .as_ref()
        .ok_or_else(|| vortex_error::vortex_err!("missing fixture slot"))?;
    assert!(ArrayRef::ptr_eq(
        probe
            .children
            .slot(1)?
            .ok_or_else(|| vortex_err!("missing fixture slot"))?
            .array(),
        source
    ));
    assert!(
        probe
            .children
            .slot(2)?
            .ok_or_else(|| vortex_err!("missing fixture slot"))?
            .state
            .as_ref()
            .is_some_and(|state| state.probe.is_none())
    );
    let mut other = ProbeState::<usize>::new(&array);
    assert!(
        other
            .children
            .slot(1)?
            .ok_or_else(|| vortex_err!("missing fixture slot"))?
            .state
            .as_ref()
            .is_some_and(|state| state.probe.is_none())
    );

    let mut moved = (probe, ());
    assert_eq!(
        std::ptr::from_mut(
            moved
                .0
                .children
                .slot(1)?
                .ok_or_else(|| vortex_err!("missing fixture slot"))?
        ),
        first
    );
    assert_eq!(
        moved
            .0
            .children
            .slot(1)?
            .ok_or_else(|| vortex_err!("missing fixture slot"))?
            .execute_scalar(1, &mut ctx)?,
        20i32.into()
    );
    Ok(())
}

#[rstest]
#[case(ProbeUsage::Once)]
#[case(ProbeUsage::Repeated)]
fn access_checks_bounds_and_nulls(#[case] usage: ProbeUsage) -> VortexResult<()> {
    let array = PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array();
    let mut ctx = crate::array_session().create_execution_ctx();
    let mut probe = array.probe(usage);
    assert!(
        probe
            .state
            .as_ref()
            .is_none_or(|state| state.probe.is_none())
    );
    assert!(probe.execute_scalar(3, &mut ctx).is_err());
    assert!(
        probe
            .state
            .as_ref()
            .is_none_or(|state| state.probe.is_none())
    );
    assert!(probe.execute_scalar(1, &mut ctx)?.is_null());
    assert_eq!(
        probe
            .state
            .as_ref()
            .is_some_and(|state| state.probe.is_some()),
        usage == ProbeUsage::Repeated
    );
    assert_eq!(
        array.execute_scalar(0, &mut ctx)?,
        Scalar::primitive(10i32, array.dtype().nullability())
    );
    assert!(array.execute_scalar(1, &mut ctx)?.is_null());
    assert!(array.execute_scalar(3, &mut ctx).is_err());
    for index in [2, 1, 0, 2] {
        assert_eq!(
            probe.execute_scalar(index, &mut ctx)?,
            array.execute_scalar(index, &mut ctx)?
        );
    }
    if usage == ProbeUsage::Once {
        assert!(probe.state.is_none());
    }
    Ok(())
}

#[rstest]
fn slot_access_uses_context_policy(
    #[values(ProbeUsage::Once, ProbeUsage::Repeated)] usage: ProbeUsage,
) -> VortexResult<()> {
    let leaf = PrimitiveArray::from_option_iter([Some(10i32), None]).into_array();
    let array = StructArray::from_fields(&[("values", leaf)])?.into_array();
    let mut ctx = crate::array_session().create_execution_ctx();
    let mut retained = ProbeState::<()>::new(&array);
    {
        let mut probe = match usage {
            ProbeUsage::Once => ProbeAccess::Once(&array),
            ProbeUsage::Repeated => retained.access(),
        };
        assert!(probe.slot(0)?.is_none());
        assert!(probe.slot(99).is_err());
        for index in [1, 0, 1, 0] {
            let mut slot = probe
                .slot(1)?
                .ok_or_else(|| vortex_err!("missing fixture slot"))?;
            assert_eq!(
                slot.execute_scalar(index, &mut ctx)?,
                slot.array().execute_scalar(index, &mut ctx)?
            );
        }
        assert!(
            probe
                .slot(1)?
                .ok_or_else(|| vortex_err!("missing fixture slot"))?
                .execute_scalar(2, &mut ctx)
                .is_err()
        );
    }
    match usage {
        ProbeUsage::Once => assert_eq!(retained.children.slots.capacity(), 0),
        ProbeUsage::Repeated => assert!(
            retained
                .children
                .slot(1)?
                .ok_or_else(|| vortex_err!("missing fixture slot"))?
                .state
                .as_ref()
                .is_some_and(|state| state.probe.is_some())
        ),
    }
    Ok(())
}

#[test]
fn validity_is_owned_by_probe_state() -> VortexResult<()> {
    let array = PrimitiveArray::from_option_iter([Some(10i32), None]).into_array();
    let mut ctx = crate::array_session().create_execution_ctx();
    let mut state = ProbeState::<()>::new(&array);
    assert!(state.validity.is_none());
    assert_eq!(
        state.access().validity()?.execute_scalar(1, &mut ctx)?,
        false.into()
    );
    assert_eq!(state.children.slots.capacity(), 0);
    let mut validity = state
        .validity
        .take()
        .ok_or_else(|| vortex_err!("missing validity probe"))?;
    drop(state);
    drop(array);
    assert_eq!(validity.execute_scalar(0, &mut ctx)?, true.into());
    assert_eq!(validity.execute_scalar(1, &mut ctx)?, false.into());
    Ok(())
}

#[rstest]
fn validity_accessor_obeys_retention_policy(
    #[values(ProbeUsage::Once, ProbeUsage::Repeated)] usage: ProbeUsage,
) -> VortexResult<()> {
    let array = PrimitiveArray::from_option_iter([Some(10i32), None]).into_array();
    let mut ctx = crate::array_session().create_execution_ctx();
    let mut retained = ProbeState::<()>::new(&array);
    for index in [0, 1, 0] {
        let mut access = match usage {
            ProbeUsage::Once => ProbeAccess::Once(&array),
            ProbeUsage::Repeated => retained.access(),
        };
        assert_eq!(
            access.validity()?.execute_scalar(index, &mut ctx)?,
            (index == 0).into()
        );
        assert!(access.validity()?.execute_scalar(2, &mut ctx).is_err());
    }
    assert_eq!(retained.validity.is_some(), usage == ProbeUsage::Repeated);
    assert_eq!(retained.children.slots.capacity(), 0);
    Ok(())
}
