// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::Cell;

use rstest::rstest;
use vortex_error::VortexResult;

use super::ProbeAccess;
use super::ProbeState;
use super::ProbeStorage;
use super::ProbeUsage;
use super::RetainedProbe;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::scalar::Scalar;

struct TrackedProbe<'a> {
    value: &'a i32,
    reads: &'a Cell<usize>,
    drops: &'a Cell<usize>,
}

impl RetainedProbe for TrackedProbe<'_> {
    fn scalar_at(&mut self, _index: usize, _ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        self.reads.set(self.reads.get() + 1);
        Ok((*self.value).into())
    }
}

impl Drop for TrackedProbe<'_> {
    fn drop(&mut self) {
        self.drops.set(self.drops.get() + 1);
    }
}

#[test]
fn borrowed_state_is_initialized_once_and_survives_moves() -> VortexResult<()> {
    let value = 42;
    let initialized = Cell::new(0);
    let reads = Cell::new(0);
    let drops = Cell::new(0);
    let mut ctx = crate::array_session().create_execution_ctx();
    let mut storage = ProbeStorage::new();
    let init = || -> Box<dyn RetainedProbe + '_> {
        initialized.set(initialized.get() + 1);
        Box::new(TrackedProbe {
            value: &value,
            reads: &reads,
            drops: &drops,
        })
    };
    assert!(storage.probe.is_none());
    assert_eq!(
        storage.get_or_init(init).scalar_at(0, &mut ctx)?,
        value.into()
    );
    let mut moved = (storage, ());
    assert_eq!(
        moved.0.get_or_init(init).scalar_at(0, &mut ctx)?,
        value.into()
    );
    assert_eq!(initialized.get(), 1);
    assert_eq!(reads.get(), 2);
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
    assert!(probe.child(0).is_err());
    assert!(probe.child(99).is_err());
    assert_eq!(probe.children.slots.capacity(), 0);

    let first = std::ptr::from_mut(probe.child(1)?);
    assert!(probe.children.slots[2].is_none());
    let (state, children) = probe.parts();
    let child = children.child(1)?;
    assert!(
        child
            .state
            .as_ref()
            .is_some_and(|state| state.probe.is_none())
    );
    assert!(child.scalar_at(2, &mut ctx).is_err());
    assert!(
        child
            .state
            .as_ref()
            .is_some_and(|state| state.probe.is_none())
    );
    assert_eq!(child.scalar_at(0, &mut ctx)?, 10i32.into());
    *state += 1;
    assert_eq!(*probe.state_mut(), 1);
    assert_eq!(std::ptr::from_mut(probe.child(1)?), first);
    assert!(
        probe
            .child(1)?
            .state
            .as_ref()
            .is_some_and(|state| state.probe.is_some())
    );

    // Identical sources in different slots still get independent probe state.
    assert_ne!(std::ptr::from_mut(probe.child(2)?), first);
    let source = array.slots()[1]
        .as_ref()
        .ok_or_else(|| vortex_error::vortex_err!("missing fixture slot"))?;
    assert!(std::ptr::eq(probe.child(1)?.array(), source));
    assert!(
        probe
            .child(2)?
            .state
            .as_ref()
            .is_some_and(|state| state.probe.is_none())
    );
    let mut other = ProbeState::<usize>::new(&array);
    assert!(
        other
            .child(1)?
            .state
            .as_ref()
            .is_some_and(|state| state.probe.is_none())
    );

    let mut moved = (probe, ());
    assert_eq!(std::ptr::from_mut(moved.0.child(1)?), first);
    assert_eq!(moved.0.child(1)?.scalar_at(1, &mut ctx)?, 20i32.into());
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
    assert!(probe.scalar_at(3, &mut ctx).is_err());
    assert!(
        probe
            .state
            .as_ref()
            .is_none_or(|state| state.probe.is_none())
    );
    for index in [2, 1, 0, 2] {
        assert_eq!(
            probe.scalar_at(index, &mut ctx)?,
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
            ProbeUsage::Repeated => ProbeAccess::Repeated(&mut retained),
        };
        assert!(probe.slot(0).is_err());
        assert!(probe.slot(99).is_err());
        for index in [1, 0, 1, 0] {
            let mut slot = probe.slot(1)?;
            assert_eq!(
                slot.execute_scalar(index, &mut ctx)?,
                slot.array().execute_scalar(index, &mut ctx)?
            );
        }
        assert!(probe.slot(1)?.execute_scalar(2, &mut ctx).is_err());
    }
    match usage {
        ProbeUsage::Once => assert_eq!(retained.children.slots.capacity(), 0),
        ProbeUsage::Repeated => assert!(
            retained
                .child(1)?
                .state
                .as_ref()
                .is_some_and(|state| state.probe.is_some())
        ),
    }
    Ok(())
}
