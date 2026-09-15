// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped registry of cast rules.
//!
//! Every cast is a [`CastRule`]. [`CastRules`] holds the rules a session consults when it executes
//! a [`Cast`](super::Cast): the first rule that accepts the source and target dtypes binds the
//! cast, and a pair no rule accepts cannot be cast in that session. A rule sees both dtypes,
//! extension metadata included, so it can accept a cast for some instances of a dtype only, such
//! as timestamps that share a timezone.
//!
//! [`CastSession`] is the session variable that owns the registry. Its [`Default`] installs the
//! [standard rules](super::standard), which are the casts Vortex performs between its own dtypes,
//! and [`TimestampCast`]. [`CastSession::empty`] installs none, so nothing casts until a rule is
//! registered. The most recently registered rule is consulted first, so a rule registered into a
//! default session adds a cast the standard rules do not perform, or replaces one they do.
//!
//! Rules apply when a cast executes, which is where the session is known. Binding a cast
//! expression therefore cannot consult them, so [`Cast::return_dtype`](super::Cast) accepts every
//! pair of dtypes and an unsupported cast fails when it executes. Reduce rules and
//! [`Scalar::cast`](crate::scalar::Scalar::cast) have no session either: they use the rules of
//! the default session, so a rule registered into another session applies to arrays executed in
//! that session, not to constants folded before execution.

use std::any::Any;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use arc_swap::ArcSwap;
use vortex_error::VortexResult;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::extension::datetime::TimestampCast;
use crate::scalar_fn::fns::cast::standard::BoolCast;
use crate::scalar_fn::fns::cast::standard::DecimalCast;
use crate::scalar_fn::fns::cast::standard::ListCast;
use crate::scalar_fn::fns::cast::standard::MapCast;
use crate::scalar_fn::fns::cast::standard::NullCast;
use crate::scalar_fn::fns::cast::standard::NullabilityCast;
use crate::scalar_fn::fns::cast::standard::PrimitiveCast;
use crate::scalar_fn::fns::cast::standard::StorageCast;
use crate::scalar_fn::fns::cast::standard::StructCast;

/// A cast bound to a concrete source and target dtype.
///
/// The function receives the array to cast, which may be of any encoding and may be lazy, and
/// returns an array of the same length whose dtype is exactly the target dtype. The result may
/// itself be lazy: the executor keeps evaluating it.
pub type CastFn = Arc<dyn Fn(ArrayRef, &mut ExecutionCtx) -> VortexResult<ArrayRef> + Send + Sync>;

/// A cast between dtypes.
pub trait CastRule: Debug + Send + Sync + 'static {
    /// Binds a cast from `source` to `target`.
    ///
    /// Returns `Ok(None)` when this rule does not cover the pair, in which case later rules are
    /// consulted. Returning an error rejects the cast outright, even if a later rule would accept
    /// it.
    fn bind(&self, source: &DType, target: &DType) -> VortexResult<Option<CastFn>>;
}

/// A type-erased, shared [`CastRule`].
pub type CastRuleRef = Arc<dyn CastRule>;

/// An ordered registry of [`CastRule`]s.
///
/// Clones share their storage, so rules registered through any clone are visible to all of them.
#[derive(Clone, Debug, Default)]
pub struct CastRules {
    rules: Arc<ArcSwap<Vec<CastRuleRef>>>,
}

impl CastRules {
    /// Creates a registry with no rules.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Registers a rule.
    ///
    /// The most recently registered rule is consulted first, so a rule registered later replaces
    /// earlier rules for every pair of dtypes it accepts.
    pub fn register<R: CastRule>(&self, rule: R) {
        let rule: CastRuleRef = Arc::new(rule);
        self.rules.rcu(|rules| {
            let mut new_rules = Vec::with_capacity(rules.len() + 1);
            new_rules.push(Arc::clone(&rule));
            new_rules.extend(rules.iter().cloned());
            new_rules
        });
    }

    /// Binds a cast from `source` to `target` using the first rule that accepts the pair.
    ///
    /// Returns `Ok(None)` when no rule accepts it.
    pub fn bind(&self, source: &DType, target: &DType) -> VortexResult<Option<CastFn>> {
        for rule in self.rules.load().iter() {
            if let Some(cast_fn) = rule.bind(source, target)? {
                return Ok(Some(cast_fn));
            }
        }
        Ok(None)
    }

    /// Returns the number of registered rules.
    pub fn len(&self) -> usize {
        self.rules.load().len()
    }

    /// Returns whether no rules are registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Session variable owning a session's [`CastRules`].
#[derive(Clone, Debug)]
pub struct CastSession {
    rules: CastRules,
}

impl CastSession {
    /// Creates a session variable with no cast rules, so no cast is possible until one is
    /// registered.
    pub fn empty() -> Self {
        Self {
            rules: CastRules::empty(),
        }
    }

    /// Returns the cast rule registry.
    pub fn rules(&self) -> &CastRules {
        &self.rules
    }
}

/// Derefs to the held [`CastRules`], so a [`CastSession`] read from a session can be used
/// wherever a `&CastRules` is expected.
impl Deref for CastSession {
    type Target = CastRules;

    fn deref(&self) -> &CastRules {
        &self.rules
    }
}

/// Installs the standard rules and [`TimestampCast`].
impl Default for CastSession {
    fn default() -> Self {
        let this = Self::empty();
        this.register(NullabilityCast);
        this.register(NullCast);
        this.register(BoolCast);
        this.register(PrimitiveCast);
        this.register(DecimalCast);
        this.register(ListCast);
        this.register(MapCast);
        this.register(StructCast);
        this.register(StorageCast);
        this.register(TimestampCast);
        this
    }
}

impl SessionVar for CastSession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension trait for accessing a session's cast rules.
pub trait CastSessionExt: SessionExt {
    /// Returns the cast rule registry, installing the default rules if the session has none.
    fn casts(&self) -> SessionGuard<'_, CastSession> {
        self.get::<CastSession>()
    }
}
impl<S: SessionExt> CastSessionExt for S {}
