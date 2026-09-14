// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped registry of cast rules.
//!
//! [`CastRules`] holds the [`CastRule`]s a session consults when it executes a
//! [`Cast`](super::Cast). Rules are consulted before the built-in casts, so a rule can add a cast
//! Vortex does not perform by itself (an integer into a string, a storage dtype into an extension
//! dtype) or replace a built-in cast between built-in dtypes. A rule sees both dtypes, extension
//! metadata included, so it can accept a cast for some instances of a dtype only, such as
//! timestamps that share a timezone.
//!
//! [`CastSession`] is the session variable that owns the registry. Its [`Default`] installs
//! vortex-array's own rules, currently the timestamp unit conversion; [`CastSession::empty`]
//! installs none.
//!
//! Rules apply when a cast executes, which is where the session is known. Two consequences
//! follow. Binding a cast expression cannot consult the rules, so
//! [`Cast::return_dtype`](super::Cast) accepts every pair of dtypes and an unsupported cast fails
//! when it executes. And the optimizer, which has no session either, folds constant and literal
//! casts with the built-in casts only: a rule that adds a cast is always reached, while a rule
//! that replaces a built-in cast does not apply to constants the optimizer folded first.

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

/// A cast bound to a concrete source and target dtype.
///
/// The function receives the array to cast, which may be of any encoding and may be lazy, and
/// returns an array of the same length whose dtype is exactly the target dtype. The result may
/// itself be lazy: the executor keeps evaluating it.
pub type CastFn = Arc<dyn Fn(ArrayRef, &mut ExecutionCtx) -> VortexResult<ArrayRef> + Send + Sync>;

/// A pluggable cast between dtypes.
pub trait CastRule: Debug + Send + Sync + 'static {
    /// Binds a cast from `source` to `target`.
    ///
    /// Returns `Ok(None)` when this rule does not cover the pair, in which case later rules and
    /// then the built-in casts are consulted. Returning an error rejects the cast outright, even
    /// if a later rule or a built-in cast would accept it.
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
    /// earlier rules, and the built-in casts, for every pair of dtypes it accepts.
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
    /// Returns `Ok(None)` when no rule accepts it, leaving the cast to the built-in casts.
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
    /// Creates a session variable with no cast rules, so only the built-in casts apply.
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

impl Default for CastSession {
    fn default() -> Self {
        let this = Self::empty();
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
