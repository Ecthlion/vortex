// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cascading array compression implementation.

mod cascade;
mod constant;
mod sample;
mod select;
mod structural;

use vortex_utils::aliases::hash_set::HashSet;

use crate::builtins::IntDictScheme;
use crate::scheme::ChildSelection;
use crate::scheme::DescendantExclusion;
use crate::scheme::Scheme;
use crate::scheme::SchemeExt;
use crate::scheme::SchemeId;

/// Synthetic scheme ID used for the compressor's own root-level cascading.
pub(crate) const ROOT_SCHEME_ID: SchemeId = SchemeId {
    name: "vortex.compressor.root",
};

/// The main compressor type implementing cascading adaptive compression.
///
/// This compressor applies adaptive compression [`Scheme`]s to arrays based on their data types and
/// characteristics. It recursively compresses nested structures like structs and lists, and chooses
/// optimal compression schemes for leaf types.
///
/// The compressor works by:
/// 1. Canonicalizing input arrays to a standard representation.
/// 2. Pre-filtering schemes by [`Scheme::matches`] and exclusion rules.
/// 3. Evaluating each matching scheme's compression estimate and resolving deferred work.
/// 4. Compressing with the best scheme and verifying the result is smaller.
///
/// No scheme may appear twice in a cascade chain. The compressor enforces this automatically
/// along with push/pull exclusion rules declared by each scheme.
///
/// Downstream crates usually wrap this type with a preconfigured scheme set. Use it directly when
/// embedding a custom fixed scheme list or testing scheme interactions.
#[derive(Debug, Clone)]
pub struct CascadingCompressor {
    /// The active compression schemes: those given, minus any that another given scheme
    /// [replaces](Scheme::replaces).
    schemes: Vec<&'static dyn Scheme>,

    /// Descendant exclusion rules for the compressor's own cascading (e.g. excluding Dict from
    /// list offsets).
    root_exclusions: Vec<DescendantExclusion>,
}

impl CascadingCompressor {
    /// Creates a new compressor with the given schemes.
    ///
    /// A scheme that another given scheme [replaces](Scheme::replaces) is dropped, so two
    /// versions of one scheme never compete. Restrict the list to the schemes whose serialized
    /// IDs the writer permits before calling this, so a newer version that is not permitted is
    /// gone before replacement and the version it replaces stays.
    ///
    /// Root-level exclusion rules (e.g. excluding Dict from list offsets) are built automatically.
    pub fn new(schemes: Vec<&'static dyn Scheme>) -> Self {
        let replaced: HashSet<SchemeId> = schemes
            .iter()
            .flat_map(|scheme| scheme.replaces())
            .collect();
        let schemes = schemes
            .into_iter()
            .filter(|scheme| !replaced.contains(&scheme.id()))
            .collect();

        // Root exclusion: exclude IntDict from list/listview offsets (monotonically
        // increasing data where dictionary encoding is wasteful).
        let root_exclusions = vec![DescendantExclusion {
            excluded: IntDictScheme.id(),
            children: ChildSelection::One(structural::root_list_children::OFFSETS),
        }];

        Self {
            schemes,
            root_exclusions,
        }
    }

    /// The schemes active for compression, in registration order.
    pub fn schemes(&self) -> &[&'static dyn Scheme] {
        &self.schemes
    }

    /// Returns whether `scheme` is active for compression.
    ///
    /// A scheme given to [`new`](Self::new) is inactive when another given scheme
    /// [replaces](Scheme::replaces) it.
    pub fn has_scheme(&self, scheme: SchemeId) -> bool {
        self.schemes
            .iter()
            .any(|candidate| candidate.id() == scheme)
    }
}

// NB: Cascading compression logic is located in `vortex-compressor/src/compressor/cascade.rs`.

#[cfg(test)]
mod tests;

#[cfg(test)]
mod replacement_tests;
