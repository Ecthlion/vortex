// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use super::*;
use crate::scheme::CompressionEstimate;
use crate::scheme::CompressorContext;
use crate::scheme::EstimateVerdict;
use crate::stats::ArrayAndStats;

static V1_ID: CachedId = CachedId::new("test.format_v1");
static V2_ID: CachedId = CachedId::new("test.format_v2");
static V3_ID: CachedId = CachedId::new("test.format_v3");
static OTHER_ID: CachedId = CachedId::new("test.other");

/// A scheme version. `produced` are the wire IDs it writes and `replaces` the versions it
/// supersedes.
struct TestScheme {
    name: &'static str,
    produced: &'static [&'static CachedId],
    replaces: &'static [&'static TestScheme],
}

impl std::fmt::Debug for TestScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

impl Scheme for TestScheme {
    fn scheme_name(&self) -> &'static str {
        self.name
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        self.produced.iter().map(|id| ***id).collect()
    }

    fn replaces(&self) -> Vec<SchemeId> {
        self.replaces.iter().map(|scheme| scheme.id()).collect()
    }

    fn num_children(&self) -> usize {
        1
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Verdict(EstimateVerdict::Skip)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        Ok(data.array().clone())
    }
}

/// The frozen format.
static V1: TestScheme = TestScheme {
    name: "test.scheme_v1",
    produced: &[&V1_ID],
    replaces: &[],
};
/// Writes the frozen format for some values and a new one for others, like wide decimals.
static V2: TestScheme = TestScheme {
    name: "test.scheme_v2",
    produced: &[&V1_ID, &V2_ID],
    replaces: &[&V1],
};
/// A format that stands alone. It names only the version before it.
static V3: TestScheme = TestScheme {
    name: "test.scheme_v3",
    produced: &[&V3_ID],
    replaces: &[&V2],
};
static OTHER: TestScheme = TestScheme {
    name: "test.other",
    produced: &[&OTHER_ID],
    replaces: &[],
};

fn active(compressor: &CascadingCompressor) -> Vec<SchemeId> {
    compressor
        .schemes()
        .iter()
        .map(|scheme| scheme.id())
        .collect()
}

#[test]
fn a_replacement_drops_the_schemes_it_names() {
    assert_eq!(
        active(&CascadingCompressor::new(vec![&V2, &V1])),
        vec![V2.id()]
    );
    assert_eq!(active(&CascadingCompressor::new(vec![&V1])), vec![V1.id()]);
}

/// V3 names only V2, but V2 names V1, and the lists of dropped schemes still apply.
#[test]
fn replacement_is_transitive() {
    assert_eq!(
        active(&CascadingCompressor::new(vec![&V3, &V2, &V1])),
        vec![V3.id()]
    );
}

/// Without V2 in the list, nothing names V1, so V3 and V1 are both active.
#[test]
fn replacement_only_follows_given_schemes() {
    assert_eq!(
        active(&CascadingCompressor::new(vec![&V3, &V1])),
        vec![V3.id(), V1.id()]
    );
}

#[test]
fn registration_order_is_preserved() {
    assert_eq!(
        active(&CascadingCompressor::new(vec![&OTHER, &V3, &V2, &V1])),
        vec![OTHER.id(), V3.id()]
    );
}

#[test]
fn replacing_an_unregistered_scheme_is_a_no_op() {
    assert_eq!(active(&CascadingCompressor::new(vec![&V2])), vec![V2.id()]);
}

#[test]
fn has_scheme_reports_the_active_version() {
    let compressor = CascadingCompressor::new(vec![&V2, &V1]);
    assert!(compressor.has_scheme(V2.id()));
    assert!(!compressor.has_scheme(V1.id()));
}
