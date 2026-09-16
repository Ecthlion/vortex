// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::distr::Uniform;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::scalar::Scalar;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedPrimitiveArray;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

const ARRAY_LEN: usize = 65_536;
/// One search over [`ARRAY_LEN`] elements is ~16 comparisons, small enough that a single search
/// measures mostly harness overhead. Search a batch per iteration instead; the targets differ, so
/// the searches cannot be folded together.
const SEARCH_TARGETS: usize = 64;
/// One null in sixteen, so validity is a bit buffer rather than a uniform `AllValid`.
const NULL_EVERY: usize = 16;

/// Sorted values, and the targets to search for.
fn fixture() -> (Buffer<i32>, Vec<i32>) {
    let mut rng = StdRng::seed_from_u64(0);
    let range = Uniform::new(0, ARRAY_LEN as i32).unwrap();

    let mut values: Vec<i32> = (0..ARRAY_LEN).map(|_| rng.sample(range)).collect();
    values.sort_unstable();
    let targets = (0..SEARCH_TARGETS).map(|_| rng.sample(range)).collect();

    (Buffer::from(values), targets)
}

/// Nulls sort first, so they occupy the front of the array rather than being interleaved.
fn nullable_fixture() -> (ArrayRef, Vec<i32>) {
    let (values, targets) = fixture();
    let null_count = values.len() / NULL_EVERY;
    let array = PrimitiveArray::from_option_iter(
        std::iter::repeat_n(None, null_count)
            .chain(values.iter().skip(null_count).copied().map(Some)),
    );
    (array.into_array(), targets)
}

/// The shape `RunEnd::find_physical_index` and `Patches::search_index` take: a non-nullable
/// array of ends or indices searched with a `usize` needle, one searcher per search.
#[divan::bench]
fn usize_needle_non_nullable(bencher: Bencher) {
    let (values, targets) = fixture();
    let array = PrimitiveArray::new(values, Validity::NonNullable).into_array();
    bencher
        .with_inputs(|| (&array, &targets, SESSION.create_execution_ctx()))
        .bench_refs(|(array, targets, ctx)| {
            let mut total = 0;
            for &target in targets.iter() {
                total += SearchSortedPrimitiveArray::<i32>::new(array, ctx)
                    .search_sorted(&(target as usize), SearchSortedSide::Right)
                    .unwrap()
                    .to_index();
            }
            total
        });
}

/// A typed needle over an array whose dtype is non-nullable: validity never has to be consulted.
#[divan::bench]
fn typed_needle_non_nullable(bencher: Bencher) {
    let (values, targets) = fixture();
    let array = PrimitiveArray::new(values, Validity::NonNullable).into_array();
    bencher
        .with_inputs(|| (&array, &targets, SESSION.create_execution_ctx()))
        .bench_refs(|(array, targets, ctx)| {
            let mut total = 0;
            for &target in targets.iter() {
                total += SearchSortedPrimitiveArray::<i32>::new(array, ctx)
                    .search_sorted(&target, SearchSortedSide::Left)
                    .unwrap()
                    .to_index();
            }
            total
        });
}

/// An `Option` needle over an array whose validity is a bit buffer: the worst case, since every
/// comparison has to resolve validity as well as read the value.
#[divan::bench]
fn option_needle_nullable(bencher: Bencher) {
    let (array, targets) = nullable_fixture();
    bencher
        .with_inputs(|| (&array, &targets, SESSION.create_execution_ctx()))
        .bench_refs(|(array, targets, ctx)| {
            let mut total = 0;
            for &target in targets.iter() {
                total += SearchSortedPrimitiveArray::<i32>::new(array, ctx)
                    .search_sorted(&Some(target), SearchSortedSide::Left)
                    .unwrap()
                    .to_index();
            }
            total
        });
}

/// The untyped path: comparisons go through `Scalar`, which cannot be specialized on the ptype.
#[divan::bench]
fn scalar_needle_non_nullable(bencher: Bencher) {
    let (values, targets) = fixture();
    let array = PrimitiveArray::new(values, Validity::NonNullable).into_array();
    let targets: Vec<Scalar> = targets.into_iter().map(Scalar::from).collect();
    bencher
        .with_inputs(|| (&array, &targets))
        .bench_refs(|(array, targets)| {
            let mut total = 0;
            for target in targets.iter() {
                total += array
                    .search_sorted(target, SearchSortedSide::Left)
                    .unwrap()
                    .to_index();
            }
            total
        });
}
