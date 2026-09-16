// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(all(target_arch = "x86_64", not(miri)))]

use std::panic::RefUnwindSafe;
use std::panic::catch_unwind;

use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;

use self::gather::Avx2Gather;
use self::gather::GatherFn;
use super::*;
use crate::arrays::fixed_width::Record;
use crate::arrays::fixed_width::record::Record4;
use crate::arrays::fixed_width::record::Record8;
use crate::dtype::UnsignedPType;

/// Pairs a record type with the AVX2 gather lane of the same width.
trait GatherRecord: Record + RefUnwindSafe {
    type Lane;
    fn from_index(index: usize) -> Self;
}

impl GatherRecord for Record4 {
    type Lane = u32;
    fn from_index(index: usize) -> Self {
        Self::from_bytes(u32::try_from(index).unwrap().to_le_bytes())
    }
}

impl GatherRecord for Record8 {
    type Lane = u64;
    fn from_index(index: usize) -> Self {
        Self::from_bytes(u64::try_from(index).unwrap().to_le_bytes())
    }
}

fn records<R: GatherRecord>(range: impl IntoIterator<Item = usize>) -> Vec<R> {
    range.into_iter().map(R::from_index).collect()
}

fn take_avx2_if_supported<R: GatherRecord, I: UnsignedPType>(
    values: &[R],
    indices: &[I],
) -> Option<Buffer<R>>
where
    Avx2Gather: GatherFn<u8, R::Lane>
        + GatherFn<u16, R::Lane>
        + GatherFn<u32, R::Lane>
        + GatherFn<u64, R::Lane>,
{
    if !is_x86_feature_detected!("avx2") {
        return None;
    }

    // SAFETY: AVX2 support was detected above, and `Record` guarantees that every byte in the
    // values is initialized.
    Some(unsafe {
        take_avx2::<R, R::Lane, I>(values, indices, &BufferAllocatorRef::statically_allocated())
    })
}

fn assert_avx2_take_panics<R, I>(values: &[R], indices: &[I], expected: &str)
where
    R: GatherRecord,
    I: UnsignedPType + RefUnwindSafe,
    Avx2Gather: GatherFn<u8, R::Lane>
        + GatherFn<u16, R::Lane>
        + GatherFn<u32, R::Lane>
        + GatherFn<u64, R::Lane>,
{
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // SAFETY: AVX2 support was detected above, and `Record` guarantees that every byte in the
    // values is initialized.
    let result = catch_unwind(|| unsafe {
        take_avx2::<R, R::Lane, I>(values, indices, &BufferAllocatorRef::statically_allocated())
    });
    let Err(payload) = result else {
        panic!("take should panic for an invalid index");
    };
    let message = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str));
    assert_eq!(message, Some(expected));
}

macro_rules! test_cases {
    (index_type => $IDX:ty, record_types => $($REC:ty),+) => {
        paste::paste! {
            $(
                #[test]
                fn [<test_avx2_take_simple_ $IDX _ $REC:snake>]() {
                    let values = records::<$REC>(1..=127);
                    let indices: Vec<$IDX> = (0..127).collect();

                    let Some(result) = take_avx2_if_supported(&values, &indices) else {
                        return;
                    };
                    assert_eq!(&values, result.as_slice());
                }

                #[test]
                fn [<test_avx2_take_empty_ $IDX _ $REC:snake>]() {
                    let values: Vec<$REC> = vec![];
                    let indices: Vec<$IDX> = (0..127).collect();

                    assert_avx2_take_panics(
                        &values,
                        &indices,
                        "cannot take a non-empty set of indices from an empty buffer",
                    );
                }

                #[test]
                fn [<test_avx2_take_invalid_ $IDX _ $REC:snake>]() {
                    let values = records::<$REC>(1..=127);
                    let indices: Vec<$IDX> = (127..=254).collect();

                    assert_avx2_take_panics(&values, &indices, "take index out of bounds");
                }
            )+
        }
    };
}

test_cases!(index_type => u8, record_types => Record4, Record8);
test_cases!(index_type => u16, record_types => Record4, Record8);
test_cases!(index_type => u32, record_types => Record4, Record8);
test_cases!(index_type => u64, record_types => Record4, Record8);

#[test]
fn last_valid_u8_index() {
    let values = records::<Record8>(0..=255);
    let indices: Vec<u8> = vec![255; 20];

    let Some(result) = take_avx2_if_supported(&values, &indices) else {
        return;
    };
    assert_eq!(&[Record8::from_index(255); 20], result.as_slice());
}

#[test]
fn last_valid_u16_index() {
    let values = records::<Record8>(0..=65535);
    let indices: Vec<u16> = vec![65535; 20];

    let Some(result) = take_avx2_if_supported(&values, &indices) else {
        return;
    };
    assert_eq!(&[Record8::from_index(65535); 20], result.as_slice());
}

#[test]
fn empty_values_and_indices() {
    let Some(result) = take_avx2_if_supported::<Record4, u32>(&[], &[]) else {
        return;
    };

    assert!(result.is_empty());
}

#[test]
fn i32_gather_addressable_length_boundary() {
    assert!(i32_gather_can_address(i32::MAX as usize + 1));
    assert!(!i32_gather_can_address(i32::MAX as usize + 2));
}

#[test]
fn invalid_index_only_in_simd_block() {
    let values = records::<Record4>([10, 20, 30]);
    let indices = vec![3u32, 0, 1, 2, 0, 1, 2, 0, 1];

    assert_avx2_take_panics(&values, &indices, "take index out of bounds");
}

#[test]
fn gather_preserves_arbitrary_record_bytes() {
    let values: Vec<Record4> = (1u32..=200)
        .map(|x| Record4::from_bytes(x.to_le_bytes()))
        .collect();
    let indices: Vec<u32> = (0..200).collect();

    let Some(result) = take_avx2_if_supported(&values, &indices) else {
        return;
    };
    assert_eq!(values.as_slice(), result.as_slice());
}

#[test]
fn u32_max_index_in_u32_lane() {
    let values = vec![Record4::default(); 8];
    // The first eight indices execute in the SIMD loop; the scalar remainder is valid.
    let indices = vec![0, u32::MAX, 2, 3, 4, 5, 6, 7, 0];

    assert_avx2_take_panics(&values, &indices, "take index out of bounds");
}

#[test]
fn u64_max_index_in_u32_lane() {
    let values = vec![Record4::default(); 8];
    // The first four indices execute in the SIMD loop; the scalar remainder is valid.
    let indices = vec![0, u64::MAX, 2, 3, 0];

    assert_avx2_take_panics(&values, &indices, "take index out of bounds");
}

#[test]
fn u64_max_index_in_u64_lane() {
    let values = vec![Record8::default(); 8];
    // The first four indices execute in the SIMD loop; the scalar remainder is valid.
    let indices = vec![0, u64::MAX, 2, 3, 0];

    assert_avx2_take_panics(&values, &indices, "take index out of bounds");
}
