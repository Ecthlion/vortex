// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DuckDB's hash functions, reimplemented.
//!
//! A bloom filter pushed down from a hash join holds the hashes of the join's build-side keys, so
//! probing it from Vortex means reproducing the hash DuckDB would have computed for the same
//! value. These functions mirror `src/common/types/hash.cpp` and
//! `src/common/vector_operations/vector_hash.cpp` in the DuckDB sources, and the
//! `hash_matches_duckdb` test pins them to `duckdb::Value::Hash`.

/// The hash DuckDB assigns to a NULL key (`duckdb::HashOp::NULL_HASH`).
pub(crate) const NULL_HASH: u64 = 0xbf58_476d_1ce4_e5b9;

/// The multiplier of `duckdb::MurmurHash64`, also used to mix in byte blocks.
const MULTIPLIER: u64 = 0xd6e8_feb8_6659_fd93;
/// The seed and multiplier `duckdb::HashBytes` mixes the byte length with.
const BYTES_SEED: u64 = 0xe17a_1465;
const BYTES_MULTIPLIER: u64 = 0xc6a4_a793_5bd1_e995;

/// `duckdb::MurmurHash64`.
#[inline]
pub fn murmur_hash64(mut x: u64) -> u64 {
    x ^= x >> 32;
    x = x.wrapping_mul(MULTIPLIER);
    x ^= x >> 32;
    x = x.wrapping_mul(MULTIPLIER);
    x ^= x >> 32;
    x
}

/// `duckdb::HashBytes`, which hashes VARCHAR and BLOB keys.
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = BYTES_SEED ^ (bytes.len() as u64).wrapping_mul(BYTES_MULTIPLIER);

    let (blocks, remainder) = bytes.as_chunks::<8>();
    for block in blocks {
        hash ^= u64::from_le_bytes(*block);
        hash = hash.wrapping_mul(MULTIPLIER);
    }

    if !remainder.is_empty() {
        // DuckDB loads the trailing bytes into a zeroed word and byte-swaps on big-endian
        // targets, which is the same as reading the zero-padded tail little-endian.
        let mut tail = [0u8; 8];
        tail[..remainder.len()].copy_from_slice(remainder);
        hash ^= u64::from_le_bytes(tail);
        hash = hash.wrapping_mul(MULTIPLIER);
    }

    murmur_hash64(hash)
}

/// The hash DuckDB computes for a value of a physical type.
pub trait DuckDbHash: Copy {
    fn duckdb_hash(self) -> u64;
}

/// DuckDB hashes every type narrower than 64 bits through `MurmurHash32`, which widens the value
/// to `uint32_t` first: signed types sign-extend before being truncated to 32 bits, unsigned
/// types zero-extend.
macro_rules! impl_narrow_hash {
    ($($ty:ty),+ $(,)?) => {
        $(impl DuckDbHash for $ty {
            #[inline]
            fn duckdb_hash(self) -> u64 {
                murmur_hash64(u64::from(self as u32))
            }
        })+
    };
}

macro_rules! impl_wide_hash {
    ($($ty:ty),+ $(,)?) => {
        $(impl DuckDbHash for $ty {
            #[inline]
            fn duckdb_hash(self) -> u64 {
                murmur_hash64(self as u64)
            }
        })+
    };
}

impl_narrow_hash!(i8, i16, i32, u8, u16, u32);
impl_wide_hash!(i64, u64);

impl DuckDbHash for bool {
    #[inline]
    fn duckdb_hash(self) -> u64 {
        // DuckDB stores BOOLEAN as `int8_t` and hashes it as such.
        i8::from(self).duckdb_hash()
    }
}

impl DuckDbHash for f32 {
    #[inline]
    fn duckdb_hash(self) -> u64 {
        murmur_hash64(u64::from(normalize_f32(self).to_bits()))
    }
}

impl DuckDbHash for f64 {
    #[inline]
    fn duckdb_hash(self) -> u64 {
        murmur_hash64(normalize_f64(self).to_bits())
    }
}

/// `duckdb::FloatingPointEqualityTransform`, which collapses negative zero and non-canonical NaNs
/// so that values comparing equal hash equally.
#[inline]
fn normalize_f32(value: f32) -> f32 {
    if value == 0.0 {
        0.0
    } else if value.is_nan() {
        f32::NAN
    } else {
        value
    }
}

#[inline]
fn normalize_f64(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else if value.is_nan() {
        f64::NAN
    } else {
        value
    }
}
