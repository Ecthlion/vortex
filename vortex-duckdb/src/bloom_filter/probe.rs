// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DuckDB's bloom filter probe, reimplemented.
//!
//! DuckDB's filter (`src/planner/filter/bloom_filter.cpp`) is a flat, power-of-two array of
//! 64-bit sectors. A key's hash picks one sector with its low bits and four bit positions inside
//! that sector with four of its bytes; the key is present when all four bits are set.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_MM_HINT_T0;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_prefetch;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use vortex::error::VortexExpect;

/// The bits of the hash that select bit positions, masked to the 0..64 range a sector spans
/// (`duckdb::SHIFT_MASK`).
const SHIFT_MASK: u64 = 0x3F3F_3F3F_3F3F_3F3F;
/// How many bits a key sets in its sector (`duckdb::N_BITS`).
const BITS_PER_KEY: usize = 4;

/// A borrowed view over the sectors of a DuckDB bloom filter.
pub struct Sectors<'a> {
    sectors: &'a [AtomicU64],
}

impl<'a> Sectors<'a> {
    /// # Panics
    ///
    /// Panics if the sector count is not a power of two, which DuckDB guarantees by rounding it
    /// up when it sizes the filter.
    pub fn new(sectors: &'a [AtomicU64]) -> Self {
        assert!(
            sectors.len().is_power_of_two(),
            "DuckDB bloom filters have a power-of-two number of sectors, got {}",
            sectors.len()
        );
        Self { sectors }
    }

    /// Hints the CPU to start loading the sector this hash selects.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        let sector = self.sectors[self.index(hash)].as_ptr();
        // SAFETY: `_mm_prefetch` only hints the cache, and never dereferences the address.
        unsafe { _mm_prefetch(sector.cast::<i8>(), _MM_HINT_T0) };
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[inline]
    pub fn prefetch(&self, _hash: u64) {}

    #[inline]
    fn index(&self, hash: u64) -> usize {
        usize::try_from(hash & (self.sectors.len() as u64 - 1))
            .vortex_expect("a sector index is smaller than the sector count")
    }

    /// Whether the filter might contain a key with this hash.
    ///
    /// False positives are possible; false negatives are not.
    #[inline]
    pub fn contains(&self, hash: u64) -> bool {
        // The build side of the join sets bits with relaxed atomic ORs, so read them the same way.
        let sector = self.sectors[self.index(hash)].load(Ordering::Relaxed);
        let mask = sector_mask(hash);
        sector & mask == mask
    }
}

/// `duckdb::GetMask`: the top [`BITS_PER_KEY`] bytes of the masked hash, each naming a bit to set.
#[inline]
fn sector_mask(hash: u64) -> u64 {
    // DuckDB reads these bytes through a `uint8_t *`, which follows the host's byte order, so the
    // mask has to be derived from the same bytes it would pick on this machine.
    #[expect(
        clippy::host_endian_bytes,
        reason = "mirrors DuckDB reading the shifts through a byte pointer"
    )]
    let shifts = (hash & SHIFT_MASK).to_ne_bytes();

    let mut mask = 0u64;
    for &shift in &shifts[8 - BITS_PER_KEY..] {
        mask |= 1u64 << shift;
    }
    mask
}
