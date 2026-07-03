// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A decoder kernel for the `vortex.fastlanes.bitpacked` encoding, for `i32`.
//!
//! This kernel matches the native encoding's semantics exactly: it links the same [`fastlanes`]
//! crate the native `BitPacked` VTable uses, so the packed layout — 1024-element FastLanes chunks
//! of `128 * bit_width` bytes in the transposed lane order — is decoded bit-for-bit identically.
//! It also honours the encoding's `offset` (a slice into the first chunk), a partial final chunk
//! (unpacked into scratch and truncated), and **patches**: values wider than `bit_width` that the
//! encoder stored separately and that overwrite the unpacked output.
//!
//! Payload layout (values pack as the unsigned physical type, `u32` for `i32`):
//!
//! ```text
//! [u8 bit_width][u8 pad][u16 offset][u32 len][u32 n_patches]
//! [n_patches * u32]  patch positions (already normalized by the patches' offset)
//! [n_patches * 4]    patch values (i32 LE)
//! [packed bytes...]  ceil((offset + len) / 1024) chunks of 128 * bit_width bytes
//! ```

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use fastlanes::BitPacking;
use vortex_wasm_guest::GuestResult;
use vortex_wasm_guest::WasmEncoding;
use vortex_wasm_guest::abi::PType;
use vortex_wasm_guest::arrow::Decoded;
use vortex_wasm_guest::arrow::DecodedPrimitive;
use vortex_wasm_guest::export_wasm_encoding;
use vortex_wasm_guest::guest_ensure;

/// FastLanes chunk size in elements.
const CHUNK: usize = 1024;

struct BitPacked;

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

impl WasmEncoding for BitPacked {
    fn decode(input: &[u8]) -> GuestResult<Decoded> {
        guest_ensure!(input.len() >= 12, "bitpacked header must be 12 bytes");
        let bit_width = input[0] as usize;
        guest_ensure!(bit_width <= 32, "bitpacked bit width exceeds u32");
        let offset = u16::from_le_bytes([input[2], input[3]]) as usize;
        guest_ensure!(offset < CHUNK, "bitpacked offset must be within one chunk");
        let len = read_u32(input, 4) as usize;
        let n_patches = read_u32(input, 8) as usize;

        let patches_end = 12 + n_patches * 8;
        guest_ensure!(
            input.len() >= patches_end,
            "bitpacked payload too short for patches"
        );
        let patch_positions = &input[12..12 + n_patches * 4];
        let patch_values = &input[12 + n_patches * 4..patches_end];
        let packed = &input[patches_end..];

        let total = offset + len;
        let num_chunks = total.div_ceil(CHUNK);
        let bytes_per_chunk = 128 * bit_width;
        let words_per_chunk = bytes_per_chunk / 4;
        guest_ensure!(
            packed.len() >= num_chunks * bytes_per_chunk,
            "bitpacked payload too short for packed chunks"
        );

        // Unpack chunk by chunk with the same fastlanes kernels the native encoding uses. The
        // packed words are copied out per chunk because the payload slice is neither
        // alignment-guaranteed nor endianness-agnostic.
        let mut values = Vec::with_capacity(len * 4);
        let mut words = Vec::with_capacity(words_per_chunk);
        let mut scratch = [0u32; CHUNK];
        for chunk in 0..num_chunks {
            words.clear();
            let base = chunk * bytes_per_chunk;
            for word in 0..words_per_chunk {
                words.push(read_u32(packed, base + word * 4));
            }
            if bit_width == 0 {
                scratch.fill(0);
            } else {
                // SAFETY: `words` holds exactly `128 * bit_width / 4` words and `scratch` exactly
                // 1024 elements, as `unchecked_unpack` requires.
                unsafe { BitPacking::unchecked_unpack(bit_width, &words, &mut scratch) };
            }

            let start = chunk * CHUNK;
            let lo = offset.saturating_sub(start).min(CHUNK);
            let hi = (total - start).min(CHUNK);
            for value in &scratch[lo..hi] {
                values.extend_from_slice(&value.to_le_bytes());
            }
        }

        // Patches overwrite the unpacked output at their (normalized) positions.
        for patch in 0..n_patches {
            let position = read_u32(patch_positions, patch * 4) as usize;
            guest_ensure!(position < len, "bitpacked patch position out of bounds");
            let value = &patch_values[patch * 4..patch * 4 + 4];
            values[position * 4..position * 4 + 4].copy_from_slice(value);
        }

        Ok(Decoded::Primitive(DecodedPrimitive {
            ptype: PType::I32,
            len,
            values,
            validity: None,
        }))
    }
}

export_wasm_encoding!(BitPacked);
