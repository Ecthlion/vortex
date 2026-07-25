// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The embeddable WASM decoder for `vortex.fastlanes.bitpacked`.
//!
//! This kernel is the portable mirror of the native `BitPacked` VTable's deserialize/decompress
//! path, consuming the encoding's **real serialized parts**:
//!
//! - **metadata**: the prost-encoded `BitPackedMetadata`
//!   (`{1: bit_width, 2: offset, 3: PatchesMetadata{1: len, 2: offset, 3: indices_ptype,
//!   4: chunk_offsets_len, 5: chunk_offsets_ptype, 6: offset_within_chunk}}`), parsed with the
//!   SDK's dependency-free proto reader;
//! - **buffers**: `[packed]` — 1024-element FastLanes chunks of `128 * bit_width` bytes, decoded
//!   with the same [`fastlanes`] crate the native encoding uses;
//! - **children**: `[patch indices, patch values, (chunk offsets),] [validity]` — declared via
//!   `vx_children` and decoded by the host.
//!
//! Semantics mirrored from `bitpack_decompress.rs`: unpack chunk by chunk (honouring the
//! encoding's `offset` into the first chunk and a partial final chunk), then overwrite patch
//! positions (`index - patches.offset`), carrying the validity child through to the output.
//!
//! Scope: 4-byte primitives (`i32`/`u32`/`f32`). Other widths are pure monomorphization of the
//! same fastlanes kernels, at ~25 KB of unrolled unpack code per width family.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use fastlanes::BitPacking;
use vortex_wasm_guest::GuestError;
use vortex_wasm_guest::GuestResult;
use vortex_wasm_guest::WasmEncoding;
use vortex_wasm_guest::abi::PType;
use vortex_wasm_guest::arrow::ChildView;
use vortex_wasm_guest::arrow::Decoded;
use vortex_wasm_guest::arrow::DecodedPrimitive;
use vortex_wasm_guest::export_wasm_encoding;
use vortex_wasm_guest::guest_ensure;
use vortex_wasm_guest::node::ChildDType;
use vortex_wasm_guest::node::ChildSpec;
use vortex_wasm_guest::node::NodeHeader;
use vortex_wasm_guest::node::NodeView;
use vortex_wasm_guest::node::ParentDType;
use vortex_wasm_guest::proto::Field;
use vortex_wasm_guest::proto::ProtoReader;

/// FastLanes chunk size in elements.
const CHUNK: usize = 1024;

/// Mirror of the native `PatchesMetadata` prost message.
#[derive(Default)]
struct PatchesMeta {
    len: u64,
    offset: u64,
    indices_ptype: Option<PType>,
    chunk_offsets_len: Option<u64>,
    chunk_offsets_ptype: Option<PType>,
}

/// Mirror of the native `BitPackedMetadata` prost message.
#[derive(Default)]
struct BitPackedMeta {
    bit_width: u32,
    offset: u32,
    patches: Option<PatchesMeta>,
}

fn parse_patches(bytes: &[u8]) -> GuestResult<PatchesMeta> {
    let mut meta = PatchesMeta::default();
    let mut reader = ProtoReader::new(bytes);
    while let Some((field, value)) = reader.next()? {
        match (field, value) {
            (1, Field::Varint(v)) => meta.len = v,
            (2, Field::Varint(v)) => meta.offset = v,
            (3, Field::Varint(v)) => {
                meta.indices_ptype =
                    Some(PType::from_discriminant(v).ok_or(GuestError::new("bad indices ptype"))?)
            }
            (4, Field::Varint(v)) => meta.chunk_offsets_len = Some(v),
            (5, Field::Varint(v)) => {
                meta.chunk_offsets_ptype = Some(
                    PType::from_discriminant(v)
                        .ok_or(GuestError::new("bad chunk offsets ptype"))?,
                )
            }
            _ => {}
        }
    }
    Ok(meta)
}

fn parse_metadata(bytes: &[u8]) -> GuestResult<BitPackedMeta> {
    let mut meta = BitPackedMeta::default();
    let mut reader = ProtoReader::new(bytes);
    while let Some((field, value)) = reader.next()? {
        match (field, value) {
            (1, Field::Varint(v)) => meta.bit_width = v as u32,
            (2, Field::Varint(v)) => meta.offset = v as u32,
            (3, Field::Bytes(b)) => meta.patches = Some(parse_patches(b)?),
            _ => {}
        }
    }
    Ok(meta)
}

struct BitPacked;

impl WasmEncoding for BitPacked {
    fn children(header: &NodeHeader<'_>) -> GuestResult<Vec<ChildSpec>> {
        let meta = parse_metadata(header.metadata)?;
        let mut specs = Vec::new();
        if let Some(patches) = &meta.patches {
            let indices_ptype = patches
                .indices_ptype
                .ok_or(GuestError::new("patches missing indices ptype"))?;
            specs.push(ChildSpec::values(
                ChildDType::Primitive(indices_ptype, false),
                patches.len,
            ));
            specs.push(ChildSpec::values(ChildDType::Parent, patches.len));
            if let (Some(co_len), Some(co_ptype)) =
                (patches.chunk_offsets_len, patches.chunk_offsets_ptype)
            {
                specs.push(ChildSpec::values(
                    ChildDType::Primitive(co_ptype, false),
                    co_len,
                ));
            }
        }
        // A trailing validity child is present iff the node has one more child than the patch
        // layout accounts for.
        if header.n_children == specs.len() + 1 {
            specs.push(ChildSpec::values(
                ChildDType::Bool(false),
                header.len as u64,
            ));
        }
        guest_ensure!(
            specs.len() == header.n_children,
            "bitpacked child count mismatch"
        );
        Ok(specs)
    }

    fn decode(node: &NodeView<'_>) -> GuestResult<Decoded> {
        let meta = parse_metadata(node.metadata)?;
        let ParentDType::Primitive(ptype) = node.parent else {
            return Err(GuestError::new("bitpacked expects a primitive dtype"));
        };
        guest_ensure!(
            ptype.byte_width() == 4,
            "this bitpacked kernel supports 4-byte primitives only"
        );
        let bit_width = meta.bit_width as usize;
        guest_ensure!(bit_width <= 32, "bitpacked bit width exceeds the ptype");
        let offset = meta.offset as usize;
        guest_ensure!(offset < CHUNK, "bitpacked offset must be within one chunk");

        guest_ensure!(node.nbuffers() == 1, "bitpacked expects one packed buffer");
        let packed = node.buffer(0)?;

        let total = offset + node.len;
        let num_chunks = total.div_ceil(CHUNK);
        let bytes_per_chunk = 128 * bit_width;
        let words_per_chunk = bytes_per_chunk / 4;
        guest_ensure!(
            packed.len() >= num_chunks * bytes_per_chunk,
            "bitpacked buffer too short for packed chunks"
        );

        // `vx_alloc` 8-aligns every host upload and wasm32 is little-endian (matching the
        // serialized format), so the packed buffer is viewed in place — no copy.
        // SAFETY: alignment is checked via the empty head; the length is checked above.
        let (head, words, _) = unsafe { packed.align_to::<u32>() };
        guest_ensure!(head.is_empty(), "packed buffer must be 4-byte aligned");

        // Unpack with the same fastlanes kernels the native encoding uses: full in-range chunks
        // decode directly into the output (mirroring the native `decode_into` fast path); only a
        // sliced first chunk and a partial trailer go through scratch.
        let mut out = alloc::vec![0u32; node.len];
        let mut scratch = [0u32; CHUNK];
        for chunk in 0..num_chunks {
            let start = chunk * CHUNK;
            let lo = offset.saturating_sub(start).min(CHUNK);
            let hi = (total - start).min(CHUNK);
            let chunk_words = &words[chunk * words_per_chunk..(chunk + 1) * words_per_chunk];

            let dst_start = start + lo - offset;
            if bit_width == 0 {
                out[dst_start..dst_start + (hi - lo)].fill(0);
            } else if lo == 0 && hi == CHUNK {
                // SAFETY: `chunk_words` holds exactly `128 * bit_width / 4` words and the
                // destination exactly 1024 elements, as `unchecked_unpack` requires.
                unsafe {
                    BitPacking::unchecked_unpack(
                        bit_width,
                        chunk_words,
                        &mut out[dst_start..dst_start + CHUNK],
                    )
                };
            } else {
                // SAFETY: as above, with a full-size scratch destination.
                unsafe { BitPacking::unchecked_unpack(bit_width, chunk_words, &mut scratch) };
                out[dst_start..dst_start + (hi - lo)].copy_from_slice(&scratch[lo..hi]);
            }
        }

        // Patches overwrite the unpacked output at `index - patches.offset`, exactly as the
        // native `apply_patches_to_uninit_range` does.
        let mut next_child = 0;
        if let Some(patches) = &meta.patches {
            let ChildView::Primitive(indices) = node.child(0)? else {
                return Err(GuestError::new("patch indices must be primitive"));
            };
            let ChildView::Primitive(patch_values) = node.child(1)? else {
                return Err(GuestError::new("patch values must be primitive"));
            };
            guest_ensure!(
                patch_values.ptype.byte_width() == 4,
                "patch values must match the parent width"
            );
            next_child = if patches.chunk_offsets_len.is_some() {
                3
            } else {
                2
            };

            for i in 0..indices.len {
                let position = indices.value_u64(i) - patches.offset;
                let position = usize::try_from(position)
                    .map_err(|_| GuestError::new("patch position overflow"))?;
                guest_ensure!(position < node.len, "patch position out of bounds");
                out[position] = u32::from_le_bytes(
                    patch_values.values[i * 4..i * 4 + 4]
                        .try_into()
                        .expect("4 bytes"),
                );
            }
        }

        // One cast, one copy: the u32 output reinterprets as little-endian bytes.
        // SAFETY: u8 has no alignment requirement and the byte length matches.
        let values =
            unsafe { core::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 4) }
                .to_vec();

        // A trailing validity child carries through to the output bitmap.
        let validity = if node.nchildren() == next_child + 1 {
            let ChildView::Bool(bits) = node.child(next_child)? else {
                return Err(GuestError::new("validity child must be boolean"));
            };
            Some(bits.bits[..node.len.div_ceil(8)].to_vec())
        } else {
            None
        };

        Ok(Decoded::Primitive(DecodedPrimitive {
            ptype,
            len: node.len,
            nullable: node.nullable,
            values,
            validity,
        }))
    }
}

export_wasm_encoding!(BitPacked);
