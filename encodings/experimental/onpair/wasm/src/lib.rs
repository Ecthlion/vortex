// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The embeddable WASM decoder for `vortex.onpair`.
//!
//! OnPair is FSST-shaped: a trained dictionary in a buffer, a stream of fixed-width codes
//! indexing it, and per-row boundaries. So this is a **value-producing** kernel like
//! `vortex.fsst` — the decompressed bytes exist nowhere in the file, so there is nothing to
//! delegate and the kernel returns a single `Materialized` node.
//!
//! Serialized parts consumed:
//! - **metadata**: prost `OnPairMetadata` `{1: uncompressed_lengths_ptype, 2: bits,
//!   3: dict_size, 4: total_tokens, 5: dict_offsets_ptype, 6: codes_ptype,
//!   7: codes_offsets_ptype}`;
//! - **buffers**: `[dict_bytes]` — the dictionary blob, over-padded by
//!   [`onpair::MAX_TOKEN_SIZE`] so the decoder's fixed-width token read stays in bounds;
//! - **children**: `[dict_offsets (dict_size + 1), codes (total_tokens),
//!   codes_offsets (len + 1), uncompressed_lengths (len), [validity]]`.
//!
//! Every child ptype is read from the metadata rather than assumed: the cascading compressor
//! narrows these integer children (`codes` to U8 when `bits <= 8`, `dict_offsets` to U16, and so
//! on), and the recorded ptype is what says how wide they actually are on disk. The kernel widens
//! them back to the `u32`/`u16` the decoder wants.
//!
//! # Untrusted input
//!
//! `onpair::Parts` is documented as built by struct literal from deserialized storage, so its
//! arrays may be corrupt — and [`onpair::Parts::validate`] exists for exactly that. Calling it
//! once up front turns a malformed dictionary or an out-of-range code into a clean kernel error
//! instead of a panic, which in a `panic = "abort"` guest would reach the host as an opaque trap.
//! The decoder was written for this threat model, so the kernel adds almost nothing to it.
//!
//! # A sliced array reads more than it needs
//!
//! `codes_offsets` bounds the run of `codes` belonging to the rows actually present, and the
//! native canonical path point-looks-up those two boundaries and slices `codes` before
//! materializing it. This kernel slices the same window, but only *after* the host has already
//! decoded and copied the whole `codes` child into guest memory: `ChildSpec` carries a dtype, a
//! length, and an access mode, and cannot ask for a row range. The bound it would need lives
//! inside another child, which `vx_children` — a single pure call made before any child is
//! decoded — cannot read. This is the same shape of gap that makes `vortex.chunked`
//! inexpressible, in a milder form: onpair still decodes correctly, it just over-reads for a
//! sliced array.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use onpair::MAX_TOKEN_SIZE;
use onpair::Parts;
use vortex_wasm_guest::GuestError;
use vortex_wasm_guest::GuestResult;
use vortex_wasm_guest::WasmEncoding;
use vortex_wasm_guest::abi::PType;
use vortex_wasm_guest::data::ChildView;
use vortex_wasm_guest::data::Decoded;
use vortex_wasm_guest::data::DecodedVarBinView;
use vortex_wasm_guest::data::PrimitiveView;
use vortex_wasm_guest::data::Validity;
use vortex_wasm_guest::dtype::DTypeExpr;
use vortex_wasm_guest::export_wasm_encoding;
use vortex_wasm_guest::guest_ensure;
use vortex_wasm_guest::node::ChildSpec;
use vortex_wasm_guest::node::NodeHeader;
use vortex_wasm_guest::node::NodeView;
use vortex_wasm_guest::plan::NodeId;
use vortex_wasm_guest::plan::PlanBuilder;
use vortex_wasm_guest::proto::Field;
use vortex_wasm_guest::proto::ProtoReader;

/// Serialized child slots.
const DICT_OFFSETS: usize = 0;
const CODES: usize = 1;
const CODES_OFFSETS: usize = 2;
const UNCOMPRESSED_LENGTHS: usize = 3;
const VALIDITY: usize = 4;

/// Mirror of the native `OnPairMetadata` prost message.
#[derive(Default)]
struct OnPairMeta {
    uncompressed_lengths_ptype: Option<PType>,
    bits: u32,
    dict_size: u32,
    total_tokens: u64,
    dict_offsets_ptype: Option<PType>,
    codes_ptype: Option<PType>,
    codes_offsets_ptype: Option<PType>,
}

impl OnPairMeta {
    /// Prost omits zero-valued fields, and `PType` discriminant 0 is `U8`.
    fn ptype(field: Option<PType>) -> PType {
        field.unwrap_or(PType::U8)
    }
}

fn parse_metadata(bytes: &[u8]) -> GuestResult<OnPairMeta> {
    let mut meta = OnPairMeta::default();
    let mut reader = ProtoReader::new(bytes);
    while let Some((field, value)) = reader.next()? {
        match (field, value) {
            (1, Field::Varint(v)) => {
                meta.uncompressed_lengths_ptype =
                    Some(PType::from_discriminant(v).ok_or(GuestError::new("bad lengths ptype"))?)
            }
            (2, Field::Varint(v)) => meta.bits = v as u32,
            (3, Field::Varint(v)) => meta.dict_size = v as u32,
            (4, Field::Varint(v)) => meta.total_tokens = v,
            (5, Field::Varint(v)) => {
                meta.dict_offsets_ptype = Some(
                    PType::from_discriminant(v).ok_or(GuestError::new("bad dict offsets ptype"))?,
                )
            }
            (6, Field::Varint(v)) => {
                meta.codes_ptype =
                    Some(PType::from_discriminant(v).ok_or(GuestError::new("bad codes ptype"))?)
            }
            (7, Field::Varint(v)) => {
                meta.codes_offsets_ptype = Some(
                    PType::from_discriminant(v)
                        .ok_or(GuestError::new("bad codes offsets ptype"))?,
                )
            }
            _ => {}
        }
    }
    Ok(meta)
}

/// Widen a narrowed integer child to `u32`, the width `onpair` wants for dictionary offsets.
fn widen_u32(values: &PrimitiveView) -> GuestResult<Vec<u32>> {
    (0..values.len)
        .map(|i| {
            u32::try_from(values.value_u64(i))
                .map_err(|_| GuestError::new("onpair dict offset exceeds u32"))
        })
        .collect()
}

/// Widen a narrowed integer child to `u16`, the width `onpair` wants for codes.
fn widen_u16(values: &PrimitiveView, range: core::ops::Range<usize>) -> GuestResult<Vec<u16>> {
    range
        .map(|i| {
            u16::try_from(values.value_u64(i))
                .map_err(|_| GuestError::new("onpair code exceeds u16"))
        })
        .collect()
}

fn primitive(node: &NodeView<'_>, slot: usize, what: &'static str) -> GuestResult<PrimitiveView> {
    match node.child(slot)? {
        ChildView::Primitive(view) => Ok(view),
        ChildView::Bool(_) => Err(GuestError::new(what)),
    }
}

struct OnPair;

impl WasmEncoding for OnPair {
    fn children(header: &NodeHeader<'_>) -> GuestResult<Vec<ChildSpec>> {
        let meta = parse_metadata(header.metadata)?;
        guest_ensure!(
            (9..=16).contains(&meta.bits),
            "onpair bits must be in 9..=16"
        );
        guest_ensure!(
            u64::from(meta.dict_size) <= 1u64 << meta.bits,
            "onpair dict_size exceeds 2^bits"
        );
        guest_ensure!(
            header.n_children == 4 || header.n_children == 5,
            "onpair expects 4 or 5 children"
        );

        let mut specs = Vec::with_capacity(header.n_children);
        specs.push(ChildSpec::values(
            DTypeExpr::primitive(OnPairMeta::ptype(meta.dict_offsets_ptype), false),
            u64::from(meta.dict_size) + 1,
        ));
        specs.push(ChildSpec::values(
            DTypeExpr::primitive(OnPairMeta::ptype(meta.codes_ptype), false),
            meta.total_tokens,
        ));
        // Row boundaries into `codes`, so len + 1 like any offsets child.
        specs.push(ChildSpec::values(
            DTypeExpr::primitive(OnPairMeta::ptype(meta.codes_offsets_ptype), false),
            header.len as u64 + 1,
        ));
        specs.push(ChildSpec::values(
            DTypeExpr::primitive(OnPairMeta::ptype(meta.uncompressed_lengths_ptype), false),
            header.len as u64,
        ));
        if header.n_children == 5 {
            specs.push(ChildSpec::values(DTypeExpr::bool(false), header.len as u64));
        }
        Ok(specs)
    }

    fn decode(node: &NodeView<'_>, plan: &mut PlanBuilder) -> GuestResult<NodeId> {
        let meta = parse_metadata(node.metadata)?;

        guest_ensure!(node.nbuffers() == 1, "onpair expects one dictionary buffer");
        let dict_bytes = node.buffer(0)?;

        let dict_offsets = primitive(node, DICT_OFFSETS, "onpair dict offsets must be primitive")?;
        let codes = primitive(node, CODES, "onpair codes must be primitive")?;
        let codes_offsets = primitive(
            node,
            CODES_OFFSETS,
            "onpair codes offsets must be primitive",
        )?;
        let lengths = primitive(
            node,
            UNCOMPRESSED_LENGTHS,
            "onpair uncompressed lengths must be primitive",
        )?;

        guest_ensure!(
            codes_offsets.len == node.len + 1,
            "onpair codes offsets must have len + 1 entries"
        );
        guest_ensure!(
            lengths.len == node.len,
            "onpair uncompressed lengths must have len entries"
        );

        // The rows present here own the contiguous window `codes_offsets[0]..codes_offsets[len]`.
        // A sliced array narrows only `codes_offsets`, keeping the whole `codes` child, so this
        // window is what the decoder must walk — mirroring the native canonical path.
        let code_start = usize::try_from(codes_offsets.value_u64(0))
            .map_err(|_| GuestError::new("onpair code start overflow"))?;
        let code_end = usize::try_from(codes_offsets.value_u64(node.len))
            .map_err(|_| GuestError::new("onpair code end overflow"))?;
        guest_ensure!(
            code_start <= code_end,
            "onpair codes offsets must be nondecreasing"
        );
        guest_ensure!(
            code_end <= codes.len,
            "onpair codes offsets end exceeds the codes child"
        );

        let dict_offsets = widen_u32(&dict_offsets)?;
        let codes = widen_u16(&codes, code_start..code_end)?;
        let parts = Parts {
            dict_bytes,
            dict_offsets: &dict_offsets,
            bits: meta.bits,
            codes: &codes,
        };

        // `Parts` is built by struct literal from file bytes, so validate before decoding: this is
        // what turns a corrupt dictionary or an out-of-range code into an error rather than a
        // panic the host can only report as a trap. Also confirms the buffer carries the
        // decoder's trailing padding.
        parts.validate().map_err(|_| {
            GuestError::new("onpair parts are not decodable: bad dictionary or out-of-range code")
        })?;
        guest_ensure!(
            dict_bytes.len() >= MAX_TOKEN_SIZE || dict_offsets.len() <= 1,
            "onpair dictionary buffer is missing decoder padding"
        );

        // The per-row lengths both size the output and split it, so they must agree with what the
        // decoder will actually write. Disagreement means the file is lying about one of them.
        let mut total = 0usize;
        for i in 0..lengths.len {
            let length = usize::try_from(lengths.value_u64(i))
                .map_err(|_| GuestError::new("onpair uncompressed length overflow"))?;
            total = total
                .checked_add(length)
                .ok_or(GuestError::new("onpair uncompressed lengths overflow"))?;
        }
        guest_ensure!(
            total == onpair::decompressed_len(parts),
            "onpair uncompressed lengths disagree with the codes stream"
        );

        let mut out: Vec<u8> = Vec::with_capacity(total);
        let written = onpair::decompress_into(parts, out.spare_capacity_mut());
        guest_ensure!(written == total, "onpair decoded an unexpected byte count");
        // SAFETY: `decompress_into` initialized exactly `written` bytes of the spare capacity
        // reserved above, and `written == total <= capacity`.
        unsafe { out.set_len(written) };

        let validity = if node.nchildren() == 5 {
            let ChildView::Bool(bits) = node.child(VALIDITY)? else {
                return Err(GuestError::new("onpair validity child must be boolean"));
            };
            Validity::Bitmap(bits.bits[..node.len.div_ceil(8)].to_vec())
        } else if node.nullable {
            Validity::AllValid
        } else {
            Validity::NonNullable
        };

        // OnPair compresses strings, but the same view layout serves Utf8 and Binary, so the
        // output type comes from the parent rather than from the layout.
        Ok(plan.materialized(
            DTypeExpr::parent(),
            Decoded::VarBinView(DecodedVarBinView::from_heap(
                out,
                (0..lengths.len).map(|i| lengths.value_u64(i) as usize),
                validity,
            )?),
        ))
    }
}

export_wasm_encoding!(OnPair);

/// `getrandom`'s custom backend. `onpair` links `rand` for dictionary training; a decoder never
/// draws randomness, so this exists only to satisfy the linker and always fails.
#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(
    _dest: *mut u8,
    _len: usize,
) -> Result<(), getrandom::Error> {
    Err(getrandom::Error::UNSUPPORTED)
}
