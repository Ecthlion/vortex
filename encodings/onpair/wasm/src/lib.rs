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
//! - **metadata**: prost `OnPairMetadata` `{1: uncompressed_lengths_ptype, 3: dict_size,
//!   4: codes_len, 5: dict_offsets_ptype, 6: codes_ptype, 7: codes_offsets_ptype}`;
//! - **buffers**: `[dict_bytes]` — the dictionary blob, read-padded by
//!   [`onpair::MAX_TOKEN_SIZE`] so the decoder's fixed-width token copy stays in bounds;
//! - **children**: `[dict_offsets (dict_size + 1), codes (codes_len), codes_offsets (len + 1),
//!   uncompressed_lengths (len), [validity]]`.
//!
//! Every child ptype is read from the metadata rather than assumed: the cascading compressor
//! narrows these integer children (`codes` to U8 for a small dictionary, `dict_offsets` to U16,
//! and so on), and the recorded ptype is what says how wide they actually are on disk. The kernel
//! widens them back to the `u32`/`u16` the decoder wants.
//!
//! # Untrusted input
//!
//! `onpair` was written for exactly this threat model. [`CompactDictionaryView::validate`]
//! checks the dictionary's offsets, token sizes, and read padding before a view exists at all,
//! and the decoders bounds-check every code — but they do so by *panicking*, which in a
//! `panic = "abort"` guest reaches the host as an opaque trap. So the kernel checks the codes
//! itself first, and a corrupt file becomes a clean kernel error instead.
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

use onpair::CompactDictionaryView;
use onpair::DictionaryView;
use onpair::Token;
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
    dict_size: u32,
    codes_len: u64,
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
            (3, Field::Varint(v)) => meta.dict_size = v as u32,
            (4, Field::Varint(v)) => meta.codes_len = v,
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

/// Widen a narrowed integer child to [`Token`], the width `onpair` wants for codes.
fn widen_tokens(values: &PrimitiveView, range: core::ops::Range<usize>) -> GuestResult<Vec<Token>> {
    range
        .map(|i| {
            Token::try_from(values.value_u64(i))
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
            meta.codes_len,
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

        // The dictionary is validated — offsets, token sizes, read padding — before a view of it
        // exists at all; this is what turns a corrupt dictionary into an error rather than a
        // panic the host can only report as a trap.
        let dict_offsets = widen_u32(&dict_offsets)?;
        let dict = CompactDictionaryView::validate(dict_bytes, &dict_offsets)
            .map_err(|_| GuestError::new("onpair dictionary is malformed"))?;

        // The decoders bounds-check codes too, but by panicking. Check first so a lying code
        // stream is a clean error as well.
        let codes = widen_tokens(&codes, code_start..code_end)?;
        let ntok = dict.num_tokens();
        guest_ensure!(
            codes.iter().all(|&code| usize::from(code) < ntok),
            "onpair code does not index the dictionary"
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
            total == onpair::decoded_len(&codes, dict),
            "onpair uncompressed lengths disagree with the codes stream"
        );

        let mut out: Vec<u8> = Vec::with_capacity(total);
        let written = onpair::try_decode_into(&codes, dict, out.spare_capacity_mut())
            .map_err(|_| GuestError::new("onpair output buffer too small"))?;
        guest_ensure!(written == total, "onpair decoded an unexpected byte count");
        // SAFETY: `try_decode_into` initialized exactly `written` bytes of the spare capacity
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
