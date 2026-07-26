// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The embeddable WASM decoder for `vortex.fsst`.
//!
//! This kernel is the portable mirror of the native `FSST` VTable's deserialize/canonicalize
//! path, consuming the encoding's **real serialized parts** (the current 3-buffer format):
//!
//! - **metadata**: the prost-encoded `FSSTMetadata`
//!   (`{1: uncompressed_lengths_ptype, 2: codes_offsets_ptype}`);
//! - **buffers**: `[symbols (u64 each), symbol_lengths, codes_bytes]`;
//! - **children**: `[uncompressed_lengths (len), codes_offsets (len + 1), [validity]]`.
//!
//! Like the native canonical path, it bulk-decompresses the whole codes heap with the same
//! [`fsst`] crate ([`fsst::Decompressor`]) the native encoding uses, then splits strings by the
//! prefix sums of `uncompressed_lengths` — which are exactly the Arrow utf8 offsets.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use fsst::Decompressor;
use fsst::Symbol;
use vortex_wasm_guest::GuestError;
use vortex_wasm_guest::GuestResult;
use vortex_wasm_guest::WasmEncoding;
use vortex_wasm_guest::abi::PType;
use vortex_wasm_guest::data::ChildView;
use vortex_wasm_guest::data::Decoded;
use vortex_wasm_guest::data::DecodedVarBinView;
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

/// Mirror of the native `FSSTMetadata` prost message.
struct FsstMeta {
    uncompressed_lengths_ptype: PType,
    codes_offsets_ptype: PType,
}

fn parse_metadata(bytes: &[u8]) -> GuestResult<FsstMeta> {
    let mut lengths_ptype = None;
    let mut offsets_ptype = None;
    let mut reader = ProtoReader::new(bytes);
    while let Some((field, value)) = reader.next()? {
        match (field, value) {
            (1, Field::Varint(v)) => {
                lengths_ptype =
                    Some(PType::from_discriminant(v).ok_or(GuestError::new("bad lengths ptype"))?)
            }
            (2, Field::Varint(v)) => {
                offsets_ptype =
                    Some(PType::from_discriminant(v).ok_or(GuestError::new("bad offsets ptype"))?)
            }
            _ => {}
        }
    }
    Ok(FsstMeta {
        // Prost omits zero-valued fields; discriminant 0 is U8.
        uncompressed_lengths_ptype: lengths_ptype.unwrap_or(PType::U8),
        codes_offsets_ptype: offsets_ptype.unwrap_or(PType::U8),
    })
}

struct Fsst;

impl WasmEncoding for Fsst {
    fn children(header: &NodeHeader<'_>) -> GuestResult<Vec<ChildSpec>> {
        let meta = parse_metadata(header.metadata)?;
        let mut specs = Vec::with_capacity(3);
        specs.push(ChildSpec::values(
            DTypeExpr::primitive(meta.uncompressed_lengths_ptype, false),
            header.len as u64,
        ));
        // VarBin offsets are len + 1.
        specs.push(ChildSpec::values(
            DTypeExpr::primitive(meta.codes_offsets_ptype, false),
            header.len as u64 + 1,
        ));
        if header.n_children == 3 {
            specs.push(ChildSpec::values(DTypeExpr::bool(false), header.len as u64));
        }
        guest_ensure!(
            specs.len() == header.n_children,
            "fsst child count mismatch"
        );
        Ok(specs)
    }

    fn decode(node: &NodeView<'_>, plan: &mut PlanBuilder) -> GuestResult<NodeId> {
        guest_ensure!(
            node.nbuffers() == 3,
            "fsst expects [symbols, symbol_lengths, codes] buffers"
        );
        let symbol_bytes = node.buffer(0)?;
        let symbol_lengths = node.buffer(1)?;
        let codes = node.buffer(2)?;
        let n_symbols = symbol_lengths.len();
        guest_ensure!(
            symbol_bytes.len() == n_symbols * 8,
            "fsst symbol buffers disagree"
        );

        // Rebuild the symbol table with the same types the native decoder uses.
        let symbols: Vec<Symbol> = (0..n_symbols)
            .map(|i| {
                Symbol::from_slice(
                    symbol_bytes[i * 8..(i + 1) * 8]
                        .try_into()
                        .expect("8 bytes per symbol"),
                )
            })
            .collect();
        let decompressor = Decompressor::new(&symbols, symbol_lengths);

        // The codes heap is bounded by the final codes offset.
        let ChildView::Primitive(codes_offsets) = node.child(1)? else {
            return Err(GuestError::new("fsst codes offsets must be primitive"));
        };
        guest_ensure!(
            codes_offsets.len == node.len + 1,
            "fsst codes offsets must have len + 1 entries"
        );
        let codes_end = usize::try_from(codes_offsets.value_u64(node.len))
            .map_err(|_| GuestError::new("fsst codes end overflow"))?;
        guest_ensure!(codes_end <= codes.len(), "fsst codes end out of bounds");

        // Bulk-decompress the whole heap (mirroring the native canonical path); the uncompressed
        // lengths' prefix sums are the output utf8 offsets.
        let values = decompressor.decompress(&codes[..codes_end]);

        let ChildView::Primitive(lengths) = node.child(0)? else {
            return Err(GuestError::new(
                "fsst uncompressed lengths must be primitive",
            ));
        };
        guest_ensure!(
            lengths.len == node.len,
            "fsst uncompressed lengths must have len entries"
        );

        let validity = if node.nchildren() == 3 {
            let ChildView::Bool(bits) = node.child(2)? else {
                return Err(GuestError::new("fsst validity child must be boolean"));
            };
            Validity::Bitmap(bits.bits[..node.len.div_ceil(8)].to_vec())
        } else if node.nullable {
            Validity::AllValid
        } else {
            Validity::NonNullable
        };

        // Emit Vortex's canonical view layout directly. The previous Arrow-shaped output used i32
        // offsets, which the host imported as `VarBin` — not canonical — costing a second full
        // conversion of the whole heap on every string decode.
        // The parent dtype carries through: FSST compresses both Utf8 and Binary, and the view
        // layout alone does not distinguish them.
        Ok(plan.materialized(
            DTypeExpr::parent(),
            Decoded::VarBinView(DecodedVarBinView::from_heap(
                values,
                (0..node.len).map(|i| lengths.value_u64(i) as usize),
                validity,
            )?),
        ))
    }
}

export_wasm_encoding!(Fsst);
