// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The embeddable WASM decoder for `vortex.runend`.
//!
//! Run-end is the canonical **structural** encoding: its output is not new data, it is the values
//! child repeated. So this kernel never touches the values child at all. It declares it
//! [`ChildMode::Reference`], expands the run ends into one gather index per output row, and
//! returns [`Decoded::Take`] — the host resolves the child in its own encoding and gathers it
//! lazily.
//!
//! That is what makes this kernel *dtype-agnostic*. The native decoder
//! (`run_end_canonicalize`, `encodings/runend/src/array.rs`) needs a separate implementation per
//! dtype — bool, primitive, varbinview — and `vortex_bail!`s on anything else. This kernel has
//! none of that: because the values child is only *named*, run-end over strings, decimals, or any
//! future dtype works with no code here, and none of it crosses the sandbox boundary.
//!
//! Serialized parts consumed:
//! - **metadata**: prost `RunEndMetadata` `{1: ends_ptype, 2: num_runs, 3: offset}`;
//! - **buffers**: none (run-end has `nbuffers() == 0`);
//! - **children**: `[ends (primitive, num_runs, Values), values (parent dtype, num_runs,
//!   Reference)]`.
//!
//! Index expansion mirrors `trimmed_ends_iter` (`encodings/runend/src/iter.rs`): each run end is
//! shifted by the array's `offset` and clamped to `len`, so a sliced run-end array decodes
//! correctly.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use vortex_wasm_guest::GuestError;
use vortex_wasm_guest::GuestResult;
use vortex_wasm_guest::WasmEncoding;
use vortex_wasm_guest::abi::PType;
use vortex_wasm_guest::arrow::ChildView;
use vortex_wasm_guest::arrow::Decoded;
use vortex_wasm_guest::arrow::DecodedPrimitive;
use vortex_wasm_guest::arrow::DecodedTake;
use vortex_wasm_guest::export_wasm_encoding;
use vortex_wasm_guest::guest_ensure;
use vortex_wasm_guest::node::ChildDType;
use vortex_wasm_guest::node::ChildSpec;
use vortex_wasm_guest::node::NodeHeader;
use vortex_wasm_guest::node::NodeView;
use vortex_wasm_guest::proto::Field;
use vortex_wasm_guest::proto::ProtoReader;

/// Serialized child slots.
const ENDS: u16 = 0;
const VALUES: u16 = 1;

/// Mirror of the native `RunEndMetadata` prost message.
#[derive(Default)]
struct RunEndMeta {
    ends_ptype: Option<PType>,
    num_runs: u64,
    offset: u64,
}

fn parse_metadata(bytes: &[u8]) -> GuestResult<RunEndMeta> {
    let mut meta = RunEndMeta::default();
    let mut reader = ProtoReader::new(bytes);
    while let Some((field, value)) = reader.next()? {
        match (field, value) {
            (1, Field::Varint(v)) => {
                meta.ends_ptype =
                    Some(PType::from_discriminant(v).ok_or(GuestError::new("bad ends ptype"))?)
            }
            (2, Field::Varint(v)) => meta.num_runs = v,
            (3, Field::Varint(v)) => meta.offset = v,
            _ => {}
        }
    }
    Ok(meta)
}

struct RunEnd;

impl WasmEncoding for RunEnd {
    fn children(header: &NodeHeader<'_>) -> GuestResult<Vec<ChildSpec>> {
        let meta = parse_metadata(header.metadata)?;
        // Prost omits zero-valued fields; discriminant 0 is U8.
        let ends_ptype = meta.ends_ptype.unwrap_or(PType::U8);
        guest_ensure!(
            matches!(ends_ptype, PType::U8 | PType::U16 | PType::U32 | PType::U64),
            "run-end ends must be an unsigned integer"
        );
        guest_ensure!(header.n_children == 2, "run-end expects exactly 2 children");

        let mut specs = Vec::with_capacity(2);
        // The ends are read here, to build the gather indices.
        specs.push(ChildSpec::values(
            ChildDType::Primitive(ends_ptype, false),
            meta.num_runs,
        ));
        // The values are only named: same dtype as the parent, whatever that is, and never copied
        // into guest memory.
        specs.push(ChildSpec::reference(ChildDType::Parent, meta.num_runs));
        Ok(specs)
    }

    fn decode(node: &NodeView<'_>) -> GuestResult<Decoded> {
        let meta = parse_metadata(node.metadata)?;
        let offset = meta.offset;

        let ChildView::Primitive(ends) = node.child(ENDS as usize)? else {
            return Err(GuestError::new("run-end ends child must be primitive"));
        };
        guest_ensure!(
            ends.len as u64 == meta.num_runs,
            "run-end ends length disagrees with num_runs"
        );

        // One run index per output row. `num_runs` bounds the index values, and the host
        // re-validates every index against the values child's length before gathering.
        guest_ensure!(
            meta.num_runs <= u32::MAX as u64,
            "run-end has too many runs for u32 indices"
        );
        let mut indices: Vec<u8> = Vec::with_capacity(node.len * 4);
        let mut filled: usize = 0;

        for run in 0..ends.len {
            if filled >= node.len {
                break;
            }
            // Mirror `trimmed_ends_iter`: shift by the slice offset, clamp to len.
            let raw = ends.value_u64(run);
            guest_ensure!(raw >= offset, "run end precedes the array offset");
            let end = (raw - offset).min(node.len as u64) as usize;
            guest_ensure!(end >= filled, "run ends must be non-decreasing");

            let index = (run as u32).to_le_bytes();
            for _ in filled..end {
                indices.extend_from_slice(&index);
            }
            filled = end;
        }
        guest_ensure!(
            filled == node.len,
            "run ends do not cover the array's length"
        );

        Ok(Decoded::Take(DecodedTake {
            values_slot: VALUES,
            indices: DecodedPrimitive {
                ptype: PType::U32,
                len: node.len,
                nullable: false,
                values: indices,
                validity: None,
            },
        }))
    }
}

export_wasm_encoding!(RunEnd);
