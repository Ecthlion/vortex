// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The embeddable WASM decoder for `vortex.runend`.
//!
//! Run-end is the canonical **structural** encoding: its output is not new data, it is the values
//! child repeated. So this kernel computes nothing about the values themselves. It expands the run
//! ends into one gather index per output row and hands them to [`Decoded::take`], the SDK's
//! generic gather over a canonical child of *any* dtype.
//!
//! That is what makes this kernel *dtype-agnostic*. The native decoder
//! (`run_end_canonicalize`, `encodings/runend/src/array.rs`) needs a separate implementation per
//! dtype — bool, primitive, varbinview — and `vortex_bail!`s on anything else. This kernel has
//! none of that: the values child arrives in Vortex's canonical layout for whatever its dtype is
//! (a struct of lists of strings, a timestamp extension, a union), and `take` follows that layout.
//!
//! Serialized parts consumed:
//! - **metadata**: prost `RunEndMetadata` `{1: ends_ptype, 2: num_runs, 3: offset}`;
//! - **buffers**: none (run-end has `nbuffers() == 0`);
//! - **children**: `[ends (primitive, num_runs), values (parent dtype, num_runs)]`.
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
use vortex_wasm_guest::data::Decoded;
use vortex_wasm_guest::dtype::DTypeExpr;
use vortex_wasm_guest::export_wasm_encoding;
use vortex_wasm_guest::guest_ensure;
use vortex_wasm_guest::node::ChildSpec;
use vortex_wasm_guest::node::NodeHeader;
use vortex_wasm_guest::node::NodeView;
use vortex_wasm_guest::proto::Field;
use vortex_wasm_guest::proto::ProtoReader;

/// Serialized child slots.
const ENDS: usize = 0;
const VALUES: usize = 1;

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

        Ok(alloc::vec![
            ChildSpec::new(DTypeExpr::primitive(ends_ptype, false), meta.num_runs),
            // Same dtype as the parent, whatever that is: the kernel never needs to know.
            ChildSpec::new(DTypeExpr::parent(), meta.num_runs),
        ])
    }

    fn decode(node: &NodeView<'_>) -> GuestResult<Decoded> {
        let meta = parse_metadata(node.metadata)?;
        let offset = meta.offset;

        let ends = node.child(ENDS)?.as_primitive()?;
        guest_ensure!(
            ends.len as u64 == meta.num_runs,
            "run-end ends length disagrees with num_runs"
        );
        guest_ensure!(
            meta.num_runs <= u32::MAX as u64,
            "run-end has too many runs for u32 indices"
        );

        // One run index per output row, mirroring `trimmed_ends_iter`: shift each end by the slice
        // offset and clamp it to len.
        let mut indices: Vec<u32> = Vec::with_capacity(node.len);
        for run in 0..ends.len {
            if indices.len() >= node.len {
                break;
            }
            let raw = ends.value_u64(run);
            guest_ensure!(raw >= offset, "run end precedes the array offset");
            let end = (raw - offset).min(node.len as u64) as usize;
            guest_ensure!(end >= indices.len(), "run ends must be non-decreasing");
            indices.resize(end, run as u32);
        }
        guest_ensure!(
            indices.len() == node.len,
            "run ends do not cover the array's length"
        );

        // `take` bounds-checks every index against the values child and follows its canonical
        // layout, whatever the dtype.
        Decoded::take(&node.child(VALUES)?, &indices)
    }
}

export_wasm_encoding!(RunEnd);
