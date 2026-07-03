// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An FSST (Fast Static Symbol Table) string decoder kernel — the first non-primitive encoding.
//!
//! FSST replaces frequent 1-8 byte substrings with 1-byte codes from a table of up to 255 symbols;
//! code 255 is an escape marking the next byte as a literal. Decompression is a simple table walk,
//! so a portable decoder is tiny even though the compressor (training, hash tables) is not — the
//! compressor stays on the write side (the `fsst` crate on the host) and only the decoder ships in
//! the file.
//!
//! The whole encoded form fits in the opaque payload, so this kernel has no child:
//!
//! ```text
//! [u32 n_symbols]
//! [n_symbols * 8]        symbol bytes (each symbol is a u64, little-endian)
//! [n_symbols]            symbol lengths (1..=8)
//! [u32 n_strings]
//! [(n_strings + 1) * 4]  u32 offsets of each compressed string in the codes region
//! [codes...]             concatenated compressed strings
//! ```
//!
//! Output: a utf8 string array as Arrow C Data Interface structs (offsets + data).

#![no_std]

extern crate alloc;

use alloc::vec::Vec;

use vortex_wasm_guest::GuestResult;
use vortex_wasm_guest::WasmEncoding;
use vortex_wasm_guest::arrow::Decoded;
use vortex_wasm_guest::arrow::DecodedUtf8;
use vortex_wasm_guest::export_wasm_encoding;
use vortex_wasm_guest::guest_ensure;

/// Code marking the next byte as a literal (fsst's `ESCAPE_CODE`).
const ESCAPE: u8 = 255;

struct Fsst;

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

impl WasmEncoding for Fsst {
    fn decode(input: &[u8]) -> GuestResult<Decoded> {
        guest_ensure!(input.len() >= 4, "fsst payload too short for symbol count");
        let n_symbols = read_u32(input, 0) as usize;
        guest_ensure!(n_symbols <= 255, "fsst symbol table too large");
        let table_end = 4 + n_symbols * 9;
        guest_ensure!(
            input.len() >= table_end + 4,
            "fsst payload too short for symbol table"
        );
        let symbols = &input[4..4 + n_symbols * 8];
        let lengths = &input[4 + n_symbols * 8..table_end];

        let n_strings = read_u32(input, table_end) as usize;
        let offsets_start = table_end + 4;
        let codes_start = offsets_start + (n_strings + 1) * 4;
        guest_ensure!(
            input.len() >= codes_start,
            "fsst payload too short for string offsets"
        );
        let code_offsets = &input[offsets_start..codes_start];
        let codes = &input[codes_start..];

        let mut values = Vec::new();
        let mut offsets = Vec::with_capacity(n_strings + 1);
        offsets.push(0i32);
        for i in 0..n_strings {
            let start = read_u32(code_offsets, i * 4) as usize;
            let end = read_u32(code_offsets, (i + 1) * 4) as usize;
            guest_ensure!(
                start <= end && end <= codes.len(),
                "fsst code offsets out of bounds"
            );

            let mut pos = start;
            while pos < end {
                let code = codes[pos];
                if code == ESCAPE {
                    guest_ensure!(pos + 1 < end, "fsst escape at end of compressed string");
                    values.push(codes[pos + 1]);
                    pos += 2;
                } else {
                    let code = code as usize;
                    guest_ensure!(code < n_symbols, "fsst code outside symbol table");
                    let len = lengths[code] as usize;
                    values.extend_from_slice(&symbols[code * 8..code * 8 + len]);
                    pos += 1;
                }
            }
            offsets.push(values.len() as i32);
        }

        Ok(Decoded::Utf8(DecodedUtf8 {
            len: n_strings,
            offsets,
            values,
            validity: None,
        }))
    }
}

export_wasm_encoding!(Fsst);
