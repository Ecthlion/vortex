// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Building and reading Arrow C Data Interface structs in the guest's own linear memory.
//!
//! Decoded arrays cross the boundary as Arrow C structs. A kernel returns its result as a
//! [`Decoded`] and the SDK lays out the structs ([`write`]); host-decoded child arrays arrive as
//! structs the SDK reads back ([`read_child`]). The layouts are plain bytes (see [`crate::abi`]),
//! so no Arrow library is needed.

use alloc::vec::Vec;

use crate::abi::ARRAY_SIZE;
use crate::abi::ARROW_FLAG_NULLABLE;
use crate::abi::PType;
use crate::abi::SCHEMA_SIZE;
use crate::abi::array;
use crate::abi::decode_result;
use crate::abi::schema;
use crate::error::GuestError;
use crate::error::GuestResult;
use crate::host::alloc_bytes;

/// What a kernel produces for a node.
///
/// Two shapes, because Vortex encodings come in two kinds. Encodings that compute *new element
/// values* (bit-packing, FSST, zstd, delta) must materialize bytes. Encodings that merely
/// **re-arrange** an existing child (run-end, dict) must not: materializing them would force the
/// guest to reproduce a dtype it may not even be able to represent, and would copy the child
/// through the sandbox twice for nothing. Those return [`Decoded::Take`] instead and let the host
/// gather.
pub enum Decoded {
    /// A materialized primitive array.
    Primitive(DecodedPrimitive),
    /// A materialized utf8 string array.
    Utf8(DecodedUtf8),
    /// The output is a referenced child gathered by guest-computed indices.
    Take(DecodedTake),
}

/// The output is child `values_slot`, gathered by `indices`.
///
/// The gathered child is declared [`ChildMode::Reference`](crate::node::ChildMode::Reference), so
/// it never enters guest memory: its dtype may be anything, including nested types, and the host
/// performs the gather lazily.
pub struct DecodedTake {
    /// The serialized child slot to gather from.
    pub values_slot: u16,
    /// One unsigned index per output element, each `< values.len()`.
    pub indices: DecodedPrimitive,
}

/// A decoded primitive array. The values buffer must hold an entry at every position; null
/// positions may contain any bytes. `validity` is an LSB-first bitmap (`ceil(len / 8)` bytes,
/// 1 = valid); it may be `None` even when `nullable` (meaning all-valid).
pub struct DecodedPrimitive {
    /// Element type.
    pub ptype: PType,
    /// Logical element count.
    pub len: usize,
    /// Whether the output dtype is nullable.
    pub nullable: bool,
    /// Little-endian values, `len * ptype.byte_width()` bytes.
    pub values: Vec<u8>,
    /// Optional validity bitmap.
    pub validity: Option<Vec<u8>>,
}

/// A decoded utf8 string array in Arrow's variable-size layout: `len + 1` byte offsets into a
/// concatenated values buffer, so string `i` is `values[offsets[i]..offsets[i + 1]]`.
pub struct DecodedUtf8 {
    /// Logical element count.
    pub len: usize,
    /// Whether the output dtype is nullable.
    pub nullable: bool,
    /// `len + 1` monotonically non-decreasing byte offsets into `values`.
    pub offsets: Vec<i32>,
    /// Concatenated utf8 bytes.
    pub values: Vec<u8>,
    /// Optional validity bitmap (LSB-first, 1 = valid).
    pub validity: Option<Vec<u8>>,
}

/// Write a [`Decoded`] as a `vx_decode` result frame in linear memory (see
/// [`decode_result`](crate::abi::decode_result)), returning its offset.
pub fn write(decoded: &Decoded) -> i32 {
    match decoded {
        Decoded::Take(take) => {
            let pair = write_primitive_structs(&take.indices);
            let mut frame = Vec::with_capacity(16);
            frame.extend_from_slice(&decode_result::TAG_TAKE.to_le_bytes());
            frame.extend_from_slice(&u32::from(take.values_slot).to_le_bytes());
            frame.extend_from_slice(&pair.0.to_le_bytes());
            frame.extend_from_slice(&pair.1.to_le_bytes());
            alloc_bytes(&frame) as i32
        }
        other => {
            let (array_ptr, schema_ptr) = write_materialized(other);
            let mut frame = Vec::with_capacity(12);
            frame.extend_from_slice(&decode_result::TAG_MATERIALIZED.to_le_bytes());
            frame.extend_from_slice(&array_ptr.to_le_bytes());
            frame.extend_from_slice(&schema_ptr.to_le_bytes());
            alloc_bytes(&frame) as i32
        }
    }
}

/// Lay out the Arrow C structs for a materialized primitive, returning `(array_ptr, schema_ptr)`.
fn write_primitive_structs(primitive: &DecodedPrimitive) -> (u32, u32) {
    let values_ptr = alloc_bytes(&primitive.values);
    let validity_ptr = primitive
        .validity
        .as_ref()
        .map(|v| alloc_bytes(v))
        .unwrap_or(0);
    write_structs(
        primitive.ptype.format_code(),
        primitive.len,
        primitive.nullable,
        &[validity_ptr, values_ptr],
    )
}

/// Lay out the Arrow C structs for a materialized output, returning `(array_ptr, schema_ptr)`.
fn write_materialized(decoded: &Decoded) -> (u32, u32) {
    match decoded {
        Decoded::Primitive(primitive) => write_primitive_structs(primitive),
        Decoded::Utf8(utf8) => {
            let mut offset_bytes = Vec::with_capacity(utf8.offsets.len() * 4);
            for offset in &utf8.offsets {
                offset_bytes.extend_from_slice(&offset.to_le_bytes());
            }
            let offsets_ptr = alloc_bytes(&offset_bytes);
            let values_ptr = alloc_bytes(&utf8.values);
            let validity_ptr = utf8.validity.as_ref().map(|v| alloc_bytes(v)).unwrap_or(0);
            write_structs(
                "u",
                utf8.len,
                utf8.nullable,
                &[validity_ptr, offsets_ptr, values_ptr],
            )
        }
        // `write` routes Take before calling this.
        Decoded::Take(_) => (0, 0),
    }
}

/// Lay out the `ArrowSchema`/`ArrowArray` structs (plus the buffer-pointer table) for an array
/// whose buffers are already in linear memory, returning `(array_ptr, schema_ptr)`.
fn write_structs(format: &str, len: usize, nullable: bool, buffer_ptrs: &[u32]) -> (u32, u32) {
    let mut format_bytes = Vec::with_capacity(format.len() + 1);
    format_bytes.extend_from_slice(format.as_bytes());
    format_bytes.push(0);
    let format_ptr = alloc_bytes(&format_bytes);

    let mut buffers = Vec::with_capacity(buffer_ptrs.len() * 4);
    for ptr in buffer_ptrs {
        buffers.extend_from_slice(&ptr.to_le_bytes());
    }
    let buffers_ptr = alloc_bytes(&buffers);

    let mut schema_buf = [0u8; SCHEMA_SIZE];
    schema_buf[schema::FORMAT..schema::FORMAT + 4].copy_from_slice(&format_ptr.to_le_bytes());
    let flags: i64 = if nullable { ARROW_FLAG_NULLABLE } else { 0 };
    schema_buf[schema::FLAGS..schema::FLAGS + 8].copy_from_slice(&flags.to_le_bytes());
    let schema_ptr = alloc_bytes(&schema_buf);

    let has_bitmap = buffer_ptrs.first().is_some_and(|&v| v != 0);
    let mut array_buf = [0u8; ARRAY_SIZE];
    array_buf[array::LENGTH..array::LENGTH + 8].copy_from_slice(&(len as i64).to_le_bytes());
    let null_count: i64 = if has_bitmap { -1 } else { 0 };
    array_buf[array::NULL_COUNT..array::NULL_COUNT + 8].copy_from_slice(&null_count.to_le_bytes());
    array_buf[array::N_BUFFERS..array::N_BUFFERS + 8]
        .copy_from_slice(&(buffer_ptrs.len() as i64).to_le_bytes());
    array_buf[array::BUFFERS..array::BUFFERS + 4].copy_from_slice(&buffers_ptr.to_le_bytes());
    let array_ptr = alloc_bytes(&array_buf);

    (array_ptr, schema_ptr)
}

/// A read-only view of a host-decoded child array delivered as Arrow C structs.
pub enum ChildView {
    /// A primitive child.
    Primitive(PrimitiveView),
    /// A boolean child (e.g. a validity bitmap).
    Bool(BoolView),
}

/// A primitive child array.
pub struct PrimitiveView {
    /// Element type.
    pub ptype: PType,
    /// Logical element count.
    pub len: usize,
    /// Little-endian values (`len * ptype.byte_width()` bytes).
    pub values: &'static [u8],
    /// Validity bitmap, if the child carries one.
    pub validity: Option<&'static [u8]>,
}

impl PrimitiveView {
    /// Read element `i` widened to `u64` (values are unsigned-reinterpreted).
    pub fn value_u64(&self, i: usize) -> u64 {
        let w = self.ptype.byte_width();
        let bytes = &self.values[i * w..(i + 1) * w];
        let mut buf = [0u8; 8];
        buf[..w].copy_from_slice(bytes);
        u64::from_le_bytes(buf)
    }
}

/// A boolean child array; values are an LSB-first bitmap.
pub struct BoolView {
    /// Logical element count.
    pub len: usize,
    /// The values bitmap (`ceil(len / 8)` bytes).
    pub bits: &'static [u8],
    /// Validity bitmap, if the child carries one.
    pub validity: Option<&'static [u8]>,
}

/// Parse the Arrow C Data Interface structs at `array_ptr`/`schema_ptr` in this module's linear
/// memory into a [`ChildView`].
///
/// # Safety
///
/// The host guarantees the structs and their buffers live in this module's memory for the duration
/// of the decode call, so the returned `'static` slices are valid until the call returns.
pub fn read_child(array_ptr: u32, schema_ptr: u32) -> GuestResult<ChildView> {
    unsafe {
        let format_ptr = load_u32(schema_ptr + schema::FORMAT as u32);
        let format = load_cstr(format_ptr)?;
        let len = load_i64(array_ptr + array::LENGTH as u32) as usize;
        let offset = load_i64(array_ptr + array::OFFSET as u32) as usize;
        if offset != 0 {
            return Err(GuestError::new("child arrays must have offset 0"));
        }
        let buffers_ptr = load_u32(array_ptr + array::BUFFERS as u32);
        let validity_ptr = load_u32(buffers_ptr);
        let values_ptr = load_u32(buffers_ptr + 4);

        let validity = if validity_ptr != 0 {
            Some(core::slice::from_raw_parts(
                validity_ptr as *const u8,
                len.div_ceil(8),
            ))
        } else {
            None
        };

        if format == "b" {
            let bits = core::slice::from_raw_parts(values_ptr as *const u8, len.div_ceil(8));
            return Ok(ChildView::Bool(BoolView {
                len,
                bits,
                validity,
            }));
        }

        let ptype = PType::from_format(format)
            .ok_or(GuestError::new("child has unsupported Arrow format"))?;
        let values = core::slice::from_raw_parts(values_ptr as *const u8, len * ptype.byte_width());
        Ok(ChildView::Primitive(PrimitiveView {
            ptype,
            len,
            values,
            validity,
        }))
    }
}

unsafe fn load_u32(off: u32) -> u32 {
    let mut b = [0u8; 4];
    unsafe { core::ptr::copy_nonoverlapping(off as *const u8, b.as_mut_ptr(), 4) };
    u32::from_le_bytes(b)
}

unsafe fn load_i64(off: u32) -> i64 {
    let mut b = [0u8; 8];
    unsafe { core::ptr::copy_nonoverlapping(off as *const u8, b.as_mut_ptr(), 8) };
    i64::from_le_bytes(b)
}

unsafe fn load_cstr(ptr: u32) -> GuestResult<&'static str> {
    // Format codes are 1-3 ASCII bytes; scan a small bound for the NUL terminator.
    for n in 0..8u32 {
        let byte = unsafe { *((ptr + n) as *const u8) };
        if byte == 0 {
            let slice = unsafe { core::slice::from_raw_parts(ptr as *const u8, n as usize) };
            return core::str::from_utf8(slice)
                .map_err(|_| GuestError::new("non-utf8 Arrow format code"));
        }
    }
    Err(GuestError::new("unterminated Arrow format code"))
}
