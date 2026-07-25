// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The data crossing the host/guest boundary, in **Vortex's own vocabulary**.
//!
//! There is deliberately no Arrow C Data Interface here. That protocol carries a schema, and this
//! boundary has no schema to carry: the host already holds the node's `DType`, and the guest
//! declares its children's dtypes itself. Shipping `ArrowSchema`/`ArrowArray` structs would mean
//! the guest writes a format string that the host parses back into a type it already knew, then
//! revalidates and converts into Vortex's representation — and for strings that conversion is
//! lossy in cost, landing on `VarBin` rather than the canonical `VarBinView`.
//!
//! So arrays cross as a **buffer table plus a small shape tag**, in exactly the layout Vortex's
//! canonical arrays already use (which for primitives and bools is byte-identical to Arrow's
//! anyway — only the schema goes away). See [`crate::abi`] for the wire formats.

use alloc::vec;
use alloc::vec::Vec;

use crate::abi::PType;
use crate::abi::child_entry;
use crate::abi::decode_result;
use crate::abi::shape;
use crate::abi::validity;
use crate::error::GuestError;
use crate::error::GuestResult;
use crate::host::alloc_bytes;

/// Values up to this length are stored inline in the view instead of the data buffer.
const MAX_INLINED: usize = 12;

/// The validity of an array, as an algebra rather than an always-materialized bitmap.
///
/// This matters on both sides: a non-nullable or all-valid array carries no bitmap across the
/// boundary at all, where the previous Arrow-shaped channel copied one for nothing.
pub enum Validity {
    /// The dtype is not nullable.
    NonNullable,
    /// Nullable, all elements valid.
    AllValid,
    /// Nullable, all elements null.
    AllInvalid,
    /// An LSB-first bitmap, `ceil(len / 8)` bytes, 1 = valid.
    Bitmap(Vec<u8>),
}

impl Validity {
    fn tag(&self) -> u8 {
        match self {
            Validity::NonNullable => validity::NON_NULLABLE,
            Validity::AllValid => validity::ALL_VALID,
            Validity::AllInvalid => validity::ALL_INVALID,
            Validity::Bitmap(_) => validity::BITMAP,
        }
    }
}

/// What a kernel produces for a node.
///
/// Encodings that compute *new element values* (bit-packing, FSST, zstd) return a materialized
/// variant. Encodings that merely **re-arrange** an existing child (run-end, dict) return
/// [`Decoded::Take`], so the child never enters guest memory and may have any dtype.
pub enum Decoded {
    /// A materialized primitive array.
    Primitive(DecodedPrimitive),
    /// A materialized boolean array.
    Bool(DecodedBool),
    /// A materialized string/binary array in Vortex's canonical view layout.
    VarBinView(DecodedVarBinView),
    /// The output is a referenced child gathered by guest-computed indices.
    Take(DecodedTake),
}

/// A materialized primitive array.
pub struct DecodedPrimitive {
    /// Element type.
    pub ptype: PType,
    /// Logical element count.
    pub len: usize,
    /// Little-endian values, `len * ptype.byte_width()` bytes.
    pub values: Vec<u8>,
    /// Element validity.
    pub validity: Validity,
}

/// A materialized boolean array.
pub struct DecodedBool {
    /// Logical element count.
    pub len: usize,
    /// LSB-first values bitmap, `ceil(len / 8)` bytes.
    pub bits: Vec<u8>,
    /// Element validity.
    pub validity: Validity,
}

/// A materialized string/binary array in Vortex's canonical layout: 16-byte views plus data
/// buffers. Build one with [`DecodedVarBinView::from_heap`] rather than by hand.
pub struct DecodedVarBinView {
    /// Logical element count.
    pub len: usize,
    /// `len * 16` bytes of views.
    pub views: Vec<u8>,
    /// The data buffers views may reference.
    pub data: Vec<Vec<u8>>,
    /// Element validity.
    pub validity: Validity,
}

impl DecodedVarBinView {
    /// Build a canonical view array from one concatenated heap plus per-element lengths.
    ///
    /// This is the shape decompressors naturally produce (FSST, zstd), and emitting views here
    /// means the host constructs `VarBinViewArray` directly — the canonical form — instead of
    /// going through an offsets-based `VarBin` and paying a second conversion.
    pub fn from_heap(
        heap: Vec<u8>,
        lengths: impl IntoIterator<Item = usize>,
        validity: Validity,
    ) -> GuestResult<Self> {
        let mut views: Vec<u8> = Vec::new();
        let mut offset = 0usize;
        let mut len = 0usize;
        for size in lengths {
            let end = offset
                .checked_add(size)
                .filter(|&end| end <= heap.len())
                .ok_or(GuestError::new("string length exceeds the heap"))?;
            let value = &heap[offset..end];
            let size_u32 =
                u32::try_from(size).map_err(|_| GuestError::new("string too long for a view"))?;

            views.extend_from_slice(&size_u32.to_le_bytes());
            if size <= MAX_INLINED {
                // Inlined: [u32 size][12 bytes of value, zero padded]
                let mut inline = [0u8; MAX_INLINED];
                inline[..size].copy_from_slice(value);
                views.extend_from_slice(&inline);
            } else {
                // Ref: [u32 size][4-byte prefix][u32 buffer_index][u32 offset]
                views.extend_from_slice(&value[..4]);
                views.extend_from_slice(&0u32.to_le_bytes());
                views.extend_from_slice(
                    &u32::try_from(offset)
                        .map_err(|_| GuestError::new("heap offset exceeds u32"))?
                        .to_le_bytes(),
                );
            }
            offset = end;
            len += 1;
        }
        Ok(Self {
            len,
            views,
            data: vec![heap],
            validity,
        })
    }
}

/// The output is child `values_slot`, gathered by `indices`.
pub struct DecodedTake {
    /// The serialized child slot to gather from.
    pub values_slot: u16,
    /// One unsigned index per output element, each `< values.len()`.
    pub indices: DecodedPrimitive,
}

/// Write a [`Decoded`] as a `vx_decode` result frame, returning its offset.
pub fn write(decoded: &Decoded) -> i32 {
    let mut frame = Vec::new();
    match decoded {
        Decoded::Take(take) => {
            frame.extend_from_slice(&decode_result::TAG_TAKE.to_le_bytes());
            frame.extend_from_slice(&u32::from(take.values_slot).to_le_bytes());
            write_array(&mut frame, DecodedRef::Primitive(&take.indices));
        }
        other => {
            frame.extend_from_slice(&decode_result::TAG_MATERIALIZED.to_le_bytes());
            write_array(&mut frame, other);
        }
    }
    alloc_bytes(&frame) as i32
}

/// A borrowed view of a materialized output, so the writer is shared between the direct and
/// `Take`-indices paths.
enum DecodedRef<'a> {
    Primitive(&'a DecodedPrimitive),
    Bool(&'a DecodedBool),
    VarBinView(&'a DecodedVarBinView),
}

impl<'a> From<&'a Decoded> for DecodedRef<'a> {
    fn from(decoded: &'a Decoded) -> Self {
        match decoded {
            Decoded::Primitive(p) => DecodedRef::Primitive(p),
            Decoded::Bool(b) => DecodedRef::Bool(b),
            Decoded::VarBinView(v) => DecodedRef::VarBinView(v),
            // `write` routes Take before reaching here.
            Decoded::Take(t) => DecodedRef::Primitive(&t.indices),
        }
    }
}

/// Append an array descriptor + buffer table to `frame` (see [`decode_result`]).
fn write_array<'a>(frame: &mut Vec<u8>, decoded: impl Into<DecodedRef<'a>>) {
    let decoded = decoded.into();
    let (shape_tag, ptype, len, validity, buffers): (u8, u8, usize, &Validity, Vec<&[u8]>) =
        match decoded {
            DecodedRef::Primitive(p) => (
                shape::PRIMITIVE,
                p.ptype as u8,
                p.len,
                &p.validity,
                vec![p.values.as_slice()],
            ),
            DecodedRef::Bool(b) => (shape::BOOL, 0, b.len, &b.validity, vec![b.bits.as_slice()]),
            DecodedRef::VarBinView(v) => {
                let mut bufs = vec![v.views.as_slice()];
                bufs.extend(v.data.iter().map(|d| d.as_slice()));
                (shape::VAR_BIN_VIEW, 0, v.len, &v.validity, bufs)
            }
        };

    let validity_ptr = match validity {
        Validity::Bitmap(bits) => alloc_bytes(bits),
        _ => 0,
    };

    frame.push(shape_tag);
    frame.push(ptype);
    frame.push(validity.tag());
    frame.push(buffers.len() as u8);
    frame.extend_from_slice(&(len as u32).to_le_bytes());
    frame.extend_from_slice(&validity_ptr.to_le_bytes());
    for buffer in buffers {
        frame.extend_from_slice(&alloc_bytes(buffer).to_le_bytes());
        frame.extend_from_slice(&(buffer.len() as u32).to_le_bytes());
    }
}

/// A read-only view of a host-supplied `Values` child.
///
/// Only primitive and boolean children are deliverable: anything else should be declared
/// [`ChildMode::Reference`](crate::node::ChildMode::Reference), which imposes no dtype limit
/// because the child never enters guest memory.
pub enum ChildView {
    /// A primitive child.
    Primitive(PrimitiveView),
    /// A boolean child.
    Bool(BoolView),
}

/// The validity of a host-supplied child.
pub enum ValidityView {
    /// The dtype is not nullable.
    NonNullable,
    /// Nullable, all elements valid.
    AllValid,
    /// Nullable, all elements null.
    AllInvalid,
    /// An LSB-first bitmap, `ceil(len / 8)` bytes, 1 = valid.
    Bitmap(&'static [u8]),
}

/// A primitive child array.
pub struct PrimitiveView {
    /// Element type.
    pub ptype: PType,
    /// Logical element count.
    pub len: usize,
    /// Little-endian values.
    pub values: &'static [u8],
    /// Element validity.
    pub validity: ValidityView,
}

impl PrimitiveView {
    /// Read element `i` widened to `u64` (values are unsigned-reinterpreted).
    pub fn value_u64(&self, i: usize) -> u64 {
        let w = self.ptype.byte_width();
        let mut buf = [0u8; 8];
        buf[..w].copy_from_slice(&self.values[i * w..(i + 1) * w]);
        u64::from_le_bytes(buf)
    }
}

/// A boolean child array; values are an LSB-first bitmap.
pub struct BoolView {
    /// Logical element count.
    pub len: usize,
    /// The values bitmap, `ceil(len / 8)` bytes.
    pub bits: &'static [u8],
    /// Element validity.
    pub validity: ValidityView,
}

/// Parse one child entry from the decode frame.
///
/// # Safety
///
/// The host wrote these buffers into this module's memory before the call and keeps them alive
/// for its duration, so the `'static` slices are valid until `vx_decode` returns.
pub(crate) fn read_child(entry: &[u8]) -> GuestResult<ChildView> {
    let shape_tag = entry[child_entry::SHAPE];
    let validity_tag = entry[child_entry::VALIDITY];
    let len = read_u32(entry, child_entry::LEN) as usize;
    let values_ptr = read_u32(entry, child_entry::VALUES_PTR);
    let values_len = read_u32(entry, child_entry::VALUES_LEN) as usize;
    let validity_ptr = read_u32(entry, child_entry::VALIDITY_PTR);

    // SAFETY: host-owned guest memory, valid for the duration of the decode call.
    let values = unsafe { core::slice::from_raw_parts(values_ptr as *const u8, values_len) };
    let validity = match validity_tag {
        validity::NON_NULLABLE => ValidityView::NonNullable,
        validity::ALL_VALID => ValidityView::AllValid,
        validity::ALL_INVALID => ValidityView::AllInvalid,
        validity::BITMAP => ValidityView::Bitmap(unsafe {
            core::slice::from_raw_parts(validity_ptr as *const u8, len.div_ceil(8))
        }),
        _ => return Err(GuestError::new("bad child validity tag")),
    };

    match shape_tag {
        shape::PRIMITIVE => {
            let ptype = PType::from_discriminant(u64::from(entry[child_entry::PTYPE]))
                .ok_or(GuestError::new("bad child ptype"))?;
            Ok(ChildView::Primitive(PrimitiveView {
                ptype,
                len,
                values,
                validity,
            }))
        }
        shape::BOOL => Ok(ChildView::Bool(BoolView {
            len,
            bits: values,
            validity,
        })),
        _ => Err(GuestError::new("bad child shape tag")),
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}
