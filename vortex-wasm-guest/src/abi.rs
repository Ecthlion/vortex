// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Host/guest ABI constants shared with the host.
//!
//! A kernel is the decoder for one Vortex array encoding. The host resolves an unknown encoding id
//! to a kernel and drives it with the array's **real serialized parts** — the same
//! `(len, metadata, buffers, children)` a native `VTable::deserialize` receives:
//!
//! 1. `vx_children(input_ptr, input_len) -> i32`: given the node header (see
//!    [`children_frame`]), the guest returns descriptors for the node's serialized children
//!    (dtype + length each) so the host can decode them natively.
//! 2. `vx_decode(input_ptr, input_len) -> i32`: the host pushes the metadata, the raw buffers, and
//!    the canonical children into guest memory in one frame (see [`decode_frame`]); the guest
//!    returns a pointer to its output, an [`array_frame`] of the node's dtype.
//!
//! There are no host callbacks during decode — the host pushes everything up front. Arrays cross
//! the boundary in **Vortex's own canonical layouts** (a recursive [`array_frame`] with a
//! [`shape`] tag), not as Arrow C structs. Types cross in the compact [`dtype`](crate::dtype)
//! encoding. These layouts MUST match `vortex-wasm`'s `convert` and `dtype` modules.

/// Host/guest ABI version.
///
/// The host refuses to run a kernel whose declared version it does not implement. Every layout in
/// this module is part of version 1, so any change to a frame, the [`array_frame`] layout, or the
/// [`dtype`](crate::dtype) grammar bumps it.
pub const ABI_VERSION: u32 = 1;

/// Host import module name the guest links against.
pub const HOST_MODULE: &str = "vortex_host";

/// The `vx_children` input frame (all integers little-endian).
///
/// ```text
/// [u64 parent_len][u32 flags][u32 n_children][u32 dtype_len][u32 metadata_len]
/// [dtype…][metadata…]
/// ```
///
/// `flags` bit 0: the parent dtype is nullable (also recoverable from the dtype itself; kept
/// because it is free and it is what most kernels actually check). `n_children` is the number of
/// serialized children present on the node (the guest uses it to detect an optional trailing
/// validity child).
pub mod children_frame {
    /// Byte offset of `parent_len`.
    pub const PARENT_LEN: usize = 0;
    /// Byte offset of `flags`.
    pub const FLAGS: usize = 8;
    /// Byte offset of `n_children`.
    pub const N_CHILDREN: usize = 12;
    /// Byte offset of `dtype_len`.
    pub const DTYPE_LEN: usize = 16;
    /// Byte offset of `metadata_len`.
    pub const METADATA_LEN: usize = 20;
    /// Total header size; the dtype then the metadata follow.
    pub const HEADER: usize = 24;
}

/// The `vx_children` output: `[u32 n]` followed by `n` variable-length descriptors:
///
/// ```text
/// [u32 dtype_len][u64 len][dtype…]
/// ```
///
/// The dtype is a [`DTypeExpr`](crate::dtype::DTypeExpr) — a literal or a derivation of the
/// parent's type. The host decodes every declared child and delivers it canonical, whatever its
/// dtype.
pub mod child_descriptor {
    /// Fixed part of one descriptor, before the dtype bytes.
    pub const HEADER: usize = 12;
    /// Byte offset of the dtype length.
    pub const DTYPE_LEN: usize = 0;
    /// Byte offset of the logical element count.
    pub const LEN: usize = 4;
}

/// Tag byte layout for the [`dtype`](crate::dtype) encoding.
pub mod dtype_tag {
    /// Mask selecting the kind (or derivation opcode) from a tag byte.
    pub const KIND_MASK: u8 = 0x3f;
    /// Bit 6: the type is nullable.
    pub const NULLABLE: u8 = 0x40;
    /// Bit 7: the low bits are a [`dtype_derivation`](super::dtype_derivation) opcode rather than
    /// a [`dtype_kind`](super::dtype_kind).
    pub const DERIVED: u8 = 0x80;
}

/// Literal dtype kinds. These are the ABI's own numbering, not Vortex's internal discriminants.
pub mod dtype_kind {
    /// The logical null type.
    pub const NULL: u8 = 0;
    /// A boolean.
    pub const BOOL: u8 = 1;
    /// A fixed-width numeric; payload is one [`PType`](super::PType) discriminant.
    pub const PRIMITIVE: u8 = 2;
    /// A fixed-precision decimal; payload is `u8` precision and `i8` scale.
    pub const DECIMAL: u8 = 3;
    /// A UTF-8 string.
    pub const UTF8: u8 = 4;
    /// Binary data.
    pub const BINARY: u8 = 5;
    /// A variable-length list; payload is the element dtype.
    pub const LIST: u8 = 6;
    /// A fixed-size list; payload is a varint size and the element dtype.
    pub const FIXED_SIZE_LIST: u8 = 7;
    /// A struct; payload is a varint field count then `(varint name_len, name, dtype)` each.
    pub const STRUCT: u8 = 8;
    /// A union; payload is a varint variant count then `(varint name_len, name, u8 type_id,
    /// dtype)` each.
    pub const UNION: u8 = 9;
    /// A dynamically typed value.
    pub const VARIANT: u8 = 10;
    /// An extension type; payload is a varint-prefixed id, varint-prefixed metadata, and the
    /// storage dtype.
    pub const EXTENSION: u8 = 11;
    /// A map; payload is a `u8` keys-sorted flag, the key dtype, then the value dtype.
    pub const MAP: u8 = 12;
}

/// Derivation opcodes: a type named as a path from the parent rather than spelled out.
///
/// Guest→host only. They exist because a literal is not always writable — an extension type needs
/// a host vtable the guest cannot synthesize — and because a kernel generic over its parent should
/// not have to name a concrete type at all.
pub mod dtype_derivation {
    /// The parent node's own dtype. No payload.
    pub const PARENT: u8 = 0;
    /// Struct field `i` of the inner type; payload is a varint index then the inner dtype.
    pub const FIELD: u8 = 1;
    /// The element type of the inner list type; payload is the inner dtype.
    pub const ELEMENT: u8 = 2;
    /// The storage type of the inner extension type; payload is the inner dtype.
    pub const STORAGE: u8 = 3;
    /// The inner type, made nullable; payload is the inner dtype.
    pub const NULLABLE: u8 = 4;
    /// The inner type, made non-nullable; payload is the inner dtype.
    pub const NON_NULLABLE: u8 = 5;
}

/// The canonical shape of an [`array_frame`]. One tag per variant of Vortex's `Canonical`, so an
/// array of any dtype can cross the boundary in either direction. The dtype is known on both
/// sides, so no schema is transmitted; the shape says how to read the frame's buffers and
/// children.
///
/// | shape | `param` | buffers | children |
/// |---|---|---|---|
/// | `NULL` | — | — | — |
/// | `BOOL` | — | LSB-first values bitmap | — |
/// | `PRIMITIVE` | [`PType`] | little-endian values | — |
/// | `DECIMAL` | `DecimalType` storage width | little-endian scaled integers | — |
/// | `VAR_BIN_VIEW` | — | 16-byte views, then data buffers | — |
/// | `LIST` | — | — | offsets, sizes, elements |
/// | `FIXED_SIZE_LIST` | — | — | elements (`len × size`) |
/// | `STRUCT` | — | — | one per field, in dtype order |
/// | `UNION` | — | — | `u8` type ids (carrying the union's validity), then one per variant |
/// | `MAP` | — | — | entries: a `LIST` of non-nullable `STRUCT { key, value }` |
/// | `EXTENSION` | — | — | storage |
///
/// `Variant` has no shape: Vortex defines no physical canonical layout for variant values yet, so
/// the host refuses to deliver or accept one, with a clear error, while the dtype channel names it.
pub mod shape {
    /// The logical null array.
    pub const NULL: u8 = 0;
    /// One buffer: an LSB-first values bitmap.
    pub const BOOL: u8 = 1;
    /// One buffer: little-endian values.
    pub const PRIMITIVE: u8 = 2;
    /// One buffer: little-endian scaled integers of the `param` storage width.
    pub const DECIMAL: u8 = 3;
    /// `1 + n` buffers: 16-byte views, then the data buffers they reference.
    pub const VAR_BIN_VIEW: u8 = 4;
    /// Three children: offsets, sizes, elements.
    pub const LIST: u8 = 5;
    /// One child: the flattened elements.
    pub const FIXED_SIZE_LIST: u8 = 6;
    /// One child per field.
    pub const STRUCT: u8 = 7;
    /// Type ids, then one child per variant.
    pub const UNION: u8 = 8;
    /// One child: the entries list.
    pub const MAP: u8 = 9;
    /// One child: the storage array.
    pub const EXTENSION: u8 = 10;
}

/// Validity, as an algebra — a non-nullable or all-valid array transmits no bitmap.
pub mod validity {
    /// The dtype is not nullable.
    pub const NON_NULLABLE: u8 = 0;
    /// Nullable, all elements valid.
    pub const ALL_VALID: u8 = 1;
    /// Nullable, all elements null.
    pub const ALL_INVALID: u8 = 2;
    /// An LSB-first bitmap of `ceil(len / 8)` bytes, 1 = valid.
    pub const BITMAP: u8 = 3;
}

/// An array, in either direction: the host writes each canonical child as one of these, and the
/// guest returns its output as one.
///
/// ```text
/// [u8 shape][u8 param][u8 validity][u8 pad][u32 len][u32 validity_ptr]
/// [u32 n_buffers][(u32 ptr, u32 len) × n_buffers]
/// [u32 n_children][array_frame × n_children]
/// ```
///
/// Children are inline, in preorder. Buffer and bitmap pointers are absolute guest offsets. See
/// [`shape`] for which buffers and children each shape carries.
pub mod array_frame {
    /// Byte offset of the [`shape`](super::shape) tag.
    pub const SHAPE: usize = 0;
    /// Byte offset of the shape parameter (ptype or decimal storage width).
    pub const PARAM: usize = 1;
    /// Byte offset of the [`validity`](super::validity) tag.
    pub const VALIDITY: usize = 2;
    /// Byte offset of the logical element count.
    pub const LEN: usize = 4;
    /// Byte offset of the validity bitmap pointer (0 unless the tag is `BITMAP`).
    pub const VALIDITY_PTR: usize = 8;
    /// Byte offset of the buffer count.
    pub const N_BUFFERS: usize = 12;
    /// Fixed part of the frame, before the buffer table.
    pub const HEADER: usize = 16;
}

/// The `vx_decode` input frame (all integers little-endian).
///
/// ```text
/// [u64 parent_len][u32 flags][u32 dtype_len][u32 metadata_len][u32 n_buffers][u32 n_children]
/// [dtype…][metadata…]
/// [(u32 buffer_ptr, u32 buffer_len) × n_buffers]
/// [u32 child_ptr × n_children]
/// ```
///
/// `flags` bit 0: the parent dtype is nullable. Buffers are the node's raw serialized buffers,
/// already copied into guest memory; each child pointer addresses an [`array_frame`] of the
/// canonical child.
pub mod decode_frame {
    /// Byte offset of `parent_len`.
    pub const PARENT_LEN: usize = 0;
    /// Byte offset of `flags`.
    pub const FLAGS: usize = 8;
    /// Byte offset of `dtype_len`.
    pub const DTYPE_LEN: usize = 12;
    /// Byte offset of `metadata_len`.
    pub const METADATA_LEN: usize = 16;
    /// Byte offset of `n_buffers`.
    pub const N_BUFFERS: usize = 20;
    /// Byte offset of `n_children`.
    pub const N_CHILDREN: usize = 24;
    /// Total header size; the dtype, metadata, buffer table, and child pointer table follow.
    pub const HEADER: usize = 28;
}

/// Frame flag bit 0: the parent dtype is nullable.
pub const FLAG_NULLABLE: u32 = 1;

/// Primitive type. The discriminants match Vortex's `PType` prost enumeration, so metadata enum
/// fields decode directly and the tag can be passed straight back to the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PType {
    /// `u8`
    U8 = 0,
    /// `u16`
    U16 = 1,
    /// `u32`
    U32 = 2,
    /// `u64`
    U64 = 3,
    /// `i8`
    I8 = 4,
    /// `i16`
    I16 = 5,
    /// `i32`
    I32 = 6,
    /// `i64`
    I64 = 7,
    /// `f16`
    F16 = 8,
    /// `f32`
    F32 = 9,
    /// `f64`
    F64 = 10,
}

impl PType {
    /// Width in bytes.
    pub const fn byte_width(self) -> usize {
        match self {
            PType::U8 | PType::I8 => 1,
            PType::U16 | PType::I16 | PType::F16 => 2,
            PType::U32 | PType::I32 | PType::F32 => 4,
            PType::U64 | PType::I64 | PType::F64 => 8,
        }
    }

    /// Parse the Vortex `PType` prost enumeration discriminant (used in encoding metadata).
    pub fn from_discriminant(value: u64) -> Option<Self> {
        Some(match value {
            0 => PType::U8,
            1 => PType::U16,
            2 => PType::U32,
            3 => PType::U64,
            4 => PType::I8,
            5 => PType::I16,
            6 => PType::I32,
            7 => PType::I64,
            8 => PType::F16,
            9 => PType::F32,
            10 => PType::F64,
            _ => return None,
        })
    }
}
