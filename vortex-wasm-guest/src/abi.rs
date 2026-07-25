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
//!    the decoded `Values` children into guest memory in one frame (see [`decode_frame`]); the
//!    guest returns a pointer to a [`decode_result`] frame.
//!
//! There are no host callbacks during decode — the host pushes everything up front. Arrays cross
//! the boundary in **Vortex's own canonical layouts** (a buffer table plus a [`shape`] tag), not
//! as Arrow C structs: the dtype is already known on both sides, so there is no schema to
//! transmit. These layouts MUST match `vortex-wasm`'s `convert` module.

/// Host/guest ABI version.
pub const ABI_VERSION: u32 = 2;

/// Host import module name the guest links against.
pub const HOST_MODULE: &str = "vortex_host";

/// The `vx_children` input frame (all integers little-endian).
///
/// ```text
/// [u64 parent_len][u32 flags][u32 n_children][u32 metadata_len][metadata…]
/// ```
///
/// `flags` bit 0: the parent dtype is nullable. `n_children` is the number of serialized children
/// present on the node (the guest uses it to detect an optional trailing validity child).
pub mod children_frame {
    /// Byte offset of `parent_len`.
    pub const PARENT_LEN: usize = 0;
    /// Byte offset of `flags`.
    pub const FLAGS: usize = 8;
    /// Byte offset of `n_children`.
    pub const N_CHILDREN: usize = 12;
    /// Byte offset of `metadata_len`.
    pub const METADATA_LEN: usize = 16;
    /// Total header size; the metadata follows.
    pub const HEADER: usize = 20;
}

/// The `vx_children` output: `[u32 n]` followed by `n` 16-byte descriptors:
///
/// ```text
/// [u8 tag][u8 ptype][u8 nullable][u8 mode][u8 pad x4][u64 len]
/// ```
pub mod child_descriptor {
    /// The child has the parent's dtype (e.g. run-end values, patch values).
    pub const TAG_PARENT: u8 = 0;
    /// A primitive child; `ptype` holds the [`PType`](super::PType) discriminant.
    pub const TAG_PRIMITIVE: u8 = 1;
    /// A boolean child (e.g. a validity bitmap).
    pub const TAG_BOOL: u8 = 2;
    /// A utf8 child.
    pub const TAG_UTF8: u8 = 3;
    /// Size of one descriptor.
    pub const SIZE: usize = 16;

    /// The guest will read this child's element bytes: the host copies its buffers into guest
    /// memory. Only primitive and boolean children are deliverable this way.
    pub const MODE_VALUES: u8 = 0;
    /// The guest only *names* this child in its result: the host resolves it lazily, in its own
    /// encoding, and never canonicalizes or copies it. A referenced child may therefore have any
    /// dtype, including nested ones the guest could not read or reproduce.
    pub const MODE_REFERENCE: u8 = 1;
}

/// How an array's buffers are to be interpreted. These are Vortex's canonical layouts, not Arrow
/// C structs: the dtype is already known on both sides, so no schema is transmitted.
pub mod shape {
    /// One buffer: little-endian values.
    pub const PRIMITIVE: u8 = 0;
    /// One buffer: an LSB-first values bitmap.
    pub const BOOL: u8 = 1;
    /// `1 + n` buffers: 16-byte views, then the data buffers they reference.
    pub const VAR_BIN_VIEW: u8 = 2;
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

/// A `Values` child in the `vx_decode` frame: a fixed 24-byte entry.
///
/// ```text
/// [u8 shape][u8 ptype][u8 validity][u8 pad][u32 len]
/// [u32 values_ptr][u32 values_len][u32 validity_ptr][u32 pad]
/// ```
pub mod child_entry {
    /// Byte offset of the [`shape`](super::shape) tag.
    pub const SHAPE: usize = 0;
    /// Byte offset of the [`PType`](super::PType) discriminant.
    pub const PTYPE: usize = 1;
    /// Byte offset of the [`validity`](super::validity) tag.
    pub const VALIDITY: usize = 2;
    /// Byte offset of the logical element count.
    pub const LEN: usize = 4;
    /// Byte offset of the values buffer pointer.
    pub const VALUES_PTR: usize = 8;
    /// Byte offset of the values buffer length.
    pub const VALUES_LEN: usize = 12;
    /// Byte offset of the validity bitmap pointer (0 unless the tag is `BITMAP`).
    pub const VALIDITY_PTR: usize = 16;
    /// Size of one entry.
    pub const SIZE: usize = 24;
}

/// The `vx_decode` result frame: `[u32 tag]` followed by a tag-specific body.
///
/// A materialized array descriptor is:
///
/// ```text
/// [u8 shape][u8 ptype][u8 validity][u8 n_buffers][u32 len][u32 validity_ptr]
/// [(u32 ptr, u32 len) x n_buffers]
/// ```
pub mod decode_result {
    /// The guest materialized the output; the body is one array descriptor.
    pub const TAG_MATERIALIZED: u32 = 0;
    /// The output is a child gathered by guest-materialized indices: `[u32 values_slot]` followed
    /// by one array descriptor for the index array. The host performs the gather, so the gathered
    /// child never crosses the boundary.
    pub const TAG_TAKE: u32 = 1;
}

/// The `vx_decode` input frame (all integers little-endian).
///
/// ```text
/// [u64 parent_len][u32 flags][u32 metadata_len][u32 n_buffers][u32 n_children]
/// [metadata…]
/// [(u32 buffer_ptr, u32 buffer_len) x n_buffers]
/// [child_entry x n_children]
/// ```
///
/// `flags` bit 0: the parent dtype is nullable. Buffers are the node's raw serialized buffers,
/// already copied into guest memory; children are the host-decoded child arrays as Arrow C
/// Data Interface struct pairs in guest memory.
pub mod decode_frame {
    /// Byte offset of `parent_len`.
    pub const PARENT_LEN: usize = 0;
    /// Byte offset of `flags`.
    pub const FLAGS: usize = 8;
    /// Byte offset of `metadata_len`.
    pub const METADATA_LEN: usize = 12;
    /// Byte offset of `n_buffers`.
    pub const N_BUFFERS: usize = 16;
    /// Byte offset of `n_children`.
    pub const N_CHILDREN: usize = 20;
    /// Total header size; metadata, then the buffer table, then the child table follow.
    pub const HEADER: usize = 24;
}

/// Frame flag bit 0: the parent dtype is nullable.
pub const FLAG_NULLABLE: u32 = 1;

/// Frame flags bits 8-15: the parent dtype's kind.
pub const PARENT_KIND_SHIFT: u32 = 8;
/// Frame flags bits 16-23: the parent's [`PType`] discriminant (when the kind is primitive).
pub const PARENT_PTYPE_SHIFT: u32 = 16;

/// Parent dtype kinds carried in the frame flags.
pub mod parent_kind {
    /// A dtype the frame cannot describe (kernels needing it should bail).
    pub const OTHER: u32 = 0;
    /// A primitive dtype; the ptype rides in bits 16-23.
    pub const PRIMITIVE: u32 = 1;
    /// A boolean dtype.
    pub const BOOL: u32 = 2;
    /// A utf8 dtype.
    pub const UTF8: u32 = 3;
}

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
