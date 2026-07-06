// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Host/guest ABI constants and the Arrow C Data Interface layout (wasm32) shared with the host.
//!
//! A kernel is the decoder for one Vortex array encoding. The host resolves an unknown encoding id
//! to a kernel and drives it with the array's **real serialized parts** — the same
//! `(len, metadata, buffers, children)` a native `VTable::deserialize` receives:
//!
//! 1. `vx_children(input_ptr, input_len) -> i32`: given the node header (see
//!    [`children_frame`]), the guest returns descriptors for the node's serialized children
//!    (dtype + length each) so the host can decode them natively.
//! 2. `vx_decode(input_ptr, input_len) -> i32`: the host pushes the metadata, the raw buffers
//!    (copied into guest memory), and the decoded children (as Arrow C structs) in one frame (see
//!    [`decode_frame`]); the guest returns a pointer to the `(array_ptr, schema_ptr)` pair of its
//!    decoded output.
//!
//! There are no host callbacks during decode — the host pushes everything up front. Decoded arrays
//! cross the boundary as the [Arrow C Data Interface]: the guest builds and reads the
//! `ArrowSchema`/`ArrowArray` structs directly (plain byte layouts). These offsets MUST match
//! `vortex-wasm`'s `arrow_ffi` module.
//!
//! [Arrow C Data Interface]: https://arrow.apache.org/docs/format/CDataInterface.html

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
/// [u8 tag][u8 ptype][u8 nullable][u8 pad x5][u64 len]
/// ```
pub mod child_descriptor {
    /// The child has the parent's dtype (e.g. patch values).
    pub const TAG_PARENT: u8 = 0;
    /// A primitive child; `ptype` holds the [`PType`](super::PType) discriminant.
    pub const TAG_PRIMITIVE: u8 = 1;
    /// A boolean child (e.g. a validity bitmap).
    pub const TAG_BOOL: u8 = 2;
    /// A utf8 child.
    pub const TAG_UTF8: u8 = 3;
    /// Size of one descriptor.
    pub const SIZE: usize = 16;
}

/// The `vx_decode` input frame (all integers little-endian).
///
/// ```text
/// [u64 parent_len][u32 flags][u32 metadata_len][u32 n_buffers][u32 n_children]
/// [metadata…]
/// [(u32 buffer_ptr, u32 buffer_len) x n_buffers]
/// [(u32 array_ptr, u32 schema_ptr) x n_children]
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

/// Size of an `ArrowSchema` struct in the wasm32 C ABI.
pub const SCHEMA_SIZE: usize = 48;
/// Size of an `ArrowArray` struct in the wasm32 C ABI.
pub const ARRAY_SIZE: usize = 64;

/// `ArrowSchema` field offsets (wasm32 C ABI: 4-byte pointers, 8-aligned `int64`).
pub mod schema {
    /// `const char* format`
    pub const FORMAT: usize = 0;
    /// `int64 flags`
    pub const FLAGS: usize = 16;
}

/// `ArrowArray` field offsets (wasm32 C ABI).
pub mod array {
    /// `int64 length`
    pub const LENGTH: usize = 0;
    /// `int64 null_count`
    pub const NULL_COUNT: usize = 8;
    /// `int64 offset`
    pub const OFFSET: usize = 16;
    /// `int64 n_buffers`
    pub const N_BUFFERS: usize = 24;
    /// `const void** buffers`
    pub const BUFFERS: usize = 40;
}

/// Arrow schema flag: the field may contain nulls.
pub const ARROW_FLAG_NULLABLE: i64 = 2;

/// Primitive type. The discriminants match Vortex's `PType` prost enumeration, so metadata
/// enum fields decode directly; the format codes match the Arrow C Data Interface.
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

    /// Arrow C Data Interface format code (no trailing NUL).
    pub const fn format_code(self) -> &'static str {
        match self {
            PType::I8 => "c",
            PType::U8 => "C",
            PType::I16 => "s",
            PType::U16 => "S",
            PType::I32 => "i",
            PType::U32 => "I",
            PType::I64 => "l",
            PType::U64 => "L",
            PType::F16 => "e",
            PType::F32 => "f",
            PType::F64 => "g",
        }
    }

    /// Parse an Arrow C Data Interface primitive format code.
    pub fn from_format(format: &str) -> Option<Self> {
        Some(match format {
            "c" => PType::I8,
            "C" => PType::U8,
            "s" => PType::I16,
            "S" => PType::U16,
            "i" => PType::I32,
            "I" => PType::U32,
            "l" => PType::I64,
            "L" => PType::U64,
            "e" => PType::F16,
            "f" => PType::F32,
            "g" => PType::F64,
            _ => return None,
        })
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
