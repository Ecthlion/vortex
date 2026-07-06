// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Views over the frames the host passes to `vx_children` and `vx_decode` (see [`crate::abi`]).

use alloc::vec::Vec;

use crate::abi::PType;
use crate::abi::child_descriptor;
use crate::abi::children_frame;
use crate::abi::decode_frame;
use crate::arrow::ChildView;
use crate::arrow::read_child;
use crate::error::GuestError;
use crate::error::GuestResult;

/// The dtype of a serialized child, declared by the kernel so the host can decode it natively.
#[derive(Clone, Copy)]
pub enum ChildDType {
    /// The parent's own dtype (e.g. patch values).
    Parent,
    /// A primitive dtype.
    Primitive(PType, bool),
    /// A boolean dtype (e.g. a validity bitmap).
    Bool(bool),
    /// A utf8 dtype.
    Utf8(bool),
}

/// One serialized child's dtype and logical length.
#[derive(Clone, Copy)]
pub struct ChildSpec {
    /// The child's dtype.
    pub dtype: ChildDType,
    /// The child's logical element count.
    pub len: u64,
}

/// The parent (node) dtype, as far as the frame flags can describe it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ParentDType {
    /// A primitive dtype.
    Primitive(PType),
    /// A boolean dtype.
    Bool,
    /// A utf8 dtype.
    Utf8,
    /// Something the frame cannot describe.
    Other,
}

fn parse_parent(flags: u32) -> GuestResult<ParentDType> {
    use crate::abi::PARENT_KIND_SHIFT;
    use crate::abi::PARENT_PTYPE_SHIFT;
    use crate::abi::parent_kind;
    Ok(match (flags >> PARENT_KIND_SHIFT) & 0xff {
        parent_kind::PRIMITIVE => ParentDType::Primitive(
            PType::from_discriminant(u64::from((flags >> PARENT_PTYPE_SHIFT) & 0xff))
                .ok_or(GuestError::new("bad parent ptype in frame flags"))?,
        ),
        parent_kind::BOOL => ParentDType::Bool,
        parent_kind::UTF8 => ParentDType::Utf8,
        _ => ParentDType::Other,
    })
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(buf)
}

/// The header the host passes to `vx_children`: the node's length, nullability, serialized child
/// count, and encoding metadata.
pub struct NodeHeader<'a> {
    /// The node's logical element count.
    pub len: usize,
    /// Whether the node's dtype is nullable.
    pub nullable: bool,
    /// The node's dtype, as far as the frame can describe it.
    pub parent: ParentDType,
    /// The number of serialized children present on the node.
    pub n_children: usize,
    /// The encoding's metadata bytes (parse with [`crate::proto`]).
    pub metadata: &'a [u8],
}

impl<'a> NodeHeader<'a> {
    /// Parse a `vx_children` input frame.
    pub fn parse(input: &'a [u8]) -> GuestResult<Self> {
        if input.len() < children_frame::HEADER {
            return Err(GuestError::new("children frame too short"));
        }
        let len = read_u64(input, children_frame::PARENT_LEN) as usize;
        let flags = read_u32(input, children_frame::FLAGS);
        let n_children = read_u32(input, children_frame::N_CHILDREN) as usize;
        let metadata_len = read_u32(input, children_frame::METADATA_LEN) as usize;
        if input.len() < children_frame::HEADER + metadata_len {
            return Err(GuestError::new("children frame metadata out of bounds"));
        }
        Ok(Self {
            len,
            nullable: flags & crate::abi::FLAG_NULLABLE != 0,
            parent: parse_parent(flags)?,
            n_children,
            metadata: &input[children_frame::HEADER..children_frame::HEADER + metadata_len],
        })
    }
}

/// Serialize child specs into the `vx_children` output and return its pointer.
pub fn write_child_specs(specs: &[ChildSpec]) -> i32 {
    let mut out = Vec::with_capacity(4 + specs.len() * child_descriptor::SIZE);
    out.extend_from_slice(&(specs.len() as u32).to_le_bytes());
    for spec in specs {
        let (tag, ptype, nullable) = match spec.dtype {
            ChildDType::Parent => (child_descriptor::TAG_PARENT, 0u8, 0u8),
            ChildDType::Primitive(ptype, nullable) => {
                (child_descriptor::TAG_PRIMITIVE, ptype as u8, nullable as u8)
            }
            ChildDType::Bool(nullable) => (child_descriptor::TAG_BOOL, 0u8, nullable as u8),
            ChildDType::Utf8(nullable) => (child_descriptor::TAG_UTF8, 0u8, nullable as u8),
        };
        out.extend_from_slice(&[tag, ptype, nullable, 0, 0, 0, 0, 0]);
        out.extend_from_slice(&spec.len.to_le_bytes());
    }
    crate::host::alloc_bytes(&out) as i32
}

/// A view of the node the host asks the kernel to decode: the real serialized parts — metadata,
/// raw buffers, and host-decoded children.
pub struct NodeView<'a> {
    /// The node's logical element count.
    pub len: usize,
    /// Whether the node's dtype is nullable.
    pub nullable: bool,
    /// The node's dtype, as far as the frame can describe it.
    pub parent: ParentDType,
    /// The encoding's metadata bytes.
    pub metadata: &'a [u8],
    input: &'a [u8],
    buffers_table: usize,
    n_buffers: usize,
    children_table: usize,
    n_children: usize,
}

impl<'a> NodeView<'a> {
    /// Parse a `vx_decode` input frame.
    pub fn parse(input: &'a [u8]) -> GuestResult<Self> {
        if input.len() < decode_frame::HEADER {
            return Err(GuestError::new("decode frame too short"));
        }
        let len = read_u64(input, decode_frame::PARENT_LEN) as usize;
        let flags = read_u32(input, decode_frame::FLAGS);
        let metadata_len = read_u32(input, decode_frame::METADATA_LEN) as usize;
        let n_buffers = read_u32(input, decode_frame::N_BUFFERS) as usize;
        let n_children = read_u32(input, decode_frame::N_CHILDREN) as usize;

        let buffers_table = decode_frame::HEADER + metadata_len;
        let children_table = buffers_table + n_buffers * 8;
        if input.len() < children_table + n_children * 8 {
            return Err(GuestError::new("decode frame tables out of bounds"));
        }
        Ok(Self {
            len,
            nullable: flags & crate::abi::FLAG_NULLABLE != 0,
            parent: parse_parent(flags)?,
            metadata: &input[decode_frame::HEADER..buffers_table],
            input,
            buffers_table,
            n_buffers,
            children_table,
            n_children,
        })
    }

    /// The number of raw serialized buffers.
    pub fn nbuffers(&self) -> usize {
        self.n_buffers
    }

    /// The `i`th raw serialized buffer, resident in guest memory.
    ///
    /// The returned slice borrows guest linear memory directly; it is valid for the duration of
    /// the decode call.
    pub fn buffer(&self, i: usize) -> GuestResult<&'a [u8]> {
        if i >= self.n_buffers {
            return Err(GuestError::new("buffer index out of bounds"));
        }
        let entry = self.buffers_table + i * 8;
        let ptr = read_u32(self.input, entry);
        let len = read_u32(self.input, entry + 4) as usize;
        // SAFETY: the host wrote this buffer into guest memory before calling vx_decode.
        Ok(unsafe { core::slice::from_raw_parts(ptr as *const u8, len) })
    }

    /// The number of host-decoded children.
    pub fn nchildren(&self) -> usize {
        self.n_children
    }

    /// The `i`th host-decoded child, as a typed view over its Arrow C structs.
    pub fn child(&self, i: usize) -> GuestResult<ChildView> {
        if i >= self.n_children {
            return Err(GuestError::new("child index out of bounds"));
        }
        let entry = self.children_table + i * 8;
        let array_ptr = read_u32(self.input, entry);
        let schema_ptr = read_u32(self.input, entry + 4);
        read_child(array_ptr, schema_ptr)
    }
}
