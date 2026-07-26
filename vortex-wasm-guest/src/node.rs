// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Views over the frames the host passes to `vx_children` and `vx_decode` (see [`crate::abi`]).

use alloc::vec::Vec;

use crate::abi::child_descriptor;
use crate::abi::child_entry;
use crate::abi::children_frame;
use crate::abi::decode_frame;
use crate::data::ChildView;
use crate::dtype::DTypeExpr;
use crate::dtype::DTypeView;
use crate::error::GuestError;
use crate::error::GuestResult;

/// How the guest intends to use a serialized child.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ChildMode {
    /// The guest reads this child's element bytes. The host canonicalizes it and copies it into
    /// guest memory, so its dtype must be one the guest can read.
    Values,
    /// The guest only *names* this child in its plan (see [`PlanBuilder::child`](crate::plan::PlanBuilder::child)).
    /// The host resolves it lazily in its own encoding and never canonicalizes or copies it —
    /// so a referenced child may have **any** dtype, including nested ones the guest could
    /// neither read nor reproduce.
    Reference,
}

/// One serialized child's dtype, logical length, and access mode.
pub struct ChildSpec {
    /// The child's dtype, as a literal or a derivation of the parent's.
    pub dtype: DTypeExpr,
    /// The child's logical element count.
    pub len: u64,
    /// Whether the guest reads this child or merely references it.
    pub mode: ChildMode,
}

impl ChildSpec {
    /// A child the guest will read.
    pub fn values(dtype: DTypeExpr, len: u64) -> Self {
        Self {
            dtype,
            len,
            mode: ChildMode::Values,
        }
    }

    /// A child the guest will only name in its plan.
    pub fn reference(dtype: DTypeExpr, len: u64) -> Self {
        Self {
            dtype,
            len,
            mode: ChildMode::Reference,
        }
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
    /// The number of serialized children present on the node.
    pub n_children: usize,
    /// The encoding's metadata bytes (parse with [`crate::proto`]).
    pub metadata: &'a [u8],
    dtype: &'a [u8],
}

impl<'a> NodeHeader<'a> {
    /// The node's dtype, in full.
    pub fn dtype(&self) -> GuestResult<DTypeView<'a>> {
        DTypeView::new(self.dtype)
    }
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
        let dtype_len = read_u32(input, children_frame::DTYPE_LEN) as usize;
        let metadata_len = read_u32(input, children_frame::METADATA_LEN) as usize;
        let metadata_start = children_frame::HEADER + dtype_len;
        if input.len() < metadata_start + metadata_len {
            return Err(GuestError::new("children frame out of bounds"));
        }
        Ok(Self {
            len,
            nullable: flags & crate::abi::FLAG_NULLABLE != 0,
            n_children,
            metadata: &input[metadata_start..metadata_start + metadata_len],
            dtype: &input[children_frame::HEADER..metadata_start],
        })
    }
}

/// Serialize child specs into the `vx_children` output and return its pointer.
pub fn write_child_specs(specs: &[ChildSpec]) -> i32 {
    let mut out = Vec::with_capacity(4 + specs.len() * (child_descriptor::HEADER + 2));
    out.extend_from_slice(&(specs.len() as u32).to_le_bytes());
    for spec in specs {
        let mode = match spec.mode {
            ChildMode::Values => child_descriptor::MODE_VALUES,
            ChildMode::Reference => child_descriptor::MODE_REFERENCE,
        };
        let dtype = spec.dtype.as_bytes();
        out.extend_from_slice(&[mode, 0, 0, 0]);
        out.extend_from_slice(&(dtype.len() as u32).to_le_bytes());
        out.extend_from_slice(&spec.len.to_le_bytes());
        out.extend_from_slice(dtype);
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
    /// The encoding's metadata bytes.
    pub metadata: &'a [u8],
    dtype: &'a [u8],
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
        let dtype_len = read_u32(input, decode_frame::DTYPE_LEN) as usize;
        let metadata_len = read_u32(input, decode_frame::METADATA_LEN) as usize;
        let n_buffers = read_u32(input, decode_frame::N_BUFFERS) as usize;
        let n_children = read_u32(input, decode_frame::N_CHILDREN) as usize;

        let metadata_start = decode_frame::HEADER + dtype_len;
        let buffers_table = metadata_start + metadata_len;
        let children_table = buffers_table + n_buffers * 8;
        if input.len() < children_table + n_children * child_entry::SIZE {
            return Err(GuestError::new("decode frame tables out of bounds"));
        }
        Ok(Self {
            len,
            nullable: flags & crate::abi::FLAG_NULLABLE != 0,
            metadata: &input[metadata_start..buffers_table],
            dtype: &input[decode_frame::HEADER..metadata_start],
            input,
            buffers_table,
            n_buffers,
            children_table,
            n_children,
        })
    }

    /// The node's dtype, in full.
    ///
    /// The real type, not a coarse kind tag: a kernel can walk a struct's fields, read a decimal's
    /// precision, or look through an extension type to its storage.
    pub fn dtype(&self) -> GuestResult<DTypeView<'a>> {
        DTypeView::new(self.dtype)
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

    /// The `i`th host-supplied `Values` child, as a typed view.
    pub fn child(&self, i: usize) -> GuestResult<ChildView> {
        if i >= self.n_children {
            return Err(GuestError::new("child index out of bounds"));
        }
        let start = self.children_table + i * child_entry::SIZE;
        crate::data::read_child(&self.input[start..start + child_entry::SIZE])
    }
}
