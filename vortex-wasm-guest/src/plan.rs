// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The decode **plan**: what a kernel returns instead of an array.
//!
//! A decoder does not have to produce bytes. Most compressed encodings are not *computing* new
//! values at all — run-end repeats a child, dict gathers one, sparse overlays exceptions on a
//! constant. For those, materializing the output inside the sandbox is pure waste: the kernel
//! copies a child in, rebuilds it element by element, and copies it back out, when the host could
//! have expressed the same array as a handful of nodes over data that never moved.
//!
//! So a kernel returns a plan — a small tree of operations over the node's children — and the host
//! evaluates it. The operations are deliberately ones Vortex already has lazy arrays for, so
//! evaluation allocates nothing: a `Take` becomes a `DictArray`, a `Slice` a `SliceArray`, a
//! `Concat` a `ChunkedArray`. Nothing is canonicalized until the scan actually asks for values,
//! and a plan over `Child` nodes touches no element data at all.
//!
//! The vocabulary is *closed on purpose*. Every operation maps to one `vortex-array` constructor,
//! which is what keeps it available: a kernel exists precisely because the reader lacks that
//! encoding, so a plan must never depend on an encoding crate that might be missing.
//!
//! # Wire format
//!
//! The plan is a flat, **postorder** array of fixed-size nodes, not a nested tree:
//!
//! ```text
//! [u32 n_nodes][u32 root][u32 aux_len][u32 reserved]
//! [node × n_nodes]          node = [u8 op][u8 flags][u16 _pad][u32 a][u32 b][u32 c]
//! [aux bytes]
//! ```
//!
//! Operands `a`/`b`/`c` are node indices, child slots, or offsets into the trailing `aux` blob for
//! payloads that do not fit (64-bit ranges, scalars, operand lists).
//!
//! Flat and postorder is the whole safety argument. A node may only reference nodes at a **lower
//! index**, which makes the plan acyclic by construction and lets the host evaluate it in a single
//! forward pass over an array — no recursion, so no depth to bound and no stack to overflow. A
//! nested encoding would have handed an untrusted file a recursive descent parser.
//!
//! Because references are by index, a node may be used more than once and is evaluated once; the
//! plan is a DAG, and sharing a subplan costs nothing.
//!
//! # Building
//!
//! [`PlanBuilder`] hands back a [`NodeId`] for each node, which is the only way to refer to it:
//!
//! ```ignore
//! let mut plan = PlanBuilder::new();
//! let values = plan.child(VALUES);
//! let indices = plan.materialized(DTypeExpr::primitive(PType::U32, false), run_indices);
//! let root = plan.take(values, indices);
//! plan.finish(root)
//! ```

use alloc::vec::Vec;

use crate::abi::PType;
use crate::abi::plan_op;
use crate::data::Decoded;
use crate::data::write_array_into;
use crate::dtype::DTypeExpr;
use crate::dtype::write_varint;
use crate::error::GuestError;
use crate::error::GuestResult;
use crate::host::alloc_bytes;

/// Size of one encoded plan node.
pub const NODE_SIZE: usize = 16;

/// A node in a [`PlanBuilder`], and the only way to reference one.
///
/// Opaque and non-forgeable: because ids only come from the builder, a plan cannot name a node
/// that does not exist or one that has not been emitted yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeId(u32);

/// Builds a decode plan.
///
/// Nodes are appended in postorder — an operand must be built before the operation that consumes
/// it, which the [`NodeId`] type enforces for free.
pub struct PlanBuilder {
    nodes: Vec<u8>,
    aux: Vec<u8>,
    count: u32,
}

impl Default for PlanBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanBuilder {
    /// An empty plan.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            aux: Vec::new(),
            count: 0,
        }
    }

    fn push(&mut self, op: u8, arg0: u32, arg1: u32) -> NodeId {
        self.nodes.push(op);
        self.nodes.push(0);
        self.nodes.extend_from_slice(&0u16.to_le_bytes());
        self.nodes.extend_from_slice(&arg0.to_le_bytes());
        self.nodes.extend_from_slice(&arg1.to_le_bytes());
        self.nodes.extend_from_slice(&0u32.to_le_bytes());
        let id = NodeId(self.count);
        self.count += 1;
        id
    }

    fn push_aux(&mut self, bytes: &[u8]) -> u32 {
        let offset = self.aux.len() as u32;
        self.aux.extend_from_slice(bytes);
        offset
    }

    /// The node's serialized child at `slot`, in its own encoding.
    ///
    /// The host resolves it lazily and never canonicalizes or copies it, so the child may have any
    /// dtype — including nested ones the kernel could neither read nor reproduce. The slot must
    /// have been declared [`ChildMode::Reference`](crate::node::ChildMode::Reference).
    pub fn child(&mut self, slot: u16) -> NodeId {
        self.push(plan_op::CHILD, u32::from(slot), 0)
    }

    /// An array the kernel built itself.
    ///
    /// The escape hatch, and the only node that moves bytes: for encodings that genuinely compute
    /// new values (bit-packing, FSST, ALP) there is nothing to re-arrange and this is the point.
    ///
    /// `dtype` is required because the buffer layout alone does not determine the type — the same
    /// view layout is both `Utf8` and `Binary`, and the same 4-byte values are `u32`, `i32`, or
    /// `f32`. Usually [`DTypeExpr::parent`](crate::dtype::DTypeExpr::parent).
    pub fn materialized(&mut self, dtype: DTypeExpr, array: Decoded) -> NodeId {
        let mut payload = Vec::new();
        payload.extend_from_slice(dtype.as_bytes());
        write_array_into(&mut payload, &array);
        let offset = self.push_aux(&payload);
        self.push(plan_op::MATERIALIZED, offset, 0)
    }

    /// `base` gathered by `indices`, one output element per index.
    ///
    /// Evaluated with `ArrayRef::take`. The host re-validates every index against `base`'s length.
    pub fn take(&mut self, base: NodeId, indices: NodeId) -> NodeId {
        self.push(plan_op::TAKE, base.0, indices.0)
    }

    /// The half-open row range `start..stop` of `base`.
    pub fn slice(&mut self, base: NodeId, start: u64, stop: u64) -> NodeId {
        let mut range = [0u8; 16];
        range[..8].copy_from_slice(&start.to_le_bytes());
        range[8..].copy_from_slice(&stop.to_le_bytes());
        let offset = self.push_aux(&range);
        self.push(plan_op::SLICE, base.0, offset)
    }

    /// The concatenation of `parts`, in order.
    ///
    /// All parts must share a dtype. Evaluated as a `ChunkedArray`, so this is a view, not a copy.
    pub fn concat(&mut self, parts: &[NodeId]) -> NodeId {
        let mut encoded = Vec::with_capacity(4 + parts.len() * 4);
        encoded.extend_from_slice(&(parts.len() as u32).to_le_bytes());
        for part in parts {
            encoded.extend_from_slice(&part.0.to_le_bytes());
        }
        let offset = self.push_aux(&encoded);
        self.push(plan_op::CONCAT, offset, 0)
    }

    /// `len` copies of a scalar.
    ///
    /// The scalar is encoded as `[dtype][is_valid][value bytes]`. Only null, boolean, and
    /// primitive scalars are expressible — enough for the fill value of a sparse encoding, which
    /// is what this exists for.
    pub fn constant(&mut self, scalar: Scalar, len: u64) -> NodeId {
        let offset = self.push_aux(&scalar.bytes);
        let len_offset = self.push_aux(&len.to_le_bytes());
        self.push(plan_op::CONSTANT, offset, len_offset)
    }

    /// `base` with its validity replaced by `mask`, a non-nullable boolean array of the same
    /// length where `true` means valid.
    ///
    /// `base` must not itself contain nulls. This is how an encoding that stores values and
    /// validity as separate children reassembles them without touching the values.
    pub fn set_validity(&mut self, base: NodeId, mask: NodeId) -> NodeId {
        self.push(plan_op::SET_VALIDITY, base.0, mask.0)
    }

    /// `base` with the elements at `indices` replaced by `values`.
    ///
    /// Not a primitive: this is `take` over the concatenation of `base` and `values`, which is
    /// exactly equivalent and needs no host operation of its own. `positions` gives, for each
    /// output row, either `None` for "keep the base value" or `Some(j)` for "use `values[j]`".
    ///
    /// The gain over patching inside the sandbox is that the patch values never enter it: they
    /// stay a [`child`](Self::child) reference and may have any dtype, where a kernel that
    /// overwrites its own output must be able to read and rewrite every value it patches.
    pub fn patch(
        &mut self,
        base: NodeId,
        base_len: usize,
        values: NodeId,
        positions: impl IntoIterator<Item = Option<u32>>,
    ) -> GuestResult<NodeId> {
        let base_len_u32 =
            u32::try_from(base_len).map_err(|_| GuestError::new("patched array too long"))?;
        let mut indices: Vec<u8> = Vec::with_capacity(base_len * 4);
        let mut n = 0usize;
        for (row, position) in positions.into_iter().enumerate() {
            let index = match position {
                None => u32::try_from(row).map_err(|_| GuestError::new("row index overflow"))?,
                Some(j) => base_len_u32
                    .checked_add(j)
                    .ok_or(GuestError::new("patch index overflow"))?,
            };
            indices.extend_from_slice(&index.to_le_bytes());
            n += 1;
        }
        let combined = self.concat(&[base, values]);
        let indices = self.materialized(
            DTypeExpr::primitive(PType::U32, false),
            Decoded::Primitive(crate::data::DecodedPrimitive {
                ptype: PType::U32,
                len: n,
                values: indices,
                validity: crate::data::Validity::NonNullable,
            }),
        );
        Ok(self.take(combined, indices))
    }

    /// Finish the plan, returning a pointer to the encoded frame.
    pub fn finish(self, root: NodeId) -> i32 {
        let mut frame = Vec::with_capacity(16 + self.nodes.len() + self.aux.len());
        frame.extend_from_slice(&self.count.to_le_bytes());
        frame.extend_from_slice(&root.0.to_le_bytes());
        frame.extend_from_slice(&(self.aux.len() as u32).to_le_bytes());
        frame.extend_from_slice(&0u32.to_le_bytes());
        frame.extend_from_slice(&self.nodes);
        frame.extend_from_slice(&self.aux);
        alloc_bytes(&frame) as i32
    }
}

/// A scalar value for [`PlanBuilder::constant`].
pub struct Scalar {
    bytes: Vec<u8>,
}

impl Scalar {
    /// A null of the given type expression.
    pub fn null(dtype: DTypeExpr) -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(dtype.as_bytes());
        bytes.push(0);
        Self { bytes }
    }

    /// A boolean.
    pub fn bool(value: bool, nullable: bool) -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(DTypeExpr::bool(nullable).as_bytes());
        bytes.push(1);
        bytes.push(u8::from(value));
        Self { bytes }
    }

    /// A primitive, given its little-endian value bytes.
    pub fn primitive(ptype: PType, value: &[u8], nullable: bool) -> GuestResult<Self> {
        if value.len() != ptype.byte_width() {
            return Err(GuestError::new(
                "scalar value width does not match its ptype",
            ));
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(DTypeExpr::primitive(ptype, nullable).as_bytes());
        bytes.push(1);
        write_varint(&mut bytes, value.len() as u64);
        bytes.extend_from_slice(value);
        Ok(Self { bytes })
    }
}
