// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Parsing, validating, and evaluating a kernel's decode **plan**.
//!
//! See the guest SDK's `plan` module for what a plan is and why. This is the side that has to
//! treat one as hostile: a plan arrives from a WebAssembly module that arrived inside a file, and
//! nothing about it may be assumed — not that indices are in range, not that lengths agree, not
//! that a node's operands exist.
//!
//! Three properties make that tractable:
//!
//! - **The plan is flat.** Nodes live in an array and reference each other by index, so evaluation
//!   is a `for` loop over a slot table. There is no recursion, so there is no depth to bound and no
//!   way to overflow the stack.
//! - **References point backwards.** A node may only name a lower index. Cycles are therefore
//!   unrepresentable rather than merely rejected, and a single forward pass always has its
//!   operands ready.
//! - **Every op is one `vortex-array` constructor.** Validation is per-op and local, and evaluation
//!   cannot reach an encoding the reader might not have — which matters, because a kernel exists
//!   precisely when the reader is missing an encoding.
//!
//! What is left to check is arithmetic: that lengths agree, that indices are in bounds, and that
//! the work the plan describes is bounded. [`Plan::evaluate`] does that per node, and the checks
//! are deliberately independent of anything the file claims about itself.

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::convert::ArrayDescriptor;
use crate::dtype;

/// Plan opcodes, mirroring the guest SDK's `abi::plan_op`.
const OP_MATERIALIZED: u8 = 0;
const OP_CHILD: u8 = 1;
const OP_TAKE: u8 = 2;
const OP_SLICE: u8 = 3;
const OP_CONCAT: u8 = 4;
const OP_CONSTANT: u8 = 5;
const OP_SET_VALIDITY: u8 = 6;

const PLAN_HEADER: usize = 16;
const NODE_SIZE: usize = 16;

/// Cap on the nodes in one plan.
///
/// Generous next to any real decoder — the widest kernel here uses four — and small enough that a
/// pathological plan cannot turn one node decode into unbounded host work.
const MAX_NODES: usize = 1024;

/// Cap on the operands of a single `Concat`.
const MAX_CONCAT_PARTS: usize = 1024;

/// Multiple of the node's own length that the plan's intermediate results may sum to.
///
/// Plan operations build lazy arrays, so an oversized intermediate costs references rather than
/// bytes up front — but the cost is real once the scan canonicalizes, and a plan that concatenates
/// the same child a thousand times is not something a decoder does. This bounds the *described*
/// work, not the work done here.
const OUTPUT_BUDGET_FACTOR: usize = 64;

/// One decoded plan node.
#[derive(Debug, Clone, Copy)]
struct Node {
    op: u8,
    a: u32,
    b: u32,
}

/// Everything the evaluator needs from the node being decoded.
pub(crate) struct PlanContext<'a> {
    /// The node's dtype.
    pub dtype: &'a DType,
    /// The node's logical length.
    pub len: usize,
    /// Resolves serialized child `slot` to an array, in its own encoding.
    ///
    /// Called at most once per referenced slot, and only for slots a plan actually names — a child
    /// no node mentions is never resolved, which is what keeps `Reference` children free.
    pub child: &'a mut dyn FnMut(usize) -> VortexResult<ArrayRef>,
}

/// A parsed, structurally valid plan.
///
/// Construction checks everything that can be checked without arrays in hand: node count, opcode
/// validity, operand indices pointing backwards, and aux payloads being present and in range.
/// Anything involving a length or a dtype is checked in [`Self::evaluate`], where the arrays exist.
pub(crate) struct Plan {
    nodes: Vec<Node>,
    root: usize,
    aux: Vec<u8>,
}

impl Plan {
    /// Parse a plan frame starting at `offset` in guest memory.
    pub(crate) fn parse(mem: &[u8], offset: usize) -> VortexResult<Self> {
        let header = mem
            .get(offset..offset + PLAN_HEADER)
            .ok_or_else(|| vortex_err!("truncated plan header"))?;
        let n_nodes = usize::try_from(read_u32(header, 0))?;
        let root = usize::try_from(read_u32(header, 4))?;
        let aux_len = usize::try_from(read_u32(header, 8))?;

        vortex_ensure!(n_nodes > 0, "a kernel returned an empty plan");
        vortex_ensure!(
            n_nodes <= MAX_NODES,
            "a kernel returned a plan with {n_nodes} nodes, more than the {MAX_NODES} allowed"
        );
        vortex_ensure!(
            root < n_nodes,
            "a plan's root node {root} is out of range for {n_nodes} nodes"
        );

        let nodes_start = offset + PLAN_HEADER;
        let aux_start = nodes_start + n_nodes * NODE_SIZE;
        let node_bytes = mem
            .get(nodes_start..aux_start)
            .ok_or_else(|| vortex_err!("truncated plan node array"))?;
        let aux = mem
            .get(aux_start..aux_start + aux_len)
            .ok_or_else(|| vortex_err!("truncated plan aux blob"))?
            .to_vec();

        let mut nodes = Vec::with_capacity(n_nodes);
        for i in 0..n_nodes {
            let raw = &node_bytes[i * NODE_SIZE..(i + 1) * NODE_SIZE];
            let node = Node {
                op: raw[0],
                a: read_u32(raw, 4),
                b: read_u32(raw, 8),
            };
            // Operands that name other nodes must point strictly backwards. This is the invariant
            // that makes the plan a DAG and lets evaluation be a single forward pass.
            let backrefs: &[u32] = match node.op {
                OP_TAKE | OP_SET_VALIDITY => &[node.a, node.b],
                OP_SLICE => &[node.a],
                OP_MATERIALIZED | OP_CHILD | OP_CONCAT | OP_CONSTANT => &[],
                other => vortex_bail!("a kernel returned an unknown plan opcode {other}"),
            };
            for &reference in backrefs {
                vortex_ensure!(
                    usize::try_from(reference)? < i,
                    "plan node {i} references node {reference}, which is not strictly before it"
                );
            }
            nodes.push(node);
        }

        let plan = Self { nodes, root, aux };
        // Concat's operand list lives in the aux blob, so its back-references are checked here,
        // once the blob is available.
        for (i, node) in plan.nodes.iter().enumerate() {
            if node.op == OP_CONCAT {
                for part in plan.concat_parts(node)? {
                    vortex_ensure!(
                        part < i,
                        "plan node {i} concatenates node {part}, which is not strictly before it"
                    );
                }
            }
        }
        Ok(plan)
    }

    fn aux_slice(&self, offset: u32, len: usize) -> VortexResult<&[u8]> {
        let offset = usize::try_from(offset)?;
        self.aux
            .get(offset..offset + len)
            .ok_or_else(|| vortex_err!("plan aux payload out of range"))
    }

    fn concat_parts(&self, node: &Node) -> VortexResult<Vec<usize>> {
        let count = usize::try_from(read_u32(self.aux_slice(node.a, 4)?, 0))?;
        vortex_ensure!(
            count <= MAX_CONCAT_PARTS,
            "a plan concatenates {count} parts, more than the {MAX_CONCAT_PARTS} allowed"
        );
        let bytes = self.aux_slice(node.a + 4, count * 4)?;
        (0..count)
            .map(|i| Ok(usize::try_from(read_u32(bytes, i * 4))?))
            .collect()
    }

    /// Evaluate the plan, returning the root array.
    pub(crate) fn evaluate(
        &self,
        ctx: &mut PlanContext<'_>,
        exec: &mut ExecutionCtx,
        guest_mem: &[u8],
    ) -> VortexResult<ArrayRef> {
        let budget = ctx
            .len
            .saturating_mul(OUTPUT_BUDGET_FACTOR)
            .max(OUTPUT_BUDGET_FACTOR);
        let mut spent = 0usize;
        let mut slots: Vec<ArrayRef> = Vec::with_capacity(self.nodes.len());

        for (i, node) in self.nodes.iter().enumerate() {
            let array = self.evaluate_node(node, &slots, ctx, exec, guest_mem)?;
            spent = spent.saturating_add(array.len());
            vortex_ensure!(
                spent <= budget,
                "a plan's intermediate results total {spent} rows for a node of {} rows, over the \
                 {OUTPUT_BUDGET_FACTOR}x budget",
                ctx.len
            );
            debug_assert_eq!(slots.len(), i);
            slots.push(array);
        }

        Ok(slots.swap_remove(self.root))
    }

    fn evaluate_node(
        &self,
        node: &Node,
        slots: &[ArrayRef],
        ctx: &mut PlanContext<'_>,
        exec: &mut ExecutionCtx,
        guest_mem: &[u8],
    ) -> VortexResult<ArrayRef> {
        // `parse` proved every back-reference is a lower index than the node being evaluated, so
        // each operand is already in `slots`.
        let operand = |index: u32| -> VortexResult<ArrayRef> {
            slots
                .get(usize::try_from(index)?)
                .cloned()
                .ok_or_else(|| vortex_err!("plan operand {index} has not been evaluated"))
        };

        match node.op {
            OP_MATERIALIZED => {
                // The descriptor bytes live in the aux blob; the buffer pointers inside it are
                // absolute guest offsets, resolved against `guest_mem`. A descriptor alone is
                // dtype-ambiguous — the same view layout is both Utf8 and Binary — so the guest
                // states the type and the array is checked against it.
                let offset = usize::try_from(node.a)?;
                let (dtype, consumed) = dtype::decode(
                    self.aux
                        .get(offset..)
                        .ok_or_else(|| vortex_err!("plan materialized node out of range"))?,
                    ctx.dtype,
                )?;
                let (descriptor, _) = ArrayDescriptor::parse(&self.aux, offset + consumed)?;
                descriptor.build(guest_mem, &dtype)
            }
            OP_CHILD => (ctx.child)(usize::try_from(node.a)?),
            OP_TAKE => {
                let base = operand(node.a)?;
                let indices = operand(node.b)?;
                validate_indices(&indices, base.len(), exec)?;
                base.take(indices)
            }
            OP_SLICE => {
                let base = operand(node.a)?;
                let range = self.slice_range(node)?;
                vortex_ensure!(
                    range.start <= range.end && range.end <= base.len(),
                    "a plan slices {}..{} of a {}-row array",
                    range.start,
                    range.end,
                    base.len()
                );
                base.slice(range)
            }
            OP_CONCAT => {
                let parts =
                    self.concat_parts(node)?
                        .into_iter()
                        .map(|part| {
                            slots.get(part).cloned().ok_or_else(|| {
                                vortex_err!("plan concat operand {part} is unevaluated")
                            })
                        })
                        .collect::<VortexResult<Vec<_>>>()?;
                vortex_ensure!(!parts.is_empty(), "a plan concatenates zero arrays");
                let dtype = parts[0].dtype().clone();
                for part in &parts[1..] {
                    vortex_ensure!(
                        part.dtype() == &dtype,
                        "a plan concatenates arrays of differing dtypes, {dtype} and {}",
                        part.dtype()
                    );
                }
                Ok(ChunkedArray::try_new(parts, dtype)?.into_array())
            }
            OP_CONSTANT => {
                let scalar = self.constant_scalar(node, ctx.dtype)?;
                let len = usize::try_from(u64::from_le_bytes(
                    self.aux_slice(node.b, 8)?
                        .try_into()
                        .map_err(|_| vortex_err!("truncated plan constant length"))?,
                ))?;
                vortex_ensure!(
                    len <= ctx.len.saturating_mul(OUTPUT_BUDGET_FACTOR).max(1),
                    "a plan builds a {len}-row constant for a node of {} rows",
                    ctx.len
                );
                Ok(ConstantArray::new(scalar, len).into_array())
            }
            OP_SET_VALIDITY => {
                let base = operand(node.a)?;
                let mask = operand(node.b)?;
                vortex_ensure!(
                    mask.len() == base.len(),
                    "a plan applies a {}-row validity mask to a {}-row array",
                    mask.len(),
                    base.len()
                );
                vortex_ensure!(
                    matches!(mask.dtype(), DType::Bool(Nullability::NonNullable)),
                    "a plan's validity mask must be a non-nullable boolean, got {}",
                    mask.dtype()
                );
                Ok(MaskedArray::try_new(base, Validity::Array(mask))?.into_array())
            }
            other => vortex_bail!("a kernel returned an unknown plan opcode {other}"),
        }
    }

    fn slice_range(&self, node: &Node) -> VortexResult<Range<usize>> {
        let bytes = self.aux_slice(node.b, 16)?;
        let start = u64::from_le_bytes(
            bytes[..8]
                .try_into()
                .map_err(|_| vortex_err!("truncated plan slice range"))?,
        );
        let stop = u64::from_le_bytes(
            bytes[8..]
                .try_into()
                .map_err(|_| vortex_err!("truncated plan slice range"))?,
        );
        Ok(usize::try_from(start)?..usize::try_from(stop)?)
    }

    fn constant_scalar(&self, node: &Node, parent: &DType) -> VortexResult<Scalar> {
        let bytes = self
            .aux
            .get(usize::try_from(node.a)?..)
            .ok_or_else(|| vortex_err!("plan constant out of range"))?;
        let (dtype, consumed) = dtype::decode(bytes, parent)?;
        let is_valid = *bytes
            .get(consumed)
            .ok_or_else(|| vortex_err!("truncated plan constant"))?;
        if is_valid == 0 {
            return Ok(Scalar::null(dtype));
        }
        let body = &bytes[consumed + 1..];
        match &dtype {
            DType::Bool(_) => {
                let value = *body
                    .first()
                    .ok_or_else(|| vortex_err!("truncated boolean constant"))?;
                Ok(Scalar::bool(value != 0, dtype.nullability()))
            }
            DType::Primitive(ptype, _) => {
                let (len, n) = read_varint(body, 0)?;
                let width = usize::try_from(len)?;
                vortex_ensure!(
                    width == ptype.byte_width(),
                    "a plan's {ptype} constant carries {width} value bytes"
                );
                let value = body
                    .get(n..n + width)
                    .ok_or_else(|| vortex_err!("truncated primitive constant"))?;
                Ok(primitive_scalar(*ptype, value, dtype.nullability())?)
            }
            other => {
                vortex_bail!("a plan's constant must be null, boolean, or primitive, got {other}")
            }
        }
    }
}

fn primitive_scalar(ptype: PType, value: &[u8], nullability: Nullability) -> VortexResult<Scalar> {
    macro_rules! read {
        ($ty:ty) => {{
            let bytes: [u8; std::mem::size_of::<$ty>()] = value
                .try_into()
                .map_err(|_| vortex_err!("bad {ptype} constant width"))?;
            <$ty>::from_le_bytes(bytes)
        }};
    }
    Ok(match ptype {
        PType::U8 => Scalar::primitive(read!(u8), nullability),
        PType::U16 => Scalar::primitive(read!(u16), nullability),
        PType::U32 => Scalar::primitive(read!(u32), nullability),
        PType::U64 => Scalar::primitive(read!(u64), nullability),
        PType::I8 => Scalar::primitive(read!(i8), nullability),
        PType::I16 => Scalar::primitive(read!(i16), nullability),
        PType::I32 => Scalar::primitive(read!(i32), nullability),
        PType::I64 => Scalar::primitive(read!(i64), nullability),
        PType::F32 => Scalar::primitive(read!(f32), nullability),
        PType::F64 => Scalar::primitive(read!(f64), nullability),
        PType::F16 => vortex_bail!("f16 constants are not supported in a plan"),
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

fn read_varint(bytes: &[u8], mut offset: usize) -> VortexResult<(u64, usize)> {
    let start = offset;
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *bytes
            .get(offset)
            .ok_or_else(|| vortex_err!("truncated varint"))?;
        offset += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, offset - start));
        }
        shift += 7;
        vortex_ensure!(shift < 64, "varint overflow");
    }
}

/// Validate gather indices produced by an untrusted kernel.
///
/// Deliberately recomputes the maximum over the materialized index buffer rather than consulting
/// `Stat::Max`: statistics are themselves attacker-controlled file data, so they must never be
/// load-bearing for a safety property. This matters because `ArrayRef::take` builds a `DictArray`,
/// whose constructor checks only that the codes are integral — an out-of-range index would
/// otherwise surface later as a panic or garbage.
pub(crate) fn validate_indices(
    indices: &ArrayRef,
    values_len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let DType::Primitive(ptype, _) = indices.dtype() else {
        vortex_bail!(
            "wasm gather indices must be primitive, got {}",
            indices.dtype()
        );
    };
    vortex_ensure!(
        ptype.is_unsigned_int(),
        "wasm gather indices must be an unsigned integer, got {ptype}"
    );
    vortex_ensure!(
        !indices.dtype().is_nullable(),
        "wasm gather indices must be non-nullable"
    );

    let primitive = indices.clone().execute::<Canonical>(ctx)?.into_primitive();
    let in_bounds = match_each_unsigned_integer_ptype!(primitive.ptype(), |P| {
        primitive
            .as_slice::<P>()
            .iter()
            .all(|&i| (i as u128) < values_len as u128)
    });
    vortex_ensure!(
        in_bounds,
        "wasm gather index out of bounds for a child of length {values_len}"
    );
    Ok(())
}

#[cfg(test)]
// Test plan frames are a few dozen bytes, so the offsets in `RawPlan` cannot overflow a u32.
#[expect(clippy::cast_possible_truncation)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;

    use super::*;

    /// Builds plan frames byte by byte, including ones no honest guest would emit.
    ///
    /// The guest SDK's builder cannot express a cycle or a dangling reference — that is the point
    /// of its `NodeId` — so testing that the host rejects them means writing the bytes directly.
    #[derive(Default)]
    struct RawPlan {
        nodes: Vec<u8>,
        aux: Vec<u8>,
        count: u32,
    }

    impl RawPlan {
        fn push(&mut self, op: u8, a: u32, b: u32) -> u32 {
            self.nodes.push(op);
            self.nodes.extend_from_slice(&[0, 0, 0]);
            self.nodes.extend_from_slice(&a.to_le_bytes());
            self.nodes.extend_from_slice(&b.to_le_bytes());
            self.nodes.extend_from_slice(&0u32.to_le_bytes());
            self.count += 1;
            self.count - 1
        }

        fn aux(&mut self, bytes: &[u8]) -> u32 {
            let offset = self.aux.len() as u32;
            self.aux.extend_from_slice(bytes);
            offset
        }

        fn child(&mut self, slot: u32) -> u32 {
            self.push(OP_CHILD, slot, 0)
        }

        fn take(&mut self, base: u32, indices: u32) -> u32 {
            self.push(OP_TAKE, base, indices)
        }

        fn slice(&mut self, base: u32, start: u64, stop: u64) -> u32 {
            let mut range = [0u8; 16];
            range[..8].copy_from_slice(&start.to_le_bytes());
            range[8..].copy_from_slice(&stop.to_le_bytes());
            let offset = self.aux(&range);
            self.push(OP_SLICE, base, offset)
        }

        fn concat(&mut self, parts: &[u32]) -> u32 {
            let mut encoded = (parts.len() as u32).to_le_bytes().to_vec();
            for part in parts {
                encoded.extend_from_slice(&part.to_le_bytes());
            }
            let offset = self.aux(&encoded);
            self.push(OP_CONCAT, offset, 0)
        }

        fn set_validity(&mut self, base: u32, mask: u32) -> u32 {
            self.push(OP_SET_VALIDITY, base, mask)
        }

        /// Serialize into a buffer laid out like guest memory.
        fn finish(self, root: u32) -> Vec<u8> {
            let mut frame = Vec::new();
            frame.extend_from_slice(&self.count.to_le_bytes());
            frame.extend_from_slice(&root.to_le_bytes());
            frame.extend_from_slice(&(self.aux.len() as u32).to_le_bytes());
            frame.extend_from_slice(&0u32.to_le_bytes());
            frame.extend_from_slice(&self.nodes);
            frame.extend_from_slice(&self.aux);
            frame
        }
    }

    fn u32_array(values: impl IntoIterator<Item = u32>) -> ArrayRef {
        let values: Buffer<u32> = values.into_iter().collect();
        PrimitiveArray::new(values, Validity::NonNullable).into_array()
    }

    /// Evaluate a plan whose children come from `children`, for a node of `len` rows.
    fn eval(frame: &[u8], len: usize, children: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
        let plan = Plan::parse(frame, 0)?;
        let mut exec = array_session().create_execution_ctx();
        let mut resolve = |slot: usize| -> VortexResult<ArrayRef> {
            children
                .get(slot)
                .cloned()
                .ok_or_else(|| vortex_err!("no child {slot}"))
        };
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        plan.evaluate(
            &mut PlanContext {
                dtype: &dtype,
                len,
                child: &mut resolve,
            },
            &mut exec,
            frame,
        )
    }

    /// `Plan` has no `Debug`, so `unwrap_err` needs a hand.
    fn unwrap_parse_err(result: VortexResult<Plan>) -> vortex_error::VortexError {
        match result {
            Ok(_) => panic!("expected the plan to be rejected"),
            Err(err) => err,
        }
    }

    fn values() -> ArrayRef {
        PrimitiveArray::new(buffer![10i32, 20, 30, 40], Validity::NonNullable).into_array()
    }

    #[test]
    fn a_take_over_a_child_gathers_it() -> VortexResult<()> {
        let mut plan = RawPlan::default();
        let base = plan.child(0);
        let indices = plan.push(OP_CHILD, 1, 0);
        let root = plan.take(base, indices);
        let frame = plan.finish(root);

        let result = eval(&frame, 5, vec![values(), u32_array([3u32, 0, 0, 1, 2])])?;
        assert_eq!(result.len(), 5);
        let mut exec = array_session().create_execution_ctx();
        let canonical = result.execute::<Canonical>(&mut exec)?.into_primitive();
        assert_eq!(canonical.as_slice::<i32>(), &[40, 10, 10, 20, 30]);
        Ok(())
    }

    #[test]
    fn a_slice_narrows_a_child() -> VortexResult<()> {
        let mut plan = RawPlan::default();
        let base = plan.child(0);
        let root = plan.slice(base, 1, 3);
        let frame = plan.finish(root);

        let result = eval(&frame, 2, vec![values()])?;
        assert_eq!(result.len(), 2);
        Ok(())
    }

    #[test]
    fn a_concat_joins_children() -> VortexResult<()> {
        let mut plan = RawPlan::default();
        let a = plan.child(0);
        let b = plan.child(0);
        let root = plan.concat(&[a, b]);
        let frame = plan.finish(root);

        assert_eq!(eval(&frame, 8, vec![values()])?.len(), 8);
        Ok(())
    }

    /// The composition the flat encoding buys: a take over a concat, which is how
    /// `PlanBuilder::patch` is expressed without a host patch primitive.
    #[test]
    fn nodes_compose_and_a_shared_node_is_evaluated_once() -> VortexResult<()> {
        let mut plan = RawPlan::default();
        let base = plan.child(0);
        let patch_values = plan.child(1);
        let combined = plan.concat(&[base, patch_values]);
        // Rows 0 and 2 keep their base values; rows 1 and 3 come from the patch child.
        let indices = plan.child(2);
        let root = plan.take(combined, indices);
        let frame = plan.finish(root);

        let result = eval(
            &frame,
            4,
            vec![
                values(),
                PrimitiveArray::new(buffer![99i32, 98], Validity::NonNullable).into_array(),
                u32_array([0u32, 4, 2, 5]),
            ],
        )?;
        let mut exec = array_session().create_execution_ctx();
        let canonical = result.execute::<Canonical>(&mut exec)?.into_primitive();
        assert_eq!(canonical.as_slice::<i32>(), &[10, 99, 30, 98]);
        Ok(())
    }

    /// The core structural invariant. A forward reference is what a cycle would have to be built
    /// from, so rejecting it at parse time makes cycles unrepresentable rather than merely caught.
    #[test]
    fn a_forward_reference_is_rejected_at_parse_time() {
        let mut plan = RawPlan::default();
        // Node 0 takes from node 1, which does not exist yet.
        plan.take(1, 2);
        plan.child(0);
        plan.child(1);
        let frame = plan.finish(0);

        let err = unwrap_parse_err(Plan::parse(&frame, 0)).to_string();
        assert!(err.contains("not strictly before it"), "{err}");
    }

    #[test]
    fn a_self_reference_is_rejected() {
        let mut plan = RawPlan::default();
        plan.take(0, 0);
        let frame = plan.finish(0);
        assert!(Plan::parse(&frame, 0).is_err());
    }

    /// Concat's operands live in the aux blob rather than the node, so they need the same check.
    #[test]
    fn a_forward_reference_inside_a_concat_is_rejected() {
        let mut plan = RawPlan::default();
        let a = plan.child(0);
        let root = plan.concat(&[a, 99]);
        let frame = plan.finish(root);
        assert!(Plan::parse(&frame, 0).is_err());
    }

    #[test]
    fn an_out_of_range_root_is_rejected() {
        let mut plan = RawPlan::default();
        plan.child(0);
        let frame = plan.finish(7);
        assert!(Plan::parse(&frame, 0).is_err());
    }

    #[test]
    fn an_empty_plan_is_rejected() {
        let frame = RawPlan::default().finish(0);
        assert!(Plan::parse(&frame, 0).is_err());
    }

    #[test]
    fn an_unknown_opcode_is_rejected() {
        let mut plan = RawPlan::default();
        plan.push(200, 0, 0);
        let frame = plan.finish(0);
        assert!(Plan::parse(&frame, 0).is_err());
    }

    #[test]
    fn a_truncated_frame_is_rejected() {
        let mut plan = RawPlan::default();
        let a = plan.child(0);
        let frame = plan.finish(a);
        for cut in 0..frame.len() {
            // Any prefix is either rejected or parses to something consistent; none may panic.
            drop(Plan::parse(&frame[..cut], 0));
        }
        // A header promising more nodes than the frame carries.
        let mut lying = frame;
        lying[0..4].copy_from_slice(&64u32.to_le_bytes());
        assert!(Plan::parse(&lying, 0).is_err());
    }

    #[test]
    fn too_many_nodes_are_rejected() {
        let mut frame = Vec::new();
        frame.extend_from_slice(&((MAX_NODES + 1) as u32).to_le_bytes());
        frame.extend_from_slice(&0u32.to_le_bytes());
        frame.extend_from_slice(&0u32.to_le_bytes());
        frame.extend_from_slice(&0u32.to_le_bytes());
        frame.resize(PLAN_HEADER + (MAX_NODES + 1) * NODE_SIZE, 0);
        let err = unwrap_parse_err(Plan::parse(&frame, 0)).to_string();
        assert!(err.contains("more than the"), "{err}");
    }

    /// Out-of-range gather indices are the classic exploit: `take` builds a `DictArray`, whose
    /// constructor only checks that codes are integral, so an unchecked index surfaces later as a
    /// panic or as data from beyond the child.
    #[test]
    fn an_out_of_bounds_gather_index_is_rejected() {
        let mut plan = RawPlan::default();
        let base = plan.child(0);
        let indices = plan.child(1);
        let root = plan.take(base, indices);
        let frame = plan.finish(root);

        // The child has 4 rows; index 4 is one past the end.
        let err = eval(&frame, 1, vec![values(), u32_array([4u32])])
            .unwrap_err()
            .to_string();
        assert!(err.contains("out of bounds"), "{err}");
    }

    #[test]
    fn signed_nullable_and_non_integer_indices_are_rejected() {
        let mut plan = RawPlan::default();
        let base = plan.child(0);
        let indices = plan.child(1);
        let root = plan.take(base, indices);
        let frame = plan.finish(root);

        // Signed.
        let signed = PrimitiveArray::new(buffer![0i32], Validity::NonNullable).into_array();
        assert!(eval(&frame, 1, vec![values(), signed]).is_err());

        // Nullable.
        let nullable = PrimitiveArray::new(buffer![0u32], Validity::AllValid).into_array();
        assert!(eval(&frame, 1, vec![values(), nullable]).is_err());
    }

    #[test]
    fn an_out_of_range_slice_is_rejected() {
        let mut plan = RawPlan::default();
        let base = plan.child(0);
        let root = plan.slice(base, 2, 99);
        let frame = plan.finish(root);
        let err = eval(&frame, 4, vec![values()]).unwrap_err().to_string();
        assert!(err.contains("slices"), "{err}");

        // And an inverted range.
        let mut plan = RawPlan::default();
        let base = plan.child(0);
        let root = plan.slice(base, 3, 1);
        let frame = plan.finish(root);
        assert!(eval(&frame, 2, vec![values()]).is_err());
    }

    #[test]
    fn concatenating_mismatched_dtypes_is_rejected() {
        let mut plan = RawPlan::default();
        let a = plan.child(0);
        let b = plan.child(1);
        let root = plan.concat(&[a, b]);
        let frame = plan.finish(root);

        let err = eval(&frame, 8, vec![values(), u32_array([1u32, 2, 3, 4])])
            .unwrap_err()
            .to_string();
        assert!(err.contains("differing dtypes"), "{err}");
    }

    #[test]
    fn a_mismatched_validity_mask_is_rejected() {
        let mut plan = RawPlan::default();
        let base = plan.child(0);
        let mask = plan.child(1);
        let root = plan.set_validity(base, mask);
        let frame = plan.finish(root);

        // Wrong length.
        let short = BoolArray::from_iter([true, false]).into_array();
        assert!(eval(&frame, 4, vec![values(), short]).is_err());

        // Not a boolean.
        let err = eval(&frame, 4, vec![values(), u32_array([1u32, 1, 1, 1])])
            .unwrap_err()
            .to_string();
        assert!(err.contains("non-nullable boolean"), "{err}");
    }

    /// A plan is cheap to write and can describe expensive work: 1024 concatenations of a child
    /// are 1024 references here but a very large array to the scan that eventually reads it.
    #[test]
    fn a_plan_describing_unbounded_output_is_rejected() {
        let mut plan = RawPlan::default();
        // Each round doubles the length, so a handful of nodes describe an enormous array.
        let mut current = plan.child(0);
        for _ in 0..12 {
            current = plan.concat(&[current, current]);
        }
        let frame = plan.finish(current);

        let err = eval(&frame, 4, vec![values()]).unwrap_err().to_string();
        assert!(err.contains("budget"), "{err}");
    }

    #[test]
    fn naming_a_child_that_does_not_exist_is_rejected() {
        let mut plan = RawPlan::default();
        let root = plan.child(9);
        let frame = plan.finish(root);
        assert!(eval(&frame, 4, vec![values()]).is_err());
    }
}
