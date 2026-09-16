// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Moving arrays across the host/guest boundary in **Vortex's own vocabulary**.
//!
//! One wire format serves both directions. The host writes each of a node's canonical children
//! into guest memory as an array frame, and the kernel returns its output as one. A frame is a
//! shape tag, a buffer table, and inline children — the canonical layouts Vortex already uses, so
//! for primitives and bools the bytes are identical to Arrow's, and strings cross as canonical
//! 16-byte views rather than offsets. Every variant of [`Canonical`] has a shape, so a kernel can
//! read a child of any dtype and produce an output of any dtype.
//!
//! The one exception is `Variant`: Vortex defines no physical canonical layout for variant values
//! (canonicalization is the identity, and storage is whatever encoding the values arrived in), so
//! there is nothing to write. Both directions refuse it with a clear error.
//!
//! Reading is dtype-driven. The host already knows what type the frame must have, so the frame
//! carries no schema; instead every shape is checked against the dtype it is supposed to be, and
//! each array is built through its validating constructor.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListView;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::NullArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::extension::ExtensionArraySlotsExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::map::MapArraySlotsExt;
use vortex_array::arrays::struct_::StructArraySlotsExt;
use vortex_array::arrays::union::UnionArraySlotsExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

/// Shape tags, mirroring the guest SDK's `abi::shape`.
const SHAPE_NULL: u8 = 0;
const SHAPE_BOOL: u8 = 1;
const SHAPE_PRIMITIVE: u8 = 2;
const SHAPE_DECIMAL: u8 = 3;
const SHAPE_VAR_BIN_VIEW: u8 = 4;
const SHAPE_LIST: u8 = 5;
const SHAPE_FIXED_SIZE_LIST: u8 = 6;
const SHAPE_STRUCT: u8 = 7;
const SHAPE_UNION: u8 = 8;
const SHAPE_MAP: u8 = 9;
const SHAPE_EXTENSION: u8 = 10;

/// Validity tags, mirroring the guest SDK's `abi::validity`.
const VALIDITY_NON_NULLABLE: u8 = 0;
const VALIDITY_ALL_VALID: u8 = 1;
const VALIDITY_ALL_INVALID: u8 = 2;
const VALIDITY_BITMAP: u8 = 3;

/// Fixed part of an array frame, mirroring `abi::array_frame::HEADER`.
const FRAME_HEADER: usize = 16;

/// Maximum array nesting either side will handle. On this side it bounds recursion over
/// attacker-controlled bytes, so a few hundred bytes of nested `LIST` tags cannot overflow the
/// host stack.
pub(crate) const MAX_DEPTH: usize = 32;

/// Cap on the total array nodes in one result, so a frame cannot describe unbounded work.
const MAX_NODES: usize = 4096;

/// Cap on the buffers one node may carry, so a malformed frame cannot drive an unbounded number
/// of copies out of guest memory.
const MAX_BUFFERS: usize = 64;

/// A writable view of a WASM guest's linear memory.
pub trait GuestMem {
    /// Allocate `len` bytes in guest memory (8-byte aligned), returning the offset.
    fn alloc(&mut self, len: u32) -> VortexResult<u32>;
    /// Write `bytes` at guest offset `off`.
    fn write(&mut self, off: u32, bytes: &[u8]) -> VortexResult<()>;
}

fn put(mem: &mut dyn GuestMem, bytes: &[u8]) -> VortexResult<u32> {
    let off = mem.alloc(u32::try_from(bytes.len().max(1))?)?;
    mem.write(off, bytes)?;
    Ok(off)
}

/// Materialize a validity into a byte-aligned bitmap, or `None` for the bitmap-free cases.
///
/// `shrink_offset` is what closes the bit-offset hazard: a sliced array's mask can start mid-byte,
/// and handing those bytes over verbatim would silently shift every validity bit.
fn encode_validity(
    validity: &Validity,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(u8, Option<ByteBuffer>)> {
    Ok(match validity {
        Validity::NonNullable => (VALIDITY_NON_NULLABLE, None),
        Validity::AllValid => (VALIDITY_ALL_VALID, None),
        Validity::AllInvalid => (VALIDITY_ALL_INVALID, None),
        Validity::Array(_) => {
            let bits = validity
                .execute_mask(len, ctx)?
                .to_bit_buffer()
                .shrink_offset();
            vortex_ensure!(
                bits.offset() == 0,
                "validity bitmap could not be byte-aligned"
            );
            let bytes = bits.inner().slice(0..len.div_ceil(8));
            (VALIDITY_BITMAP, Some(bytes))
        }
    })
}

/// Write `array`, canonicalized, into guest memory as an array frame and return the frame's
/// address.
pub(crate) fn write_array(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
    mem: &mut dyn GuestMem,
) -> VortexResult<u32> {
    let mut frame = Vec::new();
    write_array_into(array, ctx, mem, &mut frame, 0)?;
    put(mem, &frame)
}

fn host_bytes(handle: &BufferHandle) -> VortexResult<ByteBuffer> {
    handle.clone().try_to_host_sync()
}

fn write_array_into(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
    mem: &mut dyn GuestMem,
    frame: &mut Vec<u8>,
    depth: usize,
) -> VortexResult<()> {
    vortex_ensure!(
        depth <= MAX_DEPTH,
        "array nested too deeply to send to a kernel"
    );
    let len = array.len();
    let canonical = array.clone().execute::<Canonical>(ctx)?;

    // The validity written on the node itself. Shapes whose validity lives in a child (union,
    // map, extension) write `NonNullable` here and let the child speak for them.
    let own_validity = array.validity()?;
    let (shape, param, validity, buffers, children): (
        u8,
        u8,
        Validity,
        Vec<ByteBuffer>,
        Vec<ArrayRef>,
    ) = match &canonical {
        Canonical::Null(_) => (SHAPE_NULL, 0, Validity::NonNullable, vec![], vec![]),
        Canonical::Bool(bools) => {
            let bits = bools.to_bit_buffer().shrink_offset();
            vortex_ensure!(bits.offset() == 0, "bool array could not be byte-aligned");
            (
                SHAPE_BOOL,
                0,
                own_validity,
                vec![bits.inner().slice(0..len.div_ceil(8))],
                vec![],
            )
        }
        Canonical::Primitive(prims) => (
            SHAPE_PRIMITIVE,
            prims.ptype() as u8,
            own_validity,
            vec![host_bytes(prims.buffer_handle())?],
            vec![],
        ),
        Canonical::Decimal(decimals) => (
            SHAPE_DECIMAL,
            decimals.values_type() as u8,
            own_validity,
            vec![host_bytes(decimals.buffer_handle())?],
            vec![],
        ),
        Canonical::VarBinView(views) => {
            let mut buffers = vec![host_bytes(views.views_handle())?];
            for data in views.data_buffers().iter() {
                buffers.push(host_bytes(data)?);
            }
            (SHAPE_VAR_BIN_VIEW, 0, own_validity, buffers, vec![])
        }
        Canonical::List(lists) => (
            SHAPE_LIST,
            0,
            own_validity,
            vec![],
            vec![
                lists.offsets().clone(),
                lists.sizes().clone(),
                lists.elements().clone(),
            ],
        ),
        Canonical::FixedSizeList(lists) => (
            SHAPE_FIXED_SIZE_LIST,
            0,
            own_validity,
            vec![],
            vec![lists.elements().clone()],
        ),
        Canonical::Struct(structs) => (
            SHAPE_STRUCT,
            0,
            own_validity,
            vec![],
            structs.fields().to_vec(),
        ),
        Canonical::Union(unions) => {
            let mut children = vec![unions.type_ids().clone()];
            children.extend(unions.children().iter().cloned());
            (SHAPE_UNION, 0, Validity::NonNullable, vec![], children)
        }
        Canonical::Map(maps) => (
            SHAPE_MAP,
            0,
            Validity::NonNullable,
            vec![],
            vec![maps.entries().clone()],
        ),
        Canonical::Extension(ext) => (
            SHAPE_EXTENSION,
            0,
            Validity::NonNullable,
            vec![],
            vec![ext.storage().clone()],
        ),
        Canonical::Variant(_) => vortex_bail!(
            "cannot send a Variant array to a kernel: Vortex defines no canonical byte layout for \
             variant values"
        ),
    };

    let (validity_tag, validity_bits) = encode_validity(&validity, len, ctx)?;
    let validity_ptr = match validity_bits {
        Some(bits) => put(mem, bits.as_slice())?,
        None => 0,
    };

    frame.push(shape);
    frame.push(param);
    frame.push(validity_tag);
    frame.push(0);
    frame.extend_from_slice(&u32::try_from(len)?.to_le_bytes());
    frame.extend_from_slice(&validity_ptr.to_le_bytes());
    frame.extend_from_slice(&u32::try_from(buffers.len())?.to_le_bytes());
    for buffer in &buffers {
        let ptr = put(mem, buffer.as_slice())?;
        frame.extend_from_slice(&ptr.to_le_bytes());
        frame.extend_from_slice(&u32::try_from(buffer.len())?.to_le_bytes());
    }
    frame.extend_from_slice(&u32::try_from(children.len())?.to_le_bytes());
    for child in &children {
        write_array_into(child, ctx, mem, frame, depth + 1)?;
    }
    Ok(())
}

/// An array frame read out of guest memory, before it is checked against a dtype.
#[derive(Debug)]
pub(crate) struct ArrayDescriptor {
    shape: u8,
    param: u8,
    validity: u8,
    len: usize,
    validity_ptr: u32,
    buffers: Vec<(u32, u32)>,
    children: Vec<ArrayDescriptor>,
}

impl ArrayDescriptor {
    /// Parse the frame at `offset` in `mem`, returning it and the offset just past it.
    ///
    /// Structural checks only — the frame is untrusted, so every length is bounds-checked and
    /// nesting and node counts are capped. Whether it matches a dtype is [`build`](Self::build)'s
    /// job.
    pub(crate) fn parse(mem: &[u8], offset: usize) -> VortexResult<(Self, usize)> {
        let mut budget = MAX_NODES;
        Self::parse_at(mem, offset, 0, &mut budget)
    }

    fn parse_at(
        mem: &[u8],
        offset: usize,
        depth: usize,
        budget: &mut usize,
    ) -> VortexResult<(Self, usize)> {
        vortex_ensure!(depth <= MAX_DEPTH, "kernel result nested too deeply");
        *budget = budget
            .checked_sub(1)
            .ok_or_else(|| vortex_err!("kernel result has more than {MAX_NODES} array nodes"))?;
        let header = mem
            .get(offset..offset + FRAME_HEADER)
            .ok_or_else(|| vortex_err!("truncated array frame in kernel result"))?;
        let shape = header[0];
        let param = header[1];
        let validity = header[2];
        let len = usize::try_from(read_u32(header, 4)?)?;
        let validity_ptr = read_u32(header, 8)?;
        let n_buffers = usize::try_from(read_u32(header, 12)?)?;
        vortex_ensure!(
            n_buffers <= MAX_BUFFERS,
            "kernel returned {n_buffers} buffers for one array, more than the {MAX_BUFFERS} allowed"
        );

        let mut cursor = offset + FRAME_HEADER;
        let buffers = (0..n_buffers)
            .map(|i| {
                let at = cursor + i * 8;
                Ok((read_u32(mem, at)?, read_u32(mem, at + 4)?))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        cursor += n_buffers * 8;

        let n_children = usize::try_from(read_u32(mem, cursor)?)?;
        cursor += 4;
        vortex_ensure!(
            n_children <= *budget,
            "kernel result declares more array nodes than the {MAX_NODES} allowed"
        );
        let mut children = Vec::with_capacity(n_children);
        for _ in 0..n_children {
            let (child, next) = Self::parse_at(mem, cursor, depth + 1, budget)?;
            children.push(child);
            cursor = next;
        }
        Ok((
            Self {
                shape,
                param,
                validity,
                len,
                validity_ptr,
                buffers,
                children,
            },
            cursor,
        ))
    }

    fn buffer(&self, mem: &[u8], i: usize, alignment: usize) -> VortexResult<ByteBuffer> {
        let (ptr, len) = self.buffers[i];
        copy_out(mem, ptr, len as usize, alignment)
    }

    fn expect_buffers(&self, n: usize, what: &str) -> VortexResult<()> {
        vortex_ensure!(
            self.buffers.len() == n,
            "kernel returned {} buffers for {what}, expected {n}",
            self.buffers.len()
        );
        Ok(())
    }

    fn expect_children(&self, n: usize, what: &str) -> VortexResult<()> {
        vortex_ensure!(
            self.children.len() == n,
            "kernel returned {} children for {what}, expected {n}",
            self.children.len()
        );
        Ok(())
    }

    fn expect_shape(&self, shape: u8, dtype: &DType) -> VortexResult<()> {
        vortex_ensure!(
            self.shape == shape,
            "kernel returned array shape {} for dtype {dtype}, expected {shape}",
            self.shape
        );
        Ok(())
    }

    fn validity(&self, mem: &[u8], nullability: Nullability) -> VortexResult<Validity> {
        Ok(match self.validity {
            VALIDITY_NON_NULLABLE => {
                vortex_ensure!(
                    nullability == Nullability::NonNullable,
                    "kernel returned a non-nullable array for a nullable dtype"
                );
                Validity::NonNullable
            }
            VALIDITY_ALL_VALID => Validity::AllValid,
            VALIDITY_ALL_INVALID => Validity::AllInvalid,
            VALIDITY_BITMAP => {
                let bytes = copy_out(mem, self.validity_ptr, self.len.div_ceil(8), 1)?;
                Validity::Array(
                    BoolArray::new(BitBuffer::new(bytes, self.len), Validity::NonNullable)
                        .into_array(),
                )
            }
            other => vortex_bail!("kernel returned unknown validity tag {other}"),
        })
    }

    /// A non-nullable integer child, whose ptype the frame itself declares — list offsets and
    /// sizes, union type ids.
    fn build_index_child(
        &self,
        i: usize,
        mem: &[u8],
        expected_ptype: Option<PType>,
        what: &str,
    ) -> VortexResult<ArrayRef> {
        let child = &self.children[i];
        vortex_ensure!(
            child.shape == SHAPE_PRIMITIVE,
            "kernel returned a non-primitive {what}"
        );
        let ptype = PType::try_from(i32::from(child.param))
            .map_err(|_| vortex_err!("kernel returned bad ptype {} for {what}", child.param))?;
        vortex_ensure!(ptype.is_int(), "kernel returned a non-integer {what}");
        if let Some(expected) = expected_ptype {
            vortex_ensure!(
                ptype == expected,
                "kernel returned {ptype} {what}, expected {expected}"
            );
        }
        child.build_inner(
            mem,
            &DType::Primitive(ptype, Nullability::NonNullable),
            &mut None,
        )
    }

    /// Build the Vortex array this frame describes, checking every level against `dtype`.
    pub(crate) fn build(
        &self,
        mem: &[u8],
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        self.build_inner(mem, dtype, &mut Some(ctx))
    }

    /// `ctx` is only needed to validate string views; the integer children built by
    /// [`build_index_child`](Self::build_index_child) never need it.
    fn build_inner(
        &self,
        mem: &[u8],
        dtype: &DType,
        ctx: &mut Option<&mut ExecutionCtx>,
    ) -> VortexResult<ArrayRef> {
        let nullability = dtype.nullability();
        let array = match dtype {
            DType::Null => {
                self.expect_shape(SHAPE_NULL, dtype)?;
                NullArray::new(self.len).into_array()
            }
            DType::Bool(_) => {
                self.expect_shape(SHAPE_BOOL, dtype)?;
                self.expect_buffers(1, "a bool array")?;
                let bits = self.buffer(mem, 0, 1)?;
                vortex_ensure!(
                    bits.len() >= self.len.div_ceil(8),
                    "bool values bitmap is too short"
                );
                BoolArray::try_new(
                    BitBuffer::new(bits, self.len),
                    self.validity(mem, nullability)?,
                )?
                .into_array()
            }
            DType::Primitive(ptype, _) => {
                self.expect_shape(SHAPE_PRIMITIVE, dtype)?;
                self.expect_buffers(1, "a primitive array")?;
                vortex_ensure!(
                    self.param == *ptype as u8,
                    "kernel returned ptype {} for dtype {dtype}",
                    self.param
                );
                let values = self.buffer(mem, 0, ptype.byte_width())?;
                vortex_ensure!(
                    values.len() == self.len * ptype.byte_width(),
                    "primitive values buffer is {} bytes, expected {}",
                    values.len(),
                    self.len * ptype.byte_width()
                );
                PrimitiveArray::from_byte_buffer(values, *ptype, self.validity(mem, nullability)?)
                    .into_array()
            }
            DType::Decimal(decimal, _) => {
                self.expect_shape(SHAPE_DECIMAL, dtype)?;
                self.expect_buffers(1, "a decimal array")?;
                let values_type = decimal_type(self.param)?;
                let width = values_type.byte_width();
                let values = self.buffer(mem, 0, width)?;
                vortex_ensure!(
                    values.len() == self.len * width,
                    "decimal values buffer is {} bytes, expected {}",
                    values.len(),
                    self.len * width
                );
                // `try_new_handle` checks the storage is wide enough for the precision.
                DecimalArray::try_new_handle(
                    BufferHandle::new_host(values),
                    values_type,
                    *decimal,
                    self.validity(mem, nullability)?,
                )?
                .into_array()
            }
            DType::Utf8(_) | DType::Binary(_) => {
                self.expect_shape(SHAPE_VAR_BIN_VIEW, dtype)?;
                vortex_ensure!(
                    !self.buffers.is_empty(),
                    "varbinview expects a views buffer"
                );
                let views = self.buffer(mem, 0, align_of::<BinaryView>())?;
                vortex_ensure!(
                    views.len() == self.len * size_of::<BinaryView>(),
                    "views buffer is {} bytes, expected {}",
                    views.len(),
                    self.len * size_of::<BinaryView>()
                );
                let data: Arc<[ByteBuffer]> = (1..self.buffers.len())
                    .map(|i| self.buffer(mem, i, 1))
                    .collect::<VortexResult<Vec<_>>>()?
                    .into();
                let ctx = ctx
                    .as_deref_mut()
                    .ok_or_else(|| vortex_err!("string arrays cannot appear as index children"))?;
                // `try_new` validates every view's buffer index, offset+size, and utf8.
                VarBinViewArray::try_new(
                    Buffer::from_byte_buffer(views),
                    data,
                    dtype.clone(),
                    self.validity(mem, nullability)?,
                    ctx,
                )?
                .into_array()
            }
            DType::List(element, _) => {
                self.expect_shape(SHAPE_LIST, dtype)?;
                self.expect_children(3, "a list array")?;
                let offsets = self.build_index_child(0, mem, None, "list offsets")?;
                let sizes = self.build_index_child(1, mem, None, "list sizes")?;
                let elements = self.children[2].build_inner(mem, element, ctx)?;
                vortex_ensure!(
                    offsets.len() == self.len && sizes.len() == self.len,
                    "list offsets and sizes must have one entry per list"
                );
                // `try_new` checks every offset + size lands inside the elements.
                ListViewArray::try_new(elements, offsets, sizes, self.validity(mem, nullability)?)?
                    .into_array()
            }
            DType::FixedSizeList(element, size, _) => {
                self.expect_shape(SHAPE_FIXED_SIZE_LIST, dtype)?;
                self.expect_children(1, "a fixed-size list array")?;
                let elements = self.children[0].build_inner(mem, element, ctx)?;
                FixedSizeListArray::try_new(
                    elements,
                    *size,
                    self.validity(mem, nullability)?,
                    self.len,
                )?
                .into_array()
            }
            DType::Map(map, _) => {
                self.expect_shape(SHAPE_MAP, dtype)?;
                self.expect_children(1, "a map array")?;
                let entries_dtype = DType::List(Arc::new(map.entries_dtype()), nullability);
                let entries = self.children[0]
                    .build_inner(mem, &entries_dtype, ctx)?
                    .try_downcast::<ListView>()
                    .map_err(|_| vortex_err!("map entries did not build as a list view"))?;
                MapArray::try_new(map.clone(), entries)?.into_array()
            }
            DType::Struct(fields, _) => {
                self.expect_shape(SHAPE_STRUCT, dtype)?;
                self.expect_children(fields.nfields(), "a struct array")?;
                let columns = self
                    .children
                    .iter()
                    .zip(fields.fields())
                    .map(|(child, field)| child.build_inner(mem, &field, ctx))
                    .collect::<VortexResult<Vec<_>>>()?;
                StructArray::try_new(
                    fields.names().clone(),
                    columns,
                    self.len,
                    self.validity(mem, nullability)?,
                )?
                .into_array()
            }
            DType::Union(variants, _) => {
                self.expect_shape(SHAPE_UNION, dtype)?;
                self.expect_children(1 + variants.len(), "a union array")?;
                // The type ids carry the union's validity: a null tag is an outer null.
                let type_ids = self.children[0].build_inner(
                    mem,
                    &DType::Primitive(PType::U8, nullability),
                    ctx,
                )?;
                let children = self.children[1..]
                    .iter()
                    .zip(variants.variants())
                    .map(|(child, variant)| child.build_inner(mem, &variant, ctx))
                    .collect::<VortexResult<Vec<_>>>()?;
                UnionArray::try_new(type_ids, variants.clone(), children)?.into_array()
            }
            DType::Variant(_) => vortex_bail!(
                "a kernel cannot return a Variant array: Vortex defines no canonical byte layout \
                 for variant values"
            ),
            DType::Extension(ext) => {
                self.expect_shape(SHAPE_EXTENSION, dtype)?;
                self.expect_children(1, "an extension array")?;
                let storage = self.children[0].build_inner(mem, ext.storage_dtype(), ctx)?;
                ExtensionArray::try_new(ext.clone(), storage)?.into_array()
            }
        };

        vortex_ensure!(
            array.len() == self.len,
            "kernel array length {} disagrees with its frame ({})",
            array.len(),
            self.len
        );
        vortex_ensure!(
            array.dtype() == dtype,
            "kernel returned dtype {}, expected {dtype}",
            array.dtype()
        );
        Ok(array)
    }
}

fn decimal_type(param: u8) -> VortexResult<DecimalType> {
    Ok(match param {
        0 => DecimalType::I8,
        1 => DecimalType::I16,
        2 => DecimalType::I32,
        3 => DecimalType::I64,
        4 => DecimalType::I128,
        5 => DecimalType::I256,
        other => vortex_bail!("kernel returned bad decimal storage type {other}"),
    })
}

fn read_u32(mem: &[u8], off: usize) -> VortexResult<u32> {
    let bytes: [u8; 4] = mem
        .get(off..off + 4)
        .and_then(|slice| slice.try_into().ok())
        .ok_or_else(|| vortex_err!("out-of-bounds read in kernel result"))?;
    Ok(u32::from_le_bytes(bytes))
}

/// Copy `len` bytes at guest offset `ptr` out of guest memory, aligned for the type that will
/// read them. Guest memory dies with the instance, so a copy is unavoidable; the alignment is
/// free on the way.
fn copy_out(mem: &[u8], ptr: u32, len: usize, alignment: usize) -> VortexResult<ByteBuffer> {
    let start = ptr as usize;
    vortex_ensure!(
        start.checked_add(len).is_some_and(|end| end <= mem.len()),
        "kernel buffer [{start}, {start}+{len}) is outside guest memory ({})",
        mem.len()
    );
    Ok(ByteBuffer::copy_from_aligned(
        &mem[start..start + len],
        Alignment::new(alignment.max(1)),
    ))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::FieldName;
    use vortex_array::dtype::MapDType;
    use vortex_array::dtype::StructFields;
    use vortex_array::dtype::UnionVariants;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;
    use vortex_buffer::buffer;
    use vortex_session::VortexSession;

    use super::*;

    /// A `Vec`-backed [`GuestMem`] standing in for guest linear memory, so the conversion can be
    /// exercised without instantiating a wasm module.
    struct VecMem {
        mem: Vec<u8>,
    }

    impl VecMem {
        fn new() -> Self {
            // Reserve offset 0 so it reads as a null pointer.
            Self { mem: vec![0u8; 8] }
        }
    }

    impl GuestMem for VecMem {
        fn alloc(&mut self, len: u32) -> VortexResult<u32> {
            while !self.mem.len().is_multiple_of(8) {
                self.mem.push(0);
            }
            let off = u32::try_from(self.mem.len())?;
            self.mem.resize(self.mem.len() + len as usize, 0);
            Ok(off)
        }

        fn write(&mut self, off: u32, bytes: &[u8]) -> VortexResult<()> {
            self.mem[off as usize..off as usize + bytes.len()].copy_from_slice(bytes);
            Ok(())
        }
    }

    fn session() -> VortexSession {
        array_session()
    }

    /// Push an array into "guest" memory as a frame and read it straight back — the same bytes a
    /// kernel would see as a child, and the same bytes it would return as its output.
    fn round_trip(array: ArrayRef) -> VortexResult<ArrayRef> {
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let mut mem = VecMem::new();
        let ptr = write_array(&array, &mut ctx, &mut mem)?;
        let (descriptor, _) = ArrayDescriptor::parse(&mem.mem, ptr as usize)?;
        descriptor.build(&mem.mem, array.dtype(), &mut ctx)
    }

    fn assert_round_trips(array: ArrayRef) -> VortexResult<()> {
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let imported = round_trip(array.clone())?;
        assert_arrays_eq!(imported, array, &mut ctx);
        Ok(())
    }

    fn strings(values: &[Option<&str>]) -> ArrayRef {
        VarBinViewArray::from_iter_nullable_str(values.iter().copied()).into_array()
    }

    fn list_of(
        elements: ArrayRef,
        bounds: &[(i32, i32)],
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        let offsets = PrimitiveArray::new(
            Buffer::from_iter(bounds.iter().map(|(o, _)| *o)),
            Validity::NonNullable,
        )
        .into_array();
        let sizes = PrimitiveArray::new(
            Buffer::from_iter(bounds.iter().map(|(_, s)| *s)),
            Validity::NonNullable,
        )
        .into_array();
        Ok(ListViewArray::try_new(elements, offsets, sizes, validity)?.into_array())
    }

    fn struct_of(
        fields: Vec<(&str, ArrayRef)>,
        len: usize,
        validity: Validity,
    ) -> VortexResult<ArrayRef> {
        let names: Vec<FieldName> = fields.iter().map(|(name, _)| (*name).into()).collect();
        Ok(StructArray::try_new(
            names.into(),
            fields.into_iter().map(|(_, array)| array),
            len,
            validity,
        )?
        .into_array())
    }

    #[test]
    fn null_round_trips() -> VortexResult<()> {
        assert_round_trips(NullArray::new(7).into_array())
    }

    #[test]
    fn bool_round_trips_with_a_bitmap() -> VortexResult<()> {
        let bits = BitBuffer::from_indices(6, [0usize, 2, 5]);
        let validity = Validity::from_iter([true, true, false, true, true, false]);
        assert_round_trips(BoolArray::new(bits, validity).into_array())
    }

    #[rstest]
    #[case(PrimitiveArray::new(buffer![1u8, 2, 3], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![-1i16, 0, 1], Validity::AllValid))]
    #[case(PrimitiveArray::new(buffer![1u32, 2, 3, 4, 5], Validity::from_iter([true, false, true, false, true])))]
    #[case(PrimitiveArray::new(buffer![i64::MIN, 0, i64::MAX], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![1.5f32, -2.5], Validity::NonNullable))]
    #[case(PrimitiveArray::new(buffer![1.5f64, -2.5], Validity::AllInvalid))]
    fn primitives_round_trip(#[case] array: PrimitiveArray) -> VortexResult<()> {
        assert_round_trips(array.into_array())
    }

    #[rstest]
    #[case::i64_storage(DecimalArray::try_new(buffer![12345i64, -678], DecimalDType::try_new(10, 2)?, Validity::NonNullable)?)]
    #[case::i128_storage(DecimalArray::try_new(buffer![1i128 << 70, -(1i128 << 70)], DecimalDType::try_new(30, 4)?, Validity::from_iter([true, false]))?)]
    fn decimals_round_trip_with_their_storage_width(
        #[case] array: DecimalArray,
    ) -> VortexResult<()> {
        assert_round_trips(array.into_array())
    }

    #[test]
    fn strings_round_trip_as_views() -> VortexResult<()> {
        assert_round_trips(strings(&[
            Some("short"),
            None,
            Some("a string long enough to live in the data buffer"),
            Some(""),
        ]))
    }

    #[test]
    fn lists_of_strings_round_trip() -> VortexResult<()> {
        let elements = strings(&[Some("a"), Some("bb"), None, Some("dddd"), Some("e")]);
        // Out-of-order, overlapping list views, as the layout permits.
        let lists = list_of(
            elements,
            &[(3, 2), (0, 2), (1, 0), (0, 5)],
            Validity::from_iter([true, true, false, true]),
        )?;
        assert_round_trips(lists)
    }

    #[test]
    fn fixed_size_lists_round_trip() -> VortexResult<()> {
        let elements =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5, 6], Validity::NonNullable).into_array();
        let array =
            FixedSizeListArray::try_new(elements, 3, Validity::from_iter([true, false]), 2)?;
        assert_round_trips(array.into_array())
    }

    #[test]
    fn nested_structs_round_trip() -> VortexResult<()> {
        let inner = struct_of(
            vec![
                (
                    "tags",
                    list_of(
                        strings(&[Some("x"), Some("y"), Some("z")]),
                        &[(0, 2), (2, 1), (0, 0)],
                        Validity::NonNullable,
                    )?,
                ),
                (
                    "n",
                    PrimitiveArray::new(buffer![7u16, 8, 9], Validity::AllValid).into_array(),
                ),
            ],
            3,
            Validity::from_iter([true, true, false]),
        )?;
        let outer = struct_of(
            vec![
                ("inner", inner),
                (
                    "flag",
                    BoolArray::new(BitBuffer::from_indices(3, [1usize]), Validity::NonNullable)
                        .into_array(),
                ),
            ],
            3,
            Validity::NonNullable,
        )?;
        assert_round_trips(outer)
    }

    #[test]
    fn unions_round_trip_with_their_type_tags() -> VortexResult<()> {
        let names: Vec<FieldName> = vec!["int".into(), "text".into()];
        let variants = UnionVariants::try_new(
            names.into(),
            vec![
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Utf8(Nullability::Nullable),
            ],
            vec![3, 7],
        )?;
        // A null tag is an outer null.
        let type_ids = PrimitiveArray::new(
            buffer![3u8, 7, 3, 0],
            Validity::from_iter([true, true, true, false]),
        )
        .into_array();
        let array = UnionArray::try_new(
            type_ids,
            variants,
            [
                PrimitiveArray::new(buffer![1i32, 0, 3, 0], Validity::NonNullable).into_array(),
                strings(&[None, Some("seven"), None, None]),
            ],
        )?;

        // `assert_arrays_eq!` has no union comparison yet, so compare the canonical parts: the
        // type-id child (which carries the outer validity) and each variant.
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let imported = round_trip(array.clone().into_array())?;
        assert_eq!(imported.dtype(), array.dtype());
        let imported = imported
            .try_downcast::<vortex_array::arrays::Union>()
            .map_err(|_| vortex_err!("not a union"))?;
        assert_arrays_eq!(imported.type_ids(), array.type_ids(), &mut ctx);
        assert_eq!(imported.children().len(), array.children().len());
        for (got, want) in imported.children().iter().zip(array.children().iter()) {
            assert_arrays_eq!(got, want, &mut ctx);
        }
        Ok(())
    }

    #[test]
    fn maps_round_trip() -> VortexResult<()> {
        let map_dtype = MapDType::try_new(
            DType::Utf8(Nullability::NonNullable),
            DType::Primitive(PType::I64, Nullability::Nullable),
            false,
        )?;
        // Map keys are never nullable, so these are plain (non-nullable) strings.
        let keys = VarBinViewArray::from_iter_str(["a", "b", "c"]).into_array();
        let values = PrimitiveArray::new(
            buffer![1i64, 2, 3],
            Validity::from_iter([true, false, true]),
        )
        .into_array();
        let entries = struct_of(
            vec![("key", keys), ("value", values)],
            3,
            Validity::NonNullable,
        )?;
        let entries = list_of(entries, &[(0, 2), (2, 1)], Validity::NonNullable)?
            .try_downcast::<ListView>()
            .map_err(|_| vortex_err!("not a list view"))?;
        assert_round_trips(MapArray::try_new(map_dtype, entries)?.into_array())
    }

    #[test]
    fn extension_arrays_round_trip_over_their_storage() -> VortexResult<()> {
        let ext = Timestamp::new(TimeUnit::Milliseconds, Nullability::Nullable).erased();
        let storage = PrimitiveArray::new(
            buffer![1_700_000_000_000i64, 0, 42],
            Validity::from_iter([true, false, true]),
        )
        .into_array();
        assert_round_trips(ExtensionArray::try_new(ext, storage)?.into_array())
    }

    #[test]
    fn a_shape_that_does_not_match_the_dtype_is_rejected() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![1i32, 2], Validity::NonNullable).into_array();
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let mut mem = VecMem::new();
        let ptr = write_array(&array, &mut ctx, &mut mem)?;
        let (descriptor, _) = ArrayDescriptor::parse(&mem.mem, ptr as usize)?;
        // Same bytes, asked to be a bool.
        assert!(
            descriptor
                .build(&mem.mem, &DType::Bool(Nullability::NonNullable), &mut ctx)
                .is_err()
        );
        // Same bytes, asked to be a different primitive width.
        assert!(
            descriptor
                .build(
                    &mem.mem,
                    &DType::Primitive(PType::I64, Nullability::NonNullable),
                    &mut ctx
                )
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn rejects_out_of_bounds_buffer() {
        // A frame whose buffer runs past the end of guest memory must error, not panic.
        let mut mem = vec![0u8; 64];
        let at = 16;
        mem[at] = SHAPE_PRIMITIVE;
        mem[at + 1] = PType::I32 as u8;
        mem[at + 2] = VALIDITY_NON_NULLABLE;
        mem[at + 4..at + 8].copy_from_slice(&4u32.to_le_bytes());
        mem[at + 12..at + 16].copy_from_slice(&1u32.to_le_bytes());
        mem[at + 16..at + 20].copy_from_slice(&9999u32.to_le_bytes());
        mem[at + 20..at + 24].copy_from_slice(&16u32.to_le_bytes());

        let (descriptor, _) = ArrayDescriptor::parse(&mem, at).expect("frame parses");
        let session = session();
        let mut ctx = session.create_execution_ctx();
        assert!(
            descriptor
                .build(
                    &mem,
                    &DType::Primitive(PType::I32, Nullability::NonNullable),
                    &mut ctx,
                )
                .is_err()
        );
    }

    /// Recursion over attacker-controlled bytes must stop at a defined depth rather than
    /// running the host stack out.
    #[test]
    fn a_deeply_nested_frame_is_rejected_not_overflowed() {
        let mut mem = Vec::new();
        for _ in 0..(MAX_DEPTH + 8) {
            // A LIST node with three children, the third of which is the next node down.
            mem.push(SHAPE_LIST);
            mem.extend_from_slice(&[0, VALIDITY_NON_NULLABLE, 0]);
            mem.extend_from_slice(&1u32.to_le_bytes());
            mem.extend_from_slice(&0u32.to_le_bytes());
            mem.extend_from_slice(&0u32.to_le_bytes());
            mem.extend_from_slice(&3u32.to_le_bytes());
            for _ in 0..2 {
                mem.push(SHAPE_PRIMITIVE);
                mem.extend_from_slice(&[PType::I32 as u8, VALIDITY_NON_NULLABLE, 0]);
                mem.extend_from_slice(&1u32.to_le_bytes());
                mem.extend_from_slice(&0u32.to_le_bytes());
                mem.extend_from_slice(&0u32.to_le_bytes());
                mem.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        let err = ArrayDescriptor::parse(&mem, 0).unwrap_err().to_string();
        assert!(err.contains("nested too deeply"), "{err}");
    }

    #[test]
    fn an_absurd_child_count_is_rejected_before_allocating() {
        let mut mem = vec![SHAPE_STRUCT, 0, VALIDITY_NON_NULLABLE, 0];
        mem.extend_from_slice(&1u32.to_le_bytes());
        mem.extend_from_slice(&0u32.to_le_bytes());
        mem.extend_from_slice(&0u32.to_le_bytes());
        mem.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(ArrayDescriptor::parse(&mem, 0).is_err());
    }

    #[test]
    fn a_child_count_that_does_not_match_the_dtype_is_rejected() -> VortexResult<()> {
        let two_fields = struct_of(
            vec![
                (
                    "a",
                    PrimitiveArray::new(buffer![1i32], Validity::NonNullable).into_array(),
                ),
                (
                    "b",
                    PrimitiveArray::new(buffer![2i32], Validity::NonNullable).into_array(),
                ),
            ],
            1,
            Validity::NonNullable,
        )?;
        let session = session();
        let mut ctx = session.create_execution_ctx();
        let mut mem = VecMem::new();
        let ptr = write_array(&two_fields, &mut ctx, &mut mem)?;
        let (descriptor, _) = ArrayDescriptor::parse(&mem.mem, ptr as usize)?;
        let one_field = DType::Struct(
            StructFields::new(
                vec![FieldName::from("a")].into(),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let err = descriptor
            .build(&mem.mem, &one_field, &mut ctx)
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected 1"), "{err}");
        Ok(())
    }
}
