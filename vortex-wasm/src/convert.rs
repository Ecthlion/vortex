// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Moving arrays across the host/guest boundary in **Vortex's own vocabulary**.
//!
//! This replaces what was an Arrow C Data Interface binding. That protocol carries a schema, and
//! this boundary has none to carry: the host already holds the node's [`DType`] and the guest
//! declares its children's dtypes itself. The Arrow round trip therefore meant the guest wrote a
//! format string the host parsed back into a type it already knew, ran Arrow's full
//! `ArrayData::try_new` revalidation, and then converted into Vortex's representation — landing
//! strings on `VarBin`, which is *not* canonical, so every string kernel paid a second conversion.
//!
//! Instead an array is a **buffer table plus a shape tag**, in the layout Vortex's canonical
//! arrays already use. For primitives and bools those bytes are identical to Arrow's; only the
//! schema goes away. Strings cross as canonical 16-byte views plus data buffers.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

/// Shape tags, mirroring the guest SDK's `abi::shape`.
const SHAPE_PRIMITIVE: u8 = 0;
const SHAPE_BOOL: u8 = 1;
const SHAPE_VAR_BIN_VIEW: u8 = 2;

/// Validity tags, mirroring the guest SDK's `abi::validity`.
const VALIDITY_NON_NULLABLE: u8 = 0;
const VALIDITY_ALL_VALID: u8 = 1;
const VALIDITY_ALL_INVALID: u8 = 2;
const VALIDITY_BITMAP: u8 = 3;

/// Size of a `Values` child entry in the decode frame, mirroring `abi::child_entry`.
pub(crate) const CHILD_ENTRY_SIZE: usize = 24;

/// Cap on the buffers a kernel may return for one array, so a malformed descriptor cannot drive
/// an unbounded number of copies out of guest memory.
const MAX_RESULT_BUFFERS: usize = 64;

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
/// Returns the tag plus the bitmap bytes. `shrink_offset` is what closes the bit-offset hazard: a
/// sliced array's mask can start mid-byte, and handing those bytes over verbatim would silently
/// shift every validity bit.
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

/// Write a canonical child array into guest memory as a 24-byte [`CHILD_ENTRY_SIZE`] entry.
///
/// Only primitive and boolean children are deliverable: anything else must be declared
/// `Reference`, which imposes no dtype limit because the child never enters guest memory.
pub(crate) fn write_child(
    canonical: &Canonical,
    ctx: &mut ExecutionCtx,
    mem: &mut dyn GuestMem,
) -> VortexResult<[u8; CHILD_ENTRY_SIZE]> {
    let (shape, ptype, len, values, validity): (u8, u8, usize, ByteBuffer, Validity) =
        match canonical {
            Canonical::Primitive(array) => (
                SHAPE_PRIMITIVE,
                array.ptype() as u8,
                array.len(),
                array.buffer_handle().try_to_host_sync()?,
                array.validity()?,
            ),
            Canonical::Bool(array) => {
                let bits = array.to_bit_buffer().shrink_offset();
                vortex_ensure!(bits.offset() == 0, "bool child could not be byte-aligned");
                (
                    SHAPE_BOOL,
                    0,
                    array.len(),
                    bits.inner().slice(0..array.len().div_ceil(8)),
                    array.validity()?,
                )
            }
            other => vortex_bail!(
                "a kernel may only read primitive or boolean children; got {:?}. Declare this child \
             as Reference instead — referenced children are never copied into the sandbox and so \
             may have any dtype.",
                std::mem::discriminant(other)
            ),
        };

    let (validity_tag, validity_bits) = encode_validity(&validity, len, ctx)?;
    let values_ptr = put(mem, values.as_slice())?;
    let validity_ptr = match validity_bits {
        Some(bits) => put(mem, bits.as_slice())?,
        None => 0,
    };

    let mut entry = [0u8; CHILD_ENTRY_SIZE];
    entry[0] = shape;
    entry[1] = ptype;
    entry[2] = validity_tag;
    entry[4..8].copy_from_slice(&u32::try_from(len)?.to_le_bytes());
    entry[8..12].copy_from_slice(&values_ptr.to_le_bytes());
    entry[12..16].copy_from_slice(&u32::try_from(values.len())?.to_le_bytes());
    entry[16..20].copy_from_slice(&validity_ptr.to_le_bytes());
    Ok(entry)
}

/// A materialized array descriptor read out of a guest result frame.
pub(crate) struct ArrayDescriptor {
    shape: u8,
    ptype: u8,
    validity: u8,
    len: usize,
    validity_ptr: u32,
    buffers: Vec<(u32, u32)>,
}

impl ArrayDescriptor {
    /// Parse a descriptor at `offset` in `mem`, returning it and the offset just past it.
    pub(crate) fn parse(mem: &[u8], offset: usize) -> VortexResult<(Self, usize)> {
        vortex_ensure!(
            offset + 12 <= mem.len(),
            "truncated array descriptor in kernel result"
        );
        let shape = mem[offset];
        let ptype = mem[offset + 1];
        let validity = mem[offset + 2];
        let n_buffers = mem[offset + 3] as usize;
        vortex_ensure!(
            n_buffers <= MAX_RESULT_BUFFERS,
            "kernel returned {n_buffers} buffers, more than the {MAX_RESULT_BUFFERS} allowed"
        );
        let len = read_u32(mem, offset + 4)? as usize;
        let validity_ptr = read_u32(mem, offset + 8)?;

        let table = offset + 12;
        vortex_ensure!(
            table + n_buffers * 8 <= mem.len(),
            "truncated buffer table in kernel result"
        );
        let buffers = (0..n_buffers)
            .map(|i| {
                Ok((
                    read_u32(mem, table + i * 8)?,
                    read_u32(mem, table + i * 8 + 4)?,
                ))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok((
            Self {
                shape,
                ptype,
                validity,
                len,
                validity_ptr,
                buffers,
            },
            table + n_buffers * 8,
        ))
    }

    fn buffer(&self, mem: &[u8], i: usize) -> VortexResult<ByteBuffer> {
        let (ptr, len) = self.buffers[i];
        copy_out(mem, ptr, len as usize)
    }

    fn validity(&self, mem: &[u8], nullable: bool) -> VortexResult<Validity> {
        Ok(match self.validity {
            VALIDITY_NON_NULLABLE => {
                vortex_ensure!(
                    !nullable,
                    "kernel returned a non-nullable array for a nullable dtype"
                );
                Validity::NonNullable
            }
            VALIDITY_ALL_VALID => Validity::AllValid,
            VALIDITY_ALL_INVALID => Validity::AllInvalid,
            VALIDITY_BITMAP => {
                let bytes = copy_out(mem, self.validity_ptr, self.len.div_ceil(8))?;
                Validity::Array(
                    BoolArray::new(BitBuffer::new(bytes, self.len), Validity::NonNullable)
                        .into_array(),
                )
            }
            other => vortex_bail!("kernel returned unknown validity tag {other}"),
        })
    }

    /// Build the Vortex array this descriptor names, checking it against the expected dtype.
    pub(crate) fn build(&self, mem: &[u8], dtype: &DType) -> VortexResult<ArrayRef> {
        let nullable = dtype.is_nullable();
        let validity = self.validity(mem, nullable)?;

        let array = match self.shape {
            SHAPE_PRIMITIVE => {
                vortex_ensure!(self.buffers.len() == 1, "primitive expects one buffer");
                let ptype = PType::try_from(self.ptype as i32)
                    .map_err(|_| vortex_err!("kernel returned bad ptype {}", self.ptype))?;
                let values = self.buffer(mem, 0)?;
                vortex_ensure!(
                    values.len() == self.len * ptype.byte_width(),
                    "primitive values buffer is {} bytes, expected {}",
                    values.len(),
                    self.len * ptype.byte_width()
                );
                PrimitiveArray::from_byte_buffer(values, ptype, validity).into_array()
            }
            SHAPE_BOOL => {
                vortex_ensure!(self.buffers.len() == 1, "bool expects one buffer");
                let bits = self.buffer(mem, 0)?;
                vortex_ensure!(
                    bits.len() >= self.len.div_ceil(8),
                    "bool values bitmap is too short"
                );
                BoolArray::try_new(BitBuffer::new(bits, self.len), validity)?.into_array()
            }
            SHAPE_VAR_BIN_VIEW => {
                vortex_ensure!(
                    !self.buffers.is_empty(),
                    "varbinview expects a views buffer"
                );
                let views = self.buffer(mem, 0)?;
                vortex_ensure!(
                    views.len() == self.len * size_of::<BinaryView>(),
                    "views buffer is {} bytes, expected {}",
                    views.len(),
                    self.len * size_of::<BinaryView>()
                );
                let data: Arc<[ByteBuffer]> = (1..self.buffers.len())
                    .map(|i| self.buffer(mem, i))
                    .collect::<VortexResult<Vec<_>>>()?
                    .into();
                // `try_new` validates every view's buffer index, offset+size, and utf8.
                VarBinViewArray::try_new(
                    Buffer::from_byte_buffer(views),
                    data,
                    dtype.clone(),
                    validity,
                )?
                .into_array()
            }
            other => vortex_bail!("kernel returned unknown array shape {other}"),
        };

        vortex_ensure!(
            array.len() == self.len,
            "kernel array length {} disagrees with its descriptor {}",
            array.len(),
            self.len
        );
        Ok(array)
    }
}

fn read_u32(mem: &[u8], off: usize) -> VortexResult<u32> {
    let bytes: [u8; 4] = mem
        .get(off..off + 4)
        .and_then(|slice| slice.try_into().ok())
        .ok_or_else(|| vortex_err!("out-of-bounds read in kernel result"))?;
    Ok(u32::from_le_bytes(bytes))
}

fn copy_out(mem: &[u8], ptr: u32, len: usize) -> VortexResult<ByteBuffer> {
    let start = ptr as usize;
    vortex_ensure!(
        start.checked_add(len).is_some_and(|end| end <= mem.len()),
        "kernel buffer [{start}, {start}+{len}) is outside guest memory ({})",
        mem.len()
    );
    Ok(ByteBuffer::copy_from(&mem[start..start + len]))
}

#[cfg(test)]
mod tests {
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::dtype::Nullability;
    use vortex_buffer::buffer;

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

    /// Push a canonical child into "guest" memory and read it straight back as an array, which is
    /// the same descriptor a kernel would see.
    fn child_round_trip(canonical: Canonical, dtype: &DType) -> VortexResult<ArrayRef> {
        let mut ctx = array_session().create_execution_ctx();
        let mut mem = VecMem::new();
        let entry = write_child(&canonical, &mut ctx, &mut mem)?;

        // Re-frame the child entry as a result descriptor: same shape/ptype/validity, one buffer.
        let mut frame = vec![entry[0], entry[1], entry[2], 1];
        frame.extend_from_slice(&entry[4..8]);
        frame.extend_from_slice(&entry[16..20]);
        frame.extend_from_slice(&entry[8..12]);
        frame.extend_from_slice(&entry[12..16]);
        let at = mem.mem.len();
        mem.mem.extend_from_slice(&frame);

        let (descriptor, _) = ArrayDescriptor::parse(&mem.mem, at)?;
        descriptor.build(&mem.mem, dtype)
    }

    #[test]
    fn primitive_round_trip_nullable() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let validity = Validity::from_iter([true, false, true, false, true]);
        let canonical =
            Canonical::Primitive(PrimitiveArray::new(buffer![1i64, 2, 3, 4, 5], validity));

        let imported = child_round_trip(
            canonical,
            &DType::Primitive(PType::I64, Nullability::Nullable),
        )?;
        assert_eq!(imported.len(), 5);
        let values = imported
            .clone()
            .execute::<Canonical>(&mut ctx)?
            .into_primitive();
        assert_eq!(values.as_slice::<i64>(), &[1, 2, 3, 4, 5]);
        let bits = imported
            .validity()?
            .execute_mask(5, &mut ctx)?
            .to_bit_buffer();
        let valid: Vec<bool> = (0..5).map(|i| bits.value(i)).collect();
        assert_eq!(valid, vec![true, false, true, false, true]);
        Ok(())
    }

    #[test]
    fn primitive_round_trip_non_nullable() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let canonical = Canonical::Primitive(PrimitiveArray::new(
            buffer![10i32, 20, 30, 40],
            Validity::NonNullable,
        ));

        let imported = child_round_trip(
            canonical,
            &DType::Primitive(PType::I32, Nullability::NonNullable),
        )?;
        let values = imported.execute::<Canonical>(&mut ctx)?.into_primitive();
        assert_eq!(values.as_slice::<i32>(), &[10, 20, 30, 40]);
        Ok(())
    }

    #[test]
    fn bool_round_trip() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let bits = BitBuffer::from_indices(6, [0usize, 2, 5]);
        let canonical = Canonical::Bool(BoolArray::new(bits, Validity::NonNullable));

        let imported = child_round_trip(canonical, &DType::Bool(Nullability::NonNullable))?;
        let mask = imported.execute::<Canonical>(&mut ctx)?.into_bool();
        let bits = mask.to_bit_buffer();
        let values: Vec<bool> = (0..6).map(|i| bits.value(i)).collect();
        assert_eq!(values, vec![true, false, true, false, false, true]);
        Ok(())
    }

    #[test]
    fn rejects_out_of_bounds_buffer() {
        // A descriptor whose buffer runs past the end of guest memory must error, not panic.
        let mut mem = vec![0u8; 64];
        let at = 16;
        mem[at] = SHAPE_PRIMITIVE;
        mem[at + 1] = PType::I32 as u8;
        mem[at + 2] = VALIDITY_NON_NULLABLE;
        mem[at + 3] = 1;
        mem[at + 4..at + 8].copy_from_slice(&4u32.to_le_bytes());
        mem[at + 12..at + 16].copy_from_slice(&32u32.to_le_bytes());
        mem[at + 16..at + 20].copy_from_slice(&9999u32.to_le_bytes());

        let (descriptor, _) = ArrayDescriptor::parse(&mem, at).expect("descriptor parses");
        assert!(
            descriptor
                .build(
                    &mem,
                    &DType::Primitive(PType::I32, Nullability::NonNullable)
                )
                .is_err()
        );
    }
}
