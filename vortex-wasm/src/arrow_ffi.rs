// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The host side of the [Arrow C Data Interface] boundary with a WASM guest's linear memory.
//!
//! Decoded arrays cross the host/guest boundary as the Arrow C Data Interface. Because the guest
//! is wasm32 (4-byte pointers, separate address space), we cannot hand `arrow`'s `from_ffi`/
//! `to_ffi` a borrowed struct: this module reads and writes the C struct layout *as it exists in
//! the guest's 32-bit address space*, deep-copying buffers across.
//!
//! It is deliberately **not** a per-type special case. Both directions are driven by Arrow's own
//! machinery, so the full interface is supported:
//!
//! - **Schemas** round-trip through [`FFI_ArrowSchema`]: import parses the guest's recursive
//!   schema (format strings, names, flags, metadata, children, dictionaries) into a native
//!   `FFI_ArrowSchema` and lets `arrow-schema`'s own parser produce the [`Field`]; export builds
//!   the `FFI_ArrowSchema` from the field and serializes it into guest memory.
//! - **Arrays** are sized by [`arrow_data::layout`] for every [`DataType`] (validity/bitmap/
//!   fixed-width/variable-width/view buffers), recursing through children and dictionaries, and
//!   are validated by [`ArrayData::try_new`] — untrusted guest data gets Arrow's full validation
//!   (offset monotonicity, bounds, utf8) before any host code touches it.
//!
//! [Arrow C Data Interface]: https://arrow.apache.org/docs/format/CDataInterface.html

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::make_array;
use arrow_buffer::Buffer as ArrowBuffer;
use arrow_data::ArrayData;
use arrow_data::BufferSpec;
use arrow_data::layout;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::ffi::Flags;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::arrow::FromArrowArray;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

/// Size of an `ArrowSchema` struct in the wasm32 C ABI.
const SCHEMA_SIZE: usize = 48;
/// Size of an `ArrowArray` struct in the wasm32 C ABI.
const ARRAY_SIZE: usize = 64;

/// Guard against malicious guest structures (cycles are impossible, but deep chains and huge
/// child counts would otherwise be a cheap way to burn host memory).
const MAX_DEPTH: usize = 64;
const MAX_CHILDREN: usize = 4096;

/// `ArrowSchema` field offsets in the wasm32 C ABI (4-byte pointers, 8-byte/8-aligned `int64`).
mod schema {
    pub const FORMAT: usize = 0; // const char*
    pub const NAME: usize = 4; // const char*
    pub const METADATA: usize = 8; // const char* (int32-length-prefixed key/value blob)
    pub const FLAGS: usize = 16; // int64 (after 12 bytes of pointers + 4 pad)
    pub const N_CHILDREN: usize = 24; // int64
    pub const CHILDREN: usize = 32; // ArrowSchema** (ptr to array of ptrs)
    pub const DICTIONARY: usize = 36; // ArrowSchema*
}

/// `ArrowArray` field offsets in the wasm32 C ABI.
mod array {
    pub const LENGTH: usize = 0; // int64
    pub const NULL_COUNT: usize = 8; // int64
    pub const OFFSET: usize = 16; // int64
    pub const N_BUFFERS: usize = 24; // int64
    pub const N_CHILDREN: usize = 32; // int64
    pub const BUFFERS: usize = 40; // const void** (ptr to array of ptrs)
    pub const CHILDREN: usize = 44; // ArrowArray** (ptr to array of ptrs)
    pub const DICTIONARY: usize = 48; // ArrowArray*
}

fn read_u32(mem: &[u8], off: u32) -> VortexResult<u32> {
    let off = off as usize;
    vortex_ensure!(off + 4 <= mem.len(), "arrow-ffi: u32 read out of bounds");
    Ok(u32::from_le_bytes(mem[off..off + 4].try_into().expect("4")))
}

fn read_i64(mem: &[u8], off: u32) -> VortexResult<i64> {
    let off = off as usize;
    vortex_ensure!(off + 8 <= mem.len(), "arrow-ffi: i64 read out of bounds");
    Ok(i64::from_le_bytes(mem[off..off + 8].try_into().expect("8")))
}

/// Read a NUL-terminated C string.
fn read_cstr(mem: &[u8], ptr: u32) -> VortexResult<&str> {
    let start = ptr as usize;
    vortex_ensure!(start <= mem.len(), "arrow-ffi: string ptr out of bounds");
    let end = mem[start..]
        .iter()
        .position(|&b| b == 0)
        .map(|n| start + n)
        .ok_or_else(|| vortex_err!("arrow-ffi: unterminated C string"))?;
    std::str::from_utf8(&mem[start..end]).map_err(|_| vortex_err!("arrow-ffi: non-utf8 C string"))
}

/// Read an array of `n` 4-byte guest pointers.
fn read_ptr_array(mem: &[u8], ptr: u32, n: usize) -> VortexResult<Vec<u32>> {
    (0..n)
        .map(|i| read_u32(mem, ptr + (i * 4) as u32))
        .collect()
}

fn copy_bytes(mem: &[u8], ptr: u32, len: usize) -> VortexResult<ArrowBuffer> {
    let start = ptr as usize;
    vortex_ensure!(
        start.checked_add(len).is_some_and(|end| end <= mem.len()),
        "arrow-ffi: buffer [{start}, {start}+{len}) out of bounds ({})",
        mem.len()
    );
    Ok(ArrowBuffer::from(&mem[start..start + len]))
}

/// Parse the Arrow C metadata blob: `int32 n`, then `n` entries of
/// `int32 key_len, key, int32 value_len, value` (little-endian on wasm32).
fn read_metadata(mem: &[u8], ptr: u32) -> VortexResult<Vec<(String, String)>> {
    if ptr == 0 {
        return Ok(Vec::new());
    }
    let read_i32 = |off: u32| -> VortexResult<i32> {
        let off = off as usize;
        vortex_ensure!(off + 4 <= mem.len(), "arrow-ffi: metadata out of bounds");
        Ok(i32::from_le_bytes(mem[off..off + 4].try_into().expect("4")))
    };
    let read_str = |off: u32, len: usize| -> VortexResult<String> {
        let off = off as usize;
        vortex_ensure!(off + len <= mem.len(), "arrow-ffi: metadata out of bounds");
        Ok(std::str::from_utf8(&mem[off..off + len])
            .map_err(|_| vortex_err!("arrow-ffi: non-utf8 metadata"))?
            .to_string())
    };

    let mut pos = ptr;
    let n = usize::try_from(read_i32(pos)?)?;
    vortex_ensure!(
        n <= MAX_CHILDREN,
        "arrow-ffi: metadata entry count too large"
    );
    pos += 4;
    let mut entries = Vec::with_capacity(n);
    for _ in 0..n {
        let key_len = usize::try_from(read_i32(pos)?)?;
        let key = read_str(pos + 4, key_len)?;
        pos += 4 + key_len as u32;
        let value_len = usize::try_from(read_i32(pos)?)?;
        let value = read_str(pos + 4, value_len)?;
        pos += 4 + value_len as u32;
        entries.push((key, value));
    }
    Ok(entries)
}

/// Serialize metadata entries into the Arrow C metadata blob.
fn write_metadata(entries: &[(String, String)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(entries.len() as i32).to_le_bytes());
    for (key, value) in entries {
        out.extend_from_slice(&(key.len() as i32).to_le_bytes());
        out.extend_from_slice(key.as_bytes());
        out.extend_from_slice(&(value.len() as i32).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }
    out
}

/// Recursively parse an `ArrowSchema` struct out of guest memory into a native
/// [`FFI_ArrowSchema`], so `arrow-schema`'s own format parser can interpret it.
fn read_ffi_schema(mem: &[u8], ptr: u32, depth: usize) -> VortexResult<FFI_ArrowSchema> {
    vortex_ensure!(depth < MAX_DEPTH, "arrow-ffi: schema too deeply nested");

    let format = read_cstr(mem, read_u32(mem, ptr + schema::FORMAT as u32)?)?;
    let name_ptr = read_u32(mem, ptr + schema::NAME as u32)?;
    let flags = read_i64(mem, ptr + schema::FLAGS as u32)?;
    let n_children = usize::try_from(read_i64(mem, ptr + schema::N_CHILDREN as u32)?)?;
    vortex_ensure!(n_children <= MAX_CHILDREN, "arrow-ffi: too many children");

    let mut children = Vec::with_capacity(n_children);
    if n_children > 0 {
        let children_ptr = read_u32(mem, ptr + schema::CHILDREN as u32)?;
        for child_ptr in read_ptr_array(mem, children_ptr, n_children)? {
            children.push(read_ffi_schema(mem, child_ptr, depth + 1)?);
        }
    }

    let dictionary_ptr = read_u32(mem, ptr + schema::DICTIONARY as u32)?;
    let dictionary = (dictionary_ptr != 0)
        .then(|| read_ffi_schema(mem, dictionary_ptr, depth + 1))
        .transpose()?;

    let mut ffi = FFI_ArrowSchema::try_new(format, children, dictionary)
        .map_err(|e| vortex_err!("arrow-ffi: invalid schema: {e}"))?;
    if name_ptr != 0 {
        ffi = ffi
            .with_name(read_cstr(mem, name_ptr)?)
            .map_err(|e| vortex_err!("arrow-ffi: invalid schema name: {e}"))?;
    }
    ffi = ffi
        .with_flags(Flags::from_bits_truncate(flags))
        .map_err(|e| vortex_err!("arrow-ffi: invalid schema flags: {e}"))?;
    let metadata = read_metadata(mem, read_u32(mem, ptr + schema::METADATA as u32)?)?;
    if !metadata.is_empty() {
        ffi = ffi
            .with_metadata(metadata)
            .map_err(|e| vortex_err!("arrow-ffi: invalid schema metadata: {e}"))?;
    }
    Ok(ffi)
}

/// The child field data types of a nested [`DataType`], in Arrow child order.
fn child_types(dtype: &DataType) -> Vec<&DataType> {
    match dtype {
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::ListView(f)
        | DataType::LargeListView(f)
        | DataType::FixedSizeList(f, _)
        | DataType::Map(f, _) => vec![f.data_type()],
        DataType::Struct(fields) => fields.iter().map(|f| f.data_type()).collect(),
        DataType::Union(fields, _) => fields.iter().map(|(_, f)| f.data_type()).collect(),
        DataType::RunEndEncoded(run_ends, values) => {
            vec![run_ends.data_type(), values.data_type()]
        }
        _ => vec![],
    }
}

/// Whether buffer 0 (after validity) is an offsets buffer with `len + offset + 1` elements.
fn has_offsets_buffer(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::List(_)
            | DataType::LargeList(_)
            | DataType::Map(_, _)
    )
}

/// Recursively read an `ArrowArray` struct for `dtype` out of guest memory into a validated
/// [`ArrayData`].
fn read_array_data(
    mem: &[u8],
    ptr: u32,
    dtype: &DataType,
    depth: usize,
) -> VortexResult<ArrayData> {
    vortex_ensure!(depth < MAX_DEPTH, "arrow-ffi: array too deeply nested");

    let len = usize::try_from(read_i64(mem, ptr + array::LENGTH as u32)?)?;
    let offset = usize::try_from(read_i64(mem, ptr + array::OFFSET as u32)?)?;
    let n_buffers = usize::try_from(read_i64(mem, ptr + array::N_BUFFERS as u32)?)?;
    let n_children = usize::try_from(read_i64(mem, ptr + array::N_CHILDREN as u32)?)?;
    vortex_ensure!(
        n_buffers <= MAX_CHILDREN && n_children <= MAX_CHILDREN,
        "arrow-ffi: buffer/child count too large"
    );
    let _ = read_i64(mem, ptr + array::NULL_COUNT as u32)?;

    let spec = layout(dtype);
    let expected_buffers =
        spec.buffers.len() + usize::from(spec.can_contain_null_mask) + usize::from(spec.variadic);
    if spec.variadic {
        // Views: [validity, views, data0..dataN, variadic_lengths]; N >= 0.
        vortex_ensure!(
            n_buffers >= expected_buffers,
            "arrow-ffi: {dtype} expects at least {expected_buffers} buffers, got {n_buffers}"
        );
    } else {
        vortex_ensure!(
            n_buffers == expected_buffers,
            "arrow-ffi: {dtype} expects {expected_buffers} buffers, got {n_buffers}"
        );
    }

    let buffer_ptrs = if n_buffers > 0 {
        read_ptr_array(mem, read_u32(mem, ptr + array::BUFFERS as u32)?, n_buffers)?
    } else {
        Vec::new()
    };
    let mut next_buffer = 0usize;

    let null_bit_buffer = if spec.can_contain_null_mask {
        let validity_ptr = buffer_ptrs[next_buffer];
        next_buffer += 1;
        (validity_ptr != 0)
            .then(|| copy_bytes(mem, validity_ptr, (len + offset).div_ceil(8)))
            .transpose()?
    } else {
        None
    };

    let mut buffers = Vec::with_capacity(spec.buffers.len());
    if spec.variadic {
        // The views buffer: 16 bytes per element.
        let views = copy_bytes(mem, buffer_ptrs[next_buffer], (len + offset) * 16)?;
        buffers.push(views);
        next_buffer += 1;
        // The trailing buffer holds an int64 length for each variadic data buffer.
        let n_data = n_buffers - next_buffer - 1;
        let lengths = copy_bytes(mem, buffer_ptrs[n_buffers - 1], n_data * 8)?;
        for i in 0..n_data {
            let data_len = usize::try_from(i64::from_le_bytes(
                lengths.as_slice()[i * 8..i * 8 + 8].try_into().expect("8"),
            ))?;
            buffers.push(copy_bytes(mem, buffer_ptrs[next_buffer + i], data_len)?);
        }
    } else {
        let mut last_offsets: Option<ArrowBuffer> = None;
        let offsets_first = has_offsets_buffer(dtype);
        for (buffer_index, buffer_spec) in spec.buffers.iter().enumerate() {
            let buffer_ptr = buffer_ptrs[next_buffer];
            next_buffer += 1;
            let buffer = match buffer_spec {
                BufferSpec::FixedWidth { byte_width, .. } => {
                    let elements = if offsets_first && buffer_index == 0 {
                        // Offsets buffers carry one extra trailing entry.
                        len + offset + 1
                    } else {
                        len + offset
                    };
                    let buffer = copy_bytes(mem, buffer_ptr, elements * byte_width)?;
                    if offsets_first && buffer_index == 0 {
                        last_offsets = Some(buffer.clone());
                    }
                    buffer
                }
                BufferSpec::VariableWidth => {
                    // The data buffer's length is the final offset value.
                    let offsets = last_offsets
                        .as_ref()
                        .ok_or_else(|| vortex_err!("arrow-ffi: data buffer without offsets"))?;
                    let width = offsets.len() / (len + offset + 1);
                    let last = offsets.len() - width;
                    let data_len = match width {
                        4 => usize::try_from(i32::from_le_bytes(
                            offsets.as_slice()[last..last + 4].try_into().expect("4"),
                        ))?,
                        8 => usize::try_from(i64::from_le_bytes(
                            offsets.as_slice()[last..last + 8].try_into().expect("8"),
                        ))?,
                        _ => vortex_bail!("arrow-ffi: unsupported offset width {width}"),
                    };
                    copy_bytes(mem, buffer_ptr, data_len)?
                }
                BufferSpec::BitMap => copy_bytes(mem, buffer_ptr, (len + offset).div_ceil(8))?,
                BufferSpec::AlwaysNull => ArrowBuffer::from(Vec::<u8>::new()),
            };
            buffers.push(buffer);
        }
    }

    // Children: dtypes come from the DataType itself; each child reads its own length.
    let expected_children = child_types(dtype);
    vortex_ensure!(
        n_children == expected_children.len(),
        "arrow-ffi: {dtype} expects {} children, got {n_children}",
        expected_children.len()
    );
    let mut child_data = Vec::with_capacity(n_children);
    if n_children > 0 {
        let children_ptr = read_u32(mem, ptr + array::CHILDREN as u32)?;
        for (child_ptr, child_type) in read_ptr_array(mem, children_ptr, n_children)?
            .into_iter()
            .zip(expected_children)
        {
            child_data.push(read_array_data(mem, child_ptr, child_type, depth + 1)?);
        }
    }

    // A dictionary's values array hangs off the dictionary pointer and becomes child 0 of the
    // arrow-rs ArrayData (whose buffers are the keys).
    if let DataType::Dictionary(_, value_type) = dtype {
        let dictionary_ptr = read_u32(mem, ptr + array::DICTIONARY as u32)?;
        vortex_ensure!(
            dictionary_ptr != 0,
            "arrow-ffi: dictionary array missing values"
        );
        child_data.push(read_array_data(mem, dictionary_ptr, value_type, depth + 1)?);
    }

    ArrayData::try_new(
        dtype.clone(),
        len,
        null_bit_buffer,
        offset,
        buffers,
        child_data,
    )
    .map_err(|e| vortex_err!("arrow-ffi: invalid {dtype} array data: {e}"))
}

/// Import a Vortex array from Arrow C Data Interface structs in `mem`.
///
/// `array_ptr` and `schema_ptr` are wasm32 offsets to the `ArrowArray` and `ArrowSchema` structs.
pub fn import(mem: &[u8], array_ptr: u32, schema_ptr: u32) -> VortexResult<ArrayRef> {
    let ffi = read_ffi_schema(mem, schema_ptr, 0)?;
    let field = Field::try_from(&ffi).map_err(|e| vortex_err!("arrow-ffi: bad schema: {e}"))?;
    let data = read_array_data(mem, array_ptr, field.data_type(), 0)?;
    let arrow: ArrowArrayRef = make_array(data);
    ArrayRef::from_arrow(arrow.as_ref(), field.is_nullable())
}

/// A writable view of a WASM guest's linear memory, used to lay out Arrow C structs for the guest.
///
/// `alloc` allocates `len` bytes in guest memory (via the guest's `vx_alloc`) and returns the
/// offset; `write` copies bytes to a previously allocated offset.
pub trait GuestMem {
    /// Allocate `len` bytes in guest memory, returning the offset.
    fn alloc(&mut self, len: u32) -> VortexResult<u32>;
    /// Write `bytes` at guest offset `off`.
    fn write(&mut self, off: u32, bytes: &[u8]) -> VortexResult<()>;
}

fn put(mem: &mut dyn GuestMem, bytes: &[u8]) -> VortexResult<u32> {
    let off = mem.alloc(u32::try_from(bytes.len().max(1))?)?;
    mem.write(off, bytes)?;
    Ok(off)
}

fn put_cstr(mem: &mut dyn GuestMem, s: &str) -> VortexResult<u32> {
    let mut bytes = Vec::with_capacity(s.len() + 1);
    bytes.extend_from_slice(s.as_bytes());
    bytes.push(0);
    put(mem, &bytes)
}

/// Recursively serialize a native [`FFI_ArrowSchema`] into guest memory in the wasm32 layout,
/// returning the guest offset of the `ArrowSchema` struct.
fn write_schema(mem: &mut dyn GuestMem, ffi: &FFI_ArrowSchema) -> VortexResult<u32> {
    let format_ptr = put_cstr(mem, ffi.format())?;
    let name_ptr = match ffi.name() {
        Some(name) => put_cstr(mem, name)?,
        None => 0,
    };
    let metadata = ffi
        .metadata()
        .map_err(|e| vortex_err!("arrow-ffi: bad schema metadata: {e}"))?;
    let metadata_ptr = if metadata.is_empty() {
        0
    } else {
        let entries: Vec<(String, String)> = metadata.into_iter().collect();
        put(mem, &write_metadata(&entries))?
    };

    let children: Vec<u32> = ffi
        .children()
        .map(|child| write_schema(mem, child))
        .collect::<VortexResult<_>>()?;
    let children_ptr = if children.is_empty() {
        0
    } else {
        let bytes: Vec<u8> = children.iter().flat_map(|p| p.to_le_bytes()).collect();
        put(mem, &bytes)?
    };
    let dictionary_ptr = match ffi.dictionary() {
        Some(dictionary) => write_schema(mem, dictionary)?,
        None => 0,
    };

    let mut buf = vec![0u8; SCHEMA_SIZE];
    buf[schema::FORMAT..schema::FORMAT + 4].copy_from_slice(&format_ptr.to_le_bytes());
    buf[schema::NAME..schema::NAME + 4].copy_from_slice(&name_ptr.to_le_bytes());
    buf[schema::METADATA..schema::METADATA + 4].copy_from_slice(&metadata_ptr.to_le_bytes());
    let flags = ffi.flags().map(|f| f.bits()).unwrap_or(0);
    buf[schema::FLAGS..schema::FLAGS + 8].copy_from_slice(&flags.to_le_bytes());
    buf[schema::N_CHILDREN..schema::N_CHILDREN + 8]
        .copy_from_slice(&(children.len() as i64).to_le_bytes());
    buf[schema::CHILDREN..schema::CHILDREN + 4].copy_from_slice(&children_ptr.to_le_bytes());
    buf[schema::DICTIONARY..schema::DICTIONARY + 4].copy_from_slice(&dictionary_ptr.to_le_bytes());
    put(mem, &buf)
}

/// Recursively serialize an [`ArrayData`] into guest memory in the wasm32 layout, returning the
/// guest offset of the `ArrowArray` struct.
fn write_array_data(mem: &mut dyn GuestMem, data: &ArrayData) -> VortexResult<u32> {
    let spec = layout(data.data_type());

    let mut buffer_ptrs: Vec<u32> = Vec::new();
    if spec.can_contain_null_mask {
        let validity_ptr = match data.nulls() {
            Some(nulls) => {
                // The bitmap's bit offset must match the array offset for the C image to line up.
                vortex_ensure!(
                    nulls.inner().offset() == data.offset(),
                    "arrow-ffi: validity offset does not match array offset"
                );
                put(mem, nulls.inner().inner().as_slice())?
            }
            None => 0,
        };
        buffer_ptrs.push(validity_ptr);
    }
    for buffer in data.buffers() {
        buffer_ptrs.push(put(mem, buffer.as_slice())?);
    }
    if spec.variadic {
        // Append the int64 lengths of the variadic data buffers (everything after the views).
        let lengths: Vec<u8> = data.buffers()[1..]
            .iter()
            .flat_map(|b| (b.len() as i64).to_le_bytes())
            .collect();
        buffer_ptrs.push(put(mem, &lengths)?);
    }
    let buffers_ptr = if buffer_ptrs.is_empty() {
        0
    } else {
        let bytes: Vec<u8> = buffer_ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
        put(mem, &bytes)?
    };

    // A dictionary's single child is its values array, exported via the dictionary pointer.
    let (children, dictionary): (&[ArrayData], Option<&ArrayData>) =
        if matches!(data.data_type(), DataType::Dictionary(_, _)) {
            (&[], data.child_data().first())
        } else {
            (data.child_data(), None)
        };

    let child_ptrs: Vec<u32> = children
        .iter()
        .map(|child| write_array_data(mem, child))
        .collect::<VortexResult<_>>()?;
    let children_ptr = if child_ptrs.is_empty() {
        0
    } else {
        let bytes: Vec<u8> = child_ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
        put(mem, &bytes)?
    };
    let dictionary_ptr = match dictionary {
        Some(values) => write_array_data(mem, values)?,
        None => 0,
    };

    let mut buf = vec![0u8; ARRAY_SIZE];
    buf[array::LENGTH..array::LENGTH + 8].copy_from_slice(&(data.len() as i64).to_le_bytes());
    buf[array::NULL_COUNT..array::NULL_COUNT + 8]
        .copy_from_slice(&(data.null_count() as i64).to_le_bytes());
    buf[array::OFFSET..array::OFFSET + 8].copy_from_slice(&(data.offset() as i64).to_le_bytes());
    buf[array::N_BUFFERS..array::N_BUFFERS + 8]
        .copy_from_slice(&(buffer_ptrs.len() as i64).to_le_bytes());
    buf[array::N_CHILDREN..array::N_CHILDREN + 8]
        .copy_from_slice(&(child_ptrs.len() as i64).to_le_bytes());
    buf[array::BUFFERS..array::BUFFERS + 4].copy_from_slice(&buffers_ptr.to_le_bytes());
    buf[array::CHILDREN..array::CHILDREN + 4].copy_from_slice(&children_ptr.to_le_bytes());
    buf[array::DICTIONARY..array::DICTIONARY + 4].copy_from_slice(&dictionary_ptr.to_le_bytes());
    put(mem, &buf)
}

/// Export a canonical array as Arrow C Data Interface structs written into `mem`, returning the
/// `(array_ptr, schema_ptr)` offsets a guest can consume.
///
/// The conversion goes through the session's Arrow export (any Vortex dtype), so guests can
/// receive arbitrary child types, not just primitives.
pub fn export(
    canonical: &Canonical,
    ctx: &mut ExecutionCtx,
    mem: &mut dyn GuestMem,
) -> VortexResult<(u32, u32)> {
    let array = canonical.clone().into_array();
    let session = ctx.session().clone();
    let field = session.arrow().to_arrow_field("", array.dtype())?;
    let arrow = session.arrow().execute_arrow(array, Some(&field), ctx)?;

    let ffi = FFI_ArrowSchema::try_from(&field)
        .map_err(|e| vortex_err!("arrow-ffi: unexportable field: {e}"))?;
    let schema_ptr = write_schema(mem, &ffi)?;
    let array_ptr = write_array_data(mem, &arrow.to_data())?;
    Ok((array_ptr, schema_ptr))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;

    /// A `Vec`-backed [`GuestMem`] simulating guest linear memory for wasm-free tests.
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
            let off = self.mem.len() as u32;
            self.mem.resize(self.mem.len() + len as usize, 0);
            Ok(off)
        }

        fn write(&mut self, off: u32, bytes: &[u8]) -> VortexResult<()> {
            self.mem[off as usize..off as usize + bytes.len()].copy_from_slice(bytes);
            Ok(())
        }
    }

    /// Export `canonical` into a guest image and import it back.
    fn round_trip(canonical: Canonical) -> VortexResult<ArrayRef> {
        let mut ctx = array_session().create_execution_ctx();
        let mut mem = VecMem::new();
        let (array_ptr, schema_ptr) = export(&canonical, &mut ctx, &mut mem)?;
        import(&mem.mem, array_ptr, schema_ptr)
    }

    #[test]
    fn export_then_import_round_trip_nullable() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let validity = Validity::from_iter([true, false, true, false, true]);
        let canonical =
            Canonical::Primitive(PrimitiveArray::new(buffer![1i64, 2, 3, 4, 5], validity));

        let imported = round_trip(canonical)?;
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
    fn round_trip_strings() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let strings = ["one", "two", "three", "four", ""];
        let canonical = Canonical::VarBinView(VarBinViewArray::from_iter_str(strings));

        let imported = round_trip(canonical)?;
        assert_eq!(imported.len(), strings.len());
        let views = imported.execute::<Canonical>(&mut ctx)?.into_varbinview();
        for (i, expected) in strings.iter().enumerate() {
            assert_eq!(views.bytes_at(i).as_slice(), expected.as_bytes());
        }
        Ok(())
    }

    #[test]
    fn round_trip_struct() -> VortexResult<()> {
        use vortex_array::arrays::StructArray;
        use vortex_array::arrays::struct_::StructArrayExt;
        use vortex_array::validity::Validity;

        let mut ctx = array_session().create_execution_ctx();
        let ints = PrimitiveArray::new(buffer![10i32, 20, 30], Validity::NonNullable).into_array();
        let strings = VarBinViewArray::from_iter_str(["a", "bb", "ccc"]).into_array();
        let struct_array = StructArray::try_new(
            vec![Arc::from("i"), Arc::from("s")].into(),
            vec![ints, strings],
            3,
            Validity::NonNullable,
        )?;
        let canonical = struct_array.into_array().execute::<Canonical>(&mut ctx)?;

        let imported = round_trip(canonical)?;
        assert_eq!(imported.len(), 3);
        let imported = imported.execute::<Canonical>(&mut ctx)?.into_struct();
        let ints = imported
            .unmasked_field(0)
            .clone()
            .execute::<Canonical>(&mut ctx)?
            .into_primitive();
        assert_eq!(ints.as_slice::<i32>(), &[10, 20, 30]);
        let strings = imported
            .unmasked_field(1)
            .clone()
            .execute::<Canonical>(&mut ctx)?
            .into_varbinview();
        assert_eq!(strings.bytes_at(2).as_slice(), b"ccc");
        Ok(())
    }

    /// Lays out an Arrow C Data Interface image (wasm32) for a single primitive/bool array.
    struct ImageBuilder {
        mem: Vec<u8>,
    }

    impl ImageBuilder {
        fn new() -> Self {
            // Reserve a zero page so offset 0 reads as a null pointer.
            Self { mem: vec![0u8; 16] }
        }

        fn put(&mut self, bytes: &[u8]) -> u32 {
            // 8-align every region so struct int64 fields are aligned.
            while !self.mem.len().is_multiple_of(8) {
                self.mem.push(0);
            }
            let off = self.mem.len() as u32;
            self.mem.extend_from_slice(bytes);
            off
        }

        fn schema(&mut self, format: &str, nullable: bool) -> u32 {
            let mut fmt = format.as_bytes().to_vec();
            fmt.push(0);
            let format_ptr = self.put(&fmt);
            let mut s = vec![0u8; SCHEMA_SIZE];
            s[schema::FORMAT..schema::FORMAT + 4].copy_from_slice(&format_ptr.to_le_bytes());
            let flags: i64 = if nullable { Flags::NULLABLE.bits() } else { 0 };
            s[schema::FLAGS..schema::FLAGS + 8].copy_from_slice(&flags.to_le_bytes());
            self.put(&s)
        }

        fn array(&mut self, len: usize, values: &[u8], validity: Option<&[u8]>) -> u32 {
            let values_ptr = self.put(values);
            let validity_ptr = validity.map(|v| self.put(v)).unwrap_or(0);
            let mut buffers = Vec::new();
            buffers.extend_from_slice(&validity_ptr.to_le_bytes());
            buffers.extend_from_slice(&values_ptr.to_le_bytes());
            let buffers_ptr = self.put(&buffers);

            let null_count: i64 = if validity.is_some() { -1 } else { 0 };
            let mut a = vec![0u8; ARRAY_SIZE];
            a[array::LENGTH..array::LENGTH + 8].copy_from_slice(&(len as i64).to_le_bytes());
            a[array::NULL_COUNT..array::NULL_COUNT + 8].copy_from_slice(&null_count.to_le_bytes());
            a[array::OFFSET..array::OFFSET + 8].copy_from_slice(&0i64.to_le_bytes());
            a[array::N_BUFFERS..array::N_BUFFERS + 8].copy_from_slice(&2i64.to_le_bytes());
            a[array::BUFFERS..array::BUFFERS + 4].copy_from_slice(&buffers_ptr.to_le_bytes());
            self.put(&a)
        }
    }

    #[test]
    fn import_primitive_i32() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values: Vec<u8> = [10i32, 20, 30, 40]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut b = ImageBuilder::new();
        let schema_ptr = b.schema("i", false);
        let array_ptr = b.array(4, &values, None);

        let imported = import(&b.mem, array_ptr, schema_ptr)?;
        assert_eq!(imported.len(), 4);
        let canonical = imported.execute::<Canonical>(&mut ctx)?;
        assert_eq!(
            canonical.into_primitive().as_slice::<i32>(),
            &[10, 20, 30, 40]
        );
        Ok(())
    }

    #[test]
    fn import_nullable_i32() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let values: Vec<u8> = [1i32, 2, 3, 4, 5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // valid at 0,2,4 -> bits 1,0,1,0,1 -> 0b10101 = 0x15
        let validity = [0x15u8];
        let mut b = ImageBuilder::new();
        let schema_ptr = b.schema("i", true);
        let array_ptr = b.array(5, &values, Some(&validity));

        let imported = import(&b.mem, array_ptr, schema_ptr)?;
        assert_eq!(imported.len(), 5);
        let bits = imported
            .validity()?
            .execute_mask(5, &mut ctx)?
            .to_bit_buffer();
        let valid: Vec<bool> = (0..5).map(|i| bits.value(i)).collect();
        assert_eq!(valid, vec![true, false, true, false, true]);
        Ok(())
    }
}
