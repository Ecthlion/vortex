// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrays crossing the host/guest boundary, in **Vortex's own canonical vocabulary**.
//!
//! One wire format serves both directions. The host writes the node's decoded children into
//! guest memory as [`ArrayView`]s, and the kernel returns its output as a [`Decoded`], which is
//! written in the same layout. Every shape in Vortex's `Canonical` enum has a wire shape — see
//! [`crate::abi::shape`] — so a kernel can read a child of any dtype and produce an output
//! of any dtype, nested types included.
//!
//! There is deliberately no Arrow C Data Interface here. That protocol carries a schema, and this
//! boundary has none to carry: the host already holds the node's `DType`, and the guest declares
//! its children's dtypes itself. Strings cross as canonical 16-byte views plus data buffers, lists
//! as list-views (offsets and sizes, so a sublist is a slice of the elements and never copied).
//!
//! # Re-arranging encodings
//!
//! Run-end, dict, and their relatives do not compute values: their output is a child's values
//! re-arranged. [`Decoded::take`] is the generic gather that lets such a kernel materialize its
//! output over a child of *any* dtype without dtype-specific code. It is cheap where the canonical
//! layouts make it cheap — string views and list offsets are gathered, the bytes they point at are
//! copied once — and it recurses through structs, unions, maps, and extension storage.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::abi::PType;
use crate::abi::array_frame;
use crate::abi::shape;
use crate::abi::validity;
use crate::error::GuestError;
use crate::error::GuestResult;
use crate::host::alloc_bytes;

/// Values up to this length are stored inline in a view instead of the data buffer.
const MAX_INLINED: usize = 12;

/// Size of one canonical string view.
const VIEW_SIZE: usize = 16;

/// Maximum array nesting the guest will parse, matching the host's limit.
pub const MAX_DEPTH: usize = 32;

/// Storage width of a decimal array's values. Discriminants match the host's `DecimalType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DecimalType {
    /// 8-bit storage.
    I8 = 0,
    /// 16-bit storage.
    I16 = 1,
    /// 32-bit storage.
    I32 = 2,
    /// 64-bit storage.
    I64 = 3,
    /// 128-bit storage.
    I128 = 4,
    /// 256-bit storage.
    I256 = 5,
}

impl DecimalType {
    /// Width in bytes.
    pub const fn byte_width(self) -> usize {
        match self {
            DecimalType::I8 => 1,
            DecimalType::I16 => 2,
            DecimalType::I32 => 4,
            DecimalType::I64 => 8,
            DecimalType::I128 => 16,
            DecimalType::I256 => 32,
        }
    }

    /// Parse the wire discriminant.
    pub fn from_discriminant(value: u8) -> Option<Self> {
        Some(match value {
            0 => DecimalType::I8,
            1 => DecimalType::I16,
            2 => DecimalType::I32,
            3 => DecimalType::I64,
            4 => DecimalType::I128,
            5 => DecimalType::I256,
            _ => return None,
        })
    }
}

/// The validity of an array, as an algebra rather than an always-materialized bitmap.
///
/// A non-nullable or all-valid array carries no bitmap across the boundary at all.
pub enum Validity {
    /// The dtype is not nullable.
    NonNullable,
    /// Nullable, all elements valid.
    AllValid,
    /// Nullable, all elements null.
    AllInvalid,
    /// An LSB-first bitmap, `ceil(len / 8)` bytes, 1 = valid.
    Bitmap(Vec<u8>),
}

impl Validity {
    fn tag(&self) -> u8 {
        match self {
            Validity::NonNullable => validity::NON_NULLABLE,
            Validity::AllValid => validity::ALL_VALID,
            Validity::AllInvalid => validity::ALL_INVALID,
            Validity::Bitmap(_) => validity::BITMAP,
        }
    }
}

/// The validity of a host-supplied array, borrowed from guest memory.
#[derive(Clone, Copy)]
pub enum ValidityView<'a> {
    /// The dtype is not nullable.
    NonNullable,
    /// Nullable, all elements valid.
    AllValid,
    /// Nullable, all elements null.
    AllInvalid,
    /// An LSB-first bitmap, `ceil(len / 8)` bytes, 1 = valid.
    Bitmap(&'a [u8]),
}

impl ValidityView<'_> {
    /// Whether element `i` is valid.
    pub fn is_valid(&self, i: usize) -> bool {
        match self {
            ValidityView::NonNullable | ValidityView::AllValid => true,
            ValidityView::AllInvalid => false,
            ValidityView::Bitmap(bits) => bit(bits, i),
        }
    }

    /// An owned copy.
    pub fn to_owned(&self) -> Validity {
        match self {
            ValidityView::NonNullable => Validity::NonNullable,
            ValidityView::AllValid => Validity::AllValid,
            ValidityView::AllInvalid => Validity::AllInvalid,
            ValidityView::Bitmap(bits) => Validity::Bitmap(bits.to_vec()),
        }
    }

    /// The validity of the elements at `indices`, in order.
    pub fn take(&self, indices: &[u32]) -> Validity {
        match self {
            ValidityView::NonNullable => Validity::NonNullable,
            ValidityView::AllValid => Validity::AllValid,
            ValidityView::AllInvalid => Validity::AllInvalid,
            ValidityView::Bitmap(bits) => {
                let mut out = vec![0u8; indices.len().div_ceil(8)];
                for (row, &index) in indices.iter().enumerate() {
                    if bit(bits, index as usize) {
                        out[row / 8] |= 1 << (row % 8);
                    }
                }
                Validity::Bitmap(out)
            }
        }
    }
}

fn bit(bits: &[u8], i: usize) -> bool {
    bits.get(i / 8)
        .is_some_and(|byte| byte & (1 << (i % 8)) != 0)
}

/// A host-supplied array, parsed from guest memory. Borrowed: nothing is copied until a kernel
/// asks for it.
pub struct ArrayView<'a> {
    /// Logical element count.
    pub len: usize,
    /// Element validity. For shapes whose validity lives in a child (union, map, extension) this
    /// is what the host wrote for the node itself and is not authoritative.
    pub validity: ValidityView<'a>,
    /// The canonical shape and its buffers or children.
    pub kind: ArrayKind<'a>,
}

/// The canonical shape of an [`ArrayView`].
pub enum ArrayKind<'a> {
    /// The logical null array.
    Null,
    /// A boolean array; `bits` is an LSB-first bitmap.
    Bool {
        /// The values bitmap, `ceil(len / 8)` bytes.
        bits: &'a [u8],
    },
    /// A fixed-width numeric array.
    Primitive {
        /// Element type.
        ptype: PType,
        /// Little-endian values, `len * ptype.byte_width()` bytes.
        values: &'a [u8],
    },
    /// A decimal array: scaled integers of the given storage width.
    Decimal {
        /// Storage width.
        values_type: DecimalType,
        /// Little-endian values, `len * values_type.byte_width()` bytes.
        values: &'a [u8],
    },
    /// A string or binary array in the canonical view layout.
    VarBinView {
        /// `len * 16` bytes of views.
        views: &'a [u8],
        /// The data buffers views may reference.
        data: Vec<&'a [u8]>,
    },
    /// A variable-length list in the list-view layout: each list is `elements[offset..offset + size]`.
    List {
        /// Non-nullable integer offsets into `elements`, one per list.
        offsets: Box<ArrayView<'a>>,
        /// Non-nullable integer sizes, one per list.
        sizes: Box<ArrayView<'a>>,
        /// The flattened elements.
        elements: Box<ArrayView<'a>>,
    },
    /// A fixed-size list: list `i` is `elements[i * size..(i + 1) * size]`.
    FixedSizeList {
        /// The flattened elements, `len * size` long.
        elements: Box<ArrayView<'a>>,
    },
    /// A struct: one row-aligned child per field.
    Struct {
        /// The fields, in dtype order.
        fields: Vec<ArrayView<'a>>,
    },
    /// A sparse union: a `u8` tag per row selecting one of the row-aligned variants. A null tag
    /// is an outer null.
    Union {
        /// The type tags, a `u8` primitive whose validity is the union's.
        type_ids: Box<ArrayView<'a>>,
        /// The variants, in dtype order, each as long as the union.
        variants: Vec<ArrayView<'a>>,
    },
    /// A map: a list of non-nullable `{key, value}` entry structs.
    Map {
        /// The entries, a `List` of `Struct`.
        entries: Box<ArrayView<'a>>,
    },
    /// An extension array: its storage, whose dtype is the extension type's storage dtype.
    Extension {
        /// The storage array.
        storage: Box<ArrayView<'a>>,
    },
}

/// A typed view of a primitive array.
pub struct PrimitiveView<'a> {
    /// Element type.
    pub ptype: PType,
    /// Logical element count.
    pub len: usize,
    /// Little-endian values.
    pub values: &'a [u8],
    /// Element validity.
    pub validity: ValidityView<'a>,
}

impl PrimitiveView<'_> {
    /// Read element `i` widened to `u64` (values are unsigned-reinterpreted).
    pub fn value_u64(&self, i: usize) -> u64 {
        let w = self.ptype.byte_width();
        let mut buf = [0u8; 8];
        buf[..w].copy_from_slice(&self.values[i * w..(i + 1) * w]);
        u64::from_le_bytes(buf)
    }
}

/// A typed view of a boolean array.
pub struct BoolView<'a> {
    /// Logical element count.
    pub len: usize,
    /// The values bitmap, `ceil(len / 8)` bytes.
    pub bits: &'a [u8],
    /// Element validity.
    pub validity: ValidityView<'a>,
}

impl<'a> ArrayView<'a> {
    /// This array as a primitive, or an error if it is some other shape.
    pub fn as_primitive(&self) -> GuestResult<PrimitiveView<'a>> {
        match &self.kind {
            ArrayKind::Primitive { ptype, values } => Ok(PrimitiveView {
                ptype: *ptype,
                len: self.len,
                values,
                validity: self.validity,
            }),
            _ => Err(GuestError::new("array is not primitive")),
        }
    }

    /// This array as a boolean, or an error if it is some other shape.
    pub fn as_bool(&self) -> GuestResult<BoolView<'a>> {
        match &self.kind {
            ArrayKind::Bool { bits } => Ok(BoolView {
                len: self.len,
                bits,
                validity: self.validity,
            }),
            _ => Err(GuestError::new("array is not boolean")),
        }
    }

    /// Parse the array frame at guest address `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must address a frame the host wrote into this module's memory before the current
    /// call, which it keeps alive for the call's duration.
    pub(crate) unsafe fn parse(ptr: u32) -> GuestResult<Self> {
        // The frame's total size is not known before parsing it, so borrow from `ptr` to the end
        // of the address space; the parser only reads what the headers say is there.
        let bytes =
            unsafe { core::slice::from_raw_parts(ptr as *const u8, (u32::MAX - ptr) as usize) };
        Self::parse_at(bytes, 0, 0).map(|(view, _)| view)
    }

    /// Parse the frame at `offset` in `bytes`, returning the view and the offset just past it.
    fn parse_at(bytes: &'a [u8], offset: usize, depth: usize) -> GuestResult<(Self, usize)> {
        if depth > MAX_DEPTH {
            return Err(GuestError::new("array nested too deeply"));
        }
        let header = bytes
            .get(offset..offset + array_frame::HEADER)
            .ok_or(GuestError::new("truncated array frame"))?;
        let shape_tag = header[array_frame::SHAPE];
        let param = header[array_frame::PARAM];
        let validity_tag = header[array_frame::VALIDITY];
        let len = read_u32(header, array_frame::LEN) as usize;
        let validity_ptr = read_u32(header, array_frame::VALIDITY_PTR);
        let n_buffers = read_u32(header, array_frame::N_BUFFERS) as usize;

        let mut cursor = offset + array_frame::HEADER;
        let mut buffers: Vec<&'a [u8]> = Vec::with_capacity(n_buffers);
        for _ in 0..n_buffers {
            let entry = bytes
                .get(cursor..cursor + 8)
                .ok_or(GuestError::new("truncated buffer table"))?;
            let ptr = read_u32(entry, 0);
            let buffer_len = read_u32(entry, 4) as usize;
            // SAFETY: host-owned guest memory, valid for the duration of the decode call.
            buffers.push(unsafe { core::slice::from_raw_parts(ptr as *const u8, buffer_len) });
            cursor += 8;
        }
        let n_children = read_u32(
            bytes
                .get(cursor..cursor + 4)
                .ok_or(GuestError::new("truncated child count"))?,
            0,
        ) as usize;
        cursor += 4;
        let mut children = Vec::with_capacity(n_children);
        for _ in 0..n_children {
            let (child, next) = Self::parse_at(bytes, cursor, depth + 1)?;
            children.push(child);
            cursor = next;
        }

        let validity = match validity_tag {
            validity::NON_NULLABLE => ValidityView::NonNullable,
            validity::ALL_VALID => ValidityView::AllValid,
            validity::ALL_INVALID => ValidityView::AllInvalid,
            // SAFETY: as for buffers.
            validity::BITMAP => ValidityView::Bitmap(unsafe {
                core::slice::from_raw_parts(validity_ptr as *const u8, len.div_ceil(8))
            }),
            _ => return Err(GuestError::new("bad validity tag")),
        };

        let mut children = children.into_iter();
        let mut child = |what: &'static str| children.next().ok_or_else(|| GuestError::new(what));
        let buffer = |i: usize, what: &'static str| {
            buffers.get(i).copied().ok_or_else(|| GuestError::new(what))
        };
        let kind = match shape_tag {
            shape::NULL => ArrayKind::Null,
            shape::BOOL => ArrayKind::Bool {
                bits: buffer(0, "bool array missing its bitmap")?,
            },
            shape::PRIMITIVE => ArrayKind::Primitive {
                ptype: PType::from_discriminant(u64::from(param))
                    .ok_or(GuestError::new("bad ptype"))?,
                values: buffer(0, "primitive array missing its values")?,
            },
            shape::DECIMAL => ArrayKind::Decimal {
                values_type: DecimalType::from_discriminant(param)
                    .ok_or(GuestError::new("bad decimal storage type"))?,
                values: buffer(0, "decimal array missing its values")?,
            },
            shape::VAR_BIN_VIEW => ArrayKind::VarBinView {
                views: buffer(0, "view array missing its views")?,
                data: buffers.get(1..).unwrap_or(&[]).to_vec(),
            },
            shape::LIST => ArrayKind::List {
                offsets: Box::new(child("list missing offsets")?),
                sizes: Box::new(child("list missing sizes")?),
                elements: Box::new(child("list missing elements")?),
            },
            shape::FIXED_SIZE_LIST => ArrayKind::FixedSizeList {
                elements: Box::new(child("fixed-size list missing elements")?),
            },
            shape::STRUCT => ArrayKind::Struct {
                fields: children.collect(),
            },
            shape::UNION => ArrayKind::Union {
                type_ids: Box::new(child("union missing type ids")?),
                variants: children.collect(),
            },
            shape::MAP => ArrayKind::Map {
                entries: Box::new(child("map missing entries")?),
            },
            shape::EXTENSION => ArrayKind::Extension {
                storage: Box::new(child("extension missing storage")?),
            },
            _ => return Err(GuestError::new("bad array shape tag")),
        };
        Ok((
            Self {
                len,
                validity,
                kind,
            },
            cursor,
        ))
    }
}

/// A materialized primitive array.
pub struct DecodedPrimitive {
    /// Element type.
    pub ptype: PType,
    /// Logical element count.
    pub len: usize,
    /// Little-endian values, `len * ptype.byte_width()` bytes.
    pub values: Vec<u8>,
    /// Element validity.
    pub validity: Validity,
}

/// A materialized boolean array.
pub struct DecodedBool {
    /// Logical element count.
    pub len: usize,
    /// LSB-first values bitmap, `ceil(len / 8)` bytes.
    pub bits: Vec<u8>,
    /// Element validity.
    pub validity: Validity,
}

/// A materialized decimal array.
pub struct DecodedDecimal {
    /// Storage width. Must be wide enough for the dtype's precision.
    pub values_type: DecimalType,
    /// Logical element count.
    pub len: usize,
    /// Little-endian scaled integers, `len * values_type.byte_width()` bytes.
    pub values: Vec<u8>,
    /// Element validity.
    pub validity: Validity,
}

/// A materialized string/binary array in Vortex's canonical layout: 16-byte views plus data
/// buffers. Build one with [`DecodedVarBinView::from_heap`] rather than by hand.
pub struct DecodedVarBinView {
    /// Logical element count.
    pub len: usize,
    /// `len * 16` bytes of views.
    pub views: Vec<u8>,
    /// The data buffers views may reference.
    pub data: Vec<Vec<u8>>,
    /// Element validity.
    pub validity: Validity,
}

impl DecodedVarBinView {
    /// Build a canonical view array from one concatenated heap plus per-element lengths.
    ///
    /// This is the shape decompressors naturally produce (FSST, zstd), and emitting views here
    /// means the host constructs `VarBinViewArray` directly — the canonical form.
    pub fn from_heap(
        heap: Vec<u8>,
        lengths: impl IntoIterator<Item = usize>,
        validity: Validity,
    ) -> GuestResult<Self> {
        let mut views: Vec<u8> = Vec::new();
        let mut offset = 0usize;
        let mut len = 0usize;
        for size in lengths {
            let end = offset
                .checked_add(size)
                .filter(|&end| end <= heap.len())
                .ok_or(GuestError::new("string length exceeds the heap"))?;
            let value = &heap[offset..end];
            let size_u32 =
                u32::try_from(size).map_err(|_| GuestError::new("string too long for a view"))?;

            views.extend_from_slice(&size_u32.to_le_bytes());
            if size <= MAX_INLINED {
                // Inlined: [u32 size][12 bytes of value, zero padded]
                let mut inline = [0u8; MAX_INLINED];
                inline[..size].copy_from_slice(value);
                views.extend_from_slice(&inline);
            } else {
                // Ref: [u32 size][4-byte prefix][u32 buffer_index][u32 offset]
                views.extend_from_slice(&value[..4]);
                views.extend_from_slice(&0u32.to_le_bytes());
                views.extend_from_slice(
                    &u32::try_from(offset)
                        .map_err(|_| GuestError::new("heap offset exceeds u32"))?
                        .to_le_bytes(),
                );
            }
            offset = end;
            len += 1;
        }
        Ok(Self {
            len,
            views,
            data: vec![heap],
            validity,
        })
    }
}

/// An array the kernel built, to be returned to the host. Owned; mirrors [`ArrayKind`].
pub enum Decoded {
    /// The logical null array of `len` elements.
    Null {
        /// Logical element count.
        len: usize,
    },
    /// A boolean array.
    Bool(DecodedBool),
    /// A fixed-width numeric array.
    Primitive(DecodedPrimitive),
    /// A decimal array.
    Decimal(DecodedDecimal),
    /// A string or binary array.
    VarBinView(DecodedVarBinView),
    /// A list in the list-view layout.
    List {
        /// Logical element count.
        len: usize,
        /// Non-nullable integer offsets into `elements`.
        offsets: Box<Decoded>,
        /// Non-nullable integer sizes.
        sizes: Box<Decoded>,
        /// The flattened elements.
        elements: Box<Decoded>,
        /// List validity.
        validity: Validity,
    },
    /// A fixed-size list.
    FixedSizeList {
        /// Logical element count.
        len: usize,
        /// The flattened elements, `len * size` long.
        elements: Box<Decoded>,
        /// List validity.
        validity: Validity,
    },
    /// A struct.
    Struct {
        /// Logical element count.
        len: usize,
        /// The fields, in dtype order, each `len` long.
        fields: Vec<Decoded>,
        /// Row validity.
        validity: Validity,
    },
    /// A sparse union. Its validity is `type_ids`'s.
    Union {
        /// Logical element count.
        len: usize,
        /// A `u8` primitive of type tags; a null tag is an outer null.
        type_ids: Box<Decoded>,
        /// The variants, in dtype order, each `len` long.
        variants: Vec<Decoded>,
    },
    /// A map. Its validity is `entries`'s.
    Map {
        /// The entries, a `List` of non-nullable `Struct { key, value }`.
        entries: Box<Decoded>,
    },
    /// An extension array over its storage.
    Extension {
        /// The storage array.
        storage: Box<Decoded>,
    },
}

impl Decoded {
    /// Logical element count.
    pub fn len(&self) -> usize {
        match self {
            Decoded::Null { len }
            | Decoded::List { len, .. }
            | Decoded::FixedSizeList { len, .. }
            | Decoded::Struct { len, .. }
            | Decoded::Union { len, .. } => *len,
            Decoded::Bool(b) => b.len,
            Decoded::Primitive(p) => p.len,
            Decoded::Decimal(d) => d.len,
            Decoded::VarBinView(v) => v.len,
            Decoded::Map { entries } => entries.len(),
            Decoded::Extension { storage } => storage.len(),
        }
    }

    /// Whether the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// An owned copy of a host-supplied array.
    pub fn from_view(view: &ArrayView<'_>) -> Self {
        let validity = view.validity.to_owned();
        match &view.kind {
            ArrayKind::Null => Decoded::Null { len: view.len },
            ArrayKind::Bool { bits } => Decoded::Bool(DecodedBool {
                len: view.len,
                bits: bits.to_vec(),
                validity,
            }),
            ArrayKind::Primitive { ptype, values } => Decoded::Primitive(DecodedPrimitive {
                ptype: *ptype,
                len: view.len,
                values: values.to_vec(),
                validity,
            }),
            ArrayKind::Decimal {
                values_type,
                values,
            } => Decoded::Decimal(DecodedDecimal {
                values_type: *values_type,
                len: view.len,
                values: values.to_vec(),
                validity,
            }),
            ArrayKind::VarBinView { views, data } => Decoded::VarBinView(DecodedVarBinView {
                len: view.len,
                views: views.to_vec(),
                data: data.iter().map(|d| d.to_vec()).collect(),
                validity,
            }),
            ArrayKind::List {
                offsets,
                sizes,
                elements,
            } => Decoded::List {
                len: view.len,
                offsets: Box::new(Self::from_view(offsets)),
                sizes: Box::new(Self::from_view(sizes)),
                elements: Box::new(Self::from_view(elements)),
                validity,
            },
            ArrayKind::FixedSizeList { elements } => Decoded::FixedSizeList {
                len: view.len,
                elements: Box::new(Self::from_view(elements)),
                validity,
            },
            ArrayKind::Struct { fields } => Decoded::Struct {
                len: view.len,
                fields: fields.iter().map(Self::from_view).collect(),
                validity,
            },
            ArrayKind::Union { type_ids, variants } => Decoded::Union {
                len: view.len,
                type_ids: Box::new(Self::from_view(type_ids)),
                variants: variants.iter().map(Self::from_view).collect(),
            },
            ArrayKind::Map { entries } => Decoded::Map {
                entries: Box::new(Self::from_view(entries)),
            },
            ArrayKind::Extension { storage } => Decoded::Extension {
                storage: Box::new(Self::from_view(storage)),
            },
        }
    }

    /// The elements of `view` at `indices`, in order — a gather over any dtype.
    ///
    /// This is the primitive every re-arranging encoding needs. It follows the canonical layouts
    /// to their cheapest reading: string views and list offsets are gathered while the bytes they
    /// reference are copied through once; structs, unions, maps, and extension storage recurse.
    /// Every index is bounds-checked against `view.len`.
    pub fn take(view: &ArrayView<'_>, indices: &[u32]) -> GuestResult<Self> {
        if indices.iter().any(|&i| i as usize >= view.len) {
            return Err(GuestError::new("take index out of bounds"));
        }
        let len = indices.len();
        let validity = view.validity.take(indices);
        Ok(match &view.kind {
            ArrayKind::Null => Decoded::Null { len },
            ArrayKind::Bool { bits } => {
                let mut out = vec![0u8; len.div_ceil(8)];
                for (row, &index) in indices.iter().enumerate() {
                    if bit(bits, index as usize) {
                        out[row / 8] |= 1 << (row % 8);
                    }
                }
                Decoded::Bool(DecodedBool {
                    len,
                    bits: out,
                    validity,
                })
            }
            ArrayKind::Primitive { ptype, values } => Decoded::Primitive(DecodedPrimitive {
                ptype: *ptype,
                len,
                values: gather_fixed(values, ptype.byte_width(), indices),
                validity,
            }),
            ArrayKind::Decimal {
                values_type,
                values,
            } => Decoded::Decimal(DecodedDecimal {
                values_type: *values_type,
                len,
                values: gather_fixed(values, values_type.byte_width(), indices),
                validity,
            }),
            ArrayKind::VarBinView { views, data } => Decoded::VarBinView(DecodedVarBinView {
                len,
                views: gather_fixed(views, VIEW_SIZE, indices),
                data: data.iter().map(|d| d.to_vec()).collect(),
                validity,
            }),
            ArrayKind::List {
                offsets,
                sizes,
                elements,
            } => Decoded::List {
                len,
                // List-view semantics: gathering the offsets and sizes re-arranges the lists
                // without touching the elements.
                offsets: Box::new(Self::take(offsets, indices)?),
                sizes: Box::new(Self::take(sizes, indices)?),
                elements: Box::new(Self::from_view(elements)),
                validity,
            },
            ArrayKind::FixedSizeList { elements } => {
                let size = elements.len.checked_div(view.len).unwrap_or(0);
                let mut element_indices = Vec::with_capacity(len * size);
                for &index in indices {
                    let start = index as usize * size;
                    for i in start..start + size {
                        element_indices
                            .push(u32::try_from(i).map_err(|_| GuestError::new("index overflow"))?);
                    }
                }
                Decoded::FixedSizeList {
                    len,
                    elements: Box::new(Self::take(elements, &element_indices)?),
                    validity,
                }
            }
            ArrayKind::Struct { fields } => Decoded::Struct {
                len,
                fields: fields
                    .iter()
                    .map(|field| Self::take(field, indices))
                    .collect::<GuestResult<_>>()?,
                validity,
            },
            ArrayKind::Union { type_ids, variants } => Decoded::Union {
                len,
                type_ids: Box::new(Self::take(type_ids, indices)?),
                variants: variants
                    .iter()
                    .map(|variant| Self::take(variant, indices))
                    .collect::<GuestResult<_>>()?,
            },
            ArrayKind::Map { entries } => Decoded::Map {
                entries: Box::new(Self::take(entries, indices)?),
            },
            ArrayKind::Extension { storage } => Decoded::Extension {
                storage: Box::new(Self::take(storage, indices)?),
            },
        })
    }

    /// Write this array as a frame in guest memory and return its address.
    pub(crate) fn write(&self) -> u32 {
        let mut frame = Vec::new();
        self.write_into(&mut frame);
        alloc_bytes(&frame)
    }

    fn write_into(&self, frame: &mut Vec<u8>) {
        let no_validity = Validity::NonNullable;
        let (shape_tag, param, len, validity, buffers, children): (
            u8,
            u8,
            usize,
            &Validity,
            Vec<&[u8]>,
            Vec<&Decoded>,
        ) = match self {
            Decoded::Null { len } => (shape::NULL, 0, *len, &no_validity, vec![], vec![]),
            Decoded::Bool(b) => (shape::BOOL, 0, b.len, &b.validity, vec![&b.bits], vec![]),
            Decoded::Primitive(p) => (
                shape::PRIMITIVE,
                p.ptype as u8,
                p.len,
                &p.validity,
                vec![&p.values],
                vec![],
            ),
            Decoded::Decimal(d) => (
                shape::DECIMAL,
                d.values_type as u8,
                d.len,
                &d.validity,
                vec![&d.values],
                vec![],
            ),
            Decoded::VarBinView(v) => {
                let mut bufs: Vec<&[u8]> = vec![&v.views];
                bufs.extend(v.data.iter().map(|d| d.as_slice()));
                (shape::VAR_BIN_VIEW, 0, v.len, &v.validity, bufs, vec![])
            }
            Decoded::List {
                len,
                offsets,
                sizes,
                elements,
                validity,
            } => (
                shape::LIST,
                0,
                *len,
                validity,
                vec![],
                vec![offsets.as_ref(), sizes.as_ref(), elements.as_ref()],
            ),
            Decoded::FixedSizeList {
                len,
                elements,
                validity,
            } => (
                shape::FIXED_SIZE_LIST,
                0,
                *len,
                validity,
                vec![],
                vec![elements.as_ref()],
            ),
            Decoded::Struct {
                len,
                fields,
                validity,
            } => (
                shape::STRUCT,
                0,
                *len,
                validity,
                vec![],
                fields.iter().collect(),
            ),
            Decoded::Union {
                len,
                type_ids,
                variants,
            } => {
                let mut kids: Vec<&Decoded> = vec![type_ids.as_ref()];
                kids.extend(variants.iter());
                (shape::UNION, 0, *len, &no_validity, vec![], kids)
            }
            Decoded::Map { entries } => (
                shape::MAP,
                0,
                entries.len(),
                &no_validity,
                vec![],
                vec![entries.as_ref()],
            ),
            Decoded::Extension { storage } => (
                shape::EXTENSION,
                0,
                storage.len(),
                &no_validity,
                vec![],
                vec![storage.as_ref()],
            ),
        };

        let validity_ptr = match validity {
            Validity::Bitmap(bits) => alloc_bytes(bits),
            _ => 0,
        };

        frame.push(shape_tag);
        frame.push(param);
        frame.push(validity.tag());
        frame.push(0);
        frame.extend_from_slice(&(len as u32).to_le_bytes());
        frame.extend_from_slice(&validity_ptr.to_le_bytes());
        frame.extend_from_slice(&(buffers.len() as u32).to_le_bytes());
        for buffer in buffers {
            frame.extend_from_slice(&alloc_bytes(buffer).to_le_bytes());
            frame.extend_from_slice(&(buffer.len() as u32).to_le_bytes());
        }
        frame.extend_from_slice(&(children.len() as u32).to_le_bytes());
        for child in children {
            child.write_into(frame);
        }
    }
}

/// Gather `width`-byte elements at `indices` out of `values`.
fn gather_fixed(values: &[u8], width: usize, indices: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(indices.len() * width);
    for &index in indices {
        let start = index as usize * width;
        out.extend_from_slice(&values[start..start + width]);
    }
    out
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}
