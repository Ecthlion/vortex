// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The `DType` wire codec: a compact, dependency-free encoding of a Vortex logical type.
//!
//! Both directions of the ABI need to name types. The host sends the node's own dtype; the guest
//! names its children's dtypes so the host can decode them. Neither side can afford the real
//! serializations — the flatbuffer and protobuf schemas would each pull a parser into a kernel
//! whose whole point is to stay small — so this is a third encoding, sized for the boundary.
//!
//! It is a **preorder byte stream**. Every type is one tag byte plus a payload, and composite
//! types simply contain more of the same:
//!
//! ```text
//! tag      := kind | (nullable << 6) | (derived << 7)
//! kind     := 0 Null | 1 Bool | 2 Primitive | 3 Decimal | 4 Utf8 | 5 Binary
//!           | 6 List | 7 FixedSizeList | 8 Struct | 9 Union | 10 Variant | 11 Extension
//!           | 12 Map
//!
//! Primitive     : u8 ptype
//! Decimal       : u8 precision, i8 scale
//! List          : dtype
//! FixedSizeList : varint size, dtype
//! Struct        : varint n, n × (varint name_len, utf8 name, dtype)
//! Union         : varint n, n × (varint name_len, utf8 name, u8 type_id, dtype)
//! Extension     : varint id_len, utf8 id, varint meta_len, meta, dtype (storage)
//! Map           : u8 keys_sorted, dtype (key), dtype (value)
//! others        : ε
//! ```
//!
//! # Derivations
//!
//! A literal is not always writable. Extension types resolve through a host-side vtable registry,
//! so a guest cannot construct one from bytes, and a kernel generic over its parent (run-end over
//! anything, dict over anything) does not *want* to name a concrete type — it wants to say "the
//! same type I was handed". So when the tag's high bit is set the low bits are a **derivation**: a
//! type expressed as a path from the parent rather than spelled out.
//!
//! ```text
//! Parent          : ε                    the node's own dtype
//! Field           : varint i, dtype      struct field i of the inner type
//! Element         : dtype                list / fixed-size-list element, or map entry
//!                                        struct, of the inner type
//! Storage         : dtype                storage type of the inner extension type
//! Nullable        : dtype                the inner type, made nullable
//! NonNullable     : dtype                the inner type, made non-nullable
//! ```
//!
//! Derivations compose, so `NonNullable(Field(2, Parent))` is a valid type expression. This is
//! what lets a kernel handle dtypes it could never spell: it never holds the type, only a path to
//! it. The host resolves the path against the dtype it already has.
//!
//! Only the guest→host direction uses derivations. The host always writes literals — it holds the
//! real `DType` and has nothing to derive from. Extension literals carry the type's real
//! serialized metadata in both directions, and the host resolves a guest-written one through its
//! dtype registry, so a kernel may name any extension type the reader knows.

use alloc::vec::Vec;

use crate::abi::PType;
use crate::abi::dtype_derivation as derive_op;
use crate::abi::dtype_kind as kind;
use crate::abi::dtype_tag;
use crate::error::GuestError;
use crate::error::GuestResult;

/// Maximum type nesting the guest will parse, matching the host's limit.
///
/// Bounds recursion when skipping over a nested type so a deeply nested one traps at a defined
/// point instead of overflowing the guest stack.
pub const MAX_DEPTH: usize = 32;

/// Read a LEB128 unsigned varint, returning the value and the number of bytes consumed.
pub(crate) fn read_varint(bytes: &[u8], mut offset: usize) -> GuestResult<(u64, usize)> {
    let start = offset;
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *bytes
            .get(offset)
            .ok_or(GuestError::new("truncated dtype varint"))?;
        offset += 1;
        value |= u64::from(byte & 0x7f)
            .checked_shl(shift)
            .ok_or(GuestError::new("dtype varint overflow"))?;
        if byte & 0x80 == 0 {
            return Ok((value, offset - start));
        }
        shift += 7;
        if shift >= 64 {
            return Err(GuestError::new("dtype varint overflow"));
        }
    }
}

/// Append a LEB128 unsigned varint.
pub(crate) fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            return;
        }
    }
}

/// The logical kind of a [`DTypeView`], as far as the guest distinguishes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypeKind {
    /// The logical null type.
    Null,
    /// A boolean.
    Bool,
    /// A fixed-width numeric.
    Primitive(PType),
    /// A fixed-precision decimal: precision and scale.
    Decimal(u8, i8),
    /// A UTF-8 string.
    Utf8,
    /// Binary data.
    Binary,
    /// A variable-length list.
    List,
    /// A fixed-size list of `size` elements.
    FixedSizeList(u32),
    /// A struct with `n` fields.
    Struct(usize),
    /// A union with `n` variants.
    Union(usize),
    /// A dynamically typed value.
    Variant,
    /// A user-defined extension type.
    Extension,
    /// A map from keys to values; `keys_sorted` says whether each entry's keys are sorted.
    Map {
        /// Whether keys within each map value are sorted.
        keys_sorted: bool,
    },
}

/// A borrowed, lazily parsed view of an encoded Vortex `DType`.
///
/// Parsing is on demand: constructing a view validates only the tag byte, and walking into a
/// composite type happens when a caller asks for a field or element. A kernel that only needs to
/// know "is my parent a 4-byte primitive" therefore reads exactly one byte, whatever the type's
/// nesting.
#[derive(Debug, Clone, Copy)]
pub struct DTypeView<'a> {
    bytes: &'a [u8],
}

impl<'a> DTypeView<'a> {
    /// View the encoded dtype at the start of `bytes`.
    ///
    /// The slice may be longer than the type; trailing bytes are ignored, which is what makes a
    /// dtype embeddable in a larger frame.
    pub fn new(bytes: &'a [u8]) -> GuestResult<Self> {
        if bytes.is_empty() {
            return Err(GuestError::new("empty dtype"));
        }
        if bytes[0] & dtype_tag::DERIVED != 0 {
            // Derivations are a guest→host construct. The host never writes one, so seeing one
            // here means the frame is malformed.
            return Err(GuestError::new("unexpected dtype derivation from the host"));
        }
        Ok(Self { bytes })
    }

    /// Whether the type is nullable.
    pub fn nullable(&self) -> bool {
        self.bytes[0] & dtype_tag::NULLABLE != 0
    }

    /// The type's kind and parameters.
    pub fn kind(&self) -> GuestResult<DTypeKind> {
        let payload = &self.bytes[1..];
        Ok(match self.bytes[0] & dtype_tag::KIND_MASK {
            kind::NULL => DTypeKind::Null,
            kind::BOOL => DTypeKind::Bool,
            kind::PRIMITIVE => DTypeKind::Primitive(
                PType::from_discriminant(u64::from(
                    *payload.first().ok_or(GuestError::new("truncated dtype"))?,
                ))
                .ok_or(GuestError::new("bad ptype in dtype"))?,
            ),
            kind::DECIMAL => {
                if payload.len() < 2 {
                    return Err(GuestError::new("truncated decimal dtype"));
                }
                DTypeKind::Decimal(payload[0], payload[1] as i8)
            }
            kind::UTF8 => DTypeKind::Utf8,
            kind::BINARY => DTypeKind::Binary,
            kind::LIST => DTypeKind::List,
            kind::FIXED_SIZE_LIST => {
                let (size, _) = read_varint(payload, 0)?;
                DTypeKind::FixedSizeList(
                    u32::try_from(size).map_err(|_| GuestError::new("list size overflow"))?,
                )
            }
            kind::STRUCT => {
                let (n, _) = read_varint(payload, 0)?;
                DTypeKind::Struct(
                    usize::try_from(n).map_err(|_| GuestError::new("field count overflow"))?,
                )
            }
            kind::UNION => {
                let (n, _) = read_varint(payload, 0)?;
                DTypeKind::Union(
                    usize::try_from(n).map_err(|_| GuestError::new("variant count overflow"))?,
                )
            }
            kind::VARIANT => DTypeKind::Variant,
            kind::EXTENSION => DTypeKind::Extension,
            kind::MAP => DTypeKind::Map {
                keys_sorted: *payload
                    .first()
                    .ok_or(GuestError::new("truncated map dtype"))?
                    != 0,
            },
            _ => return Err(GuestError::new("bad dtype kind")),
        })
    }

    /// The [`PType`] if this is a primitive, else `None`.
    pub fn ptype(&self) -> Option<PType> {
        match self.kind() {
            Ok(DTypeKind::Primitive(ptype)) => Some(ptype),
            _ => None,
        }
    }

    /// The total encoded length of this type in bytes.
    pub fn encoded_len(&self) -> GuestResult<usize> {
        Self::skip(self.bytes, 0, 0)
    }

    /// The element type of a `List` or `FixedSizeList`.
    pub fn element(&self) -> GuestResult<DTypeView<'a>> {
        let payload_start = match self.kind()? {
            DTypeKind::List => 1,
            DTypeKind::FixedSizeList(_) => 1 + read_varint(&self.bytes[1..], 0)?.1,
            _ => return Err(GuestError::new("dtype is not a list")),
        };
        Self::new(&self.bytes[payload_start..])
    }

    /// The key type of a `Map`.
    pub fn map_key(&self) -> GuestResult<DTypeView<'a>> {
        if !matches!(self.kind()?, DTypeKind::Map { .. }) {
            return Err(GuestError::new("dtype is not a map"));
        }
        Self::new(
            self.bytes
                .get(2..)
                .ok_or(GuestError::new("truncated map dtype"))?,
        )
    }

    /// The value type of a `Map`.
    pub fn map_value(&self) -> GuestResult<DTypeView<'a>> {
        let key = self.map_key()?;
        Self::new(
            self.bytes
                .get(2 + key.encoded_len()?..)
                .ok_or(GuestError::new("truncated map dtype"))?,
        )
    }

    /// The storage type of an `Extension`.
    pub fn storage(&self) -> GuestResult<DTypeView<'a>> {
        if self.kind()? != DTypeKind::Extension {
            return Err(GuestError::new("dtype is not an extension"));
        }
        let mut offset = 1;
        for _ in 0..2 {
            let (len, n) = read_varint(self.bytes, offset)?;
            offset = offset
                .checked_add(n)
                .and_then(|o| o.checked_add(usize::try_from(len).ok()?))
                .ok_or(GuestError::new("truncated extension dtype"))?;
        }
        Self::new(
            self.bytes
                .get(offset..)
                .ok_or(GuestError::new("truncated extension dtype"))?,
        )
    }

    /// The extension type's serialized metadata — what its vtable's `serialize_metadata` wrote.
    ///
    /// The bytes are the extension's own format (a Timestamp's, say, is its prost message);
    /// the SDK does not interpret them.
    pub fn extension_metadata(&self) -> GuestResult<&'a [u8]> {
        if self.kind()? != DTypeKind::Extension {
            return Err(GuestError::new("dtype is not an extension"));
        }
        let (id_len, n) = read_varint(self.bytes, 1)?;
        let after_id =
            1 + n + usize::try_from(id_len).map_err(|_| GuestError::new("id too long"))?;
        let (meta_len, n) = read_varint(self.bytes, after_id)?;
        let start = after_id + n;
        let end = start
            .checked_add(
                usize::try_from(meta_len).map_err(|_| GuestError::new("metadata too long"))?,
            )
            .ok_or(GuestError::new("truncated extension dtype"))?;
        self.bytes
            .get(start..end)
            .ok_or(GuestError::new("truncated extension metadata"))
    }

    /// The extension type's id, as raw UTF-8 bytes.
    pub fn extension_id(&self) -> GuestResult<&'a [u8]> {
        if self.kind()? != DTypeKind::Extension {
            return Err(GuestError::new("dtype is not an extension"));
        }
        let (len, n) = read_varint(self.bytes, 1)?;
        let start = 1 + n;
        let end = start
            .checked_add(usize::try_from(len).map_err(|_| GuestError::new("id too long"))?)
            .ok_or(GuestError::new("truncated extension dtype"))?;
        self.bytes
            .get(start..end)
            .ok_or(GuestError::new("truncated extension id"))
    }

    /// The name and type of struct field `index`.
    pub fn field(&self, index: usize) -> GuestResult<(&'a [u8], DTypeView<'a>)> {
        let DTypeKind::Struct(n) = self.kind()? else {
            return Err(GuestError::new("dtype is not a struct"));
        };
        let (name, _, dtype) = self.named_entry(index, n, false)?;
        Ok((name, dtype))
    }

    /// The name, type tag, and type of union variant `index`.
    ///
    /// The tag is what the data uses to select the variant; it is not necessarily `index`.
    pub fn variant(&self, index: usize) -> GuestResult<(&'a [u8], u8, DTypeView<'a>)> {
        let DTypeKind::Union(n) = self.kind()? else {
            return Err(GuestError::new("dtype is not a union"));
        };
        self.named_entry(index, n, true)
    }

    /// Entry `index` of a struct or union body: its name, its type tag (0 for a struct), and its
    /// type. Walks the preceding entries, since they are variable-length.
    fn named_entry(
        &self,
        index: usize,
        count: usize,
        with_type_ids: bool,
    ) -> GuestResult<(&'a [u8], u8, DTypeView<'a>)> {
        if index >= count {
            return Err(GuestError::new("dtype entry index out of bounds"));
        }
        let mut offset = 1 + read_varint(&self.bytes[1..], 0)?.1;
        for i in 0..=index {
            let (name_len, n_bytes) = read_varint(self.bytes, offset)?;
            let name_start = offset + n_bytes;
            let name_end = name_start
                .checked_add(
                    usize::try_from(name_len).map_err(|_| GuestError::new("name too long"))?,
                )
                .ok_or(GuestError::new("truncated dtype entry"))?;
            let name = self
                .bytes
                .get(name_start..name_end)
                .ok_or(GuestError::new("truncated dtype entry name"))?;
            let (type_id, dtype_start) = if with_type_ids {
                let tag = *self
                    .bytes
                    .get(name_end)
                    .ok_or(GuestError::new("truncated union type tag"))?;
                (tag, name_end + 1)
            } else {
                (0, name_end)
            };
            if i == index {
                return Ok((name, type_id, Self::new(&self.bytes[dtype_start..])?));
            }
            offset = dtype_start + Self::skip(self.bytes, dtype_start, 0)?;
        }
        Err(GuestError::new("dtype entry index out of bounds"))
    }

    /// The encoded length of the type starting at `offset`.
    fn skip(bytes: &[u8], offset: usize, depth: usize) -> GuestResult<usize> {
        if depth > MAX_DEPTH {
            return Err(GuestError::new("dtype nested too deeply"));
        }
        let tag = *bytes
            .get(offset)
            .ok_or(GuestError::new("truncated dtype"))?;
        let mut len = 1usize;
        match tag & dtype_tag::KIND_MASK {
            kind::NULL | kind::BOOL | kind::UTF8 | kind::BINARY | kind::VARIANT => {}
            kind::PRIMITIVE => len += 1,
            kind::DECIMAL => len += 2,
            kind::LIST => len += Self::skip(bytes, offset + len, depth + 1)?,
            kind::FIXED_SIZE_LIST => {
                len += read_varint(bytes, offset + len)?.1;
                len += Self::skip(bytes, offset + len, depth + 1)?;
            }
            kind::STRUCT | kind::UNION => {
                // Identical bodies, except that each union entry carries a one-byte type tag.
                let tag_len = usize::from(tag & dtype_tag::KIND_MASK == kind::UNION);
                let (n, n_bytes) = read_varint(bytes, offset + len)?;
                len += n_bytes;
                for _ in 0..n {
                    let (name_len, name_bytes) = read_varint(bytes, offset + len)?;
                    len += name_bytes
                        + usize::try_from(name_len)
                            .map_err(|_| GuestError::new("name too long"))?
                        + tag_len;
                    len += Self::skip(bytes, offset + len, depth + 1)?;
                }
            }
            kind::EXTENSION => {
                for _ in 0..2 {
                    let (blob_len, n_bytes) = read_varint(bytes, offset + len)?;
                    len += n_bytes
                        + usize::try_from(blob_len)
                            .map_err(|_| GuestError::new("extension blob too long"))?;
                }
                len += Self::skip(bytes, offset + len, depth + 1)?;
            }
            kind::MAP => {
                len += 1;
                len += Self::skip(bytes, offset + len, depth + 1)?;
                len += Self::skip(bytes, offset + len, depth + 1)?;
            }
            _ => return Err(GuestError::new("bad dtype kind")),
        }
        if offset + len > bytes.len() {
            return Err(GuestError::new("truncated dtype"));
        }
        Ok(len)
    }
}

/// A type expression the guest writes to name a child's dtype.
///
/// Every Vortex dtype has a literal spelling here, extension types included
/// ([`Self::extension`] carries the id, the serialized metadata, and the storage type). Still,
/// prefer the derivations ([`Self::parent`], [`Self::field`], ...) wherever the kernel does not
/// genuinely need to know the type: a kernel that derives is automatically generic over every
/// dtype Vortex has, including ones added after the kernel was built.
pub struct DTypeExpr {
    bytes: Vec<u8>,
}

impl DTypeExpr {
    /// The parent node's own dtype.
    ///
    /// The most useful expression in the set: a kernel that only re-arranges its parent's values
    /// (run-end, dict) names its values child this way and is thereby generic over every dtype.
    pub fn parent() -> Self {
        Self {
            bytes: alloc::vec![dtype_tag::DERIVED | derive_op::PARENT],
        }
    }

    fn literal(kind: u8, nullable: bool) -> Self {
        let mut bytes = Vec::with_capacity(2);
        bytes.push(kind | if nullable { dtype_tag::NULLABLE } else { 0 });
        Self { bytes }
    }

    /// The logical null type.
    pub fn null() -> Self {
        Self::literal(kind::NULL, false)
    }

    /// A boolean.
    pub fn bool(nullable: bool) -> Self {
        Self::literal(kind::BOOL, nullable)
    }

    /// A fixed-width numeric.
    pub fn primitive(ptype: PType, nullable: bool) -> Self {
        let mut expr = Self::literal(kind::PRIMITIVE, nullable);
        expr.bytes.push(ptype as u8);
        expr
    }

    /// A fixed-precision decimal.
    pub fn decimal(precision: u8, scale: i8, nullable: bool) -> Self {
        let mut expr = Self::literal(kind::DECIMAL, nullable);
        expr.bytes.push(precision);
        expr.bytes.push(scale as u8);
        expr
    }

    /// A UTF-8 string.
    pub fn utf8(nullable: bool) -> Self {
        Self::literal(kind::UTF8, nullable)
    }

    /// Binary data.
    pub fn binary(nullable: bool) -> Self {
        Self::literal(kind::BINARY, nullable)
    }

    /// A variable-length list of `element`.
    pub fn list(element: DTypeExpr, nullable: bool) -> Self {
        let mut expr = Self::literal(kind::LIST, nullable);
        expr.bytes.extend_from_slice(&element.bytes);
        expr
    }

    /// A fixed-size list of `size` elements.
    pub fn fixed_size_list(element: DTypeExpr, size: u32, nullable: bool) -> Self {
        let mut expr = Self::literal(kind::FIXED_SIZE_LIST, nullable);
        write_varint(&mut expr.bytes, u64::from(size));
        expr.bytes.extend_from_slice(&element.bytes);
        expr
    }

    /// A struct of `(name, dtype)` fields.
    pub fn struct_(
        fields: impl IntoIterator<Item = (&'static str, DTypeExpr)>,
        nullable: bool,
    ) -> Self {
        let mut body = Vec::new();
        let mut n = 0u64;
        for (name, dtype) in fields {
            write_varint(&mut body, name.len() as u64);
            body.extend_from_slice(name.as_bytes());
            body.extend_from_slice(&dtype.bytes);
            n += 1;
        }
        let mut expr = Self::literal(kind::STRUCT, nullable);
        write_varint(&mut expr.bytes, n);
        expr.bytes.extend_from_slice(&body);
        expr
    }

    /// A union of `(name, type_id, dtype)` variants.
    ///
    /// `type_id` is the tag the data uses to select the variant, and need not be its position.
    pub fn union_(
        variants: impl IntoIterator<Item = (&'static str, u8, DTypeExpr)>,
        nullable: bool,
    ) -> Self {
        let mut body = Vec::new();
        let mut n = 0u64;
        for (name, type_id, dtype) in variants {
            write_varint(&mut body, name.len() as u64);
            body.extend_from_slice(name.as_bytes());
            body.push(type_id);
            body.extend_from_slice(&dtype.bytes);
            n += 1;
        }
        let mut expr = Self::literal(kind::UNION, nullable);
        write_varint(&mut expr.bytes, n);
        expr.bytes.extend_from_slice(&body);
        expr
    }

    /// An extension type: `id` and `metadata` as the host's registry expects them, over `storage`.
    ///
    /// The host resolves `id` in its dtype registry and hands `metadata` to that type's own
    /// deserializer, so this can name any extension type the reader has registered. For the
    /// node's own extension type, or one reachable from it, prefer [`Self::parent`] and
    /// [`Self::storage`]: they need no knowledge of the metadata format at all.
    pub fn extension(id: &str, metadata: &[u8], storage: DTypeExpr, nullable: bool) -> Self {
        let mut expr = Self::literal(kind::EXTENSION, nullable);
        write_varint(&mut expr.bytes, id.len() as u64);
        expr.bytes.extend_from_slice(id.as_bytes());
        write_varint(&mut expr.bytes, metadata.len() as u64);
        expr.bytes.extend_from_slice(metadata);
        expr.bytes.extend_from_slice(&storage.bytes);
        expr
    }

    /// A map from `key` to `value`. Keys must be non-nullable.
    pub fn map(key: DTypeExpr, value: DTypeExpr, keys_sorted: bool, nullable: bool) -> Self {
        let mut expr = Self::literal(kind::MAP, nullable);
        expr.bytes.push(u8::from(keys_sorted));
        expr.bytes.extend_from_slice(&key.bytes);
        expr.bytes.extend_from_slice(&value.bytes);
        expr
    }

    fn derived(op: u8, inner: DTypeExpr) -> Self {
        let mut bytes = Vec::with_capacity(1 + inner.bytes.len());
        bytes.push(dtype_tag::DERIVED | op);
        bytes.extend_from_slice(&inner.bytes);
        Self { bytes }
    }

    /// Struct field `index` of `inner`.
    pub fn field(inner: DTypeExpr, index: u32) -> Self {
        let mut bytes = Vec::with_capacity(2 + inner.bytes.len());
        bytes.push(dtype_tag::DERIVED | derive_op::FIELD);
        write_varint(&mut bytes, u64::from(index));
        bytes.extend_from_slice(&inner.bytes);
        Self { bytes }
    }

    /// The element type of the list `inner`, or the `{key, value}` entry struct of the map `inner`.
    pub fn element(inner: DTypeExpr) -> Self {
        Self::derived(derive_op::ELEMENT, inner)
    }

    /// The storage type of the extension type `inner`.
    pub fn storage(inner: DTypeExpr) -> Self {
        Self::derived(derive_op::STORAGE, inner)
    }

    /// `inner`, made nullable.
    pub fn nullable(inner: DTypeExpr) -> Self {
        Self::derived(derive_op::NULLABLE, inner)
    }

    /// `inner`, made non-nullable.
    pub fn non_nullable(inner: DTypeExpr) -> Self {
        Self::derived(derive_op::NON_NULLABLE, inner)
    }

    /// The encoded bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}
