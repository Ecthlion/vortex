// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The host half of the `DType` wire codec. See the guest SDK's `dtype` module for the grammar.
//!
//! Two directions, deliberately asymmetric:
//!
//! - [`encode`] writes a [`DType`] as literal bytes for the guest to inspect. The host holds the
//!   real type and has nothing to derive from, so it never emits a derivation.
//! - [`decode`] reads a type expression *written by an untrusted kernel*. That expression may be a
//!   literal or a **derivation** — a path from the node's own dtype, such as "struct field 2 of my
//!   parent, made non-nullable".
//!
//! Every `DType` Vortex defines has a literal spelling, extension types included: an extension
//! literal carries the id, the vtable's own serialized metadata, and the storage type, and the
//! host rebuilds it exactly as the file-footer path does — through the session's extension
//! registry, or as a foreign placeholder when the session allows unknown types. Derivations are a
//! convenience on top: a kernel that only re-arranges its parent's values can say "my parent's
//! type" without re-spelling (or even understanding) it.

use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::dtype::UnionVariants;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ForeignExtDType;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

/// Tag byte layout, mirroring the guest SDK's `abi::dtype_tag`.
const KIND_MASK: u8 = 0x3f;
const NULLABLE: u8 = 0x40;
const DERIVED: u8 = 0x80;

/// Literal kinds, mirroring the guest SDK's `abi::dtype_kind`.
const KIND_NULL: u8 = 0;
const KIND_BOOL: u8 = 1;
const KIND_PRIMITIVE: u8 = 2;
const KIND_DECIMAL: u8 = 3;
const KIND_UTF8: u8 = 4;
const KIND_BINARY: u8 = 5;
const KIND_LIST: u8 = 6;
const KIND_FIXED_SIZE_LIST: u8 = 7;
const KIND_STRUCT: u8 = 8;
const KIND_UNION: u8 = 9;
const KIND_VARIANT: u8 = 10;
const KIND_EXTENSION: u8 = 11;
const KIND_MAP: u8 = 12;

/// Derivation opcodes, mirroring the guest SDK's `abi::dtype_derivation`.
const DERIVE_PARENT: u8 = 0;
const DERIVE_FIELD: u8 = 1;
const DERIVE_ELEMENT: u8 = 2;
const DERIVE_STORAGE: u8 = 3;
const DERIVE_NULLABLE: u8 = 4;
const DERIVE_NON_NULLABLE: u8 = 5;

/// Maximum type nesting either side will handle.
///
/// The guest's own limit is the same. On this side it bounds recursion over attacker-controlled
/// bytes, so it is a safety property rather than a courtesy: without it a kernel could return a
/// few hundred bytes of nested `List` tags and overflow the host stack.
pub(crate) const MAX_DEPTH: usize = 32;

/// Cap on struct fields or union variants in a guest-written type, so a varint count cannot
/// drive an unbounded loop.
const MAX_FIELDS: usize = 4096;

fn write_varint(out: &mut Vec<u8>, mut value: u64) {
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

fn read_varint(bytes: &[u8], mut offset: usize) -> VortexResult<(u64, usize)> {
    let start = offset;
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *bytes
            .get(offset)
            .ok_or_else(|| vortex_err!("truncated dtype varint"))?;
        offset += 1;
        value |= u64::from(byte & 0x7f)
            .checked_shl(shift)
            .ok_or_else(|| vortex_err!("dtype varint overflow"))?;
        if byte & 0x80 == 0 {
            return Ok((value, offset - start));
        }
        shift += 7;
        vortex_ensure!(shift < 64, "dtype varint overflow");
    }
}

fn nullable_bit(dtype: &DType) -> u8 {
    if dtype.is_nullable() { NULLABLE } else { 0 }
}

/// Encode a [`DType`] as literal bytes.
pub fn encode(dtype: &DType) -> VortexResult<Vec<u8>> {
    let mut out = Vec::new();
    encode_into(&mut out, dtype, 0)?;
    Ok(out)
}

fn encode_into(out: &mut Vec<u8>, dtype: &DType, depth: usize) -> VortexResult<()> {
    vortex_ensure!(depth <= MAX_DEPTH, "dtype nested too deeply to encode");
    let n = nullable_bit(dtype);
    match dtype {
        DType::Null => out.push(KIND_NULL),
        DType::Bool(_) => out.push(KIND_BOOL | n),
        DType::Primitive(ptype, _) => {
            out.push(KIND_PRIMITIVE | n);
            out.push(*ptype as u8);
        }
        DType::Decimal(decimal, _) => {
            out.push(KIND_DECIMAL | n);
            out.push(decimal.precision());
            out.push(decimal.scale() as u8);
        }
        DType::Utf8(_) => out.push(KIND_UTF8 | n),
        DType::Binary(_) => out.push(KIND_BINARY | n),
        DType::List(element, _) => {
            out.push(KIND_LIST | n);
            encode_into(out, element, depth + 1)?;
        }
        DType::FixedSizeList(element, size, _) => {
            out.push(KIND_FIXED_SIZE_LIST | n);
            write_varint(out, u64::from(*size));
            encode_into(out, element, depth + 1)?;
        }
        DType::Struct(fields, _) => {
            out.push(KIND_STRUCT | n);
            write_varint(out, fields.nfields() as u64);
            for (name, field) in fields.names().iter().zip(fields.fields()) {
                let name = name.as_ref();
                write_varint(out, name.len() as u64);
                out.extend_from_slice(name.as_bytes());
                encode_into(out, &field, depth + 1)?;
            }
        }
        DType::Union(variants, _) => {
            // A struct entry plus the variant's type tag, which the data uses to select it and
            // need not be the variant's position.
            out.push(KIND_UNION | n);
            write_varint(out, variants.len() as u64);
            for ((name, variant), type_id) in variants
                .names()
                .iter()
                .zip(variants.variants())
                .zip(variants.type_ids())
            {
                let name = name.as_ref();
                write_varint(out, name.len() as u64);
                out.extend_from_slice(name.as_bytes());
                out.push(*type_id);
                encode_into(out, &variant, depth + 1)?;
            }
        }
        DType::Variant(_) => out.push(KIND_VARIANT | n),
        DType::Extension(ext) => {
            out.push(KIND_EXTENSION | n);
            let id = ext.id();
            write_varint(out, id.as_ref().len() as u64);
            out.extend_from_slice(id.as_ref().as_bytes());
            // The vtable's own serialization, exactly as the file footer records it, so a kernel
            // can echo the type back and the host can rebuild it through the same registry.
            let metadata = ext.serialize_metadata()?;
            write_varint(out, metadata.len() as u64);
            out.extend_from_slice(&metadata);
            encode_into(out, ext.storage_dtype(), depth + 1)?;
        }
        DType::Map(map, _) => {
            out.push(KIND_MAP | n);
            out.push(u8::from(map.keys_sorted()));
            encode_into(out, &map.key_dtype(), depth + 1)?;
            encode_into(out, &map.value_dtype(), depth + 1)?;
        }
    }
    Ok(())
}

/// Decode a type expression written by a kernel, resolving derivations against `parent`.
///
/// Returns the type and the number of bytes consumed. `parent` is the dtype of the node being
/// decoded — the anchor every derivation is relative to. `session` resolves extension literals:
/// a registered plugin rebuilds its type from the metadata; an unknown id becomes a foreign
/// placeholder if the session allows unknown types, and is an error otherwise.
pub fn decode(
    bytes: &[u8],
    parent: &DType,
    session: &VortexSession,
) -> VortexResult<(DType, usize)> {
    decode_at(bytes, 0, parent, session, 0)
}

fn decode_at(
    bytes: &[u8],
    offset: usize,
    parent: &DType,
    session: &VortexSession,
    depth: usize,
) -> VortexResult<(DType, usize)> {
    vortex_ensure!(depth <= MAX_DEPTH, "dtype expression nested too deeply");
    let tag = *bytes
        .get(offset)
        .ok_or_else(|| vortex_err!("truncated dtype expression"))?;

    if tag & DERIVED != 0 {
        return decode_derivation(bytes, offset, parent, session, depth, tag & KIND_MASK);
    }

    let nullability = if tag & NULLABLE != 0 {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    };
    let mut consumed = 1usize;
    let dtype = match tag & KIND_MASK {
        KIND_NULL => DType::Null,
        KIND_BOOL => DType::Bool(nullability),
        KIND_PRIMITIVE => {
            let discriminant = *bytes
                .get(offset + consumed)
                .ok_or_else(|| vortex_err!("truncated primitive dtype"))?;
            consumed += 1;
            let ptype = PType::try_from(i32::from(discriminant))
                .map_err(|_| vortex_err!("bad ptype {discriminant} in dtype"))?;
            DType::Primitive(ptype, nullability)
        }
        KIND_DECIMAL => {
            let payload = bytes
                .get(offset + consumed..offset + consumed + 2)
                .ok_or_else(|| vortex_err!("truncated decimal dtype"))?;
            consumed += 2;
            DType::Decimal(
                DecimalDType::try_new(payload[0], payload[1] as i8)?,
                nullability,
            )
        }
        KIND_UTF8 => DType::Utf8(nullability),
        KIND_BINARY => DType::Binary(nullability),
        KIND_LIST => {
            let (element, n) = decode_at(bytes, offset + consumed, parent, session, depth + 1)?;
            consumed += n;
            DType::List(element.into(), nullability)
        }
        KIND_FIXED_SIZE_LIST => {
            let (size, n) = read_varint(bytes, offset + consumed)?;
            consumed += n;
            let (element, n) = decode_at(bytes, offset + consumed, parent, session, depth + 1)?;
            consumed += n;
            DType::FixedSizeList(element.into(), u32::try_from(size)?, nullability)
        }
        KIND_STRUCT => {
            let (entries, n) =
                decode_named_entries(bytes, offset + consumed, parent, session, depth, false)?;
            consumed += n;
            DType::Struct(
                StructFields::new(entries.names.into(), entries.dtypes),
                nullability,
            )
        }
        KIND_UNION => {
            let (entries, n) =
                decode_named_entries(bytes, offset + consumed, parent, session, depth, true)?;
            consumed += n;
            DType::Union(
                UnionVariants::try_new(entries.names.into(), entries.dtypes, entries.type_ids)?,
                nullability,
            )
        }
        KIND_VARIANT => DType::Variant(nullability),
        KIND_EXTENSION => {
            let (id_len, n) = read_varint(bytes, offset + consumed)?;
            consumed += n;
            let id_len = usize::try_from(id_len)?;
            let id = bytes
                .get(offset + consumed..offset + consumed + id_len)
                .ok_or_else(|| vortex_err!("truncated extension dtype id"))?;
            consumed += id_len;
            let id = ExtId::from(
                std::str::from_utf8(id)
                    .map_err(|_| vortex_err!("extension dtype id is not valid UTF-8"))?,
            );

            let (metadata_len, n) = read_varint(bytes, offset + consumed)?;
            consumed += n;
            let metadata_len = usize::try_from(metadata_len)?;
            let metadata = bytes
                .get(offset + consumed..offset + consumed + metadata_len)
                .ok_or_else(|| vortex_err!("truncated extension dtype metadata"))?;
            consumed += metadata_len;

            let (storage, n) = decode_at(bytes, offset + consumed, parent, session, depth + 1)?;
            consumed += n;

            // The same resolution the file footer's dtype goes through: the registered plugin
            // validates the metadata against the storage type it is given.
            let plugin = session.dtypes().registry().get(&id);
            let ext = if let Some(plugin) = plugin {
                plugin.deserialize(metadata, storage)?
            } else if session.allows_unknown() {
                ForeignExtDType::from_parts(id, metadata.to_vec(), storage)?
            } else {
                vortex_bail!(
                    "a kernel named extension dtype {id}, which this session does not know"
                )
            };
            // An extension's nullability is its storage's; the tag bit is authoritative if the
            // kernel spelled them inconsistently.
            DType::Extension(ext.with_nullability(nullability))
        }
        KIND_MAP => {
            let keys_sorted = *bytes
                .get(offset + consumed)
                .ok_or_else(|| vortex_err!("truncated map dtype"))?
                != 0;
            consumed += 1;
            let (key, n) = decode_at(bytes, offset + consumed, parent, session, depth + 1)?;
            consumed += n;
            let (value, n) = decode_at(bytes, offset + consumed, parent, session, depth + 1)?;
            consumed += n;
            // `try_new` is where a nullable key — which Arrow maps forbid — is rejected.
            DType::Map(MapDType::try_new(key, value, keys_sorted)?, nullability)
        }
        other => vortex_bail!("bad dtype kind {other} in a kernel's type expression"),
    };
    Ok((dtype, consumed))
}

/// The named, typed entries of a struct or union literal.
struct NamedEntries {
    names: Vec<FieldName>,
    dtypes: Vec<DType>,
    /// Union type tags; empty for a struct.
    type_ids: Vec<u8>,
}

/// Decode `varint n, n × (varint name_len, name, [u8 type_id,] dtype)` — the shared body of the
/// struct and union productions, which differ only in whether each entry carries a type tag.
fn decode_named_entries(
    bytes: &[u8],
    offset: usize,
    parent: &DType,
    session: &VortexSession,
    depth: usize,
    with_type_ids: bool,
) -> VortexResult<(NamedEntries, usize)> {
    let (count, mut consumed) = read_varint(bytes, offset)?;
    let count = usize::try_from(count)?;
    vortex_ensure!(
        count <= MAX_FIELDS,
        "dtype expression declares {count} named entries, more than the {MAX_FIELDS} allowed"
    );
    let mut entries = NamedEntries {
        names: Vec::with_capacity(count),
        dtypes: Vec::with_capacity(count),
        type_ids: Vec::with_capacity(if with_type_ids { count } else { 0 }),
    };
    for _ in 0..count {
        let (name_len, n) = read_varint(bytes, offset + consumed)?;
        consumed += n;
        let name_len = usize::try_from(name_len)?;
        let name = bytes
            .get(offset + consumed..offset + consumed + name_len)
            .ok_or_else(|| vortex_err!("truncated dtype entry name"))?;
        consumed += name_len;
        entries.names.push(
            std::str::from_utf8(name)
                .map_err(|_| vortex_err!("dtype entry name is not valid UTF-8"))?
                .into(),
        );
        if with_type_ids {
            entries.type_ids.push(
                *bytes
                    .get(offset + consumed)
                    .ok_or_else(|| vortex_err!("truncated union type tag"))?,
            );
            consumed += 1;
        }
        let (dtype, n) = decode_at(bytes, offset + consumed, parent, session, depth + 1)?;
        consumed += n;
        entries.dtypes.push(dtype);
    }
    Ok((entries, consumed))
}

fn decode_derivation(
    bytes: &[u8],
    offset: usize,
    parent: &DType,
    session: &VortexSession,
    depth: usize,
    op: u8,
) -> VortexResult<(DType, usize)> {
    let mut consumed = 1usize;
    if op == DERIVE_PARENT {
        return Ok((parent.clone(), consumed));
    }

    // FIELD carries its index before the inner expression; the rest are pure unary operators.
    let index = if op == DERIVE_FIELD {
        let (index, n) = read_varint(bytes, offset + consumed)?;
        consumed += n;
        usize::try_from(index)?
    } else {
        0
    };

    let (inner, n) = decode_at(bytes, offset + consumed, parent, session, depth + 1)?;
    consumed += n;

    let derived = match op {
        DERIVE_FIELD => {
            let DType::Struct(fields, _) = &inner else {
                vortex_bail!("cannot take field {index} of non-struct dtype {inner}")
            };
            fields.field_by_index(index).ok_or_else(|| {
                vortex_err!(
                    "field index {index} out of bounds for a struct with {} fields",
                    fields.nfields()
                )
            })?
        }
        DERIVE_ELEMENT => match &inner {
            DType::List(element, _) | DType::FixedSizeList(element, ..) => element.as_ref().clone(),
            // A map's "element" is its `{key, value}` entry, which is how map arrays store it.
            DType::Map(map, _) => map.entries_dtype(),
            other => vortex_bail!("cannot take the element type of non-list dtype {other}"),
        },
        DERIVE_STORAGE => match &inner {
            DType::Extension(ext) => ext.storage_dtype().clone(),
            other => vortex_bail!("cannot take the storage type of non-extension dtype {other}"),
        },
        DERIVE_NULLABLE => inner.as_nullable(),
        DERIVE_NON_NULLABLE => inner.as_nonnullable(),
        other => vortex_bail!("bad dtype derivation opcode {other}"),
    };
    Ok((derived, consumed))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::array_session;
    use vortex_array::dtype::extension::ExtDTypeRef;
    use vortex_array::extension::datetime::Date;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;

    use super::*;

    /// Decode against a default session, which knows the built-in extension types.
    fn decode(bytes: &[u8], parent: &DType) -> VortexResult<(DType, usize)> {
        super::decode(bytes, parent, &array_session())
    }

    fn struct_of(fields: Vec<(&str, DType)>) -> DType {
        let names: Vec<FieldName> = fields.iter().map(|(name, _)| (*name).into()).collect();
        DType::Struct(
            StructFields::new(names.into(), fields.into_iter().map(|(_, d)| d).collect()),
            Nullability::NonNullable,
        )
    }

    #[rstest]
    #[case(DType::Null)]
    #[case(DType::Bool(Nullability::Nullable))]
    #[case(DType::Bool(Nullability::NonNullable))]
    #[case(DType::Primitive(PType::I64, Nullability::Nullable))]
    #[case(DType::Primitive(PType::F16, Nullability::NonNullable))]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Binary(Nullability::NonNullable))]
    #[case(DType::Variant(Nullability::Nullable))]
    fn scalar_dtypes_round_trip(#[case] dtype: DType) -> VortexResult<()> {
        let bytes = encode(&dtype)?;
        let (decoded, consumed) = decode(&bytes, &DType::Null)?;
        assert_eq!(decoded, dtype);
        assert_eq!(consumed, bytes.len());
        Ok(())
    }

    #[test]
    fn nested_dtypes_round_trip() -> VortexResult<()> {
        let dtype = struct_of(vec![
            ("a", DType::Primitive(PType::U8, Nullability::Nullable)),
            (
                "nested",
                struct_of(vec![(
                    "list",
                    DType::List(
                        DType::Utf8(Nullability::Nullable).into(),
                        Nullability::NonNullable,
                    ),
                )]),
            ),
            (
                "fsl",
                DType::FixedSizeList(
                    DType::Primitive(PType::F64, Nullability::NonNullable).into(),
                    7,
                    Nullability::Nullable,
                ),
            ),
        ]);
        let bytes = encode(&dtype)?;
        let (decoded, consumed) = decode(&bytes, &DType::Null)?;
        assert_eq!(decoded, dtype);
        assert_eq!(consumed, bytes.len());
        Ok(())
    }

    /// A union carries its variants' type tags, which need not be their positions.
    #[test]
    fn union_dtypes_round_trip_with_their_type_tags() -> VortexResult<()> {
        let names: Vec<FieldName> = vec!["int".into(), "text".into()];
        let dtype = DType::Union(
            UnionVariants::try_new(
                names.into(),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Utf8(Nullability::Nullable),
                ],
                vec![3, 7],
            )?,
            Nullability::Nullable,
        );
        let bytes = encode(&dtype)?;
        let (decoded, consumed) = decode(&bytes, &DType::Null)?;
        assert_eq!(decoded, dtype);
        assert_eq!(consumed, bytes.len());
        let DType::Union(variants, _) = decoded else {
            unreachable!()
        };
        assert_eq!(variants.type_ids(), &[3, 7]);
        Ok(())
    }

    #[test]
    fn map_dtypes_round_trip_and_derive_their_entries() -> VortexResult<()> {
        let map = MapDType::try_new(
            DType::Utf8(Nullability::NonNullable),
            DType::Primitive(PType::I64, Nullability::Nullable),
            true,
        )?;
        let entries = map.entries_dtype();
        let dtype = DType::Map(map, Nullability::Nullable);
        let bytes = encode(&dtype)?;
        let (decoded, consumed) = decode(&bytes, &DType::Null)?;
        assert_eq!(decoded, dtype);
        assert_eq!(consumed, bytes.len());
        // ELEMENT(PARENT) of a map is its `{key, value}` entry struct.
        assert_eq!(
            decode(&[DERIVED | DERIVE_ELEMENT, DERIVED | DERIVE_PARENT], &dtype)?.0,
            entries
        );
        // A literal with a nullable key is refused, as the native constructor refuses it.
        let mut nullable_key = vec![KIND_MAP, 0];
        nullable_key.extend(encode(&DType::Utf8(Nullability::Nullable))?);
        nullable_key.extend(encode(&DType::Null)?);
        assert!(decode(&nullable_key, &DType::Null).is_err());
        Ok(())
    }

    #[test]
    fn decimal_round_trips_with_a_negative_scale() -> VortexResult<()> {
        let dtype = DType::Decimal(DecimalDType::try_new(19, -3)?, Nullability::Nullable);
        let bytes = encode(&dtype)?;
        assert_eq!(decode(&bytes, &DType::Null)?.0, dtype);
        Ok(())
    }

    /// A dtype embedded in a larger frame: decoding must report what it consumed and ignore the
    /// rest, which is what lets the child-descriptor table pack dtypes back to back.
    #[test]
    fn decoding_stops_at_the_end_of_the_dtype() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let mut bytes = encode(&dtype)?;
        let prefix_len = bytes.len();
        bytes.extend_from_slice(b"trailing garbage");
        let (decoded, consumed) = decode(&bytes, &DType::Null)?;
        assert_eq!(decoded, dtype);
        assert_eq!(consumed, prefix_len);
        Ok(())
    }

    /// The point of the derivation channel: a type the guest could never spell, named by path.
    #[test]
    fn derivations_resolve_against_the_parent() -> VortexResult<()> {
        let parent = struct_of(vec![
            ("id", DType::Primitive(PType::I64, Nullability::NonNullable)),
            (
                "tags",
                DType::List(
                    DType::Utf8(Nullability::Nullable).into(),
                    Nullability::Nullable,
                ),
            ),
        ]);

        // PARENT
        assert_eq!(decode(&[DERIVED | DERIVE_PARENT], &parent)?.0, parent);

        // FIELD(1, PARENT)
        let field = [DERIVED | DERIVE_FIELD, 1, DERIVED | DERIVE_PARENT];
        assert_eq!(
            decode(&field, &parent)?.0,
            DType::List(
                DType::Utf8(Nullability::Nullable).into(),
                Nullability::Nullable
            )
        );

        // ELEMENT(FIELD(1, PARENT))
        let element = [
            DERIVED | DERIVE_ELEMENT,
            DERIVED | DERIVE_FIELD,
            1,
            DERIVED | DERIVE_PARENT,
        ];
        assert_eq!(
            decode(&element, &parent)?.0,
            DType::Utf8(Nullability::Nullable)
        );

        // NON_NULLABLE(ELEMENT(FIELD(1, PARENT)))
        let non_nullable = [
            DERIVED | DERIVE_NON_NULLABLE,
            DERIVED | DERIVE_ELEMENT,
            DERIVED | DERIVE_FIELD,
            1,
            DERIVED | DERIVE_PARENT,
        ];
        assert_eq!(
            decode(&non_nullable, &parent)?.0,
            DType::Utf8(Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn a_derivation_that_does_not_fit_the_parent_is_rejected() {
        let parent = DType::Primitive(PType::I32, Nullability::NonNullable);
        // FIELD(0, PARENT) where the parent is not a struct.
        let expr = [DERIVED | DERIVE_FIELD, 0, DERIVED | DERIVE_PARENT];
        assert!(decode(&expr, &parent).is_err());
        // ELEMENT(PARENT) where the parent is not a list.
        assert!(
            decode(
                &[DERIVED | DERIVE_ELEMENT, DERIVED | DERIVE_PARENT],
                &parent
            )
            .is_err()
        );
    }

    #[test]
    fn an_out_of_bounds_field_index_is_rejected() {
        let parent = struct_of(vec![("only", DType::Bool(Nullability::NonNullable))]);
        let expr = [DERIVED | DERIVE_FIELD, 9, DERIVED | DERIVE_PARENT];
        assert!(decode(&expr, &parent).is_err());
    }

    /// Recursion over attacker-controlled bytes must terminate at a defined point rather than
    /// running the host stack out.
    #[test]
    fn a_deeply_nested_expression_is_rejected_not_overflowed() {
        let mut bytes = vec![KIND_LIST; MAX_DEPTH + 8];
        bytes.push(KIND_BOOL);
        assert!(decode(&bytes, &DType::Null).is_err());

        let deep_derivation = vec![DERIVED | DERIVE_NULLABLE; MAX_DEPTH + 8];
        assert!(decode(&deep_derivation, &DType::Null).is_err());
    }

    #[test]
    fn a_truncated_expression_is_rejected() {
        assert!(decode(&[], &DType::Null).is_err());
        assert!(decode(&[KIND_PRIMITIVE], &DType::Null).is_err());
        assert!(decode(&[KIND_DECIMAL, 10], &DType::Null).is_err());
        assert!(decode(&[KIND_LIST], &DType::Null).is_err());
        // A struct promising a field it does not carry, and a union whose entry stops before its
        // type tag.
        assert!(decode(&[KIND_STRUCT, 1], &DType::Null).is_err());
        assert!(decode(&[KIND_UNION, 1, 1, b'a'], &DType::Null).is_err());
    }

    #[test]
    fn an_absurd_field_count_is_rejected_before_allocating() {
        // A varint field count of u64::MAX, with no field bytes behind it.
        let mut bytes = vec![KIND_STRUCT];
        write_varint(&mut bytes, u64::MAX);
        assert!(decode(&bytes, &DType::Null).is_err());
    }

    #[test]
    fn bad_kinds_and_opcodes_are_rejected() {
        assert!(decode(&[KIND_MASK], &DType::Null).is_err());
        assert!(decode(&[DERIVED | 63], &DType::Null).is_err());
        // A ptype discriminant Vortex does not define.
        assert!(decode(&[KIND_PRIMITIVE, 99], &DType::Null).is_err());
    }

    /// An extension literal carries the vtable's real metadata, so a kernel can echo a type back
    /// and the host rebuilds it through the session registry — timezone and all.
    #[rstest]
    #[case(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased())]
    #[case(Timestamp::new_with_tz(TimeUnit::Nanoseconds, Some("Europe/Warsaw".into()), Nullability::Nullable).erased())]
    #[case(Date::new(TimeUnit::Days, Nullability::Nullable).erased())]
    fn extension_types_round_trip_as_literals(#[case] ext: ExtDTypeRef) -> VortexResult<()> {
        let storage = ext.storage_dtype().clone();
        let dtype = DType::Extension(ext);
        let bytes = encode(&dtype)?;
        assert_eq!(bytes[0] & KIND_MASK, KIND_EXTENSION);
        let (decoded, consumed) = decode(&bytes, &DType::Null)?;
        assert_eq!(decoded, dtype);
        assert_eq!(consumed, bytes.len());
        // STORAGE(PARENT) still works for kernels that would rather not spell the type.
        assert_eq!(
            decode(&[DERIVED | DERIVE_STORAGE, DERIVED | DERIVE_PARENT], &dtype)?.0,
            storage
        );
        Ok(())
    }

    /// Nullability of an extension type lives on its storage; the tag bit wins if a kernel
    /// spells them inconsistently, so the result always agrees with itself.
    #[test]
    fn extension_tag_nullability_is_authoritative() -> VortexResult<()> {
        let non_nullable =
            DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullability::NonNullable).erased());
        let mut bytes = encode(&non_nullable)?;
        bytes[0] |= NULLABLE;
        let (decoded, _) = decode(&bytes, &DType::Null)?;
        assert_eq!(decoded, non_nullable.as_nullable());
        assert!(decoded.is_nullable());
        Ok(())
    }

    /// An extension id the session has no plugin for: refused by default, a foreign placeholder
    /// (readable as its storage type) when the session opts into unknown types — the same policy
    /// the file footer's dtype follows.
    #[test]
    fn unknown_extension_ids_follow_the_session_policy() -> VortexResult<()> {
        let mut bytes = vec![KIND_EXTENSION | NULLABLE];
        write_varint(&mut bytes, b"acme.money".len() as u64);
        bytes.extend_from_slice(b"acme.money");
        write_varint(&mut bytes, 3);
        bytes.extend_from_slice(&[1, 2, 3]);
        bytes.extend(encode(&DType::Primitive(
            PType::I64,
            Nullability::Nullable,
        ))?);

        let strict = array_session();
        let err = super::decode(&bytes, &DType::Null, &strict)
            .unwrap_err()
            .to_string();
        assert!(err.contains("acme.money"), "{err}");

        let lenient = array_session();
        lenient.allow_unknown();
        let (decoded, consumed) = super::decode(&bytes, &DType::Null, &lenient)?;
        assert_eq!(consumed, bytes.len());
        let DType::Extension(ext) = &decoded else {
            panic!("expected an extension dtype, got {decoded}")
        };
        assert_eq!(ext.id(), ExtId::from("acme.money"));
        assert_eq!(ext.serialize_metadata()?, vec![1, 2, 3]);
        assert_eq!(
            ext.storage_dtype(),
            &DType::Primitive(PType::I64, Nullability::Nullable)
        );
        // And it re-encodes to the same bytes, so a foreign type survives a further hop.
        assert_eq!(encode(&decoded)?, bytes);
        Ok(())
    }

    #[test]
    fn a_truncated_extension_literal_is_rejected() {
        let mut bytes = vec![KIND_EXTENSION];
        write_varint(&mut bytes, 9);
        bytes.extend_from_slice(b"short");
        assert!(decode(&bytes, &DType::Null).is_err());
        // Id complete, metadata promised but missing.
        let mut bytes = vec![KIND_EXTENSION];
        write_varint(&mut bytes, 4);
        bytes.extend_from_slice(b"vort");
        write_varint(&mut bytes, 100);
        assert!(decode(&bytes, &DType::Null).is_err());
    }

    /// One type per `DType` variant, nested inside each other, round-trips through the channel:
    /// the codec covers the whole type system, not a convenient subset.
    #[test]
    fn every_dtype_kind_round_trips_in_one_expression() -> VortexResult<()> {
        let names: Vec<FieldName> = vec!["i".into(), "s".into()];
        let union = DType::Union(
            UnionVariants::try_new(
                names.into(),
                vec![
                    DType::Primitive(PType::I16, Nullability::NonNullable),
                    DType::Utf8(Nullability::Nullable),
                ],
                vec![0, 1],
            )?,
            Nullability::Nullable,
        );
        let map = DType::Map(
            MapDType::try_new(
                DType::Utf8(Nullability::NonNullable),
                DType::Extension(Date::new(TimeUnit::Days, Nullability::Nullable).erased()),
                false,
            )?,
            Nullability::NonNullable,
        );
        let dtype = struct_of(vec![
            ("null", DType::Null),
            ("bool", DType::Bool(Nullability::Nullable)),
            (
                "prim",
                DType::Primitive(PType::F32, Nullability::NonNullable),
            ),
            (
                "dec",
                DType::Decimal(DecimalDType::try_new(38, 10)?, Nullability::Nullable),
            ),
            ("utf8", DType::Utf8(Nullability::NonNullable)),
            ("bin", DType::Binary(Nullability::Nullable)),
            (
                "list",
                DType::List(union.clone().into(), Nullability::Nullable),
            ),
            (
                "fsl",
                DType::FixedSizeList(map.clone().into(), 3, Nullability::NonNullable),
            ),
            ("union", union),
            ("variant", DType::Variant(Nullability::Nullable)),
            (
                "ext",
                DType::Extension(
                    Timestamp::new_with_tz(
                        TimeUnit::Microseconds,
                        Some("UTC".into()),
                        Nullability::NonNullable,
                    )
                    .erased(),
                ),
            ),
            ("map", map),
        ]);
        let bytes = encode(&dtype)?;
        let (decoded, consumed) = decode(&bytes, &DType::Null)?;
        assert_eq!(decoded, dtype);
        assert_eq!(consumed, bytes.len());
        Ok(())
    }
}
