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
//! Derivations are what make the channel complete rather than merely wide. Extension types resolve
//! through a vtable registry, so no byte encoding lets a guest construct one; a literal-only
//! channel would leave every extension-typed child unnameable no matter how many kinds it spelled.
//! A derivation sidesteps the problem by never having the guest hold the type at all — it names a
//! path, and the host walks it against a `DType` it already trusts.

use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

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

/// Cap on struct fields in a guest-written type, so a varint count cannot drive an unbounded loop.
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
        DType::Union(_) => out.push(KIND_UNION | n),
        DType::Variant(_) => out.push(KIND_VARIANT | n),
        DType::Extension(ext) => {
            out.push(KIND_EXTENSION | n);
            let id = ext.id();
            write_varint(out, id.as_ref().len() as u64);
            out.extend_from_slice(id.as_ref().as_bytes());
            // Extension metadata is not part of the guest-visible contract yet: a kernel can read
            // the id and the storage type, which is what distinguishes the type, and nothing here
            // could rebuild the vtable anyway.
            write_varint(out, 0);
            encode_into(out, ext.storage_dtype(), depth + 1)?;
        }
    }
    Ok(())
}

/// Decode a type expression written by a kernel, resolving derivations against `parent`.
///
/// Returns the type and the number of bytes consumed. `parent` is the dtype of the node being
/// decoded — the anchor every derivation is relative to.
pub fn decode(bytes: &[u8], parent: &DType) -> VortexResult<(DType, usize)> {
    decode_at(bytes, 0, parent, 0)
}

fn decode_at(
    bytes: &[u8],
    offset: usize,
    parent: &DType,
    depth: usize,
) -> VortexResult<(DType, usize)> {
    vortex_ensure!(depth <= MAX_DEPTH, "dtype expression nested too deeply");
    let tag = *bytes
        .get(offset)
        .ok_or_else(|| vortex_err!("truncated dtype expression"))?;

    if tag & DERIVED != 0 {
        return decode_derivation(bytes, offset, parent, depth, tag & KIND_MASK);
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
            let (element, n) = decode_at(bytes, offset + consumed, parent, depth + 1)?;
            consumed += n;
            DType::List(element.into(), nullability)
        }
        KIND_FIXED_SIZE_LIST => {
            let (size, n) = read_varint(bytes, offset + consumed)?;
            consumed += n;
            let (element, n) = decode_at(bytes, offset + consumed, parent, depth + 1)?;
            consumed += n;
            DType::FixedSizeList(element.into(), u32::try_from(size)?, nullability)
        }
        KIND_STRUCT => {
            let (n_fields, n) = read_varint(bytes, offset + consumed)?;
            consumed += n;
            let n_fields = usize::try_from(n_fields)?;
            vortex_ensure!(
                n_fields <= MAX_FIELDS,
                "dtype expression declares {n_fields} struct fields, more than the {MAX_FIELDS} allowed"
            );
            let mut names: Vec<FieldName> = Vec::with_capacity(n_fields);
            let mut fields = Vec::with_capacity(n_fields);
            for _ in 0..n_fields {
                let (name_len, n) = read_varint(bytes, offset + consumed)?;
                consumed += n;
                let name_len = usize::try_from(name_len)?;
                let name = bytes
                    .get(offset + consumed..offset + consumed + name_len)
                    .ok_or_else(|| vortex_err!("truncated struct field name"))?;
                consumed += name_len;
                names.push(
                    std::str::from_utf8(name)
                        .map_err(|_| vortex_err!("struct field name is not valid UTF-8"))?
                        .into(),
                );
                let (field, n) = decode_at(bytes, offset + consumed, parent, depth + 1)?;
                consumed += n;
                fields.push(field);
            }
            DType::Struct(StructFields::new(names.into(), fields), nullability)
        }
        KIND_UNION => DType::Union(nullability),
        KIND_VARIANT => DType::Variant(nullability),
        KIND_EXTENSION => {
            // Reconstructing an extension type needs its vtable, which lives in a host registry
            // and cannot be conjured from bytes. A kernel that needs one should derive it from the
            // parent instead of spelling it out.
            vortex_bail!(
                "a kernel cannot write an extension dtype literal; derive it from the parent instead"
            )
        }
        other => vortex_bail!("bad dtype kind {other} in a kernel's type expression"),
    };
    Ok((dtype, consumed))
}

fn decode_derivation(
    bytes: &[u8],
    offset: usize,
    parent: &DType,
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

    let (inner, n) = decode_at(bytes, offset + consumed, parent, depth + 1)?;
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
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;

    use super::*;

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
    #[case(DType::Union(Nullability::Nullable))]
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
        // A struct promising a field it does not carry.
        assert!(decode(&[KIND_STRUCT, 1], &DType::Null).is_err());
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

    /// The host writes extension types so a kernel can inspect them; the kernel cannot write one
    /// back, because rebuilding the vtable is not something bytes can do.
    #[test]
    fn extension_types_encode_but_do_not_decode_as_literals() -> VortexResult<()> {
        let ext = DType::Extension(
            Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased(),
        );
        let DType::Extension(inner) = &ext else {
            unreachable!()
        };
        let storage = inner.storage_dtype().clone();
        let bytes = encode(&ext)?;
        assert_eq!(bytes[0] & KIND_MASK, KIND_EXTENSION);
        // Round-tripping the literal is refused with a message pointing at the alternative.
        let err = decode(&bytes, &DType::Null).unwrap_err().to_string();
        assert!(err.contains("derive it from the parent"), "{err}");
        // ...and the alternative works.
        assert_eq!(
            decode(&[DERIVED | DERIVE_STORAGE, DERIVED | DERIVE_PARENT], &ext)?.0,
            storage
        );
        Ok(())
    }
}
