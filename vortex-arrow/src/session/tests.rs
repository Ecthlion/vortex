// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::DictionaryArray;
use arrow_array::FixedSizeBinaryArray;
use arrow_array::Int32Array;
use arrow_array::ListArray as ArrowListArray;
use arrow_array::StringArray;
use arrow_array::StructArray as ArrowStructArray;
use arrow_array::cast::AsArray;
use arrow_buffer::OffsetBuffer;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::Uuid as ArrowUuid;
use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::Dict;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::extension::uuid::Uuid;
use vortex_array::extension::uuid::UuidMetadata;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::*;

/// The Arrow [`Field`] for `dtype`, which every test here requires to exist.
fn arrow_field(session: &ArrowSession, name: &str, dtype: &DType) -> VortexResult<Field> {
    session
        .to_arrow_field(name, dtype)?
        .ok_or_else(|| no_arrow_type(dtype))
}

fn uuid_dtype(nullable: bool) -> DType {
    let storage = DType::FixedSizeList(
        Arc::new(DType::Primitive(PType::U8, Nullability::NonNullable)),
        16,
        nullable.into(),
    );
    DType::Extension(
        ExtDType::try_with_vtable(Uuid, UuidMetadata::default(), storage)
            .expect("uuid ext dtype")
            .erased(),
    )
}

#[test]
fn to_arrow_field_top_level_uuid_carries_extension_metadata() -> VortexResult<()> {
    let session = ArrowSession::default();
    let field = arrow_field(&session, "id", &uuid_dtype(false))?;
    assert!(has_valid_extension_type::<ArrowUuid>(&field));
    Ok(())
}

#[test]
fn to_arrow_field_struct_with_nested_uuid_preserves_metadata() -> VortexResult<()> {
    let session = ArrowSession::default();
    let dtype = DType::Struct(
        StructFields::from_iter([(FieldName::from("id"), uuid_dtype(false))]),
        Nullability::NonNullable,
    );
    let field = arrow_field(&session, "row", &dtype)?;
    let DataType::Struct(inner) = field.data_type() else {
        panic!("expected Struct, got {:?}", field.data_type());
    };
    assert_eq!(inner.len(), 1);
    assert_eq!(inner[0].data_type(), &DataType::FixedSizeBinary(16));
    assert!(has_valid_extension_type::<ArrowUuid>(&inner[0]));
    Ok(())
}

#[test]
fn to_arrow_field_list_of_uuid_preserves_metadata() -> VortexResult<()> {
    let session = ArrowSession::default();
    let dtype = DType::List(Arc::new(uuid_dtype(true)), Nullability::NonNullable);
    let field = arrow_field(&session, "ids", &dtype)?;
    let DataType::List(elem) = field.data_type() else {
        panic!("expected List, got {:?}", field.data_type());
    };
    assert!(has_valid_extension_type::<ArrowUuid>(elem));
    Ok(())
}

#[test]
fn to_arrow_field_fixed_size_list_of_uuid_preserves_metadata() -> VortexResult<()> {
    let session = ArrowSession::default();
    let dtype = DType::FixedSizeList(Arc::new(uuid_dtype(false)), 3, Nullability::NonNullable);
    let field = arrow_field(&session, "triple", &dtype)?;
    let DataType::FixedSizeList(elem, size) = field.data_type() else {
        panic!("expected FixedSizeList, got {:?}", field.data_type());
    };
    assert_eq!(*size, 3);
    assert!(has_valid_extension_type::<ArrowUuid>(elem));
    Ok(())
}

#[test]
fn schema_roundtrip_preserves_map_uuid_fields() -> VortexResult<()> {
    let session = ArrowSession::default();
    let map = DType::map(
        uuid_dtype(false),
        uuid_dtype(true),
        true,
        Nullability::Nullable,
    )?;
    let dtype = DType::Struct(
        StructFields::from_iter([(FieldName::from("ids"), map)]),
        Nullability::NonNullable,
    );

    let schema = session.to_arrow_schema(&dtype)?;
    let field = schema.field(0);
    let DataType::Map(entries, keys_sorted) = field.data_type() else {
        panic!("expected Map, got {:?}", field.data_type());
    };
    assert!(*keys_sorted);
    assert_eq!(entries.name(), "entries");
    assert!(!entries.is_nullable());
    let DataType::Struct(fields) = entries.data_type() else {
        panic!("expected map entries struct, got {:?}", entries.data_type());
    };
    assert!(has_valid_extension_type::<ArrowUuid>(&fields[0]));
    assert!(has_valid_extension_type::<ArrowUuid>(&fields[1]));
    assert!(!fields[0].is_nullable());
    assert!(fields[1].is_nullable());

    assert_eq!(session.from_arrow_schema(&schema)?, dtype);
    Ok(())
}

#[test]
fn to_arrow_schema_struct_of_struct_uuid() -> VortexResult<()> {
    let session = ArrowSession::default();
    let inner = DType::Struct(
        StructFields::from_iter([(FieldName::from("id"), uuid_dtype(true))]),
        Nullability::NonNullable,
    );
    let outer = DType::Struct(
        StructFields::from_iter([(FieldName::from("payload"), inner)]),
        Nullability::NonNullable,
    );
    let schema = session.to_arrow_schema(&outer)?;
    let payload = schema.field(0);
    let DataType::Struct(inner_fields) = payload.data_type() else {
        panic!("expected Struct, got {:?}", payload.data_type());
    };
    assert!(has_valid_extension_type::<ArrowUuid>(&inner_fields[0]));
    Ok(())
}

#[test]
fn from_arrow_field_recurses_into_nested_uuid() -> VortexResult<()> {
    let session = ArrowSession::default();
    let mut elem = Field::new("item", DataType::FixedSizeBinary(16), false);
    elem.try_with_extension_type(ArrowUuid)?;
    let outer = Field::new("ids", DataType::List(Arc::new(elem)), false);

    let dtype = session.from_arrow_field(&outer)?;
    let DType::List(inner_dt, _) = dtype else {
        panic!("expected List dtype, got {dtype}");
    };
    assert!(
        matches!(inner_dt.as_ref(), DType::Extension(ext) if ext.id() == Uuid.id()),
        "expected Uuid extension element, got {inner_dt}",
    );
    Ok(())
}

#[test]
fn schema_roundtrip_preserves_nested_uuid() -> VortexResult<()> {
    let session = ArrowSession::default();
    let dtype = DType::Struct(
        StructFields::from_iter([
            (FieldName::from("id"), uuid_dtype(false)),
            (
                FieldName::from("ids"),
                DType::List(Arc::new(uuid_dtype(true)), Nullability::NonNullable),
            ),
        ]),
        Nullability::NonNullable,
    );
    let schema = session.to_arrow_schema(&dtype)?;
    let roundtripped = session.from_arrow_schema(&schema)?;
    assert_eq!(roundtripped, dtype);
    Ok(())
}

#[test]
fn to_arrow_datatype_dispatches_plugins() -> VortexResult<()> {
    let session = ArrowSession::default();
    assert_eq!(
        session.to_arrow_datatype(&uuid_dtype(false))?,
        DataType::FixedSizeBinary(16)
    );
    assert_eq!(
        session.to_arrow_datatype(&DType::Utf8(Nullability::Nullable))?,
        DataType::Utf8View
    );
    Ok(())
}

#[test]
fn from_arrow_datatype_recurses_into_nested_extension_fields() -> VortexResult<()> {
    let session = ArrowSession::default();
    let mut elem = Field::new("item", DataType::FixedSizeBinary(16), false);
    elem.try_with_extension_type(ArrowUuid)?;
    let data_type = DataType::List(Arc::new(elem));

    let dtype = session.from_arrow_datatype(&data_type, Nullability::Nullable)?;
    let DType::List(inner_dt, Nullability::Nullable) = dtype else {
        panic!("expected nullable List dtype, got {dtype}");
    };
    assert!(
        matches!(inner_dt.as_ref(), DType::Extension(ext) if ext.id() == Uuid.id()),
        "expected Uuid extension element, got {inner_dt}",
    );
    Ok(())
}

#[test]
fn from_arrow_fields_matches_schema_conversion() -> VortexResult<()> {
    let session = ArrowSession::default();
    let fields = Fields::from(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Utf8View, true),
    ]);
    let struct_fields = session.from_arrow_fields(&fields)?;
    let schema_dtype = session.from_arrow_schema(&Schema::new(fields))?;
    assert_eq!(
        schema_dtype,
        DType::Struct(struct_fields, Nullability::NonNullable)
    );
    Ok(())
}

#[test]
fn from_arrow_field_maps_variant_without_importer() -> VortexResult<()> {
    let session = ArrowSession::default();
    let storage = DataType::Struct(
        vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, true),
        ]
        .into(),
    );
    let field = Field::new("v", storage, true).with_metadata(
        [(
            "ARROW:extension:name".to_string(),
            "arrow.parquet.variant".to_string(),
        )]
        .into(),
    );
    assert_eq!(
        session.from_arrow_field(&field)?,
        DType::Variant(Nullability::Nullable)
    );
    Ok(())
}

#[test]
fn execute_arrow_target_none_preserves_top_level_uuid_metadata() -> VortexResult<()> {
    let vortex_session = array_session();
    let mut ctx = vortex_session.create_execution_ctx();
    let session = vortex_session.arrow();

    let mut field = Field::new("id", DataType::FixedSizeBinary(16), false);
    field.try_with_extension_type(ArrowUuid)?;
    let arrow_array: ArrowArrayRef = Arc::new(FixedSizeBinaryArray::try_from_iter(
        [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
    )?);

    let vortex_array = session.from_arrow_array(arrow_array, &field)?;

    let vortex_ext = vortex_array.dtype().as_extension();
    assert!(vortex_ext.is::<Uuid>());

    let exported = session.execute_arrow(vortex_array, None, &mut ctx)?;
    assert_eq!(exported.data_type(), &DataType::FixedSizeBinary(16));
    let fsb = exported.as_fixed_size_binary();
    assert_eq!(fsb.len(), 2);
    assert_eq!(fsb.value(0), b"0123456789abcdef");
    assert_eq!(fsb.value(1), b"fedcba9876543210");
    Ok(())
}

/// Import an Arrow FixedSizeBinary UUID column as a Vortex extension array.
fn uuid_array(session: &ArrowSession) -> VortexResult<ArrayRef> {
    let mut field = Field::new("id", DataType::FixedSizeBinary(16), false);
    field.try_with_extension_type(ArrowUuid)?;
    let arrow_array: ArrowArrayRef = Arc::new(FixedSizeBinaryArray::try_from_iter(
        [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
    )?);
    session.from_arrow_array(arrow_array, &field)
}

/// Exporting a struct that contains an extension column with no target field must still route
/// the column through its export plugin *and* re-attach the Arrow extension metadata to the
/// inferred child field.
#[test]
fn execute_arrow_target_none_preserves_nested_uuid_metadata() -> VortexResult<()> {
    let vortex_session = array_session();
    let mut ctx = vortex_session.create_execution_ctx();
    let session = vortex_session.arrow();

    let uuids = uuid_array(&session)?;
    let struct_array = StructArray::try_new(
        FieldNames::from(["id"]),
        vec![uuids],
        2,
        Validity::NonNullable,
    )?
    .into_array();

    let exported = session.execute_arrow(struct_array, None, &mut ctx)?;
    let DataType::Struct(fields) = exported.data_type() else {
        panic!("expected Struct, got {:?}", exported.data_type());
    };
    assert_eq!(fields[0].data_type(), &DataType::FixedSizeBinary(16));
    assert!(has_valid_extension_type::<ArrowUuid>(&fields[0]));

    let uuids = exported.as_struct().column(0).as_fixed_size_binary();
    assert_eq!(uuids.value(0), b"0123456789abcdef");
    assert_eq!(uuids.value(1), b"fedcba9876543210");
    Ok(())
}

/// Exporting a list of extension elements with no target field must infer an element field that
/// still carries the Arrow extension metadata.
#[test]
fn execute_arrow_target_none_preserves_list_element_uuid_metadata() -> VortexResult<()> {
    let vortex_session = array_session();
    let mut ctx = vortex_session.create_execution_ctx();
    let session = vortex_session.arrow();

    let list = ListArray::try_new(
        uuid_array(&session)?,
        PrimitiveArray::from_iter([0i32, 1, 2]).into_array(),
        Validity::NonNullable,
    )?
    .into_array();

    let exported = session.execute_arrow(list, None, &mut ctx)?;
    let DataType::List(elem) = exported.data_type() else {
        panic!("expected List, got {:?}", exported.data_type());
    };
    assert_eq!(elem.data_type(), &DataType::FixedSizeBinary(16));
    assert!(has_valid_extension_type::<ArrowUuid>(elem));

    let uuids = exported.as_list::<i32>().values().as_fixed_size_binary();
    assert_eq!(uuids.value(0), b"0123456789abcdef");
    assert_eq!(uuids.value(1), b"fedcba9876543210");
    Ok(())
}

/// An Arrow run-end array whose values field carries extension metadata must import as that
/// extension, through both the dtype and the array conversion.
#[test]
fn run_end_recurses_into_extension_values() -> VortexResult<()> {
    let vortex_session = array_session();
    let mut ctx = vortex_session.create_execution_ctx();
    let session = vortex_session.arrow();

    let mut values_field = Field::new("values", DataType::FixedSizeBinary(16), false);
    values_field.try_with_extension_type(ArrowUuid)?;
    let field = Field::new(
        "id",
        DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int32, false)),
            Arc::new(values_field),
        ),
        false,
    );

    let dtype = session.from_arrow_field(&field)?;
    assert!(
        dtype.as_extension().is::<Uuid>(),
        "expected a Uuid extension dtype, got {dtype}"
    );

    let values = FixedSizeBinaryArray::try_from_iter(
        [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
    )?;
    let run_array: ArrowArrayRef = Arc::new(RunArray::<Int32Type>::try_new(
        &Int32Array::from(vec![2i32, 5]),
        &values,
    )?);

    let vortex_array = session.from_arrow_array(run_array, &field)?;
    assert_eq!(vortex_array.len(), 5);
    // The array conversion must agree with the dtype conversion.
    assert_eq!(vortex_array.dtype(), &dtype);

    // And the values must round-trip back out through the export plugin.
    let exported = session.execute_arrow(vortex_array, Some(&field), &mut ctx)?;
    assert_eq!(exported.len(), 5);
    let ree = exported
        .as_any()
        .downcast_ref::<RunArray<Int32Type>>()
        .ok_or_else(|| {
            vortex_err!(
                "expected an Int32 run-end array, got {}",
                exported.data_type()
            )
        })?;
    let values = ree.values().as_fixed_size_binary();
    assert_eq!(values.value(0), b"0123456789abcdef");
    assert_eq!(values.value(1), b"fedcba9876543210");
    Ok(())
}

/// An Arrow dictionary array cannot carry extension metadata on the values themselves, but
/// fields nested inside the values data type can. Both the dtype and the array conversion
/// must dispatch those through their importer, and must agree with each other.
#[test]
fn dictionary_recurses_into_nested_extension_values() -> VortexResult<()> {
    let session = ArrowSession::default();

    let mut elem = Field::new("item", DataType::FixedSizeBinary(16), false);
    elem.try_with_extension_type(ArrowUuid)?;

    let uuids = FixedSizeBinaryArray::try_from_iter(
        [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
    )?;
    let values = ArrowListArray::try_new(
        Arc::new(elem),
        OffsetBuffer::new(vec![0, 1, 2].into()),
        Arc::new(uuids),
        None,
    )?;
    let dict: ArrowArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(
        Int32Array::from(vec![0, 1, 0]),
        Arc::new(values),
    )?);
    let field = Field::new("ids", dict.data_type().clone(), false);

    let dtype = session.from_arrow_field(&field)?;
    let DType::List(elem_dt, _) = &dtype else {
        panic!("expected a List dtype, got {dtype}");
    };
    assert!(
        elem_dt.as_extension().is::<Uuid>(),
        "expected a Uuid extension element, got {elem_dt}"
    );

    let array = session.from_arrow_array(dict, &field)?;
    assert_eq!(array.len(), 3);
    // The array conversion must agree with the dtype conversion.
    assert_eq!(array.dtype(), &dtype);

    // Arrow's dictionary values data type does carry the nested element field, so the
    // extension survives the export as well.
    let mut ctx = array_session().create_execution_ctx();
    let exported = session.execute_arrow(array, Some(&field), &mut ctx)?;
    assert_eq!(exported.data_type(), field.data_type());
    let values = exported.as_any_dictionary().values().as_list::<i32>();
    let uuids = values.values().as_fixed_size_binary();
    assert_eq!(uuids.value(0), b"0123456789abcdef");
    assert_eq!(uuids.value(1), b"fedcba9876543210");
    Ok(())
}

/// Importing by nullability instead of by [`Field`] synthesizes an anonymous field from the
/// array's own data type, so extension metadata on *nested* fields still reaches its importer.
/// A [`bool`] and the equivalent [`Nullability`] must agree.
#[rstest]
#[case(true)]
#[case(false)]
fn from_arrow_array_by_nullability(#[case] nullable: bool) -> VortexResult<()> {
    let session = ArrowSession::default();

    let mut uuid_field = Field::new("id", DataType::FixedSizeBinary(16), false);
    uuid_field.try_with_extension_type(ArrowUuid)?;
    let uuids: ArrowArrayRef = Arc::new(FixedSizeBinaryArray::try_from_iter(
        [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
    )?);
    let arrow_struct: ArrowArrayRef = Arc::new(ArrowStructArray::try_new(
        Fields::from(vec![uuid_field]),
        vec![uuids],
        None,
    )?);

    let array = session.from_arrow_array(ArrowArrayRef::clone(&arrow_struct), nullable)?;
    assert_eq!(array.dtype().nullability(), nullable.into());
    let DType::Struct(fields, _) = array.dtype() else {
        panic!("expected a Struct dtype, got {}", array.dtype());
    };
    assert!(
        fields
            .field_by_index(0)
            .vortex_expect("struct dtype has one field")
            .as_extension()
            .is::<Uuid>(),
        "expected the nested Uuid extension to survive, got {}",
        array.dtype()
    );

    // The `Nullability` form is equivalent to the `bool` form.
    let by_nullability = session.from_arrow_array(arrow_struct, Nullability::from(nullable))?;
    assert_eq!(by_nullability.dtype(), array.dtype());
    Ok(())
}

/// A plain Arrow dictionary imports as a Vortex `Dict` array over the dictionary values,
/// matching the dtype the schema conversion reports.
#[test]
fn dictionary_imports_as_dict_encoding() -> VortexResult<()> {
    let session = ArrowSession::default();
    let dict: ArrowArrayRef = Arc::new(DictionaryArray::<Int32Type>::try_new(
        Int32Array::from(vec![0, 1, 0, 1]),
        Arc::new(StringArray::from(vec!["a", "b"])),
    )?);
    let field = Field::new("s", dict.data_type().clone(), false);

    let dtype = session.from_arrow_field(&field)?;
    assert_eq!(dtype, DType::Utf8(Nullability::NonNullable));

    let array = session.from_arrow_array(dict, &field)?;
    assert!(array.is::<Dict>(), "expected a Dict encoding, got {array}");
    assert_eq!(array.dtype(), &dtype);
    assert_eq!(array.len(), 4);
    Ok(())
}
