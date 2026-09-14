// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use rstest::rstest;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::BoolArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::assert_arrays_eq;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::extension::datetime::TimeUnit;
use vortex::extension::datetime::Timestamp;
use vortex::scalar::DecimalValue;
use vortex::scalar::Scalar;
use vortex::scalar_fn::ScalarFnVTable;
use vortex::scalar_fn::VecExecutionArgs;

use crate::SESSION;
use crate::bloom_filter::BloomFilterContains;
use crate::bloom_filter::BloomFilterProbe;
use crate::bloom_filter::hash::DuckDbHash;
use crate::bloom_filter::hash_scalar;
use crate::bloom_filter::probe::Sectors;
use crate::bloom_filter::try_new_probe;
use crate::cpp;
use crate::duckdb::BloomFilterData;
use crate::duckdb::Connection;
use crate::duckdb::Database;
use crate::duckdb::LogicalType;
use crate::duckdb::Value;

fn connection() -> Connection {
    Database::open_in_memory()
        .unwrap()
        .connect()
        .expect("in-memory connection")
}

/// Builds a probe over a filter DuckDB populated with `keys`, exactly as a hash join would.
fn probe_over(connection: &Connection, dtype: &DType, keys: &[Scalar]) -> BloomFilterProbe {
    let probe = pending_probe(dtype);
    build(connection, &probe, keys);
    probe
}

/// Builds a probe over a filter whose join has not published it yet.
fn pending_probe(dtype: &DType) -> BloomFilterProbe {
    let key_type = LogicalType::try_from(dtype).expect("a DuckDB type for the key");
    try_new_probe(BloomFilterData::new_unbuilt(), &key_type, dtype)
        .expect("DuckDB hashes this key type the way Vortex does")
}

fn build(connection: &Connection, probe: &BloomFilterProbe, keys: &[Scalar]) {
    let values: Vec<Value> = keys
        .iter()
        .cloned()
        .map(Value::try_from)
        .try_collect()
        .expect("DuckDB values for the keys");
    probe.data().build(connection, &values);
}

fn duckdb_hash(scalar: &Scalar) -> u64 {
    let value = Value::try_from(scalar.clone()).expect("a DuckDB value for the scalar");
    unsafe { cpp::duckdb_vx_value_hash(value.as_ptr()) }
}

/// The hashes Vortex derives for join keys have to be the ones DuckDB derived for the keys it
/// inserted, or the filter would rule out rows that do join.
#[rstest]
#[case::i8_negative(Scalar::primitive(-7i8, Nullability::NonNullable))]
#[case::i8(Scalar::primitive(7i8, Nullability::NonNullable))]
#[case::i16_negative(Scalar::primitive(-4242i16, Nullability::NonNullable))]
#[case::i32_negative(Scalar::primitive(-123_456i32, Nullability::NonNullable))]
#[case::i32(Scalar::primitive(123_456i32, Nullability::NonNullable))]
#[case::i64_negative(Scalar::primitive(-9_000_000_000i64, Nullability::NonNullable))]
#[case::i64(Scalar::primitive(9_000_000_000i64, Nullability::NonNullable))]
#[case::u8(Scalar::primitive(250u8, Nullability::NonNullable))]
#[case::u16(Scalar::primitive(65_000u16, Nullability::NonNullable))]
#[case::u32(Scalar::primitive(4_000_000_000u32, Nullability::NonNullable))]
#[case::u64(Scalar::primitive(18_000_000_000_000_000_000u64, Nullability::NonNullable))]
#[case::f32(Scalar::primitive(1.5f32, Nullability::NonNullable))]
#[case::f32_negative_zero(Scalar::primitive(-0.0f32, Nullability::NonNullable))]
#[case::f64(Scalar::primitive(-1.5e100f64, Nullability::NonNullable))]
#[case::f64_negative_zero(Scalar::primitive(-0.0f64, Nullability::NonNullable))]
#[case::bool_true(Scalar::bool(true, Nullability::NonNullable))]
#[case::bool_false(Scalar::bool(false, Nullability::NonNullable))]
#[case::empty_string(Scalar::utf8("", Nullability::NonNullable))]
#[case::inlined_string(Scalar::utf8("vortex", Nullability::NonNullable))]
#[case::eight_byte_string(Scalar::utf8("12345678", Nullability::NonNullable))]
#[case::twelve_byte_string(Scalar::utf8("123456789012", Nullability::NonNullable))]
#[case::long_string(Scalar::utf8(
    "a string that is far too long for DuckDB to inline into a string_t",
    Nullability::NonNullable
))]
#[case::blob(Scalar::binary(vec![0u8, 1, 2, 255], Nullability::NonNullable))]
fn hash_matches_duckdb(#[case] scalar: Scalar) -> VortexResult<()> {
    assert_eq!(hash_scalar(&scalar)?, duckdb_hash(&scalar));
    Ok(())
}

/// Timestamps and dates hash as their storage, which is the physical type DuckDB hashes them as.
#[test]
fn temporal_hash_matches_duckdb() -> VortexResult<()> {
    let ext = Timestamp::new(TimeUnit::Microseconds, Nullability::NonNullable).erased();
    let scalar = Scalar::extension_ref(
        ext,
        Scalar::primitive(1_700_000_000_000_000i64, Nullability::NonNullable),
    );
    assert_eq!(hash_scalar(&scalar)?, duckdb_hash(&scalar));
    Ok(())
}

/// Every key DuckDB inserted must be found, and the filter must still rule most other keys out.
#[test]
fn probe_agrees_with_the_filter_duckdb_built() {
    let connection = connection();
    let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    let present: Vec<Scalar> = (0..1_000i64)
        .map(|key| Scalar::primitive(key, Nullability::NonNullable))
        .collect();
    let probe = probe_over(&connection, &dtype, &present);

    let sectors = probe.data().sectors().expect("the filter was built");
    let sectors = Sectors::new(sectors);

    for key in 0..1_000i64 {
        assert!(sectors.contains(key.duckdb_hash()), "missing key {key}");
    }

    let false_positives = (1_000_000..1_010_000i64)
        .filter(|key| sectors.contains(key.duckdb_hash()))
        .count();
    assert!(
        false_positives < 100,
        "{false_positives} false positives in 10,000 absent keys is far above the rate DuckDB \
         sizes its filters for"
    );
}

/// The join publishes its filter after the scan has already been planned, so the probe has to
/// read the filter's current contents on every batch rather than capture them up front.
#[test]
fn filter_is_picked_up_once_the_join_publishes_it() -> VortexResult<()> {
    let connection = connection();
    let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    let probe = pending_probe(&dtype);

    let keys = PrimitiveArray::from_iter([1i64, 2, 3]).into_array();
    let args = VecExecutionArgs::new(vec![keys], 3);
    let mut ctx = SESSION.create_execution_ctx();

    // While the join is still building, the filter rules out nothing.
    let pending = BloomFilterContains.execute(&probe, &args, &mut ctx)?;
    assert_arrays_eq!(pending, BoolArray::from_iter([true, true, true]), &mut ctx);

    build(
        &connection,
        &probe,
        &[Scalar::primitive(2i64, Nullability::NonNullable)],
    );

    // The same expression now sees the published filter.
    let published = BloomFilterContains.execute(&probe, &args, &mut ctx)?;
    assert_arrays_eq!(
        published,
        BoolArray::from_iter([false, true, false]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn probes_strings() -> VortexResult<()> {
    let connection = connection();
    let dtype = DType::Utf8(Nullability::NonNullable);
    let probe = probe_over(
        &connection,
        &dtype,
        &[
            Scalar::utf8("vortex", Nullability::NonNullable),
            Scalar::utf8(
                "a string that is far too long for DuckDB to inline",
                Nullability::NonNullable,
            ),
        ],
    );

    let keys = VarBinViewArray::from_iter_str([
        "vortex",
        "a string that is far too long for DuckDB to inline",
        "absent",
    ])
    .into_array();
    let args = VecExecutionArgs::new(vec![keys], 3);
    let mut ctx = SESSION.create_execution_ctx();

    assert_arrays_eq!(
        BloomFilterContains.execute(&probe, &args, &mut ctx)?,
        BoolArray::from_iter([true, true, false]),
        &mut ctx
    );
    Ok(())
}

/// An extension key hashes through its storage, so a timestamp column probes the filter a join
/// built over TIMESTAMP keys.
#[test]
fn probes_extension_keys() -> VortexResult<()> {
    let connection = connection();
    let ext = Timestamp::new(TimeUnit::Microseconds, Nullability::NonNullable).erased();
    let dtype = DType::Extension(ext.clone());
    let probe = probe_over(
        &connection,
        &dtype,
        &[Scalar::extension_ref(
            ext.clone(),
            Scalar::primitive(1_700_000_000_000_000i64, Nullability::NonNullable),
        )],
    );

    let keys = ExtensionArray::new(
        ext,
        PrimitiveArray::from_iter([1_700_000_000_000_000i64, 1i64]).into_array(),
    )
    .into_array();
    let args = VecExecutionArgs::new(vec![keys], 2);
    let mut ctx = SESSION.create_execution_ctx();

    assert_arrays_eq!(
        BloomFilterContains.execute(&probe, &args, &mut ctx)?,
        BoolArray::from_iter([true, false]),
        &mut ctx
    );
    Ok(())
}

/// DuckDB hashes a NULL key to a fixed value and probes the filter with it, so a filter built
/// from an equality join's non-null keys rules NULL rows out.
#[test]
fn null_keys_probe_duckdbs_null_hash() -> VortexResult<()> {
    let connection = connection();
    let dtype = DType::Primitive(PType::I64, Nullability::Nullable);
    let probe = probe_over(
        &connection,
        &dtype,
        &[Scalar::primitive(2i64, Nullability::NonNullable)],
    );

    let keys = PrimitiveArray::from_option_iter([Some(2i64), None, Some(9)]).into_array();
    let args = VecExecutionArgs::new(vec![keys], 3);
    let mut ctx = SESSION.create_execution_ctx();

    assert_arrays_eq!(
        BloomFilterContains.execute(&probe, &args, &mut ctx)?,
        BoolArray::from_iter([true, false, false]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn constant_keys_are_probed_once() -> VortexResult<()> {
    let connection = connection();
    let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    let probe = probe_over(
        &connection,
        &dtype,
        &[Scalar::primitive(2i64, Nullability::NonNullable)],
    );

    let mut ctx = SESSION.create_execution_ctx();
    for (key, expected) in [(2i64, true), (9i64, false)] {
        let keys =
            ConstantArray::new(Scalar::primitive(key, Nullability::NonNullable), 4).into_array();
        let args = VecExecutionArgs::new(vec![keys], 4);
        assert_arrays_eq!(
            BloomFilterContains.execute(&probe, &args, &mut ctx)?,
            BoolArray::from_iter([expected; 4]),
            &mut ctx
        );
    }
    Ok(())
}

/// Vortex declines the filter for keys whose values it does not hash the way DuckDB does, rather
/// than ruling out rows that do join. The filter is always pushed as an optional filter, so
/// declining it only costs the rows it would have removed.
#[test]
fn unsupported_key_types_are_declined() {
    let decimal = DType::Decimal(
        DecimalDType::try_new(10, 2).expect("a valid decimal type"),
        Nullability::NonNullable,
    );
    let key_type = LogicalType::try_from(&decimal).expect("a DuckDB type for the key");
    assert!(try_new_probe(BloomFilterData::new_unbuilt(), &key_type, &decimal).is_none());
}

/// A filter built from one key type must not be probed with another, whose values DuckDB would
/// have hashed differently.
#[test]
fn mismatched_key_types_are_declined() {
    let i64_type = LogicalType::try_from(&DType::Primitive(PType::I64, Nullability::NonNullable))
        .expect("a DuckDB type for the key");
    let i32_column = DType::Primitive(PType::I32, Nullability::NonNullable);

    assert!(try_new_probe(BloomFilterData::new_unbuilt(), &i64_type, &i32_column).is_none());
}

/// A decimal scalar has no DuckDB hash Vortex reproduces, so hashing one is an error rather than
/// a wrong answer.
#[test]
fn hashing_an_unsupported_scalar_errors() {
    let scalar = Scalar::decimal(
        DecimalValue::I128(42),
        DecimalDType::new(10, 2),
        Nullability::NonNullable,
    );
    assert!(hash_scalar(&scalar).is_err());
}
