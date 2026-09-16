// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_array::ArrayProbe;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RepeatedArrayProbe;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::Zstd;
use crate::ZstdBuffers;
use crate::ZstdData;

/// A one-off or retained probe over `array`, the retained one living in `retained`.
fn probe_for<'a>(
    array: &'a ArrayRef,
    retained: &'a mut Option<RepeatedArrayProbe>,
    repeated: bool,
) -> ArrayProbe<'a> {
    if repeated {
        retained.insert(array.repeated_probe()).as_probe()
    } else {
        array.probe()
    }
}

/// Indices that revisit frames, jump between them and repeat, so a retained decompression is
/// exercised both within a frame and across them.
const INDICES: [u32; 11] = [0, 1, 63, 64, 128, 127, 191, 3, 65, 1, 0];

#[rstest]
fn primitive_random_access(
    #[values(false, true)] repeated: bool,
    #[values(false, true)] nullable: bool,
    #[values(false, true)] sliced: bool,
    // A single frame, then several frames sharing a trained dictionary.
    #[values(0, 16)] values_per_frame: usize,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = if nullable {
        PrimitiveArray::from_option_iter((0..200i32).map(|i| (i % 7 != 0).then_some(i * 19)))
    } else {
        PrimitiveArray::from_iter((0..200i32).map(|i| i * 19))
    };
    let encoded = Zstd::from_primitive(&input, 3, values_per_frame, &mut ctx)?.into_array();
    let range = if sliced { 3..197 } else { 0..200 };
    let source = encoded.slice(range.clone())?;
    assert!(source.is::<Zstd>());
    let input = input.slice(range)?;

    let mut actual = builder_with_capacity_in(source.dtype(), INDICES.len(), ctx.allocator());
    let mut retained = None;
    let mut probe = probe_for(&source, &mut retained, repeated);
    for index in INDICES {
        actual.append_scalar(&probe.execute_scalar(index as usize, &mut ctx)?)?;
    }
    assert_arrays_eq!(
        actual.finish(),
        input.take(PrimitiveArray::from_iter(INDICES).into_array())?,
        &mut ctx
    );
    assert!(probe.execute_scalar(source.len(), &mut ctx).is_err());
    Ok(())
}

#[rstest]
fn var_bin_random_access(
    #[values(false, true)] repeated: bool,
    #[values(false, true)] nullable: bool,
    #[values(false, true)] sliced: bool,
    #[values(0, 16)] values_per_frame: usize,
    #[values(
        DType::Utf8(Nullability::Nullable),
        DType::Binary(Nullability::Nullable)
    )]
    dtype: DType,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let values: Vec<String> = (0..200).map(|i| format!("value number {i}")).collect();
    let input = VarBinViewArray::from_iter(
        values
            .iter()
            .enumerate()
            .map(|(i, value)| (!nullable || i % 7 != 0).then_some(value.as_bytes().to_vec())),
        dtype,
    );
    let encoded = Zstd::from_var_bin_view(&input, 3, values_per_frame, &mut ctx)?.into_array();
    let range = if sliced { 3..197 } else { 0..200 };
    let source = encoded.slice(range.clone())?;
    assert!(source.is::<Zstd>());
    let input = input.into_array().slice(range)?;

    let mut actual = builder_with_capacity_in(source.dtype(), INDICES.len(), ctx.allocator());
    let mut retained = None;
    let mut probe = probe_for(&source, &mut retained, repeated);
    for index in INDICES {
        actual.append_scalar(&probe.execute_scalar(index as usize, &mut ctx)?)?;
    }
    assert_arrays_eq!(
        actual.finish(),
        input.take(PrimitiveArray::from_iter(INDICES).into_array())?,
        &mut ctx
    );
    assert!(probe.execute_scalar(source.len(), &mut ctx).is_err());
    Ok(())
}

#[rstest]
#[case(PrimitiveArray::from_iter([1u16, 9, 32768, 65535]))]
#[case(PrimitiveArray::from_iter([i64::MIN, -1, 0, i64::MAX]))]
#[case(PrimitiveArray::from_iter([1.25f64, -2.5, 0.0, f64::INFINITY]))]
fn preserves_physical_type(#[case] input: PrimitiveArray) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let encoded = Zstd::from_primitive(&input, 3, 2, &mut ctx)?.into_array();
    let mut probe = encoded.repeated_probe();
    let mut actual = builder_with_capacity_in(input.dtype(), input.len(), ctx.allocator());
    for i in 0..input.len() {
        actual.append_scalar(&probe.execute_scalar(i, &mut ctx)?)?;
    }
    assert_arrays_eq!(actual.finish(), input, &mut ctx);
    Ok(())
}

#[test]
fn all_null_access_returns_null() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = PrimitiveArray::new(vec![0i32; 128], Validity::AllInvalid);
    let encoded = Zstd::from_primitive(&input, 3, 16, &mut ctx)?.into_array();
    let mut probe = encoded.repeated_probe();
    assert!(probe.execute_scalar(42, &mut ctx)?.is_null());
    assert!(probe.execute_scalar(0, &mut ctx)?.is_null());
    Ok(())
}

#[test]
fn probe_outlives_source_and_moves() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = PrimitiveArray::from_iter(0..200u32);
    let array = Zstd::from_primitive(&input, 3, 16, &mut ctx)?.into_array();
    let probe = RepeatedArrayProbe::new(array.clone());
    drop(array);
    let mut moved = (probe, ());
    for index in [1, 5, 127, 64, 63, 0, 1, 199] {
        assert_eq!(
            moved.0.execute_scalar(index, &mut ctx)?,
            u32::try_from(index)?.into()
        );
    }
    assert!(moved.0.execute_scalar(200, &mut ctx).is_err());
    Ok(())
}

/// Metadata written before frames recorded their value count leaves it at zero, and a repeated
/// probe has to recover the same frame boundaries the decompression path does.
#[rstest]
fn reads_legacy_frame_metadata(#[values(false, true)] var_bin: bool) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let (array, mut data) = if var_bin {
        let array = VarBinViewArray::from_iter_nullable_str([
            Some("foo"),
            Some("bar"),
            None,
            Some("Lorem ipsum dolor sit amet"),
            Some("baz"),
        ]);
        // Only a single frame is recoverable for variable-width values.
        let data = ZstdData::from_var_bin_view(&array, 3, 0, &mut ctx)?;
        (array.into_array(), data)
    } else {
        let array =
            PrimitiveArray::from_option_iter((0..200i32).map(|i| (i % 7 != 0).then_some(i)));
        let data = ZstdData::from_primitive(&array, 3, 16, &mut ctx)?;
        (array.into_array(), data)
    };
    for frame in &mut data.metadata.frames {
        frame.n_values = 0;
    }
    let encoded = Zstd::try_new(array.dtype().clone(), data, array.validity()?)?.into_array();

    let mut probe = encoded.repeated_probe();
    for index in 0..array.len() {
        assert_eq!(
            probe.execute_scalar(index, &mut ctx)?,
            array.execute_scalar(index, &mut ctx)?
        );
    }
    Ok(())
}

/// Frame metadata comes straight off disk, so a repeated probe has to refuse an inconsistent
/// value count rather than reading another frame's bytes back as a value.
#[rstest]
#[case::frame_holds_fewer_values_than_claimed(1000, 0)]
#[case::frame_value_counts_overflow(u64::MAX, 1)]
#[case::missing_value_count_across_frames(0, 0)]
fn rejects_corrupt_frame_metadata(#[case] n_values: u64, #[case] frame: usize) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = VarBinViewArray::from_iter_nullable_str([
        Some("foo"),
        Some("bar"),
        None,
        Some("Lorem ipsum dolor sit amet"),
        Some("baz"),
        Some("quux"),
    ]);
    let mut data = ZstdData::from_var_bin_view(&array, 3, 3, &mut ctx)?;
    data.metadata.frames[frame].n_values = n_values;
    let encoded = Zstd::try_new(array.dtype().clone(), data, array.validity()?)?.into_array();

    assert!(
        encoded
            .repeated_probe()
            .execute_scalar(0, &mut ctx)
            .is_err()
    );
    Ok(())
}

/// Every read of a `ZstdBuffers` array decompresses the whole wrapped array, so a repeated probe
/// keeps it instead of paying for it per row.
#[rstest]
#[case(PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None, Some(5)]).into_array())]
#[case(VarBinViewArray::from_iter_nullable_str([Some("hello"), None, Some("world")]).into_array())]
fn zstd_buffers_repeated_probe(#[case] input: ArrayRef) -> VortexResult<()> {
    let session = array_session();
    let mut ctx = session.create_execution_ctx();
    let encoded = ZstdBuffers::compress(&input, 3, &session)?.into_array();

    let mut probe = encoded.repeated_probe();
    let expected: Vec<Scalar> = (0..input.len())
        .map(|index| input.execute_scalar(index, &mut ctx))
        .collect::<VortexResult<_>>()?;
    // Read twice, so the second pass runs entirely off the retained inner array.
    for _ in 0..2 {
        for (index, expected) in expected.iter().enumerate() {
            assert_eq!(&probe.execute_scalar(index, &mut ctx)?, expected);
        }
    }
    assert!(probe.execute_scalar(input.len(), &mut ctx).is_err());
    Ok(())
}
