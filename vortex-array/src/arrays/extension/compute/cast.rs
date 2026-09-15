// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Extension {
    /// Handles the two extension casts that need no extension-specific knowledge: unwrapping to
    /// the storage dtype, and changing the nullability of the same extension dtype. Every other
    /// extension cast comes from a session [`CastRule`](crate::scalar_fn::fns::cast::CastRule),
    /// consulted when the cast executes.
    fn cast(
        array: ArrayView<'_, Extension>,
        dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Option<ArrayRef>> {
        let ext_dtype = array.ext_dtype();

        if ext_dtype.storage_dtype().eq_ignore_nullability(dtype) {
            return Ok(Some(array.storage_array().cast(dtype.clone(), session)?));
        }

        let Some(target_ext_dtype) = dtype.as_extension_opt() else {
            return Ok(None);
        };
        if !ext_dtype.eq_ignore_nullability(target_ext_dtype) {
            return Ok(None);
        }

        let new_storage = array
            .storage_array()
            .cast(target_ext_dtype.storage_dtype().clone(), session)?;
        Ok(Some(
            ExtensionArray::new(target_ext_dtype.clone(), new_storage).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use super::*;
    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::executor::VortexSessionExecute;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::cast::CastSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

    #[test]
    fn cast_same_ext_dtype() {
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let storage = Buffer::<i64>::empty().into_array();

        let arr = ExtensionArray::new(ext_dtype.clone(), storage);

        let output = arr
            .clone()
            .into_array()
            .cast(DType::Extension(ext_dtype.clone()), &SESSION)
            .unwrap();
        assert_eq!(arr.len(), output.len());
        assert_eq!(arr.dtype(), output.dtype());
        assert_eq!(output.dtype(), &DType::Extension(ext_dtype));
    }

    #[test]
    fn cast_same_ext_dtype_differet_nullability() {
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let storage = Buffer::<i64>::empty().into_array();

        let arr = ExtensionArray::new(ext_dtype.clone(), storage);
        assert!(!arr.dtype().is_nullable());

        let new_dtype = DType::Extension(ext_dtype).with_nullability(Nullability::Nullable);

        let output = arr
            .clone()
            .into_array()
            .cast(new_dtype.clone(), &SESSION)
            .unwrap();
        assert_eq!(arr.len(), output.len());
        assert!(arr.dtype().eq_ignore_nullability(output.dtype()));
        assert_eq!(output.dtype(), &new_dtype);
    }

    fn timestamp_dtype(unit: TimeUnit, tz: Option<&str>, nullability: Nullability) -> DType {
        DType::Extension(Timestamp::new_with_tz(unit, tz.map(Into::into), nullability).erased())
    }

    #[rstest]
    #[case(TimeUnit::Milliseconds, TimeUnit::Nanoseconds, buffer![1i64, -2, 3], buffer![1_000_000i64, -2_000_000, 3_000_000])]
    #[case(TimeUnit::Seconds, TimeUnit::Milliseconds, buffer![1i64, 2], buffer![1_000i64, 2_000])]
    #[case(TimeUnit::Nanoseconds, TimeUnit::Milliseconds, buffer![1_500_000i64, -1_500_000], buffer![1i64, -1])]
    #[case(TimeUnit::Microseconds, TimeUnit::Seconds, buffer![2_000_000i64, 999_999], buffer![2i64, 0])]
    fn cast_timestamp_unit(
        #[case] from: TimeUnit,
        #[case] to: TimeUnit,
        #[case] storage: Buffer<i64>,
        #[case] expected: Buffer<i64>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = timestamp_dtype(from, Some("UTC"), Nullability::NonNullable);
        let target = timestamp_dtype(to, Some("UTC"), Nullability::NonNullable);
        let arr = ExtensionArray::new(
            source.as_extension_opt().vortex_expect("extension").clone(),
            storage.into_array(),
        )
        .into_array();

        let result = arr
            .cast(target.clone(), ctx.session())?
            .execute::<ExtensionArray>(&mut ctx)?;
        assert_eq!(result.dtype(), &target);
        assert_arrays_eq!(
            result.storage_array().clone(),
            expected.into_array(),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn cast_timestamp_unit_changes_nullability() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = timestamp_dtype(TimeUnit::Milliseconds, None, Nullability::NonNullable);
        let target = timestamp_dtype(TimeUnit::Microseconds, None, Nullability::Nullable);
        let arr = ExtensionArray::new(
            source.as_extension_opt().vortex_expect("extension").clone(),
            buffer![5i64].into_array(),
        )
        .into_array();

        let result = arr
            .cast(target.clone(), ctx.session())?
            .execute::<ExtensionArray>(&mut ctx)?;
        assert_eq!(result.dtype(), &target);
        assert_eq!(
            result.into_array().execute_scalar(0, &mut ctx)?,
            Scalar::extension_ref(
                target.as_extension_opt().vortex_expect("extension").clone(),
                Scalar::primitive(5_000i64, Nullability::Nullable)
            )
        );
        Ok(())
    }

    #[test]
    fn cast_constant_timestamp_unit() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = timestamp_dtype(TimeUnit::Seconds, Some("UTC"), Nullability::NonNullable);
        let target = timestamp_dtype(
            TimeUnit::Milliseconds,
            Some("UTC"),
            Nullability::NonNullable,
        );
        let scalar = Scalar::extension_ref(
            source.as_extension_opt().vortex_expect("extension").clone(),
            Scalar::from(3i64),
        );

        // The same rule serves the scalar and the constant array.
        let expected = Scalar::extension_ref(
            target.as_extension_opt().vortex_expect("extension").clone(),
            Scalar::from(3_000i64),
        );
        assert_eq!(scalar.cast(&target)?, expected);

        let constant = ConstantArray::new(scalar, 4)
            .into_array()
            .cast(target.clone(), ctx.session())?;
        assert_eq!(constant.dtype(), &target);
        assert_eq!(constant.execute_scalar(2, &mut ctx)?, expected);
        Ok(())
    }

    fn timestamp_array(dtype: &DType) -> ArrayRef {
        ExtensionArray::new(
            dtype.as_extension_opt().vortex_expect("extension").clone(),
            buffer![1i64].into_array(),
        )
        .into_array()
    }

    /// Unsupported casts bind, because the rules live in the session, and fail at execution.
    #[rstest]
    #[case::timezone_change(
        timestamp_array(&timestamp_dtype(TimeUnit::Milliseconds, Some("UTC"), Nullability::NonNullable)),
        timestamp_dtype(TimeUnit::Nanoseconds, Some("Europe/London"), Nullability::NonNullable)
    )]
    #[case::storage_to_extension_requires_a_rule(
        buffer![1i64].into_array(),
        timestamp_dtype(TimeUnit::Milliseconds, None, Nullability::NonNullable)
    )]
    #[case::wider_than_storage(
        timestamp_array(&timestamp_dtype(TimeUnit::Milliseconds, None, Nullability::NonNullable)),
        DType::Primitive(PType::F64, Nullability::NonNullable)
    )]
    fn unsupported_casts_fail_at_execution(#[case] array: ArrayRef, #[case] target: DType) {
        let mut ctx = SESSION.create_execution_ctx();
        let cast = array
            .cast(target, ctx.session())
            .vortex_expect("casts always bind");
        let result = cast.execute::<ArrayRef>(&mut ctx);
        assert!(result.is_err(), "expected error, got {result:?}");
    }

    #[test]
    fn cast_timestamp_unit_needs_the_rule() {
        let session = VortexSession::empty().with_some(CastSession::empty());
        let mut ctx = session.create_execution_ctx();
        let source = timestamp_dtype(TimeUnit::Seconds, Some("UTC"), Nullability::NonNullable);
        let target = timestamp_dtype(
            TimeUnit::Milliseconds,
            Some("UTC"),
            Nullability::NonNullable,
        );

        let result = timestamp_array(&source)
            .cast(target, ctx.session())
            .vortex_expect("casts always bind")
            .execute::<ArrayRef>(&mut ctx);
        assert!(result.is_err(), "expected error, got {result:?}");
    }

    #[test]
    fn cast_timestamp_to_i64() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ext_dtype = Timestamp::new_with_tz(
            TimeUnit::Nanoseconds,
            Some("UTC".into()),
            Nullability::NonNullable,
        )
        .erased();
        let storage = buffer![1i64, 2, 3].into_array();
        let arr = ExtensionArray::new(ext_dtype, storage).into_array();

        let result = arr.cast(
            DType::Primitive(PType::I64, Nullability::NonNullable),
            ctx.session(),
        )?;
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        assert_arrays_eq!(result, buffer![1i64, 2, 3].into_array(), &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(create_timestamp_array(TimeUnit::Milliseconds, false))]
    #[case(create_timestamp_array(TimeUnit::Microseconds, true))]
    #[case(create_timestamp_array(TimeUnit::Nanoseconds, false))]
    #[case(create_timestamp_array(TimeUnit::Seconds, true))]
    fn test_cast_extension_conformance(#[case] array: ExtensionArray) {
        test_cast_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }

    fn create_timestamp_array(time_unit: TimeUnit, nullable: bool) -> ExtensionArray {
        let ext_dtype =
            Timestamp::new_with_tz(time_unit, Some("UTC".into()), nullable.into()).erased();

        let storage = if nullable {
            PrimitiveArray::from_option_iter([
                Some(1_000_000i64), // 1 second in microseconds
                None,
                Some(2_000_000),
                Some(3_000_000),
                None,
            ])
            .into_array()
        } else {
            buffer![1_000_000i64, 2_000_000, 3_000_000, 4_000_000, 5_000_000].into_array()
        };

        ExtensionArray::new(ext_dtype, storage)
    }
}
