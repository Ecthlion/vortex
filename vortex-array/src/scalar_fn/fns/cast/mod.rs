// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Casting between dtypes.
//!
//! [`Cast`] is the cast expression. Which casts exist is decided by the [`CastRule`]s of the
//! session that executes it, held in its [`CastSession`]. The [standard rules](standard) resolve
//! to [`KernelCast`], the cast the encodings implement.

mod kernel;
mod session;
pub mod standard;

use std::fmt::Display;
use std::fmt::Formatter;

pub use kernel::*;
use prost::Message;
pub use session::*;
pub(crate) use standard::struct_fields_match_order;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::ScalarFnArray;
use crate::dtype::DType;
use crate::expr::display::ExprDisplay;
use crate::expr::expression::Expression;
use crate::expr::lit;
use crate::proto::expr as pb;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::SimplifyCtx;
use crate::scalar_fn::fns::literal::Literal;

/// A cast expression that converts values to a target data type.
#[derive(Clone)]
pub struct Cast;

impl Cast {
    /// Creates a lazy cast of `input` to `target_dtype`.
    ///
    /// Whether the cast is supported is decided when it executes, by the [`CastRules`] of the
    /// executing session.
    #[expect(clippy::new_ret_no_self, reason = "constructs the lazy result array")]
    pub fn new(input: ArrayRef, target_dtype: DType) -> ScalarFnArray {
        ScalarFnArray::try_new(Cast.bind(target_dtype), vec![input])
            .vortex_expect("Cast has one child and an infallible return dtype")
    }
}

impl ScalarFnVTable for Cast {
    type Options = DType;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.cast");
        *ID
    }

    fn serialize(&self, dtype: &DType) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::CastOpts {
                target: Some(dtype.try_into()?),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        let proto = pb::CastOpts::decode(_metadata)?.target;
        DType::from_proto(
            proto
                .as_ref()
                .ok_or_else(|| vortex_err!("Missing target dtype in Cast expression"))?,
            session,
        )
    }

    fn arity(&self, _options: &DType) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _instance: &DType, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for Cast expression", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        dtype: &DType,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "cast(")?;
        Display::fmt(expr.display_child(0), f)?;
        write!(f, " as {}", dtype)?;
        write!(f, ")")
    }

    /// Every pair of dtypes is accepted. Which casts exist is decided by the [`CastRules`] of
    /// the executing session, and no session is available when an expression is typed, so an
    /// unsupported cast fails when it executes.
    fn return_dtype(&self, dtype: &DType, _arg_dtypes: &[DType]) -> VortexResult<DType> {
        Ok(dtype.clone())
    }

    fn execute(
        &self,
        target_dtype: &DType,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;
        if input.dtype() == target_dtype {
            return Ok(input);
        }

        // The session guard is dropped at the end of the statement, before the cast runs.
        let cast_fn = ctx.session().casts().bind(input.dtype(), target_dtype)?;
        let Some(cast_fn) = cast_fn else {
            vortex_bail!(
                "Cannot cast {} to {target_dtype}: no cast rule accepts it",
                input.dtype()
            );
        };
        cast_fn(input, ctx)
    }

    fn reduce<T: ReduceNode>(&self, target_dtype: &DType, node: &T) -> VortexResult<Option<T>> {
        // Collapse the node if the child is already the target type.
        let child = node.child(0);
        if &child.node_dtype()? == target_dtype {
            return Ok(Some(child));
        }
        Ok(None)
    }

    /// Folds a cast of a literal with the session's cast rules.
    ///
    /// A failing cast (e.g. null to a non-nullable dtype) is left in place so the error surfaces
    /// at execution time rather than during optimization.
    fn simplify(
        &self,
        target_dtype: &DType,
        expr: &Expression,
        ctx: &dyn SimplifyCtx,
    ) -> VortexResult<Option<Expression>> {
        let Some(scalar) = expr.child(0).as_opt::<Literal>() else {
            return Ok(None);
        };
        Ok(scalar.cast_ctx(target_dtype, ctx.session()).ok().map(lit))
    }

    fn validity(&self, dtype: &DType, expression: &Expression) -> VortexResult<Option<Expression>> {
        Ok(Some(if dtype.is_nullable() {
            expression.child(0).validity()?
        } else {
            lit(true)
        }))
    }

    fn is_strict(&self, _instance: &DType) -> bool {
        // Cast options can pin a non-nullable output dtype instead of propagating nullability.
        false
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ConstantArray;
    use crate::arrays::NullArray;
    use crate::arrays::ScalarFn;
    use crate::arrays::StructArray;
    use crate::arrays::VarBinViewArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::cast;
    use crate::expr::get_item;
    use crate::expr::root;
    use crate::expr::test_harness;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::cast::standard::StructCast;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

    #[test]
    fn dtype() {
        let dtype = test_harness::struct_dtype();
        let target = dtype.with_nullability(Nullability::Nullable);
        assert_eq!(
            cast(root(), target.clone()).return_dtype(&dtype).unwrap(),
            target
        );
    }

    #[test]
    fn replace_children() {
        let expr = cast(root(), DType::Bool(Nullability::Nullable));
        expr.with_children(vec![root()])
            .vortex_expect("operation should succeed in test");
    }

    #[test]
    fn evaluate() {
        let test_array = StructArray::from_fields(&[
            ("a", buffer![0i32, 1, 2].into_array()),
            ("b", buffer![4i64, 5, 6].into_array()),
        ])
        .unwrap()
        .into_array();

        let expr: Expression = cast(
            get_item("a", root()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        let result = test_array.apply(&expr).unwrap();

        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
    }

    #[test]
    fn simplify_folds_cast_of_literal() -> VortexResult<()> {
        let expr = cast(
            lit(3i32),
            DType::Primitive(PType::F64, Nullability::NonNullable),
        );
        let optimized = expr.optimize(&test_harness::struct_dtype(), &SESSION)?;

        let scalar = optimized
            .as_opt::<Literal>()
            .ok_or_else(|| vortex_err!("expected a bare literal, got {optimized}"))?;
        assert_eq!(scalar, &Scalar::primitive(3.0f64, Nullability::NonNullable));
        Ok(())
    }

    #[test]
    fn simplify_folds_cast_of_decimal_literal() -> VortexResult<()> {
        let decimal = Scalar::decimal(
            DecimalValue::I128(319),
            DecimalDType::new(3, 2),
            Nullability::NonNullable,
        );
        let expr = cast(
            lit(decimal),
            DType::Primitive(PType::F64, Nullability::NonNullable),
        );
        let optimized = expr.optimize(&test_harness::struct_dtype(), &SESSION)?;

        let scalar = optimized
            .as_opt::<Literal>()
            .ok_or_else(|| vortex_err!("expected a bare literal, got {optimized}"))?;
        assert_eq!(
            scalar,
            &Scalar::primitive(3.19f64, Nullability::NonNullable)
        );
        Ok(())
    }

    #[test]
    fn simplify_leaves_failing_cast_unchanged() -> VortexResult<()> {
        let target = DType::Primitive(PType::F64, Nullability::NonNullable);
        let expr = cast(
            lit(Scalar::null(DType::Primitive(
                PType::I32,
                Nullability::Nullable,
            ))),
            target.clone(),
        );
        let optimized = expr.optimize(&test_harness::struct_dtype(), &SESSION)?;

        assert!(optimized.as_opt::<Literal>().is_none());
        assert_eq!(optimized.as_opt::<Cast>(), Some(&target));
        Ok(())
    }

    /// The rules live in the session, so typing a cast cannot reject it: unsupported casts bind
    /// and fail when they execute.
    #[rstest]
    #[case::utf8_to_i32(
        VarBinViewArray::from_iter_str(["1"]).into_array(),
        DType::Primitive(PType::I32, Nullability::NonNullable)
    )]
    #[case::i32_to_bool(
        buffer![1i32].into_array(),
        DType::Bool(Nullability::NonNullable)
    )]
    #[case::null_to_non_nullable(
        NullArray::new(1).into_array(),
        DType::Primitive(PType::I32, Nullability::NonNullable)
    )]
    #[case::storage_to_extension(
        buffer![1i64].into_array(),
        DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased())
    )]
    fn unsupported_casts_bind_and_fail_at_execution(
        #[case] array: ArrayRef,
        #[case] target: DType,
    ) {
        let source = array.dtype().clone();
        assert_eq!(
            Cast.return_dtype(&target, std::slice::from_ref(&source))
                .unwrap(),
            target
        );

        // Optimizer rules that run while the cast is built may already report the failure.
        let mut ctx = SESSION.create_execution_ctx();
        let result = array
            .cast(target.clone(), ctx.session())
            .and_then(|cast| cast.execute::<ArrayRef>(&mut ctx));
        assert!(
            result.is_err(),
            "{source} -> {target} should fail, got {result:?}"
        );
    }

    #[rstest]
    #[case(
        DType::Bool(Nullability::NonNullable),
        DType::Primitive(PType::U8, Nullability::Nullable)
    )]
    #[case(
        DType::Primitive(PType::I32, Nullability::Nullable),
        DType::Decimal(DecimalDType::new(10, 2), Nullability::Nullable)
    )]
    #[case(
        DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable),
        DType::Primitive(PType::I32, Nullability::NonNullable)
    )]
    #[case(DType::Null, DType::Utf8(Nullability::Nullable))]
    #[case(
        DType::Utf8(Nullability::NonNullable),
        DType::Utf8(Nullability::Nullable)
    )]
    #[case(
        DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::NonNullable
        ),
        DType::List(
            Arc::new(DType::Primitive(PType::I64, Nullability::Nullable)),
            Nullability::Nullable
        )
    )]
    #[case(
        DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()),
        DType::Primitive(PType::I64, Nullability::Nullable)
    )]
    #[case(
        DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()),
        DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullability::NonNullable).erased())
    )]
    fn default_rules_accept(#[case] source: DType, #[case] target: DType) -> VortexResult<()> {
        assert!(
            CastSession::default().bind(&source, &target)?.is_some(),
            "{source} -> {target}"
        );
        Ok(())
    }

    #[rstest]
    #[case(
        DType::Utf8(Nullability::NonNullable),
        DType::Binary(Nullability::NonNullable)
    )]
    #[case(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::NonNullable)
    )]
    #[case(
        DType::Primitive(PType::I64, Nullability::NonNullable),
        DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased())
    )]
    #[case(
        DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()),
        DType::Primitive(PType::F64, Nullability::NonNullable)
    )]
    #[case(
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            2,
            Nullability::NonNullable
        ),
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            3,
            Nullability::NonNullable
        )
    )]
    fn default_rules_decline(#[case] source: DType, #[case] target: DType) -> VortexResult<()> {
        assert!(
            CastSession::default().bind(&source, &target)?.is_none(),
            "{source} -> {target}"
        );
        Ok(())
    }

    #[test]
    fn struct_cast_matches_by_name_and_adds_nullable_fields() -> VortexResult<()> {
        let source = DType::Struct(
            StructFields::new(
                ["a"].into(),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );
        let ok = DType::Struct(
            StructFields::new(
                ["b", "a"].into(),
                vec![
                    DType::Utf8(Nullability::Nullable),
                    DType::Primitive(PType::I64, Nullability::NonNullable),
                ],
            ),
            Nullability::NonNullable,
        );
        assert!(StructCast.bind(&source, &ok)?.is_some());

        let added_non_nullable = DType::Struct(
            StructFields::new(
                ["a", "b"].into(),
                vec![
                    DType::Primitive(PType::I32, Nullability::NonNullable),
                    DType::Utf8(Nullability::NonNullable),
                ],
            ),
            Nullability::NonNullable,
        );
        assert!(StructCast.bind(&source, &added_non_nullable)?.is_none());
        Ok(())
    }

    /// A rule that casts any `i32` array to a constant of the target dtype, to make its effect
    /// visible.
    #[derive(Debug)]
    struct ConstantRule {
        target: DType,
        value: Scalar,
    }

    impl CastRule for ConstantRule {
        fn bind(&self, source: &DType, target: &DType) -> VortexResult<Option<CastFn>> {
            if !matches!(source, DType::Primitive(PType::I32, _)) || target != &self.target {
                return Ok(None);
            }
            let value = self.value.clone();
            Ok(Some(Arc::new(
                move |array: ArrayRef, _ctx: &mut ExecutionCtx| {
                    Ok(ConstantArray::new(value.clone(), array.len()).into_array())
                },
            )))
        }
    }

    fn i64_dtype() -> DType {
        DType::Primitive(PType::I64, Nullability::NonNullable)
    }

    fn execute_cast(session: &VortexSession, target: DType) -> VortexResult<ArrayRef> {
        let mut ctx = session.create_execution_ctx();
        buffer![1i32, 2, 3]
            .into_array()
            .cast(target, session)?
            .execute::<ArrayRef>(&mut ctx)
    }

    /// Without rules there are no casts at all: the standard casts are rules like any other.
    #[test]
    fn no_rules_no_casts() {
        let session = VortexSession::empty().with_some(CastSession::empty());
        let result = execute_cast(&session, i64_dtype());
        assert!(result.is_err(), "expected error, got {result:?}");
    }

    #[test]
    fn session_rule_replaces_a_standard_cast() -> VortexResult<()> {
        let session = array_session();
        session.casts().register(ConstantRule {
            target: i64_dtype(),
            value: Scalar::from(42i64),
        });
        let mut ctx = session.create_execution_ctx();

        let result = execute_cast(&session, i64_dtype())?;
        assert_arrays_eq!(result, buffer![42i64, 42, 42].into_array(), &mut ctx);

        // Other sessions keep the standard cast.
        let result = execute_cast(&SESSION, i64_dtype())?;
        assert_arrays_eq!(result, buffer![1i64, 2, 3].into_array(), &mut ctx);
        Ok(())
    }

    #[test]
    fn session_rule_adds_a_new_cast() -> VortexResult<()> {
        let target = DType::Utf8(Nullability::NonNullable);
        assert!(execute_cast(&SESSION, target.clone()).is_err());

        let session = array_session();
        session.casts().register(ConstantRule {
            target: target.clone(),
            value: Scalar::from("x"),
        });
        let mut ctx = session.create_execution_ctx();
        let result = execute_cast(&session, target)?;
        assert_eq!(result.execute_scalar(1, &mut ctx)?, Scalar::from("x"));
        Ok(())
    }

    #[test]
    fn newest_rule_wins() -> VortexResult<()> {
        let session = array_session();
        session.casts().register(ConstantRule {
            target: i64_dtype(),
            value: Scalar::from(1i64),
        });
        session.casts().register(ConstantRule {
            target: i64_dtype(),
            value: Scalar::from(2i64),
        });
        let mut ctx = session.create_execution_ctx();

        let result = execute_cast(&session, i64_dtype())?;
        assert_eq!(result.execute_scalar(0, &mut ctx)?, Scalar::from(2i64));
        Ok(())
    }

    #[test]
    fn rules_are_shared_between_clones() {
        let rules = CastRules::empty();
        let clone = rules.clone();
        rules.register(ConstantRule {
            target: i64_dtype(),
            value: Scalar::from(1i64),
        });
        assert_eq!(clone.len(), 1);
        assert!(!CastSession::default().is_empty());
        assert!(CastSession::empty().is_empty());
    }

    /// The optimizer has no session, so it folds constants and literals with the rules of the
    /// default session: a rule adding a cast is reached at execution, while a rule replacing a
    /// standard cast does not apply to constants folded beforehand.
    #[test]
    fn optimizer_folds_with_the_default_rules() -> VortexResult<()> {
        let utf8 = DType::Utf8(Nullability::NonNullable);
        let session = array_session();
        session.casts().register(ConstantRule {
            target: utf8.clone(),
            value: Scalar::from("x"),
        });
        let mut ctx = session.create_execution_ctx();

        // Standard cast: folded while the cast is built.
        let folded = ConstantArray::new(Scalar::from(7i32), 3)
            .into_array()
            .cast(i64_dtype(), ctx.session())?;
        assert!(
            folded.as_opt::<ScalarFn>().is_none(),
            "not folded: {folded}"
        );

        // Cast known only to the session's rule: left in place and executed through the rule.
        let deferred = ConstantArray::new(Scalar::from(7i32), 3)
            .into_array()
            .cast(utf8.clone(), ctx.session())?;
        assert!(
            deferred.as_opt::<ScalarFn>().is_some(),
            "folded: {deferred}"
        );
        assert_eq!(deferred.execute_scalar(0, &mut ctx)?, Scalar::from("x"));

        let expr = cast(lit(7i32), utf8).optimize(&test_harness::struct_dtype(), ctx.session())?;
        assert!(expr.as_opt::<Literal>().is_none(), "folded: {expr}");
        Ok(())
    }

    #[test]
    fn test_display() {
        let expr = cast(
            get_item("value", root()),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        );
        assert_eq!(expr.to_string(), "cast($.value as i64)");

        let expr2 = cast(root(), DType::Bool(Nullability::Nullable));
        assert_eq!(expr2.to_string(), "cast($ as bool?)");
    }
}
