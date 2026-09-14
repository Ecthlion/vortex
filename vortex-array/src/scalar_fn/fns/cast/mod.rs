// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;
mod session;

use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
pub use kernel::*;
use prost::Message;
pub use session::*;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyColumnar;
use crate::ArrayRef;
use crate::ArrayView;
use crate::CanonicalView;
use crate::ColumnarView;
use crate::ExecutionCtx;
use crate::arrays::Bool;
use crate::arrays::Constant;
use crate::arrays::Decimal;
use crate::arrays::Extension;
use crate::arrays::FixedSizeList;
use crate::arrays::ListView;
use crate::arrays::Map;
use crate::arrays::Null;
use crate::arrays::Primitive;
use crate::arrays::ScalarFnArray;
use crate::arrays::VarBinView;
use crate::arrays::struct_::compute::cast::struct_cast;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::MapDType;
use crate::dtype::StructFields;
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
use crate::scalar_fn::fns::literal::Literal;

/// A cast expression that converts values to a target data type.
#[derive(Clone)]
pub struct Cast;

impl Cast {
    /// Creates a lazy cast of `input` to `target_dtype`.
    ///
    /// Whether the cast is supported is decided when it executes, against the session's
    /// [`CastRules`] and the built-in casts.
    #[expect(clippy::new_ret_no_self, reason = "constructs the lazy result array")]
    pub fn new(input: ArrayRef, target_dtype: DType) -> ScalarFnArray {
        ScalarFnArray::try_new(Cast.bind(target_dtype), vec![input])
            .vortex_expect("Cast has one child and an infallible return dtype")
    }

    /// Binds the session [`CastRule`] for `source` to `target`, if one accepts the pair.
    ///
    /// The executor consults this before every built-in cast, both in [`Cast::execute`] and in
    /// the fused [`CastExecuteAdaptor`] kernels.
    pub(crate) fn session_rule(
        ctx: &ExecutionCtx,
        source: &DType,
        target: &DType,
    ) -> VortexResult<Option<CastFn>> {
        // The session guard is dropped at the end of the statement, before the cast runs.
        let cast_fn = ctx.session().casts().bind(source, target)?;
        Ok(cast_fn)
    }
}

/// Whether Vortex casts `source` to `target` without a session [`CastRule`].
///
/// This is the type-level matrix of the built-in casts: every pair accepted here is handled by
/// the canonical [`CastKernel`]s and by the scalar typed-view casts. It decides nothing about
/// values, so an accepted cast can still fail when a value does not fit the target.
///
/// A [`Null`] value casts to any nullable dtype. An extension dtype casts to its storage dtype
/// and nothing else; casts into an extension dtype always come from a rule, because the storage
/// dtype alone does not prove that values are meaningful for the extension type. Union and
/// variant dtypes only change nullability.
pub(crate) fn is_builtin_cast(source: &DType, target: &DType) -> bool {
    // Nullability may change at any nesting depth; `eq_ignore_nullability` is recursive.
    if source.eq_ignore_nullability(target) {
        return true;
    }
    match (source, target) {
        (DType::Null, _) => target.is_nullable(),
        (DType::Extension(source_ext), _) => {
            source_ext.storage_dtype().eq_ignore_nullability(target)
        }
        (_, DType::Extension(_)) => false,
        (DType::Bool(_), DType::Primitive(..)) => true,
        (DType::Primitive(..), DType::Primitive(..) | DType::Decimal(..)) => true,
        (DType::Decimal(..), DType::Decimal(..) | DType::Primitive(..)) => true,
        // A list casts to a fixed-size list when every list has the target size, which is
        // checked at execution.
        (
            DType::List(source_elem, _) | DType::FixedSizeList(source_elem, ..),
            DType::List(target_elem, _) | DType::FixedSizeList(target_elem, ..),
        ) => {
            if let (
                DType::FixedSizeList(_, source_size, _),
                DType::FixedSizeList(_, target_size, _),
            ) = (source, target)
                && source_size != target_size
            {
                return false;
            }
            is_builtin_cast(source_elem, target_elem)
        }
        (DType::Map(source_map, _), DType::Map(target_map, _)) => {
            is_builtin_map_cast(source_map, target_map)
        }
        (DType::Struct(source_fields, _), DType::Struct(target_fields, _)) => {
            is_builtin_struct_cast(source_fields, target_fields)
        }
        _ => false,
    }
}

/// A map cast may not assert sorted keys that the source does not assert.
fn is_builtin_map_cast(source: &MapDType, target: &MapDType) -> bool {
    (source.keys_sorted() || !target.keys_sorted())
        && is_builtin_cast(&source.key_dtype(), &target.key_dtype())
        && is_builtin_cast(&source.value_dtype(), &target.value_dtype())
}

/// Struct casts match fields by position when the names line up exactly, and by name otherwise,
/// in which case the target may add nullable fields (filled with nulls).
fn is_builtin_struct_cast(source: &StructFields, target: &StructFields) -> bool {
    if struct_fields_match_order(source, target) {
        return source
            .fields()
            .zip_eq(target.fields())
            .all(|(source_field, target_field)| is_builtin_cast(&source_field, &target_field));
    }
    target
        .names()
        .iter()
        .zip_eq(target.fields())
        .all(|(name, target_field)| match source.find(name) {
            None => target_field.is_nullable(),
            Some(idx) => {
                let source_field = source
                    .field_by_index(idx)
                    .vortex_expect("field index returned by find");
                is_builtin_cast(&source_field, &target_field)
            }
        })
}

/// Whether two struct dtypes have the same field names in the same order.
pub(crate) fn struct_fields_match_order(source: &StructFields, target: &StructFields) -> bool {
    source.nfields() == target.nfields()
        && source
            .names()
            .iter()
            .zip(target.names().iter())
            .all(|(a, b)| a == b)
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

    /// Every pair of dtypes is accepted. Which casts exist is decided by the session's
    /// [`CastRules`] together with the built-in casts, and no session is available when an
    /// expression is typed, so unsupported casts fail when they execute.
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

        // Session rules take precedence over the built-in casts.
        if let Some(cast_fn) = Self::session_rule(ctx, input.dtype(), target_dtype)? {
            return cast_fn(input, ctx);
        }

        let Some(columnar) = input.as_opt::<AnyColumnar>() else {
            return input.execute::<ArrayRef>(ctx)?.cast(target_dtype.clone());
        };

        match columnar {
            ColumnarView::Canonical(canonical) => {
                match cast_canonical(canonical, target_dtype, ctx)? {
                    Some(result) => Ok(result),
                    None => vortex_bail!(
                        "Cannot cast {} to {}: no cast rule and no built-in cast for {} arrays",
                        canonical.to_array_ref().dtype(),
                        target_dtype,
                        canonical.to_array_ref().encoding_id(),
                    ),
                }
            }
            ColumnarView::Constant(constant) => match cast_constant(constant, target_dtype)? {
                Some(result) => Ok(result),
                None => vortex_bail!(
                    "Cannot cast {} to {}: no cast rule and no built-in cast for constants",
                    constant.dtype(),
                    target_dtype,
                ),
            },
        }
    }

    fn reduce<T: ReduceNode>(&self, target_dtype: &DType, node: &T) -> VortexResult<Option<T>> {
        // Collapse the node if the child is already the target type.
        let child = node.child(0);
        if &child.node_dtype()? == target_dtype {
            return Ok(Some(child));
        }
        Ok(None)
    }

    fn simplify_untyped(
        &self,
        target_dtype: &DType,
        expr: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let Some(scalar) = expr.child(0).as_opt::<Literal>() else {
            return Ok(None);
        };
        // Only built-in casts fold: session rules are consulted at execution, where the session
        // is known. A failing cast (e.g. null to a non-nullable dtype) is left in place so the
        // error surfaces at execution time rather than during optimization.
        Ok(scalar.cast_builtin(target_dtype).ok().flatten().map(lit))
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

/// Cast a canonical array to the target dtype by dispatching to the appropriate
/// [`CastKernel`] for each canonical encoding.
///
/// Canonical encodings that can manipulate validity directly all implement [`CastKernel`] —
/// the kernel is the execution-time complement of their [`CastReduce`] rule and can compute
/// statistics (e.g. min of the validity array) when the reduce rule had to give up.
/// Encodings that delegate to scalars or storage (e.g. [`Null`], [`Constant`], [`Extension`])
/// only implement [`CastReduce`] because they never need execution-level information.
fn cast_canonical(
    canonical: CanonicalView<'_>,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    match canonical {
        CanonicalView::Null(a) => <Null as CastReduce>::cast(a, dtype),
        CanonicalView::Bool(a) => <Bool as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Primitive(a) => <Primitive as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Decimal(a) => <Decimal as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::VarBinView(a) => <VarBinView as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::List(a) => <ListView as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Map(a) => <Map as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::FixedSizeList(a) => <FixedSizeList as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Struct(a) => struct_cast(a, dtype, ctx),
        CanonicalView::Union(_) => vortex_bail!("Union arrays don't support casting (yet)"),
        CanonicalView::Extension(a) => <Extension as CastReduce>::cast(a, dtype),
        CanonicalView::Variant(_) => vortex_bail!("Variant arrays don't support casting"),
    }
}

/// Cast a constant array by dispatching to its [`CastReduce`] implementation.
fn cast_constant(array: ArrayView<Constant>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
    <Constant as CastReduce>::cast(array, dtype)
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
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::cast;
    use crate::expr::get_item;
    use crate::expr::root;
    use crate::expr::test_harness;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;

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
    fn simplify_folds_cast_of_builtin_literal() -> VortexResult<()> {
        let expr = cast(
            lit(3i32),
            DType::Primitive(PType::F64, Nullability::NonNullable),
        );
        let optimized = expr.optimize(&test_harness::struct_dtype())?;

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
        let optimized = expr.optimize(&test_harness::struct_dtype())?;

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
        let optimized = expr.optimize(&test_harness::struct_dtype())?;

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
        assert!(!is_builtin_cast(&source, &target));

        // Optimizer rules that run while the cast is built may already report the failure.
        let mut ctx = SESSION.create_execution_ctx();
        let result = array
            .cast(target.clone())
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
    fn builtin_cast_matrix(#[case] source: DType, #[case] target: DType) {
        assert!(is_builtin_cast(&source, &target), "{source} -> {target}");
    }

    #[rstest]
    #[case(
        DType::Utf8(Nullability::NonNullable),
        DType::Binary(Nullability::NonNullable)
    )]
    #[case(
        DType::List(
            Arc::new(DType::Utf8(Nullability::NonNullable)),
            Nullability::NonNullable
        ),
        DType::List(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            Nullability::NonNullable
        )
    )]
    #[case(
        DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()),
        DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullability::NonNullable).erased())
    )]
    #[case(
        DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()),
        DType::Primitive(PType::F64, Nullability::NonNullable)
    )]
    fn not_a_builtin_cast(#[case] source: DType, #[case] target: DType) {
        assert!(!is_builtin_cast(&source, &target), "{source} -> {target}");
    }

    #[test]
    fn builtin_struct_cast_matches_by_name_and_adds_nullable_fields() {
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
        assert!(is_builtin_cast(&source, &ok));

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
        assert!(!is_builtin_cast(&source, &added_non_nullable));
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
            .cast(target)?
            .execute::<ArrayRef>(&mut ctx)
    }

    #[test]
    fn session_rule_overrides_a_builtin_cast() -> VortexResult<()> {
        let session = array_session();
        session.casts().register(ConstantRule {
            target: i64_dtype(),
            value: Scalar::from(42i64),
        });
        let mut ctx = session.create_execution_ctx();

        let result = execute_cast(&session, i64_dtype())?;
        assert_arrays_eq!(result, buffer![42i64, 42, 42].into_array(), &mut ctx);

        // Other sessions keep the built-in cast.
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

    /// The optimizer has no session, so it folds constants and literals with the built-in casts
    /// only: a rule adding a cast is reached at execution, while a rule replacing a built-in cast
    /// does not apply to constants folded beforehand.
    #[test]
    fn optimizer_folds_builtin_casts_only() -> VortexResult<()> {
        let utf8 = DType::Utf8(Nullability::NonNullable);
        let session = array_session();
        session.casts().register(ConstantRule {
            target: utf8.clone(),
            value: Scalar::from("x"),
        });
        let mut ctx = session.create_execution_ctx();

        // Built-in cast: folded while the cast is built.
        let folded = ConstantArray::new(Scalar::from(7i32), 3)
            .into_array()
            .cast(i64_dtype())?;
        assert!(
            folded.as_opt::<ScalarFn>().is_none(),
            "not folded: {folded}"
        );

        // Rule-only cast: left in place and executed through the rule.
        let deferred = ConstantArray::new(Scalar::from(7i32), 3)
            .into_array()
            .cast(utf8.clone())?;
        assert!(
            deferred.as_opt::<ScalarFn>().is_some(),
            "folded: {deferred}"
        );
        assert_eq!(deferred.execute_scalar(0, &mut ctx)?, Scalar::from("x"));

        let expr = cast(lit(7i32), utf8).optimize(&test_harness::struct_dtype())?;
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
