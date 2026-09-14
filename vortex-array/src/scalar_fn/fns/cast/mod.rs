// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod kernel;

use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
pub use kernel::*;
use prost::Message;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
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

/// How a cast from one [`DType`] to another is carried out.
///
/// [`Cast::plan`] decides this from the two dtypes alone, so type-checking a cast (via
/// [`ScalarFnVTable::return_dtype`]) and executing it share one definition of which casts exist.
/// Kernels only implement the plan they are handed; they no longer decide whether a cast is legal.
#[derive(Debug, Clone)]
pub(crate) enum CastPlan {
    /// Source and target dtypes are identical.
    Identity,
    /// Source and target dtypes differ only in nullability.
    Nullability,
    /// A built-in cast between non-extension dtypes, or from an extension dtype to its storage
    /// dtype. Executed by the canonical [`CastKernel`]s and the scalar typed-view casts.
    Builtin,
    /// An expression over [`root()`](crate::expr::root) supplied by an extension dtype hook. It
    /// evaluates to the target dtype and replaces the cast wherever it appears.
    Rewrite(Expression),
}

impl Cast {
    /// Creates a lazy cast of `input` to `target_dtype`.
    ///
    /// # Panics
    ///
    /// Panics if the cast is unsupported; see [`Cast::try_new`].
    #[expect(clippy::new_ret_no_self, reason = "constructs the lazy result array")]
    pub fn new(input: ArrayRef, target_dtype: DType) -> ScalarFnArray {
        Self::try_new(input, target_dtype).vortex_expect("unsupported cast")
    }

    /// Creates a lazy cast of `input` to `target_dtype`.
    ///
    /// # Errors
    ///
    /// Returns an error if no cast exists between the input's dtype and `target_dtype`. Casts
    /// that exist can still fail during execution when values do not fit the target, for example
    /// a null cast to a non-nullable dtype.
    pub fn try_new(input: ArrayRef, target_dtype: DType) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(Cast.bind(target_dtype), vec![input])
    }

    /// Decides how to cast `source` to `target`, or returns an error if no such cast exists.
    ///
    /// This is the single source of truth for which casts are supported. It consults only the
    /// dtypes, so accepted casts can be planned without executing them. Value-dependent failures
    /// (overflow, nulls cast to non-nullable) are not detected here; they surface at execution.
    ///
    /// Extension dtypes take part through [`ExtVTable::cast_to`] and [`ExtVTable::cast_from`]:
    /// the source hook is consulted first, then the default cast to the extension's storage
    /// dtype, then the target hook.
    ///
    /// [`ExtVTable::cast_to`]: crate::dtype::extension::ExtVTable::cast_to
    /// [`ExtVTable::cast_from`]: crate::dtype::extension::ExtVTable::cast_from
    pub(crate) fn plan(source: &DType, target: &DType) -> VortexResult<CastPlan> {
        if source == target {
            return Ok(CastPlan::Identity);
        }

        // Nullability may change at any nesting depth; `eq_ignore_nullability` is recursive.
        if source.eq_ignore_nullability(target) {
            return Ok(CastPlan::Nullability);
        }

        // Null casts to any nullable dtype as null.
        if matches!(source, DType::Null) {
            vortex_ensure!(
                target.is_nullable(),
                "Cannot cast {source} to {target}: target dtype is non-nullable"
            );
            return Ok(CastPlan::Builtin);
        }

        if let Some(source_ext) = source.as_extension_opt() {
            if let Some(rewrite) = source_ext.cast_to(target)? {
                return Ok(CastPlan::Rewrite(rewrite));
            }
            if source_ext.storage_dtype().eq_ignore_nullability(target) {
                return Ok(CastPlan::Builtin);
            }
            if let Some(target_ext) = target.as_extension_opt()
                && let Some(rewrite) = target_ext.cast_from(source)?
            {
                return Ok(CastPlan::Rewrite(rewrite));
            }
            vortex_bail!(
                "Cannot cast {source} to {target}: extension type {} only casts to its storage \
                 dtype {}",
                source_ext.id(),
                source_ext.storage_dtype()
            );
        }

        if let Some(target_ext) = target.as_extension_opt() {
            if let Some(rewrite) = target_ext.cast_from(source)? {
                return Ok(CastPlan::Rewrite(rewrite));
            }
            vortex_bail!(
                "Cannot cast {source} to {target}: extension type {} does not accept casts from \
                 {source}",
                target_ext.id()
            );
        }

        Self::plan_builtin(source, target)?;
        Ok(CastPlan::Builtin)
    }

    /// The type-level rules for casts between non-extension dtypes.
    ///
    /// Every pair accepted here has both a canonical [`CastKernel`] and a scalar typed-view cast.
    /// Union and variant dtypes only support nullability casts, which never reach here.
    fn plan_builtin(source: &DType, target: &DType) -> VortexResult<()> {
        match (source, target) {
            (DType::Bool(_), DType::Bool(_) | DType::Primitive(..)) => Ok(()),
            (DType::Primitive(..), DType::Primitive(..) | DType::Decimal(..)) => Ok(()),
            (DType::Decimal(..), DType::Decimal(..) | DType::Primitive(..)) => Ok(()),
            (DType::Utf8(_), DType::Utf8(_)) | (DType::Binary(_), DType::Binary(_)) => Ok(()),
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
                {
                    vortex_ensure!(
                        source_size == target_size,
                        "Cannot cast {source} to {target}: fixed-size lists must have the same size"
                    );
                }
                Self::plan(source_elem, target_elem).map(drop)
            }
            (DType::Map(source_map, _), DType::Map(target_map, _)) => {
                Self::plan_map(source, target, source_map, target_map)
            }
            (DType::Struct(source_fields, _), DType::Struct(target_fields, _)) => {
                Self::plan_struct(source_fields, target_fields)
            }
            _ => vortex_bail!("Cannot cast {source} to {target}"),
        }
    }

    fn plan_map(
        source: &DType,
        target: &DType,
        source_map: &MapDType,
        target_map: &MapDType,
    ) -> VortexResult<()> {
        vortex_ensure!(
            !(target_map.keys_sorted() && !source_map.keys_sorted()),
            "Cannot cast {source} to {target}: source does not assert sorted map keys"
        );
        Self::plan(&source_map.key_dtype(), &target_map.key_dtype())?;
        Self::plan(&source_map.value_dtype(), &target_map.value_dtype())?;
        Ok(())
    }

    /// Struct casts match fields by position when the names line up exactly, and by name
    /// otherwise, in which case the target may add nullable fields (filled with nulls).
    fn plan_struct(source_fields: &StructFields, target_fields: &StructFields) -> VortexResult<()> {
        if struct_fields_match_order(source_fields, target_fields) {
            for (source_field, target_field) in
                source_fields.fields().zip_eq(target_fields.fields())
            {
                Self::plan(&source_field, &target_field)?;
            }
            return Ok(());
        }

        for (target_name, target_field) in
            target_fields.names().iter().zip_eq(target_fields.fields())
        {
            match source_fields.find(target_name) {
                None => vortex_ensure!(
                    target_field.is_nullable(),
                    "CAST for struct only supports added nullable fields, but {target_name} is \
                     non-nullable"
                ),
                Some(source_idx) => {
                    let source_field = source_fields
                        .field_by_index(source_idx)
                        .vortex_expect("field index returned by find");
                    Self::plan(&source_field, &target_field)?;
                }
            }
        }
        Ok(())
    }
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

/// Instantiates a cast rewrite over the given input node.
///
/// `rewrite` is an expression over [`root()`](crate::expr::root); every root is replaced with
/// `input`, producing a tree of the same kind as `input` (an expression or an array).
fn instantiate_rewrite<T: ReduceNode>(rewrite: &Expression, input: &T) -> VortexResult<T> {
    match rewrite {
        Expression::Root => Ok(input.clone()),
        Expression::Scalar {
            scalar_fn,
            children,
        } => {
            let children: Vec<T> = children
                .iter()
                .map(|child| instantiate_rewrite(child, input))
                .try_collect()?;
            input.new_node(scalar_fn.clone(), &children)
        }
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

    fn return_dtype(&self, dtype: &DType, arg_dtypes: &[DType]) -> VortexResult<DType> {
        // Type-checking a cast is deciding how it will run. Rejecting unsupported casts here
        // means a cast expression that binds can also be executed.
        Self::plan(&arg_dtypes[0], dtype)?;
        Ok(dtype.clone())
    }

    fn execute(
        &self,
        target_dtype: &DType,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let input = args.get(0)?;

        match Self::plan(input.dtype(), target_dtype)? {
            CastPlan::Identity => return Ok(input),
            // The rewrite replaces the cast; the executor keeps evaluating the result.
            CastPlan::Rewrite(rewrite) => return input.apply(&rewrite),
            CastPlan::Nullability | CastPlan::Builtin => {}
        }

        let Some(columnar) = input.as_opt::<AnyColumnar>() else {
            return input.execute::<ArrayRef>(ctx)?.cast(target_dtype.clone());
        };

        match columnar {
            ColumnarView::Canonical(canonical) => {
                match cast_canonical(canonical, target_dtype, ctx)? {
                    Some(result) => Ok(result),
                    None => vortex_bail!(
                        "No CastKernel to cast canonical array {} from {} to {}",
                        canonical.to_array_ref().encoding_id(),
                        canonical.to_array_ref().dtype(),
                        target_dtype,
                    ),
                }
            }
            ColumnarView::Constant(constant) => match cast_constant(constant, target_dtype)? {
                Some(result) => Ok(result),
                None => vortex_bail!(
                    "No CastReduce to cast constant array from {} to {}",
                    constant.dtype(),
                    target_dtype,
                ),
            },
        }
    }

    fn reduce<T: ReduceNode>(&self, target_dtype: &DType, node: &T) -> VortexResult<Option<T>> {
        let child = node.child(0);
        match Self::plan(&child.node_dtype()?, target_dtype)? {
            // Collapse the node if the child is already the target type.
            CastPlan::Identity => Ok(Some(child)),
            // Splice the extension rewrite into the tree so it optimizes like any expression.
            CastPlan::Rewrite(rewrite) => instantiate_rewrite(&rewrite, &child).map(Some),
            CastPlan::Nullability | CastPlan::Builtin => Ok(None),
        }
    }

    fn simplify_untyped(
        &self,
        target_dtype: &DType,
        expr: &Expression,
    ) -> VortexResult<Option<Expression>> {
        let Some(scalar) = expr.child(0).as_opt::<Literal>() else {
            return Ok(None);
        };
        // A failing cast (e.g. null to a non-nullable dtype) is left in place so the error
        // surfaces at execution time rather than during optimization.
        Ok(scalar.cast(target_dtype).ok().map(lit))
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
        // `Cast::plan` rejects union and variant casts before execution.
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

    use rstest::rstest;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect as _;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::Cast;
    use crate::IntoArray;
    use crate::arrays::StructArray;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::BoundExpression;
    use crate::expr::Expression;
    use crate::expr::cast;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::root;
    use crate::expr::test_harness;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::fns::ext_wrap::ExtWrap;
    use crate::scalar_fn::fns::literal::Literal;

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

    /// Unsupported casts are rejected when the expression is typed, not when it executes.
    #[rstest]
    #[case(
        DType::Utf8(Nullability::NonNullable),
        DType::Primitive(PType::I32, Nullability::NonNullable)
    )]
    #[case(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Utf8(Nullability::NonNullable)
    )]
    #[case(
        DType::Utf8(Nullability::NonNullable),
        DType::Binary(Nullability::NonNullable)
    )]
    #[case(
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Bool(Nullability::NonNullable)
    )]
    #[case(DType::Null, DType::Primitive(PType::I32, Nullability::NonNullable))]
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
        DType::Primitive(PType::I64, Nullability::NonNullable),
        DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased())
    )]
    fn return_dtype_rejects_unsupported_casts(#[case] source: DType, #[case] target: DType) {
        let result = Cast.return_dtype(&target, std::slice::from_ref(&source));
        assert!(
            result.is_err(),
            "{source} -> {target} should be rejected, got {result:?}"
        );

        let bound = Cast.try_new_bound_expr(target, [BoundExpression::new_root(source)]);
        assert!(bound.is_err());
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
    #[case(
        DType::Extension(Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased()),
        DType::Extension(Timestamp::new(TimeUnit::Seconds, Nullability::NonNullable).erased())
    )]
    fn return_dtype_accepts_supported_casts(
        #[case] source: DType,
        #[case] target: DType,
    ) -> VortexResult<()> {
        assert_eq!(
            Cast.return_dtype(&target, std::slice::from_ref(&source))?,
            target
        );
        Ok(())
    }

    #[test]
    fn struct_cast_plan_matches_by_name_and_adds_nullable_fields() {
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
        assert!(
            Cast.return_dtype(&ok, std::slice::from_ref(&source))
                .is_ok()
        );

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
        assert!(
            Cast.return_dtype(&added_non_nullable, std::slice::from_ref(&source))
                .is_err()
        );
    }

    #[test]
    fn extension_rewrite_is_spliced_into_expressions() -> VortexResult<()> {
        let source = DType::Extension(
            Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased(),
        );
        let target = DType::Extension(
            Timestamp::new(TimeUnit::Microseconds, Nullability::NonNullable).erased(),
        );

        let optimized = cast(root(), target.clone()).optimize(&source)?;
        assert!(
            optimized.as_opt::<Cast>().is_none(),
            "cast should be rewritten: {optimized}"
        );
        assert!(optimized.as_opt::<ExtWrap>().is_some(), "{optimized}");
        assert_eq!(optimized.return_dtype(&source)?, target);
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
