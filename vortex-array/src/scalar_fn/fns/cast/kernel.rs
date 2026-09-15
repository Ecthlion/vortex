// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The cast the encodings implement.

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyColumnar;
use crate::ArrayRef;
use crate::CanonicalView;
use crate::ColumnarView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::Bool;
use crate::arrays::ConstantArray;
use crate::arrays::Decimal;
use crate::arrays::Extension;
use crate::arrays::FixedSizeList;
use crate::arrays::ListView;
use crate::arrays::Map;
use crate::arrays::Null;
use crate::arrays::Primitive;
use crate::arrays::ScalarFnArray;
use crate::arrays::VarBinView;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::struct_::compute::cast::struct_cast;
use crate::dtype::DType;
use crate::expr::display::ExprDisplay;
use crate::expr::expression::Expression;
use crate::expr::lit;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::cast::Cast;

/// The cast the encodings implement.
///
/// [`Cast`] resolves a [`CastRule`](super::CastRule) from the session when it executes, and the
/// [standard rules](super::standard) resolve to this scalar function. It executes with the
/// [`CastKernel`] registered for the input encoding, if there is one, and otherwise with the
/// canonical kernels once the input has been executed to a canonical encoding. A constant input
/// casts its scalar.
///
/// `KernelCast` is created at execution time only. It never appears in an expression and is not
/// serializable.
#[derive(Clone, Debug)]
pub struct KernelCast;

impl KernelCast {
    /// Creates a lazy kernel cast of `input` to `target_dtype`.
    #[expect(clippy::new_ret_no_self, reason = "constructs the lazy result array")]
    pub fn new(input: ArrayRef, target_dtype: DType) -> ScalarFnArray {
        ScalarFnArray::try_new(KernelCast.bind(target_dtype), vec![input])
            .vortex_expect("KernelCast has one child and an infallible return dtype")
    }
}

impl ScalarFnVTable for KernelCast {
    type Options = DType;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.cast.kernel");
        *ID
    }

    fn serialize(&self, _dtype: &DType) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(&self, _metadata: &[u8], _session: &VortexSession) -> VortexResult<DType> {
        vortex_bail!("KernelCast is created at execution time and is not serializable")
    }

    fn arity(&self, _dtype: &DType) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _dtype: &DType, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("input"),
            _ => unreachable!("Invalid child index {} for KernelCast", child_idx),
        }
    }

    fn fmt_sql(
        &self,
        dtype: &DType,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "kernel_cast(")?;
        Display::fmt(expr.display_child(0), f)?;
        write!(f, " as {}", dtype)?;
        write!(f, ")")
    }

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

        let Some(columnar) = input.as_opt::<AnyColumnar>() else {
            // Execute the input one step at a time, so the executor can offer the cast kernel of
            // whichever encoding it becomes.
            let input = input.execute::<ArrayRef>(ctx)?;
            return Ok(Self::new(input, target_dtype.clone()).into_array());
        };

        match columnar {
            ColumnarView::Canonical(canonical) => cast_canonical(canonical, target_dtype, ctx)?
                .ok_or_else(|| {
                    vortex_err!(
                        "Cannot cast {} to {}: no cast kernel for {} arrays",
                        canonical.to_array_ref().dtype(),
                        target_dtype,
                        canonical.to_array_ref().encoding_id(),
                    )
                }),
            ColumnarView::Constant(constant) => {
                let scalar = constant.scalar().cast_kernel(target_dtype)?;
                Ok(ConstantArray::new(scalar, constant.len()).into_array())
            }
        }
    }

    fn validity(&self, dtype: &DType, expression: &Expression) -> VortexResult<Option<Expression>> {
        Ok(Some(if dtype.is_nullable() {
            expression.child(0).validity()?
        } else {
            lit(true)
        }))
    }

    fn is_strict(&self, _dtype: &DType) -> bool {
        // Casting to a non-nullable dtype pins the output nullability instead of propagating it.
        false
    }
}

/// Cast a canonical array to the target dtype by dispatching to the appropriate
/// [`CastKernel`] for each canonical encoding.
///
/// Canonical encodings that can manipulate validity directly all implement [`CastKernel`] —
/// the kernel is the execution-time complement of their [`CastReduce`] rule and can compute
/// statistics (e.g. min of the validity array) when the reduce rule had to give up.
/// Encodings that delegate to scalars or storage (e.g. [`Null`], [`Extension`]) only implement
/// [`CastReduce`] because they never need execution-level information.
fn cast_canonical(
    canonical: CanonicalView<'_>,
    dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    match canonical {
        CanonicalView::Null(a) => <Null as CastReduce>::cast(a, dtype, ctx.session()),
        CanonicalView::Bool(a) => <Bool as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Primitive(a) => <Primitive as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Decimal(a) => <Decimal as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::VarBinView(a) => <VarBinView as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::List(a) => <ListView as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Map(a) => <Map as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::FixedSizeList(a) => <FixedSizeList as CastKernel>::cast(a, dtype, ctx),
        CanonicalView::Struct(a) => struct_cast(a, dtype, ctx),
        CanonicalView::Union(_) => vortex_bail!("Union arrays don't support casting (yet)"),
        CanonicalView::Extension(a) => <Extension as CastReduce>::cast(a, dtype, ctx.session()),
        CanonicalView::Variant(_) => vortex_bail!("Variant arrays don't support casting"),
    }
}

/// Reduce rule for cast: restructure the array without reading buffers.
///
/// Encodings implement this to push cast operations through their structure.
/// For example, RunEnd pushes cast down to its values array, ZigZag transforms
/// the target dtype to unsigned and pushes to its encoded array.
///
/// Reduce rules run on [`Cast`] before it is resolved to a [`CastRule`](super::CastRule), so they
/// restructure every cast between a pair of dtypes as if it were the standard one. They receive
/// the session being optimized for, so a rule that folds values, such as a constant's, applies
/// that session's cast rules.
///
/// Returns `Ok(None)` if the rule doesn't apply to this array/dtype combination.
pub trait CastReduce: VTable {
    fn cast(
        array: ArrayView<'_, Self>,
        dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Execute kernel for cast: perform the actual value conversion, potentially reading buffers.
///
/// Canonical array types implement this to do the real type conversion work.
/// For example, PrimitiveArray converts numeric values between types.
///
/// Kernels run on [`KernelCast`], after the session has resolved the cast to the standard one.
///
/// Returns `Ok(None)` if this kernel cannot handle the given dtype conversion.
pub trait CastKernel: VTable {
    fn cast(
        array: ArrayView<'_, Self>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adapts a [`CastReduce`] impl into an [`ArrayParentReduceRule`] for `ScalarFnArray(Cast, ...)`.
#[derive(Default, Debug)]
pub struct CastReduceAdaptor<V>(pub V);

impl<V> ArrayParentReduceRule<V> for CastReduceAdaptor<V>
where
    V: CastReduce,
{
    type Parent = ExactScalarFn<Cast>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: ScalarFnArrayView<'_, Cast>,
        _child_idx: usize,
        session: &VortexSession,
    ) -> VortexResult<Option<ArrayRef>> {
        let dtype = parent.options;
        if array.dtype() == dtype {
            return Ok(Some(array.array().clone()));
        }
        <V as CastReduce>::cast(array, dtype, session)
    }
}

/// Adapts a [`CastKernel`] impl into an [`ExecuteParentKernel`] for
/// `ScalarFnArray(KernelCast, ...)`.
#[derive(Default, Debug)]
pub struct CastExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for CastExecuteAdaptor<V>
where
    V: CastKernel,
{
    type Parent = ExactScalarFn<KernelCast>;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        parent: <Self::Parent as Matcher>::Match<'_>,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let dtype = parent.options;
        if array.dtype() == dtype {
            return Ok(Some(array.array().clone()));
        }
        <V as CastKernel>::cast(array, dtype, ctx)
    }
}
