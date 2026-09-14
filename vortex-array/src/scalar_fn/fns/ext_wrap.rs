// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use prost::Message;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::dtype::DType;
use crate::dtype::extension::ExtDTypeRef;
use crate::expr::Expression;
use crate::expr::display::ExprDisplay;
use crate::proto::expr as pb;
use crate::scalar::Scalar;
use crate::scalar_fn::Arity;
use crate::scalar_fn::ChildName;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::ext_storage::ExtStorage;

/// Wraps storage values into an extension dtype.
///
/// This is the inverse of [`ExtStorage`]: the child must have exactly the storage dtype of the
/// extension dtype, and the result has the extension dtype. It is the building block for
/// extension cast rewrites (see [`ExtVTable::cast_to`](crate::dtype::extension::ExtVTable::cast_to)),
/// which compute new storage values and then wrap them back into an extension dtype.
#[derive(Clone)]
pub struct ExtWrap;

impl ScalarFnVTable for ExtWrap {
    type Options = ExtDTypeRef;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.ext.wrap");
        *ID
    }

    fn serialize(&self, ext_dtype: &ExtDTypeRef) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            pb::ExtWrapOpts {
                ext_dtype: Some((&DType::Extension(ext_dtype.clone())).try_into()?),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        let proto = pb::ExtWrapOpts::decode(metadata)?.ext_dtype;
        let dtype = DType::from_proto(
            proto
                .as_ref()
                .ok_or_else(|| vortex_err!("Missing extension dtype in ext_wrap expression"))?,
            session,
        )?;
        let DType::Extension(ext_dtype) = dtype else {
            vortex_bail!("ext_wrap() options must be an extension dtype, got {dtype}");
        };
        Ok(ext_dtype)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(1)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("storage"),
            _ => unreachable!("Invalid child index {child_idx} for ext_wrap()"),
        }
    }

    fn fmt_sql(
        &self,
        ext_dtype: &ExtDTypeRef,
        expr: &dyn ExprDisplay,
        f: &mut Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "ext_wrap(")?;
        Display::fmt(expr.display_child(0), f)?;
        write!(f, " as {ext_dtype})")
    }

    fn return_dtype(&self, ext_dtype: &ExtDTypeRef, arg_dtypes: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            &arg_dtypes[0] == ext_dtype.storage_dtype(),
            "ext_wrap() into {} requires storage dtype {}, got {}",
            ext_dtype,
            ext_dtype.storage_dtype(),
            arg_dtypes[0]
        );
        Ok(DType::Extension(ext_dtype.clone()))
    }

    fn execute(
        &self,
        ext_dtype: &ExtDTypeRef,
        args: &dyn ExecutionArgs,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let storage = args.get(0)?;

        if let Some(scalar) = storage.as_constant() {
            // Validates the storage value against the extension type.
            let ext_scalar =
                Scalar::try_new(DType::Extension(ext_dtype.clone()), scalar.into_value())?;
            return Ok(ConstantArray::new(ext_scalar, args.row_count()).into_array());
        }

        Ok(ExtensionArray::try_new(ext_dtype.clone(), storage)?.into_array())
    }

    fn reduce<T: ReduceNode>(&self, ext_dtype: &ExtDTypeRef, node: &T) -> VortexResult<Option<T>> {
        // ext_wrap(ext_storage(x)) == x when x already has the target extension dtype.
        let storage = node.child(0);
        if !storage.scalar_fn().is_some_and(|f| f.is::<ExtStorage>()) {
            return Ok(None);
        }
        let inner = storage.child(0);
        if inner.node_dtype()?.as_extension_opt() != Some(ext_dtype) {
            return Ok(None);
        }
        Ok(Some(inner))
    }

    fn validity(
        &self,
        _options: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(expression.child(0).validity()?))
    }

    fn is_strict(&self, _options: &Self::Options) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::extension::ExtensionArrayExt;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtDTypeRef;
    use crate::expr::ext_storage;
    use crate::expr::ext_wrap;
    use crate::expr::root;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar::Scalar;

    fn ext_dtype(nullability: Nullability) -> ExtDTypeRef {
        Timestamp::new(TimeUnit::Nanoseconds, nullability).erased()
    }

    #[test]
    fn wraps_storage_array() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let storage = buffer![2i64, 4, 6].into_array();
        let ext_dtype = ext_dtype(Nullability::NonNullable);

        let result = storage
            .clone()
            .apply(&ext_wrap(root(), ext_dtype.clone()))?
            .execute::<ExtensionArray>(&mut ctx)?;

        assert_eq!(result.ext_dtype(), &ext_dtype);
        assert_arrays_eq!(result.storage_array().clone(), storage, &mut ctx);
        Ok(())
    }

    #[test]
    fn wraps_constant_into_extension_scalar() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let ext_dtype = ext_dtype(Nullability::NonNullable);
        let storage = ConstantArray::new(Scalar::from(7i64), 3).into_array();

        let result = storage.apply(&ext_wrap(root(), ext_dtype.clone()))?;
        let scalar = result.execute_scalar(1, &mut ctx)?;

        assert_eq!(scalar, Scalar::extension_ref(ext_dtype, Scalar::from(7i64)));
        Ok(())
    }

    #[test]
    fn rejects_mismatched_storage_dtype() {
        let storage = buffer![2i32, 4, 6].into_array();
        let result = storage.apply(&ext_wrap(root(), ext_dtype(Nullability::NonNullable)));
        assert!(result.is_err(), "expected error, got {result:?}");

        let nullable_storage = buffer![2i64]
            .into_array()
            .apply(&ext_wrap(root(), ext_dtype(Nullability::Nullable)));
        assert!(
            nullable_storage.is_err(),
            "ext_wrap must not change nullability, got {nullable_storage:?}"
        );
    }

    #[test]
    fn wrap_of_storage_reduces_to_input() -> VortexResult<()> {
        let ext_dtype = ext_dtype(Nullability::NonNullable);
        let array =
            ExtensionArray::new(ext_dtype.clone(), buffer![1i64, 2].into_array()).into_array();

        let optimized = array
            .clone()
            .apply(&ext_wrap(ext_storage(root()), ext_dtype))?
            .optimize()?;

        assert_eq!(optimized.encoding_id(), array.encoding_id());
        assert_eq!(optimized.dtype(), array.dtype());
        assert_eq!(
            optimized
                .as_::<crate::arrays::Extension>()
                .storage_array()
                .dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        Ok(())
    }
}
