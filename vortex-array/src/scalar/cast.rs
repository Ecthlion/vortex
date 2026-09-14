// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar casting between [`DType`]s.

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastPlan;

impl Scalar {
    /// Cast this scalar to another data type.
    ///
    /// The supported casts are exactly those of the array-level cast: both are decided by
    /// [`Cast::plan`] from the two dtypes.
    ///
    /// # Errors
    ///
    /// Returns an error if the cast is not supported or if a null value is cast to a non-nullable
    /// type.
    pub fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        match Cast::plan(self.dtype(), target_dtype)? {
            CastPlan::Identity => Ok(self.clone()),
            // `try_new` rejects a null value cast to a non-nullable dtype.
            CastPlan::Nullability => Scalar::try_new(target_dtype.clone(), self.value().cloned()),
            CastPlan::Builtin => self.cast_builtin(target_dtype),
            CastPlan::Rewrite(rewrite) => self.cast_rewrite(target_dtype, &rewrite),
        }
    }

    fn cast_builtin(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        // Null casts to null. The plan already verified the target is nullable for a `Null`
        // source; a null value of any other source dtype needs the same check.
        if self.value().is_none() || matches!(self.dtype(), DType::Null) {
            vortex_ensure!(
                target_dtype.is_nullable(),
                "Cannot cast null to {target_dtype}: target type is non-nullable"
            );
            return Scalar::try_new(target_dtype.clone(), None);
        }

        match &self.dtype() {
            DType::Null => unreachable!("Handled by the null case above"),
            DType::Bool(_) => self.as_bool().cast(target_dtype),
            DType::Primitive(..) => self.as_primitive().cast(target_dtype),
            DType::Decimal(..) => self.as_decimal().cast(target_dtype),
            DType::Utf8(_) => self.as_utf8().cast(target_dtype),
            DType::Binary(_) => self.as_binary().cast(target_dtype),
            DType::List(..) | DType::FixedSizeList(..) => self.as_list().cast(target_dtype),
            DType::Map(..) => self.as_map().cast(target_dtype),
            DType::Struct(..) => self.as_struct().cast(target_dtype),
            DType::Union(..) | DType::Variant(_) => {
                unreachable!("Cast::plan rejects union and variant casts")
            }
            DType::Extension(..) => self.as_extension().cast(target_dtype),
        }
    }

    /// Evaluates an extension cast rewrite over this scalar.
    ///
    /// The rewrite is an expression, so it is evaluated over a one-element constant array. This
    /// keeps a single definition of each extension cast for arrays and scalars alike.
    #[expect(
        clippy::disallowed_methods,
        reason = "Scalar::cast takes no execution context, and the rewrite only uses built-in \
                  scalar functions, so the session contents cannot change the result"
    )]
    fn cast_rewrite(&self, target_dtype: &DType, rewrite: &Expression) -> VortexResult<Scalar> {
        if self.value().is_none() {
            vortex_ensure!(
                target_dtype.is_nullable(),
                "Cannot cast null to {target_dtype}: target type is non-nullable"
            );
            return Ok(Scalar::null(target_dtype.clone()));
        }

        let mut ctx = crate::legacy_session().create_execution_ctx();
        ConstantArray::new(self.clone(), 1)
            .into_array()
            .apply(rewrite)?
            .execute_scalar(0, &mut ctx)
    }

    /// Cast the scalar into a nullable version of its current type.
    pub fn into_nullable(self) -> Scalar {
        let (dtype, value) = self.into_parts();
        Self::try_new(dtype.as_nullable(), value)
            .vortex_expect("Casting to nullable should always succeed")
    }
}
