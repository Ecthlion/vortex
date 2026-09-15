// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar casting between [`DType`]s.

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::cast::CastSessionExt;

impl Scalar {
    /// Casts this scalar to another data type with the cast rules of the default session.
    ///
    /// Prefer [`Scalar::cast_ctx`] wherever a session is available.
    #[expect(
        clippy::disallowed_methods,
        reason = "sessionless cast is a convenience over the default session"
    )]
    pub fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        self.cast_ctx(target_dtype, crate::legacy_session())
    }

    /// Casts this scalar to another data type.
    ///
    /// The cast is resolved from the [`CastRules`](crate::scalar_fn::fns::cast::CastRules) of
    /// `session` and executed over a one-element constant array, so the same rules apply to
    /// scalars as to arrays.
    ///
    /// # Errors
    ///
    /// Returns an error if no rule casts between the two dtypes, or if the value does not fit the
    /// target, such as a null cast to a non-nullable type.
    pub fn cast_ctx(&self, target_dtype: &DType, session: &VortexSession) -> VortexResult<Scalar> {
        if self.dtype() == target_dtype {
            return Ok(self.clone());
        }

        // The session guard is dropped at the end of the statement, before the cast runs.
        let cast_fn = session.casts().bind(self.dtype(), target_dtype)?;
        let Some(cast_fn) = cast_fn else {
            vortex_bail!(
                "Cannot cast {} to {target_dtype}: no cast rule accepts it",
                self.dtype()
            );
        };
        let mut ctx = session.create_execution_ctx();
        cast_fn(ConstantArray::new(self.clone(), 1).into_array(), &mut ctx)?
            .execute_scalar(0, &mut ctx)
    }

    /// The scalar side of [`KernelCast`](crate::scalar_fn::fns::cast::KernelCast): casts the
    /// value with the typed-view casts.
    ///
    /// The pair of dtypes must already have been accepted by a cast rule that resolves to
    /// `KernelCast`; no rule is consulted here.
    pub(crate) fn cast_kernel(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        if self.dtype() == target_dtype {
            return Ok(self.clone());
        }

        // A null casts to null.
        if self.value().is_none() || matches!(self.dtype(), DType::Null) {
            vortex_ensure!(
                target_dtype.is_nullable(),
                "Cannot cast null to {target_dtype}: target type is non-nullable"
            );
            return Ok(Scalar::null(target_dtype.clone()));
        }

        if self.dtype().eq_ignore_nullability(target_dtype) {
            return Scalar::try_new(target_dtype.clone(), self.value().cloned());
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
                vortex_bail!("Cannot cast {} to {target_dtype}", self.dtype())
            }
            DType::Extension(..) => self.as_extension().cast(target_dtype),
        }
    }

    /// Cast the scalar into a nullable version of its current type.
    pub fn into_nullable(self) -> Scalar {
        let (dtype, value) = self.into_parts();
        Self::try_new(dtype.as_nullable(), value)
            .vortex_expect("Casting to nullable should always succeed")
    }
}
