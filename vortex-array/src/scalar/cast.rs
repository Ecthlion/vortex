// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar casting between [`DType`]s.

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::cast::CastSessionExt;
use crate::scalar_fn::fns::cast::is_builtin_cast;

impl Scalar {
    /// Cast this scalar to another data type.
    ///
    /// The built-in casts run directly on the scalar. Any other cast is looked up in the
    /// [`CastRules`](crate::scalar_fn::fns::cast::CastRules) of the default session and executed
    /// over a one-element constant array, so the default rules, such as timestamp unit
    /// conversion, apply to scalars as well as arrays. Rules registered into another session only
    /// apply to arrays executed in that session.
    ///
    /// # Errors
    ///
    /// Returns an error if the cast is not supported or if a null value is cast to a non-nullable
    /// type.
    pub fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        if let Some(scalar) = self.cast_builtin(target_dtype)? {
            return Ok(scalar);
        }
        self.cast_with_default_rules(target_dtype)
    }

    /// Performs a built-in cast, or returns `Ok(None)` when only a cast rule could cast between
    /// the two dtypes.
    ///
    /// The supported casts are exactly those of [`is_builtin_cast`]; the array-level cast uses the
    /// same matrix. Rules are not consulted, so this is safe to call from optimizer rules that have
    /// no session.
    pub(crate) fn cast_builtin(&self, target_dtype: &DType) -> VortexResult<Option<Scalar>> {
        if self.dtype() == target_dtype {
            return Ok(Some(self.clone()));
        }

        // A null casts wherever its dtype casts, to null.
        if self.value().is_none() || matches!(self.dtype(), DType::Null) {
            vortex_ensure!(
                target_dtype.is_nullable(),
                "Cannot cast null to {target_dtype}: target type is non-nullable"
            );
            return Ok(is_builtin_cast(self.dtype(), target_dtype)
                .then(|| Scalar::null(target_dtype.clone())));
        }

        if !is_builtin_cast(self.dtype(), target_dtype) {
            return Ok(None);
        }

        if self.dtype().eq_ignore_nullability(target_dtype) {
            return Scalar::try_new(target_dtype.clone(), self.value().cloned()).map(Some);
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
                unreachable!("union and variant dtypes only change nullability")
            }
            DType::Extension(..) => self.as_extension().cast(target_dtype),
        }
        .map(Some)
    }

    /// Casts this scalar with the cast rules of the default session.
    #[expect(
        clippy::disallowed_methods,
        reason = "Scalar::cast takes no session; the default rules are the best available stand-in"
    )]
    fn cast_with_default_rules(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        let session = crate::legacy_session();
        // The session guard is dropped at the end of the statement, before the cast runs.
        let cast_fn = session.casts().bind(self.dtype(), target_dtype)?;
        let Some(cast_fn) = cast_fn else {
            vortex_bail!(
                "Cannot cast {} to {target_dtype}: no built-in cast and no default cast rule",
                self.dtype()
            );
        };
        let mut ctx = session.create_execution_ctx();
        cast_fn(ConstantArray::new(self.clone(), 1).into_array(), &mut ctx)?
            .execute_scalar(0, &mut ctx)
    }

    /// Cast the scalar into a nullable version of its current type.
    pub fn into_nullable(self) -> Scalar {
        let (dtype, value) = self.into_parts();
        Self::try_new(dtype.as_nullable(), value)
            .vortex_expect("Casting to nullable should always succeed")
    }
}
