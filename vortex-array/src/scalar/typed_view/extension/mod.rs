// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ExtScalar`] typed view implementation.

use std::fmt;
use std::hash::Hash;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

use crate::dtype::DType;
use crate::dtype::extension::ExtDTypeRef;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A scalar value representing an extension type.
///
/// Extension types allow wrapping a storage type with custom semantics.
#[derive(Debug, Clone, Copy)]
pub struct ExtScalar<'a> {
    /// A reference to the `DType` of the extension type. This **must** be the [`DType::Extension`]
    /// variant.
    dtype: &'a DType,

    /// The extension data type reference.
    ///
    /// We store this here as a convenience so that we do not need to unwrap the dtype every time.
    ext_dtype: &'a ExtDTypeRef,

    /// The underlying scalar value, or [`None`] if null.
    value: Option<&'a ScalarValue>,
}

impl fmt::Display for ExtScalar<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(value) = self.value else {
            return write!(f, "null");
        };

        self.ext_dtype.fmt_storage_value(f, value)
    }
}

impl<'a> ExtScalar<'a> {
    /// Creates a new extension scalar from a data type and scalar value.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the dtype is an extension type and that the scalar value has
    /// been verified to be valid for the extension type.
    pub(crate) fn new_unchecked(dtype: &'a DType, value: Option<&'a ScalarValue>) -> Self {
        let DType::Extension(ext_dtype) = dtype else {
            vortex_panic!("Expected extension scalar, found {}", dtype)
        };

        Self {
            dtype,
            ext_dtype,
            value,
        }
    }

    /// Return the [`DType`] of the extension scalar.
    pub fn dtype(&self) -> &DType {
        self.dtype
    }

    /// Returns the extension data type.
    pub fn ext_dtype(&self) -> &'a ExtDTypeRef {
        self.ext_dtype
    }

    /// Returns the storage scalar of the extension scalar.
    pub fn to_storage_scalar(&self) -> Scalar {
        Scalar::try_new(self.ext_dtype.storage_dtype().clone(), self.value.cloned())
            .vortex_expect("ExtScalar is invalid")
    }

    /// Returns a reference to the underlying value
    pub fn value(&self) -> Option<&ScalarValue> {
        self.value
    }

    /// Casts this scalar to its storage dtype, or to the same extension dtype with a different
    /// nullability.
    ///
    /// Other extension casts are rewrites supplied by the extension type and are evaluated by
    /// [`Scalar::cast`]; see [`ExtVTable::cast_to`](crate::dtype::extension::ExtVTable::cast_to).
    pub(crate) fn cast(&self, target_dtype: &DType) -> VortexResult<Scalar> {
        if self.value.is_none() && !target_dtype.is_nullable() {
            vortex_bail!(
                "cannot cast null extension dtype with id {} and storage type {} to non-nullable {}",
                self.ext_dtype.id(),
                self.ext_dtype.storage_dtype(),
                target_dtype
            );
        }

        let is_storage = self
            .ext_dtype
            .storage_dtype()
            .eq_ignore_nullability(target_dtype);
        let is_same_ext = target_dtype
            .as_extension_opt()
            .is_some_and(|ext_dtype| self.ext_dtype.eq_ignore_nullability(ext_dtype));
        if is_storage || is_same_ext {
            return Scalar::try_new(target_dtype.clone(), self.value.cloned());
        }

        vortex_bail!(
            "cannot cast extension dtype with id {} and storage type {} to {}",
            self.ext_dtype.id(),
            self.ext_dtype.storage_dtype(),
            target_dtype
        );
    }
}

// TODO(connor): In the future we may want to allow implementors to customize this behavior.

impl PartialEq for ExtScalar<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.ext_dtype.eq_ignore_nullability(other.ext_dtype)
            && self.to_storage_scalar() == other.to_storage_scalar()
    }
}

impl Eq for ExtScalar<'_> {}

// Ord is not implemented since it's undefined for different Extension DTypes
impl PartialOrd for ExtScalar<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if !self.ext_dtype.eq_ignore_nullability(other.ext_dtype) {
            return None;
        }
        self.to_storage_scalar()
            .partial_cmp(&other.to_storage_scalar())
    }
}

impl Hash for ExtScalar<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ext_dtype.hash_ignore_nullability(state);
        self.to_storage_scalar().hash(state);
    }
}

#[cfg(test)]
mod tests;
