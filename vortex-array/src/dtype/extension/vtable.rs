// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;

use vortex_error::VortexResult;

use crate::dtype::DType;
use crate::dtype::extension::ExtDType;
use crate::dtype::extension::ExtId;
use crate::expr::Expression;
use crate::scalar::ScalarValue;

/// The public API for defining new extension types.
///
/// This is the non-object-safe trait that plugin authors implement to define a new extension type.
/// It specifies the type's identity, metadata, serialization, storage compatibility, and scalar
/// compatibility.
///
/// An extension dtype is not a new physical array layout. It is a logical wrapper around a storage
/// [`DType`](crate::dtype::DType) plus metadata. Implementations should keep
/// [`validate_dtype`](Self::validate_dtype)
/// strict enough that every valid storage scalar can be interpreted by
/// [`unpack_native`](Self::unpack_native).
pub trait ExtVTable: 'static + Sized + Send + Sync + Clone + Debug + Eq + Hash {
    /// Associated type containing the deserialized metadata for this extension type.
    type Metadata: 'static + Send + Sync + Clone + Debug + Display + Eq + Hash;

    /// A native Rust value that represents a scalar of the extension type.
    ///
    /// The value only represents non-null values. We denote nullable values as `Option<Value>`.
    type NativeValue<'a>: Display;

    /// Returns the ID for this extension type.
    fn id(&self) -> ExtId;

    // Methods related to the extension `DType`.

    /// Serialize the metadata into a byte vector.
    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>>;

    /// Deserialize the metadata from a byte slice.
    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata>;

    /// Validate that the given storage type is compatible with this extension type.
    ///
    /// This is called when constructing an [`ExtDType`] and should check both storage dtype and
    /// extension metadata.
    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()>;

    // Methods related to casting.

    /// Returns an expression that casts a value of this extension dtype to `target`.
    ///
    /// The expression is written over [`root()`](crate::expr::root), which stands for the input
    /// value, and must evaluate to exactly `target`. Because it is a plain expression, the same
    /// definition serves arrays, constants, and scalars, and it takes part in expression
    /// optimization like any other cast. Building it from the concrete [`ExtDType`] lets the
    /// implementation consult extension metadata, such as a timestamp's unit and timezone.
    ///
    /// Returning `Ok(None)` declines the cast. The default cast rules then apply: an extension
    /// dtype can always be cast to its storage dtype, and a `target` extension dtype gets a
    /// chance to accept the cast through [`cast_from`](Self::cast_from). Anything else is
    /// rejected when the cast is type-checked, so a cast that is accepted here can be
    /// planned without executing it.
    ///
    /// Identity and nullability-only casts never reach this hook.
    fn cast_to(ext_dtype: &ExtDType<Self>, target: &DType) -> VortexResult<Option<Expression>> {
        _ = (ext_dtype, target);
        Ok(None)
    }

    /// Returns an expression that casts a value of `source` to this extension dtype.
    ///
    /// The expression follows the same contract as [`cast_to`](Self::cast_to): it is written
    /// over [`root()`](crate::expr::root) and must evaluate to exactly this extension dtype.
    /// [`ext_wrap`](crate::expr::ext_wrap) wraps a storage-typed expression into an extension
    /// dtype.
    ///
    /// There is no default cast into an extension dtype, not even from its storage dtype,
    /// because the storage dtype alone does not prove that values are meaningful for the
    /// extension type. Extension types opt in through this hook. When `source` is itself an
    /// extension dtype, its own [`cast_to`](Self::cast_to) is consulted first.
    fn cast_from(ext_dtype: &ExtDType<Self>, source: &DType) -> VortexResult<Option<Expression>> {
        _ = (ext_dtype, source);
        Ok(None)
    }

    // Methods related to the extension scalar values.

    /// Validate the given storage value is compatible with the extension type.
    ///
    /// By default, this calls [`unpack_native()`](ExtVTable::unpack_native) and discards the
    /// result.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage [`ScalarValue`] is not compatible with the extension type.
    fn validate_scalar_value(
        ext_dtype: &ExtDType<Self>,
        storage_value: &ScalarValue,
    ) -> VortexResult<()> {
        Self::unpack_native(ext_dtype, storage_value).map(|_| ())
    }

    /// Validate and unpack a native value from the storage [`ScalarValue`].
    ///
    /// Note that [`ExtVTable::validate_dtype()`] is always called first to validate the storage
    /// [`crate::dtype::DType`], and the [`Scalar`](crate::scalar::Scalar) implementation will
    /// verify that the storage value is compatible with the storage dtype on construction.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage [`ScalarValue`] is not compatible with the extension type.
    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Self::NativeValue<'a>>;
}
