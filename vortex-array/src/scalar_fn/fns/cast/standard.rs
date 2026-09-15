// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The standard cast rules.
//!
//! These are the casts Vortex performs between its own dtypes. [`CastSession::default`] registers
//! every rule here, so a default session casts exactly these pairs plus whatever later rules add.
//! Each rule accepts a family of dtype pairs and resolves them to [`KernelCast`], the cast the
//! encodings implement through their [`CastKernel`](super::CastKernel)s and the canonical kernels.
//!
//! The rules decide from dtypes alone and say nothing about values, so an accepted cast can still
//! fail when a value does not fit the target. A nested cast, such as the elements of a list or the
//! fields of a struct, is a separate [`Cast`](super::Cast) the kernel builds, so it is resolved
//! against the session on its own and may come from any rule.
//!
//! [`CastSession::default`]: super::CastSession::default

use std::sync::Arc;

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::dtype::DType;
use crate::dtype::StructFields;
use crate::scalar_fn::fns::cast::CastFn;
use crate::scalar_fn::fns::cast::CastRule;
use crate::scalar_fn::fns::cast::KernelCast;

/// Resolves a cast to [`KernelCast`].
fn kernel_cast(target: &DType) -> CastFn {
    let target = target.clone();
    Arc::new(move |array: ArrayRef, _ctx: &mut ExecutionCtx| {
        Ok(KernelCast::new(array, target.clone()).into_array())
    })
}

/// Casts that change nullability only, at any nesting depth.
///
/// Casting to non-nullable fails at execution if the array holds a null.
#[derive(Debug)]
pub struct NullabilityCast;

impl CastRule for NullabilityCast {
    fn bind(&self, source: &DType, target: &DType) -> VortexResult<Option<CastFn>> {
        Ok(source
            .eq_ignore_nullability(target)
            .then(|| kernel_cast(target)))
    }
}

/// Casts a [`DType::Null`] array to any nullable dtype, producing nulls.
#[derive(Debug)]
pub struct NullCast;

impl CastRule for NullCast {
    fn bind(&self, source: &DType, target: &DType) -> VortexResult<Option<CastFn>> {
        Ok((matches!(source, DType::Null) && target.is_nullable()).then(|| kernel_cast(target)))
    }
}

/// Casts booleans to any primitive type, as zero and one.
#[derive(Debug)]
pub struct BoolCast;

impl CastRule for BoolCast {
    fn bind(&self, source: &DType, target: &DType) -> VortexResult<Option<CastFn>> {
        Ok(
            matches!((source, target), (DType::Bool(_), DType::Primitive(..)))
                .then(|| kernel_cast(target)),
        )
    }
}

/// Casts between primitive types, and from a primitive type to a decimal.
#[derive(Debug)]
pub struct PrimitiveCast;

impl CastRule for PrimitiveCast {
    fn bind(&self, source: &DType, target: &DType) -> VortexResult<Option<CastFn>> {
        Ok(matches!(
            (source, target),
            (
                DType::Primitive(..),
                DType::Primitive(..) | DType::Decimal(..)
            )
        )
        .then(|| kernel_cast(target)))
    }
}

/// Casts between decimal types, and from a decimal to a primitive type.
#[derive(Debug)]
pub struct DecimalCast;

impl CastRule for DecimalCast {
    fn bind(&self, source: &DType, target: &DType) -> VortexResult<Option<CastFn>> {
        Ok(matches!(
            (source, target),
            (
                DType::Decimal(..),
                DType::Decimal(..) | DType::Primitive(..)
            )
        )
        .then(|| kernel_cast(target)))
    }
}

/// Casts between list types, variable-length and fixed-size.
///
/// Two fixed-size lists must have the same size. A variable-length list casts to a fixed-size
/// list when every list has the target size, which is checked at execution. The element cast is
/// resolved on its own.
#[derive(Debug)]
pub struct ListCast;

impl CastRule for ListCast {
    fn bind(&self, source: &DType, target: &DType) -> VortexResult<Option<CastFn>> {
        let accepted = match (source, target) {
            (DType::FixedSizeList(_, source_size, _), DType::FixedSizeList(_, target_size, _)) => {
                source_size == target_size
            }
            (
                DType::List(..) | DType::FixedSizeList(..),
                DType::List(..) | DType::FixedSizeList(..),
            ) => true,
            _ => false,
        };
        Ok(accepted.then(|| kernel_cast(target)))
    }
}

/// Casts between map types.
///
/// A cast may not assert sorted keys that the source does not assert. The key and value casts are
/// resolved on their own.
#[derive(Debug)]
pub struct MapCast;

impl CastRule for MapCast {
    fn bind(&self, source: &DType, target: &DType) -> VortexResult<Option<CastFn>> {
        let (DType::Map(source_map, _), DType::Map(target_map, _)) = (source, target) else {
            return Ok(None);
        };
        Ok((source_map.keys_sorted() || !target_map.keys_sorted()).then(|| kernel_cast(target)))
    }
}

/// Casts between struct types.
///
/// Fields match by position when the names line up exactly, and by name otherwise, in which case
/// the target may add nullable fields, filled with nulls. Each field cast is resolved on its own.
#[derive(Debug)]
pub struct StructCast;

impl CastRule for StructCast {
    fn bind(&self, source: &DType, target: &DType) -> VortexResult<Option<CastFn>> {
        let (DType::Struct(source_fields, _), DType::Struct(target_fields, _)) = (source, target)
        else {
            return Ok(None);
        };
        let accepted = struct_fields_match_order(source_fields, target_fields)
            || target_fields
                .names()
                .iter()
                .zip_eq(target_fields.fields())
                .all(|(name, field)| source_fields.find(name).is_some() || field.is_nullable());
        Ok(accepted.then(|| kernel_cast(target)))
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

/// Casts an extension array to its storage dtype.
///
/// Casts into an extension dtype always come from a rule for that extension type, because the
/// storage dtype alone does not prove that values are meaningful for it.
#[derive(Debug)]
pub struct StorageCast;

impl CastRule for StorageCast {
    fn bind(&self, source: &DType, target: &DType) -> VortexResult<Option<CastFn>> {
        let DType::Extension(source_ext) = source else {
            return Ok(None);
        };
        Ok(source_ext
            .storage_dtype()
            .eq_ignore_nullability(target)
            .then(|| kernel_cast(target)))
    }
}
