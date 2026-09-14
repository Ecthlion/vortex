// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::match_each_unsigned_integer_ptype;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::validity::Validity;

fn build_with_validity(
    array: ArrayView<'_, ListView>,
    new_elements: ArrayRef,
    validity: Validity,
) -> ArrayRef {
    // SAFETY: Since `cast` is length-preserving, all of the invariants remain the same.
    unsafe {
        ListViewArray::new_unchecked(
            new_elements,
            array.offsets().clone(),
            array.sizes().clone(),
            validity,
        )
        .with_zero_copy_to_list(array.is_zero_copy_to_list())
    }
    .into_array()
}

impl CastReduce for ListView {
    fn cast(array: ArrayView<'_, ListView>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        // Check if we're casting to a `List` type.
        let Some(target_element_type) = dtype.as_list_element_opt() else {
            return Ok(None);
        };
        let Some(validity) = array
            .validity()?
            .trivially_cast_nullability(dtype.nullability(), array.len())?
        else {
            return Ok(None);
        };

        // Cast the elements to the target element type.
        let new_elements = array.elements().cast((**target_element_type).clone())?;
        Ok(Some(build_with_validity(array, new_elements, validity)))
    }
}

impl CastKernel for ListView {
    fn cast(
        array: ArrayView<'_, ListView>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if let DType::FixedSizeList(target_element_type, list_size, nullability) = dtype {
            return cast_to_fixed_size_list(
                array,
                target_element_type,
                *list_size,
                *nullability,
                ctx,
            )
            .map(Some);
        }

        let Some(target_element_type) = dtype.as_list_element_opt() else {
            return Ok(None);
        };

        let validity = array
            .validity()?
            .cast_nullability(dtype.nullability(), array.len(), ctx)?;
        let new_elements = array.elements().cast((**target_element_type).clone())?;

        Ok(Some(build_with_validity(array, new_elements, validity)))
    }
}

/// Casts a list-view array to a fixed-size list by gathering each list's elements.
///
/// Every valid list must hold exactly `list_size` elements. A fixed-size list stores `list_size`
/// slots for null rows too, so null lists are padded with copies of the first element, which
/// the null mask hides.
fn cast_to_fixed_size_list(
    array: ArrayView<'_, ListView>,
    target_element_type: &DType,
    list_size: u32,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = array.len();
    let source_validity = array.validity()?;
    let valid = source_validity.execute_mask(len, ctx)?;
    let validity = source_validity.cast_nullability(nullability, len, ctx)?;

    if list_size > 0 && array.elements().is_empty() && len > 0 {
        vortex_ensure!(
            valid.true_count() == 0,
            "Cannot cast {} to FixedSizeList[{list_size}]: lists are empty",
            array.dtype()
        );
        vortex_bail!(
            "Cannot cast {} to FixedSizeList[{list_size}]: no elements to fill null lists with",
            array.dtype()
        );
    }

    // Offsets and sizes are validated non-negative, so their unsigned reinterpretation is exact.
    let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let offsets = offsets.reinterpret_cast(offsets.ptype().to_unsigned());
    let sizes = array.sizes().clone().execute::<PrimitiveArray>(ctx)?;
    let sizes = sizes.reinterpret_cast(sizes.ptype().to_unsigned());

    let indices = match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
        match_each_unsigned_integer_ptype!(sizes.ptype(), |S| {
            fixed_size_list_indices(
                offsets.as_slice::<O>(),
                sizes.as_slice::<S>(),
                &valid,
                list_size,
                array.dtype(),
            )?
        })
    });

    let elements = array
        .elements()
        .take(indices.into_array())?
        .cast(target_element_type.clone())?;
    Ok(FixedSizeListArray::try_new(elements, list_size, validity, len)?.into_array())
}

/// Builds the element indices that gather `list_size` elements per row from a list view.
fn fixed_size_list_indices<O, S>(
    offsets: &[O],
    sizes: &[S],
    valid: &Mask,
    list_size: u32,
    dtype: &DType,
) -> VortexResult<Buffer<u64>>
where
    O: NativePType + AsPrimitive<u64>,
    S: NativePType + AsPrimitive<u64>,
{
    let list_size_usize = list_size as usize;
    let list_size = u64::from(list_size);
    let mut indices = BufferMut::<u64>::with_capacity(offsets.len() * list_size_usize);
    for (row, (&offset, &size)) in offsets.iter().zip(sizes).enumerate() {
        if !valid.value(row) {
            indices.extend(std::iter::repeat_n(0u64, list_size_usize));
            continue;
        }
        let size: u64 = size.as_();
        vortex_ensure!(
            size == list_size,
            "Cannot cast {dtype} to FixedSizeList[{list_size}]: list at index {row} has {size} \
             elements"
        );
        let start: u64 = offset.as_();
        indices.extend(start..start + list_size);
    }
    Ok(indices.freeze())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListViewArray;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn cast_list_view_to_fixed_size_list() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // Lists [1, 2], null (any size), [5, 6] over non-contiguous elements.
        let array = ListViewArray::try_new(
            buffer![1i32, 2, 9, 5, 6].into_array(),
            buffer![0u32, 2, 3].into_array(),
            buffer![2u32, 1, 2].into_array(),
            Validity::from_iter([true, false, true]),
        )?
        .into_array();
        let target = DType::FixedSizeList(
            DType::Primitive(PType::I64, Nullability::NonNullable).into(),
            2,
            Nullability::Nullable,
        );

        let result = array
            .cast(target.clone())?
            .execute::<FixedSizeListArray>(&mut ctx)?
            .into_array();
        assert_eq!(result.dtype(), &target);
        assert!(result.execute_scalar(1, &mut ctx)?.is_null());
        assert_eq!(
            result.execute_scalar(2, &mut ctx)?,
            Scalar::fixed_size_list(
                Arc::new(DType::Primitive(PType::I64, Nullability::NonNullable)),
                vec![Scalar::from(5i64), Scalar::from(6i64)],
                Nullability::Nullable,
            )
        );
        Ok(())
    }

    #[test]
    fn cast_list_view_to_fixed_size_list_rejects_wrong_sizes() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = ListViewArray::try_new(
            buffer![1i32, 2, 3].into_array(),
            buffer![0u32, 2].into_array(),
            buffer![2u32, 1].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let target = DType::FixedSizeList(
            DType::Primitive(PType::I32, Nullability::NonNullable).into(),
            2,
            Nullability::NonNullable,
        );

        // List sizes are a value-level property: the cast binds but fails at execution.
        let result = array.cast(target)?.execute::<FixedSizeListArray>(&mut ctx);
        assert!(result.is_err(), "expected error, got {result:?}");
        Ok(())
    }
}
