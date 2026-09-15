// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::FixedSizeList;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastKernel;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::validity::Validity;

fn build_with_validity(
    array: ArrayView<'_, FixedSizeList>,
    elements: ArrayRef,
    validity: Validity,
) -> ArrayRef {
    // SAFETY: The only requirements for safety here are related to lengths, and no lengths have
    // changed here. So as long as the original array is valid, this is also valid.
    unsafe { FixedSizeListArray::new_unchecked(elements, array.list_size(), validity, array.len()) }
        .into_array()
}

/// Cast implementation for [`FixedSizeListArray`].
///
/// Recursively casts the inner elements array to the target element type while preserving the list
/// structure.
impl CastReduce for FixedSizeList {
    fn cast(
        array: ArrayView<'_, FixedSizeList>,
        dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(target_element_type) = dtype.as_fixed_size_list_element_opt() else {
            return Ok(None);
        };

        let Some(validity) = array
            .validity()?
            .trivially_cast_nullability(dtype.nullability(), array.len())?
        else {
            return Ok(None);
        };
        let elements = array
            .elements()
            .cast((**target_element_type).clone(), session)?;

        Ok(Some(build_with_validity(array, elements, validity)))
    }
}

impl CastKernel for FixedSizeList {
    fn cast(
        array: ArrayView<'_, FixedSizeList>,
        dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // A fixed-size list is a list whose offsets advance by `list_size` every row.
        if let Some(target_element_type) = dtype.as_list_element_opt() {
            let validity =
                array
                    .validity()?
                    .cast_nullability(dtype.nullability(), array.len(), ctx)?;
            let elements = array
                .elements()
                .cast((**target_element_type).clone(), ctx.session())?;
            let list_size = u64::from(array.list_size());
            let offsets =
                Buffer::<u64>::from_iter((0..=array.len()).map(|row| {
                    u64::try_from(row).vortex_expect("row index fits in u64") * list_size
                }))
                .into_array();
            return Ok(Some(
                ListArray::try_new(elements, offsets, validity)?.into_array(),
            ));
        }

        let Some(target_element_type) = dtype.as_fixed_size_list_element_opt() else {
            return Ok(None);
        };

        let validity = array
            .validity()?
            .cast_nullability(dtype.nullability(), array.len(), ctx)?;
        let elements = array
            .elements()
            .cast((**target_element_type).clone(), ctx.session())?;

        Ok(Some(build_with_validity(array, elements, validity)))
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListViewArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::validity::Validity;

    #[test]
    fn cast_fixed_size_list_to_list() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let array = FixedSizeListArray::new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            2,
            Validity::from_iter([true, false, true]),
            3,
        )
        .into_array();
        let target = DType::List(
            DType::Primitive(PType::I64, Nullability::NonNullable).into(),
            Nullability::Nullable,
        );

        let result = array
            .cast(target.clone(), ctx.session())?
            .execute::<ListViewArray>(&mut ctx)?;
        assert_eq!(result.dtype(), &target);

        let expected = ListViewArray::try_new(
            buffer![1i64, 2, 3, 4, 5, 6].into_array(),
            buffer![0u64, 2, 4].into_array(),
            buffer![2u64, 2, 2].into_array(),
            Validity::from_iter([true, false, true]),
        )?
        .into_array();
        assert_arrays_eq!(result.into_array(), expected, &mut ctx);
        Ok(())
    }
}
