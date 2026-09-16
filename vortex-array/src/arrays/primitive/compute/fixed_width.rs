// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::fixed_width::FixedWidthArray;
use crate::arrays::fixed_width::take;
use crate::buffer::BufferHandle;
use crate::validity::Validity;

impl FixedWidthArray for Primitive {
    fn byte_width(array: ArrayView<'_, Self>) -> usize {
        array.ptype().byte_width()
    }

    fn values_handle(array: ArrayView<'_, Self>) -> BufferHandle {
        array.buffer_handle().clone()
    }

    fn with_values(
        array: ArrayView<'_, Self>,
        values: ByteBuffer,
        _len: usize,
        validity: Validity,
    ) -> VortexResult<PrimitiveArray> {
        Ok(PrimitiveArray::from_byte_buffer(
            values,
            array.ptype(),
            validity,
        ))
    }

    fn with_values_handle(
        array: ArrayView<'_, Self>,
        values: BufferHandle,
        _len: usize,
        validity: Validity,
    ) -> VortexResult<PrimitiveArray> {
        // SAFETY: `with_values_handle` checked that `values` holds whole records, and callers only
        // reuse or slice the input handle on record boundaries, which preserves ptype alignment.
        Ok(unsafe { PrimitiveArray::new_unchecked_from_handle(values, array.ptype(), validity) })
    }
}

impl TakeExecute for Primitive {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        take::take(array, indices, ctx)
    }
}
