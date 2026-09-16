// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::arrays::VarBinView;
use crate::arrays::VarBinViewArray;
use crate::arrays::fixed_width::FixedWidthArray;
use crate::arrays::varbinview::BinaryView;
use crate::buffer::BufferHandle;
use crate::validity::Validity;

/// A `VarBinView` is fixed-width in its views: each row is one [`BinaryView`], and the variable
/// length payload lives in separate data buffers that structural kernels carry along unchanged.
///
/// Only the kernels that move whole views without rewriting them use this. `take` and `filter`
/// keep their own implementations, which also compact the data buffers.
impl FixedWidthArray for VarBinView {
    fn byte_width(_array: ArrayView<'_, Self>) -> usize {
        size_of::<BinaryView>()
    }

    fn values_handle(array: ArrayView<'_, Self>) -> BufferHandle {
        array.views_handle().clone()
    }

    fn with_values(
        array: ArrayView<'_, Self>,
        values: ByteBuffer,
        len: usize,
        validity: Validity,
    ) -> VortexResult<VarBinViewArray> {
        let views = BufferHandle::new_host(values.aligned(Alignment::of::<BinaryView>()));
        Self::with_values_handle(array, views, len, validity)
    }

    fn with_values_handle(
        array: ArrayView<'_, Self>,
        values: BufferHandle,
        _len: usize,
        validity: Validity,
    ) -> VortexResult<VarBinViewArray> {
        let dtype = array.dtype().with_nullability(validity.nullability());
        // SAFETY: `with_values_handle` checked that `values` holds whole views, and callers only
        // reuse or slice the input handle on view boundaries. The data buffers are carried over
        // unchanged, so every buffer index the surviving views reference still resolves.
        Ok(unsafe {
            VarBinViewArray::new_handle_unchecked(
                values,
                Arc::clone(array.data_buffers()),
                dtype,
                validity,
            )
        })
    }
}
