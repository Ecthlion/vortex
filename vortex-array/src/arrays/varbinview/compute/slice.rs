// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::VarBinView;
use crate::arrays::fixed_width;
use crate::arrays::slice::SliceReduce;

impl SliceReduce for VarBinView {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        fixed_width::slice::slice(array, range)
    }
}
