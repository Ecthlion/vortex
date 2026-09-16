// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::fixed_width;
use crate::scalar_fn::fns::mask::MaskReduce;

impl MaskReduce for Decimal {
    fn mask(array: ArrayView<'_, Decimal>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        fixed_width::mask::mask(array, mask)
    }
}
