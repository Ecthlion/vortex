// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::VarBinView;
use crate::arrays::fixed_width;
use crate::scalar_fn::fns::mask::MaskReduce;

impl MaskReduce for VarBinView {
    fn mask(array: ArrayView<'_, VarBinView>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        fixed_width::mask::mask(array, mask)
    }
}

#[cfg(test)]
mod tests {
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::VarBinViewArray;
    use crate::compute::conformance::mask::test_mask_conformance;

    #[test]
    fn take_mask_var_bin_view_array() {
        test_mask_conformance(
            &VarBinViewArray::from_iter_str(["one", "two", "three", "four", "five"]).into_array(),
            &mut array_session().create_execution_ctx(),
        );

        test_mask_conformance(
            &VarBinViewArray::from_iter_nullable_str([
                Some("one"),
                None,
                Some("three"),
                Some("four"),
                Some("five"),
            ])
            .into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}
