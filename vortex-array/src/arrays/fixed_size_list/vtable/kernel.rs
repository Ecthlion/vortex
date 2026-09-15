// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::VortexSession;

use crate::ArrayVTable;
use crate::arrays::Dict;
use crate::arrays::FixedSizeList;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::CastExecuteAdaptor;
use crate::scalar_fn::fns::cast::KernelCast;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(
        KernelCast.id(),
        FixedSizeList,
        CastExecuteAdaptor(FixedSizeList),
    );
    kernels.register_execute_parent_kernel(
        Dict.id(),
        FixedSizeList,
        TakeExecuteAdaptor(FixedSizeList),
    );
}
