// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared structural operations for fixed-width canonical arrays.
//!
//! Two traits split the work. [`FixedWidthArray`] is the per-encoding adapter that exposes an
//! array as a byte buffer of `len` records plus a way to rebuild the array. [`Record`] is the
//! per-width family of kernels those bytes are dispatched to; every array whose records are
//! 1, 2, 4, 8, 16 or 32 bytes wide runs the same kernels.

mod array;
pub(crate) mod filter;
pub(crate) mod record;
pub(crate) mod take;
pub(crate) mod vtable;

pub(crate) use self::array::FixedWidthArray;
pub(crate) use self::array::with_values;
pub(crate) use self::record::Record;

/// Dispatches a runtime byte width to the [`Record`] type `$R` for every width with dedicated
/// kernels, falling back to `$fallback` for any other width.
macro_rules! match_each_record_width {
    ($byte_width:expr, | $R:ident | $body:block,_ => $fallback:block) => {
        match $byte_width {
            1 => {
                type $R = $crate::arrays::fixed_width::record::Record1;
                $body
            }
            2 => {
                type $R = $crate::arrays::fixed_width::record::Record2;
                $body
            }
            4 => {
                type $R = $crate::arrays::fixed_width::record::Record4;
                $body
            }
            8 => {
                type $R = $crate::arrays::fixed_width::record::Record8;
                $body
            }
            16 => {
                type $R = $crate::arrays::fixed_width::record::Record16;
                $body
            }
            32 => {
                type $R = $crate::arrays::fixed_width::record::Record32;
                $body
            }
            _ => $fallback,
        }
    };
}
pub(crate) use match_each_record_width;
