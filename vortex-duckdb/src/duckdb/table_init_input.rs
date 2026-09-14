// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::fmt::Result;

use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::cpp;
use crate::duckdb::TableFilterSet;
use crate::duckdb::TableFilterSetRef;

pub struct TableInitInput<'a> {
    pub input: &'a cpp::duckdb_vx_tfunc_init_input,
}

impl Debug for TableInitInput<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("TableInitInput")
            .field("column_ids", &self.column_ids())
            .field("table_filter_set", &self.table_filter_set())
            .field("execution_threads", &self.input.execution_threads)
            .finish()
    }
}

impl<'a> TableInitInput<'a> {
    pub fn new(input: &'a cpp::duckdb_vx_tfunc_init_input) -> Self {
        Self { input }
    }

    pub fn column_ids(&self) -> &[u64] {
        unsafe { std::slice::from_raw_parts(self.input.column_ids, self.input.column_ids_count) }
    }

    pub fn execution_threads(&self) -> VortexResult<usize> {
        let execution_threads = usize::try_from(self.input.execution_threads)
            .map_err(|_| vortex_err!("DuckDB execution thread count does not fit in usize"))?;
        if execution_threads == 0 {
            return Err(vortex_err!(
                "DuckDB execution thread count must be positive"
            ));
        }
        Ok(execution_threads)
    }

    /// Returns the table filter set for the table function.
    pub fn table_filter_set(&self) -> Option<&TableFilterSetRef> {
        let ptr = self.input.filters;
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { TableFilterSet::borrow(ptr) })
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use vortex::error::VortexResult;

    use super::TableInitInput;
    use crate::cpp;

    #[test]
    fn execution_threads_are_checked() -> VortexResult<()> {
        let mut input = cpp::duckdb_vx_tfunc_init_input {
            bind_data: ptr::null(),
            column_ids: ptr::null_mut(),
            column_ids_count: 0,
            filters: ptr::null_mut(),
            client_context: ptr::null_mut(),
            execution_threads: 14,
        };
        assert_eq!(TableInitInput::new(&input).execution_threads()?, 14);

        input.execution_threads = 1;
        assert_eq!(TableInitInput::new(&input).execution_threads()?, 1);

        input.execution_threads = 0;
        assert!(TableInitInput::new(&input).execution_threads().is_err());
        Ok(())
    }
}
