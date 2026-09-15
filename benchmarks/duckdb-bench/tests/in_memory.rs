// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use duckdb_bench::DuckClient;

    #[test]
    fn in_memory_clients_are_isolated_and_reopen_empty() -> anyhow::Result<()> {
        let mut first = DuckClient::new_in_memory()?;
        let second = DuckClient::new_in_memory()?;
        let memory_database =
            "SELECT 1 FROM duckdb_databases() WHERE NOT internal AND path IS NULL";
        assert_eq!(first.execute_query(memory_database)?.0, 1);
        first.execute_query("CREATE TABLE memory_isolation (i INTEGER)")?;
        let table_exists =
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'memory_isolation'";
        assert_eq!(first.execute_query(table_exists)?.0, 1);
        assert_eq!(second.execute_query(table_exists)?.0, 0);
        first.reopen()?;
        assert_eq!(first.execute_query(memory_database)?.0, 1);
        assert_eq!(first.execute_query(table_exists)?.0, 0);
        Ok(())
    }
}
