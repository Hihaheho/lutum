use sqlite_agent::SqliteDb;

/// Check SQL syntax by running `EXPLAIN <sql>` against the live database.
/// Returns true if the SQL is syntactically valid.
pub fn check_syntax(db: &SqliteDb, sql: &str) -> bool {
    db.check_syntax(sql).is_ok()
}
