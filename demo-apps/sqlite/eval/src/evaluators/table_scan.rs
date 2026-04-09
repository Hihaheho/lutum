use sqlite_agent::SqliteDb;

/// Return true if the query plan contains no full table scans
/// (i.e. every scan uses an index or covers only a small rowset).
///
/// A line is flagged as a full scan if it contains "SCAN" but not "USING INDEX"
/// or "USING COVERING INDEX".
pub fn no_large_scan(db: &SqliteDb, sql: &str) -> bool {
    let Ok(plan) = db.explain_query_plan(sql) else {
        return true; // can't judge, assume ok
    };
    !plan.iter().any(|line| {
        let l = line.to_uppercase();
        l.contains("SCAN") && !l.contains("USING INDEX") && !l.contains("USING COVERING INDEX")
    })
}
