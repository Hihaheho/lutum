use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
};

use rusqlite::Connection;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::validator::derive_preview_select;

#[derive(Debug, Error)]
pub enum DbError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("validation error: {0}")]
    Validation(#[from] super::validator::SqlValidationError),
    #[error("database id already exists: {0}")]
    AlreadyExists(String),
    #[error("unknown database id: {0}")]
    NotFound(String),
}

// ---------------------------------------------------------------------------
// Result types — all Serialize so they can be returned as tool outputs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl QueryResult {
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn empty() -> Self {
        Self {
            columns: vec![],
            rows: vec![],
        }
    }
}

impl std::fmt::Display for QueryResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.columns.is_empty() {
            return write!(f, "(empty)");
        }
        write!(f, "{}", self.columns.join(" | "))?;
        for row in &self.rows {
            write!(f, "\n{}", row.join(" | "))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModifyResult {
    pub rows_affected: u64,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DdlResult {
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TableInfo {
    pub name: String,
    pub columns: Vec<ColumnInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ColumnInfo {
    pub name: String,
    pub col_type: String,
    pub not_null: bool,
    pub primary_key: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SchemaInfo {
    pub tables: Vec<TableInfo>,
    /// Set when schema retrieval failed (e.g. unknown db_id).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl SchemaInfo {
    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            tables: vec![],
            error: Some(msg.into()),
        }
    }
}

impl std::fmt::Display for SchemaInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for t in &self.tables {
            write!(f, "{}(", t.name)?;
            let cols: Vec<String> = t
                .columns
                .iter()
                .map(|c| {
                    let mut s = format!("{} {}", c.name, c.col_type);
                    if c.primary_key {
                        s.push_str(" PK");
                    }
                    if c.not_null {
                        s.push_str(" NOT NULL");
                    }
                    s
                })
                .collect();
            write!(f, "{})", cols.join(", "))?;
            writeln!(f)?;
        }
        Ok(())
    }
}

/// A preview of the rows that would be affected by a write, without committing.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WritePreview {
    pub sql: String,
    pub rows_affected: u64,
    pub sample: QueryResult,
}

// ---------------------------------------------------------------------------
// SqliteDb
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SqliteDb {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteDb {
    pub fn open(path: &Path) -> Result<Self, DbError> {
        let conn = Connection::open(path)?;
        // Enable WAL for better concurrency
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn open_in_memory() -> Result<Self, DbError> {
        let conn = Connection::open_in_memory()?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Return the full schema of all user tables.
    pub fn get_schema(&self) -> Result<SchemaInfo, DbError> {
        let conn = self.conn.lock().unwrap();

        let table_names: Vec<String> = {
            let mut stmt = conn.prepare(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )?;
            stmt.query_map([], |r| r.get(0))?
                .collect::<Result<_, _>>()?
        };

        let mut tables = Vec::with_capacity(table_names.len());
        for name in table_names {
            let mut stmt = conn.prepare(&format!("PRAGMA table_info(\"{}\")", name))?;
            let columns: Vec<ColumnInfo> = stmt
                .query_map([], |r| {
                    Ok(ColumnInfo {
                        name: r.get(1)?,
                        col_type: r.get(2)?,
                        not_null: r.get::<_, i32>(3)? != 0,
                        primary_key: r.get::<_, i32>(5)? != 0,
                    })
                })?
                .collect::<Result<_, _>>()?;
            tables.push(TableInfo { name, columns });
        }

        Ok(SchemaInfo {
            tables,
            error: None,
        })
    }

    /// Execute a read-only query and return the result set.
    pub fn execute_read(&self, sql: &str) -> Result<QueryResult, DbError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(sql)?;
        let columns: Vec<String> = stmt.column_names().into_iter().map(String::from).collect();
        let col_count = columns.len();
        let rows: Vec<Vec<String>> = stmt
            .query_map([], |row| {
                (0..col_count)
                    .map(|i| row.get::<_, rusqlite::types::Value>(i).map(value_to_string))
                    .collect()
            })?
            .collect::<Result<_, _>>()?;
        Ok(QueryResult { columns, rows })
    }

    /// Dry-run a write: execute inside a savepoint, capture changes(), then rollback.
    /// Also returns a sample of rows that would be affected (via derived SELECT).
    pub fn dry_run_write(&self, sql: &str) -> Result<WritePreview, DbError> {
        let conn = self.conn.lock().unwrap();
        conn.execute("SAVEPOINT sqlite_agent_dry_run", [])?;

        let rows_affected: u64 = match conn.execute_batch(sql) {
            Ok(_) => {
                let n: i64 = conn
                    .query_row("SELECT changes()", [], |r| r.get(0))
                    .unwrap_or(0);
                n.max(0) as u64
            }
            Err(_) => 0,
        };

        conn.execute("ROLLBACK TO SAVEPOINT sqlite_agent_dry_run", [])?;
        conn.execute("RELEASE SAVEPOINT sqlite_agent_dry_run", [])?;

        // Best-effort: show sample rows that would be affected
        let sample = match derive_preview_select(sql) {
            Some(sel_sql) => {
                let mut stmt = conn.prepare(&sel_sql)?;
                let columns: Vec<String> =
                    stmt.column_names().into_iter().map(String::from).collect();
                let col_count = columns.len();
                let rows: Vec<Vec<String>> = stmt
                    .query_map([], |row| {
                        (0..col_count)
                            .map(|i| row.get::<_, rusqlite::types::Value>(i).map(value_to_string))
                            .collect()
                    })?
                    .collect::<Result<_, _>>()?;
                QueryResult { columns, rows }
            }
            None => QueryResult::empty(),
        };

        Ok(WritePreview {
            sql: sql.to_string(),
            rows_affected,
            sample,
        })
    }

    /// Execute a write statement (INSERT / UPDATE / DELETE). Always commits.
    pub fn execute_write(&self, sql: &str) -> Result<ModifyResult, DbError> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(sql)?;
        let rows_affected = conn.changes() as u64;
        Ok(ModifyResult {
            rows_affected,
            message: format!("{rows_affected} row(s) affected"),
        })
    }

    /// Execute a DDL statement (CREATE TABLE / ALTER TABLE).
    pub fn execute_ddl(&self, sql: &str) -> Result<DdlResult, DbError> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(sql)?;
        Ok(DdlResult {
            message: "DDL executed successfully".to_string(),
        })
    }

    /// Run EXPLAIN QUERY PLAN and return the plan lines.
    pub fn explain_query_plan(&self, sql: &str) -> Result<Vec<String>, DbError> {
        let conn = self.conn.lock().unwrap();
        let plan_sql = format!("EXPLAIN QUERY PLAN {sql}");
        let mut stmt = conn.prepare(&plan_sql)?;
        let lines: Vec<String> = stmt
            .query_map([], |r| r.get::<_, String>(3))?
            .collect::<Result<_, _>>()?;
        Ok(lines)
    }

    /// Validate SQL syntax by running EXPLAIN (no side effects).
    pub fn check_syntax(&self, sql: &str) -> Result<(), DbError> {
        let conn = self.conn.lock().unwrap();
        conn.prepare(&format!("EXPLAIN {sql}"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Multi-database output types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DatabaseListEntry {
    pub db_id: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ListDatabasesResult {
    pub databases: Vec<DatabaseListEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CreateDatabaseResult {
    pub db_id: String,
    pub path: String,
    pub message: String,
}

impl CreateDatabaseResult {
    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            db_id: String::new(),
            path: String::new(),
            message: msg.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModeRequestResult {
    pub granted: bool,
    pub message: String,
}

// ---------------------------------------------------------------------------
// DbRegistry — manages multiple named SqliteDb instances
// ---------------------------------------------------------------------------

struct DbEntry {
    db: Arc<SqliteDb>,
    path: String,
}

/// A named registry of `SqliteDb` connections.
///
/// The initial database is always registered as `"main"`.
pub struct DbRegistry {
    inner: Mutex<HashMap<String, DbEntry>>,
}

impl DbRegistry {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Register an already-open database under `db_id`. Returns `false` if the
    /// id is already taken.
    pub fn register(
        &self,
        db_id: impl Into<String>,
        db: Arc<SqliteDb>,
        path: impl Into<String>,
    ) -> bool {
        let mut m = self.inner.lock().unwrap();
        let id = db_id.into();
        if m.contains_key(&id) {
            return false;
        }
        m.insert(
            id,
            DbEntry {
                db,
                path: path.into(),
            },
        );
        true
    }

    /// Return a snapshot of all registered databases.
    pub fn list(&self) -> ListDatabasesResult {
        let m = self.inner.lock().unwrap();
        let mut databases: Vec<_> = m
            .iter()
            .map(|(id, e)| DatabaseListEntry {
                db_id: id.clone(),
                path: e.path.clone(),
            })
            .collect();
        databases.sort_by(|a, b| a.db_id.cmp(&b.db_id));
        ListDatabasesResult { databases }
    }

    /// Look up a database by id.
    pub fn get(&self, db_id: &str) -> Option<Arc<SqliteDb>> {
        self.inner.lock().unwrap().get(db_id).map(|e| e.db.clone())
    }

    /// Open a new database at `path` and register it as `db_id`.
    /// Returns `Err` if the id is already taken or the file cannot be opened.
    pub fn create(
        &self,
        db_id: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<CreateDatabaseResult, DbError> {
        let id = db_id.into();
        let path_str = path.as_ref().to_string_lossy().to_string();
        let mut m = self.inner.lock().unwrap();
        if m.contains_key(&id) {
            return Err(DbError::AlreadyExists(id));
        }
        let db = Arc::new(SqliteDb::open(path.as_ref())?);
        m.insert(
            id.clone(),
            DbEntry {
                db,
                path: path_str.clone(),
            },
        );
        Ok(CreateDatabaseResult {
            db_id: id,
            path: path_str,
            message: "database created and registered".to_string(),
        })
    }
}

impl Default for DbRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn value_to_string(v: rusqlite::types::Value) -> String {
    match v {
        rusqlite::types::Value::Null => "NULL".to_string(),
        rusqlite::types::Value::Integer(n) => n.to_string(),
        rusqlite::types::Value::Real(f) => f.to_string(),
        rusqlite::types::Value::Text(s) => s,
        rusqlite::types::Value::Blob(b) => format!("<blob {} bytes>", b.len()),
    }
}
