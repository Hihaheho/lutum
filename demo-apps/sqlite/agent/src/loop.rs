use std::sync::Arc;

use lutum::{AgentLoopError, Lutum, Session, ToolResult};
use lutum_protocol::budget::Usage;
use thiserror::Error;
use tokio::sync::{Mutex, mpsc::UnboundedSender};

use crate::{
    db::{
        CreateDatabaseResult, DbError, DbRegistry, DdlResult, ModeRequestResult, ModifyResult,
        QueryResult, SchemaInfo, SqliteDb, WritePreview, validate_sql_safety,
    },
    hooks::{AgentHooksSet, TransactionMode, WriteDecision},
    tools::{SqlTools, SqlToolsCall, SqlToolsSelector},
};

/// A single SQL execution recorded during an agent turn.
#[derive(Debug, Clone)]
pub struct SqlHistoryEntry {
    /// The SQL statement that was executed.
    pub sql: String,
    /// Result rows for SELECT statements; empty for writes/DDL.
    pub result: QueryResult,
    /// Rows affected for INSERT/UPDATE/DELETE; `None` for SELECT and DDL.
    pub rows_affected: Option<u64>,
}

/// Configuration for a single agent session.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Maximum rows a single write operation may affect.
    pub max_rows: u64,
    /// Maximum number of tool-call rounds before giving up.
    pub max_rounds: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_rows: 100,
            max_rounds: 20,
        }
    }
}

/// Output of a completed agent turn.
#[derive(Debug, Clone)]
pub struct TurnOutput {
    /// All SQL statements executed during this turn, in order.
    pub sql_history: Vec<SqlHistoryEntry>,
    /// Token usage accumulated across all rounds.
    pub usage: Usage,
}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("turn collection error: {0}")]
    Collect(String),
    #[error("database error: {0}")]
    Db(#[from] DbError),
    #[error("reached {0}-round limit without a final answer")]
    RoundLimit(usize),
}

impl<E: std::error::Error + 'static> From<AgentLoopError<E>> for AgentError {
    fn from(e: AgentLoopError<E>) -> Self {
        match e {
            AgentLoopError::RoundLimit(n) => AgentError::RoundLimit(n),
            AgentLoopError::Dispatch(e) => AgentError::Collect(e.to_string()),
            AgentLoopError::Collect(s) => AgentError::Collect(s),
        }
    }
}

/// Create a new [`Session`] initialised with the system prompt from `hooks`.
///
/// Use this instead of calling `Session::new` directly when you want the
/// [`AgentHooksSet::system_prompt`] hook to control the system prompt.
pub async fn init_session(llm: Lutum, hooks: &AgentHooksSet<'_>) -> Session {
    let mut session = Session::new(llm);
    session.push_system(hooks.system_prompt().await);
    session
}

/// Run one complete agent turn, streaming text deltas to `text_tx` if provided.
///
/// `user_message` is augmented via [`AgentHooksSet::augment_user_message`] before
/// being pushed to the session. The session is mutated in-place (committed
/// turns are appended to it).
pub async fn run_turn(
    session: &mut Session,
    registry: &DbRegistry,
    hooks: &AgentHooksSet<'_>,
    config: &AgentConfig,
    user_message: String,
    text_tx: Option<UnboundedSender<String>>,
) -> Result<TurnOutput, AgentError> {
    let formatted = hooks.augment_user_message(user_message).await;
    session.push_user(formatted);
    let sql_history: Arc<Mutex<Vec<SqlHistoryEntry>>> = Arc::new(Mutex::new(Vec::new()));

    let mode = hooks.get_transaction_mode().await;
    let available = tool_selectors_for_mode(mode);

    let history_ref = sql_history.clone();

    let mut builder = session
        .agent_loop::<SqlTools>()
        .max_rounds(config.max_rounds)
        .available_tools(available);

    if let Some(tx) = text_tx {
        builder = builder.on_text_delta(move |delta| {
            let _ = tx.send(delta);
        });
    }

    let loop_output = builder
        .run(move |call| {
            let history = history_ref.clone();
            async move { dispatch_tool(call, registry, hooks, config, &history).await }
        })
        .await
        .map_err(AgentError::from)?;

    let sql_history = Arc::try_unwrap(sql_history)
        .expect("no other Arc references after loop")
        .into_inner();

    Ok(TurnOutput {
        sql_history,
        usage: loop_output.usage,
    })
}

fn tool_selectors_for_mode(mode: TransactionMode) -> Vec<SqlToolsSelector> {
    use SqlToolsSelector::*;
    let read_tools = [ListDatabases, CreateDatabase, GetSchema, Select];
    match mode {
        TransactionMode::ReadOnly => {
            let mut v = vec![RequestWritableMode];
            v.extend(read_tools);
            v
        }
        TransactionMode::Writable => {
            let mut v = read_tools.to_vec();
            v.extend([Insert, Update, Delete, CreateTable, AlterTable, CreateIndex]);
            v
        }
    }
}

/// Dispatch a single tool call.
async fn dispatch_tool(
    call: SqlToolsCall,
    registry: &DbRegistry,
    hooks: &AgentHooksSet<'_>,
    config: &AgentConfig,
    sql_history: &Mutex<Vec<SqlHistoryEntry>>,
) -> Result<ToolResult, DbError> {
    match call {
        // ── Mode request ───────────────────────────────────────────────────
        SqlToolsCall::RequestWritableMode(c) => {
            let reason = c.input().reason.clone();
            tracing::debug!(%reason, "tool: request_writable_mode");
            let granted = hooks.approve_mode_request(&reason).await;
            Ok(c.complete(ModeRequestResult {
                granted,
                message: if granted {
                    "Write mode granted. You may now execute write operations.".to_string()
                } else {
                    "Write mode denied. Do not attempt write operations.".to_string()
                },
            })
            .unwrap())
        }

        // ── Registry operations ────────────────────────────────────────────
        SqlToolsCall::ListDatabases(c) => {
            tracing::debug!("tool: list_databases");
            Ok(c.complete(registry.list()).unwrap())
        }

        SqlToolsCall::CreateDatabase(c) => {
            let args = c.input();
            tracing::debug!(db_id = %args.db_id, path = %args.path, "tool: create_database");
            match registry.create(&args.db_id, &args.path) {
                Ok(result) => Ok(c.complete(result).unwrap()),
                Err(e) => Ok(c
                    .complete(CreateDatabaseResult::error(e.to_string()))
                    .unwrap()),
            }
        }

        // ── Schema ─────────────────────────────────────────────────────────
        SqlToolsCall::GetSchema(c) => {
            let db_id = c.input().db_id.clone();
            tracing::debug!(%db_id, "tool: get_schema");
            let Some(db) = registry.get(&db_id) else {
                return Ok(c
                    .complete(SchemaInfo::error(format!("unknown db_id: '{db_id}'")))
                    .unwrap());
            };
            let result = db.get_schema()?;
            Ok(c.complete(result).unwrap())
        }

        // ── SELECT ─────────────────────────────────────────────────────────
        SqlToolsCall::Select(c) => {
            let db_id = c.input().db_id.clone();
            let sql = c.input().sql.clone();
            tracing::debug!(%db_id, %sql, "tool: select");
            let Some(db) = registry.get(&db_id) else {
                return Ok(c
                    .complete(error_query(&format!("unknown db_id: '{db_id}'")))
                    .unwrap());
            };

            if let Err(e) = validate_sql_safety(&sql) {
                return Ok(c.complete(error_query(&e.to_string())).unwrap());
            }
            match db.execute_read(&sql) {
                Ok(qr) => {
                    sql_history.lock().await.push(SqlHistoryEntry {
                        sql,
                        result: qr.clone(),
                        rows_affected: None,
                    });
                    Ok(c.complete(qr).unwrap())
                }
                Err(e) => Ok(c.complete(error_query(&e.to_string())).unwrap()),
            }
        }

        // ── INSERT / UPDATE / DELETE ────────────────────────────────────────
        SqlToolsCall::Insert(c) => {
            let args = c.input();
            let sql = args.sql.clone();
            let db_id = args.db_id.clone();
            tracing::debug!(%db_id, %sql, "tool: insert");
            let Some(db) = registry.get(&db_id) else {
                return Ok(c
                    .complete(error_modify(&format!("unknown db_id: '{db_id}'")))
                    .unwrap());
            };
            let result = execute_write_op(&sql, &db, hooks, config).await?;
            sql_history.lock().await.push(SqlHistoryEntry {
                sql,
                result: QueryResult::empty(),
                rows_affected: Some(result.rows_affected),
            });
            Ok(c.complete(result).unwrap())
        }

        SqlToolsCall::Update(c) => {
            let args = c.input();
            let sql = args.sql.clone();
            let db_id = args.db_id.clone();
            tracing::debug!(%db_id, %sql, "tool: update");
            let Some(db) = registry.get(&db_id) else {
                return Ok(c
                    .complete(error_modify(&format!("unknown db_id: '{db_id}'")))
                    .unwrap());
            };
            let result = execute_write_op(&sql, &db, hooks, config).await?;
            sql_history.lock().await.push(SqlHistoryEntry {
                sql,
                result: QueryResult::empty(),
                rows_affected: Some(result.rows_affected),
            });
            Ok(c.complete(result).unwrap())
        }

        SqlToolsCall::Delete(c) => {
            let args = c.input();
            let sql = args.sql.clone();
            let db_id = args.db_id.clone();
            tracing::debug!(%db_id, %sql, "tool: delete");
            let Some(db) = registry.get(&db_id) else {
                return Ok(c
                    .complete(error_modify(&format!("unknown db_id: '{db_id}'")))
                    .unwrap());
            };
            let result = execute_write_op(&sql, &db, hooks, config).await?;
            sql_history.lock().await.push(SqlHistoryEntry {
                sql,
                result: QueryResult::empty(),
                rows_affected: Some(result.rows_affected),
            });
            Ok(c.complete(result).unwrap())
        }

        // ── DDL ────────────────────────────────────────────────────────────
        SqlToolsCall::CreateTable(c) => {
            let args = c.input();
            let sql = args.sql.clone();
            let db_id = args.db_id.clone();
            tracing::debug!(%db_id, %sql, "tool: create_table");
            let Some(db) = registry.get(&db_id) else {
                return Ok(c
                    .complete(error_ddl(&format!("unknown db_id: '{db_id}'")))
                    .unwrap());
            };
            let result = execute_ddl_op(&sql, &db, hooks).await?;
            sql_history.lock().await.push(SqlHistoryEntry {
                sql,
                result: QueryResult::empty(),
                rows_affected: None,
            });
            Ok(c.complete(result).unwrap())
        }

        SqlToolsCall::AlterTable(c) => {
            let args = c.input();
            let sql = args.sql.clone();
            let db_id = args.db_id.clone();
            tracing::debug!(%db_id, %sql, "tool: alter_table");
            let Some(db) = registry.get(&db_id) else {
                return Ok(c
                    .complete(error_ddl(&format!("unknown db_id: '{db_id}'")))
                    .unwrap());
            };
            let result = execute_ddl_op(&sql, &db, hooks).await?;
            sql_history.lock().await.push(SqlHistoryEntry {
                sql,
                result: QueryResult::empty(),
                rows_affected: None,
            });
            Ok(c.complete(result).unwrap())
        }

        SqlToolsCall::CreateIndex(c) => {
            let args = c.input();
            let sql = args.sql.clone();
            let db_id = args.db_id.clone();
            tracing::debug!(%db_id, %sql, "tool: create_index");
            let Some(db) = registry.get(&db_id) else {
                return Ok(c
                    .complete(error_ddl(&format!("unknown db_id: '{db_id}'")))
                    .unwrap());
            };
            let result = execute_ddl_op(&sql, &db, hooks).await?;
            sql_history.lock().await.push(SqlHistoryEntry {
                sql,
                result: QueryResult::empty(),
                rows_affected: None,
            });
            Ok(c.complete(result).unwrap())
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers — per-type error constructors
// ---------------------------------------------------------------------------

const READ_ONLY_REJECTION: &str = "rejected: currently in read-only mode — call request_writable_mode to ask the user for write access";

fn error_query(msg: &str) -> QueryResult {
    QueryResult {
        columns: vec!["error".to_string()],
        rows: vec![vec![msg.to_string()]],
    }
}

fn error_modify(msg: &str) -> ModifyResult {
    ModifyResult {
        rows_affected: 0,
        message: format!("error: {msg}"),
    }
}

fn error_ddl(msg: &str) -> DdlResult {
    DdlResult {
        message: format!("error: {msg}"),
    }
}

/// Validate and preview a write candidate before it is eligible for execution.
async fn prepare_write_candidate(
    sql: &str,
    db: &SqliteDb,
    hooks: &AgentHooksSet<'_>,
    config: &AgentConfig,
) -> Result<WritePreview, ModifyResult> {
    if let Err(e) = validate_sql_safety(sql) {
        return Err(ModifyResult {
            rows_affected: 0,
            message: format!("rejected (invalid SQL): {e}"),
        });
    }

    if let Err(e) = hooks.validate_sql(sql).await {
        return Err(ModifyResult {
            rows_affected: 0,
            message: format!("rejected (policy): {e}"),
        });
    }

    let preview = db.dry_run_write(sql).map_err(|e| ModifyResult {
        rows_affected: 0,
        message: format!("error: {e}"),
    })?;
    if preview.rows_affected > config.max_rows {
        return Err(ModifyResult {
            rows_affected: 0,
            message: format!(
                "rejected: would affect {} rows (limit is {}). Add a more restrictive WHERE clause.",
                preview.rows_affected, config.max_rows
            ),
        });
    }

    Ok(preview)
}

/// Guard + execute for write ops (INSERT / UPDATE / DELETE).
async fn execute_write_op(
    sql: &str,
    db: &SqliteDb,
    hooks: &AgentHooksSet<'_>,
    config: &AgentConfig,
) -> Result<ModifyResult, DbError> {
    if let Err(e) = validate_sql_safety(sql) {
        return Ok(ModifyResult {
            rows_affected: 0,
            message: format!("rejected (invalid SQL): {e}"),
        });
    }

    if hooks.get_transaction_mode().await == TransactionMode::ReadOnly {
        return Ok(ModifyResult {
            rows_affected: 0,
            message: READ_ONLY_REJECTION.to_string(),
        });
    }

    let preview = match prepare_write_candidate(sql, db, hooks, config).await {
        Ok(preview) => preview,
        Err(result) => return Ok(result),
    };

    match hooks.approve_write(preview).await {
        WriteDecision::Accept => Ok(db.execute_write(sql)?),
        WriteDecision::EditSql(new_sql) => {
            tracing::info!("write op SQL edited by approver");
            if let Err(result) = prepare_write_candidate(&new_sql, db, hooks, config).await {
                return Ok(result);
            }
            Ok(db.execute_write(&new_sql)?)
        }
        WriteDecision::Reject(reason) => Ok(ModifyResult {
            rows_affected: 0,
            message: format!("rejected: {reason}"),
        }),
    }
}

/// Guard + execute for DDL ops.
async fn execute_ddl_op(
    sql: &str,
    db: &SqliteDb,
    hooks: &AgentHooksSet<'_>,
) -> Result<DdlResult, DbError> {
    if let Err(e) = validate_sql_safety(sql) {
        return Ok(DdlResult {
            message: format!("rejected (invalid SQL): {e}"),
        });
    }
    if hooks.get_transaction_mode().await == TransactionMode::ReadOnly {
        return Ok(DdlResult {
            message: READ_ONLY_REJECTION.to_string(),
        });
    }
    match db.execute_ddl(sql) {
        Ok(r) => Ok(r),
        Err(e) => Ok(DdlResult {
            message: format!("error: {e}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        db::SqlValidationError,
        hooks::{ApproveWrite, GetTransactionMode, ValidateSql},
    };

    struct StaticApprover {
        decision: WriteDecision,
    }

    impl ApproveWrite for StaticApprover {
        async fn call(
            &self,
            _preview: WritePreview,
            _last: Option<WriteDecision>,
        ) -> WriteDecision {
            self.decision.clone()
        }
    }

    struct WritableMode;

    impl GetTransactionMode for WritableMode {
        async fn call(&self, _last: Option<TransactionMode>) -> TransactionMode {
            TransactionMode::Writable
        }
    }

    #[lutum::impl_hook(ValidateSql)]
    async fn reject_approved_status(sql: &str) -> Result<(), SqlValidationError> {
        if sql.contains("status = 'approved'") {
            Err(SqlValidationError::ParseError(
                "policy blocked approved status".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    fn make_db() -> SqliteDb {
        let db = SqliteDb::open_in_memory().unwrap();
        db.execute_ddl("CREATE TABLE items (id INTEGER PRIMARY KEY, status TEXT NOT NULL);")
            .unwrap();
        db.execute_write(
            "INSERT INTO items (id, status) VALUES (1, 'todo'), (2, 'todo'), (3, 'todo');",
        )
        .unwrap();
        db
    }

    fn statuses(db: &SqliteDb) -> Vec<String> {
        db.execute_read("SELECT status FROM items ORDER BY id")
            .unwrap()
            .rows
            .into_iter()
            .map(|row| row[0].clone())
            .collect()
    }

    async fn run_write(
        db: &SqliteDb,
        hooks: &AgentHooksSet<'_>,
        sql: &str,
        max_rows: u64,
    ) -> ModifyResult {
        execute_write_op(
            sql,
            db,
            hooks,
            &AgentConfig {
                max_rows,
                ..AgentConfig::default()
            },
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn edit_sql_rechecks_row_limit_before_execution() {
        let db = make_db();
        let hooks = AgentHooksSet::new()
            .with_approve_write(StaticApprover {
                decision: WriteDecision::EditSql("UPDATE items SET status = 'done'".to_string()),
            })
            .with_get_transaction_mode(WritableMode);

        let result = run_write(
            &db,
            &hooks,
            "UPDATE items SET status = 'done' WHERE id = 1",
            1,
        )
        .await;

        assert_eq!(result.rows_affected, 0);
        assert!(result.message.contains("would affect 3 rows"));
        assert_eq!(statuses(&db), vec!["todo", "todo", "todo"]);
    }

    #[tokio::test]
    async fn edit_sql_rechecks_invalid_sql_before_execution() {
        let db = make_db();
        let hooks = AgentHooksSet::new()
            .with_approve_write(StaticApprover {
                decision: WriteDecision::EditSql("UPDATE items SET".to_string()),
            })
            .with_get_transaction_mode(WritableMode);

        let result = run_write(
            &db,
            &hooks,
            "UPDATE items SET status = 'done' WHERE id = 1",
            3,
        )
        .await;

        assert_eq!(result.rows_affected, 0);
        assert!(result.message.starts_with("rejected (invalid SQL):"));
        assert_eq!(statuses(&db), vec!["todo", "todo", "todo"]);
    }

    #[tokio::test]
    async fn edit_sql_rechecks_policy_before_execution() {
        let db = make_db();
        let hooks = AgentHooksSet::new()
            .with_validate_sql(RejectApprovedStatus)
            .with_approve_write(StaticApprover {
                decision: WriteDecision::EditSql(
                    "UPDATE items SET status = 'approved' WHERE id = 1".to_string(),
                ),
            })
            .with_get_transaction_mode(WritableMode);

        let result = run_write(
            &db,
            &hooks,
            "UPDATE items SET status = 'done' WHERE id = 1",
            3,
        )
        .await;

        assert_eq!(result.rows_affected, 0);
        assert!(result.message.contains("rejected (policy):"));
        assert_eq!(statuses(&db), vec!["todo", "todo", "todo"]);
    }

    #[tokio::test]
    async fn edit_sql_executes_only_the_replacement_statement() {
        let db = make_db();
        let hooks = AgentHooksSet::new()
            .with_approve_write(StaticApprover {
                decision: WriteDecision::EditSql(
                    "UPDATE items SET status = 'done' WHERE id = 2".to_string(),
                ),
            })
            .with_get_transaction_mode(WritableMode);

        let result = run_write(
            &db,
            &hooks,
            "UPDATE items SET status = 'done' WHERE id IN (1, 2)",
            2,
        )
        .await;

        assert_eq!(result.rows_affected, 1);
        assert_eq!(statuses(&db), vec!["todo", "done", "todo"]);
    }

    #[tokio::test]
    async fn accept_executes_original_statement() {
        let db = make_db();
        let hooks = AgentHooksSet::new()
            .with_approve_write(StaticApprover {
                decision: WriteDecision::Accept,
            })
            .with_get_transaction_mode(WritableMode);

        let result = run_write(
            &db,
            &hooks,
            "UPDATE items SET status = 'done' WHERE id = 1",
            1,
        )
        .await;

        assert_eq!(result.rows_affected, 1);
        assert_eq!(statuses(&db), vec!["done", "todo", "todo"]);
    }

    #[tokio::test]
    async fn reject_leaves_original_statement_unexecuted() {
        let db = make_db();
        let hooks = AgentHooksSet::new()
            .with_approve_write(StaticApprover {
                decision: WriteDecision::Reject("declined".to_string()),
            })
            .with_get_transaction_mode(WritableMode);

        let result = run_write(
            &db,
            &hooks,
            "UPDATE items SET status = 'done' WHERE id = 1",
            1,
        )
        .await;

        assert_eq!(result.rows_affected, 0);
        assert_eq!(result.message, "rejected: declined");
        assert_eq!(statuses(&db), vec!["todo", "todo", "todo"]);
    }
}
