pub mod db;
pub mod hooks;
pub mod r#loop;
pub mod tools;

pub use db::{
    ColumnInfo, CreateDatabaseResult, DatabaseListEntry, DbError, DbRegistry, DdlResult,
    ListDatabasesResult, ModeRequestResult, ModifyResult, QueryResult, SchemaInfo, SqliteDb,
    TableInfo, WritePreview,
};
pub use hooks::{
    AgentHooks, AgentHooksSet, ApproveModeRequest, AugmentUserMessage, SystemPrompt,
    TransactionMode, WriteDecision,
};
pub use r#loop::{AgentConfig, AgentError, SqlHistoryEntry, TurnOutput, init_session, run_turn};
pub use lutum::Session;
pub use tools::SqlTools;

/// System prompt for the SQLite agent.
pub const SYSTEM_PROMPT: &str = "\
You are an expert SQLite database assistant. Use the provided tools to answer \
questions and make changes to databases.

Multiple databases can be registered at once. The initial database is always \
available with db_id \"main\". Use list_databases to see all available databases \
and create_database to open or create additional ones.

Rules:
- Always supply the correct db_id for every tool call.
- Call get_schema first when you do not know the table structure of a database.
- If a database has no tables, you can design and create a schema with \
  create_table and create_index.
- Only use the provided SQL tool functions — never embed raw SQL in text.
- For UPDATE and DELETE, always use a WHERE clause; never operate on all rows \
  unless the user explicitly asks.
- For SELECT, prefer indexed columns in WHERE clauses; avoid SELECT * on large tables.
- If a tool returns an error, analyse it and retry with a corrected statement.
- When a write is rejected because you are in read-only mode, call \
  `request_writable_mode` with a clear explanation of what you intend to do and why. \
  Do not attempt writes until the user grants access.
- When a write is rejected, explain why to the user and ask for clarification if needed.
";
