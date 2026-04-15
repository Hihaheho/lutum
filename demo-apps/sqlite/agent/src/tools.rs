use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::db::{
    CreateDatabaseResult, DdlResult, ListDatabasesResult, ModeRequestResult, ModifyResult,
    QueryResult, SchemaInfo,
};

// ---------------------------------------------------------------------------
// Tool input schemas — one per SQL operation
// ---------------------------------------------------------------------------

/// Request permission to switch from read-only to writable mode.
/// Call this when a write operation is needed but the current mode is read-only.
#[lutum::tool_input(name = "request_writable_mode", output = ModeRequestResult)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RequestWritableModeArgs {
    /// Why write access is needed — shown to the user in the approval modal.
    pub reason: String,
}

/// List all registered databases (their ids and file paths).
#[lutum::tool_input(name = "list_databases", output = ListDatabasesResult)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ListDatabasesArgs {}

/// Create (or open) a new SQLite database and register it under `db_id`.
#[lutum::tool_input(name = "create_database", output = CreateDatabaseResult)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CreateDatabaseArgs {
    /// Unique identifier for the new database (e.g. "analytics").
    pub db_id: String,
    /// File path on disk where the database should be created (e.g. "analytics.db").
    pub path: String,
}

/// Retrieve the complete schema (tables, columns, types, constraints) for a database.
#[lutum::tool_input(name = "get_schema", output = SchemaInfo)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct GetSchemaArgs {
    /// The database to inspect. Use `list_databases` to see available ids.
    pub db_id: String,
}

/// Run a SELECT query and return the result set.
#[lutum::tool_input(name = "select", output = QueryResult)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct SelectArgs {
    /// The database to query. Use `list_databases` to see available ids.
    pub db_id: String,
    /// A valid SQLite SELECT statement.
    pub sql: String,
}

/// Insert one or more rows. Provide the full INSERT statement.
#[lutum::tool_input(name = "insert", output = ModifyResult)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct InsertArgs {
    /// The database to modify. Use `list_databases` to see available ids.
    pub db_id: String,
    /// A valid SQLite INSERT statement.
    pub sql: String,
}

/// Update existing rows. Provide the full UPDATE statement with a WHERE clause.
#[lutum::tool_input(name = "update", output = ModifyResult)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct UpdateArgs {
    /// The database to modify. Use `list_databases` to see available ids.
    pub db_id: String,
    /// A valid SQLite UPDATE statement. Always include a WHERE clause.
    pub sql: String,
}

/// Delete rows. Provide the full DELETE statement with a WHERE clause.
#[lutum::tool_input(name = "delete", output = ModifyResult)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct DeleteArgs {
    /// The database to modify. Use `list_databases` to see available ids.
    pub db_id: String,
    /// A valid SQLite DELETE statement. Always include a WHERE clause.
    pub sql: String,
}

/// Create a new table. Provide the full CREATE TABLE statement.
#[lutum::tool_input(name = "create_table", output = DdlResult)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CreateTableArgs {
    /// The database to modify. Use `list_databases` to see available ids.
    pub db_id: String,
    /// A valid SQLite CREATE TABLE statement.
    pub sql: String,
}

/// Alter an existing table. Provide the full ALTER TABLE statement.
#[lutum::tool_input(name = "alter_table", output = DdlResult)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct AlterTableArgs {
    /// The database to modify. Use `list_databases` to see available ids.
    pub db_id: String,
    /// A valid SQLite ALTER TABLE statement.
    pub sql: String,
}

/// Create an index on a table column. Provide the full CREATE INDEX statement.
#[lutum::tool_input(name = "create_index", output = DdlResult)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CreateIndexArgs {
    /// The database to modify. Use `list_databases` to see available ids.
    pub db_id: String,
    /// A valid SQLite CREATE INDEX statement.
    pub sql: String,
}

/// Display tabular data to the user as an interactive table.
/// Use this tool instead of Markdown tables whenever presenting rows of data.
#[lutum::tool_input(name = "show_table", output = QueryResult)]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ShowTableArgs {
    /// Execute a SELECT query and display its result as a table.
    SelectQuery {
        /// The database to query. Use `list_databases` to see available ids.
        db_id: String,
        /// A valid SQLite SELECT statement.
        sql: String,
    },
    /// Display a table from explicit column headers and row data.
    Csv {
        /// Column headers.
        headers: Vec<String>,
        /// Row data — one inner Vec per row; all values as strings.
        rows: Vec<Vec<String>>,
    },
}

// ---------------------------------------------------------------------------
// Toolset enum
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
pub enum SqlTools {
    RequestWritableMode(RequestWritableModeArgs),
    ListDatabases(ListDatabasesArgs),
    CreateDatabase(CreateDatabaseArgs),
    GetSchema(GetSchemaArgs),
    Select(SelectArgs),
    Insert(InsertArgs),
    Update(UpdateArgs),
    Delete(DeleteArgs),
    CreateTable(CreateTableArgs),
    AlterTable(AlterTableArgs),
    CreateIndex(CreateIndexArgs),
    ShowTable(ShowTableArgs),
}
