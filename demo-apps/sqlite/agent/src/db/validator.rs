use sqlparser::{ast::Statement, dialect::SQLiteDialect, parser::Parser};
use thiserror::Error;

#[derive(Debug, Error, Clone)]
pub enum SqlValidationError {
    #[error("SQL parse error: {0}")]
    ParseError(String),
    #[error("only a single SQL statement is allowed; got {0}")]
    MultipleStatements(usize),
    #[error("statement type is not allowed: {0}")]
    ForbiddenStatement(String),
}

/// Allowed statement kinds for the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatementKind {
    Select,
    Insert,
    Update,
    Delete,
    CreateTable,
    AlterTable,
    CreateIndex,
}

/// Validate that `sql` is a single allowed statement with no dangerous constructs.
/// Uses the sqlparser AST — no regex keyword scanning.
pub fn validate_sql_safety(sql: &str) -> Result<StatementKind, SqlValidationError> {
    let mut stmts = Parser::parse_sql(&SQLiteDialect {}, sql)
        .map_err(|e| SqlValidationError::ParseError(e.to_string()))?;

    if stmts.len() != 1 {
        return Err(SqlValidationError::MultipleStatements(stmts.len()));
    }

    let kind = match stmts.remove(0) {
        Statement::Query(_) => StatementKind::Select,
        Statement::Insert(_) => StatementKind::Insert,
        Statement::Update { .. } => StatementKind::Update,
        Statement::Delete(_) => StatementKind::Delete,
        Statement::CreateTable(_) => StatementKind::CreateTable,
        Statement::AlterTable { .. } => StatementKind::AlterTable,
        Statement::CreateIndex(_) => StatementKind::CreateIndex,
        other => {
            return Err(SqlValidationError::ForbiddenStatement(
                other.to_string().chars().take(80).collect(),
            ));
        }
    };

    Ok(kind)
}

/// Try to extract a best-effort preview SELECT from an UPDATE or DELETE.
/// Returns None if extraction is not possible.
pub fn derive_preview_select(sql: &str) -> Option<String> {
    let mut stmts = Parser::parse_sql(&SQLiteDialect {}, sql).ok()?;
    if stmts.len() != 1 {
        return None;
    }
    match stmts.remove(0) {
        Statement::Update {
            table, selection, ..
        } => {
            let tbl = table.relation.to_string();
            Some(match selection {
                Some(sel) => format!("SELECT * FROM {tbl} WHERE {sel} LIMIT 10"),
                None => format!("SELECT * FROM {tbl} LIMIT 10"),
            })
        }
        Statement::Delete(del) => {
            // from is a FromTable enum; extract the first table from either variant
            use sqlparser::ast::FromTable;
            let tables = match del.from {
                FromTable::WithFromKeyword(ref v) | FromTable::WithoutKeyword(ref v) => v,
            };
            let tbl = tables.first().map(|t| t.relation.to_string())?;
            Some(match del.selection {
                Some(sel) => format!("SELECT * FROM {tbl} WHERE {sel} LIMIT 10"),
                None => format!("SELECT * FROM {tbl} LIMIT 10"),
            })
        }
        _ => None,
    }
}

/// Returns true when `sql` is a single UPDATE statement.
/// Returns false on parse error or any other statement kind.
pub fn is_update_sql(sql: &str) -> bool {
    let Ok(mut stmts) = Parser::parse_sql(&SQLiteDialect {}, sql) else {
        return false;
    };
    if stmts.len() != 1 {
        return false;
    }
    matches!(stmts.remove(0), Statement::Update { .. })
}
