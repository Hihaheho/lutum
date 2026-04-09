use crate::db::{SqlValidationError, WritePreview};

// ---------------------------------------------------------------------------
// Domain types used by hooks
// ---------------------------------------------------------------------------

/// Decision returned by the write-approval hook.
#[derive(Debug, Clone)]
pub enum WriteDecision {
    /// Execute the SQL as proposed.
    Accept,
    /// Reject execution; the reason is fed back to the model.
    Reject(String),
    /// Execute an edited version of the SQL instead.
    EditSql(String),
}

/// Current transaction mode; controls whether write tools are executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionMode {
    /// Only SELECT and schema reads are allowed.
    ReadOnly,
    /// All tools including writes may execute.
    Writable,
}

// ---------------------------------------------------------------------------
// Hook slot definitions
// ---------------------------------------------------------------------------

/// SQL safety validation — chain stops on the first error.
///
/// The default passes everything; add implementations to enforce custom rules
/// on top of the built-in sqlparser structural check that always runs in the
/// agent loop.
#[lutum::def_hook(always, chain = lutum::ShortCircuit<(), SqlValidationError>)]
pub async fn validate_sql(sql: &str) -> Result<(), SqlValidationError> {
    let _ = sql;
    Ok(())
}

/// Write-operation approval gate.
///
/// Called for every INSERT, UPDATE, DELETE before execution.  The hook
/// receives a `WritePreview` describing the SQL and how many rows would be
/// affected.  The default rejects everything — callers must register an
/// implementation (e.g. `TuiApprover` or a scripted test approver).
#[lutum::def_hook(fallback)]
pub async fn approve_write(preview: WritePreview) -> WriteDecision {
    let _ = preview;
    WriteDecision::Reject("no write approver is configured".to_string())
}

/// Current transaction mode.
///
/// Return `ReadOnly` to block all write tools, `Writable` to allow them.
/// The default is `ReadOnly` (safe); override in the TUI via `TuiModeSource`.
#[lutum::def_hook(fallback)]
pub async fn get_transaction_mode() -> TransactionMode {
    TransactionMode::ReadOnly
}

/// Write-mode request gate.
///
/// Called when the agent invokes the `request_writable_mode` tool.  Return
/// `true` to grant writable access, `false` to deny.  The default denies
/// everything — callers must register an implementation (e.g. `TuiModeRequestApprover`).
#[lutum::def_hook(fallback)]
pub async fn approve_mode_request(reason: &str) -> bool {
    let _ = reason;
    false
}

// ---------------------------------------------------------------------------
// Hook container
// ---------------------------------------------------------------------------

#[lutum::hooks]
pub struct AgentHooks {
    sql_validator: ValidateSql,
    write_approver: ApproveWrite,
    mode_source: GetTransactionMode,
    mode_request_approver: ApproveModeRequest,
}
