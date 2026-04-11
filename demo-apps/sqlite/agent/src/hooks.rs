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

#[lutum::hooks]
pub trait AgentHooks {
    /// System prompt generation.
    ///
    /// Return the full system prompt to use when initialising a new session.
    /// The default returns the built-in `SYSTEM_PROMPT` constant. Register a
    /// custom implementation to inject domain-specific instructions or context.
    #[hook(fallback)]
    async fn system_prompt() -> String {
        crate::SYSTEM_PROMPT.to_string()
    }

    /// User message augmentation.
    ///
    /// Called once per turn with the raw user message before it is pushed to
    /// the session. The default passes it through unchanged. Register a custom
    /// implementation to prepend context, reformat the message, or inject
    /// dynamic information (e.g. current time, active database schema).
    #[hook(fallback)]
    async fn augment_user_message(message: String) -> String {
        message
    }

    /// SQL safety validation — chain stops on the first error.
    ///
    /// The default passes everything; add implementations to enforce custom rules
    /// on top of the built-in sqlparser structural check that always runs in the
    /// agent loop.
    #[hook(always, chain = lutum::ShortCircuit<(), SqlValidationError>)]
    async fn validate_sql(sql: &str) -> Result<(), SqlValidationError> {
        let _ = sql;
        Ok(())
    }

    /// Write-operation approval gate.
    ///
    /// Called for every INSERT, UPDATE, DELETE before execution. The hook
    /// receives a `WritePreview` describing the SQL and how many rows would be
    /// affected. The default rejects everything.
    #[hook(fallback)]
    async fn approve_write(preview: WritePreview) -> WriteDecision {
        let _ = preview;
        WriteDecision::Reject("no write approver is configured".to_string())
    }

    /// Current transaction mode.
    ///
    /// Return `ReadOnly` to block all write tools, `Writable` to allow them.
    #[hook(fallback)]
    async fn get_transaction_mode() -> TransactionMode {
        TransactionMode::ReadOnly
    }

    /// Write-mode request gate.
    ///
    /// Called when the agent invokes the `request_writable_mode` tool.
    #[hook(fallback)]
    async fn approve_mode_request(reason: &str) -> bool {
        let _ = reason;
        false
    }
}
