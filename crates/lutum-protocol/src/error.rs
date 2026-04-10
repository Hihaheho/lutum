use std::error::Error as StdError;

use thiserror::Error;

pub type BoxError = Box<dyn StdError + Send + Sync + 'static>;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("invalid model input: {0}")]
    InvalidModelInput(#[from] crate::conversation::ModelInputValidationError),
    #[error("budget error: {0}")]
    Budget(#[source] BoxError),
    #[error("backend error: {0}")]
    Backend(#[source] BoxError),
    #[error("failed to parse tool call: {0}")]
    ToolCall(#[from] crate::toolset::ToolCallError),
    #[error("failed to decode JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("failed to decode structured output: {0}")]
    StructuredOutput(#[source] serde_json::Error),
    #[error("no-tools contract violated: {0}")]
    NoToolsContractViolation(#[from] NoToolsContractViolation),
    #[error("invalid tool constraints: required tool `{tool}` is not in the available set")]
    InvalidToolConstraints { tool: String },
    #[error("other error: {0}")]
    Other(#[source] BoxError),
}

impl AgentError {
    pub fn budget(source: impl StdError + Send + Sync + 'static) -> Self {
        Self::Budget(Box::new(source))
    }

    pub fn backend(source: impl StdError + Send + Sync + 'static) -> Self {
        Self::Backend(Box::new(source))
    }

    pub fn structured_output(source: serde_json::Error) -> Self {
        Self::StructuredOutput(source)
    }

    pub fn other(source: impl StdError + Send + Sync + 'static) -> Self {
        Self::Other(Box::new(source))
    }
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum NoToolsContractViolation {
    #[error("adapter emitted a tool call chunk during a no-tools text turn")]
    TextTurnToolCallChunk,
    #[error("adapter emitted a tool call during a no-tools text turn")]
    TextTurnToolCallReady,
    #[error("adapter completed a no-tools text turn with finish_reason=ToolCall")]
    TextTurnFinishReasonToolCall,
    #[error("adapter emitted a tool call chunk during a no-tools structured turn")]
    StructuredTurnToolCallChunk,
    #[error("adapter emitted a tool call during a no-tools structured turn")]
    StructuredTurnToolCallReady,
    #[error("adapter completed a no-tools structured turn with finish_reason=ToolCall")]
    StructuredTurnFinishReasonToolCall,
}
