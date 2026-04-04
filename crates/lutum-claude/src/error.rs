use std::env;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ClaudeError {
    #[error("ANTHROPIC_API_KEY is not set: {0}")]
    MissingApiKey(#[source] env::VarError),
    #[error("invalid HTTP header: {0}")]
    InvalidHeader(#[source] reqwest::header::InvalidHeaderValue),
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("request failed with status {status}: {message}")]
    HttpStatus {
        status: reqwest::StatusCode,
        message: String,
    },
    #[error("failed to encode or decode JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("failed to parse structured output: {0}")]
    StructuredOutput(serde_json::Error),
    #[error("missing required field `{field}` in Claude payload")]
    MissingField { field: &'static str },
    #[error("invalid Claude request: {message}")]
    InvalidRequest { message: String },
    #[error("unexpected Claude SSE payload: {message}")]
    Sse { message: String },
}
