use std::{env, time::Duration};

use lutum_protocol::{AgentError, RequestFailureKind};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OpenAiError {
    #[error("OPENAI_API_KEY is not set: {0}")]
    MissingApiKey(#[source] env::VarError),
    #[error("invalid HTTP header: {0}")]
    InvalidHeader(#[source] reqwest::header::InvalidHeaderValue),
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("request failed with status {status}: {message}")]
    HttpStatus {
        status: reqwest::StatusCode,
        message: String,
        retry_after: Option<Duration>,
    },
    #[error("failed to encode or decode JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("failed to parse structured output: {0}")]
    StructuredOutput(serde_json::Error),
    #[error("unexpected SSE payload: {message}")]
    Sse { message: String },
}

fn classify_status(status: reqwest::StatusCode) -> RequestFailureKind {
    match status.as_u16() {
        401 | 403 => RequestFailureKind::Auth,
        429 => RequestFailureKind::RateLimit,
        500..=599 => RequestFailureKind::Server,
        400..=499 => RequestFailureKind::Client,
        _ => RequestFailureKind::Unknown,
    }
}

fn classify_request_error(error: &reqwest::Error) -> RequestFailureKind {
    error
        .status()
        .map(classify_status)
        .unwrap_or(RequestFailureKind::Transport)
}

impl From<OpenAiError> for AgentError {
    fn from(error: OpenAiError) -> Self {
        match error {
            OpenAiError::Request(source) => AgentError::request(
                classify_request_error(&source),
                source.status().map(|status| status.as_u16()),
                None,
                source,
            ),
            OpenAiError::HttpStatus {
                status,
                message,
                retry_after,
            } => AgentError::request(
                classify_status(status),
                Some(status.as_u16()),
                retry_after,
                OpenAiError::HttpStatus {
                    status,
                    message,
                    retry_after,
                },
            ),
            OpenAiError::Json(source) => source.into(),
            OpenAiError::StructuredOutput(source) => AgentError::structured_output(source),
            other => AgentError::backend(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn http_status_retryable_server_error_maps_to_request_failure() {
        let error: AgentError = OpenAiError::HttpStatus {
            status: reqwest::StatusCode::BAD_GATEWAY,
            message: "upstream overloaded".into(),
            retry_after: Some(Duration::from_secs(2)),
        }
        .into();

        let failure = error.request_failure().expect("request failure");
        assert_eq!(failure.kind, RequestFailureKind::Server);
        assert_eq!(failure.status, Some(502));
        assert_eq!(failure.retry_after, Some(Duration::from_secs(2)));
    }
}
