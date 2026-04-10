use thiserror::Error;

/// Errors returned by [`crate::capture`](crate::capture) and related helpers.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CaptureError {
    /// The active subscriber stack does not include [`crate::layer`](crate::layer).
    #[error(
        "lutum-trace capture layer is not installed on the active tracing subscriber (use lutum_trace::subscriber::otel_capture_subscriber or equivalent)"
    )]
    CaptureLayerNotInstalled,
    /// No valid OpenTelemetry trace id could be resolved at capture start (see [`crate::listen_trace_id`](crate::listen_trace_id)).
    #[error(
        "could not resolve a valid OpenTelemetry trace id for capture (enter a traced span or use a subscriber with tracing-opentelemetry)"
    )]
    NoTraceId,
}
