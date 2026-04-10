use opentelemetry::trace::Tracer;
use tracing_subscriber::{Registry, layer::SubscriberExt, util::SubscriberInitExt};

use crate::layer::CaptureLayer;

/// Recommended subscriber stack: OpenTelemetry layer **inside**, capture layer **outside**
/// so per-span OTel data exists before the capture layer runs.
///
/// Use a real exporter in production; for tests see [`crate::test`].
pub fn otel_capture_subscriber<T>(tracer: T) -> impl tracing::Subscriber + Send + Sync + 'static
where
    T: Tracer + Send + Sync + 'static,
    T::Span: Send + Sync,
{
    Registry::default()
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .with(CaptureLayer)
}

/// Install [`otel_capture_subscriber`] as the global default (typically at process start).
pub fn init_global_otel_capture<T>(tracer: T) -> Result<(), tracing_subscriber::util::TryInitError>
where
    T: Tracer + Send + Sync + 'static,
    T::Span: Send + Sync,
{
    otel_capture_subscriber(tracer).try_init()
}
