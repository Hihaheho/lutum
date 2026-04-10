use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::testing::trace::NoopSpanExporter;
use opentelemetry_sdk::trace::SdkTracerProvider;
use tracing::instrument::WithSubscriber as _;

use crate::subscriber::otel_capture_subscriber;

/// Run `future` with a self-contained OTel + capture subscriber and return [`crate::Collected`].
///
/// The subscriber is active for the whole capture (including trace id bootstrap), not only
/// while `future` polls.
pub async fn collect<F, T>(future: F) -> crate::Collected<T>
where
    F: std::future::Future<Output = T>,
{
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(NoopSpanExporter::new())
        .build();
    let tracer = provider.tracer("lutum_trace_test");
    let subscriber = otel_capture_subscriber(tracer);

    async { crate::try_capture(future).await }
        .with_subscriber(subscriber)
        .await
        .expect("test subscriber always includes capture + otel layers")
}
