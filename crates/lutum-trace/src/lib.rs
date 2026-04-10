mod error;
mod filter;
mod layer;
mod snapshot;
mod store;

pub mod subscriber;
pub mod test;

use std::{
    future::Future,
    sync::{Arc, Mutex},
};

use tracing::Instrument as _;

pub use crate::error::CaptureError;
pub use crate::filter::LUTUM_CAPTURE_FIELD;
pub use crate::layer::{CaptureLayer, layer};
pub use crate::snapshot::{
    EventRecord, FieldValue, SpanNode, TraceEvent, TraceSnapshot, TraceSpanId,
};
pub use opentelemetry::TraceId;

use opentelemetry::trace::TraceContextExt as _;
use tracing_opentelemetry::OpenTelemetrySpanExt as _;

use crate::{
    layer::{CAPTURE, LISTEN_TRACE_ID, TRACE_EVENTS, ensure_capture_layer_installed},
    snapshot::build_snapshot,
    store::InnerStore,
};

/// Span target used only to obtain an OpenTelemetry trace id when none is active.
/// This span is not recorded in [`TraceSnapshot`].
pub const BOOTSTRAP_SPAN_TARGET: &str = "lutum_trace_bootstrap";

pub struct Collected<T> {
    pub output: T,
    pub trace: TraceSnapshot,
}

/// Returns the OpenTelemetry trace id for the current tracing span context, if valid.
pub fn listen_trace_id() -> Option<TraceId> {
    let cx = tracing::Span::current().context();
    let tid = cx.span().span_context().trace_id();
    if tid != TraceId::INVALID {
        Some(tid)
    } else {
        None
    }
}

/// Run `future` under a capture scope and return the result together with a
/// [`TraceSnapshot`] of Lutum-related spans and events for the same OpenTelemetry trace.
///
/// Requires [`layer()`] on a subscriber built like [`subscriber::otel_capture_subscriber`]
/// (OpenTelemetry layer **inside**, capture layer **outside**). Returns
/// [`CaptureError::CaptureLayerNotInstalled`] if the capture layer was never registered.
///
/// The subscription trace id is taken from [`listen_trace_id()`] after optionally opening
/// a short-lived bootstrap span (see [`BOOTSTRAP_SPAN_TARGET`]) when no valid trace is active.
///
/// Only spans and events that match the Lutum filter are stored: `target` is `lutum` or
/// starts with `lutum::`, or the span includes `lutum.capture = true` (see [`LUTUM_CAPTURE_FIELD`]).
pub async fn try_capture<F, T>(future: F) -> Result<Collected<T>, CaptureError>
where
    F: Future<Output = T>,
{
    try_capture_inner(future, None).await
}

/// Like [`try_capture`], but also invokes `emit` for each live [`TraceEvent`].
pub async fn try_capture_with_events<F, T, E>(
    future: F,
    emit: E,
) -> Result<Collected<T>, CaptureError>
where
    F: Future<Output = T>,
    E: Fn(TraceEvent) + Send + Sync + 'static,
{
    try_capture_inner(future, Some(Arc::new(emit))).await
}

type TraceEventSink = Arc<dyn Fn(TraceEvent) + Send + Sync>;

async fn try_capture_inner<F, T>(
    future: F,
    sink: Option<TraceEventSink>,
) -> Result<Collected<T>, CaptureError>
where
    F: Future<Output = T>,
{
    if !ensure_capture_layer_installed() {
        return Err(CaptureError::CaptureLayerNotInstalled);
    }

    let store = Arc::new(Mutex::new(InnerStore::default()));

    let output = if let Some(trace_id) = listen_trace_id() {
        TRACE_EVENTS
            .scope(
                sink,
                LISTEN_TRACE_ID.scope(Some(trace_id), CAPTURE.scope(store.clone(), future)),
            )
            .await
    } else {
        let span = tracing::trace_span!(target: BOOTSTRAP_SPAN_TARGET, "capture_bootstrap");
        let trace_id = {
            let _enter = span.enter();
            listen_trace_id().ok_or(CaptureError::NoTraceId)?
        };
        let future = future.instrument(span);
        TRACE_EVENTS
            .scope(
                sink,
                LISTEN_TRACE_ID.scope(Some(trace_id), CAPTURE.scope(store.clone(), future)),
            )
            .await
    };

    let trace = {
        let store = store.lock().unwrap_or_else(|err| err.into_inner());
        build_snapshot(&store)
    };

    Ok(Collected { output, trace })
}

#[cfg(test)]
mod capture_layer_tests {
    use std::sync::{Mutex, OnceLock};

    use super::*;

    static CAPTURE_ERR_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    #[tokio::test]
    async fn capture_without_layer_errors() {
        let lock = CAPTURE_ERR_TEST_LOCK.get_or_init(|| Mutex::new(()));
        let _guard = lock.lock().unwrap();
        crate::layer::reset_capture_layer_installed_for_test();

        let dispatch = tracing::Dispatch::new(tracing_subscriber::registry());
        let _subscriber_guard = tracing::dispatcher::set_default(&dispatch);

        match try_capture(async { () }).await {
            Err(e) => assert_eq!(e, CaptureError::CaptureLayerNotInstalled),
            Ok(_) => panic!("expected CaptureLayerNotInstalled"),
        }
    }
}
