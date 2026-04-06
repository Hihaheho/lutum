mod layer;
mod snapshot;
mod store;

pub mod test;

use std::{
    future::Future,
    sync::{Arc, Mutex},
};

pub use crate::layer::{CaptureLayer, layer};
pub use crate::snapshot::{
    EventRecord, FieldValue, SpanNode, TraceEvent, TraceSnapshot, TraceSpanId,
};

use crate::{
    layer::{CAPTURE, TRACE_EVENTS},
    snapshot::build_snapshot,
    store::InnerStore,
};

pub struct Collected<T> {
    pub output: T,
    pub trace: TraceSnapshot,
}

/// Run `future` under a capture scope and return the result together with a
/// [`TraceSnapshot`] of all spans and events emitted during execution.
///
/// Requires [`layer()`] to be installed in the active subscriber stack (either
/// via [`tracing::subscriber::set_global_default`] or
/// [`tracing::instrument::WithSubscriber`]). If the layer is absent the trace
/// will be empty but no error is returned.
///
/// # Task-local scope
///
/// The capture scope is bound to the current Tokio task. Spans and events
/// emitted by [`tokio::spawn`]'d tasks will **not** be captured. To capture
/// work in a spawned task, call [`capture`] inside that task.
pub async fn capture<F, T>(future: F) -> Collected<T>
where
    F: Future<Output = T>,
{
    capture_inner(future, None).await
}

/// Run `future` under a capture scope and synchronously emit each live
/// [`TraceEvent`] to `emit`.
///
/// Like [`capture`], this requires [`layer()`] to be installed in the active
/// subscriber stack. The sink is invoked inline from tracing callbacks, so it
/// should stay lightweight and non-blocking.
pub async fn capture_with_events<F, T, E>(future: F, emit: E) -> Collected<T>
where
    F: Future<Output = T>,
    E: Fn(TraceEvent) + Send + Sync + 'static,
{
    capture_inner(future, Some(Arc::new(emit))).await
}

type TraceEventSink = Arc<dyn Fn(TraceEvent) + Send + Sync>;

async fn capture_inner<F, T>(future: F, sink: Option<TraceEventSink>) -> Collected<T>
where
    F: Future<Output = T>,
{
    let store = Arc::new(Mutex::new(InnerStore::default()));
    let output = TRACE_EVENTS
        .scope(sink, CAPTURE.scope(store.clone(), future))
        .await;
    let trace = {
        let store = store.lock().unwrap_or_else(|err| err.into_inner());
        build_snapshot(&store)
    };

    Collected { output, trace }
}
