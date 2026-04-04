mod layer;
mod snapshot;
mod store;

pub mod test;

use std::{
    future::Future,
    sync::{Arc, Mutex},
};

pub use crate::layer::{CaptureLayer, layer};
pub use crate::snapshot::{EventRecord, FieldValue, SpanNode, TraceSnapshot};

use crate::{layer::CAPTURE, snapshot::build_snapshot, store::InnerStore};

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
    let store = Arc::new(Mutex::new(InnerStore::default()));
    let output = CAPTURE.scope(store.clone(), future).await;
    let trace = {
        let store = store.lock().unwrap_or_else(|err| err.into_inner());
        build_snapshot(&store)
    };

    Collected { output, trace }
}
