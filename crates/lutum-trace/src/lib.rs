mod filter;
mod layer;
mod raw;
mod snapshot;
mod store;

pub mod test;

use std::{
    future::Future,
    sync::{Arc, Mutex},
};

use tracing::Instrument as _;

pub use crate::filter::LUTUM_CAPTURE_FIELD;
pub use crate::layer::{CaptureLayer, layer};
pub use crate::raw::{RawTraceEntry, RawTraceSnapshot};
pub use crate::snapshot::{
    EventRecord, FieldValue, SpanNode, TraceEvent, TraceSnapshot, TraceSpanId,
};

use crate::{
    layer::{
        alloc_capture_id, ensure_capture_layer_installed, register_capture, unregister_capture,
    },
    snapshot::build_snapshot,
    store::CaptureLog,
};

pub struct Collected<T> {
    pub output: T,
    pub trace: TraceSnapshot,
}

pub struct CollectedRaw<T> {
    pub output: T,
    pub trace: TraceSnapshot,
    pub raw: RawTraceSnapshot,
}

pub async fn capture<F, T>(future: F) -> Collected<T>
where
    F: Future<Output = T>,
{
    capture_inner(future, None, false).await.into_collected()
}

pub async fn capture_with_events<F, T, E>(future: F, emit: E) -> Collected<T>
where
    F: Future<Output = T>,
    E: Fn(TraceEvent) + Send + Sync + 'static,
{
    capture_inner(future, Some(Arc::new(emit)), false)
        .await
        .into_collected()
}

pub async fn capture_raw<F, T>(future: F) -> CollectedRaw<T>
where
    F: Future<Output = T>,
{
    capture_inner(future, None, true).await
}

pub async fn capture_raw_with_events<F, T, E>(future: F, emit: E) -> CollectedRaw<T>
where
    F: Future<Output = T>,
    E: Fn(TraceEvent) + Send + Sync + 'static,
{
    capture_inner(future, Some(Arc::new(emit)), true).await
}

type EventSink = Arc<dyn Fn(TraceEvent) + Send + Sync>;

async fn capture_inner<F, T>(
    future: F,
    sink: Option<EventSink>,
    capture_raw: bool,
) -> CollectedRaw<T>
where
    F: Future<Output = T>,
{
    assert!(
        ensure_capture_layer_installed(),
        "lutum_trace::capture called without the capture layer installed on the active subscriber. \
         Install it with: tracing_subscriber::registry().with(lutum_trace::layer())"
    );

    let capture_id = alloc_capture_id();
    let log = Arc::new(CaptureLog {
        records: Mutex::new(Vec::new()),
        raw_entries: Mutex::new(Vec::new()),
        capture_raw,
        event_sink: sink,
    });
    register_capture(capture_id, Arc::clone(&log));

    let anchor = tracing::trace_span!(
        target: layer::CAPTURE_ANCHOR_TARGET,
        "capture",
        lutum.capture_id = capture_id,
    );

    let output = future.instrument(anchor).await;
    unregister_capture(capture_id);

    let records = log.records.lock().unwrap_or_else(|err| err.into_inner());
    let trace = build_snapshot(&records);
    drop(records);
    let raw_entries = log
        .raw_entries
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let raw = RawTraceSnapshot {
        entries: raw_entries.clone(),
    };

    CollectedRaw { output, trace, raw }
}

impl<T> CollectedRaw<T> {
    pub fn into_collected(self) -> Collected<T> {
        Collected {
            output: self.output,
            trace: self.trace,
        }
    }
}

#[cfg(test)]
mod capture_layer_tests {
    use super::*;

    #[tokio::test]
    #[should_panic(expected = "lutum_trace::capture called without the capture layer")]
    async fn capture_without_layer_panics() {
        use crate::layer::reset_capture_layer_installed_for_test;

        reset_capture_layer_installed_for_test();
        capture(async {}).await;
    }
}
