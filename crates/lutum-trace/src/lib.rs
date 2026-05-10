mod filter;
mod layer;
mod raw;
mod snapshot;
mod store;

pub mod test;

use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

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
    store::{CaptureLog, EventSink, SpanSink},
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

#[must_use = "capture futures do nothing unless awaited"]
pub struct Capture<F> {
    inner: CaptureFuture<F>,
}

#[must_use = "capture futures do nothing unless awaited"]
pub struct CaptureRaw<F> {
    inner: CaptureFuture<F>,
}

pub fn capture<F>(future: F) -> Capture<F>
where
    F: Future,
{
    Capture {
        inner: CaptureFuture::new(future, false),
    }
}

pub fn capture_raw<F>(future: F) -> CaptureRaw<F>
where
    F: Future,
{
    CaptureRaw {
        inner: CaptureFuture::new(future, true),
    }
}

impl<F> Capture<F> {
    pub fn listen_events<E>(mut self, emit: E) -> Self
    where
        E: Fn(TraceEvent) + Send + Sync + 'static,
    {
        self.inner.listen_events(emit);
        self
    }

    pub fn listen_spans<S>(mut self, emit: S) -> Self
    where
        S: Fn(SpanNode) + Send + Sync + 'static,
    {
        self.inner.listen_spans(emit);
        self
    }
}

impl<F> CaptureRaw<F> {
    pub fn listen_events<E>(mut self, emit: E) -> Self
    where
        E: Fn(TraceEvent) + Send + Sync + 'static,
    {
        self.inner.listen_events(emit);
        self
    }

    pub fn listen_spans<S>(mut self, emit: S) -> Self
    where
        S: Fn(SpanNode) + Send + Sync + 'static,
    {
        self.inner.listen_spans(emit);
        self
    }
}

struct CaptureFuture<F> {
    future: F,
    capture_raw: bool,
    event_sink: Option<EventSink>,
    span_sink: Option<SpanSink>,
    capture_id: Option<u64>,
    log: Option<Arc<CaptureLog>>,
    anchor: Option<tracing::Span>,
    completed: bool,
}

impl<F> CaptureFuture<F> {
    fn new(future: F, capture_raw: bool) -> Self {
        Self {
            future,
            capture_raw,
            event_sink: None,
            span_sink: None,
            capture_id: None,
            log: None,
            anchor: None,
            completed: false,
        }
    }

    fn listen_events<E>(&mut self, emit: E)
    where
        E: Fn(TraceEvent) + Send + Sync + 'static,
    {
        self.event_sink = Some(Arc::new(emit));
    }

    fn listen_spans<S>(&mut self, emit: S)
    where
        S: Fn(SpanNode) + Send + Sync + 'static,
    {
        self.span_sink = Some(Arc::new(emit));
    }

    fn start(&mut self) {
        assert!(
            ensure_capture_layer_installed(),
            "lutum_trace::capture called without the capture layer installed on the active subscriber. \
             Install it with: tracing_subscriber::registry().with(lutum_trace::layer())"
        );

        let capture_id = alloc_capture_id();
        let log = Arc::new(CaptureLog {
            records: Mutex::new(Vec::new()),
            raw_entries: Mutex::new(Vec::new()),
            capture_raw: self.capture_raw,
            event_sink: self.event_sink.take(),
            span_sink: self.span_sink.take(),
        });
        register_capture(capture_id, Arc::clone(&log));

        let anchor = tracing::trace_span!(
            target: layer::CAPTURE_ANCHOR_TARGET,
            "capture",
            lutum.capture_id = capture_id,
        );

        self.capture_id = Some(capture_id);
        self.log = Some(log);
        self.anchor = Some(anchor);
    }

    fn poll_raw(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<CollectedRaw<F::Output>>
    where
        F: Future,
    {
        // Safety: after CaptureFuture is pinned, `future` is never moved before
        // drop. The other fields are not structurally pinned.
        let this = unsafe { self.get_unchecked_mut() };
        assert!(
            !this.completed,
            "lutum_trace capture future polled after completion"
        );

        if this.capture_id.is_none() {
            this.start();
        }

        let output = {
            let anchor = this.anchor.as_ref().expect("capture anchor initialized");
            let _guard = anchor.enter();
            // Safety: `future` is structurally pinned by CaptureFuture and is not
            // moved after the outer future is pinned.
            let future = unsafe { Pin::new_unchecked(&mut this.future) };
            match future.poll(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(output) => output,
            }
        };

        let capture_id = this.capture_id.expect("capture id initialized");
        unregister_capture(capture_id);
        this.completed = true;

        let log = this.log.as_ref().expect("capture log initialized");
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

        Poll::Ready(CollectedRaw { output, trace, raw })
    }
}

impl<F> Future for Capture<F>
where
    F: Future,
{
    type Output = Collected<F::Output>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Safety: `inner` is structurally pinned with Capture.
        let inner = unsafe { self.map_unchecked_mut(|this| &mut this.inner) };
        inner
            .poll_raw(cx)
            .map(|collected_raw| collected_raw.into_collected())
    }
}

impl<F> Future for CaptureRaw<F>
where
    F: Future,
{
    type Output = CollectedRaw<F::Output>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Safety: `inner` is structurally pinned with CaptureRaw.
        let inner = unsafe { self.map_unchecked_mut(|this| &mut this.inner) };
        inner.poll_raw(cx)
    }
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
