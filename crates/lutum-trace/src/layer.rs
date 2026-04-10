use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use opentelemetry::TraceId;
use opentelemetry::trace::TraceContextExt as _;
use tracing::{Event, Id, Subscriber, span};
use tracing_opentelemetry::OpenTelemetrySpanExt as _;
use tracing_subscriber::{Layer, layer::Context, registry::LookupSpan};

use crate::filter::{CaptureInterestVisitor, lutum_target};
use crate::snapshot::{EventRecord, FieldValue, TraceEvent, TraceSpanId};
use crate::store::{FieldVisitor, InnerStore, SpanData};

/// Initial span fields stored until first `on_enter` (when OTel trace id is available).
#[derive(Debug, Default)]
pub(crate) struct LutumPendingSpan {
    pub(crate) fields: Vec<(String, FieldValue)>,
}

impl LutumPendingSpan {
    fn upsert_field(&mut self, name: String, value: FieldValue) {
        if let Some((_, existing)) = self.fields.iter_mut().find(|(key, _)| key == &name) {
            *existing = value;
            return;
        }
        self.fields.push((name, value));
    }
}

static CAPTURE_LAYER_INSTALLED: AtomicBool = AtomicBool::new(false);

pub(crate) fn ensure_capture_layer_installed() -> bool {
    CAPTURE_LAYER_INSTALLED.load(Ordering::Acquire)
}

#[cfg(test)]
pub(crate) fn reset_capture_layer_installed_for_test() {
    CAPTURE_LAYER_INSTALLED.store(false, Ordering::Release);
}

tokio::task_local! {
    pub(crate) static CAPTURE: Arc<Mutex<InnerStore>>;
}

tokio::task_local! {
    pub(crate) static LISTEN_TRACE_ID: Option<TraceId>;
}

tokio::task_local! {
    pub(crate) static TRACE_EVENTS: Option<Arc<dyn Fn(TraceEvent) + Send + Sync>>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CaptureLayer;

pub fn layer() -> CaptureLayer {
    CaptureLayer
}

fn with_store<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut InnerStore) -> R,
{
    CAPTURE
        .try_with(|arc| {
            let mut guard = arc.lock().unwrap_or_else(|err| err.into_inner());
            f(&mut guard)
        })
        .ok()
}

fn with_listen<F, R>(f: F) -> Option<R>
where
    F: FnOnce(TraceId) -> R,
{
    LISTEN_TRACE_ID
        .try_with(|opt| {
            let tid = opt.expect("LISTEN_TRACE_ID must be Some during capture");
            f(tid)
        })
        .ok()
}

fn emit_trace_event(event: TraceEvent) {
    let _ = TRACE_EVENTS.try_with(|emit| {
        if let Some(emit) = emit {
            emit(event);
        }
    });
}

fn span_otel_trace_id<'a, S>(ctx: &Context<'a, S>, id: &Id) -> Option<TraceId>
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    let span = ctx.span(id)?;
    if let Some(otel) = span.extensions().get::<tracing_opentelemetry::OtelData>()
        && let Some(tid) = otel.trace_id()
    {
        return Some(tid);
    }
    span.parent()
        .and_then(|parent| span_otel_trace_id(ctx, &parent.id()))
}

fn event_interesting(event: &Event<'_>) -> bool {
    let meta = event.metadata();
    if lutum_target(meta.target()) {
        return true;
    }
    let mut v = CaptureInterestVisitor::default();
    event.record(&mut v);
    v.lutum_capture
}

fn event_should_capture<'a, S>(event: &Event<'_>, ctx: &Context<'a, S>) -> bool
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    if event_interesting(event) {
        return true;
    }
    let Some(span) = ctx.event_span(event) else {
        return false;
    };
    let raw = span.id().into_u64();
    with_store(|store| store.active_ids.contains_key(&raw)).unwrap_or(false)
}

fn event_otel_trace_id<'a, S>(event: &Event<'_>, ctx: &Context<'a, S>) -> Option<TraceId>
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    if let Some(span) = ctx.event_span(event) {
        return span_otel_trace_id(ctx, &span.id());
    }
    let cx = tracing::Span::current().context();
    let span = cx.span();
    let tid = span.span_context().trace_id();
    if tid != TraceId::INVALID {
        Some(tid)
    } else {
        None
    }
}

fn should_mark_span(attrs: &span::Attributes<'_>) -> bool {
    if lutum_target(attrs.metadata().target()) {
        return true;
    }
    let mut v = CaptureInterestVisitor::default();
    attrs.record(&mut v);
    v.lutum_capture
}

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_register_dispatch(&self, _subscriber: &tracing::Dispatch) {
        CAPTURE_LAYER_INSTALLED.store(true, Ordering::Release);
    }

    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if !should_mark_span(attrs) {
            return;
        }

        let span = ctx.span(id).expect("span must exist after on_new_span");
        let mut visitor = FieldVisitor::default();
        attrs.record(&mut visitor);
        let (fields, _) = visitor.into_parts();
        span.extensions_mut().insert(LutumPendingSpan { fields });
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        let span = match ctx.span(id) {
            Some(s) => s,
            None => return,
        };

        if !lutum_target(span.metadata().target())
            && span.extensions().get::<LutumPendingSpan>().is_none()
        {
            return;
        }

        let Some(tid) = span_otel_trace_id(&ctx, id) else {
            return;
        };

        let _ = with_listen(|listen| {
            if tid != listen {
                return;
            }

            let raw_id = id.clone().into_u64();

            with_store(|store| {
                if store.active_ids.contains_key(&raw_id) {
                    return;
                }

                let pending = span.extensions_mut().remove::<LutumPendingSpan>();
                let fields = pending.map(|p| p.fields).unwrap_or_default();
                let fields_for_event = fields.clone();

                let meta = span.metadata();
                let raw_parent = span.parent().map(|p| p.id().into_u64());
                let parent_key = raw_parent.and_then(|rp| store.active_ids.get(&rp).copied());
                let key = store.alloc_key();

                store.spans.insert(
                    key,
                    SpanData {
                        name: meta.name(),
                        target: meta.target(),
                        level: meta.level().to_string(),
                        fields,
                        events: Vec::new(),
                        children: Vec::new(),
                    },
                );

                if let Some(pk) = parent_key {
                    if let Some(parent_span) = store.spans.get_mut(&pk) {
                        parent_span.children.push(key);
                    } else {
                        store.roots.push(key);
                    }
                } else {
                    store.roots.push(key);
                }

                store.active_ids.insert(raw_id, key);

                emit_trace_event(TraceEvent::SpanOpened {
                    span_id: TraceSpanId(key.raw()),
                    parent_span_id: parent_key.map(|p| TraceSpanId(p.raw())),
                    name: meta.name().to_string(),
                    target: meta.target().to_string(),
                    level: meta.level().to_string(),
                    fields: fields_for_event,
                });
            });
        });
    }

    fn on_record(&self, id: &Id, values: &span::Record<'_>, ctx: Context<'_, S>) {
        let span = match ctx.span(id) {
            Some(s) => s,
            None => return,
        };

        if let Some(pending) = span.extensions_mut().get_mut::<LutumPendingSpan>() {
            let mut visitor = FieldVisitor::default();
            values.record(&mut visitor);
            let (fields, _) = visitor.into_parts();
            for (name, value) in fields {
                pending.upsert_field(name, value);
            }
            return;
        }

        let raw_id = id.clone().into_u64();

        let Some(tid) = span_otel_trace_id(&ctx, id) else {
            return;
        };

        let _ = with_listen(|listen| {
            if tid != listen {
                return;
            }

            with_store(|store| {
                let Some(&key) = store.active_ids.get(&raw_id) else {
                    return;
                };

                let Some(span_data) = store.spans.get_mut(&key) else {
                    return;
                };

                let mut visitor = FieldVisitor::default();
                values.record(&mut visitor);
                let (fields, _) = visitor.into_parts();
                let fields_for_event = fields.clone();

                for (name, value) in fields {
                    span_data.upsert_field(name, value);
                }

                emit_trace_event(TraceEvent::SpanRecorded {
                    span_id: TraceSpanId(key.raw()),
                    fields: fields_for_event,
                });
            });
        });
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        if !event_should_capture(event, &ctx) {
            return;
        }

        let Some(event_tid) = event_otel_trace_id(event, &ctx) else {
            return;
        };

        let _ = with_listen(|listen| {
            if event_tid != listen {
                return;
            }

            let mut visitor = FieldVisitor::default();
            event.record(&mut visitor);
            let (fields, message) = visitor.into_parts();

            let parent = ctx.event_span(event).map(|span| span.id().into_u64());
            let record = EventRecord {
                target: event.metadata().target().to_string(),
                level: event.metadata().level().to_string(),
                message,
                fields,
            };
            let parent_span_id = parent
                .and_then(|raw| with_store(|store| store.active_ids.get(&raw).copied()))
                .flatten()
                .map(|span_id| TraceSpanId(span_id.raw()));

            with_store(|store| {
                if let Some(key) = parent.and_then(|raw| store.active_ids.get(&raw).copied())
                    && let Some(span) = store.spans.get_mut(&key)
                {
                    span.events.push(record.clone());
                    emit_trace_event(TraceEvent::Event {
                        parent_span_id,
                        record: span.events.last().cloned().expect("event just inserted"),
                    });
                    return;
                }

                store.root_events.push(record.clone());
                emit_trace_event(TraceEvent::Event {
                    parent_span_id: None,
                    record: store
                        .root_events
                        .last()
                        .cloned()
                        .expect("root event just inserted"),
                });
            });
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(&id) {
            let _ = span.extensions_mut().remove::<LutumPendingSpan>();
        }

        let Some(tid) = span_otel_trace_id(&ctx, &id) else {
            return;
        };

        let _ = with_listen(|listen| {
            if tid != listen {
                return;
            }

            with_store(|store| {
                if let Some(span_id) = store.active_ids.remove(&id.into_u64()) {
                    emit_trace_event(TraceEvent::SpanClosed {
                        span_id: TraceSpanId(span_id.raw()),
                    });
                }
            });
        });
    }
}
