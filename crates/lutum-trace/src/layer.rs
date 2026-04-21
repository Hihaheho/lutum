use std::{
    collections::HashMap,
    num::NonZeroU64,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use tracing::{Dispatch, Event, Id, Subscriber, field::Visit};
use tracing_subscriber::{Layer, layer::Context, registry::LookupSpan};

use crate::{
    filter::{event_interesting, should_mark_span},
    raw::parse_raw_entry,
    snapshot::{EventRecord, TraceEvent, TraceSpanId},
    store::{CaptureLog, CaptureRecord, FieldVisitor},
};

pub(crate) const CAPTURE_ANCHOR_TARGET: &str = "lutum_trace::capture";
pub(crate) const CAPTURE_ID_FIELD: &str = "lutum.capture_id";

static CAPTURE_REGISTRY: OnceLock<Mutex<HashMap<u64, Arc<CaptureLog>>>> = OnceLock::new();
static NEXT_CAPTURE_ID: AtomicU64 = AtomicU64::new(0);
static CAPTURE_LAYER_INSTALLED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Copy, Default)]
pub struct CaptureLayer;

pub fn layer() -> CaptureLayer {
    CaptureLayer
}

pub(crate) struct LutumCaptureLog(pub(crate) Arc<CaptureLog>);

struct AnchorCaptureId(u64);

struct CaptureIdVisitor(Option<u64>);

impl Visit for CaptureIdVisitor {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        if field.name() == CAPTURE_ID_FIELD {
            self.0 = Some(value);
        }
    }

    fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}

    fn record_i64(&mut self, _: &tracing::field::Field, _: i64) {}

    fn record_str(&mut self, _: &tracing::field::Field, _: &str) {}

    fn record_bool(&mut self, _: &tracing::field::Field, _: bool) {}

    fn record_i128(&mut self, _: &tracing::field::Field, _: i128) {}

    fn record_u128(&mut self, _: &tracing::field::Field, _: u128) {}

    fn record_f64(&mut self, _: &tracing::field::Field, _: f64) {}

    fn record_error(&mut self, _: &tracing::field::Field, _: &(dyn std::error::Error + 'static)) {}
}

fn capture_registry() -> &'static Mutex<HashMap<u64, Arc<CaptureLog>>> {
    CAPTURE_REGISTRY.get_or_init(Default::default)
}

pub(crate) fn ensure_capture_layer_installed() -> bool {
    CAPTURE_LAYER_INSTALLED.load(Ordering::Acquire)
}

#[cfg(test)]
pub(crate) fn reset_capture_layer_installed_for_test() {
    CAPTURE_LAYER_INSTALLED.store(false, Ordering::Release);
}

pub(crate) fn alloc_capture_id() -> u64 {
    NEXT_CAPTURE_ID.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn register_capture(id: u64, log: Arc<CaptureLog>) {
    capture_registry()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .insert(id, log);
}

pub(crate) fn unregister_capture(id: u64) {
    capture_registry()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .remove(&id);
}

fn lookup_capture(id: u64) -> Option<Arc<CaptureLog>> {
    capture_registry()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .get(&id)
        .cloned()
}

fn id_from_u64(id: u64) -> Option<Id> {
    Some(Id::from_non_zero_u64(NonZeroU64::new(id)?))
}

fn span_has_recorded_capture<'a, S>(ctx: &Context<'a, S>, span_id: Option<u64>) -> Option<u64>
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    let id = id_from_u64(span_id?)?;
    let span = ctx.span(&id)?;
    let has_log = span.extensions().get::<LutumCaptureLog>().is_some();
    (has_log && span.metadata().target() != CAPTURE_ANCHOR_TARGET).then_some(id.into_u64())
}

fn find_log_in_parents<'a, S>(
    ctx: &Context<'a, S>,
    mut span_id: Option<u64>,
) -> Option<Arc<CaptureLog>>
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    while let Some(raw_id) = span_id {
        let id = id_from_u64(raw_id)?;
        let span = ctx.span(&id)?;
        if let Some(ext) = span.extensions().get::<LutumCaptureLog>() {
            return Some(Arc::clone(&ext.0));
        }

        span_id = span.parent().map(|parent| parent.id().into_u64());
    }

    None
}

fn find_log_for_scope<'a, S>(ctx: &Context<'a, S>, span_id: Option<u64>) -> Option<Arc<CaptureLog>>
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    let current_id = ctx.current_span().id().map(|id| id.into_u64());
    find_log_in_parents(ctx, span_id).or_else(|| {
        if current_id == span_id {
            None
        } else {
            find_log_in_parents(ctx, current_id)
        }
    })
}

fn emit_trace_event(log: &CaptureLog, event: TraceEvent) {
    if let Some(sink) = &log.event_sink {
        sink(event);
    }
}

fn push_record(log: &CaptureLog, record: CaptureRecord) {
    log.records
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .push(record);
}

fn push_raw_entry(log: &CaptureLog, entry: crate::RawTraceEntry) {
    log.raw_entries
        .lock()
        .unwrap_or_else(|err| err.into_inner())
        .push(entry);
}

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_register_dispatch(&self, _dispatch: &Dispatch) {
        CAPTURE_LAYER_INSTALLED.store(true, Ordering::Release);
    }

    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        if attrs.metadata().target() == CAPTURE_ANCHOR_TARGET {
            let mut visitor = CaptureIdVisitor(None);
            attrs.record(&mut visitor);
            let Some(capture_id) = visitor.0 else {
                return;
            };
            let Some(log) = lookup_capture(capture_id) else {
                return;
            };
            let Some(span) = ctx.span(id) else {
                return;
            };

            let mut extensions = span.extensions_mut();
            extensions.insert(LutumCaptureLog(log));
            extensions.insert(AnchorCaptureId(capture_id));
            return;
        }

        let parent_id = if attrs.is_root() {
            None
        } else if let Some(parent) = attrs.parent() {
            Some(parent.clone().into_u64())
        } else if attrs.is_contextual() {
            ctx.current_span().id().map(|current| current.into_u64())
        } else {
            None
        };

        let Some(log) = find_log_for_scope(&ctx, parent_id) else {
            return;
        };

        let captured_parent_id = span_has_recorded_capture(&ctx, parent_id);
        if !should_mark_span(attrs) && captured_parent_id.is_none() {
            return;
        }

        let mut visitor = FieldVisitor::default();
        attrs.record(&mut visitor);
        let (fields, _) = visitor.into_parts();
        let raw_id = id.clone().into_u64();

        push_record(
            &log,
            CaptureRecord::SpanOpened {
                id: raw_id,
                parent_id: captured_parent_id,
                name: attrs.metadata().name(),
                target: attrs.metadata().target(),
                level: attrs.metadata().level().as_str(),
                fields: fields.clone(),
            },
        );
        emit_trace_event(
            &log,
            TraceEvent::SpanOpened {
                span_id: TraceSpanId(raw_id),
                parent_span_id: captured_parent_id.map(TraceSpanId),
                name: attrs.metadata().name().to_string(),
                target: attrs.metadata().target().to_string(),
                level: attrs.metadata().level().to_string(),
                fields,
            },
        );

        if let Some(span) = ctx.span(id) {
            span.extensions_mut()
                .insert(LutumCaptureLog(Arc::clone(&log)));
        }
    }

    fn on_record(&self, id: &Id, values: &tracing::span::Record<'_>, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else {
            return;
        };
        if span.metadata().target() == CAPTURE_ANCHOR_TARGET {
            return;
        }
        let Some(log) = span
            .extensions()
            .get::<LutumCaptureLog>()
            .map(|ext| Arc::clone(&ext.0))
        else {
            return;
        };

        let mut visitor = FieldVisitor::default();
        values.record(&mut visitor);
        let (fields, _) = visitor.into_parts();
        let raw_id = id.clone().into_u64();

        push_record(
            &log,
            CaptureRecord::SpanRecorded {
                id: raw_id,
                fields: fields.clone(),
            },
        );
        emit_trace_event(
            &log,
            TraceEvent::SpanRecorded {
                span_id: TraceSpanId(raw_id),
                fields,
            },
        );
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let parent_id = ctx
            .event_span(event)
            .map(|span| span.id().into_u64())
            .or_else(|| ctx.current_span().id().map(|id| id.into_u64()));
        let Some(log) = find_log_for_scope(&ctx, parent_id) else {
            return;
        };

        if event.metadata().target() == lutum_protocol::RAW_TELEMETRY_TARGET {
            if !log.capture_raw {
                return;
            }

            let mut visitor = FieldVisitor::default();
            event.record(&mut visitor);
            let (fields, message) = visitor.into_parts();
            let record = EventRecord {
                target: event.metadata().target().to_string(),
                level: event.metadata().level().to_string(),
                message,
                fields,
            };
            if let Some(entry) = parse_raw_entry(&record) {
                push_raw_entry(&log, entry);
            }
            return;
        }

        let parent_span_id = span_has_recorded_capture(&ctx, parent_id);
        if parent_span_id.is_none() && !event_interesting(event) {
            return;
        }

        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        let (fields, message) = visitor.into_parts();
        let record = EventRecord {
            target: event.metadata().target().to_string(),
            level: event.metadata().level().to_string(),
            message,
            fields,
        };

        push_record(
            &log,
            CaptureRecord::Event {
                parent_span_id,
                record: record.clone(),
            },
        );
        emit_trace_event(
            &log,
            TraceEvent::Event {
                parent_span_id: parent_span_id.map(TraceSpanId),
                record,
            },
        );
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else {
            return;
        };
        if let Some(anchor) = span.extensions().get::<AnchorCaptureId>() {
            unregister_capture(anchor.0);
            return;
        }
        let Some(log) = span
            .extensions()
            .get::<LutumCaptureLog>()
            .map(|ext| Arc::clone(&ext.0))
        else {
            return;
        };

        let raw_id = id.into_u64();
        push_record(&log, CaptureRecord::SpanClosed { id: raw_id });
        emit_trace_event(
            &log,
            TraceEvent::SpanClosed {
                span_id: TraceSpanId(raw_id),
            },
        );
    }
}
