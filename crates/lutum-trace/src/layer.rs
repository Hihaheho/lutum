use std::sync::{Arc, Mutex};

use tracing::{Event, Id, Subscriber};
use tracing_subscriber::{Layer, layer::Context, registry::LookupSpan};

use crate::{
    snapshot::EventRecord,
    store::{FieldVisitor, InnerStore, SpanData},
};

tokio::task_local! {
    pub(crate) static CAPTURE: Arc<Mutex<InnerStore>>;
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

impl<S> Layer<S> for CaptureLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        attrs.record(&mut visitor);
        let (fields, _) = visitor.into_parts();

        let raw_parent = if attrs.is_root() {
            None
        } else if let Some(parent) = attrs.parent() {
            Some(parent.clone().into_u64())
        } else if attrs.is_contextual() {
            ctx.current_span().id().map(|current| current.into_u64())
        } else {
            None
        };

        with_store(|store| {
            let parent = raw_parent.and_then(|raw| store.active_ids.get(&raw).copied());
            let key = store.alloc_key();
            let raw_id = id.clone().into_u64();

            store.spans.insert(
                key,
                SpanData {
                    name: attrs.metadata().name(),
                    target: attrs.metadata().target(),
                    level: attrs.metadata().level().to_string(),
                    fields,
                    events: Vec::new(),
                    children: Vec::new(),
                },
            );

            if let Some(parent) = parent {
                if let Some(parent_span) = store.spans.get_mut(&parent) {
                    parent_span.children.push(key);
                } else {
                    store.roots.push(key);
                }
            } else {
                store.roots.push(key);
            }

            store.active_ids.insert(raw_id, key);
        });
    }

    fn on_record(&self, id: &Id, values: &tracing::span::Record<'_>, _ctx: Context<'_, S>) {
        with_store(|store| {
            let Some(&key) = store.active_ids.get(&id.clone().into_u64()) else {
                return;
            };

            let Some(span) = store.spans.get_mut(&key) else {
                return;
            };

            let mut visitor = FieldVisitor::default();
            values.record(&mut visitor);
            let (fields, _) = visitor.into_parts();

            for (name, value) in fields {
                span.upsert_field(name, value);
            }
        });
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
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

        with_store(|store| {
            if let Some(key) = parent.and_then(|raw| store.active_ids.get(&raw).copied())
                && let Some(span) = store.spans.get_mut(&key)
            {
                span.events.push(record);
                return;
            }

            store.root_events.push(record);
        });
    }

    fn on_close(&self, id: Id, _ctx: Context<'_, S>) {
        with_store(|store| {
            store.active_ids.remove(&id.into_u64());
        });
    }
}
