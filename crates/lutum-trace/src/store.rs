use std::collections::HashMap;

use tracing::field::{Field, Visit};

use crate::snapshot::{EventRecord, FieldValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SpanKey(u64);

#[derive(Default)]
pub(crate) struct InnerStore {
    pub(crate) next_key: u64,
    pub(crate) active_ids: HashMap<u64, SpanKey>,
    pub(crate) spans: HashMap<SpanKey, SpanData>,
    pub(crate) roots: Vec<SpanKey>,
    pub(crate) root_events: Vec<EventRecord>,
}

impl InnerStore {
    pub(crate) fn alloc_key(&mut self) -> SpanKey {
        let key = SpanKey(self.next_key);
        self.next_key += 1;
        key
    }
}

pub(crate) struct SpanData {
    pub(crate) name: &'static str,
    pub(crate) target: &'static str,
    pub(crate) level: String,
    pub(crate) fields: Vec<(String, FieldValue)>,
    pub(crate) events: Vec<EventRecord>,
    pub(crate) children: Vec<SpanKey>,
}

impl SpanData {
    pub(crate) fn upsert_field(&mut self, name: String, value: FieldValue) {
        if let Some((_, existing)) = self.fields.iter_mut().find(|(key, _)| key == &name) {
            *existing = value;
            return;
        }

        self.fields.push((name, value));
    }
}

#[derive(Default)]
pub(crate) struct FieldVisitor {
    fields: Vec<(String, FieldValue)>,
    message: Option<String>,
}

impl FieldVisitor {
    pub(crate) fn into_parts(self) -> (Vec<(String, FieldValue)>, Option<String>) {
        (self.fields, self.message)
    }

    fn record_value(&mut self, field: &Field, value: FieldValue) {
        if field.name() == "message" {
            if let FieldValue::Str(message) = value {
                self.message = Some(message);
            }
            return;
        }

        self.fields.push((field.name().to_string(), value));
    }

    fn record_string(&mut self, field: &Field, value: String) {
        if field.name() == "message" {
            self.message = Some(value);
            return;
        }

        self.fields
            .push((field.name().to_string(), FieldValue::Str(value)));
    }
}

impl Visit for FieldVisitor {
    fn record_bool(&mut self, field: &Field, value: bool) {
        self.record_value(field, FieldValue::Bool(value));
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.record_value(field, FieldValue::I64(value));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.record_value(field, FieldValue::U64(value));
    }

    fn record_i128(&mut self, field: &Field, value: i128) {
        self.record_value(field, FieldValue::I128(value));
    }

    fn record_u128(&mut self, field: &Field, value: u128) {
        self.record_value(field, FieldValue::U128(value));
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.record_value(field, FieldValue::F64(value));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.record_string(field, value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.record_string(field, format!("{value:?}"));
    }

    fn record_bytes(&mut self, field: &Field, value: &[u8]) {
        self.record_string(field, format!("{value:?}"));
    }

    fn record_error(&mut self, field: &Field, value: &(dyn std::error::Error + 'static)) {
        self.record_string(field, value.to_string());
    }
}
