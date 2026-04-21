use std::sync::{Arc, Mutex};

use tracing::field::{Field, Visit};

use crate::raw::RawTraceEntry;
use crate::snapshot::{EventRecord, FieldValue, TraceEvent};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CaptureRecord {
    SpanOpened {
        id: u64,
        parent_id: Option<u64>,
        name: &'static str,
        target: &'static str,
        level: &'static str,
        fields: Vec<(String, FieldValue)>,
    },
    SpanRecorded {
        id: u64,
        fields: Vec<(String, FieldValue)>,
    },
    Event {
        parent_span_id: Option<u64>,
        record: EventRecord,
    },
    SpanClosed {
        id: u64,
    },
}

pub(crate) struct CaptureLog {
    pub(crate) records: Mutex<Vec<CaptureRecord>>,
    pub(crate) raw_entries: Mutex<Vec<RawTraceEntry>>,
    pub(crate) capture_raw: bool,
    pub(crate) event_sink: Option<Arc<dyn Fn(TraceEvent) + Send + Sync>>,
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
