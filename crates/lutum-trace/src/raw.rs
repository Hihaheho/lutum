use lutum_protocol::{CollectErrorKind, OperationKind, ParseErrorStage, RequestErrorKind};

use crate::snapshot::{EventRecord, FieldValue};

#[derive(Debug, Clone, PartialEq)]
pub enum RawTraceEntry {
    Request {
        provider: String,
        api: String,
        operation: String,
        request_id: Option<String>,
        body: String,
    },
    StreamEvent {
        provider: String,
        api: String,
        operation: String,
        request_id: Option<String>,
        sequence: u64,
        payload: String,
        event_name: Option<String>,
    },
    ParseError {
        provider: String,
        api: String,
        operation: String,
        request_id: Option<String>,
        stage: ParseErrorStage,
        payload: String,
        error: String,
    },
    RequestError {
        provider: String,
        api: String,
        operation: String,
        request_id: Option<String>,
        kind: RequestErrorKind,
        status: Option<u16>,
        payload: Option<String>,
        error: String,
        error_debug: String,
        source_chain: Vec<String>,
        is_timeout: bool,
        is_connect: bool,
        is_request: bool,
        is_body: bool,
        is_decode: bool,
    },
    CollectError {
        operation_kind: OperationKind,
        request_id: Option<String>,
        kind: CollectErrorKind,
        partial_summary: String,
        error: String,
    },
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RawTraceSnapshot {
    pub entries: Vec<RawTraceEntry>,
}

pub(crate) fn parse_raw_entry(record: &EventRecord) -> Option<RawTraceEntry> {
    let kind = field_str(record, lutum_protocol::RAW_FIELD_KIND)?;

    match kind {
        lutum_protocol::RAW_KIND_REQUEST => Some(RawTraceEntry::Request {
            provider: field_str(record, lutum_protocol::RAW_FIELD_PROVIDER)?.to_string(),
            api: field_str(record, lutum_protocol::RAW_FIELD_API)?.to_string(),
            operation: field_str(record, lutum_protocol::RAW_FIELD_OPERATION)?.to_string(),
            request_id: optional_field_str(record, lutum_protocol::RAW_FIELD_REQUEST_ID),
            body: field_str(record, lutum_protocol::RAW_FIELD_PAYLOAD)?.to_string(),
        }),
        lutum_protocol::RAW_KIND_STREAM_EVENT => Some(RawTraceEntry::StreamEvent {
            provider: field_str(record, lutum_protocol::RAW_FIELD_PROVIDER)?.to_string(),
            api: field_str(record, lutum_protocol::RAW_FIELD_API)?.to_string(),
            operation: field_str(record, lutum_protocol::RAW_FIELD_OPERATION)?.to_string(),
            request_id: optional_field_str(record, lutum_protocol::RAW_FIELD_REQUEST_ID),
            sequence: field_u64(record, lutum_protocol::RAW_FIELD_SEQUENCE)?,
            payload: field_str(record, lutum_protocol::RAW_FIELD_PAYLOAD)?.to_string(),
            event_name: optional_field_str(record, lutum_protocol::RAW_FIELD_EVENT_NAME),
        }),
        lutum_protocol::RAW_KIND_PARSE_ERROR => Some(RawTraceEntry::ParseError {
            provider: field_str(record, lutum_protocol::RAW_FIELD_PROVIDER)?.to_string(),
            api: field_str(record, lutum_protocol::RAW_FIELD_API)?.to_string(),
            operation: field_str(record, lutum_protocol::RAW_FIELD_OPERATION)?.to_string(),
            request_id: optional_field_str(record, lutum_protocol::RAW_FIELD_REQUEST_ID),
            stage: ParseErrorStage::from_name(field_str(record, lutum_protocol::RAW_FIELD_STAGE)?)?,
            payload: field_str(record, lutum_protocol::RAW_FIELD_PAYLOAD)?.to_string(),
            error: field_str(record, lutum_protocol::RAW_FIELD_ERROR)?.to_string(),
        }),
        lutum_protocol::RAW_KIND_REQUEST_ERROR => Some(RawTraceEntry::RequestError {
            provider: field_str(record, lutum_protocol::RAW_FIELD_PROVIDER)?.to_string(),
            api: field_str(record, lutum_protocol::RAW_FIELD_API)?.to_string(),
            operation: field_str(record, lutum_protocol::RAW_FIELD_OPERATION)?.to_string(),
            request_id: optional_field_str(record, lutum_protocol::RAW_FIELD_REQUEST_ID),
            kind: RequestErrorKind::from_name(field_str(
                record,
                lutum_protocol::RAW_FIELD_REQUEST_ERROR_KIND,
            )?)?,
            status: optional_field_u16(record, lutum_protocol::RAW_FIELD_STATUS),
            payload: optional_field_str(record, lutum_protocol::RAW_FIELD_PAYLOAD),
            error: field_str(record, lutum_protocol::RAW_FIELD_ERROR)?.to_string(),
            error_debug: field_str(record, lutum_protocol::RAW_FIELD_ERROR_DEBUG)
                .unwrap_or_default()
                .to_string(),
            source_chain: optional_field_str(record, lutum_protocol::RAW_FIELD_SOURCE_CHAIN)
                .map(|json| {
                    serde_json::from_str::<Vec<String>>(&json).unwrap_or_else(|_| vec![json])
                })
                .unwrap_or_default(),
            is_timeout: field_bool(record, lutum_protocol::RAW_FIELD_IS_TIMEOUT).unwrap_or(false),
            is_connect: field_bool(record, lutum_protocol::RAW_FIELD_IS_CONNECT).unwrap_or(false),
            is_request: field_bool(record, lutum_protocol::RAW_FIELD_IS_REQUEST).unwrap_or(false),
            is_body: field_bool(record, lutum_protocol::RAW_FIELD_IS_BODY).unwrap_or(false),
            is_decode: field_bool(record, lutum_protocol::RAW_FIELD_IS_DECODE).unwrap_or(false),
        }),
        lutum_protocol::RAW_KIND_COLLECT_ERROR => Some(RawTraceEntry::CollectError {
            operation_kind: operation_kind_from_str(field_str(
                record,
                lutum_protocol::RAW_FIELD_OPERATION,
            )?)?,
            request_id: optional_field_str(record, lutum_protocol::RAW_FIELD_REQUEST_ID),
            kind: CollectErrorKind::from_name(field_str(
                record,
                lutum_protocol::RAW_FIELD_COLLECT_KIND,
            )?)?,
            partial_summary: field_str(record, lutum_protocol::RAW_FIELD_PARTIAL_SUMMARY)?
                .to_string(),
            error: field_str(record, lutum_protocol::RAW_FIELD_ERROR)?.to_string(),
        }),
        _ => None,
    }
}

fn optional_field_str(record: &EventRecord, key: &str) -> Option<String> {
    field_str(record, key)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn field_str<'a>(record: &'a EventRecord, key: &str) -> Option<&'a str> {
    match record.field(key)? {
        FieldValue::Str(value) => Some(value.as_str()),
        _ => None,
    }
}

fn field_u64(record: &EventRecord, key: &str) -> Option<u64> {
    match record.field(key)? {
        FieldValue::U64(value) => Some(*value),
        FieldValue::I64(value) if *value >= 0 => Some(*value as u64),
        _ => None,
    }
}

fn field_bool(record: &EventRecord, key: &str) -> Option<bool> {
    match record.field(key)? {
        FieldValue::Bool(value) => Some(*value),
        _ => None,
    }
}

fn optional_field_u16(record: &EventRecord, key: &str) -> Option<u16> {
    field_u64(record, key)
        .filter(|value| *value > 0 && *value <= u16::MAX as u64)
        .map(|value| value as u16)
}

fn operation_kind_from_str(s: &str) -> Option<OperationKind> {
    match s {
        "text_turn" => Some(OperationKind::TextTurn),
        "structured_turn" => Some(OperationKind::StructuredTurn),
        "structured_completion" => Some(OperationKind::StructuredCompletion),
        "completion" => Some(OperationKind::Completion),
        _ => None,
    }
}
