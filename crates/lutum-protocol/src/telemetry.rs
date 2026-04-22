use crate::{extensions::RequestExtensions, llm::OperationKind};

pub const RAW_TELEMETRY_TARGET: &str = "lutum.raw";

pub const RAW_FIELD_PROVIDER: &str = "provider";
pub const RAW_FIELD_API: &str = "api";
pub const RAW_FIELD_OPERATION: &str = "operation";
pub const RAW_FIELD_KIND: &str = "kind";
pub const RAW_FIELD_REQUEST_ID: &str = "request_id";
pub const RAW_FIELD_SEQUENCE: &str = "sequence";
pub const RAW_FIELD_PAYLOAD: &str = "payload";
pub const RAW_FIELD_EVENT_NAME: &str = "event_name";
pub const RAW_FIELD_STAGE: &str = "stage";
pub const RAW_FIELD_ERROR: &str = "error";
pub const RAW_FIELD_PARTIAL_SUMMARY: &str = "partial_summary";
pub const RAW_FIELD_COLLECT_KIND: &str = "collect_kind";
pub const RAW_FIELD_REQUEST_ERROR_KIND: &str = "request_error_kind";
pub const RAW_FIELD_STATUS: &str = "status";
pub const RAW_FIELD_ERROR_DEBUG: &str = "error_debug";
pub const RAW_FIELD_SOURCE_CHAIN: &str = "source_chain";
pub const RAW_FIELD_IS_TIMEOUT: &str = "is_timeout";
pub const RAW_FIELD_IS_CONNECT: &str = "is_connect";
pub const RAW_FIELD_IS_REQUEST: &str = "is_request";
pub const RAW_FIELD_IS_BODY: &str = "is_body";
pub const RAW_FIELD_IS_DECODE: &str = "is_decode";

pub const RAW_KIND_REQUEST: &str = "request";
pub const RAW_KIND_STREAM_EVENT: &str = "stream_event";
pub const RAW_KIND_PARSE_ERROR: &str = "parse_error";
pub const RAW_KIND_COLLECT_ERROR: &str = "collect_error";
pub const RAW_KIND_REQUEST_ERROR: &str = "request_error";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParseErrorStage {
    SseParse,
    SseDecode,
    CompletionChunkDecode,
    ChatChunkDecode,
    StructuredOutputParse,
    ToolCallArgumentsParse,
}

impl ParseErrorStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SseParse => "sse_parse",
            Self::SseDecode => "sse_decode",
            Self::CompletionChunkDecode => "completion_chunk_decode",
            Self::ChatChunkDecode => "chat_chunk_decode",
            Self::StructuredOutputParse => "structured_output_parse",
            Self::ToolCallArgumentsParse => "tool_call_arguments_parse",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "sse_parse" => Some(Self::SseParse),
            "sse_decode" => Some(Self::SseDecode),
            "completion_chunk_decode" => Some(Self::CompletionChunkDecode),
            "chat_chunk_decode" => Some(Self::ChatChunkDecode),
            "structured_output_parse" => Some(Self::StructuredOutputParse),
            "tool_call_arguments_parse" => Some(Self::ToolCallArgumentsParse),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectErrorKind {
    Reduction,
    Execution,
    Handler,
    UnexpectedEof,
}

impl CollectErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Reduction => "reduction",
            Self::Execution => "execution",
            Self::Handler => "handler",
            Self::UnexpectedEof => "unexpected_eof",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "reduction" => Some(Self::Reduction),
            "execution" => Some(Self::Execution),
            "handler" => Some(Self::Handler),
            "unexpected_eof" => Some(Self::UnexpectedEof),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestErrorKind {
    Transport,
    HttpStatus,
}

impl RequestErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transport => "transport",
            Self::HttpStatus => "http_status",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "transport" => Some(Self::Transport),
            "http_status" => Some(Self::HttpStatus),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RequestErrorDebugInfo {
    pub error_debug: String,
    pub source_chain: Vec<String>,
    pub is_timeout: bool,
    pub is_connect: bool,
    pub is_request: bool,
    pub is_body: bool,
    pub is_decode: bool,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RawTelemetryConfig {
    pub request: bool,
    pub stream: bool,
    pub parse_errors: bool,
    pub collect_errors: bool,
}

impl RawTelemetryConfig {
    pub const fn all() -> Self {
        Self {
            request: true,
            stream: true,
            parse_errors: true,
            collect_errors: true,
        }
    }

    pub const fn none() -> Self {
        Self {
            request: false,
            stream: false,
            parse_errors: false,
            collect_errors: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RawTelemetryEmitter {
    provider: &'static str,
    api: &'static str,
    operation: &'static str,
    config: RawTelemetryConfig,
}

impl RawTelemetryEmitter {
    pub fn new(
        extensions: &RequestExtensions,
        provider: &'static str,
        api: &'static str,
        operation: &'static str,
    ) -> Option<Self> {
        let config = *extensions.get::<RawTelemetryConfig>()?;
        tracing::enabled!(target: RAW_TELEMETRY_TARGET, tracing::Level::TRACE).then_some(Self {
            provider,
            api,
            operation,
            config,
        })
    }

    pub fn emit_request(&self, request_id: Option<&str>, body: &str) {
        if !self.config.request {
            return;
        }

        tracing::event!(
            target: RAW_TELEMETRY_TARGET,
            tracing::Level::TRACE,
            provider = self.provider,
            api = self.api,
            operation = self.operation,
            kind = RAW_KIND_REQUEST,
            request_id = request_id.unwrap_or(""),
            payload = body,
        );
    }

    pub fn emit_stream_event(
        &self,
        request_id: Option<&str>,
        sequence: u64,
        payload: &str,
        event_name: Option<&str>,
    ) {
        if !self.config.stream {
            return;
        }

        tracing::event!(
            target: RAW_TELEMETRY_TARGET,
            tracing::Level::TRACE,
            provider = self.provider,
            api = self.api,
            operation = self.operation,
            kind = RAW_KIND_STREAM_EVENT,
            request_id = request_id.unwrap_or(""),
            sequence = sequence,
            event_name = event_name.unwrap_or(""),
            payload = payload,
        );
    }

    pub fn emit_parse_error(
        &self,
        request_id: Option<&str>,
        stage: ParseErrorStage,
        payload: &str,
        error: &str,
    ) {
        if !self.config.parse_errors {
            return;
        }

        tracing::event!(
            target: RAW_TELEMETRY_TARGET,
            tracing::Level::TRACE,
            provider = self.provider,
            api = self.api,
            operation = self.operation,
            kind = RAW_KIND_PARSE_ERROR,
            request_id = request_id.unwrap_or(""),
            stage = stage.as_str(),
            payload = payload,
            error = error,
        );
    }

    pub fn emit_request_error(
        &self,
        request_id: Option<&str>,
        request_error_kind: RequestErrorKind,
        status: Option<u16>,
        payload: Option<&str>,
        error: &str,
        debug_info: &RequestErrorDebugInfo,
    ) {
        if !self.config.request {
            return;
        }

        let source_chain =
            serde_json::to_string(&debug_info.source_chain).unwrap_or_else(|_| "[]".to_string());

        tracing::event!(
            target: RAW_TELEMETRY_TARGET,
            tracing::Level::TRACE,
            provider = self.provider,
            api = self.api,
            operation = self.operation,
            kind = RAW_KIND_REQUEST_ERROR,
            request_id = request_id.unwrap_or(""),
            request_error_kind = request_error_kind.as_str(),
            status = status.unwrap_or(0_u16),
            payload = payload.unwrap_or(""),
            error = error,
            error_debug = debug_info.error_debug.as_str(),
            source_chain = source_chain.as_str(),
            is_timeout = debug_info.is_timeout,
            is_connect = debug_info.is_connect,
            is_request = debug_info.is_request,
            is_body = debug_info.is_body,
            is_decode = debug_info.is_decode,
        );
    }
}

pub fn emit_collect_error(
    extensions: &RequestExtensions,
    operation_kind: OperationKind,
    request_id: Option<&str>,
    collect_kind: CollectErrorKind,
    partial_summary: &str,
    error: &str,
) {
    emit_collect_error_enabled(
        raw_collect_errors_enabled(extensions),
        operation_kind,
        request_id,
        collect_kind,
        partial_summary,
        error,
    );
}

pub fn raw_collect_errors_enabled(extensions: &RequestExtensions) -> bool {
    extensions
        .get::<RawTelemetryConfig>()
        .is_some_and(|config| config.collect_errors)
        && tracing::enabled!(target: RAW_TELEMETRY_TARGET, tracing::Level::TRACE)
}

pub fn emit_collect_error_enabled(
    enabled: bool,
    operation_kind: OperationKind,
    request_id: Option<&str>,
    collect_kind: CollectErrorKind,
    partial_summary: &str,
    error: &str,
) {
    if !enabled {
        return;
    }

    tracing::event!(
        target: RAW_TELEMETRY_TARGET,
        tracing::Level::TRACE,
        provider = "",
        api = "",
        operation = operation_kind_name(operation_kind),
        kind = RAW_KIND_COLLECT_ERROR,
        request_id = request_id.unwrap_or(""),
        collect_kind = collect_kind.as_str(),
        partial_summary = partial_summary,
        error = error,
    );
}

pub const fn operation_kind_name(kind: OperationKind) -> &'static str {
    match kind {
        OperationKind::TextTurn => "text_turn",
        OperationKind::StructuredTurn => "structured_turn",
        OperationKind::StructuredCompletion => "structured_completion",
        OperationKind::Completion => "completion",
    }
}
