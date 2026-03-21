use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    pin::Pin,
    sync::Arc,
};

use async_stream::try_stream;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use reqwest::{
    Client,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue},
};
use serde_json::{Value, json};
use thiserror::Error;

use serde::{Deserialize, Serialize};

use agents_protocol::{
    AgentError,
    budget::Usage,
    conversation::{
        AssistantInputItem, InputMessageRole, MessageContent, ModelInput, ModelInputItem, RawJson,
        ToolCallId, ToolMetadata, ToolName, ToolUse,
    },
    extensions::RequestExtensions,
    llm::{
        AdapterStructuredTurn, AdapterTextTurn, AdapterToolChoice, AdapterTurnConfig,
        CompletionEvent, CompletionEventStream, CompletionRequest, ErasedStructuredTurnEvent,
        ErasedStructuredTurnEventStream, ErasedTextTurnEvent, ErasedTextTurnEventStream,
        FinishReason, LlmAdapter, StreamKind,
    },
    transcript::{ItemView, ToolCallItemView, ToolResultItemView, TurnRole, TurnView},
};

pub trait ReasoningEffortResolver: Send + Sync + 'static {
    fn resolve(&self, extensions: &RequestExtensions) -> Option<OpenAiReasoningEffort>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenAiReasoningEffort {
    Low,
    Medium,
    High,
}

impl OpenAiReasoningEffort {
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

#[derive(Clone)]
pub struct OpenAiAdapter {
    client: Arc<Client>,
    api_key: Arc<str>,
    base_url: Arc<str>,
    reasoning_resolver: Option<Arc<dyn ReasoningEffortResolver>>,
}

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static>>;

impl OpenAiAdapter {
    pub fn from_env() -> Result<Self, OpenAiError> {
        let api_key = env::var("OPENAI_API_KEY").map_err(OpenAiError::MissingApiKey)?;
        Ok(Self::new(api_key))
    }

    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Arc::new(Client::new()),
            api_key: Arc::from(api_key.into()),
            base_url: Arc::from("https://api.openai.com/v1"),
            reasoning_resolver: None,
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Arc::from(base_url.into());
        self
    }

    pub fn with_reasoning_resolver(mut self, r: impl ReasoningEffortResolver + 'static) -> Self {
        self.reasoning_resolver = Some(Arc::new(r));
        self
    }

    fn request_headers(&self) -> Result<HeaderMap, OpenAiError> {
        let mut headers = HeaderMap::new();
        let bearer = format!("Bearer {}", self.api_key);
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&bearer).map_err(OpenAiError::InvalidHeader)?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(headers)
    }

    async fn send_streaming_json(
        &self,
        path: &str,
        body: Value,
    ) -> Result<ByteStream, OpenAiError> {
        let response = self
            .client
            .post(format!("{}{}", self.base_url, path))
            .headers(self.request_headers()?)
            .json(&body)
            .send()
            .await?;
        let response = error_for_status_with_body(response).await?;
        Ok(Box::pin(response.bytes_stream()))
    }

    async fn get_json(&self, path: &str) -> Result<Value, OpenAiError> {
        let response = self
            .client
            .get(format!("{}{}", self.base_url, path))
            .headers(self.request_headers()?)
            .send()
            .await?;
        let value = error_for_status_with_body(response)
            .await?
            .json::<Value>()
            .await?;
        Ok(value)
    }
}

#[derive(Debug, Error)]
pub enum OpenAiError {
    #[error("OPENAI_API_KEY is not set: {0}")]
    MissingApiKey(#[source] env::VarError),
    #[error("invalid HTTP header: {0}")]
    InvalidHeader(#[source] reqwest::header::InvalidHeaderValue),
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("request failed with status {status}: {message}")]
    HttpStatus {
        status: reqwest::StatusCode,
        message: String,
    },
    #[error("failed to encode or decode JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("failed to parse structured output: {0}")]
    StructuredOutput(serde_json::Error),
    #[error("unexpected SSE payload: {message}")]
    Sse { message: String },
}

#[async_trait::async_trait]
impl LlmAdapter for OpenAiAdapter {
    async fn responses_text(
        &self,
        input: ModelInput,
        turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        let model = turn.config.model.to_string();
        let reasoning_effort = self
            .reasoning_resolver
            .as_ref()
            .and_then(|resolver| resolver.resolve(turn.extensions.as_ref()));
        let body = build_responses_request(&input, &turn.config, reasoning_effort, None)
            .map_err(AgentError::backend)?;
        let stream = self
            .send_streaming_json("/responses", body)
            .await
            .map_err(AgentError::backend)?;
        Ok(
            Box::pin(map_text_stream(stream, model).map(|item| item.map_err(AgentError::backend)))
                as ErasedTextTurnEventStream,
        )
    }

    async fn responses_structured(
        &self,
        input: ModelInput,
        turn: AdapterStructuredTurn,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
        let model = turn.config.model.to_string();
        let reasoning_effort = self
            .reasoning_resolver
            .as_ref()
            .and_then(|resolver| resolver.resolve(turn.extensions.as_ref()));
        let output_schema = Some(json!({
            "type": "json_schema",
            "name": turn.output.schema_name,
            "strict": true,
            "schema": turn.output.schema,
        }));
        let body = build_responses_request(&input, &turn.config, reasoning_effort, output_schema)
            .map_err(AgentError::backend)?;
        let stream = self
            .send_streaming_json("/responses", body)
            .await
            .map_err(AgentError::backend)?;
        Ok(Box::pin(
            map_structured_stream(stream, model).map(|item| item.map_err(AgentError::backend)),
        ) as ErasedStructuredTurnEventStream)
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionEventStream, AgentError> {
        let body = build_completion_request(&request);
        let stream = self
            .send_streaming_json("/completions", body)
            .await
            .map_err(AgentError::backend)?;
        Ok(Box::pin(
            map_completion_stream(stream, request.model.to_string())
                .map(|item| item.map_err(AgentError::backend)),
        ) as CompletionEventStream)
    }

    async fn recover_usage(
        &self,
        kind: StreamKind,
        request_id: &str,
    ) -> Result<Option<Usage>, AgentError> {
        match kind {
            StreamKind::ResponsesText | StreamKind::ResponsesStructured => {
                let value = self
                    .get_json(&format!("/responses/{request_id}"))
                    .await
                    .map_err(AgentError::backend)?;
                Ok(Some(parse_response_usage(&value)))
            }
            StreamKind::Completion => Ok(None),
        }
    }
}

fn build_responses_request(
    input: &ModelInput,
    config: &AdapterTurnConfig,
    reasoning_effort: Option<OpenAiReasoningEffort>,
    text_format: Option<Value>,
) -> Result<Value, OpenAiError> {
    let tools = build_tool_definitions(config);
    let parallel_tool_calls = match &config.tool_choice {
        AdapterToolChoice::Required | AdapterToolChoice::Specific(_) => false,
        AdapterToolChoice::None | AdapterToolChoice::Auto => true,
    };
    let mut body = json!({
        "model": config.model,
        "input": convert_model_input(input)?,
        "stream": true,
        "tools": tools,
        "parallel_tool_calls": parallel_tool_calls,
    });

    if let Some(temperature) = config.generation.temperature {
        body["temperature"] = json!(temperature.get());
    }
    if let Some(max_output_tokens) = config.generation.max_output_tokens {
        body["max_output_tokens"] = json!(max_output_tokens);
    }
    if let Some(reasoning_effort) = reasoning_effort {
        body["reasoning"] = json!({
            "effort": reasoning_effort.as_str(),
        });
    }
    if let Some(text_format) = text_format {
        body["text"] = json!({ "format": text_format });
    }
    match &config.tool_choice {
        AdapterToolChoice::Required => {
            body["tool_choice"] = json!("required");
        }
        AdapterToolChoice::None => {
            body["tool_choice"] = json!("none");
        }
        AdapterToolChoice::Specific(name) => {
            body["tool_choice"] = json!({
                "type": "function",
                "function": {
                    "name": name,
                },
            });
        }
        AdapterToolChoice::Auto => {}
    }

    Ok(body)
}

fn build_completion_request(request: &CompletionRequest) -> Value {
    let mut body = json!({
        "model": request.model,
        "prompt": request.prompt,
        "stream": true,
    });
    if let Some(temperature) = request.options.temperature {
        body["temperature"] = json!(temperature.get());
    }
    if let Some(max_output_tokens) = request.options.max_output_tokens {
        body["max_tokens"] = json!(max_output_tokens);
    }
    if !request.options.stop.is_empty() {
        body["stop"] = json!(request.options.stop);
    }
    body
}

async fn error_for_status_with_body(
    response: reqwest::Response,
) -> Result<reqwest::Response, OpenAiError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let body = response.text().await?;
    let message = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or(body);

    Err(OpenAiError::HttpStatus { status, message })
}

fn convert_model_input(input: &ModelInput) -> Result<Vec<Value>, OpenAiError> {
    let mut items = Vec::new();
    let mut assistant_message_content = Vec::new();
    let mut replayed_tool_call_ids = BTreeSet::new();

    for item in input.items() {
        match item {
            ModelInputItem::Message { role, content } => {
                flush_assistant_message(&mut assistant_message_content, &mut items);
                items.push(json!({
                    "type": "message",
                    "role": message_role(role),
                    "content": content
                        .iter()
                        .map(message_content)
                        .collect::<Vec<_>>(),
                }));
            }
            ModelInputItem::Assistant(assistant) => {
                lower_assistant_input_item(assistant, &mut assistant_message_content, &mut items)?;
            }
            ModelInputItem::ToolUse(tool_use) => {
                flush_assistant_message(&mut assistant_message_content, &mut items);
                lower_tool_use(
                    tool_use,
                    !replayed_tool_call_ids.contains(&tool_use.id),
                    &mut items,
                )?;
            }
            ModelInputItem::Turn(committed_turn) => {
                flush_assistant_message(&mut assistant_message_content, &mut items);
                if let Some(openai_turn) = committed_turn
                    .as_ref()
                    .as_any()
                    .downcast_ref::<OpenAiCommittedTurn>()
                {
                    emit_openai_turn_exact(openai_turn, &mut items, &mut replayed_tool_call_ids)?;
                } else {
                    emit_turn_from_view(
                        committed_turn.as_ref(),
                        &mut items,
                        &mut replayed_tool_call_ids,
                    )?;
                }
            }
        }
    }
    flush_assistant_message(&mut assistant_message_content, &mut items);
    Ok(items)
}

fn lower_assistant_input_item(
    item: &AssistantInputItem,
    message_content: &mut Vec<Value>,
    out: &mut Vec<Value>,
) -> Result<(), OpenAiError> {
    match item {
        AssistantInputItem::Text(text) => {
            message_content.push(json!({ "type": "output_text", "text": text }));
        }
        AssistantInputItem::Refusal(text) => {
            message_content.push(json!({ "type": "refusal", "text": text }));
        }
        AssistantInputItem::Reasoning(text) => {
            flush_assistant_message(message_content, out);
            out.push(json!({
                "type": "reasoning",
                "summary": [{ "type": "summary_text", "text": text }],
            }));
        }
    }
    Ok(())
}

fn lower_tool_use(
    tool_use: &ToolUse,
    emit_call: bool,
    out: &mut Vec<Value>,
) -> Result<(), OpenAiError> {
    if emit_call {
        out.push(json!({
            "type": "function_call",
            "call_id": tool_use.id.as_str(),
            "name": tool_use.name.as_str(),
            "arguments": serde_json::from_str::<Value>(tool_use.arguments.get())?,
        }));
    }
    out.push(json!({
        "type": "function_call_output",
        "call_id": tool_use.id.as_str(),
        "output": tool_use.result.get(),
    }));
    Ok(())
}

fn emit_openai_turn_exact(
    turn: &OpenAiCommittedTurn,
    out: &mut Vec<Value>,
    replayed_tool_call_ids: &mut BTreeSet<ToolCallId>,
) -> Result<(), OpenAiError> {
    let mut assistant_message_content = Vec::new();
    for item in &turn.items {
        match item {
            OpenAiTurnItem::Text { content } => {
                assistant_message_content.push(json!({ "type": "output_text", "text": content }));
            }
            OpenAiTurnItem::Reasoning { content } => {
                flush_assistant_message(&mut assistant_message_content, out);
                out.push(json!({
                    "type": "reasoning",
                    "summary": [{ "type": "summary_text", "text": content }],
                }));
            }
            OpenAiTurnItem::Refusal { content } => {
                assistant_message_content.push(json!({ "type": "refusal", "text": content }));
            }
            OpenAiTurnItem::ToolCall {
                id,
                name,
                arguments,
            } => {
                replayed_tool_call_ids.insert(id.clone());
                flush_assistant_message(&mut assistant_message_content, out);
                out.push(json!({
                    "type": "function_call",
                    "call_id": id.as_str(),
                    "name": name.as_str(),
                    "arguments": serde_json::from_str::<Value>(arguments.get())?,
                }));
            }
        }
    }
    flush_assistant_message(&mut assistant_message_content, out);
    Ok(())
}

fn emit_turn_from_view(
    turn: &dyn TurnView,
    out: &mut Vec<Value>,
    replayed_tool_call_ids: &mut BTreeSet<ToolCallId>,
) -> Result<(), OpenAiError> {
    match turn.role() {
        TurnRole::Assistant => emit_assistant_turn_from_view(turn, out, replayed_tool_call_ids),
        role => emit_message_turn_from_view(role, turn, out, replayed_tool_call_ids),
    }
}

fn emit_assistant_turn_from_view(
    turn: &dyn TurnView,
    out: &mut Vec<Value>,
    replayed_tool_call_ids: &mut BTreeSet<ToolCallId>,
) -> Result<(), OpenAiError> {
    let mut assistant_message_content = Vec::new();
    for index in 0..turn.item_count() {
        let Some(item) = turn.item_at(index) else {
            continue;
        };
        if let Some(text) = item.as_text() {
            assistant_message_content.push(json!({ "type": "output_text", "text": text }));
            continue;
        }
        if let Some(text) = item.as_reasoning() {
            flush_assistant_message(&mut assistant_message_content, out);
            out.push(json!({
                "type": "reasoning",
                "summary": [{ "type": "summary_text", "text": text }],
            }));
            continue;
        }
        if let Some(text) = item.as_refusal() {
            assistant_message_content.push(json!({ "type": "refusal", "text": text }));
            continue;
        }
        if let Some(tool_call) = item.as_tool_call() {
            replayed_tool_call_ids.insert(tool_call.id.clone());
            flush_assistant_message(&mut assistant_message_content, out);
            out.push(json!({
                "type": "function_call",
                "call_id": tool_call.id.as_str(),
                "name": tool_call.name.as_str(),
                "arguments": serde_json::from_str::<Value>(tool_call.arguments.get())?,
            }));
        }
    }
    flush_assistant_message(&mut assistant_message_content, out);
    Ok(())
}

fn emit_message_turn_from_view(
    role: TurnRole,
    turn: &dyn TurnView,
    out: &mut Vec<Value>,
    replayed_tool_call_ids: &mut BTreeSet<ToolCallId>,
) -> Result<(), OpenAiError> {
    let mut message_content = Vec::new();
    for index in 0..turn.item_count() {
        let Some(item) = turn.item_at(index) else {
            continue;
        };
        if let Some(text) = item.as_text() {
            message_content.push(json!({ "type": "input_text", "text": text }));
            continue;
        }
        if let Some(text) = item.as_reasoning().or_else(|| item.as_refusal()) {
            message_content.push(json!({ "type": "input_text", "text": text }));
            continue;
        }
        if let Some(tool_call) = item.as_tool_call() {
            replayed_tool_call_ids.insert(tool_call.id.clone());
            flush_message(turn_role(role), &mut message_content, out);
            out.push(json!({
                "type": "function_call",
                "call_id": tool_call.id.as_str(),
                "name": tool_call.name.as_str(),
                "arguments": serde_json::from_str::<Value>(tool_call.arguments.get())?,
            }));
        }
        if let Some(tool_result) = item.as_tool_result() {
            replayed_tool_call_ids.insert(tool_result.id.clone());
            flush_message(turn_role(role), &mut message_content, out);
            out.push(json!({
                "type": "function_call",
                "call_id": tool_result.id.as_str(),
                "name": tool_result.name.as_str(),
                "arguments": serde_json::from_str::<Value>(tool_result.arguments.get())?,
            }));
            out.push(json!({
                "type": "function_call_output",
                "call_id": tool_result.id.as_str(),
                "output": tool_result.result.get(),
            }));
        }
    }
    flush_message(turn_role(role), &mut message_content, out);
    Ok(())
}

fn flush_assistant_message(message_content: &mut Vec<Value>, out: &mut Vec<Value>) {
    flush_message("assistant", message_content, out);
}

fn flush_message(role: &str, message_content: &mut Vec<Value>, out: &mut Vec<Value>) {
    if message_content.is_empty() {
        return;
    }
    out.push(json!({
        "type": "message",
        "role": role,
        "content": std::mem::take(message_content),
    }));
}

fn message_role(role: &InputMessageRole) -> &'static str {
    match role {
        InputMessageRole::System => "system",
        InputMessageRole::Developer => "developer",
        InputMessageRole::User => "user",
    }
}

fn turn_role(role: TurnRole) -> &'static str {
    match role {
        TurnRole::System => "system",
        TurnRole::Developer => "developer",
        TurnRole::User => "user",
        TurnRole::Assistant => "assistant",
    }
}

fn message_content(content: &MessageContent) -> Value {
    match content {
        MessageContent::Text(text) => json!({
            "type": "input_text",
            "text": text,
        }),
    }
}

fn build_tool_definitions(config: &AdapterTurnConfig) -> Vec<Value> {
    config
        .tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            })
        })
        .collect()
}

#[derive(Default)]
struct ToolCallTracker {
    buffers: BTreeMap<String, ToolCallBuffer>,
    finalized: BTreeMap<String, FinalizedToolCall>,
}

struct ToolCallBuffer {
    id: ToolCallId,
    name: ToolName,
    arguments_json: String,
}

struct FinalizedToolCall {
    id: ToolCallId,
    name: ToolName,
    arguments_json: String,
}

impl ToolCallTracker {
    fn record_delta(
        &mut self,
        key: String,
        id: ToolCallId,
        name: ToolName,
        delta: &str,
    ) -> Option<(ToolCallId, ToolName, String)> {
        if delta.is_empty() {
            return None;
        }

        let entry = self.buffers.entry(key).or_insert_with(|| ToolCallBuffer {
            id,
            name,
            arguments_json: String::new(),
        });
        entry.arguments_json.push_str(delta);
        Some((entry.id.clone(), entry.name.clone(), delta.to_string()))
    }

    fn finish(
        &mut self,
        key: String,
        id: ToolCallId,
        name: ToolName,
        explicit_arguments_json: Option<String>,
    ) -> Result<Option<ToolMetadata>, OpenAiError> {
        let arguments_json = explicit_arguments_json
            .or_else(|| {
                self.buffers
                    .remove(&key)
                    .map(|buffer| buffer.arguments_json)
            })
            .unwrap_or_else(|| "{}".to_string());

        if let Some(existing) = self.finalized.get(&key) {
            if existing.id == id
                && existing.name == name
                && existing.arguments_json == arguments_json
            {
                return Ok(None);
            }
            return Err(OpenAiError::Sse {
                message: format!(
                    "conflicting duplicate tool call completion for `{}`",
                    id.as_str()
                ),
            });
        }

        let arguments = RawJson::parse(arguments_json.clone())?;
        let metadata = ToolMetadata::new(id.clone(), name.clone(), arguments);
        self.finalized.insert(
            key,
            FinalizedToolCall {
                id,
                name,
                arguments_json,
            },
        );
        Ok(Some(metadata))
    }
}

fn response_tool_key(event: &Value) -> String {
    event["call_id"]
        .as_str()
        .or_else(|| event["item_id"].as_str())
        .unwrap_or_default()
        .to_string()
}

fn response_tool_id(event: &Value) -> ToolCallId {
    ToolCallId::from(
        event["call_id"]
            .as_str()
            .or_else(|| event["item_id"].as_str())
            .unwrap_or_default(),
    )
}

fn response_tool_name(event: &Value) -> ToolName {
    ToolName::from(event["name"].as_str().unwrap_or_default())
}

fn output_item_tool_key(event: &Value) -> String {
    event
        .pointer("/item/call_id")
        .and_then(Value::as_str)
        .or_else(|| event.pointer("/item/id").and_then(Value::as_str))
        .unwrap_or_default()
        .to_string()
}

fn output_item_tool_id(event: &Value) -> ToolCallId {
    ToolCallId::from(
        event
            .pointer("/item/call_id")
            .and_then(Value::as_str)
            .or_else(|| event.pointer("/item/id").and_then(Value::as_str))
            .unwrap_or_default(),
    )
}

fn output_item_tool_name(event: &Value) -> ToolName {
    ToolName::from(
        event
            .pointer("/item/name")
            .and_then(Value::as_str)
            .unwrap_or_default(),
    )
}

fn json_string(value: &Value) -> Result<String, OpenAiError> {
    match value {
        Value::String(value) => Ok(value.clone()),
        other => Ok(serde_json::to_string(other)?),
    }
}

fn maybe_parse_structured_output(
    buffer: &str,
    emitted_ready: &mut bool,
) -> Result<Option<RawJson>, OpenAiError> {
    if *emitted_ready || buffer.is_empty() {
        return Ok(None);
    }

    *emitted_ready = true;
    RawJson::parse(buffer.to_string())
        .map(Some)
        .map_err(OpenAiError::StructuredOutput)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BufferedTurnItemKind {
    Text,
    Reasoning,
    Refusal,
}

#[derive(Debug)]
struct BufferedTurnItem {
    kind: BufferedTurnItemKind,
    item_id: String,
    content: String,
}

fn push_buffered_content(
    pending_item: &mut Option<BufferedTurnItem>,
    committed_items: &mut Vec<OpenAiTurnItem>,
    kind: BufferedTurnItemKind,
    item_id: &str,
    delta: &str,
) {
    if delta.is_empty() {
        return;
    }

    match pending_item {
        Some(buffer) if buffer.kind == kind && buffer.item_id == item_id => {
            buffer.content.push_str(delta);
        }
        Some(_) => {
            flush_buffered_content(pending_item, committed_items);
            *pending_item = Some(BufferedTurnItem {
                kind,
                item_id: item_id.to_string(),
                content: delta.to_string(),
            });
        }
        None => {
            *pending_item = Some(BufferedTurnItem {
                kind,
                item_id: item_id.to_string(),
                content: delta.to_string(),
            });
        }
    }
}

fn replace_buffered_text(
    pending_item: &mut Option<BufferedTurnItem>,
    committed_items: &mut Vec<OpenAiTurnItem>,
    item_id: &str,
    content: &str,
) {
    if content.is_empty() {
        return;
    }

    match pending_item {
        Some(buffer) if buffer.kind == BufferedTurnItemKind::Text && buffer.item_id == item_id => {
            buffer.content.clear();
            buffer.content.push_str(content);
        }
        Some(_) => {
            flush_buffered_content(pending_item, committed_items);
            *pending_item = Some(BufferedTurnItem {
                kind: BufferedTurnItemKind::Text,
                item_id: item_id.to_string(),
                content: content.to_string(),
            });
        }
        None => {
            *pending_item = Some(BufferedTurnItem {
                kind: BufferedTurnItemKind::Text,
                item_id: item_id.to_string(),
                content: content.to_string(),
            });
        }
    }
}

fn flush_buffered_content(
    pending_item: &mut Option<BufferedTurnItem>,
    committed_items: &mut Vec<OpenAiTurnItem>,
) {
    let Some(buffer) = pending_item.take() else {
        return;
    };

    let item = match buffer.kind {
        BufferedTurnItemKind::Text => OpenAiTurnItem::Text {
            content: buffer.content,
        },
        BufferedTurnItemKind::Reasoning => OpenAiTurnItem::Reasoning {
            content: buffer.content,
        },
        BufferedTurnItemKind::Refusal => OpenAiTurnItem::Refusal {
            content: buffer.content,
        },
    };
    committed_items.push(item);
}

fn push_committed_tool_call(committed_items: &mut Vec<OpenAiTurnItem>, metadata: &ToolMetadata) {
    committed_items.push(OpenAiTurnItem::ToolCall {
        id: metadata.id.clone(),
        name: metadata.name.clone(),
        arguments: metadata.arguments.clone(),
    });
}

fn map_text_stream<S>(
    stream: S,
    fallback_model: String,
) -> impl Stream<Item = Result<ErasedTextTurnEvent, OpenAiError>> + Send + 'static
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    try_stream! {
        let mut parser = SseParser::default();
        let mut started = false;
        let mut request_id = None::<String>;
        let mut model = fallback_model;
        let mut saw_tool_call = false;
        let mut saw_refusal = false;
        let mut tool_calls = ToolCallTracker::default();
        let mut pending_item = None::<BufferedTurnItem>;
        let mut committed_items = Vec::<OpenAiTurnItem>::new();
        futures::pin_mut!(stream);

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            for payload in parser.push(&chunk)? {
                if payload == "[DONE]" {
                    break;
                }
                let event = serde_json::from_str::<Value>(&payload)?;
                let event_type = event["type"].as_str().unwrap_or_default();
                if event_type == "response.created" {
                    if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                        request_id = Some(id.to_string());
                    }
                    if let Some(event_model) = event.pointer("/response/model").and_then(Value::as_str) {
                        model = event_model.to_string();
                    }
                    if !started {
                        started = true;
                        yield ErasedTextTurnEvent::Started {
                            request_id: request_id.clone(),
                            model: model.clone(),
                        };
                    }
                    continue;
                }

                if !started {
                    started = true;
                    yield ErasedTextTurnEvent::Started {
                        request_id: request_id.clone(),
                        model: model.clone(),
                    };
                }

                match event_type {
                    "response.output_text.delta" => {
                        if let Some(delta) = event["delta"].as_str() {
                            push_buffered_content(
                                &mut pending_item,
                                &mut committed_items,
                                BufferedTurnItemKind::Text,
                                event["item_id"].as_str().unwrap_or(""),
                                delta,
                            );
                            yield ErasedTextTurnEvent::TextDelta {
                                delta: delta.to_string(),
                            };
                        }
                    }
                    "response.reasoning_summary_text.delta" | "response.reasoning.delta" => {
                        if let Some(delta) = event["delta"].as_str() {
                            push_buffered_content(
                                &mut pending_item,
                                &mut committed_items,
                                BufferedTurnItemKind::Reasoning,
                                event["item_id"].as_str().unwrap_or(""),
                                delta,
                            );
                            yield ErasedTextTurnEvent::ReasoningDelta {
                                delta: delta.to_string(),
                            };
                        }
                    }
                    "response.refusal.delta" => {
                        saw_refusal = true;
                        if let Some(delta) = event["delta"].as_str() {
                            push_buffered_content(
                                &mut pending_item,
                                &mut committed_items,
                                BufferedTurnItemKind::Refusal,
                                event["item_id"].as_str().unwrap_or(""),
                                delta,
                            );
                            yield ErasedTextTurnEvent::RefusalDelta {
                                delta: delta.to_string(),
                            };
                        }
                    }
                    "response.output_text.done" => {
                        if let Some(text) = event["text"].as_str() {
                            replace_buffered_text(
                                &mut pending_item,
                                &mut committed_items,
                                event["item_id"].as_str().unwrap_or(""),
                                text,
                            );
                        }
                    }
                    "response.function_call_arguments.delta" => {
                        if let Some(delta) = event["delta"].as_str()
                            && let Some((id, name, delta)) = tool_calls.record_delta(
                                response_tool_key(&event),
                                response_tool_id(&event),
                            response_tool_name(&event),
                            delta,
                        ) {
                                yield ErasedTextTurnEvent::ToolCallChunk {
                                    id,
                                    name,
                                    arguments_json_delta: delta,
                                };
                            }
                    }
                    "response.function_call_arguments.done" => {
                        saw_tool_call = true;
                        if let Some(invocation) = tool_calls.finish(
                            response_tool_key(&event),
                            response_tool_id(&event),
                            response_tool_name(&event),
                            event.get("arguments").map(json_string).transpose()?,
                        )? {
                            flush_buffered_content(&mut pending_item, &mut committed_items);
                            push_committed_tool_call(&mut committed_items, &invocation);
                            yield ErasedTextTurnEvent::ToolCallReady(invocation);
                        }
                    }
                    "response.output_item.done" => {
                        if event.pointer("/item/type").and_then(Value::as_str) == Some("function_call") {
                            saw_tool_call = true;
                            if let Some(invocation) = tool_calls.finish(
                                output_item_tool_key(&event),
                                output_item_tool_id(&event),
                                output_item_tool_name(&event),
                                event.pointer("/item/arguments").map(json_string).transpose()?,
                            )? {
                                flush_buffered_content(&mut pending_item, &mut committed_items);
                                push_committed_tool_call(&mut committed_items, &invocation);
                                yield ErasedTextTurnEvent::ToolCallReady(invocation);
                            }
                        }
                    }
                    "response.completed" => {
                        if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                            request_id = Some(id.to_string());
                        }
                        flush_buffered_content(&mut pending_item, &mut committed_items);
                        let finish_reason = event
                            .pointer("/response/stop_reason")
                            .and_then(Value::as_str)
                            .or_else(|| event.pointer("/response/finish_reason").and_then(Value::as_str))
                            .map(map_responses_finish_reason)
                            .unwrap_or_else(|| {
                                if saw_tool_call {
                                    FinishReason::ToolCall
                                } else if saw_refusal {
                                    FinishReason::ContentFilter
                                } else {
                                    FinishReason::Stop
                                }
                            });
                        let usage = parse_response_usage(&event["response"]);
                        yield ErasedTextTurnEvent::Completed {
                            request_id: request_id.clone(),
                            finish_reason: finish_reason.clone(),
                            usage,
                            committed_turn: Arc::new(OpenAiCommittedTurn {
                                request_id: request_id.clone(),
                                model: model.clone(),
                                items: committed_items.clone(),
                                finish_reason,
                                usage,
                            }),
                        };
                    }
                    _ => {}
                }
            }
        }
    }
}

fn map_structured_stream<S>(
    stream: S,
    fallback_model: String,
) -> impl Stream<Item = Result<ErasedStructuredTurnEvent, OpenAiError>> + Send + 'static
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    try_stream! {
        let mut parser = SseParser::default();
        let mut started = false;
        let mut request_id = None::<String>;
        let mut model = fallback_model;
        let mut saw_tool_call = false;
        let mut saw_refusal = false;
        let mut tool_calls = ToolCallTracker::default();
        let mut structured_buffer = String::new();
        let mut emitted_ready = false;
        let mut pending_item = None::<BufferedTurnItem>;
        let mut committed_items = Vec::<OpenAiTurnItem>::new();
        futures::pin_mut!(stream);

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            for payload in parser.push(&chunk)? {
                if payload == "[DONE]" {
                    break;
                }
                let event = serde_json::from_str::<Value>(&payload)?;
                let event_type = event["type"].as_str().unwrap_or_default();
                if event_type == "response.created" {
                    if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                        request_id = Some(id.to_string());
                    }
                    if let Some(event_model) = event.pointer("/response/model").and_then(Value::as_str) {
                        model = event_model.to_string();
                    }
                    if !started {
                        started = true;
                        yield ErasedStructuredTurnEvent::Started {
                            request_id: request_id.clone(),
                            model: model.clone(),
                        };
                    }
                    continue;
                }

                if !started {
                    started = true;
                    yield ErasedStructuredTurnEvent::Started {
                        request_id: request_id.clone(),
                        model: model.clone(),
                    };
                }

                match event_type {
                    "response.output_text.delta" => {
                        if let Some(delta) = event["delta"].as_str() {
                            structured_buffer.push_str(delta);
                            push_buffered_content(
                                &mut pending_item,
                                &mut committed_items,
                                BufferedTurnItemKind::Text,
                                event["item_id"].as_str().unwrap_or(""),
                                delta,
                            );
                            yield ErasedStructuredTurnEvent::StructuredOutputChunk {
                                json_delta: delta.to_string(),
                            };
                        }
                    }
                    "response.output_text.done" => {
                        if let Some(text) = event["text"].as_str() {
                            structured_buffer.clear();
                            structured_buffer.push_str(text);
                            replace_buffered_text(
                                &mut pending_item,
                                &mut committed_items,
                                event["item_id"].as_str().unwrap_or(""),
                                text,
                            );
                        }
                        if let Some(value) =
                            maybe_parse_structured_output(&structured_buffer, &mut emitted_ready)?
                        {
                            yield ErasedStructuredTurnEvent::StructuredOutputReady(value);
                        }
                    }
                    "response.reasoning_summary_text.delta" | "response.reasoning.delta" => {
                        if let Some(delta) = event["delta"].as_str() {
                            push_buffered_content(
                                &mut pending_item,
                                &mut committed_items,
                                BufferedTurnItemKind::Reasoning,
                                event["item_id"].as_str().unwrap_or(""),
                                delta,
                            );
                            yield ErasedStructuredTurnEvent::ReasoningDelta {
                                delta: delta.to_string(),
                            };
                        }
                    }
                    "response.refusal.delta" => {
                        saw_refusal = true;
                        if let Some(delta) = event["delta"].as_str() {
                            push_buffered_content(
                                &mut pending_item,
                                &mut committed_items,
                                BufferedTurnItemKind::Refusal,
                                event["item_id"].as_str().unwrap_or(""),
                                delta,
                            );
                            yield ErasedStructuredTurnEvent::RefusalDelta {
                                delta: delta.to_string(),
                            };
                        }
                    }
                    "response.function_call_arguments.delta" => {
                        if let Some(delta) = event["delta"].as_str()
                            && let Some((id, name, delta)) = tool_calls.record_delta(
                                response_tool_key(&event),
                                response_tool_id(&event),
                            response_tool_name(&event),
                            delta,
                        ) {
                                yield ErasedStructuredTurnEvent::ToolCallChunk {
                                    id,
                                    name,
                                    arguments_json_delta: delta,
                                };
                            }
                    }
                    "response.function_call_arguments.done" => {
                        saw_tool_call = true;
                        if let Some(invocation) = tool_calls.finish(
                            response_tool_key(&event),
                            response_tool_id(&event),
                            response_tool_name(&event),
                            event.get("arguments").map(json_string).transpose()?,
                        )? {
                            flush_buffered_content(&mut pending_item, &mut committed_items);
                            push_committed_tool_call(&mut committed_items, &invocation);
                            yield ErasedStructuredTurnEvent::ToolCallReady(invocation);
                        }
                    }
                    "response.output_item.done" => {
                        if event.pointer("/item/type").and_then(Value::as_str) == Some("function_call") {
                            saw_tool_call = true;
                            if let Some(invocation) = tool_calls.finish(
                                output_item_tool_key(&event),
                                output_item_tool_id(&event),
                                output_item_tool_name(&event),
                                event.pointer("/item/arguments").map(json_string).transpose()?,
                            )? {
                                flush_buffered_content(&mut pending_item, &mut committed_items);
                                push_committed_tool_call(&mut committed_items, &invocation);
                                yield ErasedStructuredTurnEvent::ToolCallReady(invocation);
                            }
                        }
                    }
                    "response.completed" => {
                        if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                            request_id = Some(id.to_string());
                        }
                        if let Some(value) =
                            maybe_parse_structured_output(&structured_buffer, &mut emitted_ready)?
                        {
                            yield ErasedStructuredTurnEvent::StructuredOutputReady(value);
                        }
                        flush_buffered_content(&mut pending_item, &mut committed_items);
                        let finish_reason = event
                            .pointer("/response/stop_reason")
                            .and_then(Value::as_str)
                            .or_else(|| event.pointer("/response/finish_reason").and_then(Value::as_str))
                            .map(map_responses_finish_reason)
                            .unwrap_or_else(|| {
                                if saw_tool_call {
                                    FinishReason::ToolCall
                                } else if saw_refusal {
                                    FinishReason::ContentFilter
                                } else {
                                    FinishReason::Stop
                                }
                            });
                        let usage = parse_response_usage(&event["response"]);
                        yield ErasedStructuredTurnEvent::Completed {
                            request_id: request_id.clone(),
                            finish_reason: finish_reason.clone(),
                            usage,
                            committed_turn: Arc::new(OpenAiCommittedTurn {
                                request_id: request_id.clone(),
                                model: model.clone(),
                                items: committed_items.clone(),
                                finish_reason,
                                usage,
                            }),
                        };
                    }
                    _ => {}
                }
            }
        }
    }
}

fn map_completion_stream<S>(
    stream: S,
    fallback_model: String,
) -> impl Stream<Item = Result<CompletionEvent, OpenAiError>> + Send + 'static
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    try_stream! {
        let mut parser = SseParser::default();
        let mut started = false;
        let mut request_id = None::<String>;
        let mut model = fallback_model;
        let mut last_usage = Usage::zero();
        let mut finished = false;
        futures::pin_mut!(stream);

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            for payload in parser.push(&chunk)? {
                if payload == "[DONE]" {
                    if started && !finished {
                        yield CompletionEvent::Completed {
                            request_id: request_id.clone(),
                            finish_reason: FinishReason::Stop,
                            usage: last_usage,
                        };
                    }
                    finished = true;
                    break;
                }
                let event = serde_json::from_str::<Value>(&payload)?;
                if !started {
                    request_id = event["id"].as_str().map(ToOwned::to_owned);
                    if let Some(event_model) = event["model"].as_str() {
                        model = event_model.to_string();
                    }
                    started = true;
                    yield CompletionEvent::Started {
                        request_id: request_id.clone(),
                        model: model.clone(),
                    };
                }

                if let Some(delta) = event.pointer("/choices/0/text").and_then(Value::as_str)
                    && !delta.is_empty() {
                        yield CompletionEvent::TextDelta(delta.to_string());
                    }
                if let Some(usage) = event.get("usage") {
                    last_usage = parse_completion_usage(usage);
                }
                if let Some(finish_reason) = event.pointer("/choices/0/finish_reason").and_then(Value::as_str) {
                    finished = true;
                    yield CompletionEvent::Completed {
                        request_id: request_id.clone(),
                        finish_reason: map_completion_finish_reason(finish_reason),
                        usage: last_usage,
                    };
                }
            }
        }
    }
}

fn parse_response_usage(value: &Value) -> Usage {
    let usage = value.get("usage").unwrap_or(value);
    let input_tokens = usage["input_tokens"].as_u64().unwrap_or_default();
    let output_tokens = usage["output_tokens"].as_u64().unwrap_or_default();
    let total_tokens = usage["total_tokens"]
        .as_u64()
        .unwrap_or(input_tokens + output_tokens);
    Usage {
        input_tokens,
        output_tokens,
        total_tokens,
        cost_micros_usd: 0,
    }
}

fn map_responses_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCall,
        other => FinishReason::Unknown(other.to_string()),
    }
}

fn parse_completion_usage(value: &Value) -> Usage {
    let prompt_tokens = value["prompt_tokens"].as_u64().unwrap_or_default();
    let total_tokens = value["total_tokens"].as_u64().unwrap_or_default();
    Usage {
        input_tokens: prompt_tokens,
        output_tokens: total_tokens.saturating_sub(prompt_tokens),
        total_tokens,
        cost_micros_usd: 0,
    }
}

fn map_completion_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Unknown(other.to_string()),
    }
}

#[derive(Default)]
struct SseParser {
    buffer: Vec<u8>,
    data_lines: Vec<String>,
}

impl SseParser {
    fn push(&mut self, chunk: &[u8]) -> Result<Vec<String>, OpenAiError> {
        self.buffer.extend_from_slice(chunk);
        let mut frames = Vec::new();

        while let Some(pos) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line = self.buffer.drain(..=pos).collect::<Vec<_>>();
            if matches!(line.last(), Some(b'\n')) {
                line.pop();
            }
            if matches!(line.last(), Some(b'\r')) {
                line.pop();
            }
            if line.is_empty() {
                if !self.data_lines.is_empty() {
                    frames.push(self.data_lines.join("\n"));
                    self.data_lines.clear();
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix(b"data:") {
                let rest = if rest.first() == Some(&b' ') {
                    &rest[1..]
                } else {
                    rest
                };
                let text = String::from_utf8(rest.to_vec()).map_err(|err| OpenAiError::Sse {
                    message: err.to_string(),
                })?;
                self.data_lines.push(text);
            }
        }

        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::{StreamExt, executor::block_on};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    use agents_protocol::{
        AdapterToolChoice, AdapterToolDefinition, AdapterTurnConfig, AssistantInputItem,
        AssistantTurnItem, AssistantTurnView, ErasedStructuredTurnEvent, ErasedTextTurnEvent,
        GenerationParams, InputMessageRole, ModelInput, ModelInputItem, ModelName, ToolUse,
    };

    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct WeatherArgs {
        city: String,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct Summary {
        answer: String,
    }

    #[test]
    fn convert_model_input_preserves_ordered_system_developer_and_assistant_items() {
        let input = ModelInput::from_items(vec![
            ModelInputItem::text(InputMessageRole::System, "policy"),
            ModelInputItem::text(InputMessageRole::Developer, "dev"),
            ModelInputItem::text(InputMessageRole::User, "hello"),
            ModelInputItem::Assistant(AssistantInputItem::Reasoning("think".into())),
            ModelInputItem::Assistant(AssistantInputItem::Text("hi".into())),
            ModelInputItem::ToolUse(ToolUse::new(
                "call-1",
                "weather",
                RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                RawJson::parse("\"sunny\"").unwrap(),
            )),
        ]);

        let items = convert_model_input(&input).unwrap();
        assert_eq!(items[0]["role"], "system");
        assert_eq!(items[1]["role"], "developer");
        assert_eq!(items[2]["role"], "user");
        assert_eq!(items[3]["type"], "reasoning");
        assert_eq!(items[4]["type"], "message");
        assert_eq!(items[5]["type"], "function_call");
        assert_eq!(items[6]["type"], "function_call_output");
    }

    #[test]
    fn convert_model_input_replays_openai_turn_items_from_turn_variant() {
        let input = ModelInput::from_items(vec![
            ModelInputItem::text(InputMessageRole::User, "hello"),
            ModelInputItem::Turn(Arc::new(OpenAiCommittedTurn {
                request_id: Some("resp_1".into()),
                model: "gpt-4.1".into(),
                items: vec![
                    OpenAiTurnItem::Reasoning {
                        content: "think".into(),
                    },
                    OpenAiTurnItem::Text {
                        content: "hi".into(),
                    },
                    OpenAiTurnItem::Refusal {
                        content: "nope".into(),
                    },
                    OpenAiTurnItem::ToolCall {
                        id: "call-1".into(),
                        name: "weather".into(),
                        arguments: RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                    },
                ],
                finish_reason: FinishReason::ToolCall,
                usage: Usage::zero(),
            })),
            ModelInputItem::ToolUse(ToolUse::new(
                "call-1",
                "weather",
                RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                RawJson::parse("\"sunny\"").unwrap(),
            )),
        ]);

        let items = convert_model_input(&input).unwrap();
        assert_eq!(items[0]["role"], "user");
        assert_eq!(items[1]["type"], "reasoning");
        assert_eq!(items[2]["type"], "message");
        assert_eq!(items[2]["content"][0]["type"], "output_text");
        assert_eq!(items[2]["content"][1]["type"], "refusal");
        assert_eq!(items[3]["type"], "function_call");
        assert_eq!(items[4]["type"], "function_call_output");
    }

    #[test]
    fn convert_model_input_projects_non_openai_turns_via_turn_view() {
        let input = ModelInput::from_items(vec![ModelInputItem::Turn(Arc::new(
            AssistantTurnView::from_items(&[
                AssistantTurnItem::Reasoning("think".into()),
                AssistantTurnItem::Text("hi".into()),
                AssistantTurnItem::ToolCall {
                    id: "call-1".into(),
                    name: "weather".into(),
                    arguments: RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                },
            ]),
        ))]);

        let items = convert_model_input(&input).unwrap();
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[1]["type"], "message");
        assert_eq!(items[1]["content"][0]["type"], "output_text");
        assert_eq!(items[2]["type"], "function_call");
    }

    #[test]
    fn responses_sse_maps_reasoning_refusal_tool_and_completion() {
        let payloads = vec![
            Ok(Bytes::from(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4.1\"}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"thinking\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.refusal.delta\",\"delta\":\"no\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.function_call_arguments.done\",\"call_id\":\"call-1\",\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\\\"Tokyo\\\"}\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )),
        ];

        let events = block_on(async {
            map_text_stream(futures::stream::iter(payloads), "gpt-4.1".into())
                .collect::<Vec<_>>()
                .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        assert!(matches!(events[0], ErasedTextTurnEvent::Started { .. }));
        assert!(matches!(
            events[1],
            ErasedTextTurnEvent::ReasoningDelta { .. }
        ));
        assert!(matches!(
            events[2],
            ErasedTextTurnEvent::RefusalDelta { .. }
        ));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, ErasedTextTurnEvent::ToolCallReady(_)))
        );
        assert!(matches!(
            events.last(),
            Some(ErasedTextTurnEvent::Completed { .. })
        ));
    }

    #[test]
    fn structured_sse_emits_ready_when_json_completes() {
        let payloads = vec![
            Ok(Bytes::from(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_2\",\"model\":\"gpt-4.1\"}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"{\\\"answer\\\":\\\"42\\\"}\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_2\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )),
        ];

        let events = block_on(async {
            map_structured_stream(futures::stream::iter(payloads), "gpt-4.1".into())
                .collect::<Vec<_>>()
                .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        assert!(events.iter().any(|event| matches!(
            event,
            ErasedStructuredTurnEvent::StructuredOutputReady(summary)
                if summary.deserialize::<Summary>().unwrap().answer == "42"
        )));
    }

    #[test]
    fn disabled_tool_mode_sends_no_tool_definitions() {
        let tools = build_tool_definitions(&AdapterTurnConfig {
            model: ModelName::new("gpt-4.1").unwrap(),
            generation: GenerationParams::default(),
            tools: Vec::<AdapterToolDefinition>::new(),
            tool_choice: AdapterToolChoice::None,
        });
        assert!(tools.is_empty());
    }

    #[test]
    fn responses_sse_deduplicates_tool_call_completion_and_uses_consistent_key() {
        let payloads = vec![
            Ok(Bytes::from(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_tool\",\"model\":\"gpt-4.1\"}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"item-1\",\"call_id\":\"call-1\",\"name\":\"weather\",\"delta\":\"{\\\"city\\\":\\\"Tokyo\\\"}\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"item-1\",\"call_id\":\"call-1\",\"name\":\"weather\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"item-1\",\"call_id\":\"call-1\",\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\\\"Tokyo\\\"}\"}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_tool\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )),
        ];

        let events = block_on(async {
            map_text_stream(futures::stream::iter(payloads), "gpt-4.1".into())
                .collect::<Vec<_>>()
                .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        let ready = events
            .iter()
            .filter_map(|event| match event {
                ErasedTextTurnEvent::ToolCallReady(invocation) => Some(invocation),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].name.as_str(), "weather");
        assert_eq!(
            ready[0]
                .arguments
                .deserialize::<WeatherArgs>()
                .unwrap()
                .city,
            "Tokyo"
        );
    }

    #[test]
    fn responses_sse_keeps_separate_text_items_by_item_id() {
        let payloads = vec![
            Ok(Bytes::from(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_text_items\",\"model\":\"gpt-4.1\"}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"text_item_0\",\"delta\":\"hel\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.done\",\"item_id\":\"text_item_0\",\"text\":\"hello\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"text_item_1\",\"delta\":\"bye\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.done\",\"item_id\":\"text_item_1\",\"text\":\"bye!\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_text_items\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )),
        ];

        let events = block_on(async {
            map_text_stream(futures::stream::iter(payloads), "gpt-4.1".into())
                .collect::<Vec<_>>()
                .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        let committed_turn = match events.last() {
            Some(ErasedTextTurnEvent::Completed { committed_turn, .. }) => committed_turn,
            other => panic!("expected completed event, got {other:?}"),
        };
        let committed_turn = committed_turn
            .as_any()
            .downcast_ref::<OpenAiCommittedTurn>()
            .expect("OpenAI committed turn");

        assert!(matches!(
            committed_turn.items.as_slice(),
            [
                OpenAiTurnItem::Text { content: first },
                OpenAiTurnItem::Text { content: second }
            ] if first == "hello" && second == "bye!"
        ));
    }

    #[test]
    fn structured_sse_waits_until_completion_before_ready_for_scalar_json() {
        let payloads = vec![
            Ok(Bytes::from(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_scalar\",\"model\":\"gpt-4.1\"}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"1\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"2\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_scalar\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )),
        ];

        let events = block_on(async {
            map_structured_stream(futures::stream::iter(payloads), "gpt-4.1".into())
                .collect::<Vec<_>>()
                .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        assert!(matches!(
            events[1],
            ErasedStructuredTurnEvent::StructuredOutputChunk { .. }
        ));
        assert!(matches!(
            events[2],
            ErasedStructuredTurnEvent::StructuredOutputChunk { .. }
        ));
        match &events[3] {
            ErasedStructuredTurnEvent::StructuredOutputReady(value) => {
                assert_eq!(value.deserialize::<u64>().unwrap(), 12);
            }
            other => panic!("expected structured output ready, got {other:?}"),
        }
        assert!(matches!(
            events[4],
            ErasedStructuredTurnEvent::Completed { .. }
        ));
    }

    #[test]
    fn completion_sse_maps_text_and_done() {
        let payloads = vec![
            Ok(Bytes::from(
                "data: {\"id\":\"cmpl-1\",\"model\":\"gpt-3.5-turbo-instruct\",\"choices\":[{\"text\":\"hello\",\"finish_reason\":null}]}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"id\":\"cmpl-1\",\"model\":\"gpt-3.5-turbo-instruct\",\"choices\":[{\"text\":\"\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"total_tokens\":2}}\n\n",
            )),
        ];
        let events = block_on(async {
            map_completion_stream(
                futures::stream::iter(payloads),
                "gpt-3.5-turbo-instruct".into(),
            )
            .collect::<Vec<_>>()
            .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
        assert!(matches!(events[0], CompletionEvent::Started { .. }));
        assert!(matches!(events[1], CompletionEvent::TextDelta(_)));
        assert!(matches!(events[2], CompletionEvent::Completed { .. }));
    }
}

// ── OpenAiCommittedTurn ───────────────────────────────────────────────────────

/// A serializable, replayable representation of a completed OpenAI assistant turn.
///
/// This is the adapter-owned exact committed turn for the OpenAI provider.
/// It derives `Serialize` and `Deserialize` so it can be persisted and
/// restored without lossy normalization through a shared library IR.
///
/// At runtime, `Session` interacts with committed turns through the erased
/// `TurnView` trait; serde roundtrip stays centered on this concrete type.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OpenAiCommittedTurn {
    /// The `request_id` returned by the OpenAI API for this turn, if available.
    pub request_id: Option<String>,
    /// The model name that produced this turn.
    pub model: String,
    /// The ordered list of items produced by the assistant during this turn.
    pub items: Vec<OpenAiTurnItem>,
    /// The reason the turn ended.
    pub finish_reason: FinishReason,
    /// Token usage reported by the API for this turn.
    pub usage: Usage,
}

/// A single item within an [`OpenAiCommittedTurn`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAiTurnItem {
    /// Plain assistant text output.
    Text { content: String },
    /// Reasoning / chain-of-thought text (o-series models).
    Reasoning { content: String },
    /// Refusal text emitted by the model.
    Refusal { content: String },
    /// A tool call requested by the assistant.
    ToolCall {
        id: ToolCallId,
        name: ToolName,
        arguments: RawJson,
    },
}

// ── TurnView impl for OpenAiCommittedTurn ─────────────────────────────────────

impl TurnView for OpenAiCommittedTurn {
    fn role(&self) -> TurnRole {
        TurnRole::Assistant
    }

    fn item_count(&self) -> usize {
        self.items.len()
    }

    fn item_at(&self, index: usize) -> Option<&dyn ItemView> {
        self.items.get(index).map(|item| item as &dyn ItemView)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ItemView for OpenAiTurnItem {
    fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { content } => Some(content),
            _ => None,
        }
    }

    fn as_reasoning(&self) -> Option<&str> {
        match self {
            Self::Reasoning { content } => Some(content),
            _ => None,
        }
    }

    fn as_refusal(&self) -> Option<&str> {
        match self {
            Self::Refusal { content } => Some(content),
            _ => None,
        }
    }

    fn as_tool_call(&self) -> Option<ToolCallItemView<'_>> {
        match self {
            Self::ToolCall {
                id,
                name,
                arguments,
            } => Some(ToolCallItemView {
                id,
                name,
                arguments,
            }),
            _ => None,
        }
    }

    fn as_tool_result(&self) -> Option<ToolResultItemView<'_>> {
        None // OpenAI assistant turns do not carry tool results
    }
}
