use std::{collections::BTreeMap, env, pin::Pin, sync::Arc};

use async_stream::try_stream;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use reqwest::{
    Client,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue},
};
use serde_json::{Value, json};
use thiserror::Error;

use agents_protocol::{
    budget::Usage,
    conversation::{
        AssistantInputItem, InputMessageRole, MessageContent, ModelInput, ModelInputItem, RawJson,
        ToolCallId, ToolMetadata, ToolName, ToolUse,
    },
    llm::{
        CompletionEvent, CompletionEventStream, CompletionRequest, FinishReason, LlmAdapter,
        ReasoningEffort, ReasoningParams, ReasoningSummary, StreamKind, StructuredTurn,
        StructuredTurnEvent, StructuredTurnEventStream, TextTurn, TextTurnEvent,
        TextTurnEventStream, TurnConfig,
    },
    structured::StructuredOutput,
    toolset::{ToolCallError, ToolPolicy, ToolSelector, Toolset},
};

#[derive(Clone)]
pub struct OpenAiAdapter {
    client: Arc<Client>,
    api_key: Arc<str>,
    base_url: Arc<str>,
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
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Arc::from(base_url.into());
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
    #[error("failed to parse tool call: {0}")]
    ToolCall(#[from] ToolCallError),
    #[error("failed to parse structured output: {0}")]
    StructuredOutput(serde_json::Error),
    #[error("unexpected SSE payload: {message}")]
    Sse { message: String },
}

#[async_trait::async_trait]
impl LlmAdapter for OpenAiAdapter {
    type Error = OpenAiError;

    async fn responses_text<T>(
        &self,
        input: ModelInput,
        turn: TextTurn<T>,
    ) -> Result<TextTurnEventStream<T, Self::Error>, Self::Error>
    where
        T: Toolset,
    {
        let model = turn.config.model.to_string();
        let body = build_responses_request::<T>(&input, &turn.config, None)?;
        let stream = self.send_streaming_json("/responses", body).await?;
        Ok(Box::pin(map_text_stream::<T, _>(stream, model)) as TextTurnEventStream<T, Self::Error>)
    }

    async fn responses_structured<T, O>(
        &self,
        input: ModelInput,
        turn: StructuredTurn<T, O>,
    ) -> Result<StructuredTurnEventStream<T, O, Self::Error>, Self::Error>
    where
        T: Toolset,
        O: StructuredOutput,
    {
        let model = turn.config.model.to_string();
        let output_schema = Some(json!({
            "type": "json_schema",
            "name": <O as StructuredOutput>::schema_name(),
            "strict": true,
            "schema": serde_json::to_value(<O as StructuredOutput>::json_schema())?,
        }));
        let body = build_responses_request::<T>(&input, &turn.config, output_schema)?;
        let stream = self.send_streaming_json("/responses", body).await?;
        Ok(Box::pin(map_structured_stream::<T, O, _>(stream, model))
            as StructuredTurnEventStream<T, O, Self::Error>)
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionEventStream<Self::Error>, Self::Error> {
        let body = build_completion_request(&request);
        let stream = self.send_streaming_json("/completions", body).await?;
        Ok(
            Box::pin(map_completion_stream(stream, request.model.to_string()))
                as CompletionEventStream<Self::Error>,
        )
    }

    async fn recover_usage(
        &self,
        kind: StreamKind,
        request_id: &str,
    ) -> Result<Option<Usage>, Self::Error> {
        match kind {
            StreamKind::ResponsesText | StreamKind::ResponsesStructured => {
                let value = self.get_json(&format!("/responses/{request_id}")).await?;
                Ok(Some(parse_response_usage(&value)))
            }
            StreamKind::Completion => Ok(None),
        }
    }
}

fn build_responses_request<T>(
    input: &ModelInput,
    config: &TurnConfig<T>,
    text_format: Option<Value>,
) -> Result<Value, OpenAiError>
where
    T: Toolset,
{
    let tools = build_tool_definitions::<T>(&config.tools)?;
    let mut body = json!({
        "model": config.model,
        "input": convert_model_input(input)?,
        "stream": true,
        "tools": tools,
        "parallel_tool_calls": !config.tools.requires_tools(),
    });

    if let Some(temperature) = config.generation.temperature {
        body["temperature"] = json!(temperature.get());
    }
    if let Some(max_output_tokens) = config.generation.max_output_tokens {
        body["max_output_tokens"] = json!(max_output_tokens);
    }
    if let Some(reasoning) = reasoning_to_json(&config.reasoning) {
        body["reasoning"] = reasoning;
    }
    if let Some(text_format) = text_format {
        body["text"] = json!({ "format": text_format });
    }
    if config.tools.requires_tools() {
        body["tool_choice"] = json!("required");
    } else if !config.tools.uses_tools() {
        body["tool_choice"] = json!("none");
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
                lower_tool_use(tool_use, &mut items)?;
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

fn lower_tool_use(tool_use: &ToolUse, out: &mut Vec<Value>) -> Result<(), OpenAiError> {
    out.push(json!({
        "type": "function_call",
        "call_id": tool_use.id.as_str(),
        "name": tool_use.name.as_str(),
        "arguments": serde_json::from_str::<Value>(tool_use.arguments.get())?,
    }));
    out.push(json!({
        "type": "function_call_output",
        "call_id": tool_use.id.as_str(),
        "output": tool_use.result.get(),
    }));
    Ok(())
}

fn flush_assistant_message(message_content: &mut Vec<Value>, out: &mut Vec<Value>) {
    if message_content.is_empty() {
        return;
    }
    out.push(json!({
        "type": "message",
        "role": "assistant",
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

fn message_content(content: &MessageContent) -> Value {
    match content {
        MessageContent::Text(text) => json!({
            "type": "input_text",
            "text": text,
        }),
    }
}

fn build_tool_definitions<T>(tool_policy: &ToolPolicy<T>) -> Result<Vec<Value>, OpenAiError>
where
    T: Toolset,
{
    if !tool_policy.uses_tools() {
        return Ok(Vec::new());
    }

    let selected = tool_policy.selected().map(|selectors| {
        selectors
            .iter()
            .map(|selector| selector.name())
            .collect::<Vec<_>>()
    });
    T::definitions()
        .iter()
        .filter(|tool| {
            selected
                .as_ref()
                .is_none_or(|names| names.contains(&tool.name))
        })
        .map(|tool| {
            Ok(json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": serde_json::to_value(tool.input_schema())?,
            }))
        })
        .collect()
}

fn reasoning_to_json(reasoning: &ReasoningParams) -> Option<Value> {
    let mut map = serde_json::Map::new();
    if let Some(effort) = reasoning.effort {
        map.insert(
            "effort".into(),
            json!(match effort {
                ReasoningEffort::Low => "low",
                ReasoningEffort::Medium => "medium",
                ReasoningEffort::High => "high",
            }),
        );
    }
    if let Some(summary) = reasoning.summary {
        map.insert(
            "summary".into(),
            json!(match summary {
                ReasoningSummary::Auto => "auto",
                ReasoningSummary::Concise => "concise",
                ReasoningSummary::Detailed => "detailed",
            }),
        );
    }
    if map.is_empty() {
        None
    } else {
        Some(Value::Object(map))
    }
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

    fn finish<T>(
        &mut self,
        key: String,
        id: ToolCallId,
        name: ToolName,
        explicit_arguments_json: Option<String>,
    ) -> Result<Option<T::ToolCall>, OpenAiError>
    where
        T: Toolset,
    {
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
        let invocation = T::parse_tool_call(
            ToolMetadata::new(id.clone(), name.clone(), arguments.clone()),
            name.as_str(),
            arguments.get(),
        )?;
        self.finalized.insert(
            key,
            FinalizedToolCall {
                id,
                name,
                arguments_json,
            },
        );
        Ok(Some(invocation))
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

fn maybe_parse_structured_output<O>(
    buffer: &str,
    emitted_ready: &mut bool,
) -> Result<Option<O>, OpenAiError>
where
    O: StructuredOutput,
{
    if *emitted_ready || buffer.is_empty() {
        return Ok(None);
    }

    let value = serde_json::from_str::<O>(buffer).map_err(OpenAiError::StructuredOutput)?;
    *emitted_ready = true;
    Ok(Some(value))
}

fn map_text_stream<T, S>(
    stream: S,
    fallback_model: String,
) -> impl Stream<Item = Result<TextTurnEvent<T>, OpenAiError>> + Send + 'static
where
    T: Toolset,
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
                        yield TextTurnEvent::Started {
                            request_id: request_id.clone(),
                            model: model.clone(),
                        };
                    }
                    continue;
                }

                if !started {
                    started = true;
                    yield TextTurnEvent::Started {
                        request_id: request_id.clone(),
                        model: model.clone(),
                    };
                }

                match event_type {
                    "response.output_text.delta" => {
                        if let Some(delta) = event["delta"].as_str() {
                            yield TextTurnEvent::TextDelta {
                                delta: delta.to_string(),
                            };
                        }
                    }
                    "response.reasoning_summary_text.delta" | "response.reasoning.delta" => {
                        if let Some(delta) = event["delta"].as_str() {
                            yield TextTurnEvent::ReasoningDelta {
                                delta: delta.to_string(),
                            };
                        }
                    }
                    "response.refusal.delta" => {
                        saw_refusal = true;
                        if let Some(delta) = event["delta"].as_str() {
                            yield TextTurnEvent::RefusalDelta {
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
                                yield TextTurnEvent::ToolCallChunk {
                                    id,
                                    name,
                                    arguments_json_delta: delta,
                                };
                            }
                    }
                    "response.function_call_arguments.done" => {
                        saw_tool_call = true;
                        if let Some(invocation) = tool_calls.finish::<T>(
                            response_tool_key(&event),
                            response_tool_id(&event),
                            response_tool_name(&event),
                            event.get("arguments").map(json_string).transpose()?,
                        )? {
                            yield TextTurnEvent::ToolCallReady(invocation);
                        }
                    }
                    "response.output_item.done" => {
                        if event.pointer("/item/type").and_then(Value::as_str) == Some("function_call") {
                            saw_tool_call = true;
                            if let Some(invocation) = tool_calls.finish::<T>(
                                output_item_tool_key(&event),
                                output_item_tool_id(&event),
                                output_item_tool_name(&event),
                                event.pointer("/item/arguments").map(json_string).transpose()?,
                            )? {
                                yield TextTurnEvent::ToolCallReady(invocation);
                            }
                        }
                    }
                    "response.completed" => {
                        if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                            request_id = Some(id.to_string());
                        }
                        yield TextTurnEvent::Completed {
                            request_id: request_id.clone(),
                            finish_reason: if saw_tool_call {
                                FinishReason::ToolCall
                            } else if saw_refusal {
                                FinishReason::ContentFilter
                            } else {
                                FinishReason::Stop
                            },
                            usage: parse_response_usage(&event["response"]),
                        };
                    }
                    _ => {}
                }
            }
        }
    }
}

fn map_structured_stream<T, O, S>(
    stream: S,
    fallback_model: String,
) -> impl Stream<Item = Result<StructuredTurnEvent<T, O>, OpenAiError>> + Send + 'static
where
    T: Toolset,
    O: StructuredOutput,
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
                        yield StructuredTurnEvent::Started {
                            request_id: request_id.clone(),
                            model: model.clone(),
                        };
                    }
                    continue;
                }

                if !started {
                    started = true;
                    yield StructuredTurnEvent::Started {
                        request_id: request_id.clone(),
                        model: model.clone(),
                    };
                }

                match event_type {
                    "response.output_text.delta" => {
                        if let Some(delta) = event["delta"].as_str() {
                            structured_buffer.push_str(delta);
                            yield StructuredTurnEvent::StructuredOutputChunk {
                                json_delta: delta.to_string(),
                            };
                        }
                    }
                    "response.output_text.done" => {
                        if let Some(text) = event["text"].as_str() {
                            structured_buffer.clear();
                            structured_buffer.push_str(text);
                        }
                        if let Some(value) =
                            maybe_parse_structured_output::<O>(&structured_buffer, &mut emitted_ready)?
                        {
                            yield StructuredTurnEvent::StructuredOutputReady(value);
                        }
                    }
                    "response.reasoning_summary_text.delta" | "response.reasoning.delta" => {
                        if let Some(delta) = event["delta"].as_str() {
                            yield StructuredTurnEvent::ReasoningDelta {
                                delta: delta.to_string(),
                            };
                        }
                    }
                    "response.refusal.delta" => {
                        saw_refusal = true;
                        if let Some(delta) = event["delta"].as_str() {
                            yield StructuredTurnEvent::RefusalDelta {
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
                                yield StructuredTurnEvent::ToolCallChunk {
                                    id,
                                    name,
                                    arguments_json_delta: delta,
                                };
                            }
                    }
                    "response.function_call_arguments.done" => {
                        saw_tool_call = true;
                        if let Some(invocation) = tool_calls.finish::<T>(
                            response_tool_key(&event),
                            response_tool_id(&event),
                            response_tool_name(&event),
                            event.get("arguments").map(json_string).transpose()?,
                        )? {
                            yield StructuredTurnEvent::ToolCallReady(invocation);
                        }
                    }
                    "response.output_item.done" => {
                        if event.pointer("/item/type").and_then(Value::as_str) == Some("function_call") {
                            saw_tool_call = true;
                            if let Some(invocation) = tool_calls.finish::<T>(
                                output_item_tool_key(&event),
                                output_item_tool_id(&event),
                                output_item_tool_name(&event),
                                event.pointer("/item/arguments").map(json_string).transpose()?,
                            )? {
                                yield StructuredTurnEvent::ToolCallReady(invocation);
                            }
                        }
                    }
                    "response.completed" => {
                        if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                            request_id = Some(id.to_string());
                        }
                        if let Some(value) =
                            maybe_parse_structured_output::<O>(&structured_buffer, &mut emitted_ready)?
                        {
                            yield StructuredTurnEvent::StructuredOutputReady(value);
                        }
                        yield StructuredTurnEvent::Completed {
                            request_id: request_id.clone(),
                            finish_reason: if saw_tool_call {
                                FinishReason::ToolCall
                            } else if saw_refusal {
                                FinishReason::ContentFilter
                            } else {
                                FinishReason::Stop
                            },
                            usage: parse_response_usage(&event["response"]),
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
    Usage {
        input_tokens: usage["input_tokens"].as_u64().unwrap_or_default(),
        output_tokens: usage["output_tokens"].as_u64().unwrap_or_default(),
        total_tokens: usage["total_tokens"].as_u64().unwrap_or_default(),
        cost_micros_usd: 0,
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
    use futures::{StreamExt, executor::block_on};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    use agents_protocol::{
        AssistantInputItem, InputMessageRole, ModelInput, ModelInputItem, ToolDef, ToolUse,
        toolset::{ToolCallWrapper, ToolInput, ToolPolicy, ToolSelector},
    };

    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct WeatherArgs {
        city: String,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct WeatherResult {
        forecast: String,
    }

    impl ToolInput for WeatherArgs {
        type Output = WeatherResult;

        const NAME: &'static str = "weather";
        const DESCRIPTION: &'static str = "Get weather";
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct WeatherArgsCall {
        metadata: ToolMetadata,
        input: WeatherArgs,
    }

    impl ToolCallWrapper for WeatherArgsCall {
        fn metadata(&self) -> &ToolMetadata {
            &self.metadata
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum ToolsCall {
        Weather(WeatherArgsCall),
    }

    impl ToolCallWrapper for ToolsCall {
        fn metadata(&self) -> &ToolMetadata {
            match self {
                Self::Weather(call) => &call.metadata,
            }
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct Tools;

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize, JsonSchema)]
    enum ToolsSelector {
        Weather,
    }

    impl ToolSelector<Tools> for ToolsSelector {
        fn name(self) -> &'static str {
            match self {
                Self::Weather => "weather",
            }
        }

        fn all() -> &'static [Self] {
            &[Self::Weather]
        }

        fn try_from_name(name: &str) -> Option<Self> {
            match name {
                "weather" => Some(Self::Weather),
                _ => None,
            }
        }
    }

    impl Toolset for Tools {
        type ToolCall = ToolsCall;
        type Selector = ToolsSelector;

        fn definitions() -> &'static [ToolDef] {
            fn weather_args_schema() -> schemars::Schema {
                schemars::schema_for!(WeatherArgs)
            }

            static DEFS: [ToolDef; 1] =
                [ToolDef::new("weather", "Get weather", weather_args_schema)];
            &DEFS
        }

        fn parse_tool_call(
            metadata: ToolMetadata,
            name: &str,
            arguments_json: &str,
        ) -> Result<Self::ToolCall, ToolCallError> {
            match name {
                "weather" => serde_json::from_str(arguments_json)
                    .map(|input| ToolsCall::Weather(WeatherArgsCall { metadata, input }))
                    .map_err(|source| ToolCallError::Deserialize {
                        name: name.to_string(),
                        source,
                    }),
                _ => Err(ToolCallError::UnknownTool {
                    name: name.to_string(),
                }),
            }
        }
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
            map_text_stream::<Tools, _>(futures::stream::iter(payloads), "gpt-4.1".into())
                .collect::<Vec<_>>()
                .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        assert!(matches!(events[0], TextTurnEvent::Started { .. }));
        assert!(matches!(events[1], TextTurnEvent::ReasoningDelta { .. }));
        assert!(matches!(events[2], TextTurnEvent::RefusalDelta { .. }));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, TextTurnEvent::ToolCallReady(_)))
        );
        assert!(matches!(
            events.last(),
            Some(TextTurnEvent::Completed { .. })
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
            map_structured_stream::<Tools, Summary, _>(
                futures::stream::iter(payloads),
                "gpt-4.1".into(),
            )
            .collect::<Vec<_>>()
            .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        assert!(events.iter().any(|event| matches!(event, StructuredTurnEvent::StructuredOutputReady(summary) if summary.answer == "42")));
    }

    #[test]
    fn disabled_tool_mode_sends_no_tool_definitions() {
        let tools = build_tool_definitions::<Tools>(&ToolPolicy::Disabled).unwrap();
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
            map_text_stream::<Tools, _>(futures::stream::iter(payloads), "gpt-4.1".into())
                .collect::<Vec<_>>()
                .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        let ready = events
            .iter()
            .filter_map(|event| match event {
                TextTurnEvent::ToolCallReady(invocation) => Some(invocation),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(ready.len(), 1);
        assert!(matches!(
            &ready[0],
            ToolsCall::Weather(WeatherArgsCall {
                input: WeatherArgs { city },
                ..
            }) if city == "Tokyo"
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
            map_structured_stream::<Tools, u64, _>(
                futures::stream::iter(payloads),
                "gpt-4.1".into(),
            )
            .collect::<Vec<_>>()
            .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        assert!(matches!(
            events[1],
            StructuredTurnEvent::StructuredOutputChunk { .. }
        ));
        assert!(matches!(
            events[2],
            StructuredTurnEvent::StructuredOutputChunk { .. }
        ));
        assert!(matches!(
            events[3],
            StructuredTurnEvent::StructuredOutputReady(12)
        ));
        assert!(matches!(events[4], StructuredTurnEvent::Completed { .. }));
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
