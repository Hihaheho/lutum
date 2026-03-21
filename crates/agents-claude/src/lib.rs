use std::{collections::BTreeMap, env, pin::Pin, sync::Arc};

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
use async_stream::try_stream;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use reqwest::{
    Client,
    header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tracing::trace;

const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const MIN_THINKING_BUDGET_TOKENS: u32 = 1024;
const MIN_RESPONSE_TOKENS_WITH_THINKING: u32 = 1024;

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static>>;

pub trait BudgetTokensResolver: Send + Sync + 'static {
    fn resolve(&self, extensions: &RequestExtensions) -> Option<u32>;
}

#[derive(Clone)]
pub struct ClaudeAdapter {
    client: Arc<Client>,
    api_key: Arc<str>,
    base_url: Arc<str>,
    budget_tokens_resolver: Option<Arc<dyn BudgetTokensResolver>>,
}

impl ClaudeAdapter {
    pub fn from_env() -> Result<Self, ClaudeError> {
        let api_key = env::var(ANTHROPIC_API_KEY_ENV).map_err(ClaudeError::MissingApiKey)?;
        Ok(Self::new(api_key))
    }

    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Arc::new(Client::new()),
            api_key: Arc::from(api_key.into()),
            base_url: normalize_base_url(DEFAULT_BASE_URL),
            budget_tokens_resolver: None,
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = normalize_base_url(base_url.into());
        self
    }

    pub fn with_budget_tokens_resolver(
        mut self,
        resolver: impl BudgetTokensResolver + 'static,
    ) -> Self {
        self.budget_tokens_resolver = Some(Arc::new(resolver));
        self
    }

    fn request_headers(&self) -> Result<HeaderMap, ClaudeError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("x-api-key"),
            HeaderValue::from_str(&self.api_key).map_err(ClaudeError::InvalidHeader)?,
        );
        headers.insert(
            HeaderName::from_static("anthropic-version"),
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        Ok(headers)
    }

    async fn send_streaming_json(
        &self,
        path: &str,
        body: Value,
    ) -> Result<ByteStream, ClaudeError> {
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

    async fn post_json(&self, path: &str, body: Value) -> Result<Value, ClaudeError> {
        let response = self
            .client
            .post(format!("{}{}", self.base_url, path))
            .headers(self.request_headers()?)
            .json(&body)
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
pub enum ClaudeError {
    #[error("ANTHROPIC_API_KEY is not set: {0}")]
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
    #[error("missing required field `{field}` in Claude payload")]
    MissingField { field: &'static str },
    #[error("invalid Claude request: {message}")]
    InvalidRequest { message: String },
    #[error("unexpected Claude SSE payload: {message}")]
    Sse { message: String },
}

#[async_trait::async_trait]
impl LlmAdapter for ClaudeAdapter {
    async fn responses_text(
        &self,
        input: ModelInput,
        turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        let model = turn.config.model.to_string();
        let thinking_budget = self
            .budget_tokens_resolver
            .as_ref()
            .and_then(|resolver| resolver.resolve(turn.extensions.as_ref()))
            .map(|budget| budget.max(MIN_THINKING_BUDGET_TOKENS));
        let body = build_messages_request(&input, &turn.config, thinking_budget, None, true)
            .map_err(AgentError::backend)?;
        let stream = self
            .send_streaming_json("/v1/messages", body)
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
        let thinking_budget = self
            .budget_tokens_resolver
            .as_ref()
            .and_then(|resolver| resolver.resolve(turn.extensions.as_ref()))
            .map(|budget| budget.max(MIN_THINKING_BUDGET_TOKENS));
        let format = Some(json!({
            "type": "json_schema",
            "schema": turn.output.schema,
        }));
        let body = build_messages_request(&input, &turn.config, thinking_budget, format, true)
            .map_err(AgentError::backend)?;
        let stream = self
            .send_streaming_json("/v1/messages", body)
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
        let fallback_model = request.model.to_string();
        let value = self
            .post_json("/v1/messages", build_completion_request(&request))
            .await
            .map_err(AgentError::backend)?;

        Ok(Box::pin(try_stream! {
            let request_id = value["id"].as_str().map(ToOwned::to_owned);
            let model = value["model"]
                .as_str()
                .map(ToOwned::to_owned)
                .unwrap_or(fallback_model);
            let finish_reason = map_claude_stop_reason(value["stop_reason"].as_str());
            let usage = parse_claude_usage(value.get("usage").unwrap_or(&Value::Null));

            yield CompletionEvent::Started {
                request_id: request_id.clone(),
                model,
            };

            let text = extract_completion_text(&value);
            if !text.is_empty() {
                yield CompletionEvent::TextDelta(text);
            }

            yield CompletionEvent::Completed {
                request_id,
                finish_reason,
                usage,
            };
        }) as CompletionEventStream)
    }

    async fn recover_usage(
        &self,
        _kind: StreamKind,
        _request_id: &str,
    ) -> Result<Option<Usage>, AgentError> {
        Ok(None)
    }
}

fn normalize_base_url(url: impl Into<String>) -> Arc<str> {
    Arc::from(url.into().trim_end_matches('/').to_string())
}

async fn error_for_status_with_body(
    response: reqwest::Response,
) -> Result<reqwest::Response, ClaudeError> {
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

    Err(ClaudeError::HttpStatus { status, message })
}

#[derive(Default)]
struct CompiledClaudeConversation {
    system: Vec<Value>,
    messages: Vec<ClaudeMessage>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClaudeRole {
    User,
    Assistant,
}

impl ClaudeRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug)]
struct ClaudeMessage {
    role: ClaudeRole,
    content: Vec<Value>,
}

impl CompiledClaudeConversation {
    fn push_system_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.system.push(text_block(text));
    }

    fn push_block(&mut self, role: ClaudeRole, block: Value) {
        if let Some(last) = self.messages.last_mut()
            && last.role == role
        {
            last.content.push(block);
            return;
        }

        self.messages.push(ClaudeMessage {
            role,
            content: vec![block],
        });
    }

    fn into_json_messages(self) -> Vec<Value> {
        self.messages
            .into_iter()
            .filter(|message| !message.content.is_empty())
            .map(|message| {
                json!({
                    "role": message.role.as_str(),
                    "content": message.content,
                })
            })
            .collect()
    }
}

fn build_messages_request(
    input: &ModelInput,
    config: &AdapterTurnConfig,
    thinking_budget: Option<u32>,
    format: Option<Value>,
    stream: bool,
) -> Result<Value, ClaudeError> {
    let compiled = compile_model_input(input)?;
    let system = compiled.system.clone();
    let max_tokens = resolve_max_tokens(config.generation.max_output_tokens, thinking_budget);
    let mut body = json!({
        "model": config.model,
        "max_tokens": max_tokens,
        "messages": compiled.into_json_messages(),
        "stream": stream,
    });

    if !system.is_empty() {
        body["system"] = Value::Array(system);
    }
    if let Some(temperature) = config.generation.temperature {
        body["temperature"] = json!(temperature.get());
    }
    if !config.tools.is_empty() {
        body["tools"] = Value::Array(build_tool_definitions(config));
    }
    if let Some(tool_choice) = build_tool_choice(config) {
        body["tool_choice"] = tool_choice;
    }
    if let Some(budget_tokens) = thinking_budget {
        body["thinking"] = json!({
            "type": "enabled",
            "budget_tokens": budget_tokens,
        });
    }
    if let Some(format) = format {
        body["output_config"] = json!({
            "format": format,
        });
    }

    Ok(body)
}

fn build_completion_request(request: &CompletionRequest) -> Value {
    let mut body = json!({
        "model": request.model,
        "max_tokens": request.options.max_output_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "messages": [{
            "role": "user",
            "content": [text_block(&request.prompt)],
        }],
    });

    if let Some(temperature) = request.options.temperature {
        body["temperature"] = json!(temperature.get());
    }
    if !request.options.stop.is_empty() {
        body["stop_sequences"] = json!(request.options.stop);
    }

    body
}

fn resolve_max_tokens(explicit: Option<u32>, thinking_budget: Option<u32>) -> u32 {
    let base = explicit.unwrap_or(DEFAULT_MAX_TOKENS).max(1);
    if let Some(budget) = thinking_budget {
        return base.max(budget.saturating_add(MIN_RESPONSE_TOKENS_WITH_THINKING));
    }
    base
}

fn build_tool_definitions(config: &AdapterTurnConfig) -> Vec<Value> {
    config
        .tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            })
        })
        .collect()
}

fn build_tool_choice(config: &AdapterTurnConfig) -> Option<Value> {
    match &config.tool_choice {
        AdapterToolChoice::None => None,
        AdapterToolChoice::Auto => Some(json!({ "type": "auto" })),
        AdapterToolChoice::Required => {
            if config.tools.len() == 1 {
                Some(json!({
                    "type": "tool",
                    "name": config.tools[0].name,
                }))
            } else {
                Some(json!({ "type": "any" }))
            }
        }
        AdapterToolChoice::Specific(name) => Some(json!({
            "type": "tool",
            "name": name,
        })),
    }
}

fn compile_model_input(input: &ModelInput) -> Result<CompiledClaudeConversation, ClaudeError> {
    let mut compiled = CompiledClaudeConversation::default();

    for item in input.items() {
        match item {
            ModelInputItem::Message { role, content } => {
                emit_message(role, content.iter(), &mut compiled)?;
            }
            ModelInputItem::Assistant(item) => {
                compiled.push_block(ClaudeRole::Assistant, assistant_replay_block(item));
            }
            ModelInputItem::ToolUse(tool_use) => emit_tool_use(tool_use, &mut compiled)?,
            ModelInputItem::Turn(turn) => {
                if let Some(claude_turn) =
                    turn.as_ref().as_any().downcast_ref::<ClaudeCommittedTurn>()
                {
                    emit_claude_turn_exact(claude_turn, &mut compiled)?;
                } else {
                    emit_turn_from_view(turn.as_ref(), &mut compiled)?;
                }
            }
        }
    }

    Ok(compiled)
}

fn emit_message<'a>(
    role: &InputMessageRole,
    content: impl IntoIterator<Item = &'a MessageContent>,
    compiled: &mut CompiledClaudeConversation,
) -> Result<(), ClaudeError> {
    match role {
        InputMessageRole::System | InputMessageRole::Developer => {
            for item in content {
                let MessageContent::Text(text) = item;
                compiled.push_system_text(text);
            }
        }
        InputMessageRole::User => {
            for item in content {
                compiled.push_block(ClaudeRole::User, message_content_block(item));
            }
        }
    }

    Ok(())
}

fn emit_tool_use(
    tool_use: &ToolUse,
    compiled: &mut CompiledClaudeConversation,
) -> Result<(), ClaudeError> {
    compiled.push_block(
        ClaudeRole::Assistant,
        tool_use_block(&tool_use.id, &tool_use.name, &tool_use.arguments)?,
    );
    compiled.push_block(ClaudeRole::User, tool_result_block(tool_use)?);
    Ok(())
}

fn emit_claude_turn_exact(
    turn: &ClaudeCommittedTurn,
    compiled: &mut CompiledClaudeConversation,
) -> Result<(), ClaudeError> {
    for item in &turn.items {
        match item {
            ClaudeTurnItem::Text { content } => {
                compiled.push_block(ClaudeRole::Assistant, text_block(content));
            }
            ClaudeTurnItem::Thinking { content, signature } => {
                compiled.push_block(
                    ClaudeRole::Assistant,
                    json!({
                        "type": "thinking",
                        "thinking": content,
                        "signature": signature,
                    }),
                );
            }
            ClaudeTurnItem::Reasoning { content } | ClaudeTurnItem::Refusal { content } => {
                compiled.push_block(ClaudeRole::Assistant, text_block(content));
            }
            ClaudeTurnItem::ToolCall {
                id,
                name,
                arguments,
            } => {
                compiled.push_block(ClaudeRole::Assistant, tool_use_block(id, name, arguments)?);
            }
        }
    }

    Ok(())
}

fn emit_turn_from_view(
    turn: &dyn TurnView,
    compiled: &mut CompiledClaudeConversation,
) -> Result<(), ClaudeError> {
    match turn.role() {
        TurnRole::System | TurnRole::Developer => {
            for index in 0..turn.item_count() {
                let Some(item) = turn.item_at(index) else {
                    continue;
                };
                if let Some(text) = item.as_text() {
                    compiled.push_system_text(text);
                    continue;
                }
                if let Some(text) = item.as_reasoning().or_else(|| item.as_refusal()) {
                    compiled.push_system_text(text);
                    continue;
                }
                if item.as_tool_call().is_some() || item.as_tool_result().is_some() {
                    return Err(ClaudeError::InvalidRequest {
                        message: format!(
                            "cannot lower tool items from a {:?} turn into Claude input",
                            turn.role()
                        ),
                    });
                }
            }
        }
        TurnRole::User => {
            for index in 0..turn.item_count() {
                let Some(item) = turn.item_at(index) else {
                    continue;
                };
                if let Some(text) = item.as_text() {
                    compiled.push_block(ClaudeRole::User, text_block(text));
                    continue;
                }
                if let Some(text) = item.as_reasoning().or_else(|| item.as_refusal()) {
                    compiled.push_block(ClaudeRole::User, text_block(text));
                    continue;
                }
                if let Some(tool_result) = item.as_tool_result() {
                    compiled
                        .push_block(ClaudeRole::User, tool_result_block_from_view(tool_result)?);
                    continue;
                }
                if item.as_tool_call().is_some() {
                    return Err(ClaudeError::InvalidRequest {
                        message:
                            "cannot lower a user turn item into a Claude assistant tool_use block"
                                .to_string(),
                    });
                }
            }
        }
        TurnRole::Assistant => {
            for index in 0..turn.item_count() {
                let Some(item) = turn.item_at(index) else {
                    continue;
                };
                if let Some(text) = item.as_text() {
                    compiled.push_block(ClaudeRole::Assistant, text_block(text));
                    continue;
                }
                if let Some(text) = item.as_reasoning().or_else(|| item.as_refusal()) {
                    compiled.push_block(ClaudeRole::Assistant, text_block(text));
                    continue;
                }
                if let Some(tool_call) = item.as_tool_call() {
                    compiled.push_block(
                        ClaudeRole::Assistant,
                        tool_use_block(tool_call.id, tool_call.name, tool_call.arguments)?,
                    );
                    continue;
                }
                if item.as_tool_result().is_some() {
                    return Err(ClaudeError::InvalidRequest {
                        message: "cannot lower an assistant tool_result item into Claude input"
                            .to_string(),
                    });
                }
            }
        }
    }

    Ok(())
}

fn message_content_block(content: &MessageContent) -> Value {
    match content {
        MessageContent::Text(text) => text_block(text),
    }
}

fn assistant_replay_block(item: &AssistantInputItem) -> Value {
    match item {
        AssistantInputItem::Text(text)
        | AssistantInputItem::Reasoning(text)
        | AssistantInputItem::Refusal(text) => text_block(text),
    }
}

fn text_block(text: &str) -> Value {
    json!({
        "type": "text",
        "text": text,
    })
}

fn tool_use_block(
    id: &ToolCallId,
    name: &ToolName,
    arguments: &RawJson,
) -> Result<Value, ClaudeError> {
    Ok(json!({
        "type": "tool_use",
        "id": id.as_str(),
        "name": name.as_str(),
        "input": parse_tool_input(arguments)?,
    }))
}

fn tool_result_block(tool_use: &ToolUse) -> Result<Value, ClaudeError> {
    Ok(json!({
        "type": "tool_result",
        "tool_use_id": tool_use.id.as_str(),
        "content": tool_result_content(&tool_use.result)?,
    }))
}

fn tool_result_block_from_view(tool_result: ToolResultItemView<'_>) -> Result<Value, ClaudeError> {
    Ok(json!({
        "type": "tool_result",
        "tool_use_id": tool_result.id.as_str(),
        "content": tool_result_content(tool_result.result)?,
    }))
}

fn parse_tool_input(arguments: &RawJson) -> Result<Value, ClaudeError> {
    let value = serde_json::from_str::<Value>(arguments.get())?;
    if value.is_object() {
        return Ok(value);
    }

    Err(ClaudeError::InvalidRequest {
        message: "Claude tool inputs must be JSON objects".to_string(),
    })
}

fn tool_result_content(result: &RawJson) -> Result<Value, ClaudeError> {
    let value = serde_json::from_str::<Value>(result.get())?;
    Ok(match value {
        Value::String(text) => Value::String(text),
        other => Value::String(serde_json::to_string(&other)?),
    })
}

#[derive(Debug)]
enum ContentBlockState {
    Text {
        content: String,
    },
    Thinking {
        content: String,
        signature: String,
    },
    ToolUse {
        id: ToolCallId,
        name: ToolName,
        initial_json: Option<String>,
        delta_json: String,
        finalized_arguments: Option<RawJson>,
    },
    Unsupported {
        kind: String,
    },
}

impl ContentBlockState {
    fn push_text_delta(&mut self, delta: &str) -> Result<(), ClaudeError> {
        match self {
            Self::Text { content } => {
                content.push_str(delta);
                Ok(())
            }
            other => Err(ClaudeError::Sse {
                message: format!("received text delta for non-text block: {other:?}"),
            }),
        }
    }

    fn push_thinking_delta(&mut self, delta: &str) -> Result<(), ClaudeError> {
        match self {
            Self::Thinking { content, .. } => {
                content.push_str(delta);
                Ok(())
            }
            other => Err(ClaudeError::Sse {
                message: format!("received thinking delta for non-thinking block: {other:?}"),
            }),
        }
    }

    fn push_signature_delta(&mut self, delta: &str) -> Result<(), ClaudeError> {
        match self {
            Self::Thinking { signature, .. } => {
                signature.push_str(delta);
                Ok(())
            }
            other => Err(ClaudeError::Sse {
                message: format!("received signature delta for non-thinking block: {other:?}"),
            }),
        }
    }

    fn push_tool_delta(&mut self, delta: &str) -> Result<(ToolCallId, ToolName), ClaudeError> {
        match self {
            Self::ToolUse {
                id,
                name,
                delta_json,
                ..
            } => {
                delta_json.push_str(delta);
                Ok((id.clone(), name.clone()))
            }
            other => Err(ClaudeError::Sse {
                message: format!("received tool delta for non-tool block: {other:?}"),
            }),
        }
    }

    fn finalize_tool_call(&mut self) -> Result<Option<ToolMetadata>, ClaudeError> {
        match self {
            Self::ToolUse {
                id,
                name,
                initial_json,
                delta_json,
                finalized_arguments,
            } => {
                if let Some(arguments) = finalized_arguments.clone() {
                    return Ok(Some(ToolMetadata::new(id.clone(), name.clone(), arguments)));
                }

                let arguments_json = if delta_json.is_empty() {
                    initial_json.clone().unwrap_or_else(|| "{}".to_string())
                } else {
                    delta_json.clone()
                };
                let arguments = RawJson::parse(arguments_json)?;
                *finalized_arguments = Some(arguments.clone());
                Ok(Some(ToolMetadata::new(id.clone(), name.clone(), arguments)))
            }
            Self::Unsupported { .. } => Ok(None),
            other => Err(ClaudeError::Sse {
                message: format!("received content_block_stop for non-tool block: {other:?}"),
            }),
        }
    }
}

fn map_text_stream<S>(
    stream: S,
    fallback_model: String,
) -> impl Stream<Item = Result<ErasedTextTurnEvent, ClaudeError>> + Send + 'static
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    try_stream! {
        let mut parser = ClaudeSseParser::default();
        let mut started = false;
        let mut request_id = None::<String>;
        let mut model = fallback_model;
        let mut usage = Usage::zero();
        let mut stop_reason = None::<String>;
        let mut blocks = BTreeMap::<usize, ContentBlockState>::new();
        futures::pin_mut!(stream);

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            for frame in parser.push(&chunk)? {
                if let Some(event) = frame.event.as_deref() {
                    if event == "ping" {
                        trace!("ignoring Claude ping event");
                        continue;
                    }
                }

                let payload = if frame.data.is_empty() {
                    Value::Null
                } else {
                    serde_json::from_str::<Value>(&frame.data)?
                };
                let event_type = frame
                    .event
                    .as_deref()
                    .or_else(|| payload["type"].as_str())
                    .unwrap_or_default();

                if event_type == "message_start" {
                    request_id = payload.pointer("/message/id").and_then(Value::as_str).map(ToOwned::to_owned);
                    if let Some(event_model) = payload.pointer("/message/model").and_then(Value::as_str) {
                        model = event_model.to_string();
                    }
                    usage = parse_claude_usage(payload.pointer("/message/usage").unwrap_or(&Value::Null));
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
                    "content_block_start" => {
                        let index = required_usize(&payload, "/index")?;
                        let kind = required_str(&payload, "/content_block/type")?;
                        let state = match kind {
                            "text" => ContentBlockState::Text {
                                content: payload
                                    .pointer("/content_block/text")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string(),
                            },
                            "thinking" => ContentBlockState::Thinking {
                                content: payload
                                    .pointer("/content_block/thinking")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string(),
                                signature: payload
                                    .pointer("/content_block/signature")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string(),
                            },
                            "tool_use" => ContentBlockState::ToolUse {
                                id: ToolCallId::from(required_str(&payload, "/content_block/id")?),
                                name: ToolName::from(required_str(&payload, "/content_block/name")?),
                                initial_json: payload.pointer("/content_block/input").and_then(non_null_json_string),
                                delta_json: String::new(),
                                finalized_arguments: None,
                            },
                            other => ContentBlockState::Unsupported {
                                kind: other.to_string(),
                            },
                        };
                        blocks.insert(index, state);
                    }
                    "content_block_delta" => {
                        let index = required_usize(&payload, "/index")?;
                        let delta_type = required_str(&payload, "/delta/type")?;
                        let Some(block) = blocks.get_mut(&index) else {
                            Err(ClaudeError::Sse {
                                message: format!("received delta for unknown block index {index}"),
                            })?
                        };
                        match delta_type {
                            "text_delta" => {
                                let delta = required_str(&payload, "/delta/text")?;
                                block.push_text_delta(delta)?;
                                yield ErasedTextTurnEvent::TextDelta {
                                    delta: delta.to_string(),
                                };
                            }
                            "thinking_delta" => {
                                let delta = required_str(&payload, "/delta/thinking")?;
                                block.push_thinking_delta(delta)?;
                                yield ErasedTextTurnEvent::ReasoningDelta {
                                    delta: delta.to_string(),
                                };
                            }
                            "signature_delta" => {
                                let delta = required_str(&payload, "/delta/signature")?;
                                block.push_signature_delta(delta)?;
                            }
                            "input_json_delta" => {
                                let delta = required_str(&payload, "/delta/partial_json")?;
                                let (id, name) = block.push_tool_delta(delta)?;
                                if !delta.is_empty() {
                                    yield ErasedTextTurnEvent::ToolCallChunk {
                                        id,
                                        name,
                                        arguments_json_delta: delta.to_string(),
                                    };
                                }
                            }
                            "citations_delta" => {}
                            other => {
                                trace!("ignoring Claude content delta type {other}");
                            }
                        }
                    }
                    "content_block_stop" => {
                        let index = required_usize(&payload, "/index")?;
                        if let Some(metadata) = blocks
                            .get_mut(&index)
                            .ok_or_else(|| ClaudeError::Sse {
                                message: format!("received stop for unknown block index {index}"),
                            })?
                            .finalize_tool_call()?
                        {
                            yield ErasedTextTurnEvent::ToolCallReady(metadata);
                        }
                    }
                    "message_delta" => {
                        stop_reason = payload.pointer("/delta/stop_reason").and_then(Value::as_str).map(ToOwned::to_owned);
                        usage = parse_claude_usage(payload.get("usage").unwrap_or(&Value::Null));
                    }
                    "message_stop" => {
                        let finish_reason = map_claude_stop_reason(stop_reason.as_deref());
                        let committed_turn = Arc::new(ClaudeCommittedTurn {
                            request_id: request_id.clone(),
                            model: model.clone(),
                            items: finalize_committed_items(&mut blocks, &finish_reason)?,
                            finish_reason: finish_reason.clone(),
                            usage,
                        });
                        yield ErasedTextTurnEvent::Completed {
                            request_id: request_id.clone(),
                            finish_reason,
                            usage,
                            committed_turn,
                        };
                    }
                    "error" => Err(parse_error_event(&payload))?,
                    other => {
                        trace!("ignoring Claude SSE event {other}");
                    }
                }
            }
        }
    }
}

fn map_structured_stream<S>(
    stream: S,
    fallback_model: String,
) -> impl Stream<Item = Result<ErasedStructuredTurnEvent, ClaudeError>> + Send + 'static
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    try_stream! {
        let mut parser = ClaudeSseParser::default();
        let mut started = false;
        let mut request_id = None::<String>;
        let mut model = fallback_model;
        let mut usage = Usage::zero();
        let mut stop_reason = None::<String>;
        let mut blocks = BTreeMap::<usize, ContentBlockState>::new();
        let mut structured_buffer = String::new();
        let mut emitted_ready = false;
        futures::pin_mut!(stream);

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            for frame in parser.push(&chunk)? {
                if let Some(event) = frame.event.as_deref() {
                    if event == "ping" {
                        trace!("ignoring Claude ping event");
                        continue;
                    }
                }

                let payload = if frame.data.is_empty() {
                    Value::Null
                } else {
                    serde_json::from_str::<Value>(&frame.data)?
                };
                let event_type = frame
                    .event
                    .as_deref()
                    .or_else(|| payload["type"].as_str())
                    .unwrap_or_default();

                if event_type == "message_start" {
                    request_id = payload.pointer("/message/id").and_then(Value::as_str).map(ToOwned::to_owned);
                    if let Some(event_model) = payload.pointer("/message/model").and_then(Value::as_str) {
                        model = event_model.to_string();
                    }
                    usage = parse_claude_usage(payload.pointer("/message/usage").unwrap_or(&Value::Null));
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
                    "content_block_start" => {
                        let index = required_usize(&payload, "/index")?;
                        let kind = required_str(&payload, "/content_block/type")?;
                        let state = match kind {
                            "text" => ContentBlockState::Text {
                                content: payload
                                    .pointer("/content_block/text")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string(),
                            },
                            "thinking" => ContentBlockState::Thinking {
                                content: payload
                                    .pointer("/content_block/thinking")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string(),
                                signature: payload
                                    .pointer("/content_block/signature")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string(),
                            },
                            "tool_use" => ContentBlockState::ToolUse {
                                id: ToolCallId::from(required_str(&payload, "/content_block/id")?),
                                name: ToolName::from(required_str(&payload, "/content_block/name")?),
                                initial_json: payload.pointer("/content_block/input").and_then(non_null_json_string),
                                delta_json: String::new(),
                                finalized_arguments: None,
                            },
                            other => ContentBlockState::Unsupported {
                                kind: other.to_string(),
                            },
                        };
                        blocks.insert(index, state);
                    }
                    "content_block_delta" => {
                        let index = required_usize(&payload, "/index")?;
                        let delta_type = required_str(&payload, "/delta/type")?;
                        let Some(block) = blocks.get_mut(&index) else {
                            Err(ClaudeError::Sse {
                                message: format!("received delta for unknown block index {index}"),
                            })?
                        };
                        match delta_type {
                            "text_delta" => {
                                let delta = required_str(&payload, "/delta/text")?;
                                block.push_text_delta(delta)?;
                                structured_buffer.push_str(delta);
                                yield ErasedStructuredTurnEvent::StructuredOutputChunk {
                                    json_delta: delta.to_string(),
                                };
                            }
                            "thinking_delta" => {
                                let delta = required_str(&payload, "/delta/thinking")?;
                                block.push_thinking_delta(delta)?;
                                yield ErasedStructuredTurnEvent::ReasoningDelta {
                                    delta: delta.to_string(),
                                };
                            }
                            "signature_delta" => {
                                let delta = required_str(&payload, "/delta/signature")?;
                                block.push_signature_delta(delta)?;
                            }
                            "input_json_delta" => {
                                let delta = required_str(&payload, "/delta/partial_json")?;
                                let (id, name) = block.push_tool_delta(delta)?;
                                if !delta.is_empty() {
                                    yield ErasedStructuredTurnEvent::ToolCallChunk {
                                        id,
                                        name,
                                        arguments_json_delta: delta.to_string(),
                                    };
                                }
                            }
                            "citations_delta" => {}
                            other => {
                                trace!("ignoring Claude content delta type {other}");
                            }
                        }
                    }
                    "content_block_stop" => {
                        let index = required_usize(&payload, "/index")?;
                        if let Some(metadata) = blocks
                            .get_mut(&index)
                            .ok_or_else(|| ClaudeError::Sse {
                                message: format!("received stop for unknown block index {index}"),
                            })?
                            .finalize_tool_call()?
                        {
                            yield ErasedStructuredTurnEvent::ToolCallReady(metadata);
                        }
                    }
                    "message_delta" => {
                        stop_reason = payload.pointer("/delta/stop_reason").and_then(Value::as_str).map(ToOwned::to_owned);
                        usage = parse_claude_usage(payload.get("usage").unwrap_or(&Value::Null));
                    }
                    "message_stop" => {
                        if !emitted_ready && !structured_buffer.trim().is_empty() {
                            emitted_ready = true;
                            let value = RawJson::parse(structured_buffer.clone())
                                .map_err(ClaudeError::StructuredOutput)?;
                            yield ErasedStructuredTurnEvent::StructuredOutputReady(value);
                        }

                        let finish_reason = map_claude_stop_reason(stop_reason.as_deref());
                        let committed_turn = Arc::new(ClaudeCommittedTurn {
                            request_id: request_id.clone(),
                            model: model.clone(),
                            items: finalize_committed_items(&mut blocks, &finish_reason)?,
                            finish_reason: finish_reason.clone(),
                            usage,
                        });
                        yield ErasedStructuredTurnEvent::Completed {
                            request_id: request_id.clone(),
                            finish_reason,
                            usage,
                            committed_turn,
                        };
                    }
                    "error" => Err(parse_error_event(&payload))?,
                    other => {
                        trace!("ignoring Claude SSE event {other}");
                    }
                }
            }
        }
    }
}

fn finalize_committed_items(
    blocks: &mut BTreeMap<usize, ContentBlockState>,
    finish_reason: &FinishReason,
) -> Result<Vec<ClaudeTurnItem>, ClaudeError> {
    let refusal_mode = matches!(finish_reason, FinishReason::ContentFilter);
    let mut items = Vec::new();

    for (_index, mut block) in std::mem::take(blocks) {
        match &mut block {
            ContentBlockState::Text { content } => {
                if content.is_empty() {
                    continue;
                }
                if refusal_mode {
                    items.push(ClaudeTurnItem::Refusal {
                        content: content.clone(),
                    });
                } else {
                    items.push(ClaudeTurnItem::Text {
                        content: content.clone(),
                    });
                }
            }
            ContentBlockState::Thinking { content, signature } => {
                if !content.is_empty() {
                    if signature.is_empty() {
                        items.push(ClaudeTurnItem::Reasoning {
                            content: content.clone(),
                        });
                    } else {
                        items.push(ClaudeTurnItem::Thinking {
                            content: content.clone(),
                            signature: signature.clone(),
                        });
                    }
                }
            }
            ContentBlockState::ToolUse { .. } => {
                if let Some(metadata) = block.finalize_tool_call()? {
                    items.push(ClaudeTurnItem::ToolCall {
                        id: metadata.id,
                        name: metadata.name,
                        arguments: metadata.arguments,
                    });
                }
            }
            ContentBlockState::Unsupported { kind } => {
                trace!("ignoring unsupported Claude content block kind {kind}");
            }
        }
    }

    Ok(items)
}

fn parse_error_event(payload: &Value) -> ClaudeError {
    let message = payload
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or("Claude returned an error event")
        .to_string();
    ClaudeError::Sse { message }
}

fn parse_claude_usage(value: &Value) -> Usage {
    let input_tokens = value["input_tokens"].as_u64().unwrap_or_default()
        + value["cache_creation_input_tokens"]
            .as_u64()
            .unwrap_or_default()
        + value["cache_read_input_tokens"]
            .as_u64()
            .unwrap_or_default();
    let output_tokens = value["output_tokens"].as_u64().unwrap_or_default();

    Usage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens + output_tokens,
        cost_micros_usd: 0,
    }
}

fn map_claude_stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn") | Some("stop_sequence") | None => FinishReason::Stop,
        Some("max_tokens") => FinishReason::Length,
        Some("tool_use") => FinishReason::ToolCall,
        Some("refusal") => FinishReason::ContentFilter,
        Some(other) => FinishReason::Unknown(other.to_string()),
    }
}

fn extract_completion_text(value: &Value) -> String {
    let mut text = String::new();
    for block in value["content"].as_array().into_iter().flatten() {
        if block["type"].as_str() == Some("text")
            && let Some(delta) = block["text"].as_str()
        {
            text.push_str(delta);
        }
    }
    text
}

fn required_str<'a>(value: &'a Value, pointer: &'static str) -> Result<&'a str, ClaudeError> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or(ClaudeError::MissingField { field: pointer })
}

fn required_usize(value: &Value, pointer: &'static str) -> Result<usize, ClaudeError> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .ok_or(ClaudeError::MissingField { field: pointer })
}

fn non_null_json_string(value: &Value) -> Option<String> {
    if value.is_null() {
        None
    } else {
        serde_json::to_string(value).ok()
    }
}

#[derive(Default)]
struct ClaudeSseParser {
    buffer: Vec<u8>,
    event: Option<String>,
    data_lines: Vec<String>,
}

#[derive(Debug)]
struct ClaudeSseFrame {
    event: Option<String>,
    data: String,
}

impl ClaudeSseParser {
    fn push(&mut self, chunk: &[u8]) -> Result<Vec<ClaudeSseFrame>, ClaudeError> {
        self.buffer.extend_from_slice(chunk);
        let mut frames = Vec::new();

        while let Some(pos) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let mut line = self.buffer.drain(..=pos).collect::<Vec<_>>();
            if line.last() == Some(&b'\n') {
                line.pop();
            }
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line = String::from_utf8(line).map_err(|err| ClaudeError::Sse {
                message: format!("invalid UTF-8 in SSE stream: {err}"),
            })?;

            if line.is_empty() {
                self.finish_frame(&mut frames);
                continue;
            }
            if line.starts_with(':') {
                continue;
            }

            let (field, value) = line
                .split_once(':')
                .map_or((line.as_str(), ""), |(field, value)| {
                    (field, value.strip_prefix(' ').unwrap_or(value))
                });
            match field {
                "event" => self.event = Some(value.to_string()),
                "data" => self.data_lines.push(value.to_string()),
                _ => {}
            }
        }

        Ok(frames)
    }

    fn finish_frame(&mut self, frames: &mut Vec<ClaudeSseFrame>) {
        if self.event.is_none() && self.data_lines.is_empty() {
            return;
        }

        frames.push(ClaudeSseFrame {
            event: self.event.take(),
            data: self.data_lines.join("\n"),
        });
        self.data_lines.clear();
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClaudeCommittedTurn {
    pub request_id: Option<String>,
    pub model: String,
    pub items: Vec<ClaudeTurnItem>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClaudeTurnItem {
    Text {
        content: String,
    },
    Thinking {
        content: String,
        signature: String,
    },
    Reasoning {
        content: String,
    },
    Refusal {
        content: String,
    },
    ToolCall {
        id: ToolCallId,
        name: ToolName,
        arguments: RawJson,
    },
}

impl TurnView for ClaudeCommittedTurn {
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

impl ItemView for ClaudeTurnItem {
    fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { content } => Some(content),
            _ => None,
        }
    }

    fn as_reasoning(&self) -> Option<&str> {
        match self {
            Self::Thinking { content, .. } => Some(content),
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
        None
    }
}
