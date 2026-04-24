use std::{
    collections::{BTreeMap, HashMap},
    env,
    pin::Pin,
    sync::{Arc, Mutex},
    time::Duration,
};

use async_stream::try_stream;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use lutum_protocol::{
    AgentError,
    budget::Usage,
    conversation::{
        AssistantInputItem, InputMessageRole, MessageContent, ModelInput, ModelInputItem, RawJson,
        ToolCallId, ToolMetadata, ToolName, ToolResult,
    },
    extensions::RequestExtensions,
    llm::{
        AdapterStructuredTurn, AdapterTextTurn, AdapterToolChoice, AdapterTurnConfig,
        ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
        ErasedTextTurnEventStream, FinishReason, ModelName, OperationKind, TurnAdapter,
        UsageRecoveryAdapter,
    },
    telemetry::{ParseErrorStage, RawTelemetryEmitter, RequestErrorDebugInfo, RequestErrorKind},
    transcript::{ToolResultItemView, TurnRole, TurnView},
};
use reqwest::{
    Client,
    header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue, RETRY_AFTER},
};
use serde::Serialize;
use serde_json::Value;
use tracing::trace;

use crate::{
    error::ClaudeError,
    messages::{
        ClaudeCommittedTurn, ClaudeContentBlock, ClaudeMessage, ClaudeRole, ClaudeTool,
        ClaudeToolChoice, ClaudeTurnItem, ContentBlockDeltaEvent, ContentBlockStartEvent,
        ContentBlockStopEvent, ErrorEvent, MessageDeltaEvent, MessageStartEvent, MessagesRequest,
        OutputConfig, OutputFormat, SseContentBlock, SseContentDelta, SseEvent, SystemBlock,
        TextBlock, ThinkingBlock, ThinkingConfig, ThinkingKind, ToolResultBlock, ToolUseBlock,
    },
    sse::ClaudeSseParser,
};

const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const MIN_THINKING_BUDGET_TOKENS: u32 = 1024;
const MIN_RESPONSE_TOKENS_WITH_THINKING: u32 = 1024;

#[cfg(not(target_family = "wasm"))]
type ByteStream =
    Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + Sync + 'static>>;
#[cfg(target_family = "wasm")]
type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + 'static>>;
type UsageCache = Arc<Mutex<HashMap<String, Usage>>>;

pub trait FallbackSerializer: Send + Sync {
    fn apply(&self, request: &mut MessagesRequest);
}

/// Built-in prompt-caching serializer.
///
/// Applies `cache_control: { type: "ephemeral" }` at three cache breakpoints:
///   1. Last system block  — covers the system prompt prefix
///   2. Last tool          — covers system + all tool definitions (static across turns)
///   3. Last content block of the second-to-last message — covers conversation history
struct BuiltinCacheSerializer;

impl FallbackSerializer for BuiltinCacheSerializer {
    fn apply(&self, request: &mut MessagesRequest) {
        use crate::messages::{CacheControl, ClaudeContentBlock};

        // 1. Last system block
        if let Some(blocks) = request.system.as_mut()
            && let Some(last) = blocks.last_mut()
        {
            last.cache_control = Some(CacheControl::ephemeral());
        }

        // 2. Last tool definition
        if let Some(tools) = request.tools.as_mut()
            && let Some(last) = tools.last_mut()
        {
            last.cache_control = Some(CacheControl::ephemeral());
        }

        // 3. Last content block of the second-to-last message (penultimate turn)
        let n = request.messages.len();
        if n >= 2 {
            let msg = &mut request.messages[n - 2];
            if let Some(block) = msg.content.last_mut() {
                match block {
                    ClaudeContentBlock::Text(b) => {
                        b.cache_control = Some(CacheControl::ephemeral())
                    }
                    ClaudeContentBlock::ToolResult(b) => {
                        b.cache_control = Some(CacheControl::ephemeral())
                    }
                    _ => {}
                }
            }
        }
    }
}

#[lutum_macros::hooks]
pub trait ClaudeHooks {
    #[hook(singleton)]
    async fn select_claude_model(_extensions: &RequestExtensions, default: ModelName) -> ModelName {
        default
    }

    #[hook(singleton)]
    async fn resolve_budget_tokens(
        _extensions: &RequestExtensions,
        default: Option<u32>,
    ) -> Option<u32> {
        default
    }
}

fn serialize_raw_body<T: Serialize>(body: &T) -> Result<String, ClaudeError> {
    serde_json::to_string(body).map_err(ClaudeError::Json)
}

fn emit_claude_parse_error(
    raw: Option<&RawTelemetryEmitter>,
    request_id: Option<&str>,
    stage: ParseErrorStage,
    payload: &str,
    error: &impl std::fmt::Display,
) {
    if let Some(raw) = raw {
        raw.emit_parse_error(request_id, stage, payload, &error.to_string());
    }
}

fn emit_claude_request_error(
    raw: Option<&RawTelemetryEmitter>,
    request_id: Option<&str>,
    request_error_kind: RequestErrorKind,
    status: Option<reqwest::StatusCode>,
    payload: Option<&str>,
    error: &str,
    debug_info: &RequestErrorDebugInfo,
) {
    if let Some(raw) = raw {
        raw.emit_request_error(
            request_id,
            request_error_kind,
            status.map(|status| status.as_u16()),
            payload,
            error,
            debug_info,
        );
    }
}

fn reqwest_request_error_debug_info(error: &reqwest::Error) -> RequestErrorDebugInfo {
    let mut source_chain = Vec::new();
    let mut current = std::error::Error::source(error);
    while let Some(source) = current {
        source_chain.push(source.to_string());
        current = source.source();
    }

    RequestErrorDebugInfo {
        error_debug: format!("{error:?}"),
        source_chain,
        is_timeout: error.is_timeout(),
        #[cfg(not(target_family = "wasm"))]
        is_connect: error.is_connect(),
        #[cfg(target_family = "wasm")]
        is_connect: false,
        is_request: error.is_request(),
        is_body: error.is_body(),
        is_decode: error.is_decode(),
    }
}

fn basic_request_error_debug_info(error: &impl std::fmt::Debug) -> RequestErrorDebugInfo {
    RequestErrorDebugInfo {
        error_debug: format!("{error:?}"),
        ..RequestErrorDebugInfo::default()
    }
}

fn retry_after_from_headers(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

fn emit_claude_stream_error(
    raw: Option<&RawTelemetryEmitter>,
    request_id: Option<&str>,
    source: reqwest::Error,
) -> ClaudeError {
    let debug_info = reqwest_request_error_debug_info(&source);
    let error = ClaudeError::Request(source);
    emit_claude_request_error(
        raw,
        request_id,
        RequestErrorKind::Transport,
        None,
        None,
        &error.to_string(),
        &debug_info,
    );
    error
}

#[derive(Clone)]
pub struct ClaudeAdapter {
    client: Arc<Client>,
    api_key: Arc<str>,
    base_url: Arc<str>,
    default_model: ModelName,
    default_thinking_budget: Option<u32>,
    hooks: ClaudeHooksSet<'static>,
    fallback_serializer: Option<Arc<dyn FallbackSerializer>>,
    usage_cache: UsageCache,
}

#[derive(Default)]
struct CompiledClaudeConversation {
    system: Vec<SystemBlock>,
    messages: Vec<ClaudeMessage>,
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
    Unsupported,
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
            default_model: ModelName::new("claude-opus-4-5").unwrap(),
            default_thinking_budget: None,
            hooks: ClaudeHooksSet::new(),
            fallback_serializer: None,
            usage_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = normalize_base_url(base_url.into());
        self
    }

    pub fn with_default_model(mut self, model: ModelName) -> Self {
        self.default_model = model;
        self
    }

    pub fn with_default_thinking_budget(mut self, budget_tokens: u32) -> Self {
        self.default_thinking_budget = Some(budget_tokens);
        self
    }

    pub fn with_hooks(mut self, hooks: ClaudeHooksSet<'static>) -> Self {
        self.hooks = hooks;
        self
    }

    pub fn set_hooks(&mut self, hooks: ClaudeHooksSet<'static>) {
        self.hooks = hooks;
    }

    pub fn with_select_claude_model(mut self, hook: impl SelectClaudeModel + 'static) -> Self {
        self.hooks = self.hooks.with_select_claude_model(hook);
        self
    }

    pub fn set_select_claude_model(&mut self, hook: impl SelectClaudeModel + 'static) {
        self.hooks.register_select_claude_model(hook);
    }

    pub fn with_resolve_budget_tokens(mut self, hook: impl ResolveBudgetTokens + 'static) -> Self {
        self.hooks = self.hooks.with_resolve_budget_tokens(hook);
        self
    }

    pub fn set_resolve_budget_tokens(&mut self, hook: impl ResolveBudgetTokens + 'static) {
        self.hooks.register_resolve_budget_tokens(hook);
    }

    pub fn set_fallback_serializer(&mut self, s: Box<dyn FallbackSerializer>) {
        self.fallback_serializer = Some(s.into());
    }

    /// Enable the built-in 3-breakpoint prompt-caching strategy.
    ///
    /// Cache breakpoints are inserted at: the last system block, the last tool
    /// definition, and the last content block of the penultimate message. This
    /// covers the system prompt, tool schema, and conversation history —
    /// the three parts of the context that grow slowly across turns.
    pub fn with_prompt_caching(mut self) -> Self {
        self.set_fallback_serializer(Box::new(BuiltinCacheSerializer));
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

    fn prepare_messages_request(
        &self,
        input: &ModelInput,
        config: &AdapterTurnConfig,
        model: &str,
        thinking_budget: Option<u32>,
        format: Option<OutputFormat>,
        stream: bool,
    ) -> Result<MessagesRequest, ClaudeError> {
        let mut request =
            build_messages_request(input, config, model, thinking_budget, format, stream)?;
        if let Some(serializer) = self.fallback_serializer.as_ref() {
            serializer.apply(&mut request);
        }
        tracing::debug!(
            system_blocks = request.system.as_ref().map(|s| s.len()).unwrap_or(0),
            last_block_has_cache_control = request
                .system
                .as_ref()
                .and_then(|s| s.last())
                .and_then(|b| b.cache_control.as_ref())
                .is_some(),
            tools = request.tools.as_ref().map(|t| t.len()).unwrap_or(0),
            last_tool_has_cache_control = request
                .tools
                .as_ref()
                .and_then(|t| t.last())
                .and_then(|t| t.cache_control.as_ref())
                .is_some(),
            messages = request.messages.len(),
            "prepared MessagesRequest",
        );
        Ok(request)
    }

    async fn send_streaming_json<T>(
        &self,
        path: &str,
        body: &T,
        raw: Option<&RawTelemetryEmitter>,
    ) -> Result<ByteStream, ClaudeError>
    where
        T: Serialize + ?Sized,
    {
        let response = match self
            .client
            .post(format!("{}{}", self.base_url, path))
            .headers(self.request_headers()?)
            .json(body)
            .send()
            .await
        {
            Ok(response) => response,
            Err(source) => {
                let debug_info = reqwest_request_error_debug_info(&source);
                let error = ClaudeError::Request(source);
                emit_claude_request_error(
                    raw,
                    None,
                    RequestErrorKind::Transport,
                    None,
                    None,
                    &error.to_string(),
                    &debug_info,
                );
                return Err(error);
            }
        };
        let response = error_for_status_with_body(raw, response).await?;
        Ok(Box::pin(response.bytes_stream()))
    }
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
impl TurnAdapter for ClaudeAdapter {
    async fn text_turn(
        &self,
        input: ModelInput,
        turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        let raw =
            RawTelemetryEmitter::new(turn.extensions.as_ref(), "claude", "messages", "text_turn");
        let model = self
            .hooks
            .select_claude_model(turn.extensions.as_ref(), self.default_model.clone())
            .await;
        let thinking_budget = self
            .hooks
            .resolve_budget_tokens(turn.extensions.as_ref(), self.default_thinking_budget)
            .await
            .map(|budget| budget.max(MIN_THINKING_BUDGET_TOKENS));
        let body = self
            .prepare_messages_request(
                &input,
                &turn.config,
                model.as_ref(),
                thinking_budget,
                None,
                true,
            )
            .map_err(AgentError::from)?;
        if let Some(raw) = raw.as_ref() {
            raw.emit_request(None, &serialize_raw_body(&body).map_err(AgentError::from)?);
        }
        let stream = self
            .send_streaming_json("/v1/messages", &body, raw.as_ref())
            .await
            .map_err(AgentError::from)?;
        Ok(Box::pin(
            map_text_stream(stream, model.into(), Arc::clone(&self.usage_cache), raw)
                .map(|item| item.map_err(AgentError::from)),
        ) as ErasedTextTurnEventStream)
    }

    async fn structured_turn(
        &self,
        input: ModelInput,
        turn: AdapterStructuredTurn,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
        let raw = RawTelemetryEmitter::new(
            turn.extensions.as_ref(),
            "claude",
            "messages",
            "structured_turn",
        );
        let model = self
            .hooks
            .select_claude_model(turn.extensions.as_ref(), self.default_model.clone())
            .await;
        let thinking_budget = self
            .hooks
            .resolve_budget_tokens(turn.extensions.as_ref(), self.default_thinking_budget)
            .await
            .map(|budget| budget.max(MIN_THINKING_BUDGET_TOKENS));
        let mut schema = turn.output.schema.clone();
        require_no_additional_properties(&mut schema);
        let format = Some(OutputFormat::JsonSchema { schema });
        let body = self
            .prepare_messages_request(
                &input,
                &turn.config,
                model.as_ref(),
                thinking_budget,
                format,
                true,
            )
            .map_err(AgentError::from)?;
        if let Some(raw) = raw.as_ref() {
            raw.emit_request(None, &serialize_raw_body(&body).map_err(AgentError::from)?);
        }
        let stream = self
            .send_streaming_json("/v1/messages", &body, raw.as_ref())
            .await
            .map_err(AgentError::from)?;
        Ok(Box::pin(
            map_structured_stream(stream, model.into(), Arc::clone(&self.usage_cache), raw)
                .map(|item| item.map_err(AgentError::from)),
        ) as ErasedStructuredTurnEventStream)
    }
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
impl UsageRecoveryAdapter for ClaudeAdapter {
    async fn recover_usage(
        &self,
        kind: OperationKind,
        request_id: &str,
    ) -> Result<Option<Usage>, AgentError> {
        match kind {
            OperationKind::TextTurn
            | OperationKind::StructuredTurn
            | OperationKind::StructuredCompletion => {
                let usage = take_cached_usage(&self.usage_cache, request_id);
                trace!(
                    ?kind,
                    request_id,
                    ?usage,
                    "recovered Claude usage from cache"
                );
                Ok(usage)
            }
            OperationKind::Completion => Ok(None),
        }
    }
}

impl CompiledClaudeConversation {
    fn push_system_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.system.push(SystemBlock::new(text));
    }

    fn push_block(&mut self, role: ClaudeRole, block: ClaudeContentBlock) {
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

    fn into_messages(self) -> Vec<ClaudeMessage> {
        self.messages
            .into_iter()
            .filter(|message| !message.content.is_empty())
            .collect()
    }
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
            // Text and Thinking blocks are finalized via flush_blocks; nothing to yield here.
            Self::Unsupported | Self::Text { .. } | Self::Thinking { .. } => Ok(None),
        }
    }
}

fn build_messages_request(
    input: &ModelInput,
    config: &AdapterTurnConfig,
    model: &str,
    thinking_budget: Option<u32>,
    format: Option<OutputFormat>,
    stream: bool,
) -> Result<MessagesRequest, ClaudeError> {
    let compiled = compile_model_input(input)?;
    let system = (!compiled.system.is_empty()).then_some(compiled.system.clone());
    let max_tokens = resolve_max_tokens(config.generation.max_output_tokens, thinking_budget);
    let tools = (!config.tools.is_empty()).then(|| build_tool_definitions(config));

    Ok(MessagesRequest {
        model: model.to_string(),
        max_tokens,
        messages: compiled.into_messages(),
        stream: Some(stream),
        system,
        temperature: config
            .generation
            .temperature
            .map(|temperature| temperature.get()),
        tools,
        tool_choice: build_tool_choice(config),
        thinking: thinking_budget.map(|budget_tokens| ThinkingConfig {
            kind: ThinkingKind::Enabled,
            budget_tokens,
        }),
        output_config: format.map(|format| OutputConfig { format }),
        stop_sequences: None,
        models: None,
    })
}

fn resolve_max_tokens(explicit: Option<u32>, thinking_budget: Option<u32>) -> u32 {
    let base = explicit.unwrap_or(DEFAULT_MAX_TOKENS).max(1);
    if let Some(budget) = thinking_budget {
        return base.max(budget.saturating_add(MIN_RESPONSE_TOKENS_WITH_THINKING));
    }
    base
}

fn build_tool_definitions(config: &AdapterTurnConfig) -> Vec<ClaudeTool> {
    config
        .tools
        .iter()
        .map(|tool| ClaudeTool {
            name: tool.name.clone(),
            description: Some(tool.description.clone()),
            input_schema: tool.input_schema.clone(),
            cache_control: None,
        })
        .collect()
}

fn build_tool_choice(config: &AdapterTurnConfig) -> Option<ClaudeToolChoice> {
    match &config.tool_choice {
        AdapterToolChoice::None => None,
        AdapterToolChoice::Auto => Some(ClaudeToolChoice::Auto {
            disable_parallel_tool_use: None,
        }),
        AdapterToolChoice::Required => {
            if config.tools.len() == 1 {
                Some(ClaudeToolChoice::Tool {
                    name: config.tools[0].name.clone(),
                    disable_parallel_tool_use: None,
                })
            } else {
                Some(ClaudeToolChoice::Any {
                    disable_parallel_tool_use: None,
                })
            }
        }
        AdapterToolChoice::Specific(name) => Some(ClaudeToolChoice::Tool {
            name: name.clone(),
            disable_parallel_tool_use: None,
        }),
    }
}

fn compile_model_input(input: &ModelInput) -> Result<CompiledClaudeConversation, ClaudeError> {
    let mut compiled = CompiledClaudeConversation::default();
    // Track whether the immediately preceding item was a committed Turn that contained
    // tool calls. When true, a following ToolResult item must only emit the user-side
    // tool_result — the assistant-side tool_use blocks are already in the Turn.
    let mut prev_was_tool_turn = false;

    for item in input.items() {
        match item {
            ModelInputItem::Message { role, content } => {
                emit_message(role, content.iter(), &mut compiled)?;
                prev_was_tool_turn = false;
            }
            ModelInputItem::Assistant(item) => {
                compiled.push_block(ClaudeRole::Assistant, assistant_replay_block(item));
                prev_was_tool_turn = false;
            }
            ModelInputItem::ToolResult(tool_result) => {
                if prev_was_tool_turn {
                    // The assistant's tool_use block was already emitted by the preceding
                    // committed Turn. Only emit the user-side tool_result.
                    emit_tool_result(tool_result, &mut compiled)?;
                } else {
                    emit_tool_use(tool_result, &mut compiled)?;
                }
                // ToolResult items form a contiguous group after a Turn; keep the flag set.
            }
            ModelInputItem::Turn(turn) => {
                let has_tool_calls;
                if let Some(claude_turn) =
                    turn.as_ref().as_any().downcast_ref::<ClaudeCommittedTurn>()
                {
                    has_tool_calls = claude_turn
                        .items
                        .iter()
                        .any(|i| matches!(i, ClaudeTurnItem::ToolCall { .. }));
                    emit_claude_turn_exact(claude_turn, &mut compiled)?;
                } else {
                    has_tool_calls = (0..turn.item_count())
                        .filter_map(|i| turn.item_at(i))
                        .any(|v| v.as_tool_call().is_some());
                    emit_turn_from_view(turn.as_ref(), &mut compiled)?;
                }
                prev_was_tool_turn = has_tool_calls;
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

/// Emit a standalone tool call+result pair (no preceding committed Turn).
fn emit_tool_use(
    tool_result: &ToolResult,
    compiled: &mut CompiledClaudeConversation,
) -> Result<(), ClaudeError> {
    compiled.push_block(
        ClaudeRole::Assistant,
        tool_use_block(&tool_result.id, &tool_result.name, &tool_result.arguments)?,
    );
    compiled.push_block(ClaudeRole::User, tool_result_block(tool_result)?);
    Ok(())
}

/// Emit only the user-side tool_result for a ToolResult that follows a committed Turn.
/// The assistant-side tool_use block is already present from the Turn replay.
fn emit_tool_result(
    tool_result: &ToolResult,
    compiled: &mut CompiledClaudeConversation,
) -> Result<(), ClaudeError> {
    compiled.push_block(ClaudeRole::User, tool_result_block(tool_result)?);
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
                    ClaudeContentBlock::Thinking(ThinkingBlock {
                        thinking: content.clone(),
                        signature: signature.clone(),
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

fn message_content_block(content: &MessageContent) -> ClaudeContentBlock {
    match content {
        MessageContent::Text(text) => text_block(text),
    }
}

fn assistant_replay_block(item: &AssistantInputItem) -> ClaudeContentBlock {
    match item {
        AssistantInputItem::Text(text)
        | AssistantInputItem::Reasoning(text)
        | AssistantInputItem::Refusal(text) => text_block(text),
    }
}

fn text_block(text: &str) -> ClaudeContentBlock {
    ClaudeContentBlock::Text(TextBlock {
        text: text.to_string(),
        cache_control: None,
    })
}

fn tool_use_block(
    id: &ToolCallId,
    name: &ToolName,
    arguments: &RawJson,
) -> Result<ClaudeContentBlock, ClaudeError> {
    Ok(ClaudeContentBlock::ToolUse(ToolUseBlock {
        id: id.clone(),
        name: name.clone(),
        input: parse_tool_input(arguments)?,
    }))
}

fn tool_result_block(tool_result: &ToolResult) -> Result<ClaudeContentBlock, ClaudeError> {
    Ok(ClaudeContentBlock::ToolResult(ToolResultBlock {
        tool_use_id: tool_result.id.clone(),
        content: tool_result_content(&tool_result.result)?,
        cache_control: None,
    }))
}

fn tool_result_block_from_view(
    tool_result: ToolResultItemView<'_>,
) -> Result<ClaudeContentBlock, ClaudeError> {
    Ok(ClaudeContentBlock::ToolResult(ToolResultBlock {
        tool_use_id: tool_result.id.clone(),
        content: tool_result_content(tool_result.result)?,
        cache_control: None,
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

fn map_text_stream<S>(
    stream: S,
    fallback_model: String,
    usage_cache: UsageCache,
    raw: Option<RawTelemetryEmitter>,
) -> impl Stream<Item = Result<ErasedTextTurnEvent, ClaudeError>> + lutum_protocol::MaybeSend + 'static
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + lutum_protocol::MaybeSend + 'static,
{
    try_stream! {
        let mut parser = ClaudeSseParser::default();
        let mut started = false;
        let mut request_id = None::<String>;
        let mut model = fallback_model;
        let mut usage = Usage::zero();
        let mut cache_creation_tokens = 0u64;
        let mut cache_read_tokens = 0u64;
        let mut stop_reason = None::<String>;
        let mut blocks = BTreeMap::<usize, ContentBlockState>::new();
        let mut raw_sequence = 0_u64;
        futures::pin_mut!(stream);

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(source) => Err(emit_claude_stream_error(
                    raw.as_ref(),
                    request_id.as_deref(),
                    source,
                ))?,
            };
            for frame in parser.push(&chunk)? {
                raw_sequence += 1;
                if let Some(raw) = raw.as_ref() {
                    raw.emit_stream_event(
                        request_id.as_deref(),
                        raw_sequence,
                        &frame.data,
                        frame.event.as_deref(),
                    );
                }
                let event = match parse_sse_event(&frame.event, &frame.data) {
                    Ok(event) => event,
                    Err(err) => {
                        emit_claude_parse_error(
                            raw.as_ref(),
                            request_id.as_deref(),
                            ParseErrorStage::SseParse,
                            &frame.data,
                            &err,
                        );
                        Err(err)?
                    }
                };

                if matches!(event, SseEvent::Ping(_)) {
                    trace!("ignoring Claude ping event");
                    continue;
                }

                if let SseEvent::MessageStart(MessageStartEvent { message }) = &event {
                    request_id = Some(message.id.clone());
                    model = message.model.clone();
                    cache_creation_tokens += message.usage.cache_creation_input_tokens.unwrap_or(0);
                    cache_read_tokens += message.usage.cache_read_input_tokens.unwrap_or(0);
                    tracing::debug!(
                        input_tokens = message.usage.input_tokens,
                        cache_creation = message.usage.cache_creation_input_tokens,
                        cache_read = message.usage.cache_read_input_tokens,
                        "message_start usage",
                    );
                    usage = message.usage.clone().into_protocol_usage();
                    cache_usage(&usage_cache, &message.id, usage);
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

                match event {
                    SseEvent::ContentBlockStart(ContentBlockStartEvent {
                        index,
                        content_block,
                    }) => {
                        blocks.insert(index, start_block_state(content_block));
                    }
                    SseEvent::ContentBlockDelta(ContentBlockDeltaEvent { index, delta }) => {
                        let Some(block) = blocks.get_mut(&index) else {
                            Err(ClaudeError::Sse {
                                message: format!("received delta for unknown block index {index}"),
                            })?
                        };
                        match delta {
                            SseContentDelta::TextDelta { text } => {
                                block.push_text_delta(&text)?;
                                yield ErasedTextTurnEvent::TextDelta { delta: text };
                            }
                            SseContentDelta::ThinkingDelta { thinking } => {
                                block.push_thinking_delta(&thinking)?;
                                yield ErasedTextTurnEvent::ReasoningDelta { delta: thinking };
                            }
                            SseContentDelta::SignatureDelta { signature } => {
                                block.push_signature_delta(&signature)?;
                            }
                            SseContentDelta::InputJsonDelta { partial_json } => {
                                let (id, name) = block.push_tool_delta(&partial_json)?;
                                if !partial_json.is_empty() {
                                    yield ErasedTextTurnEvent::ToolCallChunk {
                                        id,
                                        name,
                                        arguments_json_delta: partial_json,
                                    };
                                }
                            }
                            SseContentDelta::CitationsDelta { .. } => {}
                        }
                    }
                    SseEvent::ContentBlockStop(ContentBlockStopEvent { index }) => {
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
                    SseEvent::MessageDelta(MessageDeltaEvent { delta, usage: next_usage }) => {
                        stop_reason = delta.stop_reason;
                        cache_creation_tokens += next_usage.cache_creation_input_tokens.unwrap_or(0);
                        cache_read_tokens += next_usage.cache_read_input_tokens.unwrap_or(0);
                        usage = next_usage.into_protocol_usage();
                        if let Some(request_id) = request_id.as_deref() {
                            cache_usage(&usage_cache, request_id, usage);
                        }
                    }
                    SseEvent::MessageStop(_) => {
                        let finish_reason = map_claude_stop_reason(stop_reason.as_deref());
                        let final_usage = Usage {
                            cache_creation_tokens,
                            cache_read_tokens,
                            ..usage
                        };
                        let committed_turn = Arc::new(ClaudeCommittedTurn {
                            request_id: request_id.clone(),
                            model: model.clone(),
                            items: finalize_committed_items(&mut blocks, &finish_reason)?,
                            finish_reason: finish_reason.clone(),
                            usage: final_usage,
                            cache_creation_input_tokens: cache_creation_tokens,
                            cache_read_input_tokens: cache_read_tokens,
                        });
                        if let Some(request_id) = request_id.as_deref() {
                            remove_cached_usage(&usage_cache, request_id);
                        }
                        yield ErasedTextTurnEvent::Completed {
                            request_id: request_id.clone(),
                            finish_reason,
                            usage: final_usage,
                            committed_turn,
                        };
                    }
                    SseEvent::Error(error) => Err(parse_error_event(error))?,
                    SseEvent::Ping(_) | SseEvent::MessageStart(_) => {}
                }
            }
        }
    }
}

fn map_structured_stream<S>(
    stream: S,
    fallback_model: String,
    usage_cache: UsageCache,
    raw: Option<RawTelemetryEmitter>,
) -> impl Stream<Item = Result<ErasedStructuredTurnEvent, ClaudeError>>
+ lutum_protocol::MaybeSend
+ 'static
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + lutum_protocol::MaybeSend + 'static,
{
    try_stream! {
        let mut parser = ClaudeSseParser::default();
        let mut started = false;
        let mut request_id = None::<String>;
        let mut model = fallback_model;
        let mut usage = Usage::zero();
        let mut cache_creation_tokens = 0u64;
        let mut cache_read_tokens = 0u64;
        let mut stop_reason = None::<String>;
        let mut blocks = BTreeMap::<usize, ContentBlockState>::new();
        let mut structured_buffer = String::new();
        let mut emitted_ready = false;
        let mut raw_sequence = 0_u64;
        futures::pin_mut!(stream);

        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(source) => Err(emit_claude_stream_error(
                    raw.as_ref(),
                    request_id.as_deref(),
                    source,
                ))?,
            };
            for frame in parser.push(&chunk)? {
                raw_sequence += 1;
                if let Some(raw) = raw.as_ref() {
                    raw.emit_stream_event(
                        request_id.as_deref(),
                        raw_sequence,
                        &frame.data,
                        frame.event.as_deref(),
                    );
                }
                let event = match parse_sse_event(&frame.event, &frame.data) {
                    Ok(event) => event,
                    Err(err) => {
                        emit_claude_parse_error(
                            raw.as_ref(),
                            request_id.as_deref(),
                            ParseErrorStage::SseParse,
                            &frame.data,
                            &err,
                        );
                        Err(err)?
                    }
                };

                if matches!(event, SseEvent::Ping(_)) {
                    trace!("ignoring Claude ping event");
                    continue;
                }

                if let SseEvent::MessageStart(MessageStartEvent { message }) = &event {
                    request_id = Some(message.id.clone());
                    model = message.model.clone();
                    cache_creation_tokens += message.usage.cache_creation_input_tokens.unwrap_or(0);
                    cache_read_tokens += message.usage.cache_read_input_tokens.unwrap_or(0);
                    tracing::debug!(
                        input_tokens = message.usage.input_tokens,
                        cache_creation = message.usage.cache_creation_input_tokens,
                        cache_read = message.usage.cache_read_input_tokens,
                        "message_start usage",
                    );
                    usage = message.usage.clone().into_protocol_usage();
                    cache_usage(&usage_cache, &message.id, usage);
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

                match event {
                    SseEvent::ContentBlockStart(ContentBlockStartEvent {
                        index,
                        content_block,
                    }) => {
                        blocks.insert(index, start_block_state(content_block));
                    }
                    SseEvent::ContentBlockDelta(ContentBlockDeltaEvent { index, delta }) => {
                        let Some(block) = blocks.get_mut(&index) else {
                            Err(ClaudeError::Sse {
                                message: format!("received delta for unknown block index {index}"),
                            })?
                        };
                        match delta {
                            SseContentDelta::TextDelta { text } => {
                                block.push_text_delta(&text)?;
                                structured_buffer.push_str(&text);
                                yield ErasedStructuredTurnEvent::StructuredOutputChunk {
                                    json_delta: text,
                                };
                            }
                            SseContentDelta::ThinkingDelta { thinking } => {
                                block.push_thinking_delta(&thinking)?;
                                yield ErasedStructuredTurnEvent::ReasoningDelta { delta: thinking };
                            }
                            SseContentDelta::SignatureDelta { signature } => {
                                block.push_signature_delta(&signature)?;
                            }
                            SseContentDelta::InputJsonDelta { partial_json } => {
                                let (id, name) = block.push_tool_delta(&partial_json)?;
                                if !partial_json.is_empty() {
                                    yield ErasedStructuredTurnEvent::ToolCallChunk {
                                        id,
                                        name,
                                        arguments_json_delta: partial_json,
                                    };
                                }
                            }
                            SseContentDelta::CitationsDelta { .. } => {}
                        }
                    }
                    SseEvent::ContentBlockStop(ContentBlockStopEvent { index }) => {
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
                    SseEvent::MessageDelta(MessageDeltaEvent { delta, usage: next_usage }) => {
                        stop_reason = delta.stop_reason;
                        cache_creation_tokens += next_usage.cache_creation_input_tokens.unwrap_or(0);
                        cache_read_tokens += next_usage.cache_read_input_tokens.unwrap_or(0);
                        usage = next_usage.into_protocol_usage();
                        if let Some(request_id) = request_id.as_deref() {
                            cache_usage(&usage_cache, request_id, usage);
                        }
                    }
                    SseEvent::MessageStop(_) => {
                        if !emitted_ready && !structured_buffer.trim().is_empty() {
                            emitted_ready = true;
                            let value = match RawJson::parse(structured_buffer.clone()) {
                                Ok(value) => value,
                                Err(err) => {
                                    emit_claude_parse_error(
                                        raw.as_ref(),
                                        request_id.as_deref(),
                                        ParseErrorStage::StructuredOutputParse,
                                        &structured_buffer,
                                        &err,
                                    );
                                    Err(ClaudeError::StructuredOutput(err))?
                                }
                            };
                            yield ErasedStructuredTurnEvent::StructuredOutputReady(value);
                        }

                        let finish_reason = map_claude_stop_reason(stop_reason.as_deref());
                        let final_usage = Usage {
                            cache_creation_tokens,
                            cache_read_tokens,
                            ..usage
                        };
                        let committed_turn = Arc::new(ClaudeCommittedTurn {
                            request_id: request_id.clone(),
                            model: model.clone(),
                            items: finalize_committed_items(&mut blocks, &finish_reason)?,
                            finish_reason: finish_reason.clone(),
                            usage: final_usage,
                            cache_creation_input_tokens: cache_creation_tokens,
                            cache_read_input_tokens: cache_read_tokens,
                        });
                        if let Some(request_id) = request_id.as_deref() {
                            remove_cached_usage(&usage_cache, request_id);
                        }
                        yield ErasedStructuredTurnEvent::Completed {
                            request_id: request_id.clone(),
                            finish_reason,
                            usage: final_usage,
                            committed_turn,
                        };
                    }
                    SseEvent::Error(error) => Err(parse_error_event(error))?,
                    SseEvent::Ping(_) | SseEvent::MessageStart(_) => {}
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

    for (_, mut block) in std::mem::take(blocks) {
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
            ContentBlockState::Unsupported => {
                trace!("ignoring unsupported Claude content block");
            }
        }
    }

    Ok(items)
}

fn normalize_base_url(url: impl Into<String>) -> Arc<str> {
    Arc::from(url.into().trim_end_matches('/').to_string())
}

async fn error_for_status_with_body(
    raw: Option<&RawTelemetryEmitter>,
    response: reqwest::Response,
) -> Result<reqwest::Response, ClaudeError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let retry_after = retry_after_from_headers(response.headers());
    let body = response.text().await?;
    let message = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| body.clone());

    let error = ClaudeError::HttpStatus {
        status,
        message,
        retry_after,
    };
    emit_claude_request_error(
        raw,
        None,
        RequestErrorKind::HttpStatus,
        Some(status),
        Some(&body),
        &error.to_string(),
        &basic_request_error_debug_info(&error),
    );
    Err(error)
}

fn parse_sse_event(event: &Option<String>, data: &str) -> Result<SseEvent, ClaudeError> {
    if data.is_empty() {
        if event.as_deref() == Some("ping") {
            return serde_json::from_str::<SseEvent>(r#"{"type":"ping"}"#)
                .map_err(ClaudeError::Json);
        }
        return Err(ClaudeError::Sse {
            message: "received empty Claude SSE payload".to_string(),
        });
    }
    serde_json::from_str::<SseEvent>(data).map_err(ClaudeError::Json)
}

fn start_block_state(content_block: SseContentBlock) -> ContentBlockState {
    match content_block {
        SseContentBlock::Text(block) => ContentBlockState::Text {
            content: block.text,
        },
        SseContentBlock::Thinking(block) => ContentBlockState::Thinking {
            content: block.thinking,
            signature: block.signature,
        },
        SseContentBlock::ToolUse(block) => ContentBlockState::ToolUse {
            id: block.id,
            name: block.name,
            initial_json: (!block.input.is_null()).then(|| block.input.to_string()),
            delta_json: String::new(),
            finalized_arguments: None,
        },
        SseContentBlock::RedactedThinking { .. } | SseContentBlock::Unsupported => {
            ContentBlockState::Unsupported
        }
    }
}

fn parse_error_event(payload: ErrorEvent) -> ClaudeError {
    ClaudeError::Sse {
        message: payload.error.message,
    }
}

fn cache_usage(usage_cache: &UsageCache, request_id: &str, usage: Usage) {
    usage_cache_lock(usage_cache).insert(request_id.to_string(), usage);
}

fn remove_cached_usage(usage_cache: &UsageCache, request_id: &str) {
    usage_cache_lock(usage_cache).remove(request_id);
}

fn take_cached_usage(usage_cache: &UsageCache, request_id: &str) -> Option<Usage> {
    usage_cache_lock(usage_cache).remove(request_id)
}

fn usage_cache_lock(usage_cache: &UsageCache) -> std::sync::MutexGuard<'_, HashMap<String, Usage>> {
    usage_cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
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

/// Recursively add `"additionalProperties": false` to every JSON object schema
/// that declares `"type": "object"`. The Claude API requires this for structured
/// output schemas; `schemars` does not emit it by default.
fn require_no_additional_properties(schema: &mut serde_json::Value) {
    match schema {
        serde_json::Value::Object(map) => {
            if map.get("type").and_then(|t| t.as_str()) == Some("object") {
                map.entry("additionalProperties")
                    .or_insert(serde_json::Value::Bool(false));
            }
            for value in map.values_mut() {
                require_no_additional_properties(value);
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr.iter_mut() {
                require_no_additional_properties(item);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use futures::{StreamExt, executor::block_on};

    use lutum_protocol::{
        AdapterToolChoice, AdapterTurnConfig, ErasedTextTurnEvent, GenerationParams, ModelInput,
        ModelInputItem, ModelName, OperationKind, ParseErrorStage, RawTelemetryConfig,
        RequestErrorDebugInfo, RequestErrorKind, RequestExtensions, UsageRecoveryAdapter,
        budget::Usage,
    };
    use lutum_trace::RawTraceEntry;

    use super::*;

    #[lutum_macros::impl_hook(SelectClaudeModel)]
    async fn prefer_claude_sonnet(
        _extensions: &RequestExtensions,
        _default: ModelName,
    ) -> ModelName {
        ModelName::new("claude-sonnet-4-5").unwrap()
    }

    #[lutum_macros::impl_hook(SelectClaudeModel)]
    async fn prefer_claude_haiku(
        _extensions: &RequestExtensions,
        _default: ModelName,
    ) -> ModelName {
        ModelName::new("claude-haiku-4-5").unwrap()
    }

    fn raw_extensions() -> RequestExtensions {
        let mut extensions = RequestExtensions::new();
        extensions.insert(RawTelemetryConfig::all());
        extensions
    }

    #[test]
    fn prepare_messages_request_uses_explicit_model() {
        let adapter = ClaudeAdapter::new("test-key");

        let input = ModelInput::from_items(vec![ModelInputItem::text(
            lutum_protocol::InputMessageRole::User,
            "hello",
        )]);
        let config = AdapterTurnConfig {
            generation: GenerationParams::default(),
            tools: Vec::new(),
            tool_choice: AdapterToolChoice::Auto,
        };

        let request = adapter
            .prepare_messages_request(&input, &config, "claude-sonnet-override", None, None, true)
            .unwrap();

        assert_eq!(request.model, "claude-sonnet-override");

        let payloads = vec![Ok(Bytes::from(
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ))];
        let events = block_on(async {
            map_text_stream(
                futures::stream::iter(payloads),
                "claude-sonnet-override".to_string(),
                Arc::new(Mutex::new(HashMap::new())),
                None,
            )
            .collect::<Vec<_>>()
            .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        assert!(matches!(
            &events[0],
            ErasedTextTurnEvent::Started { model, .. } if model == "claude-sonnet-override"
        ));
    }

    #[test]
    fn fallback_serializer_is_applied_to_request() {
        struct TaggingSerializer;
        impl FallbackSerializer for TaggingSerializer {
            fn apply(&self, request: &mut MessagesRequest) {
                request.models = Some(vec!["fallback-model".to_string()]);
            }
        }

        let mut adapter = ClaudeAdapter::new("test-key");
        adapter.set_fallback_serializer(Box::new(TaggingSerializer));

        let input = ModelInput::from_items(vec![ModelInputItem::text(
            lutum_protocol::InputMessageRole::User,
            "hello",
        )]);
        let config = AdapterTurnConfig {
            generation: GenerationParams::default(),
            tools: Vec::new(),
            tool_choice: AdapterToolChoice::Auto,
        };

        let request = adapter
            .prepare_messages_request(&input, &config, "claude-sonnet", None, None, true)
            .unwrap();

        assert_eq!(request.model, "claude-sonnet");
        assert_eq!(request.models, Some(vec!["fallback-model".to_string()]));
    }

    #[test]
    fn select_claude_model_uses_last_registered_singleton_override() {
        let hooks = ClaudeHooksSet::new()
            .with_select_claude_model(PreferClaudeSonnet)
            .with_select_claude_model(PreferClaudeHaiku);

        let selected = block_on(hooks.select_claude_model(
            &RequestExtensions::new(),
            ModelName::new("claude-opus-4-5").unwrap(),
        ));

        assert_eq!(selected.as_str(), "claude-haiku-4-5");
    }

    #[test]
    fn recover_usage_returns_last_seen_usage_for_interrupted_stream() {
        let adapter = ClaudeAdapter::new("test-key");
        let payloads = vec![
            Ok(Bytes::from(
                "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_interrupted\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}\n\n",
            )),
            Ok(Bytes::from(
                "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":25,\"output_tokens\":15,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0}}\n\n",
            )),
        ];

        let events = block_on(async {
            map_text_stream(
                futures::stream::iter(payloads),
                "fallback-model".to_string(),
                Arc::clone(&adapter.usage_cache),
                None,
            )
            .collect::<Vec<_>>()
            .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        assert!(matches!(
            events.as_slice(),
            [ErasedTextTurnEvent::Started {
                request_id: Some(request_id),
                model,
            }] if request_id == "msg_interrupted" && model == "claude-sonnet"
        ));

        let recovered =
            block_on(adapter.recover_usage(OperationKind::TextTurn, "msg_interrupted")).unwrap();
        assert_eq!(
            recovered,
            Some(Usage {
                input_tokens: 25,
                output_tokens: 15,
                total_tokens: 40,
                cost_micros_usd: 0,
                ..Usage::zero()
            })
        );
        assert_eq!(
            block_on(adapter.recover_usage(OperationKind::TextTurn, "msg_interrupted")).unwrap(),
            None
        );
    }

    #[test]
    fn completed_stream_removes_cached_usage() {
        let adapter = ClaudeAdapter::new("test-key");
        let payloads = vec![
            Ok(Bytes::from(
                "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_done\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}\n\n",
            )),
            Ok(Bytes::from(
                "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"input_tokens\":25,\"output_tokens\":15,\"cache_creation_input_tokens\":0,\"cache_read_input_tokens\":0}}\n\n",
            )),
            Ok(Bytes::from(
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
            )),
        ];

        let events = block_on(async {
            map_text_stream(
                futures::stream::iter(payloads),
                "fallback-model".to_string(),
                Arc::clone(&adapter.usage_cache),
                None,
            )
            .collect::<Vec<_>>()
            .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        assert!(matches!(
            events.last(),
            Some(ErasedTextTurnEvent::Completed {
                request_id: Some(request_id),
                usage,
                ..
            }) if request_id == "msg_done" && *usage == (Usage {
                input_tokens: 25,
                output_tokens: 15,
                total_tokens: 40,
                cost_micros_usd: 0,
                ..Usage::zero()
            })
        ));
        assert_eq!(
            block_on(adapter.recover_usage(OperationKind::TextTurn, "msg_done")).unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn raw_trace_captures_request_and_frame_payloads() {
        let adapter = ClaudeAdapter::new("test-key");
        let input = ModelInput::from_items(vec![ModelInputItem::text(
            lutum_protocol::InputMessageRole::User,
            "hello",
        )]);
        let config = AdapterTurnConfig {
            generation: GenerationParams::default(),
            tools: Vec::new(),
            tool_choice: AdapterToolChoice::Auto,
        };
        let request = adapter
            .prepare_messages_request(&input, &config, "claude-sonnet", None, None, true)
            .unwrap();
        let request_body = serialize_raw_body(&request).unwrap();

        let collected = lutum_trace::test::collect_raw(async move {
            let extensions = raw_extensions();
            let raw = RawTelemetryEmitter::new(&extensions, "claude", "messages", "text_turn");
            raw.unwrap().emit_request(None, &request_body);
            map_text_stream(
                futures::stream::iter(vec![
                    Ok(Bytes::from(
                        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_raw\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
                    )),
                    Ok(Bytes::from(
                        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                    )),
                ]),
                "claude-sonnet".to_string(),
                Arc::new(Mutex::new(HashMap::new())),
                raw,
            )
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
        })
        .await;

        assert!(matches!(
            collected.raw.entries.first(),
            Some(RawTraceEntry::Request { provider, api, operation, body, .. })
                if provider == "claude"
                    && api == "messages"
                    && operation == "text_turn"
                    && body.contains("\"model\":\"claude-sonnet\"")
        ));
        assert!(collected.raw.entries.iter().any(|entry| matches!(
            entry,
            RawTraceEntry::StreamEvent { request_id, event_name, payload, .. }
                if request_id.as_deref() == Some("msg_raw")
                    && event_name.as_deref() == Some("message_stop")
                    && payload.contains("\"type\":\"message_stop\"")
        )));
    }

    #[tokio::test]
    async fn raw_trace_captures_http_status_request_errors() {
        let collected = lutum_trace::test::collect_raw(async {
            let extensions = raw_extensions();
            let raw = RawTelemetryEmitter::new(&extensions, "claude", "messages", "text_turn");
            let error = ClaudeError::HttpStatus {
                status: reqwest::StatusCode::BAD_GATEWAY,
                message: "upstream overloaded".into(),
                retry_after: None,
            };
            emit_claude_request_error(
                raw.as_ref(),
                None,
                RequestErrorKind::HttpStatus,
                Some(reqwest::StatusCode::BAD_GATEWAY),
                Some("{\"error\":{\"message\":\"upstream overloaded\"}}"),
                &error.to_string(),
                &RequestErrorDebugInfo {
                    error_debug: format!("{error:?}"),
                    ..RequestErrorDebugInfo::default()
                },
            );
        })
        .await;

        assert!(matches!(
            collected.raw.entries.as_slice(),
            [
                RawTraceEntry::RequestError {
                    provider,
                    api,
                    operation,
                    request_id,
                    kind,
                    status,
                    payload,
                    error,
                    error_debug,
                    source_chain,
                    is_timeout,
                    is_connect,
                    is_request,
                    is_body,
                    is_decode,
                },
            ] if provider == "claude"
                && api == "messages"
                && operation == "text_turn"
                && request_id.is_none()
                && *kind == RequestErrorKind::HttpStatus
                && *status == Some(reqwest::StatusCode::BAD_GATEWAY.as_u16())
                && payload.as_deref() == Some("{\"error\":{\"message\":\"upstream overloaded\"}}")
                && error.contains("request failed with status")
                && error_debug.contains("HttpStatus")
                && source_chain.is_empty()
                && !is_timeout
                && !is_connect
                && !is_request
                && !is_body
                && !is_decode
        ));
    }

    #[tokio::test]
    async fn raw_trace_captures_sse_parse_errors() {
        let collected = lutum_trace::test::collect_raw(async {
            let extensions = raw_extensions();
            let raw = RawTelemetryEmitter::new(&extensions, "claude", "messages", "text_turn");
            map_text_stream(
                futures::stream::iter(vec![Ok(Bytes::from(
                    "event: content_block_delta\ndata: not-json\n\n",
                ))]),
                "claude-sonnet".to_string(),
                Arc::new(Mutex::new(HashMap::new())),
                raw,
            )
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
        })
        .await;

        assert!(collected.output.is_err());
        assert!(matches!(
            collected.raw.entries.as_slice(),
            [
                RawTraceEntry::StreamEvent { payload, event_name, .. },
                RawTraceEntry::ParseError { stage, payload: error_payload, .. },
            ] if payload == "not-json"
                && event_name.as_deref() == Some("content_block_delta")
                && *stage == ParseErrorStage::SseParse
                && error_payload == "not-json"
        ));
    }

    #[tokio::test]
    async fn raw_trace_captures_structured_output_parse_errors() {
        let collected = lutum_trace::test::collect_raw(async {
            let extensions = raw_extensions();
            let raw =
                RawTelemetryEmitter::new(&extensions, "claude", "messages", "structured_turn");
            map_structured_stream(
                futures::stream::iter(vec![
                    Ok(Bytes::from(
                        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_structured_bad\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
                    )),
                    Ok(Bytes::from(
                        "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                    )),
                    Ok(Bytes::from(
                        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"{\"}}\n\n",
                    )),
                    Ok(Bytes::from(
                        "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                    )),
                    Ok(Bytes::from(
                        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
                    )),
                ]),
                "claude-sonnet".to_string(),
                Arc::new(Mutex::new(HashMap::new())),
                raw,
            )
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
        })
        .await;

        assert!(collected.output.is_err());
        assert!(collected.raw.entries.iter().any(|entry| matches!(
            entry,
            RawTraceEntry::ParseError {
                request_id,
                stage,
                payload,
                ..
            } if request_id.as_deref() == Some("msg_structured_bad")
                && *stage == ParseErrorStage::StructuredOutputParse
                && payload == "{"
        )));
    }
}
