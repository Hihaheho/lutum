use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    pin::Pin,
    sync::Arc,
};

use async_stream::try_stream;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use lutum_protocol::{
    AgentError, FinishReason,
    budget::Usage,
    conversation::{
        AssistantInputItem, InputMessageRole, MessageContent, ModelInput, ModelInputItem, RawJson,
        ToolCallId, ToolMetadata, ToolName, ToolResult,
    },
    extensions::RequestExtensions,
    hooks::HookRegistry,
    llm::{
        AdapterStructuredCompletionRequest, AdapterStructuredTurn, AdapterTextTurn,
        AdapterToolChoice, AdapterTurnConfig, CompletionAdapter, CompletionEvent,
        CompletionEventStream, CompletionRequest as ProtocolCompletionRequest,
        ErasedStructuredCompletionEvent, ErasedStructuredCompletionEventStream,
        ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
        ErasedTextTurnEventStream, ModelName, OperationKind, TurnAdapter, UsageRecoveryAdapter,
    },
    transcript::{TurnRole, TurnView},
};
use reqwest::{
    Client,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    completion::CompletionRequest,
    error::OpenAiError,
    responses::{
        FunctionCallItem, FunctionCallOutputItem, FunctionToolChoice, InputContent, InputItem,
        InputMessage, InputTextContent, MessageRole, OpenAiCommittedTurn, OpenAiReasoningEffort,
        OpenAiTool, OpenAiTurnItem, OutputTextContent, ReasoningItem, RefusalContent,
        ResponseFunctionCallArgumentsDeltaEvent, ResponseFunctionCallArgumentsDoneEvent,
        ResponseOutputItem, SseEvent, SummaryText, TextFormat, ToolChoice,
    },
    sse::SseParser,
};

pub trait FallbackSerializer: Send + Sync {
    fn apply_to_responses(&self, request: &mut crate::responses::ResponsesRequest);
    fn apply_to_completion(&self, request: &mut CompletionRequest);
}

#[lutum_macros::def_global_hook(singleton)]
pub async fn select_openai_model(_extensions: &RequestExtensions, default: ModelName) -> ModelName {
    default
}

#[lutum_macros::def_global_hook(singleton)]
pub async fn resolve_reasoning_effort(
    _extensions: &RequestExtensions,
) -> Option<OpenAiReasoningEffort> {
    None
}

/// A snapshot of tool name information available at the time an SSE decode error occurred.
/// Passed to [`SseEventRecoveryHook`] so implementations can reconstruct missing fields.
pub struct SseHints {
    tool_names: BTreeMap<String, String>,
}

impl SseHints {
    /// Look up the tool name for a given key (call_id or item_id).
    pub fn tool_name_for(&self, key: &str) -> Option<&str> {
        self.tool_names.get(key).map(String::as_str)
    }
}

/// Hook called when an SSE event payload fails to deserialize.
/// Implementations can attempt to recover by supplying a reconstructed [`SseEvent`].
///
/// The primary use case is compensating for provider-specific SSE format deviations
/// not already handled by the built-in decoder.
pub trait SseEventRecoveryHook: Send + Sync {
    fn recover_event(
        &self,
        payload: &str,
        error: &serde_json::Error,
        hints: &SseHints,
    ) -> Result<Option<SseEvent>, OpenAiError>;
}

#[derive(Clone)]
pub struct OpenAiAdapter {
    client: Arc<Client>,
    api_key: Arc<str>,
    base_url: Arc<str>,
    default_model: ModelName,
    fallback_serializer: Option<Arc<dyn FallbackSerializer>>,
    sse_event_recovery_hook: Option<Arc<dyn SseEventRecoveryHook>>,
}

type ByteStream =
    Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send + Sync + 'static>>;

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
            default_model: ModelName::new("gpt-4.1").unwrap(),
            fallback_serializer: None,
            sse_event_recovery_hook: None,
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Arc::from(base_url.into());
        self
    }

    pub fn with_default_model(mut self, model: ModelName) -> Self {
        self.default_model = model;
        self
    }

    pub fn set_fallback_serializer(&mut self, serializer: Box<dyn FallbackSerializer>) {
        self.fallback_serializer = Some(serializer.into());
    }

    pub fn with_sse_event_recovery_hook(
        mut self,
        hook: impl SseEventRecoveryHook + 'static,
    ) -> Self {
        self.sse_event_recovery_hook = Some(Arc::new(hook));
        self
    }

    pub fn set_sse_event_recovery_hook(&mut self, hook: Box<dyn SseEventRecoveryHook>) {
        self.sse_event_recovery_hook = Some(hook.into());
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

    fn prepare_responses_request(
        &self,
        input: &ModelInput,
        config: &AdapterTurnConfig,
        model: &str,
        reasoning_effort: Option<OpenAiReasoningEffort>,
        text_format: Option<TextFormat>,
    ) -> Result<crate::responses::ResponsesRequest, OpenAiError> {
        let mut request =
            build_responses_request(input, config, model, reasoning_effort, text_format)?;
        if let Some(serializer) = self.fallback_serializer.as_ref() {
            serializer.apply_to_responses(&mut request);
        }
        Ok(request)
    }

    fn prepare_completion_request(
        &self,
        request: &ProtocolCompletionRequest,
        model: &str,
    ) -> CompletionRequest {
        let mut body = build_completion_request(request, model);
        if let Some(serializer) = self.fallback_serializer.as_ref() {
            serializer.apply_to_completion(&mut body);
        }
        body
    }

    fn prepare_structured_completion_request(
        &self,
        request: &AdapterStructuredCompletionRequest,
        model: &str,
    ) -> crate::responses::ResponsesRequest {
        let mut body = build_structured_completion_request(request, model);
        if let Some(serializer) = self.fallback_serializer.as_ref() {
            serializer.apply_to_responses(&mut body);
        }
        body
    }

    async fn send_streaming_json<T>(&self, path: &str, body: &T) -> Result<ByteStream, OpenAiError>
    where
        T: Serialize + ?Sized,
    {
        let response = self
            .client
            .post(format!("{}{}", self.base_url, path))
            .headers(self.request_headers()?)
            .json(body)
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

#[async_trait::async_trait]
impl TurnAdapter for OpenAiAdapter {
    async fn text_turn(
        &self,
        input: ModelInput,
        turn: AdapterTextTurn,
        hooks: &HookRegistry,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        let model = hooks
            .select_openai_model(turn.extensions.as_ref(), self.default_model.clone())
            .await;
        let reasoning_effort = hooks
            .resolve_reasoning_effort(turn.extensions.as_ref())
            .await;
        let body = self
            .prepare_responses_request(&input, &turn.config, model.as_ref(), reasoning_effort, None)
            .map_err(AgentError::backend)?;
        let stream = self
            .send_streaming_json("/responses", &body)
            .await
            .map_err(AgentError::backend)?;
        Ok(Box::pin(
            map_text_stream(stream, model.into(), self.sse_event_recovery_hook.clone())
                .map(|item| item.map_err(AgentError::backend)),
        ) as ErasedTextTurnEventStream)
    }

    async fn structured_turn(
        &self,
        input: ModelInput,
        turn: AdapterStructuredTurn,
        hooks: &HookRegistry,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
        let model = hooks
            .select_openai_model(turn.extensions.as_ref(), self.default_model.clone())
            .await;
        let reasoning_effort = hooks
            .resolve_reasoning_effort(turn.extensions.as_ref())
            .await;
        let text_format = Some(TextFormat::JsonSchema {
            name: turn.output.schema_name.clone(),
            schema: turn.output.schema.clone(),
            description: None,
            strict: Some(true),
        });
        let body = self
            .prepare_responses_request(
                &input,
                &turn.config,
                model.as_ref(),
                reasoning_effort,
                text_format,
            )
            .map_err(AgentError::backend)?;
        let stream = self
            .send_streaming_json("/responses", &body)
            .await
            .map_err(AgentError::backend)?;
        Ok(Box::pin(
            map_structured_stream(stream, model.into(), self.sse_event_recovery_hook.clone())
                .map(|item| item.map_err(AgentError::backend)),
        ) as ErasedStructuredTurnEventStream)
    }
}

#[async_trait::async_trait]
impl CompletionAdapter for OpenAiAdapter {
    async fn completion(
        &self,
        request: ProtocolCompletionRequest,
        extensions: &RequestExtensions,
        hooks: &HookRegistry,
    ) -> Result<CompletionEventStream, AgentError> {
        let model = hooks
            .select_openai_model(extensions, self.default_model.clone())
            .await;
        let body = self.prepare_completion_request(&request, model.as_ref());
        let stream = self
            .send_streaming_json("/completions", &body)
            .await
            .map_err(AgentError::backend)?;
        Ok(Box::pin(
            map_completion_stream(stream, model.into())
                .map(|item| item.map_err(AgentError::backend)),
        ) as CompletionEventStream)
    }

    async fn structured_completion(
        &self,
        request: AdapterStructuredCompletionRequest,
        extensions: &RequestExtensions,
        hooks: &HookRegistry,
    ) -> Result<ErasedStructuredCompletionEventStream, AgentError> {
        let model = hooks
            .select_openai_model(extensions, self.default_model.clone())
            .await;
        let body = self.prepare_structured_completion_request(&request, model.as_ref());
        let stream = self
            .send_streaming_json("/responses", &body)
            .await
            .map_err(AgentError::backend)?;
        let stream =
            map_structured_stream(stream, model.into(), self.sse_event_recovery_hook.clone())
                .map(|item| item.map_err(AgentError::backend));
        Ok(
            Box::pin(stream.map(|item| item.and_then(map_erased_structured_completion_event)))
                as ErasedStructuredCompletionEventStream,
        )
    }
}

fn build_structured_completion_request(
    request: &AdapterStructuredCompletionRequest,
    model: &str,
) -> crate::responses::ResponsesRequest {
    let mut input = Vec::new();
    if let Some(system) = request.system.as_deref() {
        input.push(InputItem::Message(InputMessage::new(
            MessageRole::System,
            vec![InputContent::InputText(InputTextContent::new(system))],
        )));
    }
    input.push(InputItem::Message(InputMessage::new(
        MessageRole::User,
        vec![InputContent::InputText(InputTextContent::new(
            request.prompt.clone(),
        ))],
    )));

    crate::responses::ResponsesRequest {
        model: model.to_string(),
        input,
        stream: true,
        tools: Vec::new(),
        parallel_tool_calls: false,
        temperature: request
            .generation
            .temperature
            .map(|temperature| temperature.get()),
        max_output_tokens: request.generation.max_output_tokens,
        reasoning: None,
        text: Some(crate::responses::ResponsesTextConfig {
            format: TextFormat::JsonSchema {
                name: request.output.schema_name.clone(),
                schema: request.output.schema.clone(),
                description: None,
                strict: Some(true),
            },
        }),
        tool_choice: None,
        models: None,
        seed: request.generation.seed,
    }
}

fn map_erased_structured_completion_event(
    event: ErasedStructuredTurnEvent,
) -> Result<ErasedStructuredCompletionEvent, AgentError> {
    match event {
        ErasedStructuredTurnEvent::Started { request_id, model } => {
            Ok(ErasedStructuredCompletionEvent::Started { request_id, model })
        }
        ErasedStructuredTurnEvent::StructuredOutputChunk { json_delta } => {
            Ok(ErasedStructuredCompletionEvent::StructuredOutputChunk { json_delta })
        }
        ErasedStructuredTurnEvent::StructuredOutputReady(raw) => {
            Ok(ErasedStructuredCompletionEvent::StructuredOutputReady(raw))
        }
        ErasedStructuredTurnEvent::ReasoningDelta { delta } => {
            Ok(ErasedStructuredCompletionEvent::ReasoningDelta { delta })
        }
        ErasedStructuredTurnEvent::RefusalDelta { delta } => {
            Ok(ErasedStructuredCompletionEvent::RefusalDelta { delta })
        }
        ErasedStructuredTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            ..
        } => Ok(ErasedStructuredCompletionEvent::Completed {
            request_id,
            finish_reason,
            usage,
        }),
        ErasedStructuredTurnEvent::ToolCallChunk { .. }
        | ErasedStructuredTurnEvent::ToolCallReady(_) => {
            Err(AgentError::backend(OpenAiError::Sse {
                message: "structured completion does not support tool calls".to_string(),
            }))
        }
    }
}

#[async_trait::async_trait]
impl UsageRecoveryAdapter for OpenAiAdapter {
    async fn recover_usage(
        &self,
        kind: OperationKind,
        request_id: &str,
    ) -> Result<Option<Usage>, AgentError> {
        match kind {
            OperationKind::TextTurn
            | OperationKind::StructuredTurn
            | OperationKind::StructuredCompletion => {
                let value = self
                    .get_json(&format!("/responses/{request_id}"))
                    .await
                    .map_err(AgentError::backend)?;
                Ok(Some(parse_response_usage(&value)))
            }
            OperationKind::Completion => Ok(None),
        }
    }
}

fn build_responses_request(
    input: &ModelInput,
    config: &AdapterTurnConfig,
    model: &str,
    reasoning_effort: Option<OpenAiReasoningEffort>,
    text_format: Option<TextFormat>,
) -> Result<crate::responses::ResponsesRequest, OpenAiError> {
    let tools = build_tool_definitions(config);
    let parallel_tool_calls = match &config.tool_choice {
        AdapterToolChoice::Required | AdapterToolChoice::Specific(_) => false,
        AdapterToolChoice::None | AdapterToolChoice::Auto => true,
    };
    let tool_choice = match &config.tool_choice {
        AdapterToolChoice::Required => Some(ToolChoice::Required),
        AdapterToolChoice::None => Some(ToolChoice::None),
        AdapterToolChoice::Specific(name) => {
            Some(ToolChoice::Function(FunctionToolChoice::new(name.clone())))
        }
        AdapterToolChoice::Auto => None,
    };

    Ok(crate::responses::ResponsesRequest {
        model: model.to_string(),
        input: convert_model_input(input)?,
        stream: true,
        tools,
        parallel_tool_calls,
        temperature: config
            .generation
            .temperature
            .map(|temperature| temperature.get()),
        max_output_tokens: config.generation.max_output_tokens,
        reasoning: reasoning_effort
            .map(|effort| crate::responses::ResponsesReasoningConfig { effort }),
        text: text_format.map(|format| crate::responses::ResponsesTextConfig { format }),
        tool_choice,
        models: None,
        seed: config.generation.seed,
    })
}

fn build_completion_request(request: &ProtocolCompletionRequest, model: &str) -> CompletionRequest {
    CompletionRequest {
        model: model.to_string(),
        prompt: request.prompt.clone(),
        stream: true,
        temperature: request
            .options
            .temperature
            .map(|temperature| temperature.get()),
        max_tokens: request.options.max_output_tokens,
        stop: request.options.stop.clone(),
        models: None,
    }
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

fn convert_model_input(input: &ModelInput) -> Result<Vec<InputItem>, OpenAiError> {
    let mut items = Vec::new();
    let mut assistant_message_content = Vec::new();
    let mut replayed_tool_call_ids = BTreeSet::new();

    for item in input.items() {
        match item {
            ModelInputItem::Message { role, content } => {
                flush_assistant_message(&mut assistant_message_content, &mut items);
                items.push(InputItem::Message(InputMessage::new(
                    message_role(role),
                    content.iter().map(message_content).collect(),
                )));
            }
            ModelInputItem::Assistant(assistant) => {
                lower_assistant_input_item(assistant, &mut assistant_message_content, &mut items)?;
            }
            ModelInputItem::ToolResult(tool_result) => {
                flush_assistant_message(&mut assistant_message_content, &mut items);
                lower_tool_result(
                    tool_result,
                    !replayed_tool_call_ids.contains(&tool_result.id),
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
    message_content: &mut Vec<InputContent>,
    out: &mut Vec<InputItem>,
) -> Result<(), OpenAiError> {
    match item {
        AssistantInputItem::Text(text) => {
            message_content.push(InputContent::OutputText(OutputTextContent::new(
                text.clone(),
            )));
        }
        AssistantInputItem::Refusal(text) => {
            message_content.push(InputContent::Refusal(RefusalContent::new(text.clone())));
        }
        AssistantInputItem::Reasoning(text) => {
            flush_assistant_message(message_content, out);
            out.push(InputItem::Reasoning(ReasoningItem::new(vec![
                SummaryText::new(text.clone()),
            ])));
        }
    }
    Ok(())
}

fn lower_tool_result(
    tool_result: &ToolResult,
    emit_call: bool,
    out: &mut Vec<InputItem>,
) -> Result<(), OpenAiError> {
    if emit_call {
        out.push(InputItem::FunctionCall(FunctionCallItem::new(
            tool_result.id.as_str(),
            tool_result.name.as_str(),
            tool_result.arguments.get(),
        )));
    }
    out.push(InputItem::FunctionCallOutput(FunctionCallOutputItem::new(
        tool_result.id.as_str(),
        Value::String(tool_result.result.get().to_string()),
    )));
    Ok(())
}

fn emit_openai_turn_exact(
    turn: &OpenAiCommittedTurn,
    out: &mut Vec<InputItem>,
    replayed_tool_call_ids: &mut BTreeSet<ToolCallId>,
) -> Result<(), OpenAiError> {
    let mut assistant_message_content = Vec::new();
    for item in &turn.items {
        match item {
            OpenAiTurnItem::Text { content } => {
                assistant_message_content.push(InputContent::OutputText(OutputTextContent::new(
                    content.clone(),
                )));
            }
            OpenAiTurnItem::Reasoning { content } => {
                flush_assistant_message(&mut assistant_message_content, out);
                out.push(InputItem::Reasoning(ReasoningItem::new(vec![
                    SummaryText::new(content.clone()),
                ])));
            }
            OpenAiTurnItem::Refusal { content } => {
                assistant_message_content
                    .push(InputContent::Refusal(RefusalContent::new(content.clone())));
            }
            OpenAiTurnItem::ToolCall {
                id,
                name,
                arguments,
            } => {
                replayed_tool_call_ids.insert(id.clone());
                flush_assistant_message(&mut assistant_message_content, out);
                out.push(InputItem::FunctionCall(FunctionCallItem::new(
                    id.as_str(),
                    name.as_str(),
                    arguments.get(),
                )));
            }
        }
    }
    flush_assistant_message(&mut assistant_message_content, out);
    Ok(())
}

fn emit_turn_from_view(
    turn: &dyn TurnView,
    out: &mut Vec<InputItem>,
    replayed_tool_call_ids: &mut BTreeSet<ToolCallId>,
) -> Result<(), OpenAiError> {
    match turn.role() {
        TurnRole::Assistant => emit_assistant_turn_from_view(turn, out, replayed_tool_call_ids),
        role => emit_message_turn_from_view(role, turn, out, replayed_tool_call_ids),
    }
}

fn emit_assistant_turn_from_view(
    turn: &dyn TurnView,
    out: &mut Vec<InputItem>,
    replayed_tool_call_ids: &mut BTreeSet<ToolCallId>,
) -> Result<(), OpenAiError> {
    let mut assistant_message_content = Vec::new();
    for index in 0..turn.item_count() {
        let Some(item) = turn.item_at(index) else {
            continue;
        };
        if let Some(text) = item.as_text() {
            assistant_message_content.push(InputContent::OutputText(OutputTextContent::new(text)));
            continue;
        }
        if let Some(text) = item.as_reasoning() {
            flush_assistant_message(&mut assistant_message_content, out);
            out.push(InputItem::Reasoning(ReasoningItem::new(vec![
                SummaryText::new(text.to_string()),
            ])));
            continue;
        }
        if let Some(text) = item.as_refusal() {
            assistant_message_content.push(InputContent::Refusal(RefusalContent::new(text)));
            continue;
        }
        if let Some(tool_call) = item.as_tool_call() {
            replayed_tool_call_ids.insert(tool_call.id.clone());
            flush_assistant_message(&mut assistant_message_content, out);
            out.push(InputItem::FunctionCall(FunctionCallItem::new(
                tool_call.id.as_str(),
                tool_call.name.as_str(),
                tool_call.arguments.get(),
            )));
        }
    }
    flush_assistant_message(&mut assistant_message_content, out);
    Ok(())
}

fn emit_message_turn_from_view(
    role: TurnRole,
    turn: &dyn TurnView,
    out: &mut Vec<InputItem>,
    replayed_tool_call_ids: &mut BTreeSet<ToolCallId>,
) -> Result<(), OpenAiError> {
    let mut message_content = Vec::new();
    for index in 0..turn.item_count() {
        let Some(item) = turn.item_at(index) else {
            continue;
        };
        if let Some(text) = item.as_text() {
            message_content.push(InputContent::InputText(InputTextContent::new(text)));
            continue;
        }
        if let Some(text) = item.as_reasoning().or_else(|| item.as_refusal()) {
            message_content.push(InputContent::InputText(InputTextContent::new(text)));
            continue;
        }
        if let Some(tool_call) = item.as_tool_call() {
            replayed_tool_call_ids.insert(tool_call.id.clone());
            flush_message(turn_role(role), &mut message_content, out);
            out.push(InputItem::FunctionCall(FunctionCallItem::new(
                tool_call.id.as_str(),
                tool_call.name.as_str(),
                tool_call.arguments.get(),
            )));
        }
        if let Some(tool_result) = item.as_tool_result() {
            replayed_tool_call_ids.insert(tool_result.id.clone());
            flush_message(turn_role(role), &mut message_content, out);
            out.push(InputItem::FunctionCall(FunctionCallItem::new(
                tool_result.id.as_str(),
                tool_result.name.as_str(),
                tool_result.arguments.get(),
            )));
            out.push(InputItem::FunctionCallOutput(FunctionCallOutputItem::new(
                tool_result.id.as_str(),
                Value::String(tool_result.result.get().to_string()),
            )));
        }
    }
    flush_message(turn_role(role), &mut message_content, out);
    Ok(())
}

fn flush_assistant_message(message_content: &mut Vec<InputContent>, out: &mut Vec<InputItem>) {
    flush_message(MessageRole::Assistant, message_content, out);
}

fn flush_message(
    role: MessageRole,
    message_content: &mut Vec<InputContent>,
    out: &mut Vec<InputItem>,
) {
    if message_content.is_empty() {
        return;
    }
    out.push(InputItem::Message(InputMessage::new(
        role,
        std::mem::take(message_content),
    )));
}

fn message_role(role: &InputMessageRole) -> MessageRole {
    match role {
        InputMessageRole::System => MessageRole::System,
        InputMessageRole::Developer => MessageRole::Developer,
        InputMessageRole::User => MessageRole::User,
    }
}

fn turn_role(role: TurnRole) -> MessageRole {
    match role {
        TurnRole::System => MessageRole::System,
        TurnRole::Developer => MessageRole::Developer,
        TurnRole::User => MessageRole::User,
        TurnRole::Assistant => MessageRole::Assistant,
    }
}

fn message_content(content: &MessageContent) -> InputContent {
    match content {
        MessageContent::Text(text) => InputContent::InputText(InputTextContent::new(text.clone())),
    }
}

fn build_tool_definitions(config: &AdapterTurnConfig) -> Vec<OpenAiTool> {
    config
        .tools
        .iter()
        .map(|tool| OpenAiTool::function(&tool.name, &tool.description, tool.input_schema.clone()))
        .collect()
}

#[derive(Default)]
struct ToolCallTracker {
    buffers: BTreeMap<String, ToolCallBuffer>,
    finalized: BTreeMap<String, FinalizedToolCall>,
}

struct ToolCallBuffer {
    id: ToolCallId,
    name: Option<ToolName>,
    arguments_json: String,
}

struct FinalizedToolCall {
    id: ToolCallId,
    name: ToolName,
    arguments_json: String,
}

impl ToolCallTracker {
    fn observe_call(&mut self, key: String, id: ToolCallId, name: ToolName) {
        let entry = self.buffers.entry(key).or_insert_with(|| ToolCallBuffer {
            id: id.clone(),
            name: Some(name.clone()),
            arguments_json: String::new(),
        });
        if entry.name.is_none() {
            entry.name = Some(name);
        }
    }

    fn peek_name(&self, key: &str) -> Option<&ToolName> {
        self.buffers
            .get(key)
            .and_then(|buffer| buffer.name.as_ref())
            .or_else(|| self.finalized.get(key).map(|finalized| &finalized.name))
    }

    fn record_delta(
        &mut self,
        key: String,
        id: ToolCallId,
        name: Option<ToolName>,
        delta: &str,
    ) -> Option<(ToolCallId, ToolName, String)> {
        if delta.is_empty() {
            return None;
        }

        let entry = self.buffers.entry(key).or_insert_with(|| ToolCallBuffer {
            id: id.clone(),
            name: name.clone(),
            arguments_json: String::new(),
        });
        if entry.name.is_none()
            && let Some(name) = name
        {
            entry.name = Some(name);
        }
        entry.arguments_json.push_str(delta);
        entry
            .name
            .as_ref()
            .map(|name| (entry.id.clone(), name.clone(), delta.to_string()))
    }

    fn finish(
        &mut self,
        key: String,
        id: ToolCallId,
        name: Option<ToolName>,
        explicit_arguments_json: Option<String>,
    ) -> Result<Option<ToolMetadata>, OpenAiError> {
        let buffered = self.buffers.remove(&key);

        // Prefer the id registered by observe_call (from response.output_item.added) over
        // the event-provided id. Some providers (e.g. Ollama with Qwen) use item_id in
        // delta/done events but call_id in output_item events, so the buffer holds the
        // authoritative call_id while the event carries only the item_id.
        let resolved_id = buffered.as_ref().map(|b| b.id.clone()).unwrap_or(id);

        let arguments_json = if let Some(args) = explicit_arguments_json {
            args
        } else {
            buffered
                .as_ref()
                .and_then(|buffer| {
                    if buffer.arguments_json.is_empty() {
                        None
                    } else {
                        Some(buffer.arguments_json.clone())
                    }
                })
                .unwrap_or_else(|| "{}".to_string())
        };

        let resolved_name = name
            .or_else(|| buffered.and_then(|buffer| buffer.name))
            .or_else(|| {
                self.finalized
                    .get(&key)
                    .map(|finalized| finalized.name.clone())
            });

        let Some(resolved_name) = resolved_name else {
            return Ok(None);
        };

        if let Some(existing) = self.finalized.get(&key) {
            if existing.id == resolved_id
                && existing.name == resolved_name
                && existing.arguments_json == arguments_json
            {
                return Ok(None);
            }
            return Err(OpenAiError::Sse {
                message: format!(
                    "conflicting duplicate tool call completion for `{}`",
                    resolved_id.as_str()
                ),
            });
        }

        // Cross-key dedup: some providers (e.g. Ollama with Qwen) emit both
        // FunctionCallArgumentsDone (keyed by item_id) and ResponseOutputItemDone (keyed by
        // call_id) for the same logical tool call. After resolving to the canonical id, a
        // matching entry under a different key means the call was already finalized.
        //
        // If name and arguments also match it is a true duplicate — suppress it.
        // If they differ (e.g. the first event had a partial/empty buffer and this one carries
        // the full payload), update the stored record with the authoritative data. The
        // ToolCallReady that was already yielded cannot be retracted, but at least the committed
        // transcript entry is corrected for future replay.
        if let Some((existing_key, existing)) =
            self.finalized.iter_mut().find(|(_, f)| f.id == resolved_id)
        {
            let _ = existing_key; // key differs by design in the cross-key case
            if existing.name != resolved_name || existing.arguments_json != arguments_json {
                existing.name = resolved_name;
                existing.arguments_json = arguments_json;
            }
            return Ok(None);
        }

        let arguments = RawJson::parse(arguments_json.clone())?;
        let metadata = ToolMetadata::new(resolved_id.clone(), resolved_name.clone(), arguments);
        self.finalized.insert(
            key,
            FinalizedToolCall {
                id: resolved_id,
                name: resolved_name,
                arguments_json,
            },
        );
        Ok(Some(metadata))
    }

    fn to_hints(&self) -> SseHints {
        let mut tool_names = BTreeMap::new();
        for (key, buffer) in &self.buffers {
            if let Some(name) = &buffer.name {
                tool_names.insert(key.clone(), name.as_str().to_string());
            }
        }
        for (key, finalized) in &self.finalized {
            tool_names
                .entry(key.clone())
                .or_insert_with(|| finalized.name.as_str().to_string());
        }
        SseHints { tool_names }
    }
}

fn response_tool_key_from_delta(event: &ResponseFunctionCallArgumentsDeltaEvent) -> String {
    event
        .call_id
        .as_deref()
        .or(event.item_id.as_deref())
        .unwrap_or_default()
        .to_string()
}

fn response_tool_id_from_delta(event: &ResponseFunctionCallArgumentsDeltaEvent) -> ToolCallId {
    ToolCallId::from(
        event
            .call_id
            .as_deref()
            .or(event.item_id.as_deref())
            .unwrap_or_default(),
    )
}

fn response_tool_key_from_done(event: &ResponseFunctionCallArgumentsDoneEvent) -> String {
    event
        .call_id
        .as_deref()
        .or(event.item_id.as_deref())
        .unwrap_or_default()
        .to_string()
}

fn response_tool_id_from_done(event: &ResponseFunctionCallArgumentsDoneEvent) -> ToolCallId {
    ToolCallId::from(
        event
            .call_id
            .as_deref()
            .or(event.item_id.as_deref())
            .unwrap_or_default(),
    )
}

fn output_item_tool_key(item: &FunctionCallItem) -> String {
    if !item.call_id.is_empty() {
        item.call_id.clone()
    } else {
        item.id.clone().unwrap_or_default()
    }
}

fn output_item_tool_id(item: &FunctionCallItem) -> ToolCallId {
    ToolCallId::from(if !item.call_id.is_empty() {
        item.call_id.as_str()
    } else {
        item.id.as_deref().unwrap_or_default()
    })
}

fn output_item_tool_name(item: &FunctionCallItem) -> ToolName {
    ToolName::from(item.name.as_str())
}

fn response_output_item_function_call(item: ResponseOutputItem) -> Option<FunctionCallItem> {
    match item {
        ResponseOutputItem::FunctionCall {
            id,
            call_id,
            name,
            arguments,
            namespace,
            status,
        } => Some(FunctionCallItem {
            arguments,
            call_id,
            name,
            item_type: "function_call".to_string(),
            id,
            namespace,
            status,
        }),
        _ => None,
    }
}

fn sse_decode_error(payload: &str, err: serde_json::Error) -> OpenAiError {
    const MAX_LEN: usize = 400;
    let snippet = if payload.len() > MAX_LEN {
        format!("{}...", &payload[..MAX_LEN])
    } else {
        payload.to_string()
    };
    OpenAiError::Sse {
        message: format!("{err}; payload={snippet}"),
    }
}

fn decode_sse_event(
    payload: &str,
    recovery_hook: Option<&Arc<dyn SseEventRecoveryHook>>,
    tool_calls: &ToolCallTracker,
) -> Result<SseEvent, OpenAiError> {
    match serde_json::from_str::<SseEvent>(payload) {
        Ok(event) => Ok(event),
        Err(err) => {
            if let Some(hook) = recovery_hook {
                let hints = tool_calls.to_hints();
                if let Some(event) = hook.recover_event(payload, &err, &hints)? {
                    return Ok(event);
                }
            }
            Err(sse_decode_error(payload, err))
        }
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
    recovery_hook: Option<Arc<dyn SseEventRecoveryHook>>,
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
                let event = decode_sse_event(&payload, recovery_hook.as_ref(), &tool_calls)?;
                if let SseEvent::ResponseCreated(created) = &event {
                    request_id = Some(created.response.id.clone());
                    if let Some(event_model) = created.response.model.as_ref() {
                        model = event_model.clone();
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

                match event {
                    SseEvent::ResponseOutputTextDelta(event) => {
                        push_buffered_content(
                            &mut pending_item,
                            &mut committed_items,
                            BufferedTurnItemKind::Text,
                            &event.item_id,
                            &event.delta,
                        );
                        yield ErasedTextTurnEvent::TextDelta {
                            delta: event.delta,
                        };
                    }
                    SseEvent::ResponseReasoningSummaryTextDelta(event) => {
                        push_buffered_content(
                            &mut pending_item,
                            &mut committed_items,
                            BufferedTurnItemKind::Reasoning,
                            event.item_id.as_deref().unwrap_or(""),
                            &event.delta,
                        );
                        yield ErasedTextTurnEvent::ReasoningDelta {
                            delta: event.delta,
                        };
                    }
                    SseEvent::ResponseReasoningDelta(event) => {
                        push_buffered_content(
                            &mut pending_item,
                            &mut committed_items,
                            BufferedTurnItemKind::Reasoning,
                            event.item_id.as_deref().unwrap_or(""),
                            &event.delta,
                        );
                        yield ErasedTextTurnEvent::ReasoningDelta {
                            delta: event.delta,
                        };
                    }
                    SseEvent::ResponseRefusalDelta(event) => {
                        saw_refusal = true;
                        push_buffered_content(
                            &mut pending_item,
                            &mut committed_items,
                            BufferedTurnItemKind::Refusal,
                            event.item_id.as_deref().unwrap_or(""),
                            &event.delta,
                        );
                        yield ErasedTextTurnEvent::RefusalDelta {
                            delta: event.delta,
                        };
                    }
                    SseEvent::ResponseOutputTextDone(event) => {
                        replace_buffered_text(
                            &mut pending_item,
                            &mut committed_items,
                            &event.item_id,
                            &event.text,
                        );
                    }
                    SseEvent::ResponseFunctionCallArgumentsDelta(event) => {
                        let key = response_tool_key_from_delta(&event);
                        if let Some((id, name, delta)) = tool_calls.record_delta(
                            key.clone(),
                            response_tool_id_from_delta(&event),
                            tool_calls.peek_name(&key).cloned(),
                            &event.delta,
                        ) {
                            yield ErasedTextTurnEvent::ToolCallChunk {
                                id,
                                name,
                                arguments_json_delta: delta,
                            };
                        }
                    }
                    SseEvent::ResponseFunctionCallArgumentsDone(event) => {
                        saw_tool_call = true;
                        if let Some(invocation) = tool_calls.finish(
                            response_tool_key_from_done(&event),
                            response_tool_id_from_done(&event),
                            event.name.as_deref().map(ToolName::from),
                            event.arguments.clone(),
                        )? {
                            flush_buffered_content(&mut pending_item, &mut committed_items);
                            push_committed_tool_call(&mut committed_items, &invocation);
                            yield ErasedTextTurnEvent::ToolCallReady(invocation);
                        }
                    }
                    SseEvent::ResponseOutputItemAdded(event) => {
                        if let Some(item) = response_output_item_function_call(event.item) {
                            let key = output_item_tool_key(&item);
                            let id = output_item_tool_id(&item);
                            let name = output_item_tool_name(&item);
                            // Also register under item id so that providers (e.g. Ollama) that
                            // use item_id (not call_id) in delta/done events can resolve the name.
                            if let Some(item_id) = &item.id
                                && !item_id.is_empty() && *item_id != key {
                                    tool_calls.observe_call(item_id.clone(), id.clone(), name.clone());
                                }
                            tool_calls.observe_call(key, id, name);
                        }
                    }
                    SseEvent::ResponseOutputItemDone(event) => {
                        if let Some(item) = response_output_item_function_call(event.item) {
                            saw_tool_call = true;
                            if let Some(invocation) = tool_calls.finish(
                                output_item_tool_key(&item),
                                output_item_tool_id(&item),
                                Some(output_item_tool_name(&item)),
                                Some(item.arguments),
                            )? {
                                flush_buffered_content(&mut pending_item, &mut committed_items);
                                push_committed_tool_call(&mut committed_items, &invocation);
                                yield ErasedTextTurnEvent::ToolCallReady(invocation);
                            }
                        }
                    }
                    SseEvent::ResponseCompleted(event) => {
                        request_id = Some(event.response.id.clone());
                        flush_buffered_content(&mut pending_item, &mut committed_items);
                        let finish_reason = event
                            .response
                            .stop_reason
                            .as_deref()
                            .or(event.response.finish_reason.as_deref())
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
                        let usage = parse_response_usage_value(&event.response.usage);
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
                    SseEvent::ResponseInProgress(_)
                    | SseEvent::ResponseContentPartAdded(_)
                    | SseEvent::ResponseContentPartDone(_)
                    | SseEvent::ResponseReasoningSummaryTextDone(_)
                    | SseEvent::Unknown => {}
                    SseEvent::ResponseCreated(_) => {}
                }
            }
        }
    }
}

fn map_structured_stream<S>(
    stream: S,
    fallback_model: String,
    recovery_hook: Option<Arc<dyn SseEventRecoveryHook>>,
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
                let event = decode_sse_event(&payload, recovery_hook.as_ref(), &tool_calls)?;
                if let SseEvent::ResponseCreated(created) = &event {
                    request_id = Some(created.response.id.clone());
                    if let Some(event_model) = created.response.model.as_ref() {
                        model = event_model.clone();
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

                match event {
                    SseEvent::ResponseOutputTextDelta(event) => {
                        structured_buffer.push_str(&event.delta);
                        push_buffered_content(
                            &mut pending_item,
                            &mut committed_items,
                            BufferedTurnItemKind::Text,
                            &event.item_id,
                            &event.delta,
                        );
                        yield ErasedStructuredTurnEvent::StructuredOutputChunk {
                            json_delta: event.delta,
                        };
                    }
                    SseEvent::ResponseOutputTextDone(event) => {
                        structured_buffer.clear();
                        structured_buffer.push_str(&event.text);
                        replace_buffered_text(
                            &mut pending_item,
                            &mut committed_items,
                            &event.item_id,
                            &event.text,
                        );
                        if let Some(value) =
                            maybe_parse_structured_output(&structured_buffer, &mut emitted_ready)?
                        {
                            yield ErasedStructuredTurnEvent::StructuredOutputReady(value);
                        }
                    }
                    SseEvent::ResponseReasoningSummaryTextDelta(event) => {
                        push_buffered_content(
                            &mut pending_item,
                            &mut committed_items,
                            BufferedTurnItemKind::Reasoning,
                            event.item_id.as_deref().unwrap_or(""),
                            &event.delta,
                        );
                        yield ErasedStructuredTurnEvent::ReasoningDelta {
                            delta: event.delta,
                        };
                    }
                    SseEvent::ResponseReasoningDelta(event) => {
                        push_buffered_content(
                            &mut pending_item,
                            &mut committed_items,
                            BufferedTurnItemKind::Reasoning,
                            event.item_id.as_deref().unwrap_or(""),
                            &event.delta,
                        );
                        yield ErasedStructuredTurnEvent::ReasoningDelta {
                            delta: event.delta,
                        };
                    }
                    SseEvent::ResponseRefusalDelta(event) => {
                        saw_refusal = true;
                        push_buffered_content(
                            &mut pending_item,
                            &mut committed_items,
                            BufferedTurnItemKind::Refusal,
                            event.item_id.as_deref().unwrap_or(""),
                            &event.delta,
                        );
                        yield ErasedStructuredTurnEvent::RefusalDelta {
                            delta: event.delta,
                        };
                    }
                    SseEvent::ResponseFunctionCallArgumentsDelta(event) => {
                        let key = response_tool_key_from_delta(&event);
                        if let Some((id, name, delta)) = tool_calls.record_delta(
                            key.clone(),
                            response_tool_id_from_delta(&event),
                            tool_calls.peek_name(&key).cloned(),
                            &event.delta,
                        ) {
                            yield ErasedStructuredTurnEvent::ToolCallChunk {
                                id,
                                name,
                                arguments_json_delta: delta,
                            };
                        }
                    }
                    SseEvent::ResponseFunctionCallArgumentsDone(event) => {
                        saw_tool_call = true;
                        if let Some(invocation) = tool_calls.finish(
                            response_tool_key_from_done(&event),
                            response_tool_id_from_done(&event),
                            event.name.as_deref().map(ToolName::from),
                            event.arguments.clone(),
                        )? {
                            flush_buffered_content(&mut pending_item, &mut committed_items);
                            push_committed_tool_call(&mut committed_items, &invocation);
                            yield ErasedStructuredTurnEvent::ToolCallReady(invocation);
                        }
                    }
                    SseEvent::ResponseOutputItemAdded(event) => {
                        if let Some(item) = response_output_item_function_call(event.item) {
                            let key = output_item_tool_key(&item);
                            let id = output_item_tool_id(&item);
                            let name = output_item_tool_name(&item);
                            if let Some(item_id) = &item.id
                                && !item_id.is_empty() && *item_id != key {
                                    tool_calls.observe_call(item_id.clone(), id.clone(), name.clone());
                                }
                            tool_calls.observe_call(key, id, name);
                        }
                    }
                    SseEvent::ResponseOutputItemDone(event) => {
                        if let Some(item) = response_output_item_function_call(event.item) {
                            saw_tool_call = true;
                            if let Some(invocation) = tool_calls.finish(
                                output_item_tool_key(&item),
                                output_item_tool_id(&item),
                                Some(output_item_tool_name(&item)),
                                Some(item.arguments),
                            )? {
                                flush_buffered_content(&mut pending_item, &mut committed_items);
                                push_committed_tool_call(&mut committed_items, &invocation);
                                yield ErasedStructuredTurnEvent::ToolCallReady(invocation);
                            }
                        }
                    }
                    SseEvent::ResponseCompleted(event) => {
                        request_id = Some(event.response.id.clone());
                        if let Some(value) =
                            maybe_parse_structured_output(&structured_buffer, &mut emitted_ready)?
                        {
                            yield ErasedStructuredTurnEvent::StructuredOutputReady(value);
                        }
                        flush_buffered_content(&mut pending_item, &mut committed_items);
                        let finish_reason = event
                            .response
                            .stop_reason
                            .as_deref()
                            .or(event.response.finish_reason.as_deref())
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
                        let usage = parse_response_usage_value(&event.response.usage);
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
                    SseEvent::ResponseInProgress(_)
                    | SseEvent::ResponseContentPartAdded(_)
                    | SseEvent::ResponseContentPartDone(_)
                    | SseEvent::ResponseReasoningSummaryTextDone(_)
                    | SseEvent::Unknown => {}
                    SseEvent::ResponseCreated(_) => {}
                }
            }
        }
    }
}

#[derive(Deserialize)]
struct CompletionChunk {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<CompletionChoice>,
    #[serde(default)]
    usage: Option<CompletionUsage>,
}

#[derive(Deserialize)]
struct CompletionChoice {
    #[serde(default)]
    text: String,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct CompletionUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
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
                let event = serde_json::from_str::<CompletionChunk>(&payload)?;
                if !started {
                    request_id = event.id.clone();
                    if let Some(event_model) = event.model.clone() {
                        model = event_model;
                    }
                    started = true;
                    yield CompletionEvent::Started {
                        request_id: request_id.clone(),
                        model: model.clone(),
                    };
                }

                if let Some(choice) = event.choices.first() {
                    if !choice.text.is_empty() {
                        yield CompletionEvent::TextDelta(choice.text.clone());
                    }
                    if let Some(finish_reason) = choice.finish_reason.as_deref() {
                        finished = true;
                        yield CompletionEvent::Completed {
                            request_id: request_id.clone(),
                            finish_reason: map_completion_finish_reason(finish_reason),
                            usage: last_usage,
                        };
                    }
                }
                if let Some(usage) = event.usage {
                    last_usage = parse_completion_usage(&usage);
                }
            }
        }
    }
}

fn parse_response_usage(value: &Value) -> Usage {
    let usage = value.get("usage").unwrap_or(value);
    parse_response_usage_value(usage)
}

fn parse_response_usage_value(usage: &Value) -> Usage {
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

fn parse_completion_usage(value: &CompletionUsage) -> Usage {
    Usage {
        input_tokens: value.prompt_tokens,
        output_tokens: value.total_tokens.saturating_sub(value.prompt_tokens),
        total_tokens: value.total_tokens,
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::{StreamExt, executor::block_on};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    use lutum_protocol::{
        AdapterToolChoice, AdapterToolDefinition, AdapterTurnConfig, AssistantInputItem,
        AssistantTurnItem, AssistantTurnView, ErasedStructuredTurnEvent, ErasedTextTurnEvent,
        GenerationParams, HookRegistry, InputMessageRole, ModelInput, ModelInputItem, ModelName,
        RequestExtensions, ToolResult,
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

    struct TaggingFallbackSerializer;

    impl FallbackSerializer for TaggingFallbackSerializer {
        fn apply_to_responses(&self, request: &mut crate::responses::ResponsesRequest) {
            request.models = Some(vec!["fallback".to_string()]);
        }

        fn apply_to_completion(&self, request: &mut CompletionRequest) {
            request.models = Some(vec!["fallback".to_string()]);
        }
    }

    #[lutum_macros::hook(SelectOpenaiModel)]
    async fn prefer_gpt_4_1(_extensions: &RequestExtensions, _default: ModelName) -> ModelName {
        ModelName::new("gpt-4.1").unwrap()
    }

    #[lutum_macros::hook(SelectOpenaiModel)]
    async fn prefer_gpt_4_1_mini(
        _extensions: &RequestExtensions,
        _default: ModelName,
    ) -> ModelName {
        ModelName::new("gpt-4.1-mini").unwrap()
    }

    #[test]
    fn prepare_responses_request_uses_explicit_model() {
        let adapter = OpenAiAdapter::new("test-key");

        let input =
            ModelInput::from_items(vec![ModelInputItem::text(InputMessageRole::User, "hello")]);
        let config = AdapterTurnConfig {
            generation: GenerationParams::default(),
            tools: Vec::new(),
            tool_choice: AdapterToolChoice::Auto,
        };

        let request = adapter
            .prepare_responses_request(&input, &config, "gpt-4.1-override", None, None)
            .unwrap();

        assert_eq!(request.model, "gpt-4.1-override");

        let payloads = vec![Ok(Bytes::from(
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4.1-override\",\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
        ))];
        let events = block_on(async {
            map_text_stream(
                futures::stream::iter(payloads),
                "gpt-4.1-override".to_string(),
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
            ErasedTextTurnEvent::Started { model, .. } if model == "gpt-4.1-override"
        ));
    }

    #[test]
    fn fallback_serializer_is_applied_to_requests() {
        let mut adapter = OpenAiAdapter::new("test-key");
        adapter.set_fallback_serializer(Box::new(TaggingFallbackSerializer));

        let input =
            ModelInput::from_items(vec![ModelInputItem::text(InputMessageRole::User, "hello")]);
        let config = AdapterTurnConfig {
            generation: GenerationParams::default(),
            tools: Vec::new(),
            tool_choice: AdapterToolChoice::Auto,
        };

        let responses_request = adapter
            .prepare_responses_request(&input, &config, "gpt-4.1", None, None)
            .unwrap();
        let completion_request =
            adapter.prepare_completion_request(&ProtocolCompletionRequest::new("hello"), "gpt-4.1");

        assert_eq!(responses_request.models, Some(vec!["fallback".to_string()]));
        assert_eq!(
            completion_request.models,
            Some(vec!["fallback".to_string()])
        );
    }

    #[test]
    fn select_openai_model_uses_last_registered_singleton_override() {
        let hooks = HookRegistry::new()
            .register_select_openai_model(PreferGpt41)
            .register_select_openai_model(PreferGpt41Mini);

        let selected = block_on(hooks.select_openai_model(
            &RequestExtensions::new(),
            ModelName::new("gpt-4.1-nano").unwrap(),
        ));

        assert_eq!(selected.as_str(), "gpt-4.1-mini");
    }

    #[test]
    fn prepare_responses_request_propagates_seed() {
        let adapter = OpenAiAdapter::new("test-key");
        let input =
            ModelInput::from_items(vec![ModelInputItem::text(InputMessageRole::User, "hello")]);
        let config = AdapterTurnConfig {
            generation: GenerationParams {
                temperature: None,
                max_output_tokens: Some(128),
                seed: Some(42),
            },
            tools: Vec::new(),
            tool_choice: AdapterToolChoice::Auto,
        };

        let request = adapter
            .prepare_responses_request(&input, &config, "gpt-4.1", None, None)
            .unwrap();

        assert_eq!(request.seed, Some(42));
    }

    #[test]
    fn convert_model_input_preserves_ordered_system_developer_and_assistant_items() {
        let input = ModelInput::from_items(vec![
            ModelInputItem::text(InputMessageRole::System, "policy"),
            ModelInputItem::text(InputMessageRole::Developer, "dev"),
            ModelInputItem::text(InputMessageRole::User, "hello"),
            ModelInputItem::Assistant(AssistantInputItem::Reasoning("think".into())),
            ModelInputItem::Assistant(AssistantInputItem::Text("hi".into())),
            ModelInputItem::ToolResult(ToolResult::new(
                "call-1",
                "weather",
                RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                RawJson::parse("\"sunny\"").unwrap(),
            )),
        ]);

        let items = convert_model_input(&input).unwrap();
        assert!(matches!(
            &items[0],
            InputItem::Message(InputMessage {
                role: MessageRole::System,
                ..
            })
        ));
        assert!(matches!(
            &items[1],
            InputItem::Message(InputMessage {
                role: MessageRole::Developer,
                ..
            })
        ));
        assert!(matches!(
            &items[2],
            InputItem::Message(InputMessage {
                role: MessageRole::User,
                ..
            })
        ));
        assert!(matches!(&items[3], InputItem::Reasoning(_)));
        assert!(matches!(&items[4], InputItem::Message(_)));
        assert!(matches!(&items[5], InputItem::FunctionCall(_)));
        assert!(matches!(&items[6], InputItem::FunctionCallOutput(_)));
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
            ModelInputItem::ToolResult(ToolResult::new(
                "call-1",
                "weather",
                RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                RawJson::parse("\"sunny\"").unwrap(),
            )),
        ]);

        let items = convert_model_input(&input).unwrap();
        assert!(matches!(&items[0], InputItem::Message(_)));
        assert!(matches!(&items[1], InputItem::Reasoning(_)));
        assert!(matches!(&items[2], InputItem::Message(_)));
        assert!(matches!(&items[3], InputItem::FunctionCall(_)));
        assert!(matches!(&items[4], InputItem::FunctionCallOutput(_)));
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
        assert!(matches!(&items[0], InputItem::Reasoning(_)));
        assert!(matches!(&items[1], InputItem::Message(_)));
        assert!(matches!(&items[2], InputItem::FunctionCall(_)));
    }

    #[test]
    fn responses_sse_maps_reasoning_refusal_tool_and_completion() {
        let payloads = vec![
            Ok(Bytes::from(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":null}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"thinking\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.refusal.delta\",\"delta\":\"no\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.function_call_arguments.done\",\"call_id\":\"call-1\",\"output_index\":0,\"sequence_number\":0,\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\\\"Tokyo\\\"}\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )),
        ];

        let events = block_on(async {
            map_text_stream(futures::stream::iter(payloads), "gpt-4.1".into(), None)
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
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_2\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":null}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"{\\\"answer\\\":\\\"42\\\"}\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_2\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )),
        ];

        let events = block_on(async {
            map_structured_stream(futures::stream::iter(payloads), "gpt-4.1".into(), None)
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
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_tool\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":null}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"item-1\",\"call_id\":\"call-1\",\"output_index\":0,\"sequence_number\":0,\"delta\":\"{\\\"city\\\":\\\"Tokyo\\\"}\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"item-1\",\"call_id\":\"call-1\",\"output_index\":0,\"sequence_number\":0,\"name\":\"weather\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"sequence_number\":0,\"item\":{\"type\":\"function_call\",\"id\":\"item-1\",\"call_id\":\"call-1\",\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\\\"Tokyo\\\"}\"}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_tool\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )),
        ];

        let events = block_on(async {
            map_text_stream(futures::stream::iter(payloads), "gpt-4.1".into(), None)
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
    fn responses_sse_done_without_name_uses_buffered_tool_name() {
        let payloads = vec![
            Ok(Bytes::from(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_optional_name\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":null}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"sequence_number\":0,\"item\":{\"type\":\"function_call\",\"id\":\"item-1\",\"call_id\":\"call-1\",\"name\":\"weather\",\"arguments\":\"\"}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"item-1\",\"call_id\":\"call-1\",\"output_index\":0,\"sequence_number\":1,\"arguments\":\"{\\\"city\\\":\\\"Tokyo\\\"}\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_optional_name\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )),
        ];

        let events = block_on(async {
            map_text_stream(futures::stream::iter(payloads), "gpt-4.1".into(), None)
                .collect::<Vec<_>>()
                .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        let ready: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                ErasedTextTurnEvent::ToolCallReady(invocation) => Some(invocation),
                _ => None,
            })
            .collect();
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
    fn responses_sse_recovery_hook_can_restore_missing_tool_name() {
        struct TestRecoveryHook;

        impl SseEventRecoveryHook for TestRecoveryHook {
            fn recover_event(
                &self,
                payload: &str,
                _error: &serde_json::Error,
                hints: &SseHints,
            ) -> Result<Option<SseEvent>, OpenAiError> {
                #[derive(Deserialize)]
                #[serde(tag = "type")]
                enum Wire {
                    #[serde(rename = "response.function_call_arguments.done")]
                    Done {
                        #[serde(default)]
                        item_id: Option<String>,
                        #[serde(default)]
                        call_id: Option<String>,
                        #[serde(default)]
                        output_index: usize,
                        #[serde(default)]
                        sequence_number: u64,
                        #[serde(default)]
                        arguments: Option<String>,
                    },
                }

                let Ok(wire) = serde_json::from_str::<Wire>(payload) else {
                    return Ok(None);
                };

                match wire {
                    Wire::Done {
                        item_id,
                        call_id,
                        output_index,
                        sequence_number,
                        arguments,
                    } => {
                        let key = call_id
                            .as_deref()
                            .or(item_id.as_deref())
                            .unwrap_or_default();
                        let Some(name) = hints.tool_name_for(key).map(ToOwned::to_owned) else {
                            return Ok(None);
                        };

                        use crate::responses::ResponseFunctionCallArgumentsDoneEvent;

                        Ok(Some(SseEvent::ResponseFunctionCallArgumentsDone(
                            ResponseFunctionCallArgumentsDoneEvent {
                                item_id,
                                call_id,
                                output_index,
                                sequence_number,
                                name: Some(name),
                                arguments,
                                event_type: Default::default(),
                            },
                        )))
                    }
                }
            }
        }

        let payloads = vec![
            Ok(Bytes::from(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_recover\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":null}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"sequence_number\":0,\"item\":{\"type\":\"function_call\",\"id\":\"item-1\",\"call_id\":\"call-1\",\"name\":\"weather\",\"arguments\":\"\"}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.function_call_arguments.done\",\"item_id\":\"item-1\",\"call_id\":\"call-1\",\"output_index\":0,\"sequence_number\":1,\"arguments\":\"{\\\"city\\\":\\\"Tokyo\\\"}\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_recover\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )),
        ];

        let events = block_on(async {
            map_text_stream(
                futures::stream::iter(payloads),
                "gpt-4.1".into(),
                Some(Arc::new(TestRecoveryHook)),
            )
            .collect::<Vec<_>>()
            .await
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

        let ready: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                ErasedTextTurnEvent::ToolCallReady(invocation) => Some(invocation),
                _ => None,
            })
            .collect();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].name.as_str(), "weather");
    }

    #[test]
    fn responses_sse_keeps_separate_text_items_by_item_id() {
        let payloads = vec![
            Ok(Bytes::from(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_text_items\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":null}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"text_item_0\",\"output_index\":0,\"content_index\":0,\"delta\":\"hel\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.done\",\"item_id\":\"text_item_0\",\"output_index\":0,\"content_index\":0,\"text\":\"hello\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"text_item_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"bye\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.done\",\"item_id\":\"text_item_1\",\"output_index\":0,\"content_index\":0,\"text\":\"bye!\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_text_items\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )),
        ];

        let events = block_on(async {
            map_text_stream(futures::stream::iter(payloads), "gpt-4.1".into(), None)
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
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_scalar\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":null}}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"1\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.delta\",\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,\"delta\":\"2\"}\n\n",
            )),
            Ok(Bytes::from(
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_scalar\",\"model\":\"gpt-4.1\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":2,\"total_tokens\":3}}}\n\n",
            )),
        ];

        let events = block_on(async {
            map_structured_stream(futures::stream::iter(payloads), "gpt-4.1".into(), None)
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

    #[test]
    fn parses_response_created_event_through_typed_enum() {
        let payload = r#"{"type":"response.created","response":{"id":"resp_1","model":"gpt-4.1","output":[],"usage":null}}"#;
        let event = serde_json::from_str::<SseEvent>(payload).unwrap();
        assert!(matches!(event, SseEvent::ResponseCreated(_)));
    }

    #[test]
    fn function_call_item_accepts_string_arguments_from_sse() {
        let value = serde_json::from_str::<Value>(
            r#"{"type":"function_call","id":"fc_1","call_id":"call_1","name":"weather","arguments":"{\"city\":\"Tokyo\"}"}"#,
        )
        .unwrap();
        let item = serde_json::from_value::<FunctionCallItem>(value).unwrap();
        assert_eq!(item.arguments, "{\"city\":\"Tokyo\"}".to_string());
    }
}
