use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
};

use futures::stream;

use thiserror::Error;

use lutum_protocol::{
    AgentError, RequestExtensions,
    budget::Usage,
    conversation::{AssistantTurnItem, ModelInput, RawJson, ToolCallId, ToolMetadata, ToolName},
    hooks::HookRegistry,
    llm::{
        AdapterStructuredCompletionRequest, AdapterStructuredTurn, AdapterTextTurn,
        CompletionAdapter, CompletionEvent, CompletionEventStream, CompletionRequest,
        ErasedStructuredCompletionEvent, ErasedStructuredCompletionEventStream,
        ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
        ErasedTextTurnEventStream, FinishReason, OperationKind, TurnAdapter, UsageRecoveryAdapter,
    },
    transcript::AssistantTurnView,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RawTextTurnEvent {
    Started {
        request_id: Option<String>,
        model: String,
    },
    TextDelta {
        delta: String,
    },
    ReasoningDelta {
        delta: String,
    },
    RefusalDelta {
        delta: String,
    },
    ToolCallChunk {
        id: ToolCallId,
        name: ToolName,
        arguments_json_delta: String,
    },
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RawStructuredTurnEvent {
    Started {
        request_id: Option<String>,
        model: String,
    },
    StructuredOutputChunk {
        json_delta: String,
    },
    ReasoningDelta {
        delta: String,
    },
    RefusalDelta {
        delta: String,
    },
    ToolCallChunk {
        id: ToolCallId,
        name: ToolName,
        arguments_json_delta: String,
    },
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RawCompletionEvent {
    Started {
        request_id: Option<String>,
        model: String,
    },
    TextDelta(String),
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RawStructuredCompletionEvent {
    Started {
        request_id: Option<String>,
        model: String,
    },
    StructuredOutputChunk {
        json_delta: String,
    },
    ReasoningDelta {
        delta: String,
    },
    RefusalDelta {
        delta: String,
    },
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
    },
}

#[derive(Clone, Debug)]
pub struct MockTextScenario {
    events: Vec<Result<RawTextTurnEvent, MockError>>,
}

impl MockTextScenario {
    pub fn events(events: Vec<Result<RawTextTurnEvent, MockError>>) -> Self {
        Self { events }
    }
}

#[derive(Clone, Debug)]
pub struct MockStructuredScenario {
    events: Vec<Result<RawStructuredTurnEvent, MockError>>,
}

impl MockStructuredScenario {
    pub fn events(events: Vec<Result<RawStructuredTurnEvent, MockError>>) -> Self {
        Self { events }
    }
}

#[derive(Clone, Debug)]
pub struct MockCompletionScenario {
    events: Vec<Result<RawCompletionEvent, MockError>>,
}

impl MockCompletionScenario {
    pub fn events(events: Vec<Result<RawCompletionEvent, MockError>>) -> Self {
        Self { events }
    }
}

#[derive(Clone, Debug)]
pub struct MockStructuredCompletionScenario {
    events: Vec<Result<RawStructuredCompletionEvent, MockError>>,
}

impl MockStructuredCompletionScenario {
    pub fn events(events: Vec<Result<RawStructuredCompletionEvent, MockError>>) -> Self {
        Self { events }
    }
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum MockError {
    #[error("no mock text scenario configured")]
    MissingTextScenario,
    #[error("no mock structured scenario configured")]
    MissingStructuredScenario,
    #[error("no mock completion scenario configured")]
    MissingCompletionScenario,
    #[error("no mock structured completion scenario configured")]
    MissingStructuredCompletionScenario,
    #[error("failed to deserialize structured output: {0}")]
    StructuredOutput(String),
    #[error("failed to deserialize tool call: {0}")]
    ToolCall(String),
    #[error("synthetic adapter failure: {message}")]
    Synthetic { message: String },
}

#[derive(Clone, Default)]
pub struct MockLlmAdapter {
    text_turns: Arc<Mutex<VecDeque<MockTextScenario>>>,
    structured_turns: Arc<Mutex<VecDeque<MockStructuredScenario>>>,
    completions: Arc<Mutex<VecDeque<MockCompletionScenario>>>,
    structured_completions: Arc<Mutex<VecDeque<MockStructuredCompletionScenario>>>,
    recovered_usage: Arc<Mutex<BTreeMap<(OperationKind, String), Usage>>>,
    recover_usage_errors: Arc<Mutex<BTreeMap<(OperationKind, String), MockError>>>,
}

impl MockLlmAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_text_scenario(self, scenario: MockTextScenario) -> Self {
        self.text_turns.lock().unwrap().push_back(scenario);
        self
    }

    pub fn with_structured_scenario(self, scenario: MockStructuredScenario) -> Self {
        self.structured_turns.lock().unwrap().push_back(scenario);
        self
    }

    pub fn with_completion_scenario(self, scenario: MockCompletionScenario) -> Self {
        self.completions.lock().unwrap().push_back(scenario);
        self
    }

    pub fn with_structured_completion_scenario(
        self,
        scenario: MockStructuredCompletionScenario,
    ) -> Self {
        self.structured_completions
            .lock()
            .unwrap()
            .push_back(scenario);
        self
    }

    pub fn with_recovered_usage(
        self,
        kind: OperationKind,
        request_id: impl Into<String>,
        usage: Usage,
    ) -> Self {
        self.recovered_usage
            .lock()
            .unwrap()
            .insert((kind, request_id.into()), usage);
        self
    }

    pub fn with_recover_usage_error(
        self,
        kind: OperationKind,
        request_id: impl Into<String>,
        error: MockError,
    ) -> Self {
        self.recover_usage_errors
            .lock()
            .unwrap()
            .insert((kind, request_id.into()), error);
        self
    }
}

#[async_trait::async_trait]
impl TurnAdapter for MockLlmAdapter {
    async fn text_turn(
        &self,
        _input: ModelInput,
        _turn: AdapterTextTurn,
        _hooks: &HookRegistry,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        let scenario = self
            .text_turns
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| AgentError::backend(MockError::MissingTextScenario))?;
        let mut tool_buffers: BTreeMap<ToolCallId, (ToolName, String)> = BTreeMap::new();
        let mut committed_items = Vec::<AssistantTurnItem>::new();

        let events = scenario
            .events
            .into_iter()
            .flat_map(move |event| match event {
                Ok(RawTextTurnEvent::Started { request_id, model }) => {
                    vec![Ok(ErasedTextTurnEvent::Started { request_id, model })]
                }
                Ok(RawTextTurnEvent::TextDelta { delta }) => {
                    push_or_extend_text(&mut committed_items, &delta);
                    vec![Ok(ErasedTextTurnEvent::TextDelta { delta })]
                }
                Ok(RawTextTurnEvent::ReasoningDelta { delta }) => {
                    push_or_extend_reasoning(&mut committed_items, &delta);
                    vec![Ok(ErasedTextTurnEvent::ReasoningDelta { delta })]
                }
                Ok(RawTextTurnEvent::RefusalDelta { delta }) => {
                    push_or_extend_refusal(&mut committed_items, &delta);
                    vec![Ok(ErasedTextTurnEvent::RefusalDelta { delta })]
                }
                Ok(RawTextTurnEvent::ToolCallChunk {
                    id,
                    name,
                    arguments_json_delta,
                }) => {
                    let entry = tool_buffers
                        .entry(id.clone())
                        .or_insert_with(|| (name.clone(), String::new()));
                    entry.1.push_str(&arguments_json_delta);
                    let mut out = vec![Ok(ErasedTextTurnEvent::ToolCallChunk {
                        id: id.clone(),
                        name: name.clone(),
                        arguments_json_delta,
                    })];
                    if is_complete_json(&entry.1) {
                        let arguments =
                            RawJson::parse(entry.1.clone()).expect("complete mock JSON is valid");
                        push_tool_call(
                            &mut committed_items,
                            id.clone(),
                            entry.0.clone(),
                            arguments.clone(),
                        );
                        out.push(Ok(ErasedTextTurnEvent::ToolCallReady(ToolMetadata::new(
                            id.clone(),
                            entry.0.clone(),
                            arguments,
                        ))));
                        tool_buffers.remove(&id);
                    }
                    out
                }
                Ok(RawTextTurnEvent::Completed {
                    request_id,
                    finish_reason,
                    usage,
                }) => vec![Ok(ErasedTextTurnEvent::Completed {
                    request_id,
                    finish_reason,
                    usage,
                    committed_turn: Arc::new(AssistantTurnView::from_items(&committed_items)),
                })],
                Err(err) => vec![Err(AgentError::backend(err))],
            })
            .collect::<Vec<_>>();

        Ok(Box::pin(stream::iter(events)) as ErasedTextTurnEventStream)
    }

    async fn structured_turn(
        &self,
        _input: ModelInput,
        _turn: AdapterStructuredTurn,
        _hooks: &HookRegistry,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
        let scenario = self
            .structured_turns
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| AgentError::backend(MockError::MissingStructuredScenario))?;
        let mut structured_buffer = String::new();
        let mut tool_buffers: BTreeMap<ToolCallId, (ToolName, String)> = BTreeMap::new();
        let mut committed_items = Vec::<AssistantTurnItem>::new();

        let events = scenario
            .events
            .into_iter()
            .flat_map(move |event| match event {
                Ok(RawStructuredTurnEvent::Started { request_id, model }) => {
                    vec![Ok(ErasedStructuredTurnEvent::Started { request_id, model })]
                }
                Ok(RawStructuredTurnEvent::StructuredOutputChunk { json_delta }) => {
                    structured_buffer.push_str(&json_delta);
                    push_or_extend_text(&mut committed_items, &json_delta);
                    vec![Ok(ErasedStructuredTurnEvent::StructuredOutputChunk {
                        json_delta,
                    })]
                }
                Ok(RawStructuredTurnEvent::ReasoningDelta { delta }) => {
                    push_or_extend_reasoning(&mut committed_items, &delta);
                    vec![Ok(ErasedStructuredTurnEvent::ReasoningDelta { delta })]
                }
                Ok(RawStructuredTurnEvent::RefusalDelta { delta }) => {
                    push_or_extend_refusal(&mut committed_items, &delta);
                    vec![Ok(ErasedStructuredTurnEvent::RefusalDelta { delta })]
                }
                Ok(RawStructuredTurnEvent::ToolCallChunk {
                    id,
                    name,
                    arguments_json_delta,
                }) => {
                    let entry = tool_buffers
                        .entry(id.clone())
                        .or_insert_with(|| (name.clone(), String::new()));
                    entry.1.push_str(&arguments_json_delta);
                    let mut out = vec![Ok(ErasedStructuredTurnEvent::ToolCallChunk {
                        id: id.clone(),
                        name: name.clone(),
                        arguments_json_delta,
                    })];
                    if is_complete_json(&entry.1) {
                        let arguments =
                            RawJson::parse(entry.1.clone()).expect("complete mock JSON is valid");
                        push_tool_call(
                            &mut committed_items,
                            id.clone(),
                            entry.0.clone(),
                            arguments.clone(),
                        );
                        out.push(Ok(ErasedStructuredTurnEvent::ToolCallReady(
                            ToolMetadata::new(id.clone(), entry.0.clone(), arguments),
                        )));
                        tool_buffers.remove(&id);
                    }
                    out
                }
                Ok(RawStructuredTurnEvent::Completed {
                    request_id,
                    finish_reason,
                    usage,
                }) => {
                    let mut out = Vec::new();
                    if !structured_buffer.is_empty() {
                        match RawJson::parse(structured_buffer.clone()) {
                            Ok(value) => out
                                .push(Ok(ErasedStructuredTurnEvent::StructuredOutputReady(value))),
                            Err(err) => out.push(Err(AgentError::structured_output(err))),
                        }
                    }
                    out.push(Ok(ErasedStructuredTurnEvent::Completed {
                        request_id,
                        finish_reason,
                        usage,
                        committed_turn: Arc::new(AssistantTurnView::from_items(&committed_items)),
                    }));
                    out
                }
                Err(err) => vec![Err(AgentError::backend(err))],
            })
            .collect::<Vec<_>>();

        Ok(Box::pin(stream::iter(events)) as ErasedStructuredTurnEventStream)
    }
}

#[async_trait::async_trait]
impl CompletionAdapter for MockLlmAdapter {
    async fn completion(
        &self,
        _request: CompletionRequest,
        _extensions: &RequestExtensions,
        _hooks: &HookRegistry,
    ) -> Result<CompletionEventStream, AgentError> {
        let scenario = self
            .completions
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| AgentError::backend(MockError::MissingCompletionScenario))?;
        let events = scenario.events.into_iter().map(|event| match event {
            Ok(RawCompletionEvent::Started { request_id, model }) => {
                Ok(CompletionEvent::Started { request_id, model })
            }
            Ok(RawCompletionEvent::TextDelta(delta)) => Ok(CompletionEvent::TextDelta(delta)),
            Ok(RawCompletionEvent::Completed {
                request_id,
                finish_reason,
                usage,
            }) => Ok(CompletionEvent::Completed {
                request_id,
                finish_reason,
                usage,
            }),
            Err(err) => Err(AgentError::backend(err)),
        });

        Ok(Box::pin(stream::iter(events.collect::<Vec<_>>())) as CompletionEventStream)
    }

    async fn structured_completion(
        &self,
        _request: AdapterStructuredCompletionRequest,
        _extensions: &RequestExtensions,
        _hooks: &HookRegistry,
    ) -> Result<ErasedStructuredCompletionEventStream, AgentError> {
        let scenario = self
            .structured_completions
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| AgentError::backend(MockError::MissingStructuredCompletionScenario))?;
        let mut structured_buffer = String::new();
        let events = scenario.events.into_iter().flat_map(|event| match event {
            Ok(RawStructuredCompletionEvent::Started { request_id, model }) => {
                vec![Ok(ErasedStructuredCompletionEvent::Started {
                    request_id,
                    model,
                })]
            }
            Ok(RawStructuredCompletionEvent::StructuredOutputChunk { json_delta }) => {
                structured_buffer.push_str(&json_delta);
                let mut out = vec![Ok(ErasedStructuredCompletionEvent::StructuredOutputChunk {
                    json_delta,
                })];
                if is_complete_json(&structured_buffer) {
                    match RawJson::parse(structured_buffer.clone()) {
                        Ok(value) => out.push(Ok(
                            ErasedStructuredCompletionEvent::StructuredOutputReady(value),
                        )),
                        Err(err) => out.push(Err(AgentError::structured_output(err))),
                    }
                }
                out
            }
            Ok(RawStructuredCompletionEvent::ReasoningDelta { delta }) => {
                vec![Ok(ErasedStructuredCompletionEvent::ReasoningDelta {
                    delta,
                })]
            }
            Ok(RawStructuredCompletionEvent::RefusalDelta { delta }) => {
                vec![Ok(ErasedStructuredCompletionEvent::RefusalDelta { delta })]
            }
            Ok(RawStructuredCompletionEvent::Completed {
                request_id,
                finish_reason,
                usage,
            }) => vec![Ok(ErasedStructuredCompletionEvent::Completed {
                request_id,
                finish_reason,
                usage,
            })],
            Err(err) => vec![Err(AgentError::backend(err))],
        });

        Ok(Box::pin(stream::iter(events.collect::<Vec<_>>()))
            as ErasedStructuredCompletionEventStream)
    }
}

#[async_trait::async_trait]
impl UsageRecoveryAdapter for MockLlmAdapter {
    async fn recover_usage(
        &self,
        kind: OperationKind,
        request_id: &str,
    ) -> Result<Option<Usage>, AgentError> {
        if let Some(err) = self
            .recover_usage_errors
            .lock()
            .unwrap()
            .get(&(kind, request_id.to_string()))
            .cloned()
        {
            return Err(AgentError::backend(err));
        }

        Ok(self
            .recovered_usage
            .lock()
            .unwrap()
            .get(&(kind, request_id.to_string()))
            .copied())
    }
}

fn is_complete_json(input: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(input).is_ok()
}

fn push_or_extend_text(items: &mut Vec<AssistantTurnItem>, delta: &str) {
    if delta.is_empty() {
        return;
    }
    match items.last_mut() {
        Some(AssistantTurnItem::Text(existing)) => existing.push_str(delta),
        _ => items.push(AssistantTurnItem::Text(delta.to_string())),
    }
}

fn push_or_extend_reasoning(items: &mut Vec<AssistantTurnItem>, delta: &str) {
    if delta.is_empty() {
        return;
    }
    match items.last_mut() {
        Some(AssistantTurnItem::Reasoning(existing)) => existing.push_str(delta),
        _ => items.push(AssistantTurnItem::Reasoning(delta.to_string())),
    }
}

fn push_or_extend_refusal(items: &mut Vec<AssistantTurnItem>, delta: &str) {
    if delta.is_empty() {
        return;
    }
    match items.last_mut() {
        Some(AssistantTurnItem::Refusal(existing)) => existing.push_str(delta),
        _ => items.push(AssistantTurnItem::Refusal(delta.to_string())),
    }
}

fn push_tool_call(
    items: &mut Vec<AssistantTurnItem>,
    id: ToolCallId,
    name: ToolName,
    arguments: RawJson,
) {
    items.push(AssistantTurnItem::ToolCall {
        id,
        name,
        arguments,
    });
}
