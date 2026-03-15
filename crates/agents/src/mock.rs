use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
};

use futures::stream;

use thiserror::Error;

use agents_protocol::{
    AgentError,
    budget::Usage,
    conversation::{ModelInput, RawJson, ToolCallId, ToolMetadata, ToolName},
    llm::{
        AdapterStructuredTurn, AdapterTextTurn, CompletionEvent, CompletionEventStream,
        CompletionRequest, ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream,
        ErasedTextTurnEvent, ErasedTextTurnEventStream, FinishReason, LlmAdapter, StreamKind,
    },
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

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum MockError {
    #[error("no mock text scenario configured")]
    MissingTextScenario,
    #[error("no mock structured scenario configured")]
    MissingStructuredScenario,
    #[error("no mock completion scenario configured")]
    MissingCompletionScenario,
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
    recovered_usage: Arc<Mutex<BTreeMap<(StreamKind, String), Usage>>>,
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

    pub fn with_recovered_usage(
        self,
        kind: StreamKind,
        request_id: impl Into<String>,
        usage: Usage,
    ) -> Self {
        self.recovered_usage
            .lock()
            .unwrap()
            .insert((kind, request_id.into()), usage);
        self
    }
}

#[async_trait::async_trait]
impl LlmAdapter for MockLlmAdapter {
    async fn responses_text(
        &self,
        _input: ModelInput,
        _turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        let scenario = self
            .text_turns
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| AgentError::backend(MockError::MissingTextScenario))?;
        let mut tool_buffers: BTreeMap<ToolCallId, (ToolName, String)> = BTreeMap::new();

        let events = scenario
            .events
            .into_iter()
            .flat_map(move |event| match event {
                Ok(RawTextTurnEvent::Started { request_id, model }) => {
                    vec![Ok(ErasedTextTurnEvent::Started { request_id, model })]
                }
                Ok(RawTextTurnEvent::TextDelta { delta }) => {
                    vec![Ok(ErasedTextTurnEvent::TextDelta { delta })]
                }
                Ok(RawTextTurnEvent::ReasoningDelta { delta }) => {
                    vec![Ok(ErasedTextTurnEvent::ReasoningDelta { delta })]
                }
                Ok(RawTextTurnEvent::RefusalDelta { delta }) => {
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
                })],
                Err(err) => vec![Err(AgentError::backend(err))],
            })
            .collect::<Vec<_>>();

        Ok(Box::pin(stream::iter(events)) as ErasedTextTurnEventStream)
    }

    async fn responses_structured(
        &self,
        _input: ModelInput,
        _turn: AdapterStructuredTurn,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
        let scenario = self
            .structured_turns
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| AgentError::backend(MockError::MissingStructuredScenario))?;
        let mut structured_buffer = String::new();
        let mut tool_buffers: BTreeMap<ToolCallId, (ToolName, String)> = BTreeMap::new();

        let events = scenario
            .events
            .into_iter()
            .flat_map(move |event| match event {
                Ok(RawStructuredTurnEvent::Started { request_id, model }) => {
                    vec![Ok(ErasedStructuredTurnEvent::Started { request_id, model })]
                }
                Ok(RawStructuredTurnEvent::StructuredOutputChunk { json_delta }) => {
                    structured_buffer.push_str(&json_delta);
                    vec![Ok(ErasedStructuredTurnEvent::StructuredOutputChunk {
                        json_delta,
                    })]
                }
                Ok(RawStructuredTurnEvent::ReasoningDelta { delta }) => {
                    vec![Ok(ErasedStructuredTurnEvent::ReasoningDelta { delta })]
                }
                Ok(RawStructuredTurnEvent::RefusalDelta { delta }) => {
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
                    }));
                    out
                }
                Err(err) => vec![Err(AgentError::backend(err))],
            })
            .collect::<Vec<_>>();

        Ok(Box::pin(stream::iter(events)) as ErasedStructuredTurnEventStream)
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
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

    async fn recover_usage(
        &self,
        kind: StreamKind,
        request_id: &str,
    ) -> Result<Option<Usage>, AgentError> {
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
