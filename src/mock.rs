use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
};

use futures::stream;

use thiserror::Error;

use crate::{
    budget::Usage,
    conversation::{ModelInput, RawJson, ToolCallId, ToolMetadata, ToolName},
    llm::{
        CompletionEvent, CompletionEventStream, CompletionRequest, FinishReason, LlmAdapter,
        StreamKind, StructuredTurnEvent, StructuredTurnEventStream, StructuredTurnRequest,
        TextTurnEvent, TextTurnEventStream, TextTurnRequest,
    },
    structured::StructuredOutput,
    toolset::Toolset,
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
    type Error = MockError;

    async fn responses_text<T>(
        &self,
        _input: ModelInput,
        _turn: TextTurnRequest<T>,
    ) -> Result<TextTurnEventStream<T, Self::Error>, Self::Error>
    where
        T: Toolset,
    {
        let scenario = self
            .text_turns
            .lock()
            .unwrap()
            .pop_front()
            .ok_or(MockError::MissingTextScenario)?;
        let mut tool_buffers: BTreeMap<ToolCallId, (ToolName, String)> = BTreeMap::new();

        let events = scenario
            .events
            .into_iter()
            .flat_map(move |event| match event {
                Ok(RawTextTurnEvent::Started { request_id, model }) => {
                    vec![Ok(TextTurnEvent::Started { request_id, model })]
                }
                Ok(RawTextTurnEvent::TextDelta { delta }) => {
                    vec![Ok(TextTurnEvent::TextDelta { delta })]
                }
                Ok(RawTextTurnEvent::ReasoningDelta { delta }) => {
                    vec![Ok(TextTurnEvent::ReasoningDelta { delta })]
                }
                Ok(RawTextTurnEvent::RefusalDelta { delta }) => {
                    vec![Ok(TextTurnEvent::RefusalDelta { delta })]
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
                    let mut out = vec![Ok(TextTurnEvent::ToolCallChunk {
                        id: id.clone(),
                        name: name.clone(),
                        arguments_json_delta,
                    })];
                    if is_complete_json(&entry.1) {
                        let arguments =
                            RawJson::parse(entry.1.clone()).expect("complete mock JSON is valid");
                        let arguments_json = arguments.get().to_string();
                        match T::parse_tool_call(
                            ToolMetadata::new(id.clone(), entry.0.clone(), arguments),
                            entry.0.as_str(),
                            &arguments_json,
                        ) {
                            Ok(tool_call) => out.push(Ok(TextTurnEvent::ToolCallReady(tool_call))),
                            Err(err) => out.push(Err(MockError::ToolCall(err.to_string()))),
                        }
                        tool_buffers.remove(&id);
                    }
                    out
                }
                Ok(RawTextTurnEvent::Completed {
                    request_id,
                    finish_reason,
                    usage,
                }) => vec![Ok(TextTurnEvent::Completed {
                    request_id,
                    finish_reason,
                    usage,
                })],
                Err(err) => vec![Err(err)],
            })
            .collect::<Vec<_>>();

        Ok(Box::pin(stream::iter(events)) as TextTurnEventStream<T, Self::Error>)
    }

    async fn responses_structured<T, O>(
        &self,
        _input: ModelInput,
        _turn: StructuredTurnRequest<T, O>,
    ) -> Result<StructuredTurnEventStream<T, O, Self::Error>, Self::Error>
    where
        T: Toolset,
        O: StructuredOutput,
    {
        let scenario = self
            .structured_turns
            .lock()
            .unwrap()
            .pop_front()
            .ok_or(MockError::MissingStructuredScenario)?;
        let mut structured_buffer = String::new();
        let mut tool_buffers: BTreeMap<ToolCallId, (ToolName, String)> = BTreeMap::new();

        let events = scenario
            .events
            .into_iter()
            .flat_map(move |event| match event {
                Ok(RawStructuredTurnEvent::Started { request_id, model }) => {
                    vec![Ok(StructuredTurnEvent::Started { request_id, model })]
                }
                Ok(RawStructuredTurnEvent::StructuredOutputChunk { json_delta }) => {
                    structured_buffer.push_str(&json_delta);
                    vec![Ok(StructuredTurnEvent::StructuredOutputChunk {
                        json_delta,
                    })]
                }
                Ok(RawStructuredTurnEvent::ReasoningDelta { delta }) => {
                    vec![Ok(StructuredTurnEvent::ReasoningDelta { delta })]
                }
                Ok(RawStructuredTurnEvent::RefusalDelta { delta }) => {
                    vec![Ok(StructuredTurnEvent::RefusalDelta { delta })]
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
                    let mut out = vec![Ok(StructuredTurnEvent::ToolCallChunk {
                        id: id.clone(),
                        name: name.clone(),
                        arguments_json_delta,
                    })];
                    if is_complete_json(&entry.1) {
                        let arguments =
                            RawJson::parse(entry.1.clone()).expect("complete mock JSON is valid");
                        let arguments_json = arguments.get().to_string();
                        match T::parse_tool_call(
                            ToolMetadata::new(id.clone(), entry.0.clone(), arguments),
                            entry.0.as_str(),
                            &arguments_json,
                        ) {
                            Ok(tool_call) => {
                                out.push(Ok(StructuredTurnEvent::ToolCallReady(tool_call)))
                            }
                            Err(err) => out.push(Err(MockError::ToolCall(err.to_string()))),
                        }
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
                        match serde_json::from_str::<O>(&structured_buffer) {
                            Ok(value) => {
                                out.push(Ok(StructuredTurnEvent::StructuredOutputReady(value)))
                            }
                            Err(err) => out.push(Err(MockError::StructuredOutput(err.to_string()))),
                        }
                    }
                    out.push(Ok(StructuredTurnEvent::Completed {
                        request_id,
                        finish_reason,
                        usage,
                    }));
                    out
                }
                Err(err) => vec![Err(err)],
            })
            .collect::<Vec<_>>();

        Ok(Box::pin(stream::iter(events)) as StructuredTurnEventStream<T, O, Self::Error>)
    }

    async fn completion(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionEventStream<Self::Error>, Self::Error> {
        let scenario = self
            .completions
            .lock()
            .unwrap()
            .pop_front()
            .ok_or(MockError::MissingCompletionScenario)?;
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
            Err(err) => Err(err),
        });

        Ok(Box::pin(stream::iter(events.collect::<Vec<_>>()))
            as CompletionEventStream<Self::Error>)
    }

    async fn recover_usage(
        &self,
        kind: StreamKind,
        request_id: &str,
    ) -> Result<Option<Usage>, Self::Error> {
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
