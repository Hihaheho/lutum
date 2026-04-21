use std::collections::BTreeMap;

use monostate::MustBe;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::responses::{MessageRole, OutputTextContent, SummaryText};

/// ```
/// use lutum_openai::responses::SseEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type": "response.output_text.delta",
///       "item_id": "msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654",
///       "output_index": 0,
///       "content_index": 0,
///       "delta": "Hi"
///     }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<SseEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<SseEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug)]
pub enum SseEvent {
    ResponseCreated(ResponseCreatedEvent),
    ResponseInProgress(ResponseInProgressEvent),
    ResponseOutputItemAdded(ResponseOutputItemAddedEvent),
    ResponseContentPartAdded(ResponseContentPartAddedEvent),
    ResponseOutputTextDelta(ResponseOutputTextDeltaEvent),
    ResponseOutputTextDone(ResponseOutputTextDoneEvent),
    ResponseContentPartDone(ResponseContentPartDoneEvent),
    ResponseOutputItemDone(ResponseOutputItemDoneEvent),
    ResponseReasoningSummaryTextDelta(ResponseReasoningSummaryTextDeltaEvent),
    ResponseReasoningSummaryTextDone(ResponseReasoningSummaryTextDoneEvent),
    ResponseReasoningDelta(ResponseReasoningDeltaEvent),
    ResponseRefusalDelta(ResponseRefusalDeltaEvent),
    ResponseFunctionCallArgumentsDelta(ResponseFunctionCallArgumentsDeltaEvent),
    ResponseFunctionCallArgumentsDone(ResponseFunctionCallArgumentsDoneEvent),
    ResponseCompleted(ResponseCompletedEvent),
    Unknown(serde_json::Value),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseOutputItem {
    #[serde(rename = "message")]
    Message(ResponseOutputMessage),
    #[serde(rename = "function_call")]
    FunctionCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        name: String,
        arguments: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        call_id: String,
        output: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        summary: Vec<SummaryText>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<Vec<Value>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseOutputMessage {
    pub content: Vec<ResponseOutputContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub role: MessageRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseOutputContent {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(default)]
        annotations: Vec<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        logprobs: Option<Vec<Value>>,
    },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
enum SseEventWire {
    #[serde(rename = "response.created")]
    Created { response: ResponseObject },
    #[serde(rename = "response.in_progress")]
    InProgress { response: ResponseObject },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: usize,
        #[serde(default)]
        sequence_number: u64,
        item: ResponseOutputItem,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: OutputTextContent,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: OutputTextContent,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        output_index: usize,
        #[serde(default)]
        sequence_number: u64,
        item: ResponseOutputItem,
    },
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        item_id: Option<String>,
    },
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone {
        item_id: String,
        output_index: usize,
        summary_index: usize,
        text: String,
        sequence_number: u64,
    },
    #[serde(rename = "response.reasoning.delta")]
    ReasoningDelta {
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        item_id: Option<String>,
    },
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta {
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        item_id: Option<String>,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        item_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        sequence_number: u64,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        #[serde(skip_serializing_if = "Option::is_none")]
        item_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        #[serde(default)]
        output_index: usize,
        #[serde(default)]
        sequence_number: u64,
        #[serde(default)]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        arguments: Option<String>,
    },
    #[serde(rename = "response.completed")]
    Completed { response: ResponseObject },
    /// Catch-all for any event types not yet handled by this crate.
    #[serde(other)]
    Unknown,
}

impl From<SseEventWire> for SseEvent {
    fn from(value: SseEventWire) -> Self {
        match value {
            SseEventWire::Created { response } => Self::ResponseCreated(ResponseCreatedEvent {
                response,
                event_type: Default::default(),
            }),
            SseEventWire::InProgress { response } => {
                Self::ResponseInProgress(ResponseInProgressEvent {
                    response,
                    event_type: Default::default(),
                })
            }
            SseEventWire::OutputItemAdded {
                output_index,
                sequence_number,
                item,
            } => Self::ResponseOutputItemAdded(ResponseOutputItemAddedEvent {
                output_index,
                sequence_number,
                item,
                event_type: Default::default(),
            }),
            SseEventWire::ContentPartAdded {
                item_id,
                output_index,
                content_index,
                part,
            } => Self::ResponseContentPartAdded(ResponseContentPartAddedEvent {
                item_id,
                output_index,
                content_index,
                part,
                event_type: Default::default(),
            }),
            SseEventWire::OutputTextDelta {
                item_id,
                output_index,
                content_index,
                delta,
            } => Self::ResponseOutputTextDelta(ResponseOutputTextDeltaEvent {
                item_id,
                output_index,
                content_index,
                delta,
                event_type: Default::default(),
            }),
            SseEventWire::OutputTextDone {
                item_id,
                output_index,
                content_index,
                text,
            } => Self::ResponseOutputTextDone(ResponseOutputTextDoneEvent {
                item_id,
                output_index,
                content_index,
                text,
                event_type: Default::default(),
            }),
            SseEventWire::ContentPartDone {
                item_id,
                output_index,
                content_index,
                part,
            } => Self::ResponseContentPartDone(ResponseContentPartDoneEvent {
                item_id,
                output_index,
                content_index,
                part,
                event_type: Default::default(),
            }),
            SseEventWire::OutputItemDone {
                output_index,
                sequence_number,
                item,
            } => Self::ResponseOutputItemDone(ResponseOutputItemDoneEvent {
                output_index,
                sequence_number,
                item,
                event_type: Default::default(),
            }),
            SseEventWire::ReasoningSummaryTextDelta { delta, item_id } => {
                Self::ResponseReasoningSummaryTextDelta(ResponseReasoningSummaryTextDeltaEvent {
                    delta,
                    item_id,
                    event_type: Default::default(),
                })
            }
            SseEventWire::ReasoningSummaryTextDone {
                item_id,
                output_index,
                summary_index,
                text,
                sequence_number,
            } => Self::ResponseReasoningSummaryTextDone(ResponseReasoningSummaryTextDoneEvent {
                item_id,
                output_index,
                summary_index,
                text,
                sequence_number,
                event_type: Default::default(),
            }),
            // Unreachable via the custom Deserialize impl, which handles Unknown
            // before calling From. Only reachable if someone calls From directly.
            SseEventWire::Unknown => Self::Unknown(serde_json::Value::Null),
            SseEventWire::ReasoningDelta { delta, item_id } => {
                Self::ResponseReasoningDelta(ResponseReasoningDeltaEvent {
                    delta,
                    item_id,
                    event_type: Default::default(),
                })
            }
            SseEventWire::RefusalDelta { delta, item_id } => {
                Self::ResponseRefusalDelta(ResponseRefusalDeltaEvent {
                    delta,
                    item_id,
                    event_type: Default::default(),
                })
            }
            SseEventWire::FunctionCallArgumentsDelta {
                item_id,
                call_id,
                output_index,
                sequence_number,
                delta,
            } => {
                Self::ResponseFunctionCallArgumentsDelta(ResponseFunctionCallArgumentsDeltaEvent {
                    item_id,
                    call_id,
                    output_index,
                    sequence_number,
                    delta,
                    event_type: Default::default(),
                })
            }
            SseEventWire::FunctionCallArgumentsDone {
                item_id,
                call_id,
                output_index,
                sequence_number,
                name,
                arguments,
            } => Self::ResponseFunctionCallArgumentsDone(ResponseFunctionCallArgumentsDoneEvent {
                item_id,
                call_id,
                output_index,
                sequence_number,
                name,
                arguments,
                event_type: Default::default(),
            }),
            SseEventWire::Completed { response } => {
                Self::ResponseCompleted(ResponseCompletedEvent {
                    response,
                    event_type: Default::default(),
                })
            }
        }
    }
}

impl From<SseEvent> for SseEventWire {
    fn from(value: SseEvent) -> Self {
        match value {
            SseEvent::ResponseCreated(ResponseCreatedEvent { response, .. }) => {
                Self::Created { response }
            }
            SseEvent::ResponseInProgress(ResponseInProgressEvent { response, .. }) => {
                Self::InProgress { response }
            }
            SseEvent::ResponseOutputItemAdded(ResponseOutputItemAddedEvent {
                output_index,
                sequence_number,
                item,
                ..
            }) => Self::OutputItemAdded {
                output_index,
                sequence_number,
                item,
            },
            SseEvent::ResponseContentPartAdded(ResponseContentPartAddedEvent {
                item_id,
                output_index,
                content_index,
                part,
                ..
            }) => Self::ContentPartAdded {
                item_id,
                output_index,
                content_index,
                part,
            },
            SseEvent::ResponseOutputTextDelta(ResponseOutputTextDeltaEvent {
                item_id,
                output_index,
                content_index,
                delta,
                ..
            }) => Self::OutputTextDelta {
                item_id,
                output_index,
                content_index,
                delta,
            },
            SseEvent::ResponseOutputTextDone(ResponseOutputTextDoneEvent {
                item_id,
                output_index,
                content_index,
                text,
                ..
            }) => Self::OutputTextDone {
                item_id,
                output_index,
                content_index,
                text,
            },
            SseEvent::ResponseContentPartDone(ResponseContentPartDoneEvent {
                item_id,
                output_index,
                content_index,
                part,
                ..
            }) => Self::ContentPartDone {
                item_id,
                output_index,
                content_index,
                part,
            },
            SseEvent::ResponseOutputItemDone(ResponseOutputItemDoneEvent {
                output_index,
                sequence_number,
                item,
                ..
            }) => Self::OutputItemDone {
                output_index,
                sequence_number,
                item,
            },
            SseEvent::ResponseReasoningSummaryTextDelta(
                ResponseReasoningSummaryTextDeltaEvent { delta, item_id, .. },
            ) => Self::ReasoningSummaryTextDelta { delta, item_id },
            SseEvent::ResponseReasoningSummaryTextDone(ResponseReasoningSummaryTextDoneEvent {
                item_id,
                output_index,
                summary_index,
                text,
                sequence_number,
                ..
            }) => Self::ReasoningSummaryTextDone {
                item_id,
                output_index,
                summary_index,
                text,
                sequence_number,
            },
            SseEvent::Unknown(_) => Self::Unknown, // Value is lost on Into; Unknown round-trips as unit
            SseEvent::ResponseReasoningDelta(ResponseReasoningDeltaEvent {
                delta,
                item_id,
                ..
            }) => Self::ReasoningDelta { delta, item_id },
            SseEvent::ResponseRefusalDelta(ResponseRefusalDeltaEvent {
                delta, item_id, ..
            }) => Self::RefusalDelta { delta, item_id },
            SseEvent::ResponseFunctionCallArgumentsDelta(
                ResponseFunctionCallArgumentsDeltaEvent {
                    item_id,
                    call_id,
                    output_index,
                    sequence_number,
                    delta,
                    ..
                },
            ) => Self::FunctionCallArgumentsDelta {
                item_id,
                call_id,
                output_index,
                sequence_number,
                delta,
            },
            SseEvent::ResponseFunctionCallArgumentsDone(
                ResponseFunctionCallArgumentsDoneEvent {
                    item_id,
                    call_id,
                    output_index,
                    sequence_number,
                    name,
                    arguments,
                    ..
                },
            ) => Self::FunctionCallArgumentsDone {
                item_id,
                call_id,
                output_index,
                sequence_number,
                name,
                arguments,
            },
            SseEvent::ResponseCompleted(ResponseCompletedEvent { response, .. }) => {
                Self::Completed { response }
            }
        }
    }
}

impl PartialEq for SseEvent {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Unknown(a), Self::Unknown(b)) => a == b,
            _ => serde_json::to_string(self).ok() == serde_json::to_string(other).ok(),
        }
    }
}

impl serde::Serialize for SseEvent {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        SseEventWire::from(self.clone()).serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for SseEvent {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        match serde_json::from_value::<SseEventWire>(value.clone()) {
            Ok(SseEventWire::Unknown) => Ok(Self::Unknown(value)),
            Ok(wire) => Ok(Self::from(wire)),
            Err(e) => Err(serde::de::Error::custom(e)),
        }
    }
}

/// ```
/// use lutum_openai::responses::ResponseObject;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "id":"resp_67c9fdcecf488190bdd9a0409de3a1ec07b8b0ad4e5eb654",
///       "object":"response",
///       "created_at":1741290958,
///       "status":"in_progress",
///       "error":null,
///       "incomplete_details":null,
///       "instructions":"You are a helpful assistant.",
///       "max_output_tokens":null,
///       "model":"gpt-5.4",
///       "output":[],
///       "parallel_tool_calls":true,
///       "previous_response_id":null,
///       "reasoning":{"effort":null,"summary":null},
///       "store":true,
///       "temperature":1.0,
///       "text":{"format":{"type":"text"}},
///       "tool_choice":"auto",
///       "tools":[],
///       "top_p":1.0,
///       "truncation":"disabled",
///       "usage":null,
///       "user":null,
///       "metadata":{}
///     }"#,
/// )
/// .unwrap();
/// let response = serde_json::from_value::<ResponseObject>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&response).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseObject>(json).unwrap(), response);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseObject {
    pub id: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub output: Vec<ResponseOutputItem>,
    #[serde(default)]
    pub usage: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// ```
/// use lutum_openai::responses::ResponseCreatedEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type":"response.created",
///       "response":{
///         "id":"resp_67c9fdcecf488190bdd9a0409de3a1ec07b8b0ad4e5eb654",
///         "object":"response",
///         "created_at":1741290958,
///         "status":"in_progress",
///         "error":null,
///         "incomplete_details":null,
///         "instructions":"You are a helpful assistant.",
///         "max_output_tokens":null,
///         "model":"gpt-5.4",
///         "output":[],
///         "parallel_tool_calls":true,
///         "previous_response_id":null,
///         "reasoning":{"effort":null,"summary":null},
///         "store":true,
///         "temperature":1.0,
///         "text":{"format":{"type":"text"}},
///         "tool_choice":"auto",
///         "tools":[],
///         "top_p":1.0,
///         "truncation":"disabled",
///         "usage":null,
///         "user":null,
///         "metadata":{}
///       }
///     }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseCreatedEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseCreatedEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseCreatedEvent {
    pub response: ResponseObject,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.created"),
}

/// ```
/// use lutum_openai::responses::ResponseInProgressEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type":"response.in_progress",
///       "response":{
///         "id":"resp_67c9fdcecf488190bdd9a0409de3a1ec07b8b0ad4e5eb654",
///         "object":"response",
///         "created_at":1741290958,
///         "status":"in_progress",
///         "error":null,
///         "incomplete_details":null,
///         "instructions":"You are a helpful assistant.",
///         "max_output_tokens":null,
///         "model":"gpt-5.4",
///         "output":[],
///         "parallel_tool_calls":true,
///         "previous_response_id":null,
///         "reasoning":{"effort":null,"summary":null},
///         "store":true,
///         "temperature":1.0,
///         "text":{"format":{"type":"text"}},
///         "tool_choice":"auto",
///         "tools":[],
///         "top_p":1.0,
///         "truncation":"disabled",
///         "usage":null,
///         "user":null,
///         "metadata":{}
///       }
///     }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseInProgressEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseInProgressEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseInProgressEvent {
    pub response: ResponseObject,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.in_progress"),
}

/// ```
/// use lutum_openai::responses::ResponseOutputItemAddedEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type":"response.output_item.added",
///       "output_index":0,
///       "sequence_number":0,
///       "item":{
///         "id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654",
///         "type":"message",
///         "status":"in_progress",
///         "role":"assistant",
///         "content":[]
///       }
///     }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseOutputItemAddedEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseOutputItemAddedEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseOutputItemAddedEvent {
    pub output_index: usize,
    #[serde(default)]
    pub sequence_number: u64,
    pub item: ResponseOutputItem,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.output_item.added"),
}

/// ```
/// use lutum_openai::responses::ResponseContentPartAddedEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type":"response.content_part.added",
///       "item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654",
///       "output_index":0,
///       "content_index":0,
///       "part":{"type":"output_text","text":"","annotations":[]}
///     }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseContentPartAddedEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseContentPartAddedEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseContentPartAddedEvent {
    pub item_id: String,
    pub output_index: usize,
    pub content_index: usize,
    pub part: OutputTextContent,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.content_part.added"),
}

/// ```
/// use lutum_openai::responses::ResponseOutputTextDeltaEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type":"response.output_text.delta",
///       "item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654",
///       "output_index":0,
///       "content_index":0,
///       "delta":"Hi"
///     }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseOutputTextDeltaEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseOutputTextDeltaEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseOutputTextDeltaEvent {
    pub item_id: String,
    pub output_index: usize,
    pub content_index: usize,
    pub delta: String,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.output_text.delta"),
}

/// ```
/// use lutum_openai::responses::ResponseOutputTextDoneEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type":"response.output_text.done",
///       "item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654",
///       "output_index":0,
///       "content_index":0,
///       "text":"Hi there! How can I assist you today?"
///     }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseOutputTextDoneEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseOutputTextDoneEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseOutputTextDoneEvent {
    pub item_id: String,
    pub output_index: usize,
    pub content_index: usize,
    pub text: String,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.output_text.done"),
}

/// ```
/// use lutum_openai::responses::ResponseContentPartDoneEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type":"response.content_part.done",
///       "item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654",
///       "output_index":0,
///       "content_index":0,
///       "part":{
///         "type":"output_text",
///         "text":"Hi there! How can I assist you today?",
///         "annotations":[]
///       }
///     }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseContentPartDoneEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseContentPartDoneEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseContentPartDoneEvent {
    pub item_id: String,
    pub output_index: usize,
    pub content_index: usize,
    pub part: OutputTextContent,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.content_part.done"),
}

/// ```
/// use lutum_openai::responses::ResponseOutputItemDoneEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type":"response.output_item.done",
///       "output_index":0,
///       "sequence_number":0,
///       "item":{
///         "id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654",
///         "type":"message",
///         "status":"completed",
///         "role":"assistant",
///         "content":[
///           {
///             "type":"output_text",
///             "text":"Hi there! How can I assist you today?",
///             "annotations":[]
///           }
///         ]
///       }
///     }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseOutputItemDoneEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseOutputItemDoneEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseOutputItemDoneEvent {
    pub output_index: usize,
    #[serde(default)]
    pub sequence_number: u64,
    pub item: ResponseOutputItem,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.output_item.done"),
}

/// ```
/// use lutum_openai::responses::ResponseReasoningSummaryTextDeltaEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{ "type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": "The classic tongue twister..." }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseReasoningSummaryTextDeltaEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseReasoningSummaryTextDeltaEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseReasoningSummaryTextDeltaEvent {
    pub delta: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.reasoning_summary_text.delta"),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseReasoningSummaryTextDoneEvent {
    pub item_id: String,
    pub output_index: usize,
    pub summary_index: usize,
    pub text: String,
    pub sequence_number: u64,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.reasoning_summary_text.done"),
}

/// ```
/// use lutum_openai::responses::ResponseReasoningDeltaEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{ "type": "response.reasoning.delta", "item_id": "rs_1", "delta": "The classic tongue twister..." }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseReasoningDeltaEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseReasoningDeltaEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseReasoningDeltaEvent {
    pub delta: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.reasoning.delta"),
}

/// ```
/// use lutum_openai::responses::ResponseRefusalDeltaEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{ "type": "response.refusal.delta", "item_id": "msg_1", "delta": "I can't help with that request." }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseRefusalDeltaEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseRefusalDeltaEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseRefusalDeltaEvent {
    pub delta: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.refusal.delta"),
}

/// ```
/// use lutum_openai::responses::ResponseFunctionCallArgumentsDeltaEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type": "response.function_call_arguments.delta",
///       "item_id": "fc_67ca09c6bedc8190a7abfec07b1a1332096610f474011cc0",
///       "call_id": "call_unLAR8MvFNptuiZK6K6HCy5k",
///       "output_index": 0,
///       "sequence_number": 0,
///       "delta": "{\"location\":\"Boston, MA\""
///     }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseFunctionCallArgumentsDeltaEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseFunctionCallArgumentsDeltaEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseFunctionCallArgumentsDeltaEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(default)]
    pub output_index: usize,
    #[serde(default)]
    pub sequence_number: u64,
    pub delta: String,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.function_call_arguments.delta"),
}

/// ```
/// use lutum_openai::responses::ResponseFunctionCallArgumentsDoneEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type": "response.function_call_arguments.done",
///       "item_id": "fc_67ca09c6bedc8190a7abfec07b1a1332096610f474011cc0",
///       "call_id": "call_unLAR8MvFNptuiZK6K6HCy5k",
///       "output_index": 0,
///       "sequence_number": 0,
///       "name": "get_current_weather",
///       "arguments": "{\"location\":\"Boston, MA\",\"unit\":\"celsius\"}"
///     }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseFunctionCallArgumentsDoneEvent>(json.clone()).unwrap();
/// assert_eq!(event.name, Some("get_current_weather".into()));
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseFunctionCallArgumentsDoneEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseFunctionCallArgumentsDoneEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(default)]
    pub output_index: usize,
    #[serde(default)]
    pub sequence_number: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.function_call_arguments.done"),
}

/// ```
/// use lutum_openai::responses::ResponseCompletedEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type":"response.completed",
///       "response":{
///         "id":"resp_67c9fdcecf488190bdd9a0409de3a1ec07b8b0ad4e5eb654",
///         "object":"response",
///         "created_at":1741290958,
///         "status":"completed",
///         "error":null,
///         "incomplete_details":null,
///         "instructions":"You are a helpful assistant.",
///         "max_output_tokens":null,
///         "model":"gpt-5.4",
///         "output":[
///           {
///             "id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654",
///             "type":"message",
///             "status":"completed",
///             "role":"assistant",
///             "content":[
///               {
///                 "type":"output_text",
///                 "text":"Hi there! How can I assist you today?",
///                 "annotations":[]
///               }
///             ]
///           }
///         ],
///         "parallel_tool_calls":true,
///         "previous_response_id":null,
///         "reasoning":{"effort":null,"summary":null},
///         "store":true,
///         "temperature":1.0,
///         "text":{"format":{"type":"text"}},
///         "tool_choice":"auto",
///         "tools":[],
///         "top_p":1.0,
///         "truncation":"disabled",
///         "usage":{"input_tokens":37,"output_tokens":11,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":48},
///         "user":null,
///         "metadata":{}
///       }
///     }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseCompletedEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseCompletedEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseCompletedEvent {
    pub response: ResponseObject,
    #[serde(rename = "type")]
    pub event_type: MustBe!("response.completed"),
}

#[cfg(test)]
mod tests {
    use super::ResponseContentPartAddedEvent;

    #[test]
    fn response_content_part_added_allows_missing_annotations() {
        let json = serde_json::json!({
            "type": "response.content_part.added",
            "item_id": "rs_tmp_vzapvvz1nn",
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "reasoning_text",
                "text": ""
            }
        });

        let event = serde_json::from_value::<ResponseContentPartAddedEvent>(json).unwrap();

        assert!(event.part.annotations.is_empty());
        assert_eq!(event.part.item_type, "reasoning_text");
        assert_eq!(event.part.text, "");
    }
}
