use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::responses::{InputItem, OutputTextContent};

/// ```
/// use agents_openai::responses::SseEvent;
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "SseEventWire", into = "SseEventWire")]
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
    Unknown,
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
        item: InputItem,
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
        item: InputItem,
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
        name: String,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        #[serde(skip_serializing_if = "Option::is_none")]
        item_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        call_id: Option<String>,
        name: String,
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
                event_type: "response.created".to_string(),
            }),
            SseEventWire::InProgress { response } => {
                Self::ResponseInProgress(ResponseInProgressEvent {
                    response,
                    event_type: "response.in_progress".to_string(),
                })
            }
            SseEventWire::OutputItemAdded { output_index, item } => {
                Self::ResponseOutputItemAdded(ResponseOutputItemAddedEvent {
                    output_index,
                    item,
                    event_type: "response.output_item.added".to_string(),
                })
            }
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
                event_type: "response.content_part.added".to_string(),
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
                event_type: "response.output_text.delta".to_string(),
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
                event_type: "response.output_text.done".to_string(),
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
                event_type: "response.content_part.done".to_string(),
            }),
            SseEventWire::OutputItemDone { output_index, item } => {
                Self::ResponseOutputItemDone(ResponseOutputItemDoneEvent {
                    output_index,
                    item,
                    event_type: "response.output_item.done".to_string(),
                })
            }
            SseEventWire::ReasoningSummaryTextDelta { delta, item_id } => {
                Self::ResponseReasoningSummaryTextDelta(ResponseReasoningSummaryTextDeltaEvent {
                    delta,
                    item_id,
                    event_type: "response.reasoning_summary_text.delta".to_string(),
                })
            }
            SseEventWire::ReasoningSummaryTextDone {
                item_id,
                output_index,
                summary_index,
                text,
                sequence_number,
            } => {
                Self::ResponseReasoningSummaryTextDone(ResponseReasoningSummaryTextDoneEvent {
                    item_id,
                    output_index,
                    summary_index,
                    text,
                    sequence_number,
                    event_type: "response.reasoning_summary_text.done".to_string(),
                })
            }
            SseEventWire::Unknown => Self::Unknown,
            SseEventWire::ReasoningDelta { delta, item_id } => {
                Self::ResponseReasoningDelta(ResponseReasoningDeltaEvent {
                    delta,
                    item_id,
                    event_type: "response.reasoning.delta".to_string(),
                })
            }
            SseEventWire::RefusalDelta { delta, item_id } => {
                Self::ResponseRefusalDelta(ResponseRefusalDeltaEvent {
                    delta,
                    item_id,
                    event_type: "response.refusal.delta".to_string(),
                })
            }
            SseEventWire::FunctionCallArgumentsDelta {
                item_id,
                call_id,
                name,
                delta,
            } => {
                Self::ResponseFunctionCallArgumentsDelta(ResponseFunctionCallArgumentsDeltaEvent {
                    item_id,
                    call_id,
                    name,
                    delta,
                    event_type: "response.function_call_arguments.delta".to_string(),
                })
            }
            SseEventWire::FunctionCallArgumentsDone {
                item_id,
                call_id,
                name,
                arguments,
            } => Self::ResponseFunctionCallArgumentsDone(ResponseFunctionCallArgumentsDoneEvent {
                item_id,
                call_id,
                name,
                arguments,
                event_type: "response.function_call_arguments.done".to_string(),
            }),
            SseEventWire::Completed { response } => {
                Self::ResponseCompleted(ResponseCompletedEvent {
                    response,
                    event_type: "response.completed".to_string(),
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
                item,
                ..
            }) => Self::OutputItemAdded { output_index, item },
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
                item,
                ..
            }) => Self::OutputItemDone { output_index, item },
            SseEvent::ResponseReasoningSummaryTextDelta(
                ResponseReasoningSummaryTextDeltaEvent { delta, item_id, .. },
            ) => Self::ReasoningSummaryTextDelta { delta, item_id },
            SseEvent::ResponseReasoningSummaryTextDone(
                ResponseReasoningSummaryTextDoneEvent {
                    item_id,
                    output_index,
                    summary_index,
                    text,
                    sequence_number,
                    ..
                },
            ) => Self::ReasoningSummaryTextDone {
                item_id,
                output_index,
                summary_index,
                text,
                sequence_number,
            },
            SseEvent::Unknown => Self::Unknown,
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
                    name,
                    delta,
                    ..
                },
            ) => Self::FunctionCallArgumentsDelta {
                item_id,
                call_id,
                name,
                delta,
            },
            SseEvent::ResponseFunctionCallArgumentsDone(
                ResponseFunctionCallArgumentsDoneEvent {
                    item_id,
                    call_id,
                    name,
                    arguments,
                    ..
                },
            ) => Self::FunctionCallArgumentsDone {
                item_id,
                call_id,
                name,
                arguments,
            },
            SseEvent::ResponseCompleted(ResponseCompletedEvent { response, .. }) => {
                Self::Completed { response }
            }
        }
    }
}

/// ```
/// use agents_openai::responses::ResponseObject;
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
    pub output: Vec<InputItem>,
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
/// use agents_openai::responses::ResponseCreatedEvent;
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
    pub event_type: String,
}

/// ```
/// use agents_openai::responses::ResponseInProgressEvent;
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
    pub event_type: String,
}

/// ```
/// use agents_openai::responses::ResponseOutputItemAddedEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type":"response.output_item.added",
///       "output_index":0,
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
    pub item: InputItem,
    #[serde(rename = "type")]
    pub event_type: String,
}

/// ```
/// use agents_openai::responses::ResponseContentPartAddedEvent;
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
    pub event_type: String,
}

/// ```
/// use agents_openai::responses::ResponseOutputTextDeltaEvent;
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
    pub event_type: String,
}

/// ```
/// use agents_openai::responses::ResponseOutputTextDoneEvent;
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
    pub event_type: String,
}

/// ```
/// use agents_openai::responses::ResponseContentPartDoneEvent;
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
    pub event_type: String,
}

/// ```
/// use agents_openai::responses::ResponseOutputItemDoneEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type":"response.output_item.done",
///       "output_index":0,
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
    pub item: InputItem,
    #[serde(rename = "type")]
    pub event_type: String,
}

/// ```
/// use agents_openai::responses::ResponseReasoningSummaryTextDeltaEvent;
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
    pub event_type: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseReasoningSummaryTextDoneEvent {
    pub item_id: String,
    pub output_index: usize,
    pub summary_index: usize,
    pub text: String,
    pub sequence_number: u64,
    #[serde(rename = "type")]
    pub event_type: String,
}

/// ```
/// use agents_openai::responses::ResponseReasoningDeltaEvent;
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
    pub event_type: String,
}

/// ```
/// use agents_openai::responses::ResponseRefusalDeltaEvent;
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
    pub event_type: String,
}

/// ```
/// use agents_openai::responses::ResponseFunctionCallArgumentsDeltaEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type": "response.function_call_arguments.delta",
///       "item_id": "fc_67ca09c6bedc8190a7abfec07b1a1332096610f474011cc0",
///       "call_id": "call_unLAR8MvFNptuiZK6K6HCy5k",
///       "name": "get_current_weather",
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
    pub name: String,
    pub delta: String,
    #[serde(rename = "type")]
    pub event_type: String,
}

/// ```
/// use agents_openai::responses::ResponseFunctionCallArgumentsDoneEvent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type": "response.function_call_arguments.done",
///       "item_id": "fc_67ca09c6bedc8190a7abfec07b1a1332096610f474011cc0",
///       "call_id": "call_unLAR8MvFNptuiZK6K6HCy5k",
///       "name": "get_current_weather",
///       "arguments": "{\"location\":\"Boston, MA\",\"unit\":\"celsius\"}"
///     }"#,
/// )
/// .unwrap();
/// let event = serde_json::from_value::<ResponseFunctionCallArgumentsDoneEvent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&event).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseFunctionCallArgumentsDoneEvent>(json).unwrap(), event);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponseFunctionCallArgumentsDoneEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    #[serde(rename = "type")]
    pub event_type: String,
}

/// ```
/// use agents_openai::responses::ResponseCompletedEvent;
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
    pub event_type: String,
}
