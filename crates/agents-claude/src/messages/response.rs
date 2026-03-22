use serde::{Deserialize, Serialize};

use crate::messages::{ClaudeRole, TextBlock, ThinkingBlock, ToolUseBlock};

/// Raw Claude SSE event.
///
/// ```
/// use agents_claude::messages::response::{
///     MessageDelta, MessageDeltaEvent, MessageDeltaUsage, SseEvent,
/// };
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "type": "message_delta",
///         "delta": {"stop_reason": "end_turn", "stop_sequence": null},
///         "usage": {"output_tokens": 15}
///     }"#,
/// ).unwrap();
/// let value = SseEvent::MessageDelta(MessageDeltaEvent {
///     delta: MessageDelta {
///         stop_reason: Some("end_turn".to_string()),
///         stop_sequence: None,
///     },
///     usage: MessageDeltaUsage {
///         input_tokens: None,
///         output_tokens: Some(15),
///         cache_creation_input_tokens: None,
///         cache_read_input_tokens: None,
///     },
/// });
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<SseEvent>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SseEvent {
    MessageStart(MessageStartEvent),
    ContentBlockStart(ContentBlockStartEvent),
    ContentBlockDelta(ContentBlockDeltaEvent),
    ContentBlockStop(ContentBlockStopEvent),
    MessageDelta(MessageDeltaEvent),
    MessageStop(MessageStopEvent),
    Ping(PingEvent),
    Error(ErrorEvent),
}

/// SSE message_start payload.
///
/// ```
/// use agents_claude::messages::response::{
///     MessageDeltaUsage, MessageStartEvent, MessageType, SseMessage,
/// };
/// use agents_claude::messages::content::ClaudeRole;
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "message": {
///             "id": "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY",
///             "type": "message",
///             "role": "assistant",
///             "content": [],
///             "model": "claude-3-5-haiku-20241022",
///             "stop_reason": null,
///             "stop_sequence": null,
///             "usage": {"input_tokens": 25, "output_tokens": 1}
///         }
///     }"#,
/// ).unwrap();
/// let value = MessageStartEvent {
///     message: SseMessage {
///         id: "msg_1nZdL29xx5MUA1yADyHTEsnR8uuvGzszyY".to_string(),
///         message_type: MessageType::Message,
///         role: ClaudeRole::Assistant,
///         content: Vec::new(),
///         model: "claude-3-5-haiku-20241022".to_string(),
///         stop_reason: None,
///         stop_sequence: None,
///         usage: MessageDeltaUsage {
///             input_tokens: Some(25),
///             output_tokens: Some(1),
///             cache_creation_input_tokens: None,
///             cache_read_input_tokens: None,
///         },
///     },
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<MessageStartEvent>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MessageStartEvent {
    pub message: SseMessage,
}

/// Claude message payload embedded in streaming events.
///
/// ```
/// use agents_claude::messages::response::{
///     MessageDeltaUsage, MessageType, SseContentBlock, SseMessage,
/// };
/// use agents_claude::messages::content::{ClaudeRole, TextBlock};
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "id": "msg_123",
///         "type": "message",
///         "role": "assistant",
///         "content": [{"type": "text", "text": "Hi, I'm Claude."}],
///         "model": "claude-opus-4-6",
///         "stop_reason": "end_turn",
///         "stop_sequence": null,
///         "usage": {"input_tokens": 25, "output_tokens": 15}
///     }"#,
/// ).unwrap();
/// let value = SseMessage {
///     id: "msg_123".to_string(),
///     message_type: MessageType::Message,
///     role: ClaudeRole::Assistant,
///     content: vec![SseContentBlock::Text(TextBlock {
///         text: "Hi, I'm Claude.".to_string(),
///     })],
///     model: "claude-opus-4-6".to_string(),
///     stop_reason: Some("end_turn".to_string()),
///     stop_sequence: None,
///     usage: MessageDeltaUsage {
///         input_tokens: Some(25),
///         output_tokens: Some(15),
///         cache_creation_input_tokens: None,
///         cache_read_input_tokens: None,
///     },
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<SseMessage>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SseMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: MessageType,
    pub role: ClaudeRole,
    pub content: Vec<SseContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: MessageDeltaUsage,
}

/// SSE content_block_start payload.
///
/// ```
/// use agents_claude::messages::response::{ContentBlockStartEvent, SseContentBlock};
/// use agents_claude::messages::content::TextBlock;
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "index": 0,
///         "content_block": {"type": "text", "text": "Hi, I'm Claude."}
///     }"#,
/// ).unwrap();
/// let value = ContentBlockStartEvent {
///     index: 0,
///     content_block: SseContentBlock::Text(TextBlock {
///         text: "Hi, I'm Claude.".to_string(),
///     }),
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ContentBlockStartEvent>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContentBlockStartEvent {
    pub index: usize,
    pub content_block: SseContentBlock,
}

/// SSE content block.
///
/// ```
/// use agents_claude::messages::content::ThinkingBlock;
/// use agents_claude::messages::response::SseContentBlock;
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "type": "thinking",
///         "thinking": "Let me solve this step by step.",
///         "signature": "sig_123"
///     }"#,
/// ).unwrap();
/// let value = SseContentBlock::Thinking(ThinkingBlock {
///     thinking: "Let me solve this step by step.".to_string(),
///     signature: "sig_123".to_string(),
/// });
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<SseContentBlock>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SseContentBlock {
    Text(TextBlock),
    Thinking(ThinkingBlock),
    ToolUse(ToolUseBlock),
    RedactedThinking {
        data: String,
    },
    #[serde(other)]
    Unsupported,
}

/// SSE content_block_delta payload.
///
/// ```
/// use agents_claude::messages::response::{ContentBlockDeltaEvent, SseContentDelta};
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "index": 0,
///         "delta": {"type": "text_delta", "text": "ello frien"}
///     }"#,
/// ).unwrap();
/// let value = ContentBlockDeltaEvent {
///     index: 0,
///     delta: SseContentDelta::TextDelta {
///         text: "ello frien".to_string(),
///     },
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ContentBlockDeltaEvent>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContentBlockDeltaEvent {
    pub index: usize,
    pub delta: SseContentDelta,
}

/// SSE delta payload.
///
/// ```
/// use agents_claude::messages::response::SseContentDelta;
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{"type":"input_json_delta","partial_json":"{\"location\": \"San Fra"}"#,
/// ).unwrap();
/// let value = SseContentDelta::InputJsonDelta {
///     partial_json: "{\"location\": \"San Fra".to_string(),
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<SseContentDelta>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SseContentDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    CitationsDelta { citation: serde_json::Value },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
}

/// SSE content_block_stop payload.
///
/// ```
/// use agents_claude::messages::response::ContentBlockStopEvent;
///
/// let json = serde_json::from_str::<serde_json::Value>(r#"{"index":0}"#).unwrap();
/// let value = ContentBlockStopEvent { index: 0 };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ContentBlockStopEvent>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ContentBlockStopEvent {
    pub index: usize,
}

/// SSE message_delta payload.
///
/// ```
/// use agents_claude::messages::response::{MessageDelta, MessageDeltaEvent, MessageDeltaUsage};
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "delta": {"stop_reason": "end_turn", "stop_sequence": null},
///         "usage": {"output_tokens": 15}
///     }"#,
/// ).unwrap();
/// let value = MessageDeltaEvent {
///     delta: MessageDelta {
///         stop_reason: Some("end_turn".to_string()),
///         stop_sequence: None,
///     },
///     usage: MessageDeltaUsage {
///         input_tokens: None,
///         output_tokens: Some(15),
///         cache_creation_input_tokens: None,
///         cache_read_input_tokens: None,
///     },
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<MessageDeltaEvent>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MessageDeltaEvent {
    pub delta: MessageDelta,
    pub usage: MessageDeltaUsage,
}

/// SSE message delta body.
///
/// ```
/// use agents_claude::messages::response::MessageDelta;
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{"stop_reason":"tool_use","stop_sequence":null}"#,
/// ).unwrap();
/// let value = MessageDelta {
///     stop_reason: Some("tool_use".to_string()),
///     stop_sequence: None,
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<MessageDelta>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

/// SSE usage fragment.
///
/// ```
/// use agents_claude::messages::response::MessageDeltaUsage;
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{"input_tokens":25,"output_tokens":1}"#,
/// ).unwrap();
/// let value = MessageDeltaUsage {
///     input_tokens: Some(25),
///     output_tokens: Some(1),
///     cache_creation_input_tokens: None,
///     cache_read_input_tokens: None,
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<MessageDeltaUsage>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MessageDeltaUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
}

impl MessageDeltaUsage {
    pub(crate) fn into_protocol_usage(self) -> agents_protocol::budget::Usage {
        let input_tokens = self.input_tokens.unwrap_or_default()
            + self.cache_creation_input_tokens.unwrap_or_default()
            + self.cache_read_input_tokens.unwrap_or_default();
        let output_tokens = self.output_tokens.unwrap_or_default();

        agents_protocol::budget::Usage {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cost_micros_usd: 0,
        }
    }
}

/// SSE message_stop payload.
///
/// ```
/// use agents_claude::messages::response::MessageStopEvent;
///
/// let json = serde_json::from_str::<serde_json::Value>("{}").unwrap();
/// let value = MessageStopEvent {};
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<MessageStopEvent>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MessageStopEvent {}

/// SSE ping payload.
///
/// ```
/// use agents_claude::messages::response::PingEvent;
///
/// let json = serde_json::from_str::<serde_json::Value>("{}").unwrap();
/// let value = PingEvent {};
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<PingEvent>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PingEvent {}

/// SSE error payload.
///
/// ```
/// use agents_claude::messages::response::{ErrorEvent, StreamError};
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "error": {
///             "type": "overloaded_error",
///             "message": "Overloaded",
///             "details": null
///         }
///     }"#,
/// ).unwrap();
/// let value = ErrorEvent {
///     error: StreamError {
///         error_type: "overloaded_error".to_string(),
///         message: "Overloaded".to_string(),
///         details: None,
///     },
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ErrorEvent>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErrorEvent {
    pub error: StreamError,
}

/// Claude SSE error details.
///
/// ```
/// use agents_claude::messages::response::StreamError;
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{"type":"overloaded_error","message":"Overloaded","details":null}"#,
/// ).unwrap();
/// let value = StreamError {
///     error_type: "overloaded_error".to_string(),
///     message: "Overloaded".to_string(),
///     details: None,
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<StreamError>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
    #[serde(default)]
    pub details: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Claude streaming message type.
///
/// ```
/// use agents_claude::messages::response::MessageType;
///
/// let json = serde_json::from_str::<serde_json::Value>(r#""message""#).unwrap();
/// let value = MessageType::Message;
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<MessageType>(json).unwrap(), value);
/// ```
pub enum MessageType {
    Message,
}
