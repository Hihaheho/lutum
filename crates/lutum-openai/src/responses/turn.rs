use lutum_protocol::{
    FinishReason,
    budget::Usage,
    conversation::{RawJson, ToolCallId, ToolName},
    transcript::{ItemView, ToolCallItemView, ToolResultItemView, TurnRole, TurnView},
};
use serde::{Deserialize, Serialize};

/// ```
/// use lutum_openai::OpenAiCommittedTurn;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "request_id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
///       "model": "gpt-5.4",
///       "items": [
///         {
///           "type": "text",
///           "content": "In a peaceful grove beneath a silver moon..."
///         }
///       ],
///       "finish_reason": "Stop",
///       "usage": {
///         "input_tokens": 36,
///         "output_tokens": 87,
///         "total_tokens": 123,
///         "cost_micros_usd": 0,
///         "cache_creation_tokens": 0,
///         "cache_read_tokens": 0
///       }
///     }"#,
/// )
/// .unwrap();
/// let turn = serde_json::from_value::<OpenAiCommittedTurn>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&turn).unwrap(), json);
/// assert_eq!(serde_json::from_value::<OpenAiCommittedTurn>(json).unwrap(), turn);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpenAiCommittedTurn {
    pub request_id: Option<String>,
    pub model: String,
    pub items: Vec<OpenAiTurnItem>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

/// ```
/// use lutum_openai::OpenAiTurnItem;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type": "text",
///       "content": "The classic tongue twister..."
///     }"#,
/// )
/// .unwrap();
/// let item = serde_json::from_value::<OpenAiTurnItem>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&item).unwrap(), json);
/// assert_eq!(serde_json::from_value::<OpenAiTurnItem>(json).unwrap(), item);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAiTurnItem {
    Text {
        content: String,
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

impl TurnView for OpenAiCommittedTurn {
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

impl ItemView for OpenAiTurnItem {
    fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { content } => Some(content),
            _ => None,
        }
    }

    fn as_reasoning(&self) -> Option<&str> {
        match self {
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
