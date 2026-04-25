use lutum_protocol::{
    budget::Usage,
    conversation::{RawJson, ToolCallId, ToolName},
    llm::FinishReason,
    transcript::{ItemView, ToolCallItemView, ToolResultItemView, TurnRole, TurnView},
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Claude committed turn.
///
/// ```
/// use lutum_claude::messages::turn::{ClaudeCommittedTurn, ClaudeTurnItem};
/// use lutum_protocol::{
///     budget::Usage,
///     conversation::{RawJson, ToolCallId, ToolName},
///     llm::FinishReason,
/// };
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "request_id": "msg_123",
///         "model": "claude-opus-4-6",
///         "items": [
///             {"type": "text", "content": "B)"},
///             {
///                 "type": "tool_call",
///                 "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
///                 "name": "get_stock_price",
///                 "arguments": {"ticker": "^GSPC"}
///             }
///         ],
///         "finish_reason": "ToolCall",
///         "usage": {
///             "input_tokens": 25,
///             "output_tokens": 15,
///             "total_tokens": 40,
///             "cost_micros_usd": 0,
///             "cache_creation_tokens": 0,
///             "cache_read_tokens": 0
///         },
///         "cache_creation_input_tokens": 0,
///         "cache_read_input_tokens": 0
///     }"#,
/// ).unwrap();
/// let value = ClaudeCommittedTurn {
///     request_id: Some("msg_123".to_string()),
///     model: "claude-opus-4-6".to_string(),
///     items: vec![
///         ClaudeTurnItem::Text {
///             content: "B)".to_string(),
///         },
///         ClaudeTurnItem::ToolCall {
///             id: ToolCallId::from("toolu_01D7FLrfh4GYq7yT1ULFeyMV"),
///             name: ToolName::from("get_stock_price"),
///             arguments: RawJson::parse(r#"{"ticker":"^GSPC"}"#).unwrap(),
///         },
///     ],
///     finish_reason: FinishReason::ToolCall,
///     usage: Usage {
///         input_tokens: 25,
///         output_tokens: 15,
///         total_tokens: 40,
///         cost_micros_usd: 0,
///         cache_creation_tokens: 0,
///         cache_read_tokens: 0,
///     },
///     cache_creation_input_tokens: 0,
///     cache_read_input_tokens: 0,
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ClaudeCommittedTurn>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClaudeCommittedTurn {
    pub request_id: Option<String>,
    pub model: String,
    pub items: Vec<ClaudeTurnItem>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    /// Tokens written to the prompt cache in this turn.
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    /// Tokens read from the prompt cache in this turn.
    #[serde(default)]
    pub cache_read_input_tokens: u64,
}

impl PartialEq for ClaudeCommittedTurn {
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
            && self.model == other.model
            && self.items == other.items
            && self.finish_reason == other.finish_reason
            && self.usage == other.usage
    }
}

/// Claude committed turn item.
///
/// ```
/// use lutum_claude::messages::turn::ClaudeTurnItem;
/// use lutum_protocol::conversation::{RawJson, ToolCallId, ToolName};
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "type": "tool_call",
///         "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
///         "name": "get_stock_price",
///         "arguments": {"ticker": "^GSPC"}
///     }"#,
/// ).unwrap();
/// let value = ClaudeTurnItem::ToolCall {
///     id: ToolCallId::from("toolu_01D7FLrfh4GYq7yT1ULFeyMV"),
///     name: ToolName::from("get_stock_price"),
///     arguments: RawJson::parse(r#"{"ticker":"^GSPC"}"#).unwrap(),
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ClaudeTurnItem>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq)]
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

impl Serialize for ClaudeTurnItem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ClaudeTurnItemRepr::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ClaudeTurnItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = ClaudeTurnItemReprOwned::deserialize(deserializer)?;
        match repr {
            ClaudeTurnItemReprOwned::Text { content } => Ok(Self::Text { content }),
            ClaudeTurnItemReprOwned::Thinking { content, signature } => {
                Ok(Self::Thinking { content, signature })
            }
            ClaudeTurnItemReprOwned::Reasoning { content } => Ok(Self::Reasoning { content }),
            ClaudeTurnItemReprOwned::Refusal { content } => Ok(Self::Refusal { content }),
            ClaudeTurnItemReprOwned::ToolCall {
                id,
                name,
                arguments,
            } => Ok(Self::ToolCall {
                id,
                name,
                arguments: RawJson::from_serializable(&arguments)
                    .map_err(serde::de::Error::custom)?,
            }),
        }
    }
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

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClaudeTurnItemRepr<'a> {
    Text {
        content: &'a str,
    },
    Thinking {
        content: &'a str,
        signature: &'a str,
    },
    Reasoning {
        content: &'a str,
    },
    Refusal {
        content: &'a str,
    },
    ToolCall {
        id: &'a ToolCallId,
        name: &'a ToolName,
        arguments: serde_json::Value,
    },
}

impl<'a> From<&'a ClaudeTurnItem> for ClaudeTurnItemRepr<'a> {
    fn from(value: &'a ClaudeTurnItem) -> Self {
        match value {
            ClaudeTurnItem::Text { content } => Self::Text { content },
            ClaudeTurnItem::Thinking { content, signature } => {
                Self::Thinking { content, signature }
            }
            ClaudeTurnItem::Reasoning { content } => Self::Reasoning { content },
            ClaudeTurnItem::Refusal { content } => Self::Refusal { content },
            ClaudeTurnItem::ToolCall {
                id,
                name,
                arguments,
            } => Self::ToolCall {
                id,
                name,
                arguments: serde_json::from_str(arguments.get()).expect("validated JSON"),
            },
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ClaudeTurnItemReprOwned {
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
        arguments: serde_json::Value,
    },
}
