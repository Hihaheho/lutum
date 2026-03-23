use agents_protocol::conversation::{ToolCallId, ToolName};
use serde::{Deserialize, Serialize};

/// Claude message role.
///
/// ```
/// use agents_claude::messages::content::ClaudeRole;
///
/// let json = serde_json::from_str::<serde_json::Value>(r#""user""#).unwrap();
/// let value = ClaudeRole::User;
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ClaudeRole>(json).unwrap(), value);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ClaudeRole {
    User,
    Assistant,
}

/// Claude message payload.
///
/// ```
/// use agents_claude::messages::content::{
///     ClaudeContentBlock, ClaudeMessage, ClaudeRole, TextBlock,
/// };
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "role": "user",
///         "content": [{"type": "text", "text": "Hello, Claude"}]
///     }"#,
/// ).unwrap();
/// let value = ClaudeMessage {
///     role: ClaudeRole::User,
///     content: vec![ClaudeContentBlock::Text(TextBlock {
///         text: "Hello, Claude".to_string(),
///     })],
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ClaudeMessage>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClaudeMessage {
    pub role: ClaudeRole,
    pub content: Vec<ClaudeContentBlock>,
}

/// Claude content block.
///
/// ```
/// use agents_claude::messages::content::{ClaudeContentBlock, ToolUseBlock};
/// use agents_protocol::conversation::{ToolCallId, ToolName};
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "type": "tool_use",
///         "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
///         "name": "get_stock_price",
///         "input": {"ticker": "^GSPC"}
///     }"#,
/// ).unwrap();
/// let value = ClaudeContentBlock::ToolUse(ToolUseBlock {
///     id: ToolCallId::from("toolu_01D7FLrfh4GYq7yT1ULFeyMV"),
///     name: ToolName::from("get_stock_price"),
///     input: serde_json::from_str(r#"{"ticker":"^GSPC"}"#).unwrap(),
/// });
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ClaudeContentBlock>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClaudeContentBlock {
    Text(TextBlock),
    ToolUse(ToolUseBlock),
    ToolResult(ToolResultBlock),
    Thinking(ThinkingBlock),
}

/// Text content block.
///
/// ```
/// use agents_claude::messages::content::TextBlock;
///
/// let json = serde_json::from_str::<serde_json::Value>(r#"{"text":"Hello, Claude"}"#).unwrap();
/// let value = TextBlock {
///     text: "Hello, Claude".to_string(),
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<TextBlock>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TextBlock {
    pub text: String,
}

/// Tool use block.
///
/// ```
/// use agents_claude::messages::content::ToolUseBlock;
/// use agents_protocol::conversation::{ToolCallId, ToolName};
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
///         "name": "get_stock_price",
///         "input": {"ticker": "^GSPC"}
///     }"#,
/// ).unwrap();
/// let value = ToolUseBlock {
///     id: ToolCallId::from("toolu_01D7FLrfh4GYq7yT1ULFeyMV"),
///     name: ToolName::from("get_stock_price"),
///     input: serde_json::from_str(r#"{"ticker":"^GSPC"}"#).unwrap(),
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ToolUseBlock>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolUseBlock {
    pub id: ToolCallId,
    pub name: ToolName,
    pub input: serde_json::Value,
}

/// Tool result block.
///
/// ```
/// use agents_claude::messages::content::ToolResultBlock;
/// use agents_protocol::conversation::ToolCallId;
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
///         "content": "259.75 USD"
///     }"#,
/// ).unwrap();
/// let value = ToolResultBlock {
///     tool_use_id: ToolCallId::from("toolu_01D7FLrfh4GYq7yT1ULFeyMV"),
///     content: serde_json::from_str(r#""259.75 USD""#).unwrap(),
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ToolResultBlock>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolResultBlock {
    pub tool_use_id: ToolCallId,
    pub content: serde_json::Value,
}

/// Thinking block.
///
/// ```
/// use agents_claude::messages::content::ThinkingBlock;
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "thinking": "Let me solve this step by step.",
///         "signature": "sig_123"
///     }"#,
/// ).unwrap();
/// let value = ThinkingBlock {
///     thinking: "Let me solve this step by step.".to_string(),
///     signature: "sig_123".to_string(),
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ThinkingBlock>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ThinkingBlock {
    pub thinking: String,
    /// Anthropic's spec requires this field. Ollama's Anthropic-compat endpoint omits it,
    /// so we accept a missing field as an empty string.
    ///
    /// **Important:** do not round-trip thinking blocks with an empty signature back to
    /// Anthropic (e.g. in multi-turn tool loops) — Anthropic uses the signature for
    /// integrity verification and will reject an empty value.
    #[serde(default)]
    pub signature: String,
}
