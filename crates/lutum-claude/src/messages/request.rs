use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::messages::{ClaudeMessage, ClaudeTool};

/// Claude messages request.
///
/// ```
/// use lutum_claude::messages::{
///     ClaudeContentBlock, ClaudeMessage, ClaudeRole, MessagesRequest, OutputConfig, OutputFormat,
///     SystemBlock, TextBlock, ThinkingConfig, ThinkingKind,
/// };
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "model": "claude-opus-4-6",
///         "max_tokens": 1024,
///         "messages": [
///             {
///                 "role": "user",
///                 "content": [{"type": "text", "text": "Hello, world"}]
///             }
///         ],
///         "system": [{"type": "text", "text": "Today's date is 2023-01-01."}],
///         "thinking": {"type": "enabled", "budget_tokens": 2048},
///         "output_config": {
///             "format": {
///                 "type": "json_schema",
///                 "schema": {"type": "object"}
///             }
///         }
///     }"#,
/// ).unwrap();
/// let value = MessagesRequest {
///     model: "claude-opus-4-6".to_string(),
///     max_tokens: 1024,
///     messages: vec![ClaudeMessage {
///         role: ClaudeRole::User,
///         content: vec![ClaudeContentBlock::Text(TextBlock {
///             text: "Hello, world".to_string(),
///             cache_control: None,
///         })],
///     }],
///     stream: None,
///     system: Some(vec![SystemBlock::new("Today's date is 2023-01-01.")]),
///     temperature: None,
///     tools: None,
///     tool_choice: None,
///     thinking: Some(ThinkingConfig {
///         kind: ThinkingKind::Enabled,
///         budget_tokens: 2048,
///     }),
///     output_config: Some(OutputConfig {
///         format: OutputFormat::JsonSchema {
///             schema: serde_json::from_str(r#"{"type":"object"}"#).unwrap(),
///         },
///     }),
///     stop_sequences: None,
///     models: None,
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<MessagesRequest>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<ClaudeMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<SystemBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ClaudeTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<crate::messages::ClaudeToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
}

/// System prompt text block.
///
/// ```
/// use lutum_claude::messages::request::SystemBlock;
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{"type":"text","text":"Today's date is 2023-01-01."}"#,
/// ).unwrap();
/// let value = SystemBlock::new("Today's date is 2023-01-01.");
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<SystemBlock>(json).unwrap(), value);
/// ```
/// Cache control configuration for a system block.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub kind: String,
}

impl CacheControl {
    /// Ephemeral cache control — cache this block for the current session.
    pub fn ephemeral() -> Self {
        Self {
            kind: "ephemeral".to_string(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SystemBlock {
    pub text: String,
    pub cache_control: Option<CacheControl>,
}

impl SystemBlock {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            cache_control: None,
        }
    }

    pub fn with_cache_control(mut self, cc: CacheControl) -> Self {
        self.cache_control = Some(cc);
        self
    }
}

impl From<String> for SystemBlock {
    fn from(text: String) -> Self {
        Self::new(text)
    }
}

impl Serialize for SystemBlock {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SystemBlockRepr {
            kind: "text",
            text: &self.text,
            cache_control: self.cache_control.as_ref(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SystemBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = SystemBlockOwnedRepr::deserialize(deserializer)?;
        if repr.kind != "text" {
            return Err(serde::de::Error::custom(
                "Claude system blocks must use type `text`",
            ));
        }
        Ok(Self {
            text: repr.text,
            cache_control: repr.cache_control,
        })
    }
}

/// Thinking configuration.
///
/// ```
/// use lutum_claude::messages::request::{ThinkingConfig, ThinkingKind};
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{"type":"enabled","budget_tokens":2048}"#,
/// ).unwrap();
/// let value = ThinkingConfig {
///     kind: ThinkingKind::Enabled,
///     budget_tokens: 2048,
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ThinkingConfig>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub kind: ThinkingKind,
    pub budget_tokens: u32,
}

/// Thinking mode.
///
/// ```
/// use lutum_claude::messages::request::ThinkingKind;
///
/// let json = serde_json::from_str::<serde_json::Value>(r#""enabled""#).unwrap();
/// let value = ThinkingKind::Enabled;
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ThinkingKind>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingKind {
    Enabled,
}

/// Output configuration.
///
/// ```
/// use lutum_claude::messages::request::{OutputConfig, OutputFormat};
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "format": {
///             "type": "json_schema",
///             "schema": {"type": "object"}
///         }
///     }"#,
/// ).unwrap();
/// let value = OutputConfig {
///     format: OutputFormat::JsonSchema {
///         schema: serde_json::from_str(r#"{"type":"object"}"#).unwrap(),
///     },
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<OutputConfig>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutputConfig {
    pub format: OutputFormat,
}

/// Output format.
///
/// ```
/// use lutum_claude::messages::request::OutputFormat;
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "type": "json_schema",
///         "schema": {"type": "object"}
///     }"#,
/// ).unwrap();
/// let value = OutputFormat::JsonSchema {
///     schema: serde_json::from_str(r#"{"type":"object"}"#).unwrap(),
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<OutputFormat>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputFormat {
    JsonSchema { schema: serde_json::Value },
}

#[derive(Serialize)]
struct SystemBlockRepr<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<&'a CacheControl>,
}

#[derive(Deserialize)]
struct SystemBlockOwnedRepr {
    #[serde(rename = "type")]
    kind: String,
    text: String,
    #[serde(default)]
    cache_control: Option<CacheControl>,
}
