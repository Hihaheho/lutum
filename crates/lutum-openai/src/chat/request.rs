use serde::{Deserialize, Serialize};

use crate::responses::OpenAiReasoningEffort;

use super::{
    message::ChatMessageParam,
    output::ServiceTier,
    tool::{ChatTool, ChatToolChoice, ResponseFormat},
};

/// Options for streaming Chat Completions responses.
///
/// ```
/// use lutum_openai::chat::ChatStreamOptions;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{ "include_usage": true }"#).unwrap();
/// let val = serde_json::from_value::<ChatStreamOptions>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatStreamOptions>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatStreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_obfuscation: Option<bool>,
}

/// Stop sequences for Chat Completions — either a single string or an array of strings.
///
/// ```
/// use lutum_openai::chat::ChatStop;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"["stop", "end"]"#).unwrap();
/// let val = serde_json::from_value::<ChatStop>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatStop>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatStop {
    Single(String),
    Multiple(Vec<String>),
}

/// Request body for the OpenAI Chat Completions API (`POST /v1/chat/completions`).
///
/// ```
/// use lutum_openai::chat::ChatCompletionRequest;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{ "messages": [], "model": "gpt-4o", "stream": true }"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<ChatCompletionRequest>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatCompletionRequest>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    // Required
    pub messages: Vec<ChatMessageParam>,
    pub model: String,

    // Optional – sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    // Optional – output limits
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Deprecated — use `max_completion_tokens` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    // Optional – logprobs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    // Optional – stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<ChatStop>,

    // Optional – tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    // Optional – response format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    // Optional – reasoning
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<OpenAiReasoningEffort>,

    // Optional – streaming
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatStreamOptions>,

    // Optional – storage & routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// OpenRouter fallback models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,

    // Optional – user identity
    /// Deprecated — use `safety_identifier` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
}
