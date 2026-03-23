use serde::{Deserialize, Serialize};

use crate::responses::{InputItem, OpenAiTool, ToolChoice};

/// ```
/// use agents_openai::responses::ResponsesRequest;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "model": "gpt-5.4",
///       "input": [
///         {
///           "role": "user",
///           "content": [
///             { "type": "input_text", "text": "What is the weather like in Boston today?" }
///           ],
///           "type": "message"
///         }
///       ],
///       "stream": true,
///       "tools": [
///         {
///           "type": "function",
///           "name": "get_current_weather",
///           "description": "Get the current weather in a given location",
///           "parameters": {
///             "type": "object",
///             "properties": {
///               "location": {
///                 "type": "string",
///                 "description": "The city and state, e.g. San Francisco, CA"
///               },
///               "unit": {
///                 "type": "string",
///                 "enum": ["celsius", "fahrenheit"]
///               }
///             },
///             "required": ["location", "unit"]
///           },
///           "strict": true
///         }
///       ],
///       "parallel_tool_calls": true,
///       "reasoning": { "effort": "xhigh" },
///       "text": { "format": { "type": "text" } },
///       "tool_choice": "required",
///       "seed": 42
///     }"#,
/// )
/// .unwrap();
/// let request = serde_json::from_value::<ResponsesRequest>(json.clone()).unwrap();
/// assert_eq!(request.seed, Some(42));
/// assert_eq!(serde_json::to_value(&request).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponsesRequest>(json).unwrap(), request);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: Vec<InputItem>,
    pub stream: bool,
    pub tools: Vec<OpenAiTool>,
    pub parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponsesReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponsesTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// ```
/// use agents_openai::responses::ResponsesReasoningConfig;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{ "effort": "xhigh" }"#).unwrap();
/// let config = serde_json::from_value::<ResponsesReasoningConfig>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&config).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponsesReasoningConfig>(json).unwrap(), config);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponsesReasoningConfig {
    pub effort: OpenAiReasoningEffort,
}

/// ```
/// use agents_openai::responses::ResponsesTextConfig;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{ "format": { "type": "text" } }"#).unwrap();
/// let config = serde_json::from_value::<ResponsesTextConfig>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&config).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponsesTextConfig>(json).unwrap(), config);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResponsesTextConfig {
    pub format: TextFormat,
}

/// ```
/// use agents_openai::responses::TextFormat;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{ "type": "text" }"#).unwrap();
/// let format = serde_json::from_value::<TextFormat>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&format).unwrap(), json);
/// assert_eq!(serde_json::from_value::<TextFormat>(json).unwrap(), format);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextFormat {
    Text,
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    JsonObject,
}

/// ```
/// use agents_openai::OpenAiReasoningEffort;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#""xhigh""#).unwrap();
/// let effort = serde_json::from_value::<OpenAiReasoningEffort>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&effort).unwrap(), json);
/// assert_eq!(serde_json::from_value::<OpenAiReasoningEffort>(json).unwrap(), effort);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenAiReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

/// ```
/// use agents_openai::responses::ReasoningEffort;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#""none""#).unwrap();
/// let effort = serde_json::from_value::<ReasoningEffort>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&effort).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ReasoningEffort>(json).unwrap(), effort);
/// ```
pub use OpenAiReasoningEffort as ReasoningEffort;
