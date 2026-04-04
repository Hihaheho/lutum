use serde::{Deserialize, Serialize};

/// ```
/// use lutum_openai::CompletionRequest;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "model": "gpt-3.5-turbo-instruct",
///       "prompt": "hello",
///       "stream": true,
///       "temperature": 1.0,
///       "max_tokens": 32,
///       "stop": ["\n\n"],
///       "models": ["openrouter/fallback-1"]
///     }"#,
/// )
/// .unwrap();
/// let request = serde_json::from_value::<CompletionRequest>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&request).unwrap(), json);
/// assert_eq!(serde_json::from_value::<CompletionRequest>(json).unwrap(), request);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(rename = "max_tokens", skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
}
