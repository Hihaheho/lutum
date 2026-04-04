use serde::{Deserialize, Serialize};

/// Claude tool definition.
///
/// ```
/// use lutum_claude::messages::tool::ClaudeTool;
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{
///         "name": "get_stock_price",
///         "description": "Get the current stock price for a given ticker symbol.",
///         "input_schema": {
///             "type": "object",
///             "properties": {
///                 "ticker": {
///                     "type": "string",
///                     "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
///                 }
///             },
///             "required": ["ticker"]
///         }
///     }"#,
/// ).unwrap();
/// let value = ClaudeTool {
///     name: "get_stock_price".to_string(),
///     description: Some("Get the current stock price for a given ticker symbol.".to_string()),
///     input_schema: serde_json::from_str(
///         r#"{
///             "type": "object",
///             "properties": {
///                 "ticker": {
///                     "type": "string",
///                     "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
///                 }
///             },
///             "required": ["ticker"]
///         }"#,
///     ).unwrap(),
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ClaudeTool>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClaudeTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// Claude tool choice.
///
/// ```
/// use lutum_claude::messages::tool::ClaudeToolChoice;
///
/// let json = serde_json::from_str::<serde_json::Value>(
///     r#"{"type":"tool","name":"get_stock_price"}"#,
/// ).unwrap();
/// let value = ClaudeToolChoice::Tool {
///     name: "get_stock_price".to_string(),
///     disable_parallel_tool_use: None,
/// };
///
/// assert_eq!(serde_json::to_value(&value).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ClaudeToolChoice>(json).unwrap(), value);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClaudeToolChoice {
    Auto {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    None,
}
