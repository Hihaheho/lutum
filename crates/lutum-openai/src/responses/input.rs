use serde::{Deserialize, Serialize};
use serde_json::Value;

/// ```
/// use lutum_openai::responses::InputItem;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "role": "user",
///       "content": [{ "type": "input_text", "text": "what is in this image?" }],
///       "type": "message"
///     }"#,
/// )
/// .unwrap();
/// let item = serde_json::from_value::<InputItem>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&item).unwrap(), json);
/// assert_eq!(serde_json::from_value::<InputItem>(json).unwrap(), item);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputItem {
    Message(InputMessage),
    FunctionCall(FunctionCallItem),
    FunctionCallOutput(FunctionCallOutputItem),
    Reasoning(ReasoningItem),
}

impl From<InputMessage> for InputItem {
    fn from(value: InputMessage) -> Self {
        Self::Message(value)
    }
}

impl From<FunctionCallItem> for InputItem {
    fn from(value: FunctionCallItem) -> Self {
        Self::FunctionCall(value)
    }
}

impl From<FunctionCallOutputItem> for InputItem {
    fn from(value: FunctionCallOutputItem) -> Self {
        Self::FunctionCallOutput(value)
    }
}

impl From<ReasoningItem> for InputItem {
    fn from(value: ReasoningItem) -> Self {
        Self::Reasoning(value)
    }
}

/// ```
/// use lutum_openai::responses::InputMessage;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "role": "user",
///       "content": [{ "type": "input_text", "text": "what is in this image?" }],
///       "type": "message"
///     }"#,
/// )
/// .unwrap();
/// let message = serde_json::from_value::<InputMessage>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&message).unwrap(), json);
/// assert_eq!(serde_json::from_value::<InputMessage>(json).unwrap(), message);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputMessage {
    pub content: Vec<InputContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub role: MessageRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(rename = "type")]
    pub item_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
}

impl InputMessage {
    pub fn new(role: MessageRole, content: Vec<InputContent>) -> Self {
        Self {
            content,
            id: None,
            role,
            status: None,
            item_type: "message".to_string(),
            phase: None,
        }
    }
}

/// ```
/// use lutum_openai::responses::MessageRole;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#""assistant""#).unwrap();
/// let role = serde_json::from_value::<MessageRole>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&role).unwrap(), json);
/// assert_eq!(serde_json::from_value::<MessageRole>(json).unwrap(), role);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Developer,
}

/// ```
/// use lutum_openai::responses::InputContent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{ "type": "input_text", "text": "what is in this image?" }"#,
/// )
/// .unwrap();
/// let content = serde_json::from_value::<InputContent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&content).unwrap(), json);
/// assert_eq!(serde_json::from_value::<InputContent>(json).unwrap(), content);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputContent {
    OutputText(OutputTextContent),
    Refusal(RefusalContent),
    InputText(InputTextContent),
}

impl From<InputTextContent> for InputContent {
    fn from(value: InputTextContent) -> Self {
        Self::InputText(value)
    }
}

impl From<OutputTextContent> for InputContent {
    fn from(value: OutputTextContent) -> Self {
        Self::OutputText(value)
    }
}

impl From<RefusalContent> for InputContent {
    fn from(value: RefusalContent) -> Self {
        Self::Refusal(value)
    }
}

/// ```
/// use lutum_openai::responses::InputTextContent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{ "type": "input_text", "text": "what is in this image?" }"#,
/// )
/// .unwrap();
/// let content = serde_json::from_value::<InputTextContent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&content).unwrap(), json);
/// assert_eq!(serde_json::from_value::<InputTextContent>(json).unwrap(), content);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputTextContent {
    pub text: String,
    #[serde(rename = "type")]
    pub item_type: String,
}

impl InputTextContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            item_type: "input_text".to_string(),
        }
    }
}

/// ```
/// use lutum_openai::responses::OutputTextContent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type": "output_text",
///       "text": "Hi there! How can I assist you today?",
///       "annotations": []
///     }"#,
/// )
/// .unwrap();
/// let content = serde_json::from_value::<OutputTextContent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&content).unwrap(), json);
/// assert_eq!(serde_json::from_value::<OutputTextContent>(json).unwrap(), content);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutputTextContent {
    #[serde(default)]
    pub annotations: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<Value>>,
    pub text: String,
    #[serde(rename = "type")]
    pub item_type: String,
}

impl OutputTextContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            annotations: Vec::new(),
            logprobs: None,
            text: text.into(),
            item_type: "output_text".to_string(),
        }
    }
}

/// ```
/// use lutum_openai::responses::RefusalContent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{ "type": "refusal", "refusal": "I can't help with that request." }"#,
/// )
/// .unwrap();
/// let content = serde_json::from_value::<RefusalContent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&content).unwrap(), json);
/// assert_eq!(serde_json::from_value::<RefusalContent>(json).unwrap(), content);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RefusalContent {
    pub refusal: String,
    #[serde(rename = "type")]
    pub item_type: String,
}

impl RefusalContent {
    pub fn new(refusal: impl Into<String>) -> Self {
        Self {
            refusal: refusal.into(),
            item_type: "refusal".to_string(),
        }
    }
}

/// ```
/// use lutum_openai::responses::FunctionCallItem;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type": "function_call",
///       "id": "fc_67ca09c6bedc8190a7abfec07b1a1332096610f474011cc0",
///       "call_id": "call_unLAR8MvFNptuiZK6K6HCy5k",
///       "name": "get_current_weather",
///       "arguments": "{\"location\":\"Boston, MA\",\"unit\":\"celsius\"}",
///       "status": "completed"
///     }"#,
/// )
/// .unwrap();
/// let item = serde_json::from_value::<FunctionCallItem>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&item).unwrap(), json);
/// assert_eq!(serde_json::from_value::<FunctionCallItem>(json).unwrap(), item);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunctionCallItem {
    pub arguments: String,
    pub call_id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub item_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

impl FunctionCallItem {
    pub fn new(
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            arguments: arguments.into(),
            call_id: call_id.into(),
            name: name.into(),
            item_type: "function_call".to_string(),
            id: None,
            namespace: None,
            status: None,
        }
    }
}

/// ```
/// use lutum_openai::responses::FunctionCallOutputItem;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type": "function_call_output",
///       "call_id": "call_unLAR8MvFNptuiZK6K6HCy5k",
///       "output": "{\"temperature\":\"72\"}"
///     }"#,
/// )
/// .unwrap();
/// let item = serde_json::from_value::<FunctionCallOutputItem>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&item).unwrap(), json);
/// assert_eq!(serde_json::from_value::<FunctionCallOutputItem>(json).unwrap(), item);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunctionCallOutputItem {
    pub call_id: String,
    pub output: Value,
    #[serde(rename = "type")]
    pub item_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

impl FunctionCallOutputItem {
    pub fn new(call_id: impl Into<String>, output: Value) -> Self {
        Self {
            call_id: call_id.into(),
            output,
            item_type: "function_call_output".to_string(),
            id: None,
            status: None,
        }
    }
}

/// ```
/// use lutum_openai::responses::ReasoningItem;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "id": "rs_67ccd7eca01881908ff0b5146584e408072912b2993db808",
///       "summary": [{ "type": "summary_text", "text": "The classic tongue twister..." }],
///       "type": "reasoning",
///       "status": "completed"
///     }"#,
/// )
/// .unwrap();
/// let item = serde_json::from_value::<ReasoningItem>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&item).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ReasoningItem>(json).unwrap(), item);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReasoningItem {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub summary: Vec<SummaryText>,
    #[serde(rename = "type")]
    pub item_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

impl ReasoningItem {
    pub fn new(summary: Vec<SummaryText>) -> Self {
        Self {
            id: None,
            summary,
            item_type: "reasoning".to_string(),
            content: None,
            encrypted_content: None,
            status: None,
        }
    }

    pub fn summary_text(text: impl Into<String>) -> Self {
        Self::new(vec![SummaryText::new(text)])
    }
}

/// ```
/// use lutum_openai::responses::SummaryText;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{ "type": "summary_text", "text": "The classic tongue twister..." }"#,
/// )
/// .unwrap();
/// let item = serde_json::from_value::<SummaryText>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&item).unwrap(), json);
/// assert_eq!(serde_json::from_value::<SummaryText>(json).unwrap(), item);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SummaryText {
    pub text: String,
    #[serde(rename = "type")]
    pub item_type: String,
}

impl SummaryText {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            item_type: "summary_text".to_string(),
        }
    }
}
