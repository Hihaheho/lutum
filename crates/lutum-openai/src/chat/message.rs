use serde::{Deserialize, Serialize};

/// Image detail level for vision inputs.
///
/// ```
/// use lutum_openai::chat::ImageDetail;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#""auto""#).unwrap();
/// let val = serde_json::from_value::<ImageDetail>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ImageDetail>(json).unwrap(), val);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
}

/// Audio format for input audio content parts.
///
/// ```
/// use lutum_openai::chat::AudioFormat;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#""wav""#).unwrap();
/// let val = serde_json::from_value::<AudioFormat>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<AudioFormat>(json).unwrap(), val);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    Wav,
    Mp3,
}

/// Image URL reference for vision inputs.
///
/// ```
/// use lutum_openai::chat::ImageUrl;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"url":"https://example.com/img.png"}"#).unwrap();
/// let val = serde_json::from_value::<ImageUrl>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ImageUrl>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

/// Base64-encoded audio for input audio content parts.
///
/// ```
/// use lutum_openai::chat::InputAudio;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"data":"base64==","format":"mp3"}"#).unwrap();
/// let val = serde_json::from_value::<InputAudio>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<InputAudio>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InputAudio {
    pub data: String,
    pub format: AudioFormat,
}

/// File reference for file content parts.
///
/// ```
/// use lutum_openai::chat::FileInput;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"file_id":"file-abc"}"#).unwrap();
/// let val = serde_json::from_value::<FileInput>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<FileInput>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FileInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}

/// Content part for user messages: text, image, audio, or file.
///
/// ```
/// use lutum_openai::chat::ChatContentPart;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"type":"text","text":"hello"}"#).unwrap();
/// let val = serde_json::from_value::<ChatContentPart>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatContentPart>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
    InputAudio { input_audio: InputAudio },
    File { file: FileInput },
}

/// Content part for assistant messages: text or refusal.
///
/// ```
/// use lutum_openai::chat::AssistantContentPart;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"type":"text","text":"hello"}"#).unwrap();
/// let val = serde_json::from_value::<AssistantContentPart>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<AssistantContentPart>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantContentPart {
    Text { text: String },
    Refusal { refusal: String },
}

/// Content for developer/system/tool messages: plain string or array of text parts.
///
/// ```
/// use lutum_openai::chat::ChatTextContent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#""hello world""#).unwrap();
/// let val = serde_json::from_value::<ChatTextContent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatTextContent>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatTextContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

/// Content for user messages: plain string or array of content parts.
///
/// ```
/// use lutum_openai::chat::ChatUserContent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#""hello""#).unwrap();
/// let val = serde_json::from_value::<ChatUserContent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatUserContent>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatUserContent {
    Text(String),
    Parts(Vec<ChatContentPart>),
}

/// Content for assistant messages: plain string or array of text/refusal parts.
///
/// ```
/// use lutum_openai::chat::AssistantContent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#""I cannot help with that.""#).unwrap();
/// let val = serde_json::from_value::<AssistantContent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<AssistantContent>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AssistantContent {
    Text(String),
    Parts(Vec<AssistantContentPart>),
}

/// Reference to a previous audio response, used in assistant message input.
///
/// ```
/// use lutum_openai::chat::ChatAssistantAudioRef;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"id":"audio_abc123"}"#).unwrap();
/// let val = serde_json::from_value::<ChatAssistantAudioRef>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatAssistantAudioRef>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChatAssistantAudioRef {
    pub id: String,
}

/// The function that was called, as returned by the model in a tool call.
///
/// ```
/// use lutum_openai::chat::ChatFunctionCallArgs;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{"arguments":"{\"city\":\"Boston\"}","name":"get_weather"}"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<ChatFunctionCallArgs>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatFunctionCallArgs>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChatFunctionCallArgs {
    pub arguments: String,
    pub name: String,
}

/// Deprecated — use `tool_calls` instead. Kept for round-trip fidelity.
pub type ChatFunctionCallLegacy = ChatFunctionCallArgs;

/// Function tool call in a message (input or output).
///
/// ```
/// use lutum_openai::chat::ChatMessageFunctionToolCall;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{"id":"call_abc","type":"function","function":{"arguments":"{}","name":"get_weather"}}"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<ChatMessageFunctionToolCall>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatMessageFunctionToolCall>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatMessageFunctionToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub function: ChatFunctionCallArgs,
}

impl ChatMessageFunctionToolCall {
    pub fn new(id: impl Into<String>, function: ChatFunctionCallArgs) -> Self {
        Self {
            id: id.into(),
            type_: "function".into(),
            function,
        }
    }
}

/// Content of a custom tool call.
///
/// ```
/// use lutum_openai::chat::ChatMessageCustomCallContent;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"input":"value","name":"my_tool"}"#).unwrap();
/// let val = serde_json::from_value::<ChatMessageCustomCallContent>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatMessageCustomCallContent>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChatMessageCustomCallContent {
    pub input: String,
    pub name: String,
}

/// Custom tool call in a message (input or output).
///
/// ```
/// use lutum_openai::chat::ChatMessageCustomToolCall;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{"id":"call_xyz","type":"custom","custom":{"input":"val","name":"my_tool"}}"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<ChatMessageCustomToolCall>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatMessageCustomToolCall>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatMessageCustomToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub custom: ChatMessageCustomCallContent,
}

impl ChatMessageCustomToolCall {
    pub fn new(id: impl Into<String>, custom: ChatMessageCustomCallContent) -> Self {
        Self {
            id: id.into(),
            type_: "custom".into(),
            custom,
        }
    }
}

/// A tool call in a message — either a function call or a custom tool call.
///
/// ```
/// use lutum_openai::chat::ChatMessageToolCall;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{"id":"call_abc","type":"function","function":{"arguments":"{}","name":"get_weather"}}"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<ChatMessageToolCall>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatMessageToolCall>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatMessageToolCall {
    Function(ChatMessageFunctionToolCall),
    Custom(ChatMessageCustomToolCall),
}

/// Developer-provided instructions (replaces `system` for o1+ models).
///
/// Serialized with `"role": "developer"` injected by [`ChatMessageParam`].
///
/// ```
/// use lutum_openai::chat::ChatDeveloperMessage;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"content":"You are helpful."}"#).unwrap();
/// let val = serde_json::from_value::<ChatDeveloperMessage>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatDeveloperMessage>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatDeveloperMessage {
    pub content: ChatTextContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// System-level instructions (use `developer` for o1+ models).
///
/// Serialized with `"role": "system"` injected by [`ChatMessageParam`].
///
/// ```
/// use lutum_openai::chat::ChatSystemMessage;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"content":"Be concise."}"#).unwrap();
/// let val = serde_json::from_value::<ChatSystemMessage>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatSystemMessage>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatSystemMessage {
    pub content: ChatTextContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message sent by the end user.
///
/// Serialized with `"role": "user"` injected by [`ChatMessageParam`].
///
/// ```
/// use lutum_openai::chat::ChatUserMessage;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"content":"Hello!"}"#).unwrap();
/// let val = serde_json::from_value::<ChatUserMessage>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatUserMessage>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatUserMessage {
    pub content: ChatUserContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Message representing a prior assistant turn, including any tool calls it made.
///
/// Serialized with `"role": "assistant"` injected by [`ChatMessageParam`].
///
/// ```
/// use lutum_openai::chat::ChatAssistantMessage;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"content":"Hi there!"}"#).unwrap();
/// let val = serde_json::from_value::<ChatAssistantMessage>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatAssistantMessage>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatAssistantMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<AssistantContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ChatAssistantAudioRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatMessageToolCall>>,
    /// Deprecated — use `tool_calls` instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<ChatFunctionCallLegacy>,
}

/// Tool result message sent after a tool call.
///
/// Serialized with `"role": "tool"` injected by [`ChatMessageParam`].
///
/// ```
/// use lutum_openai::chat::ChatToolMessage;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{"content":"72°F, sunny.","tool_call_id":"call_abc"}"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<ChatToolMessage>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatToolMessage>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatToolMessage {
    pub content: ChatTextContent,
    pub tool_call_id: String,
}

/// Deprecated function result message.
///
/// Serialized with `"role": "function"` injected by [`ChatMessageParam`].
///
/// ```
/// use lutum_openai::chat::ChatFunctionMessage;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{"content":"72°F","name":"get_weather"}"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<ChatFunctionMessage>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatFunctionMessage>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatFunctionMessage {
    pub content: String,
    pub name: String,
}

/// A single message in a chat conversation — any role.
///
/// The `"role"` field in JSON is the enum discriminant; it is not stored in
/// the inner structs.
///
/// ```
/// use lutum_openai::chat::ChatMessageParam;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{"role":"user","content":"Hello!"}"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<ChatMessageParam>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatMessageParam>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessageParam {
    Developer(ChatDeveloperMessage),
    System(ChatSystemMessage),
    User(ChatUserMessage),
    Assistant(ChatAssistantMessage),
    Tool(ChatToolMessage),
    Function(ChatFunctionMessage),
}
