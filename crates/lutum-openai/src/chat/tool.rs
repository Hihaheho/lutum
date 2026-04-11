use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as DeError};
use serde_json::Value;

/// Discriminant enum for tool type fields.
///
/// ```
/// use lutum_openai::chat::ChatToolKind;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#""function""#).unwrap();
/// let val = serde_json::from_value::<ChatToolKind>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatToolKind>(json).unwrap(), val);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatToolKind {
    Function,
    Custom,
    AllowedTools,
}

/// Definition of a function tool.
///
/// ```
/// use lutum_openai::chat::FunctionDefinition;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"name":"get_weather"}"#).unwrap();
/// let val = serde_json::from_value::<FunctionDefinition>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<FunctionDefinition>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Function tool for the Chat Completions API.
///
/// ```
/// use lutum_openai::chat::{ChatFunctionTool, FunctionDefinition};
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{"type":"function","function":{"name":"get_weather"}}"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<ChatFunctionTool>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatFunctionTool>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatFunctionTool {
    #[serde(rename = "type")]
    pub type_: ChatToolKind,
    pub function: FunctionDefinition,
}

impl ChatFunctionTool {
    pub fn new(function: FunctionDefinition) -> Self {
        Self {
            type_: ChatToolKind::Function,
            function,
        }
    }
}

/// Syntax for custom tool grammar definitions.
///
/// ```
/// use lutum_openai::chat::GrammarSyntax;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#""lark""#).unwrap();
/// let val = serde_json::from_value::<GrammarSyntax>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<GrammarSyntax>(json).unwrap(), val);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GrammarSyntax {
    Lark,
    Regex,
}

/// A grammar definition for a custom tool input format.
///
/// ```
/// use lutum_openai::chat::{Grammar, GrammarSyntax};
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{"definition":"[a-z]+","syntax":"regex"}"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<Grammar>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<Grammar>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Grammar {
    pub definition: String,
    pub syntax: GrammarSyntax,
}

/// Input format constraint for a custom tool.
///
/// ```
/// use lutum_openai::chat::CustomToolFormat;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"type":"text"}"#).unwrap();
/// let val = serde_json::from_value::<CustomToolFormat>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<CustomToolFormat>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum CustomToolFormat {
    Text,
    Grammar { grammar: Grammar },
}

/// Properties of a custom tool.
///
/// ```
/// use lutum_openai::chat::CustomToolDefinition;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"name":"my_tool"}"#).unwrap();
/// let val = serde_json::from_value::<CustomToolDefinition>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<CustomToolDefinition>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomToolDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<CustomToolFormat>,
}

/// Custom tool for the Chat Completions API.
///
/// ```
/// use lutum_openai::chat::{ChatCustomTool, CustomToolDefinition};
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{"type":"custom","custom":{"name":"my_tool"}}"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<ChatCustomTool>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatCustomTool>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatCustomTool {
    #[serde(rename = "type")]
    pub type_: ChatToolKind,
    pub custom: CustomToolDefinition,
}

impl ChatCustomTool {
    pub fn new(custom: CustomToolDefinition) -> Self {
        Self {
            type_: ChatToolKind::Custom,
            custom,
        }
    }
}

/// A tool the model may call — either a function tool or a custom tool.
///
/// ```
/// use lutum_openai::chat::{ChatTool, FunctionDefinition};
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{"type":"function","function":{"name":"get_weather"}}"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<ChatTool>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatTool>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatTool {
    Function(ChatFunctionTool),
    Custom(ChatCustomTool),
}

/// The name of a specific tool, used in named tool choice.
///
/// ```
/// use lutum_openai::chat::NamedToolInner;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"name":"my_fn"}"#).unwrap();
/// let val = serde_json::from_value::<NamedToolInner>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<NamedToolInner>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NamedToolInner {
    pub name: String,
}

/// Forces the model to call a specific function tool.
///
/// ```
/// use lutum_openai::chat::ChatNamedFunctionToolChoice;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{"type":"function","function":{"name":"my_fn"}}"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<ChatNamedFunctionToolChoice>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatNamedFunctionToolChoice>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChatNamedFunctionToolChoice {
    #[serde(rename = "type")]
    pub type_: ChatToolKind,
    pub function: NamedToolInner,
}

impl ChatNamedFunctionToolChoice {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            type_: ChatToolKind::Function,
            function: NamedToolInner { name: name.into() },
        }
    }
}

/// Forces the model to call a specific custom tool.
///
/// ```
/// use lutum_openai::chat::ChatNamedCustomToolChoice;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{"type":"custom","custom":{"name":"my_tool"}}"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<ChatNamedCustomToolChoice>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatNamedCustomToolChoice>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChatNamedCustomToolChoice {
    #[serde(rename = "type")]
    pub type_: ChatToolKind,
    pub custom: NamedToolInner,
}

impl ChatNamedCustomToolChoice {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            type_: ChatToolKind::Custom,
            custom: NamedToolInner { name: name.into() },
        }
    }
}

/// Whether the model picks from allowed tools automatically or must call one.
///
/// ```
/// use lutum_openai::chat::ChatAllowedToolsMode;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#""auto""#).unwrap();
/// let val = serde_json::from_value::<ChatAllowedToolsMode>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatAllowedToolsMode>(json).unwrap(), val);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatAllowedToolsMode {
    Auto,
    Required,
}

/// Constrains the model to a pre-defined set of allowed tools.
///
/// ```
/// use lutum_openai::chat::ChatAllowedToolsConfig;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"mode":"auto","tools":[]}"#).unwrap();
/// let val = serde_json::from_value::<ChatAllowedToolsConfig>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatAllowedToolsConfig>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatAllowedToolsConfig {
    pub mode: ChatAllowedToolsMode,
    pub tools: Vec<Value>,
}

/// Tool choice that constrains the model to a pre-defined set of tools.
///
/// ```
/// use lutum_openai::chat::ChatAllowedToolChoice;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{"type":"allowed_tools","allowed_tools":{"mode":"auto","tools":[]}}"#,
/// )
/// .unwrap();
/// let val = serde_json::from_value::<ChatAllowedToolChoice>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatAllowedToolChoice>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatAllowedToolChoice {
    #[serde(rename = "type")]
    pub type_: ChatToolKind,
    pub allowed_tools: ChatAllowedToolsConfig,
}

impl ChatAllowedToolChoice {
    pub fn new(config: ChatAllowedToolsConfig) -> Self {
        Self {
            type_: ChatToolKind::AllowedTools,
            allowed_tools: config,
        }
    }
}

/// Controls which tool the model calls.
///
/// ```
/// use lutum_openai::chat::ChatToolChoice;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#""auto""#).unwrap();
/// let val = serde_json::from_value::<ChatToolChoice>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ChatToolChoice>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum ChatToolChoice {
    None,
    Auto,
    Required,
    NamedFunction(ChatNamedFunctionToolChoice),
    NamedCustom(ChatNamedCustomToolChoice),
    AllowedTools(ChatAllowedToolChoice),
}

impl Serialize for ChatToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::None => serializer.serialize_str("none"),
            Self::Auto => serializer.serialize_str("auto"),
            Self::Required => serializer.serialize_str("required"),
            Self::NamedFunction(c) => c.serialize(serializer),
            Self::NamedCustom(c) => c.serialize(serializer),
            Self::AllowedTools(c) => c.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ChatToolChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire {
            Literal(String),
            NamedFunction(ChatNamedFunctionToolChoice),
            NamedCustom(ChatNamedCustomToolChoice),
            AllowedTools(ChatAllowedToolChoice),
        }

        match Wire::deserialize(deserializer)? {
            Wire::Literal(s) => match s.as_str() {
                "none" => Ok(Self::None),
                "auto" => Ok(Self::Auto),
                "required" => Ok(Self::Required),
                other => Err(D::Error::unknown_variant(
                    other,
                    &["none", "auto", "required"],
                )),
            },
            Wire::NamedFunction(c) => Ok(Self::NamedFunction(c)),
            Wire::NamedCustom(c) => Ok(Self::NamedCustom(c)),
            Wire::AllowedTools(c) => Ok(Self::AllowedTools(c)),
        }
    }
}

/// JSON Schema configuration for structured output.
///
/// ```
/// use lutum_openai::chat::JsonSchemaConfig;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"name":"my_schema"}"#).unwrap();
/// let val = serde_json::from_value::<JsonSchemaConfig>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<JsonSchemaConfig>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JsonSchemaConfig {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Output format constraint for the model response.
///
/// ```
/// use lutum_openai::chat::ResponseFormat;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#"{"type":"text"}"#).unwrap();
/// let val = serde_json::from_value::<ResponseFormat>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&val).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ResponseFormat>(json).unwrap(), val);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    Text,
    JsonSchema { json_schema: JsonSchemaConfig },
    JsonObject,
}
