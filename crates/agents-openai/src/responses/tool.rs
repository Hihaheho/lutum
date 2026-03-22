use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as DeError};
use serde_json::Value;

/// ```
/// use agents_openai::responses::OpenAiTool;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{
///       "type": "function",
///       "description": "Get the current weather in a given location",
///       "name": "get_current_weather",
///       "parameters": {
///         "type": "object",
///         "properties": {
///           "location": {
///             "type": "string",
///             "description": "The city and state, e.g. San Francisco, CA"
///           },
///           "unit": {
///             "type": "string",
///             "enum": ["celsius", "fahrenheit"]
///           }
///         },
///         "required": ["location", "unit"]
///       },
///       "strict": true
///     }"#,
/// )
/// .unwrap();
/// let tool = serde_json::from_value::<OpenAiTool>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&tool).unwrap(), json);
/// assert_eq!(serde_json::from_value::<OpenAiTool>(json).unwrap(), tool);
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpenAiTool {
    pub name: String,
    pub parameters: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    pub r#type: OpenAiToolKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl OpenAiTool {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            name: name.into(),
            parameters,
            strict: None,
            r#type: OpenAiToolKind::Function,
            defer_loading: None,
            description: Some(description.into()),
        }
    }
}

/// ```
/// use agents_openai::responses::OpenAiToolKind;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#""function""#).unwrap();
/// let kind = serde_json::from_value::<OpenAiToolKind>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&kind).unwrap(), json);
/// assert_eq!(serde_json::from_value::<OpenAiToolKind>(json).unwrap(), kind);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiToolKind {
    Function,
}

/// ```
/// use agents_openai::responses::FunctionToolChoice;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(
///     r#"{ "type": "function", "name": "get_current_weather" }"#,
/// )
/// .unwrap();
/// let choice = serde_json::from_value::<FunctionToolChoice>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&choice).unwrap(), json);
/// assert_eq!(serde_json::from_value::<FunctionToolChoice>(json).unwrap(), choice);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FunctionToolChoice {
    pub name: String,
    pub r#type: OpenAiToolKind,
}

impl FunctionToolChoice {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            r#type: OpenAiToolKind::Function,
        }
    }
}

/// ```
/// use agents_openai::responses::ToolChoice;
/// use serde_json::Value;
///
/// let json = serde_json::from_str::<Value>(r#""required""#).unwrap();
/// let choice = serde_json::from_value::<ToolChoice>(json.clone()).unwrap();
/// assert_eq!(serde_json::to_value(&choice).unwrap(), json);
/// assert_eq!(serde_json::from_value::<ToolChoice>(json).unwrap(), choice);
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function(FunctionToolChoice),
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::None => serializer.serialize_str("none"),
            Self::Auto => serializer.serialize_str("auto"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function(choice) => choice.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum ToolChoiceWire {
            Literal(String),
            Function(FunctionToolChoice),
        }

        match ToolChoiceWire::deserialize(deserializer)? {
            ToolChoiceWire::Literal(value) => match value.as_str() {
                "none" => Ok(Self::None),
                "auto" => Ok(Self::Auto),
                "required" => Ok(Self::Required),
                other => Err(D::Error::unknown_variant(
                    other,
                    &["none", "auto", "required"],
                )),
            },
            ToolChoiceWire::Function(choice) => Ok(Self::Function(choice)),
        }
    }
}
