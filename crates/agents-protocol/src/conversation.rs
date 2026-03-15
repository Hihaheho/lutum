use std::{borrow::Borrow, collections::BTreeMap, fmt};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::DeserializeOwned};
use serde_json::value::RawValue;
use thiserror::Error;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelInput {
    items: Vec<ModelInputItem>,
}

impl ModelInput {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn from_items(items: Vec<ModelInputItem>) -> Self {
        Self { items }
    }

    pub fn items(&self) -> &[ModelInputItem] {
        &self.items
    }

    pub fn into_items(self) -> Vec<ModelInputItem> {
        self.items
    }

    pub fn push(&mut self, item: ModelInputItem) {
        self.items.push(item);
    }

    pub fn system(mut self, text: impl Into<String>) -> Self {
        self.push(ModelInputItem::text(InputMessageRole::System, text));
        self
    }

    pub fn developer(mut self, text: impl Into<String>) -> Self {
        self.push(ModelInputItem::text(InputMessageRole::Developer, text));
        self
    }

    pub fn user(mut self, text: impl Into<String>) -> Self {
        self.push(ModelInputItem::text(InputMessageRole::User, text));
        self
    }

    pub fn assistant_text(mut self, text: impl Into<String>) -> Self {
        self.push(ModelInputItem::assistant_text(text));
        self
    }

    pub fn assistant_reasoning(mut self, text: impl Into<String>) -> Self {
        self.push(ModelInputItem::assistant_reasoning(text));
        self
    }

    pub fn assistant_refusal(mut self, text: impl Into<String>) -> Self {
        self.push(ModelInputItem::assistant_refusal(text));
        self
    }

    pub fn tool_use(mut self, tool_use: ToolUse) -> Self {
        self.push(ModelInputItem::tool_use(tool_use));
        self
    }

    pub fn append_assistant_turn(
        &mut self,
        turn: AssistantTurn,
        tool_uses: impl IntoIterator<Item = ToolUse>,
    ) -> Result<(), AssistantTurnInputError> {
        self.items.extend(turn.into_input_items(tool_uses)?);
        Ok(())
    }

    pub fn validate(&self) -> Result<(), ModelInputValidationError> {
        if self.items.is_empty() {
            return Err(ModelInputValidationError::Empty);
        }

        let mut tool_uses = std::collections::BTreeSet::new();
        for item in &self.items {
            if let ModelInputItem::ToolUse(tool_use) = item
                && !tool_uses.insert(tool_use.id.clone())
            {
                return Err(ModelInputValidationError::DuplicateToolUseId {
                    id: tool_use.id.clone(),
                });
            }
        }

        Ok(())
    }
}

impl From<Vec<ModelInputItem>> for ModelInput {
    fn from(items: Vec<ModelInputItem>) -> Self {
        Self::from_items(items)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ModelInputItem {
    Message {
        role: InputMessageRole,
        content: NonEmpty<MessageContent>,
    },
    Assistant(AssistantInputItem),
    ToolUse(ToolUse),
}

impl ModelInputItem {
    pub fn message(role: InputMessageRole, content: NonEmpty<MessageContent>) -> Self {
        Self::Message { role, content }
    }

    pub fn text(role: InputMessageRole, text: impl Into<String>) -> Self {
        Self::Message {
            role,
            content: NonEmpty::one(MessageContent::Text(text.into())),
        }
    }

    pub fn assistant(item: AssistantInputItem) -> Self {
        Self::Assistant(item)
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self::Assistant(AssistantInputItem::Text(text.into()))
    }

    pub fn assistant_reasoning(text: impl Into<String>) -> Self {
        Self::Assistant(AssistantInputItem::Reasoning(text.into()))
    }

    pub fn assistant_refusal(text: impl Into<String>) -> Self {
        Self::Assistant(AssistantInputItem::Refusal(text.into()))
    }

    pub fn tool_use(tool_use: ToolUse) -> Self {
        Self::ToolUse(tool_use)
    }

    pub fn tool_use_parts(
        id: impl Into<ToolCallId>,
        name: impl Into<ToolName>,
        arguments: RawJson,
        result: RawJson,
    ) -> Self {
        Self::ToolUse(ToolUse::new(id, name, arguments, result))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum InputMessageRole {
    System,
    Developer,
    User,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum MessageContent {
    Text(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Assistant-authored request items that can be replayed into a future model input.
///
/// This is intentionally narrower than [`AssistantTurnItem`]: tool calls are represented
/// as [`ToolUse`] at the surrounding [`ModelInputItem`] level so call/result pairs stay bundled.
pub enum AssistantInputItem {
    Text(String),
    Reasoning(String),
    Refusal(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolUse {
    pub id: ToolCallId,
    pub name: ToolName,
    pub arguments: RawJson,
    pub result: RawJson,
}

impl ToolUse {
    pub fn new(
        id: impl Into<ToolCallId>,
        name: impl Into<ToolName>,
        arguments: RawJson,
        result: RawJson,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
            result,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolMetadata {
    pub id: ToolCallId,
    pub name: ToolName,
    pub arguments: RawJson,
}

impl ToolMetadata {
    pub fn new(id: impl Into<ToolCallId>, name: impl Into<ToolName>, arguments: RawJson) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    pub fn into_tool_use(self, result: RawJson) -> ToolUse {
        ToolUse::new(self.id, self.name, self.arguments, result)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Canonical assistant output for a completed turn.
///
/// This remains richer than [`AssistantInputItem`] because the model can emit tool calls that
/// are not yet paired with tool results at response time.
pub struct AssistantTurn {
    items: NonEmpty<AssistantTurnItem>,
}

impl AssistantTurn {
    pub fn new(items: NonEmpty<AssistantTurnItem>) -> Self {
        Self { items }
    }

    pub fn from_items(items: Vec<AssistantTurnItem>) -> Result<Self, EmptyNonEmptyError> {
        Ok(Self::new(NonEmpty::try_from_vec(items)?))
    }

    pub fn items(&self) -> &[AssistantTurnItem] {
        self.items.as_slice()
    }

    pub fn items_non_empty(&self) -> &NonEmpty<AssistantTurnItem> {
        &self.items
    }

    pub fn into_items(self) -> NonEmpty<AssistantTurnItem> {
        self.items
    }

    pub fn text(text: impl Into<String>) -> Self {
        Self::new(NonEmpty::one(AssistantTurnItem::Text(text.into())))
    }

    pub fn reasoning(text: impl Into<String>) -> Self {
        Self::new(NonEmpty::one(AssistantTurnItem::Reasoning(text.into())))
    }

    pub fn refusal(text: impl Into<String>) -> Self {
        Self::new(NonEmpty::one(AssistantTurnItem::Refusal(text.into())))
    }

    pub fn tool_call(
        id: impl Into<ToolCallId>,
        name: impl Into<ToolName>,
        arguments: RawJson,
    ) -> Self {
        Self::new(NonEmpty::one(AssistantTurnItem::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        }))
    }

    pub fn assistant_text(&self) -> String {
        let mut text = String::new();
        for item in self.items() {
            if let AssistantTurnItem::Text(delta) = item {
                text.push_str(delta);
            }
        }
        text
    }

    pub fn into_input_items(
        self,
        tool_uses: impl IntoIterator<Item = ToolUse>,
    ) -> Result<Vec<ModelInputItem>, AssistantTurnInputError> {
        let mut tool_use_map = BTreeMap::new();
        for tool_use in tool_uses {
            let duplicate_id = tool_use.id.clone();
            if tool_use_map
                .insert(duplicate_id.clone(), tool_use)
                .is_some()
            {
                return Err(AssistantTurnInputError::DuplicateToolUse { id: duplicate_id });
            }
        }

        let mut input = Vec::new();
        for item in self.items.into_vec() {
            match item {
                AssistantTurnItem::Text(text) => {
                    input.push(ModelInputItem::Assistant(AssistantInputItem::Text(text)));
                }
                AssistantTurnItem::Reasoning(text) => {
                    input.push(ModelInputItem::Assistant(AssistantInputItem::Reasoning(
                        text,
                    )));
                }
                AssistantTurnItem::Refusal(text) => {
                    input.push(ModelInputItem::Assistant(AssistantInputItem::Refusal(text)));
                }
                AssistantTurnItem::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    let Some(tool_use) = tool_use_map.remove(&id) else {
                        return Err(AssistantTurnInputError::MissingToolUse { id });
                    };
                    if tool_use.name != name {
                        return Err(AssistantTurnInputError::MismatchedToolName {
                            id,
                            expected: name,
                            actual: tool_use.name,
                        });
                    }
                    if tool_use.arguments != arguments {
                        return Err(AssistantTurnInputError::MismatchedToolArguments {
                            id,
                            expected: arguments,
                            actual: tool_use.arguments,
                        });
                    }
                    input.push(ModelInputItem::ToolUse(tool_use));
                }
            }
        }

        if let Some((id, _)) = tool_use_map.into_iter().next() {
            return Err(AssistantTurnInputError::ExtraToolUse { id });
        }

        Ok(input)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Canonical assistant output items for a completed turn.
///
/// Tool calls exist only on the response side. Once a call has been paired with a tool result for
/// replay, it is represented as [`ModelInputItem::ToolUse`] instead.
pub enum AssistantTurnItem {
    Text(String),
    Reasoning(String),
    Refusal(String),
    ToolCall {
        id: ToolCallId,
        name: ToolName,
        arguments: RawJson,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NonEmpty<T>(Vec<T>);

impl<T> NonEmpty<T> {
    pub fn one(item: T) -> Self {
        Self(vec![item])
    }

    pub fn try_from_vec(items: Vec<T>) -> Result<Self, EmptyNonEmptyError> {
        if items.is_empty() {
            Err(EmptyNonEmptyError)
        } else {
            Ok(Self(items))
        }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.0.iter()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.0
    }
}

impl<T> TryFrom<Vec<T>> for NonEmpty<T> {
    type Error = EmptyNonEmptyError;

    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        Self::try_from_vec(value)
    }
}

impl<T> From<NonEmpty<T>> for Vec<T> {
    fn from(value: NonEmpty<T>) -> Self {
        value.0
    }
}

impl<T> Serialize for NonEmpty<T>
where
    T: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for NonEmpty<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let values = Vec::<T>::deserialize(deserializer)?;
        Self::try_from_vec(values).map_err(serde::de::Error::custom)
    }
}

impl<T> Borrow<[T]> for NonEmpty<T> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

#[derive(Debug, Error, Clone, Copy, Eq, PartialEq)]
#[error("non-empty collection must contain at least one element")]
pub struct EmptyNonEmptyError;

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum ModelInputValidationError {
    #[error("model input must contain at least one item")]
    Empty,
    #[error("duplicate tool use id `{id}` in model input")]
    DuplicateToolUseId { id: ToolCallId },
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum AssistantTurnInputError {
    #[error("assistant turn references missing tool use `{id}`")]
    MissingToolUse { id: ToolCallId },
    #[error("assistant turn received duplicate tool use `{id}`")]
    DuplicateToolUse { id: ToolCallId },
    #[error("assistant turn received extra tool use `{id}`")]
    ExtraToolUse { id: ToolCallId },
    #[error("assistant turn tool call `{id}` expected tool name `{expected}`, got `{actual}`")]
    MismatchedToolName {
        id: ToolCallId,
        expected: ToolName,
        actual: ToolName,
    },
    #[error("assistant turn tool call `{id}` received mismatched arguments")]
    MismatchedToolArguments {
        id: ToolCallId,
        expected: RawJson,
        actual: RawJson,
    },
}

#[derive(Serialize, Deserialize)]
#[serde(transparent)]
pub struct RawJson(Box<RawValue>);

impl RawJson {
    pub fn parse(json: impl Into<String>) -> Result<Self, serde_json::Error> {
        RawValue::from_string(json.into()).map(Self)
    }

    pub fn from_serializable<T>(value: &T) -> Result<Self, serde_json::Error>
    where
        T: Serialize,
    {
        RawValue::from_string(serde_json::to_string(value)?).map(Self)
    }

    pub fn get(&self) -> &str {
        self.0.get()
    }

    pub fn deserialize<T>(&self) -> Result<T, serde_json::Error>
    where
        T: DeserializeOwned,
    {
        serde_json::from_str(self.get())
    }
}

impl Clone for RawJson {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl fmt::Debug for RawJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RawJson").field(&self.get()).finish()
    }
}

impl PartialEq for RawJson {
    fn eq(&self, other: &Self) -> bool {
        self.get() == other.get()
    }
}

impl Eq for RawJson {}

impl PartialOrd for RawJson {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RawJson {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.get().cmp(other.get())
    }
}

impl std::hash::Hash for RawJson {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.get().hash(state);
    }
}

impl fmt::Display for RawJson {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.get())
    }
}

impl From<Box<RawValue>> for RawJson {
    fn from(value: Box<RawValue>) -> Self {
        Self(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolCallId(String);

impl ToolCallId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ToolCallId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ToolCallId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ToolCallId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ToolName(String);

impl ToolName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ToolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ToolName {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ToolName {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    use crate::toolset::ToolInput;

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct WeatherArgs {
        city: String,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct WeatherResult {
        forecast: String,
    }

    impl ToolInput for WeatherArgs {
        type Output = WeatherResult;

        const NAME: &'static str = "weather";
        const DESCRIPTION: &'static str = "Get weather";
    }

    #[test]
    fn raw_json_rejects_invalid_json() {
        assert!(RawJson::parse("{").is_err());
        assert_eq!(
            RawJson::parse("{\"ok\":true}").unwrap().get(),
            "{\"ok\":true}"
        );
    }

    #[test]
    fn non_empty_rejects_empty_vectors() {
        assert!(NonEmpty::<String>::try_from_vec(vec![]).is_err());
    }

    #[test]
    fn model_input_validation_rejects_duplicate_tool_use_ids() {
        let input = ModelInput::from_items(vec![
            ModelInputItem::text(InputMessageRole::User, "hello"),
            ModelInputItem::tool_use_parts(
                "call-1",
                "weather",
                RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                RawJson::parse("\"sunny\"").unwrap(),
            ),
            ModelInputItem::tool_use_parts(
                "call-1",
                "weather",
                RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                RawJson::parse("\"rainy\"").unwrap(),
            ),
        ]);

        assert_eq!(
            input.validate().unwrap_err(),
            ModelInputValidationError::DuplicateToolUseId {
                id: ToolCallId::from("call-1"),
            }
        );
    }

    #[test]
    fn tool_input_serializes_result() {
        let tool_use = WeatherArgs::tool_use(
            ToolMetadata::new(
                "call-1",
                "weather",
                RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
            ),
            WeatherResult {
                forecast: "sunny".into(),
            },
        )
        .unwrap();

        assert_eq!(tool_use.result.get(), "{\"forecast\":\"sunny\"}");
    }

    #[test]
    fn assistant_turn_into_input_items_preserves_order() {
        let turn = AssistantTurn::from_items(vec![
            AssistantTurnItem::Reasoning("think".into()),
            AssistantTurnItem::Text("before".into()),
            AssistantTurnItem::ToolCall {
                id: ToolCallId::from("call-1"),
                name: ToolName::from("weather"),
                arguments: RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
            },
            AssistantTurnItem::Text("after".into()),
        ])
        .unwrap();

        let items = turn
            .into_input_items(vec![ToolUse::new(
                "call-1",
                "weather",
                RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                RawJson::parse("\"sunny\"").unwrap(),
            )])
            .unwrap();

        assert!(matches!(
            items[0],
            ModelInputItem::Assistant(AssistantInputItem::Reasoning(_))
        ));
        assert!(matches!(
            items[1],
            ModelInputItem::Assistant(AssistantInputItem::Text(_))
        ));
        assert!(matches!(items[2], ModelInputItem::ToolUse(_)));
        assert!(matches!(
            items[3],
            ModelInputItem::Assistant(AssistantInputItem::Text(_))
        ));
    }

    #[test]
    fn assistant_turn_into_input_items_rejects_missing_tool_use() {
        let turn = AssistantTurn::tool_call(
            "call-1",
            "weather",
            RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
        );

        assert_eq!(
            turn.into_input_items(Vec::<ToolUse>::new()).unwrap_err(),
            AssistantTurnInputError::MissingToolUse {
                id: ToolCallId::from("call-1"),
            }
        );
    }

    #[test]
    fn assistant_turn_into_input_items_rejects_extra_tool_use() {
        let turn = AssistantTurn::text("done");

        assert_eq!(
            turn.into_input_items(vec![ToolUse::new(
                "call-1",
                "weather",
                RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                RawJson::parse("\"sunny\"").unwrap(),
            )])
            .unwrap_err(),
            AssistantTurnInputError::ExtraToolUse {
                id: ToolCallId::from("call-1"),
            }
        );
    }

    #[test]
    fn assistant_turn_into_input_items_rejects_duplicate_tool_use() {
        let turn = AssistantTurn::tool_call(
            "call-1",
            "weather",
            RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
        );

        assert_eq!(
            turn.into_input_items(vec![
                ToolUse::new(
                    "call-1",
                    "weather",
                    RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                    RawJson::parse("\"sunny\"").unwrap(),
                ),
                ToolUse::new(
                    "call-1",
                    "weather",
                    RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                    RawJson::parse("\"rainy\"").unwrap(),
                ),
            ])
            .unwrap_err(),
            AssistantTurnInputError::DuplicateToolUse {
                id: ToolCallId::from("call-1"),
            }
        );
    }

    #[test]
    fn assistant_turn_into_input_items_rejects_mismatched_tool_name() {
        let turn = AssistantTurn::tool_call(
            "call-1",
            "weather",
            RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
        );

        assert_eq!(
            turn.into_input_items(vec![ToolUse::new(
                "call-1",
                "forecast",
                RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                RawJson::parse("\"sunny\"").unwrap(),
            )])
            .unwrap_err(),
            AssistantTurnInputError::MismatchedToolName {
                id: ToolCallId::from("call-1"),
                expected: ToolName::from("weather"),
                actual: ToolName::from("forecast"),
            }
        );
    }

    #[test]
    fn assistant_turn_into_input_items_rejects_mismatched_tool_arguments() {
        let turn = AssistantTurn::tool_call(
            "call-1",
            "weather",
            RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
        );

        assert_eq!(
            turn.into_input_items(vec![ToolUse::new(
                "call-1",
                "weather",
                RawJson::parse("{\"city\":\"Osaka\"}").unwrap(),
                RawJson::parse("\"sunny\"").unwrap(),
            )])
            .unwrap_err(),
            AssistantTurnInputError::MismatchedToolArguments {
                id: ToolCallId::from("call-1"),
                expected: RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                actual: RawJson::parse("{\"city\":\"Osaka\"}").unwrap(),
            }
        );
    }
}
