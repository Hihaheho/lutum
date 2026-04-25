use std::{borrow::Borrow, fmt, ops::Deref};

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::DeserializeOwned};
use serde_json::value::RawValue;
use thiserror::Error;

use crate::transcript::CommittedTurn;

#[derive(Clone, Debug, Default)]
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

    pub fn items_mut(&mut self) -> &mut Vec<ModelInputItem> {
        &mut self.items
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

    pub fn tool_result(mut self, tool_result: ToolResult) -> Self {
        self.push(ModelInputItem::tool_result(tool_result));
        self
    }

    /// Remove ephemeral turns from the input.
    ///
    /// An item is an ephemeral turn if it is a [`ModelInputItem::Turn`] whose view reports
    /// `ephemeral() == true` (e.g. an `EphemeralTurnView`). Non-turn ephemerality is tracked
    /// by `Session` itself and stripped there; this method only handles the turn case.
    pub fn remove_ephemeral_turns(&mut self) {
        self.items.retain(|item| match item {
            ModelInputItem::Turn(turn) => !turn.ephemeral(),
            _ => true,
        });
    }

    pub fn validate(&self) -> Result<(), ModelInputValidationError> {
        if self.items.is_empty() {
            return Err(ModelInputValidationError::Empty);
        }

        let mut tool_results = std::collections::BTreeSet::new();
        for item in &self.items {
            if let ModelInputItem::ToolResult(tool_result) = item
                && !tool_results.insert(tool_result.id.clone())
            {
                return Err(ModelInputValidationError::DuplicateToolResultId {
                    id: tool_result.id.clone(),
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

/// Per-request metadata identifying [`ModelInputItem`] positions that are ephemeral.
///
/// `Session` attaches this to [`crate::RequestExtensions`] for session-originated turns so
/// adapters can place provider-specific cache breakpoints before volatile context without
/// changing the canonical request algebra.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EphemeralInputIndices {
    indices: Vec<usize>,
}

impl EphemeralInputIndices {
    pub fn new(indices: impl IntoIterator<Item = usize>) -> Self {
        let mut indices = indices.into_iter().collect::<Vec<_>>();
        indices.sort_unstable();
        indices.dedup();
        Self { indices }
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn contains(&self, index: usize) -> bool {
        self.indices.binary_search(&index).is_ok()
    }
}

#[derive(Clone, Debug)]
pub enum ModelInputItem {
    Message {
        role: InputMessageRole,
        content: NonEmpty<MessageContent>,
    },
    Assistant(AssistantInputItem),
    ToolResult(ToolResult),
    Turn(CommittedTurn),
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

    pub fn tool_result(tool_result: ToolResult) -> Self {
        Self::ToolResult(tool_result)
    }

    pub fn turn(committed_turn: CommittedTurn) -> Self {
        Self::Turn(committed_turn)
    }

    pub fn tool_result_parts(
        id: impl Into<ToolCallId>,
        name: impl Into<ToolName>,
        arguments: RawJson,
        result: RawJson,
    ) -> Self {
        Self::ToolResult(ToolResult::new(id, name, arguments, result))
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
/// as [`ToolResult`] at the surrounding [`ModelInputItem`] level so call/result pairs stay bundled.
pub enum AssistantInputItem {
    Text(String),
    Reasoning(String),
    Refusal(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub id: ToolCallId,
    pub name: ToolName,
    pub arguments: RawJson,
    pub result: RawJson,
}

pub const REJECTED_TOOL_RESULT_PREFIX: &str = "__lutum_rejected__: ";

impl ToolResult {
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

    pub fn rejection_reason(&self) -> Option<String> {
        self.result
            .deserialize::<String>()
            .ok()?
            .strip_prefix(REJECTED_TOOL_RESULT_PREFIX)
            .map(ToOwned::to_owned)
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

    pub fn into_tool_result(self, result: RawJson) -> ToolResult {
        ToolResult::new(self.id, self.name, self.arguments, result)
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
}

/// An assistant turn that has been produced by the LLM but not yet committed to a session or
/// model input.
///
/// This is the only type that carries a `commit_into()` method. Once committed, only the turn
/// content (`AssistantTurn`) is accessible from the resulting `TextTurnResult`.
///
/// Drop without committing or discarding triggers a `#[must_use]` lint.
#[derive(Debug)]
#[must_use = "call .commit_into(), or .discard() to explicitly opt out of committing"]
pub struct UncommittedAssistantTurn {
    inner: AssistantTurn,
    committed_turn: CommittedTurn,
}

impl UncommittedAssistantTurn {
    /// Construct from raw parts. Intended for adapter implementations and internal protocol use.
    pub fn new(inner: AssistantTurn, committed_turn: CommittedTurn) -> Self {
        Self {
            inner,
            committed_turn,
        }
    }

    /// Commit this turn into a `ModelInput`, appending it to the ordered input.
    pub fn commit_into(self, input: &mut ModelInput) {
        input.push(ModelInputItem::Turn(self.committed_turn));
    }

    /// Explicitly discard this turn without committing it.
    ///
    /// Use this to opt out of committing when you have intentionally decided not to record
    /// the turn in the transcript.
    pub fn discard(self) {}

    /// Access the assistant turn content.
    pub fn assistant_turn(&self) -> &AssistantTurn {
        &self.inner
    }

    /// Concatenate all `Text` items in this turn.
    pub fn assistant_text(&self) -> String {
        self.inner.assistant_text()
    }
}

impl Deref for UncommittedAssistantTurn {
    type Target = AssistantTurn;

    fn deref(&self) -> &AssistantTurn {
        &self.inner
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Canonical assistant output items for a completed turn.
///
/// Tool calls exist only on the response side. Once a call has been paired with a tool result for
/// replay, it is represented as [`ModelInputItem::ToolResult`] instead.
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
    #[error("duplicate tool result id `{id}` in model input")]
    DuplicateToolResultId { id: ToolCallId },
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum AssistantTurnInputError {
    #[error("assistant turn references missing tool result `{id}`")]
    MissingToolResult { id: ToolCallId },
    #[error("assistant turn received duplicate tool result `{id}`")]
    DuplicateToolResult { id: ToolCallId },
    #[error("assistant turn received extra tool result `{id}`")]
    ExtraToolResult { id: ToolCallId },
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
    fn model_input_validation_rejects_duplicate_tool_result_ids() {
        let input = ModelInput::from_items(vec![
            ModelInputItem::text(InputMessageRole::User, "hello"),
            ModelInputItem::tool_result_parts(
                "call-1",
                "weather",
                RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                RawJson::parse("\"sunny\"").unwrap(),
            ),
            ModelInputItem::tool_result_parts(
                "call-1",
                "weather",
                RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                RawJson::parse("\"rainy\"").unwrap(),
            ),
        ]);

        assert_eq!(
            input.validate().unwrap_err(),
            ModelInputValidationError::DuplicateToolResultId {
                id: ToolCallId::from("call-1"),
            }
        );
    }

    #[test]
    fn tool_input_serializes_result() {
        let tool_result = WeatherArgs::tool_result(
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

        assert_eq!(tool_result.result.get(), "{\"forecast\":\"sunny\"}");
    }
}
