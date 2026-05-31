use std::{fmt, future::Future};

use schemars::{JsonSchema, Schema, schema_for};
use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::{
    budget::Usage,
    conversation::{
        AssistantTurn, AssistantTurnItem, REJECTED_TOOL_RESULT_PREFIX, ToolMetadata, ToolResult,
    },
    extensions::RequestExtensions,
    hooks::{HookFuture, HookObject, boxed_hook_future},
    llm::{AdapterToolDefinition, FinishReason},
};

#[derive(Clone, Copy)]
pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    schema: fn() -> Schema,
}

impl fmt::Debug for ToolDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolDef")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

impl ToolDef {
    pub const fn new(
        name: &'static str,
        description: &'static str,
        schema: fn() -> Schema,
    ) -> Self {
        Self {
            name,
            description,
            schema,
        }
    }

    pub fn input_schema(&self) -> Schema {
        (self.schema)()
    }

    pub fn for_input<Input>() -> Self
    where
        Input: ToolInput,
    {
        Self::new(Input::NAME, Input::DESCRIPTION, || schema_for!(Input))
    }
}

/// A tool whose schema is determined at runtime.
///
/// Dynamic tools are registered per turn via `with_dynamic_tools(...)` on a
/// toolset that declares a `#[dynamic]` variant. Lutum does not inspect or
/// validate the schema beyond forwarding it to the configured adapter.
#[derive(
    Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct DynamicTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

impl DynamicTool {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// A model-issued call to a dynamic tool.
///
/// User code dispatches by inspecting [`name`](Self::name) and
/// [`arguments`](Self::arguments), then returns a [`HandledDynamicTool`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DynamicToolCall {
    metadata: ToolMetadata,
}

impl DynamicToolCall {
    pub fn new(metadata: ToolMetadata) -> Self {
        Self { metadata }
    }

    pub fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    pub fn name(&self) -> &str {
        self.metadata.name.as_str()
    }

    pub fn arguments(&self) -> &crate::conversation::RawJson {
        &self.metadata.arguments
    }

    pub fn handled(self, output: crate::conversation::RawJson) -> HandledDynamicTool {
        HandledDynamicTool {
            metadata: self.metadata,
            output,
        }
    }
}

impl ToolCallWrapper for DynamicToolCall {
    fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }
}

/// A dynamic tool call paired with user-provided raw JSON output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HandledDynamicTool {
    metadata: ToolMetadata,
    output: crate::conversation::RawJson,
}

impl HandledDynamicTool {
    pub fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    pub fn output(&self) -> &crate::conversation::RawJson {
        &self.output
    }

    pub fn into_parts(self) -> (ToolMetadata, crate::conversation::RawJson) {
        (self.metadata, self.output)
    }
}

impl IntoToolResult for HandledDynamicTool {
    fn into_tool_result(self) -> Result<ToolResult, ToolResultError> {
        let (metadata, output) = self.into_parts();
        Ok(metadata.into_tool_result(output))
    }
}

#[derive(Debug, Error)]
pub enum ToolCallError {
    #[error("unknown tool `{name}`")]
    UnknownTool { name: String },
    #[error("failed to deserialize tool call for `{name}`: {source}")]
    Deserialize {
        name: String,
        #[source]
        source: serde_json::Error,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContinueSuggestionReason {
    RecoverableToolCallIssue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecoverableToolCallIssueReason {
    NotAvailable,
    UnknownTool,
    InvalidArguments,
}

impl RecoverableToolCallIssueReason {
    pub fn from_tool_call_error(error: ToolCallError) -> Self {
        match error {
            ToolCallError::UnknownTool { .. } => Self::UnknownTool,
            ToolCallError::Deserialize { .. } => Self::InvalidArguments,
        }
    }

    pub fn rejection_reason(&self, metadata: &ToolMetadata) -> String {
        match self {
            Self::NotAvailable => format!(
                "tool `{}` is not available in this round",
                metadata.name.as_str()
            ),
            Self::UnknownTool => format!(
                "tool `{}` is not recognized in this toolset or round",
                metadata.name.as_str()
            ),
            Self::InvalidArguments => format!(
                "tool `{}` arguments did not match the expected schema",
                metadata.name.as_str()
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecoverableToolCallIssue {
    pub metadata: ToolMetadata,
    pub reason: RecoverableToolCallIssueReason,
}

impl RecoverableToolCallIssue {
    pub fn new(metadata: ToolMetadata, reason: RecoverableToolCallIssueReason) -> Self {
        Self { metadata, reason }
    }

    pub fn not_available(metadata: ToolMetadata) -> Self {
        Self::new(metadata, RecoverableToolCallIssueReason::NotAvailable)
    }

    pub fn from_tool_call_error(metadata: ToolMetadata, error: ToolCallError) -> Self {
        Self {
            metadata,
            reason: RecoverableToolCallIssueReason::from_tool_call_error(error),
        }
    }

    pub fn rejection_reason(&self) -> String {
        self.reason.rejection_reason(&self.metadata)
    }
}

#[derive(Debug, Error)]
pub enum ToolResultError {
    #[error("tool metadata for `{actual}` does not match expected tool `{expected}`")]
    MismatchedToolName {
        expected: &'static str,
        actual: String,
    },
    #[error("failed to serialize tool output: {0}")]
    Serialize(#[from] serde_json::Error),
}

#[derive(Debug, Error)]
pub enum ToolExecutionError<E> {
    #[error("tool execution failed: {0}")]
    Execute(E),
    #[error("failed to build tool result: {0}")]
    ToolResult(#[from] ToolResultError),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HandledTool<I, O> {
    metadata: ToolMetadata,
    input: I,
    output: O,
}

impl<I, O> HandledTool<I, O> {
    pub fn new(metadata: ToolMetadata, input: I, output: O) -> Self {
        Self {
            metadata,
            input,
            output,
        }
    }

    pub fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    pub fn input(&self) -> &I {
        &self.input
    }

    pub fn output(&self) -> &O {
        &self.output
    }

    pub fn into_parts(self) -> (ToolMetadata, I, O) {
        (self.metadata, self.input, self.output)
    }
}

impl<I, O> HandledTool<I, O>
where
    I: ToolInput<Output = O>,
{
    pub fn into_tool_result(self) -> Result<ToolResult, ToolResultError> {
        let (metadata, _input, output) = self.into_parts();
        I::tool_result(metadata, output)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolDecision<I, O> {
    RunNormally(I),
    Complete(O),
    Reject(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RejectedToolSource {
    Hook,
    Policy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RejectedToolCall<C> {
    source: RejectedToolSource,
    metadata: ToolMetadata,
    call: Option<C>,
    reason: String,
}

impl<C> RejectedToolCall<C> {
    pub fn from_metadata(
        source: RejectedToolSource,
        metadata: ToolMetadata,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            source,
            metadata,
            call: None,
            reason: reason.into(),
        }
    }

    pub fn source(&self) -> RejectedToolSource {
        self.source
    }

    pub fn metadata(&self) -> &ToolMetadata {
        &self.metadata
    }

    pub fn call(&self) -> Option<&C> {
        self.call.as_ref()
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }

    pub fn into_parts(self) -> (RejectedToolSource, ToolMetadata, Option<C>, String) {
        (self.source, self.metadata, self.call, self.reason)
    }
}

impl<C> RejectedToolCall<C>
where
    C: ToolCallWrapper,
{
    pub fn from_call(source: RejectedToolSource, call: C, reason: impl Into<String>) -> Self {
        Self {
            source,
            metadata: call.metadata().clone(),
            call: Some(call),
            reason: reason.into(),
        }
    }
}

impl<C> RejectedToolCall<C> {
    pub fn map_call<D>(self, f: impl FnOnce(C) -> D) -> RejectedToolCall<D> {
        RejectedToolCall {
            source: self.source,
            metadata: self.metadata,
            call: self.call.map(f),
            reason: self.reason,
        }
    }
}

#[derive(Debug)]
pub enum ToolHookOutcome<C, H> {
    Handled(H),
    Unhandled(C),
    Rejected(RejectedToolCall<C>),
}

pub trait IntoToolResult {
    fn into_tool_result(self) -> Result<ToolResult, ToolResultError>;
}

impl IntoToolResult for ToolResult {
    fn into_tool_result(self) -> Result<ToolResult, ToolResultError> {
        Ok(self)
    }
}

impl<I, O> IntoToolResult for HandledTool<I, O>
where
    I: ToolInput<Output = O>,
{
    fn into_tool_result(self) -> Result<ToolResult, ToolResultError> {
        self.into_tool_result()
    }
}

impl<C> IntoToolResult for RejectedToolCall<C> {
    fn into_tool_result(self) -> Result<ToolResult, ToolResultError> {
        let (_, metadata, _, reason) = self.into_parts();
        let result = crate::conversation::RawJson::from_serializable(&format!(
            "{REJECTED_TOOL_RESULT_PREFIX}{reason}"
        ))?;
        Ok(metadata.into_tool_result(result))
    }
}

pub trait ToolInput:
    Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync + 'static
{
    type Output: Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync + 'static;

    const NAME: &'static str;
    const DESCRIPTION: &'static str;

    fn tool_result(
        metadata: ToolMetadata,
        output: Self::Output,
    ) -> Result<ToolResult, ToolResultError> {
        if metadata.name.as_str() != Self::NAME {
            return Err(ToolResultError::MismatchedToolName {
                expected: Self::NAME,
                actual: metadata.name.as_str().to_string(),
            });
        }
        let result = crate::conversation::RawJson::from_serializable(&output)?;
        Ok(metadata.into_tool_result(result))
    }
}

pub trait ToolCallWrapper {
    fn metadata(&self) -> &ToolMetadata;
}

impl ToolCallWrapper for std::convert::Infallible {
    fn metadata(&self) -> &ToolMetadata {
        match *self {}
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ToolCallFallbackError {
    #[error("fallback parser did not recover any tool calls")]
    NoToolCall,
    #[error("fallback parser returned an empty assistant turn")]
    EmptyAssistantTurn,
    #[error("fallback parser recovered unavailable tool `{name}`")]
    UnavailableTool { name: String },
    #[error("fallback parser recovered `{actual}` but required `{expected}`")]
    WrongRequiredTool { expected: String, actual: String },
    #[error(
        "fallback parser returned {tool_calls} typed calls but assistant turn contains {assistant_tool_calls} tool-call items"
    )]
    ToolCallCountMismatch {
        assistant_tool_calls: usize,
        tool_calls: usize,
    },
    #[error("fallback parser recovered inconsistent typed call metadata at index {index}")]
    MismatchedTypedCall { index: usize },
    #[error("fallback parser could not parse recovered tool call `{name}`: {message}")]
    ToolCallParse { name: String, message: String },
    #[error("fallback parser failed: {message}")]
    Parser { message: String },
}

#[derive(Clone, Debug)]
pub struct RecoveredTextToolCalls<T: Toolset> {
    pub assistant_turn: AssistantTurn,
    pub tool_calls: Vec<T::ToolCall>,
}

impl<T> RecoveredTextToolCalls<T>
where
    T: Toolset,
{
    pub fn new(assistant_turn: AssistantTurn, tool_calls: Vec<T::ToolCall>) -> Self {
        Self {
            assistant_turn,
            tool_calls,
        }
    }

    pub fn from_parts(
        items: Vec<AssistantTurnItem>,
        tool_calls: Vec<T::ToolCall>,
    ) -> Result<Self, ToolCallFallbackError> {
        let assistant_turn = AssistantTurn::from_items(items)
            .map_err(|_| ToolCallFallbackError::EmptyAssistantTurn)?;
        Ok(Self::new(assistant_turn, tool_calls))
    }

    pub fn from_items(items: Vec<AssistantTurnItem>) -> Result<Self, ToolCallFallbackError> {
        let assistant_turn = AssistantTurn::from_items(items)
            .map_err(|_| ToolCallFallbackError::EmptyAssistantTurn)?;
        Self::from_assistant_turn(assistant_turn)
    }

    pub fn from_assistant_turn(
        assistant_turn: AssistantTurn,
    ) -> Result<Self, ToolCallFallbackError> {
        let mut tool_calls = Vec::new();
        for item in assistant_turn.items() {
            let AssistantTurnItem::ToolCall {
                id,
                name,
                arguments,
            } = item
            else {
                continue;
            };
            let metadata = ToolMetadata::new(id.clone(), name.clone(), arguments.clone());
            let tool_name = name.as_str().to_string();
            let tool_call = T::parse_tool_call(metadata).map_err(|source| {
                ToolCallFallbackError::ToolCallParse {
                    name: tool_name,
                    message: source.to_string(),
                }
            })?;
            tool_calls.push(tool_call);
        }

        Ok(Self::new(assistant_turn, tool_calls))
    }
}

pub struct TextToolCallFallbackContext<'a, T: Toolset> {
    pub assistant_turn: &'a AssistantTurn,
    pub constraints: &'a ToolConstraints<T>,
    pub tool_definitions: &'a [AdapterToolDefinition],
    pub requirement: &'a ToolRequirement<T::Selector>,
    pub request_id: Option<&'a str>,
    pub model: &'a str,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub event_count: u32,
    pub extensions: &'a RequestExtensions,
}

impl<'a, T> TextToolCallFallbackContext<'a, T>
where
    T: Toolset,
{
    pub fn assistant_text(&self) -> String {
        self.assistant_turn.assistant_text()
    }
}

pub trait TextToolCallFallbackParser<T: Toolset>: HookObject {
    fn parse_fallback_tool_calls(
        &self,
        cx: &TextToolCallFallbackContext<'_, T>,
    ) -> Result<Option<RecoveredTextToolCalls<T>>, ToolCallFallbackError>;
}

impl<T, F> TextToolCallFallbackParser<T> for F
where
    T: Toolset,
    F: Fn(
            &TextToolCallFallbackContext<'_, T>,
        ) -> Result<Option<RecoveredTextToolCalls<T>>, ToolCallFallbackError>
        + HookObject,
{
    fn parse_fallback_tool_calls(
        &self,
        cx: &TextToolCallFallbackContext<'_, T>,
    ) -> Result<Option<RecoveredTextToolCalls<T>>, ToolCallFallbackError> {
        self(cx)
    }
}

pub trait ToolSelector<T: ?Sized>:
    Copy
    + Clone
    + fmt::Debug
    + Eq
    + PartialEq
    + std::hash::Hash
    + Serialize
    + DeserializeOwned
    + JsonSchema
    + Send
    + Sync
    + 'static
{
    fn name(self) -> &'static str;

    fn definition(self) -> &'static ToolDef;

    fn all() -> &'static [Self];

    fn try_from_name(name: &str) -> Option<Self>;
}

/// Extension of [`Toolset`] that supports batch hook application via [`ToolHooks`].
///
/// Implemented automatically by `#[derive(Toolset)]`.
pub trait HookableToolset: Toolset {
    type HandledCall: IntoToolResult + Clone + fmt::Debug + Send + Sync + 'static;
}

/// Abstraction over something that can intercept tool calls for a [`HookableToolset`].
///
/// Implemented automatically by `#[derive(Toolset)]` for the generated `ToolsHooks` struct.
/// A blanket impl covers `Fn(T::ToolCall) -> Fut` closures.
pub trait ToolHooks<T: HookableToolset>: HookObject {
    #[allow(clippy::type_complexity)]
    fn hook_call<'a>(
        &'a self,
        call: T::ToolCall,
    ) -> HookFuture<'a, ToolHookOutcome<T::ToolCall, T::HandledCall>>;
}

impl<T, F, Fut> ToolHooks<T> for F
where
    T: HookableToolset,
    F: Fn(T::ToolCall) -> Fut + HookObject,
    Fut: Future<Output = ToolHookOutcome<T::ToolCall, T::HandledCall>>
        + crate::hooks::MaybeSend
        + 'static,
{
    fn hook_call<'a>(
        &'a self,
        call: T::ToolCall,
    ) -> HookFuture<'a, ToolHookOutcome<T::ToolCall, T::HandledCall>> {
        boxed_hook_future(self(call))
    }
}

pub trait Toolset: Send + Sync + 'static {
    type ToolCall: ToolCallWrapper + Clone + fmt::Debug + PartialEq + Send + Sync + 'static;
    type Selector: ToolSelector<Self>;

    fn definitions() -> &'static [ToolDef];

    fn has_dynamic_slot() -> bool {
        false
    }

    fn definitions_for<I>(selectors: I) -> Vec<&'static ToolDef>
    where
        I: IntoIterator<Item = Self::Selector>,
    {
        selectors
            .into_iter()
            .map(|selector| selector.definition())
            .collect()
    }

    /// Selectors that are **on by default** — that is, all selectors except
    /// those marked with `#[tool(off)]` or `#[toolset(off)]` in the
    /// `#[derive(Toolset)]` macro. Used by [`ToolAvailability::Default`] and
    /// [`ToolAvailability::DefaultPlus`] to expand the effective toolset.
    ///
    /// The default implementation returns every selector reported by
    /// [`ToolSelector::all`], which matches the behaviour of toolsets that do
    /// not mark any variants as off-by-default.
    fn default_selectors() -> Vec<Self::Selector> {
        Self::Selector::all().to_vec()
    }

    /// Parse an assembled provider tool call into this toolset's typed call enum.
    ///
    /// For toolsets that declare a dynamic slot, the generated parser wraps
    /// fallback names as dynamic calls because dynamic registration is turn
    /// scoped and not available to this pure parser. Turn execution validates
    /// registered dynamic names before calling this method; direct callers
    /// should apply the same registry check when they need that distinction.
    fn parse_tool_call(metadata: ToolMetadata) -> Result<Self::ToolCall, ToolCallError>;
}

/// Marker trait for toolsets that declare a `#[dynamic]` variant.
pub trait HasDynamicSlot: Toolset {}

/// Policy describing which tools the model is allowed to call on a turn.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolAvailability<S> {
    /// Every tool in the toolset is available — including variants marked
    /// `#[tool(off)]` / `#[toolset(off)]`.
    All,
    /// Only tools that are on by default are available (see
    /// [`Toolset::default_selectors`]). Variants marked off-by-default are
    /// hidden unless explicitly re-enabled via [`ToolAvailability::DefaultPlus`]
    /// or [`ToolAvailability::Only`].
    Default,
    /// Only the listed selectors are available. This is an explicit whitelist
    /// that ignores default-on/off status.
    Only(Vec<S>),
    /// The union of [`ToolAvailability::Default`] and the listed selectors.
    /// Typical use: re-enable a specific skill's tools while keeping the rest
    /// of the default-on set intact.
    DefaultPlus(Vec<S>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolRequirement<S> {
    Optional,
    AtLeastOne,
    Specific(S),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolConstraints<T: Toolset> {
    pub available: ToolAvailability<T::Selector>,
    pub requirement: ToolRequirement<T::Selector>,
    /// Per-turn description overrides. When a selector appears here, its static
    /// `ToolDef::description` is replaced with this string before the request is
    /// sent to the adapter. Last entry wins when the same selector appears more
    /// than once.
    pub description_overrides: Vec<(T::Selector, String)>,
    /// Runtime tools appended to the request alongside typed tool definitions.
    pub dynamic_tools: Vec<DynamicTool>,
}

impl<T: Toolset> Default for ToolConstraints<T> {
    fn default() -> Self {
        Self {
            // Default-off variants (`#[tool(off)]` / `#[toolset(off)]`) stay
            // hidden unless the caller explicitly opts into `All`, `Only`, or
            // `DefaultPlus`. For toolsets with no off-by-default variants this
            // behaves identically to the old `All` default.
            available: ToolAvailability::Default,
            requirement: ToolRequirement::Optional,
            description_overrides: Vec::new(),
            dynamic_tools: Vec::new(),
        }
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Eq,
    PartialEq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
pub enum NoToolSelector {}

impl ToolSelector<NoTools> for NoToolSelector {
    fn name(self) -> &'static str {
        match self {}
    }

    fn definition(self) -> &'static ToolDef {
        match self {}
    }

    fn all() -> &'static [Self] {
        &[]
    }

    fn try_from_name(_name: &str) -> Option<Self> {
        None
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoTools;

impl Toolset for NoTools {
    type ToolCall = std::convert::Infallible;
    type Selector = NoToolSelector;

    fn definitions() -> &'static [ToolDef] {
        &[]
    }

    fn parse_tool_call(metadata: ToolMetadata) -> Result<Self::ToolCall, ToolCallError> {
        Err(ToolCallError::UnknownTool {
            name: metadata.name.as_str().to_string(),
        })
    }
}
