use std::{fmt, future::Future, pin::Pin};

use schemars::{JsonSchema, Schema, schema_for};
use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::conversation::{ToolMetadata, ToolResult};

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

pub enum ToolHookOutcome<C, H> {
    Handled(H),
    Unhandled(C),
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
pub trait ToolHooks<T: HookableToolset>: Send + Sync {
    #[allow(clippy::type_complexity)]
    fn hook_call<'a>(
        &'a self,
        call: T::ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolHookOutcome<T::ToolCall, T::HandledCall>> + Send + 'a>>;
}

impl<T, F, Fut> ToolHooks<T> for F
where
    T: HookableToolset,
    F: Fn(T::ToolCall) -> Fut + Send + Sync,
    Fut: Future<Output = ToolHookOutcome<T::ToolCall, T::HandledCall>> + Send + 'static,
{
    fn hook_call<'a>(
        &'a self,
        call: T::ToolCall,
    ) -> Pin<Box<dyn Future<Output = ToolHookOutcome<T::ToolCall, T::HandledCall>> + Send + 'a>>
    {
        Box::pin(self(call))
    }
}

pub trait Toolset: Send + Sync + 'static {
    type ToolCall: ToolCallWrapper + Clone + fmt::Debug + Eq + PartialEq + Send + Sync + 'static;
    type Selector: ToolSelector<Self>;

    fn definitions() -> &'static [ToolDef];

    fn definitions_for<I>(selectors: I) -> Vec<&'static ToolDef>
    where
        I: IntoIterator<Item = Self::Selector>,
    {
        selectors
            .into_iter()
            .map(|selector| selector.definition())
            .collect()
    }

    fn parse_tool_call(metadata: ToolMetadata) -> Result<Self::ToolCall, ToolCallError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolAvailability<S> {
    All,
    Only(Vec<S>),
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
}

impl<T: Toolset> Default for ToolConstraints<T> {
    fn default() -> Self {
        Self {
            available: ToolAvailability::All,
            requirement: ToolRequirement::Optional,
            description_overrides: Vec::new(),
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
