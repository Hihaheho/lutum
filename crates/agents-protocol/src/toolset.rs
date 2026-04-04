use std::fmt;

use schemars::{JsonSchema, Schema, schema_for};
use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::conversation::{ToolMetadata, ToolUse};

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
pub enum ToolUseError {
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
    #[error("failed to build tool use: {0}")]
    ToolUse(#[from] ToolUseError),
}

pub trait ToolInput:
    Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync + 'static
{
    type Output: Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync + 'static;

    const NAME: &'static str;
    const DESCRIPTION: &'static str;

    fn tool_use(metadata: ToolMetadata, output: Self::Output) -> Result<ToolUse, ToolUseError> {
        if metadata.name.as_str() != Self::NAME {
            return Err(ToolUseError::MismatchedToolName {
                expected: Self::NAME,
                actual: metadata.name.as_str().to_string(),
            });
        }
        let result = crate::conversation::RawJson::from_serializable(&output)?;
        Ok(metadata.into_tool_use(result))
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

#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub enum ToolPolicy<T: Toolset> {
    #[default]
    Disabled,
    AllowAll,
    AllowOnly(Vec<T::Selector>),
    RequireAll,
    RequireOnly(Vec<T::Selector>),
}

impl<T> ToolPolicy<T>
where
    T: Toolset,
{
    pub fn allow_only(selectors: impl IntoIterator<Item = T::Selector>) -> Self {
        let selectors = selectors.into_iter().collect::<Vec<_>>();
        if selectors.is_empty() {
            Self::Disabled
        } else {
            Self::AllowOnly(selectors)
        }
    }

    pub fn require_only(selectors: impl IntoIterator<Item = T::Selector>) -> Self {
        let selectors = selectors.into_iter().collect::<Vec<_>>();
        if selectors.is_empty() {
            Self::Disabled
        } else {
            Self::RequireOnly(selectors)
        }
    }

    pub fn uses_tools(&self) -> bool {
        !matches!(self, Self::Disabled)
    }

    pub fn requires_tools(&self) -> bool {
        matches!(self, Self::RequireAll | Self::RequireOnly(_))
    }

    pub fn selected(&self) -> Option<&[T::Selector]> {
        match self {
            Self::AllowOnly(selectors) | Self::RequireOnly(selectors) => Some(selectors.as_slice()),
            _ => None,
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
