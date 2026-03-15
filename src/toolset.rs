use std::{fmt, marker::PhantomData};

use schemars::{JsonSchema, Schema, schema_for};
use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::conversation::{NonEmpty, ToolMetadata, ToolUse};

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

pub trait Toolset: Send + Sync + 'static {
    type ToolCall: ToolCallWrapper + Clone + fmt::Debug + Eq + PartialEq + Send + Sync + 'static;

    fn definitions() -> &'static [ToolDef];

    fn parse_tool_call(
        metadata: ToolMetadata,
        name: &str,
        arguments_json: &str,
    ) -> Result<Self::ToolCall, ToolCallError>;
}

pub trait SupportsTool<I: ToolInput>: Toolset {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ToolSubsetMarker<S> {
    _marker: PhantomData<fn() -> S>,
}

impl<S> Default for ToolSubsetMarker<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> ToolSubsetMarker<S> {
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

pub trait ToolSubset<T: Toolset> {
    fn tool_names() -> Vec<&'static str>;
}

impl<T> ToolSubset<T> for ToolSubsetMarker<()>
where
    T: Toolset,
{
    fn tool_names() -> Vec<&'static str> {
        Vec::new()
    }
}

macro_rules! impl_tool_subset_tuple {
    ($(($($name:ident),+)),+ $(,)?) => {
        $(
            impl<T, $($name),+> ToolSubset<T> for ToolSubsetMarker<($($name,)+)>
            where
                T: Toolset $(+ SupportsTool<$name>)+,
                $($name: ToolInput,)+
            {
                fn tool_names() -> Vec<&'static str> {
                    vec![$($name::NAME),+]
                }
            }
        )+
    };
}

impl_tool_subset_tuple!(
    (A),
    (A, B),
    (A, B, C),
    (A, B, C, D),
    (A, B, C, D, E),
    (A, B, C, D, E, F),
    (A, B, C, D, E, F, G),
    (A, B, C, D, E, F, G, H),
    (A, B, C, D, E, F, G, H, I),
    (A, B, C, D, E, F, G, H, I, J),
    (A, B, C, D, E, F, G, H, I, J, K),
    (A, B, C, D, E, F, G, H, I, J, K, L)
);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ToolRef<I: ToolInput> {
    _marker: PhantomData<fn() -> I>,
}

impl<I: ToolInput> Default for ToolRef<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: ToolInput> ToolRef<I> {
    pub const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    pub fn name(self) -> &'static str {
        I::NAME
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct ToolSelection<T: Toolset> {
    names: NonEmpty<&'static str>,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Clone for ToolSelection<T>
where
    T: Toolset,
{
    fn clone(&self) -> Self {
        Self {
            names: self.names.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T> ToolSelection<T>
where
    T: Toolset,
{
    pub fn from_names(names: NonEmpty<&'static str>) -> Self {
        Self {
            names,
            _marker: PhantomData,
        }
    }

    pub fn from_subset<S>() -> Option<Self>
    where
        S: ToolSubset<T>,
    {
        NonEmpty::try_from_vec(S::tool_names())
            .ok()
            .map(Self::from_names)
    }

    pub fn names(&self) -> &[&'static str] {
        self.names.as_slice()
    }
}

pub enum ToolMode<T: Toolset> {
    Disabled,
    AutoAll,
    AutoOnly(ToolSelection<T>),
    RequiredAll,
    RequiredOnly(ToolSelection<T>),
}

impl<T> Default for ToolMode<T>
where
    T: Toolset,
{
    fn default() -> Self {
        Self::Disabled
    }
}

impl<T> Clone for ToolMode<T>
where
    T: Toolset,
{
    fn clone(&self) -> Self {
        match self {
            Self::Disabled => Self::Disabled,
            Self::AutoAll => Self::AutoAll,
            Self::AutoOnly(selection) => Self::AutoOnly(selection.clone()),
            Self::RequiredAll => Self::RequiredAll,
            Self::RequiredOnly(selection) => Self::RequiredOnly(selection.clone()),
        }
    }
}

impl<T> fmt::Debug for ToolMode<T>
where
    T: Toolset,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disabled => f.write_str("Disabled"),
            Self::AutoAll => f.write_str("AutoAll"),
            Self::AutoOnly(selection) => {
                f.debug_tuple("AutoOnly").field(&selection.names()).finish()
            }
            Self::RequiredAll => f.write_str("RequiredAll"),
            Self::RequiredOnly(selection) => f
                .debug_tuple("RequiredOnly")
                .field(&selection.names())
                .finish(),
        }
    }
}

impl<T> PartialEq for ToolMode<T>
where
    T: Toolset,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Disabled, Self::Disabled) => true,
            (Self::AutoAll, Self::AutoAll) => true,
            (Self::AutoOnly(left), Self::AutoOnly(right)) => left.names() == right.names(),
            (Self::RequiredAll, Self::RequiredAll) => true,
            (Self::RequiredOnly(left), Self::RequiredOnly(right)) => left.names() == right.names(),
            _ => false,
        }
    }
}

impl<T> Eq for ToolMode<T> where T: Toolset {}

impl<T> ToolMode<T>
where
    T: Toolset,
{
    pub fn uses_tools(&self) -> bool {
        !matches!(self, Self::Disabled)
    }

    pub fn selected_names(&self) -> Option<&[&'static str]> {
        match self {
            Self::AutoOnly(selection) | Self::RequiredOnly(selection) => Some(selection.names()),
            _ => None,
        }
    }

    pub fn requires_tools(&self) -> bool {
        matches!(self, Self::RequiredAll | Self::RequiredOnly(_))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoTools;

impl Toolset for NoTools {
    type ToolCall = std::convert::Infallible;

    fn definitions() -> &'static [ToolDef] {
        &[]
    }

    fn parse_tool_call(
        _metadata: ToolMetadata,
        name: &str,
        _arguments_json: &str,
    ) -> Result<Self::ToolCall, ToolCallError> {
        Err(ToolCallError::UnknownTool {
            name: name.to_string(),
        })
    }
}
