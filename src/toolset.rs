use std::{fmt, marker::PhantomData};

use schemars::{JsonSchema, Schema, schema_for};
use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::conversation::NonEmpty;

#[derive(Clone, Copy)]
pub struct ToolDef<Call, Result> {
    pub name: &'static str,
    pub description: &'static str,
    schema: fn() -> Schema,
    _marker: PhantomData<fn() -> (Call, Result)>,
}

impl<Call, Result> fmt::Debug for ToolDef<Call, Result> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolDef")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

impl<Call, Result> ToolDef<Call, Result> {
    pub const fn new(
        name: &'static str,
        description: &'static str,
        schema: fn() -> Schema,
    ) -> Self {
        Self {
            name,
            description,
            schema,
            _marker: PhantomData,
        }
    }

    pub fn input_schema(&self) -> Schema {
        (self.schema)()
    }

    pub fn for_input<Input>(name: &'static str, description: &'static str) -> Self
    where
        Input: JsonSchema,
    {
        Self::new(name, description, || schema_for!(Input))
    }

    pub fn tool_ref(&self) -> ToolRef<Call, Result> {
        ToolRef::new(self.name)
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
    #[error("failed to serialize tool result: {0}")]
    Serialize(#[from] serde_json::Error),
}

pub trait Toolset: Send + Sync + 'static {
    type Call: Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync + 'static;
    type Result: Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync + 'static;

    fn definitions() -> &'static [ToolDef<Self::Call, Self::Result>];

    fn parse_call(name: &str, arguments_json: &str) -> Result<Self::Call, ToolCallError>;

    fn serialize_result(result: &Self::Result) -> Result<String, ToolResultError> {
        serde_json::to_string(result).map_err(ToolResultError::from)
    }
}

pub type ToolRefFor<T> = ToolRef<<T as Toolset>::Call, <T as Toolset>::Result>;

#[derive(Default)]
pub enum ToolMode<T: Toolset> {
    #[default]
    Disabled,
    AutoAll,
    AutoOnly(NonEmpty<ToolRefFor<T>>),
    RequiredAll,
    RequiredOnly(NonEmpty<ToolRefFor<T>>),
}

impl<T> Clone for ToolMode<T>
where
    T: Toolset,
{
    fn clone(&self) -> Self {
        match self {
            Self::Disabled => Self::Disabled,
            Self::AutoAll => Self::AutoAll,
            Self::AutoOnly(selected) => Self::AutoOnly(selected.clone()),
            Self::RequiredAll => Self::RequiredAll,
            Self::RequiredOnly(selected) => Self::RequiredOnly(selected.clone()),
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
            Self::AutoOnly(selected) => {
                let names = selected.iter().map(ToolRef::name).collect::<Vec<_>>();
                f.debug_tuple("AutoOnly").field(&names).finish()
            }
            Self::RequiredAll => f.write_str("RequiredAll"),
            Self::RequiredOnly(selected) => {
                let names = selected.iter().map(ToolRef::name).collect::<Vec<_>>();
                f.debug_tuple("RequiredOnly").field(&names).finish()
            }
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
            (Self::AutoOnly(left), Self::AutoOnly(right)) => names_equal(left, right),
            (Self::RequiredAll, Self::RequiredAll) => true,
            (Self::RequiredOnly(left), Self::RequiredOnly(right)) => names_equal(left, right),
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

    pub fn selected_refs(&self) -> Option<&[ToolRefFor<T>]> {
        match self {
            Self::AutoOnly(selected) | Self::RequiredOnly(selected) => Some(selected.as_slice()),
            _ => None,
        }
    }

    pub fn requires_tools(&self) -> bool {
        matches!(self, Self::RequiredAll | Self::RequiredOnly(_))
    }
}

pub struct ToolRef<Call, Result> {
    name: &'static str,
    _marker: PhantomData<fn() -> (Call, Result)>,
}

impl<Call, Result> Clone for ToolRef<Call, Result> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Call, Result> Copy for ToolRef<Call, Result> {}

impl<Call, Result> fmt::Debug for ToolRef<Call, Result> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ToolRef").field(&self.name).finish()
    }
}

impl<Call, Result> PartialEq for ToolRef<Call, Result> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl<Call, Result> Eq for ToolRef<Call, Result> {}

impl<Call, Result> PartialOrd for ToolRef<Call, Result> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<Call, Result> Ord for ToolRef<Call, Result> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.name.cmp(other.name)
    }
}

impl<Call, Result> std::hash::Hash for ToolRef<Call, Result> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl<Call, Result> ToolRef<Call, Result> {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            _marker: PhantomData,
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }
}

fn names_equal<Call, Result>(
    left: &NonEmpty<ToolRef<Call, Result>>,
    right: &NonEmpty<ToolRef<Call, Result>>,
) -> bool {
    left.as_slice()
        .iter()
        .map(ToolRef::name)
        .eq(right.as_slice().iter().map(ToolRef::name))
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, serde::Deserialize, JsonSchema)]
pub struct NoToolCall;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, serde::Deserialize, JsonSchema)]
pub struct NoToolResult;

#[derive(Clone, Copy, Debug, Default)]
pub struct NoTools;

impl Toolset for NoTools {
    type Call = NoToolCall;
    type Result = NoToolResult;

    fn definitions() -> &'static [ToolDef<Self::Call, Self::Result>] {
        &[]
    }

    fn parse_call(name: &str, _arguments_json: &str) -> Result<Self::Call, ToolCallError> {
        Err(ToolCallError::UnknownTool {
            name: name.to_string(),
        })
    }
}
