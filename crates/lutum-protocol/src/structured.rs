use std::{any::type_name, borrow::Cow};

use schemars::{JsonSchema, Schema, schema_for};
use serde::{Serialize, de::DeserializeOwned};

pub trait StructuredOutput:
    Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync + 'static
{
    fn schema_name() -> Cow<'static, str> {
        Cow::Owned(type_name::<Self>().to_string())
    }

    fn json_schema() -> Schema {
        schema_for!(Self)
    }
}

impl<T> StructuredOutput for T where
    T: Serialize + DeserializeOwned + JsonSchema + Clone + Send + Sync + 'static
{
}

/// Model-authored output that could not be decoded as the requested structured type.
/// This is a completed output, not a transport or execution failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuredOutputIssue {
    pub raw: String,
    pub message: String,
}

impl StructuredOutputIssue {
    pub fn new(raw: impl Into<String>, error: impl std::fmt::Display) -> Self {
        Self {
            raw: raw.into(),
            message: error.to_string(),
        }
    }
}
