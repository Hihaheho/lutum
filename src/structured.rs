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
