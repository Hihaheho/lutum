use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum Tools {
    #[dynamic]
    DynamicA(lutum::DynamicTool),
    #[dynamic]
    DynamicB(lutum::DynamicTool),
}

fn main() {}
