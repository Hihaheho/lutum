use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchResult {
    answer: String,
}

/// Search indexed documents
#[lutum::tool_input(name = "search", output = SearchResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchInput {
    query: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum Tools {
    Search(SearchInput),
}

fn main() {
    let ctx = lutum::Lutum::new(
        std::sync::Arc::new(lutum::MockLlmAdapter::new()),
        lutum::SharedPoolBudgetManager::new(lutum::SharedPoolBudgetOptions::default()),
    );
    let _turn = ctx
        .text_turn(lutum::ModelInput::new().user("hello"))
        .tools::<Tools>()
        .with_dynamic_tools([lutum::DynamicTool::new(
            "weather",
            "Weather",
            serde_json::json!({"type": "object"}),
        )]);
}
