use agents::{TextTurn, ToolPolicy};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[agents::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, agents::Toolset)]
enum AppTools {
    Weather(WeatherArgs),
}

#[agents::tool_input(name = "search", output = SearchResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchArgs {
    query: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchResult {
    answer: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, agents::Toolset)]
enum OtherTools {
    Search(SearchArgs),
}

fn main() {
    let _turn = {
        let mut turn =
            TextTurn::<AppTools>::new(agents::ModelName::new("gpt-4.1-mini").unwrap());
        turn.config.tools = ToolPolicy::allow_only(vec![
            AppToolsSelector::Weather,
            OtherToolsSelector::Search,
        ]);
        turn
    };
}
