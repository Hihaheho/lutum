use futures::executor::block_on;
use lutum::{RawJson, ToolDecision, ToolHookOutcome, ToolMetadata, Toolset};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchResult {
    hits: Vec<String>,
}

#[lutum::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[lutum::tool_input(name = "search", output = SearchResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchArgs {
    query: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum Tools {
    Weather(WeatherArgs),
    Search(SearchArgs),
}

#[lutum::impl_hook(WeatherHook)]
async fn hooked_forecast(
    _metadata: &lutum::ToolMetadata,
    input: WeatherArgs,
) -> ToolDecision<WeatherArgs, WeatherResult> {
    let city = input.city;
    ToolDecision::Complete(WeatherResult {
        forecast: format!("hooked:{city}"),
    })
}

fn main() {
    let call = Tools::parse_tool_call(ToolMetadata::new(
        "call-1",
        "weather",
        RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
    ))
    .unwrap();
    let hooks = ToolsHooksSet::new().with_weather_hook(HookedForecast);

    block_on(async move {
        match call.hook(&hooks).await {
            ToolHookOutcome::Handled(ToolsHandled::Weather(handled)) => {
                let _ = handled.metadata();
                let _ = handled.input();
                let _ = handled.output();
            }
            ToolHookOutcome::Handled(ToolsHandled::Search(_)) => unreachable!(),
            ToolHookOutcome::Unhandled(_) => unreachable!(),
            ToolHookOutcome::Rejected(_) => unreachable!(),
        }
    });
}
