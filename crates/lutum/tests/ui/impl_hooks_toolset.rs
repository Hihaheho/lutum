use futures::executor::block_on;
use lutum::{RawJson, ToolDecision, ToolHookOutcome, ToolMetadata, Toolset};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

#[lutum::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum Tools {
    Weather(WeatherArgs),
}

struct Cache<'a> {
    prefix: &'a str,
}

#[lutum::impl_hooks(ToolsHooksSet)]
impl<'a> ToolsHooks for Cache<'a> {
    async fn weather_hook(
        &self,
        _metadata: &ToolMetadata,
        input: WeatherArgs,
    ) -> ToolDecision<WeatherArgs, WeatherResult> {
        ToolDecision::Complete(WeatherResult {
            forecast: format!("{}:{}", self.prefix, input.city),
        })
    }
}

fn main() {
    let prefix = String::from("cached");
    let cache = Cache { prefix: &prefix };
    let hooks = ToolsHooksSet::new().with_hooks(&cache);
    let call = Tools::parse_tool_call(ToolMetadata::new(
        "call-1",
        "weather",
        RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
    ))
    .unwrap();

    block_on(async {
        match call.hook(&hooks).await {
            ToolHookOutcome::Handled(ToolsHandled::Weather(handled)) => {
                assert_eq!(handled.output().forecast, "cached:Tokyo");
            }
            other => panic!("unexpected outcome: {other:?}"),
        }
    });
}
