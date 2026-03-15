use std::convert::Infallible;

use agents::{RawJson, ToolMetadata, ToolUse, Toolset};
use futures::executor::block_on;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

#[agents::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, agents::Toolset)]
enum RawTools {
    Weather(WeatherArgs),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct AppCtx {
    prefix: String,
}

#[agents::tool_fn(skip(app_ctx, tenant))]
/// Get weather with application context.
async fn get_weather(
    app_ctx: &AppCtx,
    tenant: u64,
    city: String,
) -> Result<WeatherResult, Infallible> {
    Ok(WeatherResult {
        forecast: format!("{}:{tenant}:{city}", app_ctx.prefix),
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, agents::Toolset)]
enum FnTools {
    GetWeather(GetWeather),
}

#[test]
fn tool_input_wrapper_builds_tool_use() {
    let tool_call = RawTools::parse_tool_call(
        ToolMetadata::new(
            "call-1",
            "weather",
            RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
        ),
        "weather",
        "{\"city\":\"Tokyo\"}",
    )
    .unwrap();

    let tool_use = match tool_call {
        RawToolsCall::Weather(call) => call
            .tool_use(WeatherResult {
                forecast: "sunny".into(),
            })
            .unwrap(),
    };

    assert_eq!(
        tool_use,
        ToolUse::new(
            "call-1",
            "weather",
            RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
            RawJson::parse("{\"forecast\":\"sunny\"}").unwrap(),
        )
    );
}

#[test]
fn tool_fn_wrapper_executes_with_skipped_args() {
    let tool_call = FnTools::parse_tool_call(
        ToolMetadata::new(
            "call-2",
            "get_weather",
            RawJson::parse("{\"city\":\"Osaka\"}").unwrap(),
        ),
        "get_weather",
        "{\"city\":\"Osaka\"}",
    )
    .unwrap();

    let tool_use = match tool_call {
        FnToolsCall::GetWeather(call) => block_on(call.call(
            &AppCtx {
                prefix: "wx".into(),
            },
            7,
        ))
        .unwrap(),
    };

    assert_eq!(tool_use.result.get(), "{\"forecast\":\"wx:7:Osaka\"}");
}
