use std::convert::Infallible;

use futures::executor::block_on;
use lutum::{RawJson, ToolMetadata, ToolResult, Toolset};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

/// Get weather input.
/// Includes the requested city.
#[lutum::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum RawTools {
    CurrentWeather(WeatherArgs),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct AppCtx {
    prefix: String,
}

#[lutum::tool_fn(skip(app_ctx, tenant))]
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum FnTools {
    GetWeather(GetWeather),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
struct ScaleResult {
    scaled: f64,
}

/// Scale a floating-point value.
#[lutum::tool_input(name = "scale", output = ScaleResult)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
struct ScaleArgs {
    value: f64,
    factor: f64,
}

#[lutum::tool_fn]
/// Normalize a floating-point score.
async fn normalize_score(score: f64) -> Result<ScaleResult, Infallible> {
    Ok(ScaleResult {
        scaled: score / 100.0,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum FloatTools {
    Scale(ScaleArgs),
    NormalizeScore(NormalizeScore),
}

#[test]
fn tool_input_wrapper_builds_tool_result() {
    assert_eq!(RawToolsSelector::CurrentWeather.name(), "weather");
    assert_eq!(
        RawToolsSelector::CurrentWeather.definition().name,
        "weather"
    );
    assert_eq!(
        <WeatherArgs as lutum::ToolInput>::DESCRIPTION,
        "Get weather input.\nIncludes the requested city."
    );
    assert_eq!(
        RawToolsSelector::CurrentWeather.definition().description,
        "Get weather input.\nIncludes the requested city."
    );
    assert_eq!(
        RawToolsSelector::try_from_name("weather"),
        Some(RawToolsSelector::CurrentWeather)
    );
    assert_eq!(
        serde_json::to_string(&RawToolsSelector::CurrentWeather).unwrap(),
        "\"weather\""
    );
    assert_eq!(
        serde_json::from_str::<RawToolsSelector>("\"weather\"").unwrap(),
        RawToolsSelector::CurrentWeather
    );
    let schema = serde_json::to_value(schemars::schema_for!(RawToolsSelector)).unwrap();
    assert_eq!(schema["enum"][0], "weather");
    let selected_defs = RawTools::definitions_for([RawToolsSelector::CurrentWeather]);
    assert_eq!(selected_defs.len(), 1);
    assert_eq!(selected_defs[0].name, "weather");

    let tool_call = RawTools::parse_tool_call(ToolMetadata::new(
        "call-1",
        "weather",
        RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
    ))
    .unwrap();

    assert!(matches!(
        tool_call.selector(),
        RawToolsSelector::CurrentWeather
    ));
    let typed_tool = tool_call.clone().into_input();
    assert!(matches!(
        typed_tool,
        RawTools::CurrentWeather(WeatherArgs { ref city }) if city == "Tokyo"
    ));

    let tool_result = match tool_call {
        RawToolsCall::CurrentWeather(call) => {
            assert_eq!(call.input().city, "Tokyo");
            let input: WeatherArgs = call.clone().into();
            assert_eq!(input.city, "Tokyo");
            call.complete(WeatherResult {
                forecast: "sunny".into(),
            })
            .unwrap()
        }
    };

    assert_eq!(
        tool_result,
        ToolResult::new(
            "call-1",
            "weather",
            RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
            RawJson::parse("{\"forecast\":\"sunny\"}").unwrap(),
        )
    );
}

#[test]
fn tool_fn_wrapper_executes_with_skipped_args() {
    assert_eq!(
        <GetWeather as lutum::ToolInput>::DESCRIPTION,
        "Get weather with application context."
    );
    assert_eq!(
        FnToolsSelector::GetWeather.definition().description,
        "Get weather with application context."
    );

    let tool_call = FnTools::parse_tool_call(ToolMetadata::new(
        "call-2",
        "get_weather",
        RawJson::parse("{\"city\":\"Osaka\"}").unwrap(),
    ))
    .unwrap();

    let tool_result = match tool_call {
        FnToolsCall::GetWeather(call) => {
            assert_eq!(call.input().city, "Osaka");
            block_on(call.call(
                &AppCtx {
                    prefix: "wx".into(),
                },
                7,
            ))
            .unwrap()
        }
    };

    assert_eq!(tool_result.result.get(), "{\"forecast\":\"wx:7:Osaka\"}");
}

#[test]
fn float_tool_inputs_do_not_require_eq() {
    let tool_call = FloatTools::parse_tool_call(ToolMetadata::new(
        "call-3",
        "scale",
        RawJson::parse("{\"value\":2.5,\"factor\":4.0}").unwrap(),
    ))
    .unwrap();

    let tool_result = match tool_call {
        FloatToolsCall::Scale(call) => {
            assert_eq!(call.input().value, 2.5);
            assert_eq!(call.input().factor, 4.0);
            call.complete(ScaleResult { scaled: 10.0 }).unwrap()
        }
        FloatToolsCall::NormalizeScore(_) => panic!("expected scale tool call"),
    };

    assert_eq!(tool_result.result.get(), "{\"scaled\":10.0}");

    let tool_call = FloatTools::parse_tool_call(ToolMetadata::new(
        "call-4",
        "normalize_score",
        RawJson::parse("{\"score\":42.0}").unwrap(),
    ))
    .unwrap();

    let tool_result = match tool_call {
        FloatToolsCall::NormalizeScore(call) => {
            assert_eq!(call.input().score, 42.0);
            block_on(call.call()).unwrap()
        }
        FloatToolsCall::Scale(_) => panic!("expected normalize_score tool call"),
    };

    assert_eq!(tool_result.result.get(), "{\"scaled\":0.42}");
}
