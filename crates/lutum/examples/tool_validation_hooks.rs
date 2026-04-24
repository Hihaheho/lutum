use std::sync::Arc;

use futures::executor::block_on;
use lutum::{
    FinishReason, MockLlmAdapter, MockTextScenario, ModelInputItem, RawTextTurnEvent, Session,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, TextStepOutcomeWithTools, ToolDecision,
    ToolMetadata, Usage,
};
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

#[lutum::impl_hook(WeatherHook)]
async fn reject_blank_city(
    _metadata: &ToolMetadata,
    input: WeatherArgs,
) -> ToolDecision<WeatherArgs, WeatherResult> {
    if input.city.trim().is_empty() {
        ToolDecision::Reject("city must not be blank".into())
    } else {
        ToolDecision::RunNormally(input)
    }
}

#[lutum::impl_hook(WeatherHook)]
async fn reject_atlantis(
    _metadata: &ToolMetadata,
    input: WeatherArgs,
) -> ToolDecision<WeatherArgs, WeatherResult> {
    if input.city.eq_ignore_ascii_case("Atlantis") {
        ToolDecision::Reject("fictional locations are blocked".into())
    } else {
        ToolDecision::RunNormally(input)
    }
}

fn execute_weather(input: &WeatherArgs) -> WeatherResult {
    WeatherResult {
        forecast: format!("live forecast for {}: 24C and sunny", input.city),
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("req-tool-validation-hooks".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "call-weather-1".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "call-weather-2".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Atlantis\"}".into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("req-tool-validation-hooks".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage::zero(),
        }),
    ]));

    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new(ctx);
    session.push_user("Check Tokyo and Atlantis weather.");

    let hooks = ToolsHooksSet::new()
        .with_weather_hook(RejectBlankCity)
        .with_weather_hook(RejectAtlantis);

    let round = match session
        .text_turn()
        .tools::<Tools>()
        .available_tools([ToolsSelector::Weather])
        .collect()
        .await?
    {
        TextStepOutcomeWithTools::NeedsTools(round) => round,
        TextStepOutcomeWithTools::Finished(result) => {
            println!(
                "assistant replied without tools: {}",
                result.assistant_text()
            );
            return Ok(());
        }
    };

    let plan = round.apply_hooks(&hooks).await;

    println!("pending calls: {}", plan.pending.len());
    println!("rejected calls: {}", plan.rejected.len());

    let pending_results: Vec<_> = plan
        .pending
        .iter()
        .map(|call| match call {
            ToolsCall::Weather(call) => {
                println!("executing weather({})", call.input().city);
                call.clone().complete(execute_weather(call.input()))
            }
        })
        .collect::<Result<_, _>>()?;

    plan.commit(&mut session, pending_results)?;

    println!("\nCommitted tool results:");
    for item in session.input().items() {
        if let ModelInputItem::ToolResult(result) = item {
            if let Some(reason) = result.rejection_reason() {
                println!("- {}: rejected ({reason})", result.name);
            } else {
                let weather: WeatherResult = result.result.deserialize()?;
                println!("- {}: {}", result.name, weather.forecast);
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    block_on(run())
}
