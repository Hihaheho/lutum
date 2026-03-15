use agents::{
    FinishReason, Marker, MockLlmAdapter, MockTextScenario, Session, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, TextStepOutcome, TextTurn, ToolPolicy, Usage, UsageEstimate,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
struct AppMarker;

impl Marker for AppMarker {
    fn span_name(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("parallel_tools")
    }
}

#[agents::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
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
enum Tools {
    Weather(WeatherArgs),
    Search(SearchArgs),
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(agents::RawTextTurnEvent::Started {
            request_id: Some("req-parallel".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(agents::RawTextTurnEvent::ToolCallChunk {
            id: "call-weather".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
        }),
        Ok(agents::RawTextTurnEvent::ToolCallChunk {
            id: "call-search".into(),
            name: "search".into(),
            arguments_json_delta: "{\"query\":\"best ramen\"}".into(),
        }),
        Ok(agents::RawTextTurnEvent::Completed {
            request_id: Some("req-parallel".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage {
                total_tokens: 12,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = agents::Context::<AppMarker>::new(budget, adapter);
    let mut session = Session::new(ctx, AppMarker);
    session.push_user("Get the weather and search for ramen.");

    let outcome = session
        .prepare_text(
            {
                let mut turn = TextTurn::<Tools>::new(agents::ModelName::new("gpt-4.1-mini")?);
                turn.config.tools =
                    ToolPolicy::allow_only(vec![ToolsSelector::Weather, ToolsSelector::Search]);
                turn
            },
            UsageEstimate::zero(),
        )
        .await?
        .collect_noop()
        .await?;

    match outcome {
        TextStepOutcome::NeedsToolResults(round) => {
            let tool_uses = round
                .tool_calls
                .iter()
                .cloned()
                .map(|tool_call| match tool_call {
                    ToolsCall::Weather(call) => call
                        .tool_use(WeatherResult {
                            forecast: "sunny".into(),
                        })
                        .unwrap(),
                    ToolsCall::Search(call) => call
                        .tool_use(SearchResult {
                            answer: "Try a shop near the station.".into(),
                        })
                        .unwrap(),
                })
                .collect::<Vec<_>>();
            session.commit_tool_round(round, tool_uses)?;
        }
        TextStepOutcome::Finished(_) => unreachable!(),
    }

    println!("items={}", session.snapshot().items().len());
    Ok(())
}
