//! # call_budget — per-tool call limits with live description injection
//!
//! Shows two complementary features working together:
//!
//! 1. **Call budget** — each tool is given a maximum call count.  Before every
//!    round, tools whose budget is exhausted are removed from `available_tools`.
//!
//! 2. **Dynamic descriptions** — before each round, the remaining call count is
//!    read from the budget and injected into the tool description via
//!    `.describe_many_tools()`.  This way the model can reason about how many
//!    times it can still use a tool.
//!
//! The example runs against a `MockLlmAdapter` so it works without an API key.
//!
//! ```
//! cargo run -p lutum --example call_budget
//! ```

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use futures::executor::block_on;
use lutum::{
    FinishReason, MockLlmAdapter, MockTextScenario, RawTextTurnEvent, Session,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, TextStepOutcomeWithTools, Usage,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── tool definitions ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchResult {
    hits: Vec<String>,
}

/// Get current weather for a city.
#[lutum::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

/// Search the web for information.
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

// ── mock scenarios ────────────────────────────────────────────────────────────

fn scenario_round1() -> MockTextScenario {
    // model calls weather("Tokyo") and search("lunch near Tokyo Station")
    MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("req-1".into()),
            model: "mock".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "call-w1".into(),
            name: "weather".into(),
            arguments_json_delta: r#"{"city":"Tokyo"}"#.into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "call-s1".into(),
            name: "search".into(),
            arguments_json_delta: r#"{"query":"lunch near Tokyo Station"}"#.into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("req-1".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage::zero(),
        }),
    ])
}

fn scenario_round2() -> MockTextScenario {
    // search is now exhausted; model only calls weather("Osaka")
    MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("req-2".into()),
            model: "mock".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "call-w2".into(),
            name: "weather".into(),
            arguments_json_delta: r#"{"city":"Osaka"}"#.into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("req-2".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage::zero(),
        }),
    ])
}

fn scenario_round3() -> MockTextScenario {
    // weather budget also exhausted; model returns final text answer
    MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("req-3".into()),
            model: "mock".into(),
        }),
        Ok(RawTextTurnEvent::TextDelta {
            delta: "Tokyo is sunny (24 °C). Osaka is cloudy (19 °C). \
                    Lunch tip: try Tsukiji Sushi near Tokyo Station."
                .into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("req-3".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
        }),
    ])
}

// ── tool dispatch ─────────────────────────────────────────────────────────────

fn dispatch(call: ToolsCall) -> lutum::ToolResult {
    match call {
        ToolsCall::Weather(c) => {
            let city = &c.input().city;
            let forecast = match city.as_str() {
                "Tokyo" => "sunny and 24 °C".to_string(),
                "Osaka" => "cloudy and 19 °C".to_string(),
                other => format!("unknown city: {other}"),
            };
            println!("  [tool] weather({city}) → {forecast}");
            c.complete(WeatherResult { forecast }).unwrap()
        }
        ToolsCall::Search(c) => {
            let query = &c.input().query;
            let hits = vec![
                format!("Tsukiji Sushi near {query}"),
                format!("Ramen Spot near {query}"),
            ];
            println!("  [tool] search({query:?}) → {} hits", hits.len());
            c.complete(SearchResult { hits }).unwrap()
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(scenario_round1())
        .with_text_scenario(scenario_round2())
        .with_text_scenario(scenario_round3());

    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new(ctx);
    session.push_user(
        "Check Tokyo and Osaka weather, and search for lunch near Tokyo Station. \
         Summarise everything in one reply.",
    );

    // call budget: selector -> (used, limit)
    let budget: Arc<Mutex<HashMap<ToolsSelector, (usize, usize)>>> = Arc::new(Mutex::new(
        [
            (ToolsSelector::Weather, (0, 2)),
            (ToolsSelector::Search, (0, 1)),
        ]
        .into(),
    ));

    let mut round_number = 0;

    loop {
        round_number += 1;

        // tools with remaining budget > 0
        let available: Vec<ToolsSelector> = {
            let b = budget.lock().unwrap();
            let mut v: Vec<ToolsSelector> = b
                .iter()
                .filter(|&(_, &(used, limit))| used < limit)
                .map(|(&sel, _)| sel)
                .collect();
            v.sort_by_key(|s| s.name()); // deterministic ordering
            v
        };

        // build per-tool description overrides from current budget state
        let desc_overrides: Vec<(ToolsSelector, String)> = {
            let b = budget.lock().unwrap();
            available
                .iter()
                .map(|&sel| {
                    let (used, limit) = b[&sel];
                    let desc = format!(
                        "{} ({} calls remaining)",
                        sel.definition().description,
                        limit - used
                    );
                    (sel, desc)
                })
                .collect()
        };

        println!(
            "\n=== round {round_number} | available: [{}] ===",
            available
                .iter()
                .map(|s| s.name())
                .collect::<Vec<_>>()
                .join(", ")
        );
        for (sel, desc) in &desc_overrides {
            println!("  description override for {:?}: {desc}", sel.name());
        }

        let outcome = session
            .text_turn()
            .tools::<Tools>()
            .available_tools(available.iter().cloned())
            .describe_many_tools(desc_overrides)
            .collect()
            .await?;

        match outcome {
            TextStepOutcomeWithTools::Finished(result) => {
                println!("\nAnswer: {}", result.assistant_text().trim());
                break;
            }
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let mut tool_results = Vec::with_capacity(round.tool_calls.len());

                for call in round.tool_calls.iter().cloned() {
                    // record this call in the budget before dispatching
                    {
                        let mut b = budget.lock().unwrap();
                        if let Some((used, _)) = b.get_mut(&call.selector()) {
                            *used += 1;
                        }
                    }
                    tool_results.push(dispatch(call));
                }

                round.commit(&mut session, tool_results)?;
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    block_on(run())
}
