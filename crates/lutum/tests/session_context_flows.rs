use futures::executor::block_on;
use lutum::{
    FinishReason, InputMessageRole, Lutum, MockLlmAdapter, MockStructuredScenario,
    MockTextScenario, ModelInput, ModelInputItem, Session, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, TextStepOutcomeWithTools, Usage,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Contact {
    email: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

#[lutum::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchResult {
    answer: String,
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

#[test]
fn direct_context_text_turn_collects_without_session_helpers() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-direct".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "direct control".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-direct".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 6,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Lutum::new(Arc::new(adapter), budget);
    let input = ModelInput::from_items(vec![ModelInputItem::text(
        InputMessageRole::User,
        "Run without session helpers.",
    )]);

    let result = block_on(ctx.text_turn(input).collect()).unwrap();

    assert_eq!(result.request_id.as_deref(), Some("req-direct"));
    assert_eq!(result.finish_reason, FinishReason::Stop);
    assert_eq!(result.assistant_text(), "direct control");
}

#[test]
fn structured_session_turn_auto_commits_on_collect() {
    let adapter =
        MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(lutum::RawStructuredTurnEvent::Started {
                request_id: Some("req-structured".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(lutum::RawStructuredTurnEvent::StructuredOutputChunk {
                json_delta: "{\"email\":\"user@example.com\"}".into(),
            }),
            Ok(lutum::RawStructuredTurnEvent::Completed {
                request_id: Some("req-structured".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 8,
                    ..Usage::zero()
                },
            }),
        ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = lutum::Lutum::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);
    session.push_user("Extract the email address.");
    let before_len = session.input().items().len();

    let result = block_on(async { session.structured_turn::<Contact>().collect().await }).unwrap();

    // collect() auto-commits
    assert_eq!(session.input().items().len(), before_len + 1);
    assert_eq!(session.list_turns().count(), 1);

    assert!(matches!(
        result.semantic,
        lutum::StructuredTurnOutcome::Structured(Contact { ref email })
            if email == "user@example.com"
    ));
}

#[test]
fn session_commits_parallel_tool_results_in_order() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-parallel".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::ToolCallChunk {
            id: "call-weather".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
        }),
        Ok(lutum::RawTextTurnEvent::ToolCallChunk {
            id: "call-search".into(),
            name: "search".into(),
            arguments_json_delta: "{\"query\":\"best ramen\"}".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-parallel".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage {
                total_tokens: 12,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = lutum::Lutum::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);
    session.push_user("Get the weather and search for ramen.");
    let before_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    let outcome = block_on(async {
        session
            .text_turn()
            .tools::<Tools>()
            .allow_only(vec![ToolsSelector::Weather, ToolsSelector::Search])
            .collect()
            .await
            .unwrap()
    });

    assert_eq!(session.input().items().len(), before_len);
    assert_eq!(session.list_turns().count(), before_turns);

    match outcome {
        TextStepOutcomeWithTools::NeedsTools(round) => {
            assert_eq!(round.tool_count(), 2);
            assert_eq!(
                round.expect_at_most_one().unwrap_err(),
                lutum::ToolRoundArityError::ExpectedAtMostOne { actual: 2 }
            );
            assert_eq!(round.tool_calls.len(), 2);
            assert!(matches!(
                &round.tool_calls[0],
                ToolsCall::Weather(call) if call.input.city.as_str() == "Tokyo"
            ));
            assert!(matches!(
                &round.tool_calls[1],
                ToolsCall::Search(call) if call.input.query.as_str() == "best ramen"
            ));

            let tool_results = round
                .tool_calls
                .iter()
                .cloned()
                .map(|tool_call| match tool_call {
                    ToolsCall::Weather(call) => call
                        .complete(WeatherResult {
                            forecast: "sunny".into(),
                        })
                        .unwrap(),
                    ToolsCall::Search(call) => call
                        .complete(SearchResult {
                            answer: "Try a shop near the station.".into(),
                        })
                        .unwrap(),
                })
                .collect::<Vec<_>>();
            round.commit(&mut session, tool_results).unwrap();
        }
        TextStepOutcomeWithTools::Finished(_) => unreachable!(),
    }

    assert_eq!(session.input().items().len(), before_len + 3);
    assert_eq!(session.list_turns().count(), 1);
}
