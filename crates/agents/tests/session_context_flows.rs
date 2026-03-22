use agents::{
    Context, FinishReason, InputMessageRole, MockLlmAdapter, MockStructuredScenario,
    MockTextScenario, ModelInput, ModelInputItem, NoTools, RequestExtensions, Session,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, StructuredStepOutcome, StructuredTurn,
    TextStepOutcome, TextTurn, ToolPolicy, Usage, UsageEstimate,
};
use futures::executor::block_on;
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

#[agents::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchResult {
    answer: String,
}

#[agents::tool_input(name = "search", output = SearchResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchArgs {
    query: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, agents::Toolset)]
enum Tools {
    Weather(WeatherArgs),
    Search(SearchArgs),
}

#[test]
fn direct_context_text_turn_collects_without_session_helpers() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(agents::RawTextTurnEvent::Started {
            request_id: Some("req-direct".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(agents::RawTextTurnEvent::TextDelta {
            delta: "direct control".into(),
        }),
        Ok(agents::RawTextTurnEvent::Completed {
            request_id: Some("req-direct".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 6,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Context::new(Arc::new(adapter), budget);
    let input = ModelInput::from_items(vec![ModelInputItem::text(
        InputMessageRole::User,
        "Run without session helpers.",
    )]);

    let pending = block_on(ctx.text_turn(
        RequestExtensions::new(),
        input,
        TextTurn::<NoTools>::new(agents::ModelName::new("gpt-4.1-mini").unwrap()),
        UsageEstimate::zero(),
    ))
    .unwrap();
    let result = block_on(pending.collect_noop()).unwrap();

    assert_eq!(result.request_id.as_deref(), Some("req-direct"));
    assert_eq!(result.finish_reason, FinishReason::Stop);
    assert_eq!(result.assistant_text(), "direct control");
}

#[test]
fn structured_session_turn_is_only_applied_after_commit() {
    let adapter =
        MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(agents::RawStructuredTurnEvent::Started {
                request_id: Some("req-structured".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(agents::RawStructuredTurnEvent::StructuredOutputChunk {
                json_delta: "{\"email\":\"user@example.com\"}".into(),
            }),
            Ok(agents::RawStructuredTurnEvent::Completed {
                request_id: Some("req-structured".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 8,
                    ..Usage::zero()
                },
            }),
        ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = agents::Context::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);
    session.push_user("Extract the email address.");
    let before_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    let outcome = block_on(async {
        session
            .prepare_structured(
                RequestExtensions::new(),
                StructuredTurn::<NoTools, Contact>::new(
                    agents::ModelName::new("gpt-4.1-mini").unwrap(),
                ),
                UsageEstimate::zero(),
            )
            .await
            .unwrap()
            .collect_noop()
            .await
            .unwrap()
    });

    assert_eq!(session.input().items().len(), before_len);
    assert_eq!(session.list_turns().count(), before_turns);

    match outcome {
        StructuredStepOutcome::Finished(result) => {
            assert!(matches!(
                result.semantic,
                agents::StructuredTurnOutcome::Structured(Contact { ref email })
                    if email == "user@example.com"
            ));
            session.commit_structured(result);
        }
        StructuredStepOutcome::NeedsToolResults(_) => unreachable!(),
    }

    assert_eq!(session.input().items().len(), before_len + 1);
    assert_eq!(session.list_turns().count(), 1);
}

#[test]
fn session_commits_parallel_tool_results_in_order() {
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
    let ctx = agents::Context::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);
    session.push_user("Get the weather and search for ramen.");
    let before_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    let outcome = block_on(async {
        session
            .prepare_text(
                RequestExtensions::new(),
                {
                    let mut turn =
                        TextTurn::<Tools>::new(agents::ModelName::new("gpt-4.1-mini").unwrap());
                    turn.config.tools =
                        ToolPolicy::allow_only(vec![ToolsSelector::Weather, ToolsSelector::Search]);
                    turn
                },
                UsageEstimate::zero(),
            )
            .await
            .unwrap()
            .collect_noop()
            .await
            .unwrap()
    });

    assert_eq!(session.input().items().len(), before_len);
    assert_eq!(session.list_turns().count(), before_turns);

    match outcome {
        TextStepOutcome::NeedsToolResults(round) => {
            assert_eq!(round.tool_calls.len(), 2);
            assert!(matches!(
                &round.tool_calls[0],
                ToolsCall::Weather(call) if call.input.city.as_str() == "Tokyo"
            ));
            assert!(matches!(
                &round.tool_calls[1],
                ToolsCall::Search(call) if call.input.query.as_str() == "best ramen"
            ));

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
            session.commit_tool_round(round, tool_uses).unwrap();
        }
        TextStepOutcome::Finished(_) => unreachable!(),
    }

    assert_eq!(session.input().items().len(), before_len + 3);
    assert_eq!(session.list_turns().count(), 1);
}
