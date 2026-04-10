use lutum::{
    CommitTurn, FinishReason, MockLlmAdapter, MockStructuredScenario, MockTextScenario, Session,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, StructuredStepOutcomeWithTools,
    TextStepOutcomeWithTools, Usage,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[lutum::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum Tools {
    Weather(WeatherArgs),
}

#[test]
fn collect_auto_commits_collect_staged_does_not() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-session-1".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "hello".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-session-1".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 4,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = lutum::Lutum::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);
    session.push_user("Hi.");
    let before_len = session.input().items().len();

    // collect() auto-commits immediately
    let result =
        futures::executor::block_on(async { session.text_turn().collect().await }).unwrap();

    assert_eq!(result.assistant_text(), "hello");
    assert_eq!(session.input().items().len(), before_len + 1);
    assert_eq!(session.list_turns().count(), 1);
}

#[test]
fn collect_staged_does_not_commit_until_explicit() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-session-staged".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "hello staged".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-session-staged".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 4,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = lutum::Lutum::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);
    session.push_user("Hi.");
    let before_len = session.input().items().len();

    let staged =
        futures::executor::block_on(async { session.text_turn().collect_staged().await }).unwrap();

    // Not committed yet
    assert_eq!(session.input().items().len(), before_len);
    assert_eq!(session.list_turns().count(), 0);

    // Commit explicitly via CommitTurn trait
    staged.turn.commit(&mut session);

    assert_eq!(session.input().items().len(), before_len + 1);
    assert_eq!(session.list_turns().count(), 1);
}

#[test]
fn tool_round_is_only_applied_on_explicit_commit() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-session-2".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::ToolCallChunk {
            id: "call-1".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-session-2".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage {
                total_tokens: 7,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = lutum::Lutum::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);
    session.push_user("Check weather.");
    let before_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    let outcome = futures::executor::block_on(async {
        session
            .text_turn()
            .tools::<Tools>()
            .available_tools(vec![ToolsSelector::Weather])
            .collect()
            .await
            .unwrap()
    });

    // NeedsTools does NOT auto-commit
    assert_eq!(session.input().items().len(), before_len);
    assert_eq!(session.list_turns().count(), before_turns);

    match outcome {
        TextStepOutcomeWithTools::NeedsTools(round) => {
            assert_eq!(round.tool_count(), 1);
            // Non-consuming expect_at_most_one and expect_one
            assert!(matches!(
                round.expect_at_most_one().unwrap(),
                Some(ToolsCall::Weather(_))
            ));
            assert!(matches!(round.expect_one().unwrap(), ToolsCall::Weather(_)));
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
                })
                .collect::<Vec<_>>();
            round.commit(&mut session, tool_results).unwrap();
        }
        TextStepOutcomeWithTools::Finished(_) => unreachable!(),
    }

    assert_eq!(session.input().items().len(), before_len + 2);
    assert_eq!(session.list_turns().count(), 1);
}

#[test]
fn session_auto_commits_across_multiple_turns() {
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(lutum::RawTextTurnEvent::Started {
                request_id: Some("req-step-1".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(lutum::RawTextTurnEvent::TextDelta {
                delta: "first step".into(),
            }),
            Ok(lutum::RawTextTurnEvent::Completed {
                request_id: Some("req-step-1".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 4,
                    ..Usage::zero()
                },
            }),
        ]))
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(lutum::RawTextTurnEvent::Started {
                request_id: Some("req-step-2".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(lutum::RawTextTurnEvent::TextDelta {
                delta: "second step".into(),
            }),
            Ok(lutum::RawTextTurnEvent::Completed {
                request_id: Some("req-step-2".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 4,
                    ..Usage::zero()
                },
            }),
        ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = lutum::Lutum::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);

    // collect() auto-commits each turn; no explicit commit needed
    for prompt in ["step one", "step two"] {
        session.push_user(prompt);
        futures::executor::block_on(async { session.text_turn().collect().await }).unwrap();
    }

    assert_eq!(session.input().items().len(), 4);
    let committed = session
        .list_turns()
        .map(|turn| {
            turn.item_at(0)
                .and_then(|item| item.as_text())
                .unwrap()
                .to_string()
        })
        .collect::<Vec<_>>();
    assert_eq!(committed, vec!["first step", "second step"]);
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Summary {
    answer: String,
}

#[test]
fn structured_tool_round_stays_explicit_until_commit() {
    let adapter =
        MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(lutum::RawStructuredTurnEvent::Started {
                request_id: Some("req-session-3".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(lutum::RawStructuredTurnEvent::ToolCallChunk {
                id: "call-1".into(),
                name: "weather".into(),
                arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
            }),
            Ok(lutum::RawStructuredTurnEvent::Completed {
                request_id: Some("req-session-3".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage {
                    total_tokens: 6,
                    ..Usage::zero()
                },
            }),
        ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = lutum::Lutum::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);
    session.push_user("Plan with a tool.");
    let before_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    let outcome = futures::executor::block_on(async {
        session
            .structured_turn::<Summary>()
            .tools::<Tools>()
            .available_tools(vec![ToolsSelector::Weather])
            .collect()
            .await
            .unwrap()
    });

    assert_eq!(session.input().items().len(), before_len);
    assert_eq!(session.list_turns().count(), before_turns);

    match outcome {
        StructuredStepOutcomeWithTools::NeedsTools(round) => {
            let tool_results = round
                .tool_calls
                .iter()
                .cloned()
                .map(|tool_call| match tool_call {
                    ToolsCall::Weather(call) => call
                        .complete(WeatherResult {
                            forecast: "windy".into(),
                        })
                        .unwrap(),
                })
                .collect::<Vec<_>>();
            round.commit(&mut session, tool_results).unwrap();
        }
        StructuredStepOutcomeWithTools::Finished(_) => unreachable!(),
    }

    assert_eq!(session.input().items().len(), before_len + 2);
    assert_eq!(session.list_turns().count(), 1);
}
