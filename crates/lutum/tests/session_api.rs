use lutum::{
    AssistantTurnView, CommitTurn, EphemeralTurnView, FinishReason, MockLlmAdapter,
    MockStructuredScenario, MockTextScenario, Session, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, StructuredStepOutcomeWithTools, TextStepOutcomeWithTools, TurnView,
    Usage,
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

#[test]
fn structured_tool_parse_failure_recovers_as_tool_round() {
    let adapter =
        MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(lutum::RawStructuredTurnEvent::Started {
                request_id: Some("req-session-parse".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(lutum::RawStructuredTurnEvent::ToolCallChunk {
                id: "call-bad".into(),
                name: "weather".into(),
                arguments_json_delta: "{}".into(),
            }),
            Ok(lutum::RawStructuredTurnEvent::Completed {
                request_id: Some("req-session-parse".into()),
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
    session.push_user("Plan with a recoverable tool failure.");
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
            assert!(round.tool_calls.is_empty());
            assert_eq!(round.recoverable_tool_call_issues().len(), 1);
            assert_eq!(
                round.continue_suggestion(),
                Some(lutum::ContinueSuggestionReason::RecoverableToolCallIssue)
            );
            round
                .commit(&mut session, Vec::<lutum::ToolResult>::new())
                .unwrap();
        }
        StructuredStepOutcomeWithTools::Finished(_) => unreachable!(),
    }

    assert_eq!(session.input().items().len(), before_len + 2);
    assert_eq!(session.list_turns().count(), 1);
}

#[test]
fn ephemeral_turn_view_returns_ephemeral_true() {
    let inner = Arc::new(AssistantTurnView::from_items(&[])) as lutum::CommittedTurn;
    let ephemeral = EphemeralTurnView::new(inner.clone());
    assert!(
        ephemeral.ephemeral(),
        "EphemeralTurnView::ephemeral() should be true"
    );
    assert!(
        !inner.ephemeral(),
        "plain CommittedTurn::ephemeral() should be false"
    );
}

#[test]
fn push_ephemeral_turn_visible_in_input_but_not_in_list_turns() {
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let adapter = MockLlmAdapter::new();
    let ctx = lutum::Lutum::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);
    session.push_user("Hello.");

    let inner = Arc::new(AssistantTurnView::from_items(&[])) as lutum::CommittedTurn;
    let before_input_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    session.push_ephemeral_turn(inner);

    // Ephemeral turn IS visible in session.input() so callers can see what will be sent.
    assert_eq!(
        session.input().items().len(),
        before_input_len + 1,
        "push_ephemeral_turn should add to session.input()"
    );
    // But it is excluded from list_turns() because it is not a committed turn.
    assert_eq!(
        session.list_turns().count(),
        before_turns,
        "push_ephemeral_turn should not appear in list_turns()"
    );
}

#[test]
fn ephemeral_turn_is_cleared_after_collect() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-ephemeral-1".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "response".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-ephemeral-1".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 5,
                ..Usage::zero()
            },
        }),
    ]));
    let observed = adapter.clone();
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = lutum::Lutum::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);
    session.push_user("Hello.");

    let ephemeral_turn = Arc::new(AssistantTurnView::from_items(&[])) as lutum::CommittedTurn;
    session.push_ephemeral_turn(ephemeral_turn);

    assert_eq!(session.list_turns().count(), 0);

    // collect() — ephemeral turn goes into the snapshot sent to the model,
    // then is cleared from the session. Only the new committed turn remains.
    futures::executor::block_on(async { session.text_turn().collect().await }).unwrap();

    assert_eq!(observed.observed_ephemeral_indices(), vec![vec![1]]);
    assert_eq!(
        session.list_turns().count(),
        1,
        "only the newly committed assistant turn should be in the session"
    );
    // The ephemeral turn is gone — if it had been persisted we'd see 2 turns.
    assert_eq!(
        session.input().items().len(),
        2, // push_user + committed assistant turn
        "ephemeral turn must not be persisted in session.input()"
    );
}

#[test]
fn ephemeral_message_indices_are_attached_to_session_turn_request() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-ephemeral-message-1".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "response".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-ephemeral-message-1".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 5,
                ..Usage::zero()
            },
        }),
    ]));
    let observed = adapter.clone();
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = lutum::Lutum::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);

    session.push_user("Stable prompt.");
    session.push_ephemeral_user("Dynamic prompt.");

    futures::executor::block_on(async { session.text_turn().collect().await }).unwrap();

    assert_eq!(observed.observed_ephemeral_indices(), vec![vec![1]]);
    assert_eq!(
        session.input().items().len(),
        2,
        "ephemeral message must be stripped before commit"
    );
}
