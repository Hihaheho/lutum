use lutum::{
    AssistantTurnItem, AssistantTurnView, CommitTurn, EphemeralTurnView, FinishReason,
    MockLlmAdapter, MockStructuredScenario, MockTextScenario, RawJson, RecoveredTextToolCalls,
    Session, SharedPoolBudgetManager, SharedPoolBudgetOptions, StructuredStepOutcomeWithTools,
    TextStepOutcomeWithTools, TextToolCollectError, TextToolErrorDirective,
    TextToolHandlerDirective, ToolCallFallbackError, TurnView, Usage,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, sync::Arc};

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

#[lutum::tool_input(name = "v_weather", output = ValidationWeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct ValidationWeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct ValidationWeatherResult {
    forecast: String,
}

#[lutum::tool_input(name = "v_search", output = SearchResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchArgs {
    query: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchResult {
    snippets: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum ValidationTools {
    ValidationWeather(ValidationWeatherArgs),
    Search(SearchArgs),
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
    let mut session = Session::new();
    session.push_user("Hi.");
    let before_len = session.input().items().len();

    // collect() auto-commits immediately
    let result =
        futures::executor::block_on(async { session.text_turn(&ctx).collect().await }).unwrap();

    assert_eq!(result.assistant_text(), "hello");
    assert_eq!(session.input().items().len(), before_len + 1);
    assert_eq!(session.list_turns().count(), 1);
}

#[test]
fn session_can_switch_lutum_instances_between_turns() {
    let adapter_a = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-session-a".into()),
            model: "model-a".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "from a".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-session-a".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 4,
                ..Usage::zero()
            },
        }),
    ]));
    let adapter_b = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-session-b".into()),
            model: "model-b".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "from b".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-session-b".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 4,
                ..Usage::zero()
            },
        }),
    ]));
    let llm_a = lutum::Lutum::new(
        Arc::new(adapter_a),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let llm_b = lutum::Lutum::new(
        Arc::new(adapter_b),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();

    session.push_user("first");
    let first =
        futures::executor::block_on(async { session.text_turn(&llm_a).collect().await }).unwrap();

    session.push_user("second");
    let second =
        futures::executor::block_on(async { session.text_turn(&llm_b).collect().await }).unwrap();

    assert_eq!(first.model, "model-a");
    assert_eq!(first.assistant_text(), "from a");
    assert_eq!(second.model, "model-b");
    assert_eq!(second.assistant_text(), "from b");
    assert_eq!(session.list_turns().count(), 2);

    let committed = session
        .list_turns()
        .map(|turn| {
            turn.item_at(0)
                .and_then(|item| item.as_text())
                .unwrap()
                .to_string()
        })
        .collect::<Vec<_>>();
    assert_eq!(committed, vec!["from a", "from b"]);
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
    let mut session = Session::new();
    session.push_user("Hi.");
    let before_len = session.input().items().len();

    let staged =
        futures::executor::block_on(async { session.text_turn(&ctx).collect_staged().await })
            .unwrap();

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
    let mut session = Session::new();
    session.push_user("Check weather.");
    let before_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    let outcome = futures::executor::block_on(async {
        session
            .text_turn(&ctx)
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
        TextStepOutcomeWithTools::FinishedNoOutput(_) => unreachable!(),
    }

    assert_eq!(session.input().items().len(), before_len + 2);
    assert_eq!(session.list_turns().count(), 1);
}

#[test]
fn tools_text_no_output_completion_does_not_commit_turn() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-no-output".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-no-output".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 3,
                ..Usage::zero()
            },
        }),
    ]));
    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();
    session.push_user("Maybe call a tool.");
    let before_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    let outcome = futures::executor::block_on(async {
        session
            .text_turn(&ctx)
            .tools::<Tools>()
            .collect()
            .await
            .unwrap()
    });

    match outcome {
        TextStepOutcomeWithTools::FinishedNoOutput(result) => {
            assert_eq!(result.request_id.as_deref(), Some("req-no-output"));
            assert_eq!(result.model, "gpt-4.1-mini");
            assert_eq!(result.finish_reason, FinishReason::Stop);
            assert_eq!(result.usage.total_tokens, 3);
            assert_eq!(result.cumulative_usage.total_tokens, 3);
        }
        TextStepOutcomeWithTools::Finished(_) => panic!("expected no-output completion"),
        TextStepOutcomeWithTools::NeedsTools(_) => panic!("expected no-output completion"),
    }

    assert_eq!(session.input().items().len(), before_len);
    assert_eq!(session.list_turns().count(), before_turns);
}

#[test]
fn required_tool_text_only_completion_errors_without_commit() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-required-text".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: r#"{"tool":"weather","arguments":{"city":"Tokyo"}}"#.into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-required-text".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 5,
                ..Usage::zero()
            },
        }),
    ]));
    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();
    session.push_user("Call a tool.");
    let before_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    let err = futures::executor::block_on(async {
        session
            .text_turn(&ctx)
            .tools::<Tools>()
            .require_any_tool()
            .collect()
            .await
    })
    .unwrap_err();

    match err {
        lutum::CollectError::Reduction {
            source:
                lutum::TextTurnReductionError::UnmetToolRequirement {
                    requirement,
                    request_id,
                    ..
                },
            ..
        } => {
            assert_eq!(request_id.as_deref(), Some("req-required-text"));
            assert_eq!(requirement, "at_least_one");
        }
        other => panic!("expected unmet required-tool reduction error, got {other:?}"),
    }
    assert_eq!(session.input().items().len(), before_len);
    assert_eq!(session.list_turns().count(), before_turns);
}

#[test]
fn optional_tool_text_only_completion_still_auto_commits() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-optional-text".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "plain answer".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-optional-text".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 5,
                ..Usage::zero()
            },
        }),
    ]));
    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();
    session.push_user("Tool optional.");
    let before_len = session.input().items().len();

    let outcome = futures::executor::block_on(async {
        session.text_turn(&ctx).tools::<Tools>().collect().await
    })
    .unwrap();

    match outcome {
        TextStepOutcomeWithTools::Finished(result) => {
            assert_eq!(result.assistant_text(), "plain answer");
        }
        TextStepOutcomeWithTools::NeedsTools(_) => panic!("expected finished text"),
        TextStepOutcomeWithTools::FinishedNoOutput(_) => panic!("expected finished text"),
    }
    assert_eq!(session.input().items().len(), before_len + 1);
    assert_eq!(session.list_turns().count(), 1);
}

#[test]
fn tool_text_collect_staged_does_not_auto_commit_finished_turn() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-tools-staged-text".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "stage me".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-tools-staged-text".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 5,
                ..Usage::zero()
            },
        }),
    ]));
    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();
    session.push_user("Tool optional.");
    let before_len = session.input().items().len();

    let staged = futures::executor::block_on(async {
        session
            .text_turn(&ctx)
            .tools::<Tools>()
            .collect_staged()
            .await
    })
    .unwrap();

    match staged {
        lutum::StagedTextStepOutcomeWithTools::Finished(result) => {
            assert_eq!(result.assistant_text(), "stage me");
            assert_eq!(session.input().items().len(), before_len);
            result.turn.commit(&mut session);
        }
        lutum::StagedTextStepOutcomeWithTools::NeedsTools(_) => {
            panic!("expected staged finished turn")
        }
        lutum::StagedTextStepOutcomeWithTools::FinishedNoOutput(_) => {
            panic!("expected staged finished turn")
        }
    }

    assert_eq!(session.input().items().len(), before_len + 1);
    assert_eq!(session.list_turns().count(), 1);
}

#[test]
fn required_specific_tool_text_fallback_recovers_needs_tools_round() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-recover-tool".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "I will check.\n```json\n{\"tool\":\"weather\"}\n```".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-recover-tool".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 8,
                ..Usage::zero()
            },
        }),
    ]));
    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();
    session.push_user("Check weather.");
    let before_len = session.input().items().len();

    let outcome = futures::executor::block_on(async {
        session
            .text_turn(&ctx)
            .tools::<Tools>()
            .available_tools(vec![ToolsSelector::Weather])
            .require_tool(ToolsSelector::Weather)
            .recover_tool_calls_with(|_cx: &lutum::TextToolCallFallbackContext<'_, Tools>| {
                RecoveredTextToolCalls::<Tools>::from_items(vec![
                    AssistantTurnItem::Text("I will check.\n".into()),
                    AssistantTurnItem::ToolCall {
                        id: "fallback-call-1".into(),
                        name: "weather".into(),
                        arguments: RawJson::parse(r#"{"city":"Tokyo"}"#).unwrap(),
                    },
                ])
                .map(Some)
            })
            .collect()
            .await
    })
    .unwrap();

    assert_eq!(session.input().items().len(), before_len);
    let TextStepOutcomeWithTools::NeedsTools(round) = outcome else {
        panic!("expected recovered tool round");
    };
    assert_eq!(round.tool_count(), 1);
    let ToolsCall::Weather(call) = round.expect_one().unwrap();
    assert_eq!(call.input.city, "Tokyo");
    let result = call
        .clone()
        .complete(WeatherResult {
            forecast: "sunny".into(),
        })
        .unwrap();
    round.commit(&mut session, vec![result]).unwrap();

    assert_eq!(session.input().items().len(), before_len + 2);
    assert_eq!(session.list_turns().count(), 1);
    let turn = session.list_turns().next().unwrap();
    assert_eq!(turn.item_count(), 2);
    assert_eq!(turn.item_at(0).unwrap().as_text(), Some("I will check.\n"));
    let tool_call = turn.item_at(1).unwrap().as_tool_call().unwrap();
    assert_eq!(tool_call.id.as_str(), "fallback-call-1");
    assert_eq!(tool_call.name.as_str(), "weather");
    assert_eq!(tool_call.arguments.get(), r#"{"city":"Tokyo"}"#);
}

#[test]
fn required_tool_text_fallback_none_errors_without_commit() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-recover-none".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "not a call".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-recover-none".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 4,
                ..Usage::zero()
            },
        }),
    ]));
    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();
    session.push_user("Call a tool.");
    let before_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    let err = futures::executor::block_on(async {
        session
            .text_turn(&ctx)
            .tools::<Tools>()
            .require_any_tool()
            .recover_tool_calls_with(|_cx: &lutum::TextToolCallFallbackContext<'_, Tools>| {
                Ok::<_, ToolCallFallbackError>(None)
            })
            .collect()
            .await
    })
    .unwrap_err();

    match err {
        lutum::CollectError::Reduction {
            source:
                lutum::TextTurnReductionError::ToolCallFallback {
                    source: ToolCallFallbackError::NoToolCall,
                    request_id,
                    ..
                },
            ..
        } => assert_eq!(request_id.as_deref(), Some("req-recover-none")),
        other => panic!("expected fallback no-call reduction error, got {other:?}"),
    }
    assert_eq!(session.input().items().len(), before_len);
    assert_eq!(session.list_turns().count(), before_turns);
}

#[test]
fn controlled_handler_can_return_tool_round_from_text_delta() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-controlled-early".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "I will check.\n```json\n{\"tool\":\"weather\"}\n```".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-controlled-early".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 9,
                ..Usage::zero()
            },
        }),
    ]));
    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();
    session.push_user("Check weather.");
    let before_len = session.input().items().len();

    let outcome = futures::executor::block_on(async {
        session
            .text_turn(&ctx)
            .tools::<Tools>()
            .available_tools([ToolsSelector::Weather])
            .require_any_tool()
            .collect_controlled_with(
                |event: &lutum::TextTurnEventWithTools<Tools>,
                 cx: &lutum::TextToolHandlerContext<'_, Tools>|
                 -> Result<_, Infallible> {
                    if let lutum::TextTurnEventWithTools::TextDelta { delta } = event
                        && delta.contains("```json")
                    {
                        let recovered = cx
                            .recover_tool_calls_from_items(vec![
                                AssistantTurnItem::Text("I will check.\n".into()),
                                AssistantTurnItem::ToolCall {
                                    id: "controlled-call-1".into(),
                                    name: "weather".into(),
                                    arguments: RawJson::parse(r#"{"city":"Tokyo"}"#).unwrap(),
                                },
                            ])
                            .unwrap();
                        return Ok(TextToolHandlerDirective::Return(
                            lutum::SyntheticTextToolTurn::needs_tools(recovered),
                        ));
                    }
                    Ok(TextToolHandlerDirective::Continue)
                },
            )
            .await
    })
    .unwrap();

    assert_eq!(session.input().items().len(), before_len);
    let TextStepOutcomeWithTools::NeedsTools(round) = outcome else {
        panic!("expected controlled tool round");
    };
    assert_eq!(round.usage, Usage::zero());
    let ToolsCall::Weather(call) = round.expect_one().unwrap();
    assert_eq!(call.input.city, "Tokyo");
}

#[test]
fn controlled_synthetic_finished_respects_required_tool_constraint() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-controlled-required-bypass".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "done".into(),
        }),
    ]));
    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();
    session.push_user("Check weather.");
    let before_len = session.input().items().len();

    let err = futures::executor::block_on(async {
        session
            .text_turn(&ctx)
            .tools::<Tools>()
            .require_any_tool()
            .collect_controlled_with(
                |event: &lutum::TextTurnEventWithTools<Tools>,
                 _cx: &lutum::TextToolHandlerContext<'_, Tools>|
                 -> Result<_, Infallible> {
                    if matches!(event, lutum::TextTurnEventWithTools::TextDelta { .. }) {
                        let assistant_turn =
                            lutum::AssistantTurn::from_items(vec![AssistantTurnItem::Text(
                                "done".into(),
                            )])
                            .unwrap();
                        return Ok(TextToolHandlerDirective::Return(
                            lutum::SyntheticTextToolTurn::finished(assistant_turn),
                        ));
                    }
                    Ok(TextToolHandlerDirective::Continue)
                },
            )
            .await
    })
    .unwrap_err();

    assert_eq!(session.input().items().len(), before_len);
    assert!(matches!(
        err,
        lutum::CollectError::Reduction {
            source: lutum::TextTurnReductionError::UnmetToolRequirement { .. },
            ..
        }
    ));
}

struct LengthRecoveryHandler {
    recovered: Option<RecoveredTextToolCalls<Tools>>,
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
impl lutum::TextToolEventHandler<Tools> for LengthRecoveryHandler {
    type Error = ToolCallFallbackError;

    async fn on_event(
        &mut self,
        event: &lutum::TextTurnEventWithTools<Tools>,
        cx: &lutum::TextToolHandlerContext<Tools>,
    ) -> Result<TextToolHandlerDirective<Tools>, Self::Error> {
        if let lutum::TextTurnEventWithTools::TextDelta { delta } = event
            && delta.contains("```json")
        {
            self.recovered = Some(cx.recover_tool_calls_from_items(vec![
                AssistantTurnItem::Text("I will check.\n".into()),
                AssistantTurnItem::ToolCall {
                    id: "controlled-length-call-1".into(),
                    name: "weather".into(),
                    arguments: RawJson::parse(r#"{"city":"Tokyo"}"#).unwrap(),
                },
            ])?);
        }
        Ok(TextToolHandlerDirective::Continue)
    }

    async fn on_error(
        &mut self,
        error: TextToolCollectError<'_>,
        _cx: &lutum::TextToolHandlerContext<Tools>,
    ) -> Result<TextToolErrorDirective<Tools>, Self::Error> {
        if matches!(
            error,
            TextToolCollectError::Reduction(lutum::TextTurnReductionError::OutputLimitExceeded(_))
        ) {
            let recovered = self
                .recovered
                .take()
                .ok_or(ToolCallFallbackError::NoToolCall)?;
            return Ok(TextToolErrorDirective::Return(
                lutum::SyntheticTextToolTurn::needs_tools(recovered),
            ));
        }
        Ok(TextToolErrorDirective::Propagate)
    }
}

#[test]
fn controlled_handler_can_recover_output_limit_as_tool_round() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-controlled-length".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "I will check.\n```json\n{\"tool\":\"weather\"}\n```".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-controlled-length".into()),
            finish_reason: FinishReason::Length,
            usage: Usage {
                total_tokens: 12,
                ..Usage::zero()
            },
        }),
    ]));
    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();
    session.push_user("Check weather.");

    let outcome = futures::executor::block_on(async {
        session
            .text_turn(&ctx)
            .tools::<Tools>()
            .available_tools([ToolsSelector::Weather])
            .require_any_tool()
            .collect_controlled_with(LengthRecoveryHandler { recovered: None })
            .await
    })
    .unwrap();

    let TextStepOutcomeWithTools::NeedsTools(round) = outcome else {
        panic!("expected recovered tool round");
    };
    assert_eq!(round.usage.total_tokens, 12);
    let ToolsCall::Weather(call) = round.expect_one().unwrap();
    assert_eq!(call.input.city, "Tokyo");
}

#[test]
fn controlled_handler_propagates_output_limit_by_default() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-controlled-length-propagate".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "partial".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-controlled-length-propagate".into()),
            finish_reason: FinishReason::Length,
            usage: Usage {
                total_tokens: 7,
                ..Usage::zero()
            },
        }),
    ]));
    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();
    session.push_user("Check weather.");

    let err = futures::executor::block_on(async {
        session
            .text_turn(&ctx)
            .tools::<Tools>()
            .collect_controlled_with(
                |_event: &lutum::TextTurnEventWithTools<Tools>,
                 _cx: &lutum::TextToolHandlerContext<'_, Tools>|
                 -> Result<_, Infallible> {
                    Ok(TextToolHandlerDirective::Continue)
                },
            )
            .await
    })
    .unwrap_err();

    assert!(matches!(
        err,
        lutum::CollectError::Reduction {
            source: lutum::TextTurnReductionError::OutputLimitExceeded(_),
            ..
        }
    ));
}

fn recovered_validation_search_call<T>(
    cx: &lutum::TextToolHandlerContext<'_, T>,
) -> Result<TextToolHandlerDirective<T>, ToolCallFallbackError>
where
    T: lutum::Toolset,
{
    let recovered = cx.recover_tool_calls_from_items(vec![
        AssistantTurnItem::Text("searching\n".into()),
        AssistantTurnItem::ToolCall {
            id: "validation-call-1".into(),
            name: "v_search".into(),
            arguments: RawJson::parse(r#"{"query":"rust"}"#).unwrap(),
        },
    ])?;
    Ok(TextToolHandlerDirective::Return(
        lutum::SyntheticTextToolTurn::needs_tools(recovered),
    ))
}

#[test]
fn controlled_context_rejects_unavailable_recovered_tool() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-controlled-unavailable".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "search".into(),
        }),
    ]));
    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();
    session.push_user("Search.");

    let err = futures::executor::block_on(async {
        session
            .text_turn(&ctx)
            .tools::<ValidationTools>()
            .available_tools([ValidationToolsSelector::ValidationWeather])
            .collect_controlled_with(
                |event: &lutum::TextTurnEventWithTools<ValidationTools>,
                 cx: &lutum::TextToolHandlerContext<'_, ValidationTools>| {
                    if matches!(event, lutum::TextTurnEventWithTools::TextDelta { .. }) {
                        return recovered_validation_search_call(cx);
                    }
                    Ok(TextToolHandlerDirective::Continue)
                },
            )
            .await
    })
    .unwrap_err();

    assert!(matches!(
        err,
        lutum::CollectError::Handler {
            source: ToolCallFallbackError::UnavailableTool { ref name },
            ..
        } if name == "v_search"
    ));
}

#[test]
fn controlled_context_rejects_wrong_required_recovered_tool() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-controlled-wrong-required".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::TextDelta {
            delta: "search".into(),
        }),
    ]));
    let ctx = lutum::Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new();
    session.push_user("Search.");

    let err = futures::executor::block_on(async {
        session
            .text_turn(&ctx)
            .tools::<ValidationTools>()
            .available_tools([
                ValidationToolsSelector::ValidationWeather,
                ValidationToolsSelector::Search,
            ])
            .require_tool(ValidationToolsSelector::ValidationWeather)
            .collect_controlled_with(
                |event: &lutum::TextTurnEventWithTools<ValidationTools>,
                 cx: &lutum::TextToolHandlerContext<'_, ValidationTools>| {
                    if matches!(event, lutum::TextTurnEventWithTools::TextDelta { .. }) {
                        return recovered_validation_search_call(cx);
                    }
                    Ok(TextToolHandlerDirective::Continue)
                },
            )
            .await
    })
    .unwrap_err();

    assert!(matches!(
        err,
        lutum::CollectError::Handler {
            source: ToolCallFallbackError::WrongRequiredTool {
                ref expected,
                ref actual
            },
            ..
        } if expected == "v_weather" && actual == "v_search"
    ));
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
    let mut session = Session::new();

    // collect() auto-commits each turn; no explicit commit needed
    for prompt in ["step one", "step two"] {
        session.push_user(prompt);
        futures::executor::block_on(async { session.text_turn(&ctx).collect().await }).unwrap();
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
    let mut session = Session::new();
    session.push_user("Plan with a tool.");
    let before_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    let outcome = futures::executor::block_on(async {
        session
            .structured_turn::<Summary>(&ctx)
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
    let mut session = Session::new();
    session.push_user("Plan with a recoverable tool failure.");
    let before_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    let outcome = futures::executor::block_on(async {
        session
            .structured_turn::<Summary>(&ctx)
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
    let mut session = Session::new();
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
    let mut session = Session::new();
    session.push_user("Hello.");

    let ephemeral_turn = Arc::new(AssistantTurnView::from_items(&[])) as lutum::CommittedTurn;
    session.push_ephemeral_turn(ephemeral_turn);

    assert_eq!(session.list_turns().count(), 0);

    // collect() — ephemeral turn goes into the snapshot sent to the model,
    // then is cleared from the session. Only the new committed turn remains.
    futures::executor::block_on(async { session.text_turn(&ctx).collect().await }).unwrap();

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
    let mut session = Session::new();

    session.push_user("Stable prompt.");
    session.push_ephemeral_user("Dynamic prompt.");

    futures::executor::block_on(async { session.text_turn(&ctx).collect().await }).unwrap();

    assert_eq!(observed.observed_ephemeral_indices(), vec![vec![1]]);
    assert_eq!(
        session.input().items().len(),
        2,
        "ephemeral message must be stripped before commit"
    );
}
