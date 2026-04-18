use async_trait::async_trait;
use futures::executor::block_on;

use lutum::{
    AssistantTurnItem, AssistantTurnView, BudgetManager, CollectError, EventHandler, FinishReason,
    HandlerContext, HandlerDirective, InputMessageRole, Lutum, MockError, MockLlmAdapter,
    MockStructuredScenario, MockTextScenario, ModelInput, ModelInputItem, OperationKind,
    RequestBudget, RequestExtensions, SharedPoolBudgetManager, SharedPoolBudgetOptions,
    StructuredTurnOutcome, TextTurnEventWithTools, TextTurnReducerWithTools,
    TextTurnStateWithTools, ToolMetadata, Usage, UsageEstimate,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Summary {
    answer: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum Tools {
    Weather(WeatherArgs),
}

fn input() -> ModelInput {
    ModelInput::from_items(vec![ModelInputItem::text(InputMessageRole::User, "hello")])
}

fn weather_turn<'a>(turn: lutum::TextTurn<'a>) -> lutum::TextTurnWithTools<'a, Tools> {
    turn.tools::<Tools>()
        .available_tools(vec![ToolsSelector::Weather])
}

fn shared_pool_budget_error(err: &lutum::AgentError) -> &lutum::SharedPoolBudgetError {
    match err {
        lutum::AgentError::Budget(source) => source
            .downcast_ref::<lutum::SharedPoolBudgetError>()
            .expect("shared pool budget error source"),
        other => panic!("expected budget error, got {other}"),
    }
}

fn test_budget() -> SharedPoolBudgetManager {
    SharedPoolBudgetManager::new(SharedPoolBudgetOptions {
        capacity_tokens: 100,
        capacity_cost_micros_usd: 1_000,
        stop_threshold_tokens: 0,
        stop_threshold_cost_micros_usd: 0,
    })
}

struct StopOnTextDelta;

#[async_trait]
impl EventHandler<TextTurnEventWithTools<Tools>, TextTurnStateWithTools<Tools>>
    for StopOnTextDelta
{
    type Error = std::convert::Infallible;

    async fn on_event(
        &mut self,
        event: &TextTurnEventWithTools<Tools>,
        _cx: &HandlerContext<TextTurnStateWithTools<Tools>>,
    ) -> Result<HandlerDirective, Self::Error> {
        Ok(
            if matches!(event, TextTurnEventWithTools::TextDelta { .. }) {
                HandlerDirective::Stop
            } else {
                HandlerDirective::Continue
            },
        )
    }
}

struct FailOnTextDelta;

#[async_trait]
impl EventHandler<TextTurnEventWithTools<Tools>, TextTurnStateWithTools<Tools>>
    for FailOnTextDelta
{
    type Error = MockError;

    async fn on_event(
        &mut self,
        event: &TextTurnEventWithTools<Tools>,
        _cx: &HandlerContext<TextTurnStateWithTools<Tools>>,
    ) -> Result<HandlerDirective, Self::Error> {
        if matches!(event, TextTurnEventWithTools::TextDelta { .. }) {
            Err(MockError::Synthetic {
                message: "handler failed".into(),
            })
        } else {
            Ok(HandlerDirective::Continue)
        }
    }
}

#[test]
fn text_turn_collects_assistant_output_and_tool_calls() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::mock::RawTextTurnEvent::Started {
            request_id: Some("req-1".into()),
            model: "gpt-4.1".into(),
        }),
        Ok(lutum::mock::RawTextTurnEvent::TextDelta {
            delta: "looking up ".into(),
        }),
        Ok(lutum::mock::RawTextTurnEvent::ToolCallChunk {
            id: "call-1".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
        }),
        Ok(lutum::mock::RawTextTurnEvent::Completed {
            request_id: Some("req-1".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage {
                total_tokens: 12,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Lutum::new(Arc::new(adapter), budget);
    let pending = block_on(weather_turn(ctx.text_turn(input())).start()).unwrap();
    let result = block_on(pending.collect()).unwrap();

    assert_eq!(result.assistant_text(), "looking up ");
    assert_eq!(result.tool_calls.len(), 1);
    assert!(matches!(
        &result.turn.items()[0],
        AssistantTurnItem::Text(text) if text == "looking up "
    ));
    assert!(matches!(
        &result.turn.items()[1],
        AssistantTurnItem::ToolCall { .. }
    ));
}

#[test]
fn structured_turn_collects_typed_output_and_appends_assistant_item() {
    let adapter =
        MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(lutum::mock::RawStructuredTurnEvent::Started {
                request_id: Some("req-2".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(lutum::mock::RawStructuredTurnEvent::StructuredOutputChunk {
                json_delta: "{\"answer\":\"42\"}".into(),
            }),
            Ok(lutum::mock::RawStructuredTurnEvent::Completed {
                request_id: Some("req-2".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 9,
                    ..Usage::zero()
                },
            }),
        ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Lutum::new(Arc::new(adapter), budget);
    let result = block_on(ctx.structured_turn::<Summary>(input()).collect()).unwrap();

    assert!(matches!(
        result.semantic,
        StructuredTurnOutcome::Structured(Summary { ref answer }) if answer == "42"
    ));
    assert!(matches!(
        &result.assistant_turn.items()[0],
        AssistantTurnItem::Text(text) if text == "{\"answer\":\"42\"}"
    ));
}

#[test]
fn recorded_events_reduce_to_same_result_as_collect() {
    let arguments = lutum::RawJson::parse("{\"city\":\"Tokyo\"}").unwrap();
    let events = vec![
        TextTurnEventWithTools::<Tools>::Started {
            request_id: Some("req-r".into()),
            model: "gpt-4.1".into(),
        },
        TextTurnEventWithTools::<Tools>::TextDelta {
            delta: "checking ".into(),
        },
        TextTurnEventWithTools::<Tools>::ToolCallReady(ToolsCall::Weather(WeatherArgsCall {
            metadata: ToolMetadata::new("call-1", "weather", arguments),
            input: WeatherArgs {
                city: "Tokyo".into(),
            },
        })),
        TextTurnEventWithTools::<Tools>::Completed {
            request_id: Some("req-r".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage {
                total_tokens: 5,
                ..Usage::zero()
            },
            committed_turn: Arc::new(AssistantTurnView::from_items(&[])),
        },
    ];

    let mut reducer = TextTurnReducerWithTools::<Tools>::new();
    for event in &events {
        reducer.apply(event).unwrap();
    }
    let reduced = reducer.into_result().unwrap();

    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::mock::RawTextTurnEvent::Started {
            request_id: Some("req-r".into()),
            model: "gpt-4.1".into(),
        }),
        Ok(lutum::mock::RawTextTurnEvent::TextDelta {
            delta: "checking ".into(),
        }),
        Ok(lutum::mock::RawTextTurnEvent::ToolCallChunk {
            id: "call-1".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
        }),
        Ok(lutum::mock::RawTextTurnEvent::Completed {
            request_id: Some("req-r".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage {
                total_tokens: 5,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Lutum::new(Arc::new(adapter), budget);
    let pending = block_on(weather_turn(ctx.text_turn(input())).start()).unwrap();
    let collected = block_on(pending.collect()).unwrap();

    assert_eq!(*reduced.turn, *collected.turn);
    assert_eq!(reduced.tool_calls, collected.tool_calls);
    assert_eq!(reduced.finish_reason, collected.finish_reason);
    assert_eq!(reduced.usage, collected.usage);
}

#[test]
fn handler_stop_returns_partial_including_triggering_event_and_releases_budget() {
    let budget = test_budget();
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::mock::RawTextTurnEvent::Started {
            request_id: Some("req-stop".into()),
            model: "gpt-4.1".into(),
        }),
        Ok(lutum::mock::RawTextTurnEvent::TextDelta { delta: "he".into() }),
    ]));
    let ctx = Lutum::new(Arc::new(adapter), budget.clone());
    let pending = block_on(
        ctx.text_turn(input())
            .tools::<Tools>()
            .ext(UsageEstimate {
                total_tokens: 10,
                ..UsageEstimate::zero()
            })
            .start(),
    )
    .unwrap();

    let err = block_on(pending.collect_with(StopOnTextDelta)).unwrap_err();

    match err {
        CollectError::Stopped { partial } => {
            assert!(matches!(
                partial.assistant_turn.as_slice(),
                [AssistantTurnItem::Text(text)] if text == "he"
            ));
        }
        other => panic!("unexpected error: {other:?}"),
    }

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 100);
}

#[test]
fn recovery_failure_does_not_replace_stopped_error() {
    let budget = test_budget();
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(lutum::mock::RawTextTurnEvent::Started {
                request_id: Some("req-stop-recovery-error".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(lutum::mock::RawTextTurnEvent::TextDelta { delta: "he".into() }),
        ]))
        .with_recover_usage_error(
            OperationKind::TextTurn,
            "req-stop-recovery-error",
            MockError::Synthetic {
                message: "recovery failed".into(),
            },
        );
    let ctx = Lutum::new(Arc::new(adapter), budget.clone());
    let pending = block_on(
        ctx.text_turn(input())
            .tools::<Tools>()
            .ext(UsageEstimate {
                total_tokens: 10,
                ..UsageEstimate::zero()
            })
            .start(),
    )
    .unwrap();

    let err = block_on(pending.collect_with(StopOnTextDelta)).unwrap_err();

    match err {
        CollectError::Stopped { partial } => {
            assert_eq!(
                partial.request_id.as_deref(),
                Some("req-stop-recovery-error")
            );
            assert!(matches!(
                partial.assistant_turn.as_slice(),
                [AssistantTurnItem::Text(text)] if text == "he"
            ));
        }
        other => panic!("unexpected error: {other:?}"),
    }

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 100);
}

#[test]
fn recovery_failure_does_not_replace_handler_error() {
    let budget = test_budget();
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(lutum::mock::RawTextTurnEvent::Started {
                request_id: Some("req-handler-recovery-error".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(lutum::mock::RawTextTurnEvent::TextDelta { delta: "he".into() }),
        ]))
        .with_recover_usage_error(
            OperationKind::TextTurn,
            "req-handler-recovery-error",
            MockError::Synthetic {
                message: "recovery failed".into(),
            },
        );
    let ctx = Lutum::new(Arc::new(adapter), budget.clone());
    let pending = block_on(
        ctx.text_turn(input())
            .tools::<Tools>()
            .ext(UsageEstimate {
                total_tokens: 10,
                ..UsageEstimate::zero()
            })
            .start(),
    )
    .unwrap();

    let err = block_on(pending.collect_with(FailOnTextDelta)).unwrap_err();

    match err {
        CollectError::Handler { source, partial } => {
            assert_eq!(
                source,
                MockError::Synthetic {
                    message: "handler failed".into(),
                }
            );
            assert_eq!(
                partial.request_id.as_deref(),
                Some("req-handler-recovery-error")
            );
            assert!(matches!(
                partial.assistant_turn.as_slice(),
                [AssistantTurnItem::Text(text)] if text == "he"
            ));
        }
        other => panic!("unexpected error: {other:?}"),
    }

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 100);
}

#[test]
fn recovery_failure_does_not_replace_unexpected_eof() {
    let budget = test_budget();
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(lutum::mock::RawTextTurnEvent::Started {
                request_id: Some("req-eof-recovery-error".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(lutum::mock::RawTextTurnEvent::TextDelta { delta: "he".into() }),
        ]))
        .with_recover_usage_error(
            OperationKind::TextTurn,
            "req-eof-recovery-error",
            MockError::Synthetic {
                message: "recovery failed".into(),
            },
        );
    let ctx = Lutum::new(Arc::new(adapter), budget.clone());
    let pending = block_on(
        ctx.text_turn(input())
            .tools::<Tools>()
            .ext(UsageEstimate {
                total_tokens: 10,
                ..UsageEstimate::zero()
            })
            .start(),
    )
    .unwrap();

    let err = block_on(pending.collect()).unwrap_err();

    match err {
        CollectError::UnexpectedEof { partial } => {
            assert_eq!(
                partial.request_id.as_deref(),
                Some("req-eof-recovery-error")
            );
            assert!(matches!(
                partial.assistant_turn.as_slice(),
                [AssistantTurnItem::Text(text)] if text == "he"
            ));
        }
        other => panic!("unexpected error: {other:?}"),
    }

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 100);
}

#[test]
fn into_stream_releases_reserved_budget_without_collect() {
    let budget = test_budget();
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![]));
    let ctx = Lutum::new(Arc::new(adapter), budget.clone());
    let pending = block_on(
        ctx.text_turn(input())
            .tools::<Tools>()
            .ext(UsageEstimate {
                total_tokens: 10,
                ..UsageEstimate::zero()
            })
            .start(),
    )
    .unwrap();

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 90);

    let _stream = pending.into_stream();

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 100);
}

#[test]
fn adapter_error_uses_recovered_usage_when_available() {
    let budget = test_budget();
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(lutum::mock::RawTextTurnEvent::Started {
                request_id: Some("req-recover".into()),
                model: "gpt-4.1".into(),
            }),
            Err(MockError::Synthetic {
                message: "boom".into(),
            }),
        ]))
        .with_recovered_usage(
            OperationKind::TextTurn,
            "req-recover",
            Usage {
                total_tokens: 5,
                ..Usage::zero()
            },
        );
    let ctx = Lutum::new(Arc::new(adapter), budget.clone());
    let pending = block_on(
        ctx.text_turn(input())
            .tools::<Tools>()
            .ext(UsageEstimate {
                total_tokens: 10,
                ..UsageEstimate::zero()
            })
            .start(),
    )
    .unwrap();

    let err = block_on(pending.collect()).unwrap_err();
    assert!(matches!(err, CollectError::Execution { .. }));
    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 95);
}

#[test]
fn tool_call_deserialize_error_is_collected_as_recoverable_failure() {
    let budget = test_budget();
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::mock::RawTextTurnEvent::Started {
            request_id: Some("req-bad-tool".into()),
            model: "gpt-4.1".into(),
        }),
        Ok(lutum::mock::RawTextTurnEvent::TextDelta {
            delta: "looking up ".into(),
        }),
        Ok(lutum::mock::RawTextTurnEvent::ToolCallChunk {
            id: "call-bad".into(),
            name: "weather".into(),
            arguments_json_delta: "{}".into(),
        }),
        Ok(lutum::mock::RawTextTurnEvent::Completed {
            request_id: Some("req-bad-tool".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage::zero(),
        }),
    ]));
    let ctx = Lutum::new(Arc::new(adapter), budget.clone());
    let pending = block_on(
        weather_turn(ctx.text_turn(input()))
            .ext(UsageEstimate {
                total_tokens: 10,
                ..UsageEstimate::zero()
            })
            .start(),
    )
    .unwrap();

    let result = block_on(pending.collect()).unwrap();

    assert_eq!(result.request_id.as_deref(), Some("req-bad-tool"));
    assert_eq!(result.assistant_text(), "looking up ");
    assert!(result.tool_calls.is_empty());
    assert_eq!(result.recoverable_tool_call_issues.len(), 1);
    assert_eq!(
        result.continue_suggestion,
        Some(lutum::ContinueSuggestionReason::RecoverableToolCallIssue)
    );
    assert_eq!(
        result.recoverable_tool_call_issues[0].reason,
        lutum::RecoverableToolCallIssueReason::InvalidArguments
    );
    assert_eq!(
        result.recoverable_tool_call_issues[0]
            .metadata
            .name
            .as_str(),
        "weather"
    );

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 100);
}

#[test]
fn structured_output_deserialize_error_surfaces_as_execution_error() {
    let budget = test_budget();
    let adapter =
        MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(lutum::mock::RawStructuredTurnEvent::Started {
                request_id: Some("req-bad-structured".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(lutum::mock::RawStructuredTurnEvent::StructuredOutputChunk {
                json_delta: "{\"answer\":42}".into(),
            }),
            Ok(lutum::mock::RawStructuredTurnEvent::Completed {
                request_id: Some("req-bad-structured".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 8,
                    ..Usage::zero()
                },
            }),
        ]));
    let ctx = Lutum::new(Arc::new(adapter), budget.clone());
    let pending = block_on(
        ctx.structured_turn::<Summary>(input())
            .ext(UsageEstimate {
                total_tokens: 10,
                ..UsageEstimate::zero()
            })
            .start(),
    )
    .unwrap();

    let err = block_on(pending.collect()).unwrap_err();

    match err {
        CollectError::Execution { source, partial } => {
            assert!(matches!(source, lutum::AgentError::StructuredOutput(_)));
            assert_eq!(partial.request_id.as_deref(), Some("req-bad-structured"));
            assert!(matches!(
                partial.assistant_turn.as_slice(),
                [AssistantTurnItem::Text(text)] if text == "{\"answer\":42}"
            ));
            assert!(partial.structured.is_none());
            assert!(partial.finish_reason.is_none());
        }
        other => panic!("unexpected error: {other:?}"),
    }

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 100);
}

#[test]
fn request_budget_is_enforced_per_turn() {
    let adapter = MockLlmAdapter::new();
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Lutum::new(Arc::new(adapter), budget);

    let err = match block_on(
        ctx.text_turn(input())
            .tools::<Tools>()
            .budget(RequestBudget::from_tokens(16))
            .ext(UsageEstimate {
                total_tokens: 32,
                ..UsageEstimate::zero()
            })
            .start(),
    ) {
        Ok(_) => panic!("request should have been rejected by the per-request budget"),
        Err(err) => err,
    };

    assert!(matches!(
        shared_pool_budget_error(&err),
        lutum::SharedPoolBudgetError::RequestBudgetExceeded {
            requested_tokens: 32,
            budget_tokens: Some(16),
            ..
        }
    ));
}
