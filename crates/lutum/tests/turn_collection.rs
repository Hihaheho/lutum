use async_trait::async_trait;
use futures::executor::block_on;

use lutum::{
    AssistantTurnItem, AssistantTurnView, BudgetManager, CollectError, EventHandler, EventHandlers,
    FinishReason, HandlerContext, HandlerDirective, InputMessageRole, Lutum, MockError,
    MockLlmAdapter, MockStructuredScenario, MockTextScenario, ModelInput, ModelInputItem,
    OperationKind, RequestBudget, RequestExtensions, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, StructuredTurnOutcome, TextToolCollectError, TextToolErrorDirective,
    TextToolEventHandler, TextToolHandlerContext, TextToolHandlerDirective, TextTurnEvent,
    TextTurnEventWithTools, TextTurnReducerWithTools, TextTurnState, TextTurnStateWithTools,
    ToolMetadata, Usage, UsageEstimate, UsageRecoveryAdapter,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

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

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl EventHandler<TextTurnEventWithTools<Tools>, TextTurnStateWithTools<Tools>>
    for StopOnTextDelta
{
    async fn on_event(
        &mut self,
        event: &TextTurnEventWithTools<Tools>,
        _cx: &HandlerContext<TextTurnStateWithTools<Tools>>,
    ) -> lutum::HandlerResult {
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

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl EventHandler<TextTurnEventWithTools<Tools>, TextTurnStateWithTools<Tools>>
    for FailOnTextDelta
{
    async fn on_event(
        &mut self,
        event: &TextTurnEventWithTools<Tools>,
        _cx: &HandlerContext<TextTurnStateWithTools<Tools>>,
    ) -> lutum::HandlerResult {
        if matches!(event, TextTurnEventWithTools::TextDelta { .. }) {
            Err(MockError::Synthetic {
                message: "handler failed".into(),
            }
            .into())
        } else {
            Ok(HandlerDirective::Continue)
        }
    }
}

fn ordered_text_adapter(request_id: &str) -> MockLlmAdapter {
    MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::mock::RawTextTurnEvent::Started {
            request_id: Some(request_id.into()),
            model: "gpt-4.1".into(),
        }),
        Ok(lutum::mock::RawTextTurnEvent::TextDelta {
            delta: "hello".into(),
        }),
        Ok(lutum::mock::RawTextTurnEvent::Completed {
            request_id: Some(request_id.into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 5,
                ..Usage::zero()
            },
        }),
    ]))
}

fn synthetic_finished_turn<T: lutum::Toolset>() -> lutum::SyntheticTextToolTurn<T> {
    let assistant_turn =
        lutum::AssistantTurn::from_items(vec![AssistantTurnItem::Text("synthetic".into())])
            .unwrap();
    lutum::SyntheticTextToolTurn::finished(assistant_turn)
}

struct ReturnFinishedOnTextDelta {
    label: &'static str,
    seen: Arc<Mutex<Vec<&'static str>>>,
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl TextToolEventHandler<Tools> for ReturnFinishedOnTextDelta {
    async fn on_event(
        &mut self,
        event: &TextTurnEventWithTools<Tools>,
        _cx: &TextToolHandlerContext<Tools>,
    ) -> lutum::TextToolHandlerResult<Tools> {
        if matches!(event, TextTurnEventWithTools::TextDelta { .. }) {
            self.seen.lock().unwrap().push(self.label);
            return Ok(TextToolHandlerDirective::Return(synthetic_finished_turn()));
        }
        Ok(TextToolHandlerDirective::Continue)
    }
}

struct RecoverUnexpectedEof {
    label: &'static str,
    seen: Arc<Mutex<Vec<&'static str>>>,
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl TextToolEventHandler<Tools> for RecoverUnexpectedEof {
    async fn on_event(
        &mut self,
        _event: &TextTurnEventWithTools<Tools>,
        _cx: &TextToolHandlerContext<Tools>,
    ) -> lutum::TextToolHandlerResult<Tools> {
        Ok(TextToolHandlerDirective::Continue)
    }

    async fn on_error(
        &mut self,
        error: TextToolCollectError<'_>,
        _cx: &TextToolHandlerContext<Tools>,
    ) -> lutum::TextToolErrorHandlerResult<Tools> {
        if matches!(error, TextToolCollectError::UnexpectedEof) {
            self.seen.lock().unwrap().push(self.label);
            return Ok(TextToolErrorDirective::Return(synthetic_finished_turn()));
        }
        Ok(TextToolErrorDirective::Propagate)
    }
}

#[test]
fn builder_event_handlers_run_in_registration_order() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let ctx = Lutum::new(
        Arc::new(ordered_text_adapter("req-handler-order")),
        test_budget(),
    );
    let first_seen = Arc::clone(&seen);
    let second_seen = Arc::clone(&seen);

    let result = block_on(
        ctx.text_turn(input())
            .on_event(
                move |event: &TextTurnEvent,
                      _cx: &HandlerContext<TextTurnState>|
                      -> lutum::HandlerResult {
                    if matches!(event, TextTurnEvent::TextDelta { .. }) {
                        first_seen.lock().unwrap().push("first");
                    }
                    Ok(HandlerDirective::Continue)
                },
            )
            .on_event(
                move |event: &TextTurnEvent,
                      _cx: &HandlerContext<TextTurnState>|
                      -> lutum::HandlerResult {
                    if matches!(event, TextTurnEvent::TextDelta { .. }) {
                        second_seen.lock().unwrap().push("second");
                    }
                    Ok(HandlerDirective::Continue)
                },
            )
            .collect(),
    )
    .unwrap();

    assert_eq!(result.assistant_text(), "hello");
    assert_eq!(*seen.lock().unwrap(), vec!["first", "second"]);
}

#[test]
fn first_stop_short_circuits_ordered_handlers() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let ctx = Lutum::new(
        Arc::new(ordered_text_adapter("req-handler-stop")),
        test_budget(),
    );
    let first_seen = Arc::clone(&seen);
    let second_seen = Arc::clone(&seen);

    let err = block_on(ctx.text_turn(input()).collect_with((
        move |event: &TextTurnEvent, _cx: &HandlerContext<TextTurnState>| -> lutum::HandlerResult {
            if matches!(event, TextTurnEvent::TextDelta { .. }) {
                first_seen.lock().unwrap().push("stop");
                return Ok(HandlerDirective::Stop);
            }
            Ok(HandlerDirective::Continue)
        },
        move |event: &TextTurnEvent, _cx: &HandlerContext<TextTurnState>| -> lutum::HandlerResult {
            if matches!(event, TextTurnEvent::TextDelta { .. }) {
                second_seen.lock().unwrap().push("after-stop");
            }
            Ok(HandlerDirective::Continue)
        },
    )))
    .unwrap_err();

    assert!(matches!(err, CollectError::Stopped { .. }));
    assert_eq!(*seen.lock().unwrap(), vec!["stop"]);
}

#[test]
fn first_error_short_circuits_ordered_handlers() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let ctx = Lutum::new(
        Arc::new(ordered_text_adapter("req-handler-error")),
        test_budget(),
    );
    let first_seen = Arc::clone(&seen);
    let second_seen = Arc::clone(&seen);

    let err = block_on(ctx.text_turn(input()).collect_with((
        move |event: &TextTurnEvent, _cx: &HandlerContext<TextTurnState>| -> lutum::HandlerResult {
            if matches!(event, TextTurnEvent::TextDelta { .. }) {
                first_seen.lock().unwrap().push("error");
                return Err(MockError::Synthetic {
                    message: "handler failed".into(),
                }
                .into());
            }
            Ok(HandlerDirective::Continue)
        },
        move |event: &TextTurnEvent, _cx: &HandlerContext<TextTurnState>| -> lutum::HandlerResult {
            if matches!(event, TextTurnEvent::TextDelta { .. }) {
                second_seen.lock().unwrap().push("after-error");
            }
            Ok(HandlerDirective::Continue)
        },
    )))
    .unwrap_err();

    let CollectError::Handler { source, .. } = err else {
        panic!("expected handler error");
    };
    assert!(source.downcast_backend::<MockError>().is_some());
    assert_eq!(*seen.lock().unwrap(), vec!["error"]);
}

#[test]
fn collect_with_appends_after_builder_event_handlers() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let ctx = Lutum::new(
        Arc::new(ordered_text_adapter("req-handler-append")),
        test_budget(),
    );
    let first_seen = Arc::clone(&seen);
    let second_seen = Arc::clone(&seen);

    let staged = block_on(
        ctx.text_turn(input())
            .on_event(
                move |event: &TextTurnEvent,
                      _cx: &HandlerContext<TextTurnState>|
                      -> lutum::HandlerResult {
                    if matches!(event, TextTurnEvent::TextDelta { .. }) {
                        first_seen.lock().unwrap().push("builder");
                    }
                    Ok(HandlerDirective::Continue)
                },
            )
            .collect_with(
                move |event: &TextTurnEvent,
                      _cx: &HandlerContext<TextTurnState>|
                      -> lutum::HandlerResult {
                    if matches!(event, TextTurnEvent::TextDelta { .. }) {
                        second_seen.lock().unwrap().push("collect_with");
                    }
                    Ok(HandlerDirective::Continue)
                },
            ),
    )
    .unwrap();

    assert_eq!(staged.assistant_text(), "hello");
    assert_eq!(*seen.lock().unwrap(), vec!["builder", "collect_with"]);
}

#[test]
fn event_handlers_container_preserves_order() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let ctx = Lutum::new(
        Arc::new(ordered_text_adapter("req-handler-container")),
        test_budget(),
    );
    let first_seen = Arc::clone(&seen);
    let second_seen = Arc::clone(&seen);
    let handlers: EventHandlers<'_, TextTurnEvent, TextTurnState> = EventHandlers::new()
        .with(
            move |event: &TextTurnEvent,
                  _cx: &HandlerContext<TextTurnState>|
                  -> lutum::HandlerResult {
                if matches!(event, TextTurnEvent::TextDelta { .. }) {
                    first_seen.lock().unwrap().push("first");
                }
                Ok(HandlerDirective::Continue)
            },
        )
        .with(
            move |event: &TextTurnEvent,
                  _cx: &HandlerContext<TextTurnState>|
                  -> lutum::HandlerResult {
                if matches!(event, TextTurnEvent::TextDelta { .. }) {
                    second_seen.lock().unwrap().push("second");
                }
                Ok(HandlerDirective::Continue)
            },
        );

    let _ = block_on(ctx.text_turn(input()).collect_with(handlers)).unwrap();

    assert_eq!(*seen.lock().unwrap(), vec!["first", "second"]);
}

#[test]
fn controlled_return_short_circuits_ordered_handlers() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let ctx = Lutum::new(
        Arc::new(ordered_text_adapter("req-controlled-return-order")),
        test_budget(),
    );

    let outcome = block_on(
        ctx.text_turn(input())
            .tools::<Tools>()
            .collect_controlled_with((
                ReturnFinishedOnTextDelta {
                    label: "first",
                    seen: Arc::clone(&seen),
                },
                ReturnFinishedOnTextDelta {
                    label: "second",
                    seen: Arc::clone(&seen),
                },
            )),
    )
    .unwrap();

    assert!(matches!(
        outcome,
        lutum::TextStepOutcomeWithTools::Finished(_)
    ));
    assert_eq!(*seen.lock().unwrap(), vec!["first"]);
}

#[test]
fn controlled_on_error_recovery_short_circuits_ordered_handlers() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![Ok(
        lutum::mock::RawTextTurnEvent::Started {
            request_id: Some("req-controlled-error-order".into()),
            model: "gpt-4.1".into(),
        },
    )]));
    let ctx = Lutum::new(Arc::new(adapter), test_budget());

    let outcome = block_on(
        ctx.text_turn(input())
            .tools::<Tools>()
            .collect_controlled_with((
                RecoverUnexpectedEof {
                    label: "first",
                    seen: Arc::clone(&seen),
                },
                RecoverUnexpectedEof {
                    label: "second",
                    seen: Arc::clone(&seen),
                },
            )),
    )
    .unwrap();

    assert!(matches!(
        outcome,
        lutum::TextStepOutcomeWithTools::Finished(_)
    ));
    assert_eq!(*seen.lock().unwrap(), vec!["first"]);
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
    let result = match block_on(pending.collect()).unwrap() {
        lutum::StagedTextTurnOutcomeWithTools::Turn(result) => result,
        lutum::StagedTextTurnOutcomeWithTools::FinishedNoOutput(_) => {
            panic!("expected assistant turn")
        }
    };

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
    let reduced = match reducer.into_result().unwrap() {
        lutum::StagedTextTurnOutcomeWithTools::Turn(result) => result,
        lutum::StagedTextTurnOutcomeWithTools::FinishedNoOutput(_) => {
            panic!("expected assistant turn")
        }
    };

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
    let collected = match block_on(pending.collect()).unwrap() {
        lutum::StagedTextTurnOutcomeWithTools::Turn(result) => result,
        lutum::StagedTextTurnOutcomeWithTools::FinishedNoOutput(_) => {
            panic!("expected assistant turn")
        }
    };

    assert_eq!(*reduced.turn, *collected.turn);
    assert_eq!(reduced.tool_calls, collected.tool_calls);
    assert_eq!(reduced.finish_reason, collected.finish_reason);
    assert_eq!(reduced.usage, collected.usage);
}

#[test]
fn handler_stop_returns_partial_including_triggering_event_and_accounts_budget() {
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

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 90);
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

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 90);
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
                source.downcast_backend::<MockError>(),
                Some(&MockError::Synthetic {
                    message: "handler failed".into(),
                })
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

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 90);
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

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 90);
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
fn text_turn_length_finish_reason_surfaces_as_reduction_error() {
    let budget = test_budget();
    let usage = Usage {
        input_tokens: 14,
        output_tokens: 16,
        total_tokens: 30,
        ..Usage::zero()
    };
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::mock::RawTextTurnEvent::Started {
            request_id: Some("req-length".into()),
            model: "qwen3.5:9b".into(),
        }),
        Ok(lutum::mock::RawTextTurnEvent::ReasoningDelta {
            delta: "thinking".into(),
        }),
        Ok(lutum::mock::RawTextTurnEvent::Completed {
            request_id: Some("req-length".into()),
            finish_reason: FinishReason::Length,
            usage,
        }),
    ]));
    let ctx = Lutum::new(Arc::new(adapter), budget.clone());
    let pending = block_on(
        ctx.text_turn(input())
            .ext(UsageEstimate {
                total_tokens: 40,
                ..UsageEstimate::zero()
            })
            .start(),
    )
    .unwrap();

    let err = block_on(pending.collect()).unwrap_err();

    match err {
        CollectError::Reduction {
            source: lutum::TextTurnReductionError::OutputLimitExceeded(limit),
            partial,
        } => {
            assert_eq!(limit.model, "qwen3.5:9b");
            assert_eq!(limit.request_id.as_deref(), Some("req-length"));
            assert_eq!(limit.usage, usage);
            assert_eq!(partial.finish_reason, Some(FinishReason::Length));
            assert!(matches!(
                partial.assistant_turn.as_slice(),
                [AssistantTurnItem::Reasoning(text)] if text == "thinking"
            ));
        }
        other => panic!("unexpected error: {other:?}"),
    }

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 70);
}

#[test]
fn structured_turn_length_takes_precedence_over_missing_semantic() {
    let budget = test_budget();
    let usage = Usage {
        input_tokens: 20,
        output_tokens: 64,
        total_tokens: 84,
        ..Usage::zero()
    };
    let adapter =
        MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(lutum::mock::RawStructuredTurnEvent::Started {
                request_id: Some("req-structured-length".into()),
                model: "qwen3.5:9b".into(),
            }),
            Ok(lutum::mock::RawStructuredTurnEvent::ReasoningDelta {
                delta: "thinking only".into(),
            }),
            Ok(lutum::mock::RawStructuredTurnEvent::Completed {
                request_id: Some("req-structured-length".into()),
                finish_reason: FinishReason::Length,
                usage,
            }),
        ]));
    let ctx = Lutum::new(Arc::new(adapter), budget.clone());
    let pending = block_on(
        ctx.structured_turn::<Summary>(input())
            .ext(UsageEstimate {
                total_tokens: 90,
                ..UsageEstimate::zero()
            })
            .start(),
    )
    .unwrap();

    let err = block_on(pending.collect()).unwrap_err();

    match err {
        CollectError::Reduction {
            source: lutum::StructuredTurnReductionError::OutputLimitExceeded(limit),
            partial,
        } => {
            assert_eq!(limit.model, "qwen3.5:9b");
            assert_eq!(limit.request_id.as_deref(), Some("req-structured-length"));
            assert_eq!(limit.usage, usage);
            assert_eq!(partial.finish_reason, Some(FinishReason::Length));
            assert!(partial.structured.is_none());
        }
        other => panic!("unexpected error: {other:?}"),
    }

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 16);
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
    let adapter = Arc::new(adapter);
    let ctx = Lutum::new(adapter.clone(), budget.clone())
        .with_recovery(adapter as Arc<dyn UsageRecoveryAdapter>);
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

    let result = match block_on(pending.collect()).unwrap() {
        lutum::StagedTextTurnOutcomeWithTools::Turn(result) => result,
        lutum::StagedTextTurnOutcomeWithTools::FinishedNoOutput(_) => {
            panic!("expected assistant turn")
        }
    };

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

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 90);
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
