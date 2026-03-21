use async_trait::async_trait;
use futures::executor::block_on;

use agents::{
    AssistantTurnItem, AssistantTurnView, BudgetManager, CollectError, Context, EventHandler,
    FinishReason, HandlerContext, HandlerDirective, InputMessageRole, MockError, MockLlmAdapter,
    MockStructuredScenario, MockTextScenario, ModelInput, ModelInputItem, RequestBudget,
    RequestExtensions, SharedPoolBudgetManager, SharedPoolBudgetOptions, StreamKind,
    StructuredTurn, StructuredTurnOutcome, TextTurn, TextTurnEvent, TextTurnReducer, ToolMetadata,
    ToolPolicy, Usage, UsageEstimate,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[agents::tool_input(name = "weather", output = WeatherResult)]
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, agents::Toolset)]
enum Tools {
    Weather(WeatherArgs),
}

fn input() -> ModelInput {
    ModelInput::from_items(vec![ModelInputItem::text(InputMessageRole::User, "hello")])
}

fn weather_turn(model: &str) -> TextTurn<Tools> {
    let mut turn = TextTurn::new(agents::ModelName::new(model).unwrap());
    turn.config.tools = ToolPolicy::allow_only(vec![ToolsSelector::Weather]);
    turn
}

fn shared_pool_budget_error(err: &agents::AgentError) -> &agents::SharedPoolBudgetError {
    match err {
        agents::AgentError::Budget(source) => source
            .downcast_ref::<agents::SharedPoolBudgetError>()
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

fn extensions() -> RequestExtensions {
    RequestExtensions::new()
}

struct StopOnTextDelta;

#[async_trait]
impl EventHandler<TextTurnEvent<Tools>, agents::TextTurnState<Tools>> for StopOnTextDelta {
    type Error = std::convert::Infallible;

    async fn on_event(
        &mut self,
        event: &TextTurnEvent<Tools>,
        _cx: &HandlerContext<agents::TextTurnState<Tools>>,
    ) -> Result<HandlerDirective, Self::Error> {
        Ok(if matches!(event, TextTurnEvent::TextDelta { .. }) {
            HandlerDirective::Stop
        } else {
            HandlerDirective::Continue
        })
    }
}

#[test]
fn text_turn_collects_assistant_output_and_tool_calls() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(agents::mock::RawTextTurnEvent::Started {
            request_id: Some("req-1".into()),
            model: "gpt-4.1".into(),
        }),
        Ok(agents::mock::RawTextTurnEvent::TextDelta {
            delta: "looking up ".into(),
        }),
        Ok(agents::mock::RawTextTurnEvent::ToolCallChunk {
            id: "call-1".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
        }),
        Ok(agents::mock::RawTextTurnEvent::Completed {
            request_id: Some("req-1".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage {
                total_tokens: 12,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Context::new(budget, adapter);
    let turn = weather_turn("gpt-4.1");
    let pending =
        block_on(ctx.responses_text(extensions(), input(), turn, UsageEstimate::zero())).unwrap();
    let result = block_on(pending.collect_noop()).unwrap();

    assert_eq!(result.assistant_text(), "looking up ");
    assert_eq!(result.tool_calls.len(), 1);
    assert!(matches!(
        &result.assistant_turn.items()[0],
        AssistantTurnItem::Text(text) if text == "looking up "
    ));
    assert!(matches!(
        &result.assistant_turn.items()[1],
        AssistantTurnItem::ToolCall { .. }
    ));
}

#[test]
fn structured_turn_collects_typed_output_and_appends_assistant_item() {
    let adapter =
        MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(agents::mock::RawStructuredTurnEvent::Started {
                request_id: Some("req-2".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(
                agents::mock::RawStructuredTurnEvent::StructuredOutputChunk {
                    json_delta: "{\"answer\":\"42\"}".into(),
                },
            ),
            Ok(agents::mock::RawStructuredTurnEvent::Completed {
                request_id: Some("req-2".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 9,
                    ..Usage::zero()
                },
            }),
        ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Context::new(budget, adapter);
    let turn = StructuredTurn::<Tools, Summary>::new(agents::ModelName::new("gpt-4.1").unwrap());
    let pending =
        block_on(ctx.responses_structured(extensions(), input(), turn, UsageEstimate::zero()))
            .unwrap();
    let result = block_on(pending.collect_noop()).unwrap();

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
    let arguments = agents::RawJson::parse("{\"city\":\"Tokyo\"}").unwrap();
    let events = vec![
        TextTurnEvent::<Tools>::Started {
            request_id: Some("req-r".into()),
            model: "gpt-4.1".into(),
        },
        TextTurnEvent::<Tools>::TextDelta {
            delta: "checking ".into(),
        },
        TextTurnEvent::<Tools>::ToolCallReady(ToolsCall::Weather(WeatherArgsCall {
            metadata: ToolMetadata::new("call-1", "weather", arguments),
            input: WeatherArgs {
                city: "Tokyo".into(),
            },
        })),
        TextTurnEvent::<Tools>::Completed {
            request_id: Some("req-r".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage {
                total_tokens: 5,
                ..Usage::zero()
            },
            committed_turn: Arc::new(AssistantTurnView::from_items(&[])),
        },
    ];

    let mut reducer = TextTurnReducer::<Tools>::new();
    for event in &events {
        reducer.apply(event).unwrap();
    }
    let reduced = reducer.into_result().unwrap();

    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(agents::mock::RawTextTurnEvent::Started {
            request_id: Some("req-r".into()),
            model: "gpt-4.1".into(),
        }),
        Ok(agents::mock::RawTextTurnEvent::TextDelta {
            delta: "checking ".into(),
        }),
        Ok(agents::mock::RawTextTurnEvent::ToolCallChunk {
            id: "call-1".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
        }),
        Ok(agents::mock::RawTextTurnEvent::Completed {
            request_id: Some("req-r".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage {
                total_tokens: 5,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Context::new(budget, adapter);
    let pending = block_on(ctx.responses_text(
        extensions(),
        input(),
        weather_turn("gpt-4.1"),
        UsageEstimate::zero(),
    ))
    .unwrap();
    let collected = block_on(pending.collect_noop()).unwrap();

    assert_eq!(reduced.assistant_turn, collected.assistant_turn);
    assert_eq!(reduced.tool_calls, collected.tool_calls);
    assert_eq!(reduced.finish_reason, collected.finish_reason);
    assert_eq!(reduced.usage, collected.usage);
}

#[test]
fn handler_stop_returns_partial_including_triggering_event_and_releases_budget() {
    let budget = test_budget();
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(agents::mock::RawTextTurnEvent::Started {
            request_id: Some("req-stop".into()),
            model: "gpt-4.1".into(),
        }),
        Ok(agents::mock::RawTextTurnEvent::TextDelta { delta: "he".into() }),
    ]));
    let ctx = Context::new(budget.clone(), adapter);
    let pending = block_on(ctx.responses_text(
        extensions(),
        input(),
        TextTurn::<Tools>::new(agents::ModelName::new("gpt-4.1").unwrap()),
        UsageEstimate {
            total_tokens: 10,
            ..UsageEstimate::zero()
        },
    ))
    .unwrap();

    let err = block_on(pending.collect(StopOnTextDelta)).unwrap_err();

    match err {
        CollectError::Stopped { partial } => {
            assert_eq!(partial.assistant_text(), "he");
        }
        other => panic!("unexpected error: {other:?}"),
    }

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 100);
}

#[test]
fn into_stream_releases_reserved_budget_without_collect() {
    let budget = test_budget();
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![]));
    let ctx = Context::new(budget.clone(), adapter);
    let pending = block_on(ctx.responses_text(
        extensions(),
        input(),
        TextTurn::<Tools>::new(agents::ModelName::new("gpt-4.1").unwrap()),
        UsageEstimate {
            total_tokens: 10,
            ..UsageEstimate::zero()
        },
    ))
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
            Ok(agents::mock::RawTextTurnEvent::Started {
                request_id: Some("req-recover".into()),
                model: "gpt-4.1".into(),
            }),
            Err(MockError::Synthetic {
                message: "boom".into(),
            }),
        ]))
        .with_recovered_usage(
            StreamKind::ResponsesText,
            "req-recover",
            Usage {
                total_tokens: 5,
                ..Usage::zero()
            },
        );
    let ctx = Context::new(budget.clone(), adapter);
    let pending = block_on(ctx.responses_text(
        extensions(),
        input(),
        TextTurn::<Tools>::new(agents::ModelName::new("gpt-4.1").unwrap()),
        UsageEstimate {
            total_tokens: 10,
            ..UsageEstimate::zero()
        },
    ))
    .unwrap();

    let err = block_on(pending.collect_noop()).unwrap_err();
    assert!(matches!(err, CollectError::Execution { .. }));
    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 95);
}

#[test]
fn tool_call_deserialize_error_surfaces_as_execution_error() {
    let budget = test_budget();
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(agents::mock::RawTextTurnEvent::Started {
            request_id: Some("req-bad-tool".into()),
            model: "gpt-4.1".into(),
        }),
        Ok(agents::mock::RawTextTurnEvent::TextDelta {
            delta: "looking up ".into(),
        }),
        Ok(agents::mock::RawTextTurnEvent::ToolCallChunk {
            id: "call-bad".into(),
            name: "weather".into(),
            arguments_json_delta: "{}".into(),
        }),
    ]));
    let ctx = Context::new(budget.clone(), adapter);
    let pending = block_on(ctx.responses_text(
        extensions(),
        input(),
        weather_turn("gpt-4.1"),
        UsageEstimate {
            total_tokens: 10,
            ..UsageEstimate::zero()
        },
    ))
    .unwrap();

    let err = block_on(pending.collect_noop()).unwrap_err();

    match err {
        CollectError::Execution { source, partial } => {
            assert!(matches!(
                source,
                agents::AgentError::ToolCall(agents::ToolCallError::Deserialize { ref name, .. })
                    if name == "weather"
            ));
            assert_eq!(partial.request_id.as_deref(), Some("req-bad-tool"));
            assert_eq!(partial.assistant_text(), "looking up ");
            assert!(partial.tool_calls.is_empty());
            assert!(partial.finish_reason.is_none());
        }
        other => panic!("unexpected error: {other:?}"),
    }

    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 100);
}

#[test]
fn structured_output_deserialize_error_surfaces_as_execution_error() {
    let budget = test_budget();
    let adapter =
        MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(agents::mock::RawStructuredTurnEvent::Started {
                request_id: Some("req-bad-structured".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(
                agents::mock::RawStructuredTurnEvent::StructuredOutputChunk {
                    json_delta: "{\"answer\":42}".into(),
                },
            ),
            Ok(agents::mock::RawStructuredTurnEvent::Completed {
                request_id: Some("req-bad-structured".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 8,
                    ..Usage::zero()
                },
            }),
        ]));
    let ctx = Context::new(budget.clone(), adapter);
    let pending = block_on(ctx.responses_structured(
        extensions(),
        input(),
        StructuredTurn::<Tools, Summary>::new(agents::ModelName::new("gpt-4.1").unwrap()),
        UsageEstimate {
            total_tokens: 10,
            ..UsageEstimate::zero()
        },
    ))
    .unwrap();

    let err = block_on(pending.collect_noop()).unwrap_err();

    match err {
        CollectError::Execution { source, partial } => {
            assert!(matches!(source, agents::AgentError::StructuredOutput(_)));
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
    let ctx = Context::new(budget, adapter);

    let err = match block_on(ctx.responses_text(
        extensions(),
        input(),
        {
            let mut turn = TextTurn::<Tools>::new(agents::ModelName::new("gpt-4.1").unwrap());
            turn.config.budget = RequestBudget::from_tokens(16);
            turn
        },
        UsageEstimate {
            total_tokens: 32,
            ..UsageEstimate::zero()
        },
    )) {
        Ok(_) => panic!("request should have been rejected by the per-request budget"),
        Err(err) => err,
    };

    assert!(matches!(
        shared_pool_budget_error(&err),
        agents::SharedPoolBudgetError::RequestBudgetExceeded {
            requested_tokens: 32,
            budget_tokens: Some(16),
            ..
        }
    ));
}
