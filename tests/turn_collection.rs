use async_trait::async_trait;
use futures::executor::block_on;

use agents::{
    AssistantTurnItem, BudgetManager, CollectError, Context, EventHandler, FinishReason,
    HandlerContext, HandlerDirective, InputMessageRole, Marker, MockError, MockLlmAdapter,
    MockStructuredScenario, MockTextScenario, ModelInput, ModelInputItem, NonEmpty, RequestBudget,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, StreamKind, StructuredTurnOutcome,
    StructuredTurnRequest, TextTurnEvent, TextTurnReducer, TextTurnRequest, ToolCallError, ToolDef,
    ToolMode, Toolset, Usage, UsageEstimate,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
struct AppMarker;

impl Marker for AppMarker {
    fn span_name(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("turn")
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
enum Calls {
    Weather(WeatherArgs),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
enum Results {
    Weather { forecast: String },
}

#[derive(Clone, Copy, Debug, Default)]
struct Tools;

impl Toolset for Tools {
    type Call = Calls;
    type Result = Results;

    fn definitions() -> &'static [ToolDef<Self::Call, Self::Result>] {
        fn weather_args_schema() -> schemars::Schema {
            schemars::schema_for!(WeatherArgs)
        }

        static DEFS: [ToolDef<Calls, Results>; 1] =
            [ToolDef::new("weather", "Get weather", weather_args_schema)];
        &DEFS
    }

    fn parse_call(name: &str, arguments_json: &str) -> Result<Self::Call, ToolCallError> {
        match name {
            "weather" => serde_json::from_str(arguments_json)
                .map(Calls::Weather)
                .map_err(|source| ToolCallError::Deserialize {
                    name: name.to_string(),
                    source,
                }),
            _ => Err(ToolCallError::UnknownTool {
                name: name.to_string(),
            }),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Summary {
    answer: String,
}

fn input() -> ModelInput {
    ModelInput::from_items(vec![ModelInputItem::text(InputMessageRole::User, "hello")])
}

struct StopOnTextDelta;

#[async_trait]
impl EventHandler<TextTurnEvent<Tools>, AppMarker, agents::TextTurnState<Tools>>
    for StopOnTextDelta
{
    type Error = std::convert::Infallible;

    async fn on_event(
        &mut self,
        event: &TextTurnEvent<Tools>,
        _cx: &HandlerContext<AppMarker, agents::TextTurnState<Tools>>,
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
    let ctx: Context<AppMarker, _, _> = Context::new(budget, adapter);
    let turn = TextTurnRequest::<Tools>::new("gpt-4.1").with_tool_mode(ToolMode::AutoOnly(
        NonEmpty::one(Tools::definitions()[0].tool_ref()),
    ));
    let pending =
        block_on(ctx.responses_text(AppMarker, input(), turn, UsageEstimate::zero())).unwrap();
    let result = block_on(pending.collect_noop()).unwrap();

    assert_eq!(result.assistant_text(), "looking up ");
    assert_eq!(result.typed_tool_calls.len(), 1);
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
    let ctx: Context<AppMarker, _, _> = Context::new(budget, adapter);
    let turn = StructuredTurnRequest::<Tools, Summary>::new("gpt-4.1");
    let pending =
        block_on(ctx.responses_structured(AppMarker, input(), turn, UsageEstimate::zero()))
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
        TextTurnEvent::<Tools>::ToolCallReady(agents::TypedToolInvocation {
            id: "call-1".into(),
            name: "weather".into(),
            call: Calls::Weather(WeatherArgs {
                city: "Tokyo".into(),
            }),
            arguments,
        }),
        TextTurnEvent::<Tools>::Completed {
            request_id: Some("req-r".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage {
                total_tokens: 5,
                ..Usage::zero()
            },
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
    let ctx: Context<AppMarker, _, _> = Context::new(budget, adapter);
    let pending = block_on(ctx.responses_text(
        AppMarker,
        input(),
        TextTurnRequest::<Tools>::new("gpt-4.1").with_tool_mode(ToolMode::AutoOnly(NonEmpty::one(
            Tools::definitions()[0].tool_ref(),
        ))),
        UsageEstimate::zero(),
    ))
    .unwrap();
    let collected = block_on(pending.collect_noop()).unwrap();

    assert_eq!(reduced.assistant_turn, collected.assistant_turn);
    assert_eq!(reduced.typed_tool_calls, collected.typed_tool_calls);
    assert_eq!(reduced.finish_reason, collected.finish_reason);
    assert_eq!(reduced.usage, collected.usage);
}

#[test]
fn handler_stop_returns_partial_including_triggering_event_and_releases_budget() {
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions {
        capacity_tokens: 100,
        capacity_cost_micros_usd: 1_000,
        stop_threshold_tokens: 0,
        stop_threshold_cost_micros_usd: 0,
    });
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(agents::mock::RawTextTurnEvent::Started {
            request_id: Some("req-stop".into()),
            model: "gpt-4.1".into(),
        }),
        Ok(agents::mock::RawTextTurnEvent::TextDelta { delta: "he".into() }),
    ]));
    let ctx: Context<AppMarker, _, _> = Context::new(budget.clone(), adapter);
    let pending = block_on(ctx.responses_text(
        AppMarker,
        input(),
        TextTurnRequest::<Tools>::new("gpt-4.1"),
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

    assert_eq!(budget.remaining(&AppMarker).tokens, 100);
}

#[test]
fn adapter_error_uses_recovered_usage_when_available() {
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions {
        capacity_tokens: 100,
        capacity_cost_micros_usd: 1_000,
        stop_threshold_tokens: 0,
        stop_threshold_cost_micros_usd: 0,
    });
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
    let ctx: Context<AppMarker, _, _> = Context::new(budget.clone(), adapter);
    let pending = block_on(ctx.responses_text(
        AppMarker,
        input(),
        TextTurnRequest::<Tools>::new("gpt-4.1"),
        UsageEstimate {
            total_tokens: 10,
            ..UsageEstimate::zero()
        },
    ))
    .unwrap();

    let err = block_on(pending.collect_noop()).unwrap_err();
    assert!(matches!(err, CollectError::Adapter { .. }));
    assert_eq!(budget.remaining(&AppMarker).tokens, 95);
}

#[test]
fn request_budget_is_enforced_per_turn() {
    let adapter = MockLlmAdapter::new();
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx: Context<AppMarker, _, _> = Context::new(budget, adapter);

    let err = match block_on(
        ctx.responses_text(
            AppMarker,
            input(),
            TextTurnRequest::<Tools>::builder()
                .model("gpt-4.1")
                .budget(RequestBudget::from_tokens(16))
                .build(),
            UsageEstimate {
                total_tokens: 32,
                ..UsageEstimate::zero()
            },
        ),
    ) {
        Ok(_) => panic!("request should have been rejected by the per-request budget"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        agents::ContextError::Budget(agents::SharedPoolBudgetError::RequestBudgetExceeded {
            requested_tokens: 32,
            budget_tokens: Some(16),
            ..
        })
    ));
}
