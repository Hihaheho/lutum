use agents::{
    FinishReason, Marker, MockLlmAdapter, MockStructuredScenario, MockTextScenario, NoTools,
    Session, SharedPoolBudgetManager, SharedPoolBudgetOptions, StructuredStepOutcome,
    TextStepOutcome, TextTurn, ToolPolicy, Usage, UsageEstimate,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
struct AppMarker;

impl Marker for AppMarker {
    fn span_name(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("session_api")
    }
}

#[agents::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, agents::Toolset)]
enum Tools {
    Weather(WeatherArgs),
}

#[test]
fn prepare_and_collect_do_not_mutate_transcript_before_commit() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(agents::RawTextTurnEvent::Started {
            request_id: Some("req-session-1".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(agents::RawTextTurnEvent::TextDelta {
            delta: "hello".into(),
        }),
        Ok(agents::RawTextTurnEvent::Completed {
            request_id: Some("req-session-1".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 4,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = agents::Context::<AppMarker, _, _>::new(budget, adapter);
    let mut session = Session::new(ctx, AppMarker);
    session.push_user("Hi.");
    let before = session.snapshot();

    let outcome = futures::executor::block_on(async {
        session
            .prepare_text(
                TextTurn::<NoTools>::new(agents::ModelName::new("gpt-4.1-mini").unwrap()),
                UsageEstimate::zero(),
            )
            .await
            .unwrap()
            .collect_noop()
            .await
            .unwrap()
    });

    assert_eq!(session.snapshot(), before);

    match outcome {
        TextStepOutcome::Finished(result) => session.commit_text(result).unwrap(),
        TextStepOutcome::NeedsToolResults(_) => unreachable!(),
    }

    assert!(session.snapshot().items().len() > before.items().len());
}

#[test]
fn tool_round_is_only_applied_on_explicit_commit() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(agents::RawTextTurnEvent::Started {
            request_id: Some("req-session-2".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(agents::RawTextTurnEvent::ToolCallChunk {
            id: "call-1".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
        }),
        Ok(agents::RawTextTurnEvent::Completed {
            request_id: Some("req-session-2".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage {
                total_tokens: 7,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = agents::Context::<AppMarker, _, _>::new(budget, adapter);
    let mut session = Session::new(ctx, AppMarker);
    session.push_user("Check weather.");
    let before = session.snapshot();

    let outcome = futures::executor::block_on(async {
        session
            .prepare_text(
                {
                    let mut turn =
                        TextTurn::<Tools>::new(agents::ModelName::new("gpt-4.1-mini").unwrap());
                    turn.config.tools = ToolPolicy::allow_only(vec![ToolsSelector::Weather]);
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

    assert_eq!(session.snapshot(), before);

    match outcome {
        TextStepOutcome::NeedsToolResults(round) => {
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
                })
                .collect::<Vec<_>>();
            session.commit_tool_round(round, tool_uses).unwrap();
        }
        TextStepOutcome::Finished(_) => unreachable!(),
    }

    assert!(session.snapshot().items().len() > before.items().len());
}

#[test]
fn snapshot_round_trips_through_session() {
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let adapter = MockLlmAdapter::new();
    let ctx = agents::Context::<AppMarker, _, _>::new(budget, adapter);
    let mut session = Session::new(ctx.clone(), AppMarker);
    session.push_system("Be exact.");
    session.push_user("Hello.");

    let snapshot = session.snapshot();
    let restored = Session::from_snapshot(ctx, AppMarker, snapshot.clone());

    assert_eq!(snapshot, restored.snapshot());
}

#[test]
fn session_can_drive_a_stateful_step_loop() {
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(agents::RawTextTurnEvent::Started {
                request_id: Some("req-step-1".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(agents::RawTextTurnEvent::TextDelta {
                delta: "first step".into(),
            }),
            Ok(agents::RawTextTurnEvent::Completed {
                request_id: Some("req-step-1".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 4,
                    ..Usage::zero()
                },
            }),
        ]))
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(agents::RawTextTurnEvent::Started {
                request_id: Some("req-step-2".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(agents::RawTextTurnEvent::TextDelta {
                delta: "second step".into(),
            }),
            Ok(agents::RawTextTurnEvent::Completed {
                request_id: Some("req-step-2".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 4,
                    ..Usage::zero()
                },
            }),
        ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = agents::Context::<AppMarker, _, _>::new(budget, adapter);
    let mut session = Session::new(ctx, AppMarker);

    for prompt in ["step one", "step two"] {
        session.push_user(prompt);
        let outcome = futures::executor::block_on(async {
            session
                .prepare_text(
                    TextTurn::<NoTools>::new(agents::ModelName::new("gpt-4.1-mini").unwrap()),
                    UsageEstimate::zero(),
                )
                .await
                .unwrap()
                .collect_noop()
                .await
                .unwrap()
        });
        match outcome {
            TextStepOutcome::Finished(result) => session.commit_text(result).unwrap(),
            TextStepOutcome::NeedsToolResults(_) => unreachable!(),
        }
    }

    assert!(session.snapshot().items().len() >= 4);
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Summary {
    answer: String,
}

#[test]
fn structured_tool_round_stays_explicit_until_commit() {
    let adapter =
        MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(agents::RawStructuredTurnEvent::Started {
                request_id: Some("req-session-3".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(agents::RawStructuredTurnEvent::ToolCallChunk {
                id: "call-1".into(),
                name: "weather".into(),
                arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
            }),
            Ok(agents::RawStructuredTurnEvent::Completed {
                request_id: Some("req-session-3".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage {
                    total_tokens: 6,
                    ..Usage::zero()
                },
            }),
        ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = agents::Context::<AppMarker, _, _>::new(budget, adapter);
    let mut session = Session::new(ctx, AppMarker);
    session.push_user("Plan with a tool.");
    let before = session.snapshot();

    let outcome = futures::executor::block_on(async {
        session
            .prepare_structured(
                {
                    let mut turn = agents::StructuredTurn::<Tools, Summary>::new(
                        agents::ModelName::new("gpt-4.1-mini").unwrap(),
                    );
                    turn.config.tools = ToolPolicy::allow_only(vec![ToolsSelector::Weather]);
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

    assert_eq!(session.snapshot(), before);

    match outcome {
        StructuredStepOutcome::NeedsToolResults(round) => {
            let tool_uses = round
                .tool_calls
                .iter()
                .cloned()
                .map(|tool_call| match tool_call {
                    ToolsCall::Weather(call) => call
                        .tool_use(WeatherResult {
                            forecast: "windy".into(),
                        })
                        .unwrap(),
                })
                .collect::<Vec<_>>();
            session.commit_tool_round(round, tool_uses).unwrap();
        }
        StructuredStepOutcome::Finished(_) => unreachable!(),
    }

    assert!(session.snapshot().items().len() > before.items().len());
}
