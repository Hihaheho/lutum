use agents::{
    FinishReason, MockLlmAdapter, MockStructuredScenario, MockTextScenario, NoTools,
    RequestExtensions, Session, SharedPoolBudgetManager, SharedPoolBudgetOptions,
    StructuredStepOutcome, TextStepOutcome, TextTurn, ToolPolicy, Usage, UsageEstimate,
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
    let ctx = agents::Context::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);
    session.push_user("Hi.");
    let before_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    let outcome = futures::executor::block_on(async {
        session
            .prepare_text(
                RequestExtensions::new(),
                TextTurn::<NoTools>::new(agents::ModelName::new("gpt-4.1-mini").unwrap()),
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
        TextStepOutcome::Finished(result) => session.commit_text(result),
        TextStepOutcome::NeedsToolResults(_) => unreachable!(),
    }

    assert_eq!(session.input().items().len(), before_len + 1);
    assert_eq!(session.list_turns().count(), 1);
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
    let ctx = agents::Context::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);
    session.push_user("Check weather.");
    let before_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    let outcome = futures::executor::block_on(async {
        session
            .prepare_text(
                RequestExtensions::new(),
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

    assert_eq!(session.input().items().len(), before_len);
    assert_eq!(session.list_turns().count(), before_turns);

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

    assert_eq!(session.input().items().len(), before_len + 2);
    assert_eq!(session.list_turns().count(), 1);
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
    let ctx = agents::Context::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);

    for prompt in ["step one", "step two"] {
        session.push_user(prompt);
        let outcome = futures::executor::block_on(async {
            session
                .prepare_text(
                    RequestExtensions::new(),
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
            TextStepOutcome::Finished(result) => session.commit_text(result),
            TextStepOutcome::NeedsToolResults(_) => unreachable!(),
        }
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
    let ctx = agents::Context::new(Arc::new(adapter), budget);
    let mut session = Session::new(ctx);
    session.push_user("Plan with a tool.");
    let before_len = session.input().items().len();
    let before_turns = session.list_turns().count();

    let outcome = futures::executor::block_on(async {
        session
            .prepare_structured(
                RequestExtensions::new(),
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

    assert_eq!(session.input().items().len(), before_len);
    assert_eq!(session.list_turns().count(), before_turns);

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

    assert_eq!(session.input().items().len(), before_len + 2);
    assert_eq!(session.list_turns().count(), 1);
}
