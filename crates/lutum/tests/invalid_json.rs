use std::sync::Arc;

use futures::{StreamExt, executor::block_on};
use lutum::{
    BudgetManager, FinishReason, Lutum, MockLlmAdapter, MockStructuredCompletionScenario,
    MockStructuredScenario, MockTextScenario, RawStructuredCompletionEvent, RawStructuredTurnEvent,
    RawTextTurnEvent, RecoverableToolCallIssueReason, RequestExtensions, Session,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, StructuredTurnOutcome,
    TextStepOutcomeWithTools, TextTurnEventWithTools, ToolArguments, ToolCallWrapper, Usage,
    UsageEstimate,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Answer {
    answer: String,
}

#[lutum::tool_input(name = "weather", output = String)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Weather {
    city: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug, lutum::Toolset)]
enum Tools {
    Weather(Weather),
}

fn budget() -> SharedPoolBudgetManager {
    SharedPoolBudgetManager::new(SharedPoolBudgetOptions {
        capacity_tokens: 100,
        ..SharedPoolBudgetOptions::default()
    })
}
fn usage() -> Usage {
    Usage {
        total_tokens: 8,
        ..Usage::zero()
    }
}
fn structured_scenario(raw: &str) -> MockStructuredScenario {
    MockStructuredScenario::events(vec![
        Ok(RawStructuredTurnEvent::Started {
            request_id: Some("invalid".into()),
            model: "mock".into(),
        }),
        Ok(RawStructuredTurnEvent::StructuredOutputChunk {
            json_delta: raw.into(),
        }),
        Ok(RawStructuredTurnEvent::Completed {
            request_id: Some("invalid".into()),
            finish_reason: FinishReason::Stop,
            usage: usage(),
        }),
    ])
}

#[test]
fn malformed_and_schema_mismatched_structured_turns_preserve_usage_and_history() {
    for raw in ["{", "{\"answer\":42}", "{\"other\":true}", "   "] {
        let budget = budget();
        let ctx = Lutum::new(
            Arc::new(MockLlmAdapter::new().with_structured_scenario(structured_scenario(raw))),
            budget.clone(),
        );
        let mut session = Session::new();
        session.push_user("Answer in JSON");
        let result = block_on(
            session
                .structured_turn::<Answer>()
                .ext(UsageEstimate {
                    total_tokens: 10,
                    ..UsageEstimate::zero()
                })
                .collect(&ctx),
        )
        .unwrap();
        assert_eq!(result.usage, usage());
        assert_eq!(result.request_id.as_deref(), Some("invalid"));
        let StructuredTurnOutcome::Invalid(issue) = result.semantic else {
            panic!("expected invalid output")
        };
        assert_eq!(issue.raw, raw);
        assert!(!issue.message.is_empty());
        assert_eq!(result.assistant_turn.assistant_text(), raw);
        assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 92);
        assert!(
            session
                .input()
                .items()
                .iter()
                .any(|item| matches!(item, lutum::ModelInputItem::Turn(_)))
        );
    }
}

#[test]
fn structured_completion_returns_invalid_output_and_usage() {
    for raw in ["{", "{\"answer\":42}", "[1,2]"] {
        let adapter = Arc::new(MockLlmAdapter::new().with_structured_completion_scenario(
            MockStructuredCompletionScenario::events(vec![
                Ok(RawStructuredCompletionEvent::Started {
                    request_id: Some("completion".into()),
                    model: "mock".into(),
                }),
                Ok(RawStructuredCompletionEvent::StructuredOutputChunk {
                    json_delta: raw.into(),
                }),
                Ok(RawStructuredCompletionEvent::Completed {
                    request_id: Some("completion".into()),
                    finish_reason: FinishReason::Stop,
                    usage: usage(),
                }),
            ]),
        ));
        let ctx = Lutum::from_parts(adapter.clone(), adapter, budget());
        let result = block_on(ctx.structured_completion::<Answer>("answer").collect()).unwrap();
        assert_eq!(result.usage, usage());
        let StructuredTurnOutcome::Invalid(issue) = result.semantic else {
            panic!("expected invalid output")
        };
        assert_eq!(issue.raw, raw);
    }
}

fn text_scenario() -> MockTextScenario {
    MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("tools".into()),
            model: "mock".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "bad".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "good".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("tools".into()),
            finish_reason: FinishReason::ToolCall,
            usage: usage(),
        }),
    ])
}

#[test]
fn malformed_tool_arguments_do_not_discard_valid_calls_and_can_be_rejected() {
    let ctx = Lutum::new(
        Arc::new(MockLlmAdapter::new().with_text_scenario(text_scenario())),
        budget(),
    );
    let mut session = Session::new();
    session.push_user("weather");
    let outcome = block_on(session.text_turn().tools::<Tools>().collect(&ctx)).unwrap();
    let TextStepOutcomeWithTools::NeedsTools(round) = outcome else {
        panic!("expected tool round")
    };
    assert_eq!(round.tool_calls.len(), 1);
    let issue = &round.recoverable_tool_call_issues()[0];
    assert_eq!(
        issue.reason,
        RecoverableToolCallIssueReason::InvalidArguments
    );
    assert_eq!(issue.metadata.arguments.get(), "{\"city\":");
    assert!(issue.message.as_ref().unwrap().contains("EOF"));
    // Dropping/committing a rejected call remains an explicit user-code decision.
    let good = round.tool_calls[0].metadata().clone();
    let result = good.into_tool_result(lutum::RawJson::from_serializable(&"sunny").unwrap());
    round.commit_into(session.input_mut(), [result]).unwrap();
    let turns: Vec<_> = session.list_turns().collect();
    let bad = (0..turns[0].item_count())
        .filter_map(|i| turns[0].item_at(i).unwrap().as_tool_call())
        .find(|call| call.id.as_str() == "bad")
        .unwrap();
    assert_eq!(bad.arguments.get(), "{\"city\":");
}

#[test]
fn malformed_tool_stream_reaches_completed() {
    let ctx = Lutum::new(
        Arc::new(MockLlmAdapter::new().with_text_scenario(text_scenario())),
        budget(),
    );
    let mut session = Session::new();
    session.push_user("weather");
    let pending = block_on(session.text_turn().tools::<Tools>().start(&ctx)).unwrap();
    let events = block_on(pending.into_stream().collect::<Vec<_>>());
    assert!(events.iter().all(Result::is_ok));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, Ok(TextTurnEventWithTools::ToolCallIssue(_))))
    );
    assert!(
        matches!(events.last(), Some(Ok(TextTurnEventWithTools::Completed { usage: actual, .. })) if *actual == usage())
    );
}

#[test]
fn raw_tool_arguments_roundtrip_invalid_json_without_fabricating_json() {
    for raw in ["{", "{  \"city\":\"Tokyo\" }", "\"text\""] {
        let arguments = ToolArguments::new(raw);
        let stored = serde_json::to_string(&arguments).unwrap();
        let decoded: ToolArguments = serde_json::from_str(&stored).unwrap();
        assert_eq!(decoded.get(), raw);
    }
}

#[lutum::tool_input(name = "other", output = String)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Other {
    value: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum DynamicTools {
    Other(Other),
    #[dynamic]
    Dynamic(lutum::DynamicTool),
}

#[test]
fn dynamic_tools_do_not_surface_malformed_json_as_executable_calls() {
    let ctx = Lutum::new(
        Arc::new(MockLlmAdapter::new().with_text_scenario(text_scenario())),
        budget(),
    );
    let mut session = Session::new();
    session.push_user("weather");
    let outcome = block_on(
        session
            .text_turn()
            .tools::<DynamicTools>()
            .with_dynamic_tools([lutum::DynamicTool::new(
                "weather",
                "weather",
                serde_json::json!({"type":"object"}),
            )])
            .collect(&ctx),
    )
    .unwrap();
    let TextStepOutcomeWithTools::NeedsTools(round) = outcome else {
        panic!("expected tool round")
    };
    assert_eq!(round.tool_calls.len(), 1);
    assert_eq!(
        round.recoverable_tool_call_issues()[0].reason,
        RecoverableToolCallIssueReason::InvalidArguments
    );
}

#[test]
fn structured_tools_with_malformed_arguments_complete_as_a_tool_round() {
    let ctx = Lutum::new(
        Arc::new(
            MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
                Ok(RawStructuredTurnEvent::Started {
                    request_id: Some("tools".into()),
                    model: "mock".into(),
                }),
                Ok(RawStructuredTurnEvent::ToolCallChunk {
                    id: "bad".into(),
                    name: "weather".into(),
                    arguments_json_delta: "{".into(),
                }),
                Ok(RawStructuredTurnEvent::Completed {
                    request_id: Some("tools".into()),
                    finish_reason: FinishReason::ToolCall,
                    usage: usage(),
                }),
            ])),
        ),
        budget(),
    );
    let mut session = Session::new();
    session.push_user("weather");
    let outcome = block_on(
        session
            .structured_turn::<Answer>()
            .tools::<Tools>()
            .collect(&ctx),
    )
    .unwrap();
    let lutum::StructuredStepOutcomeWithTools::NeedsTools(round) = outcome else {
        panic!("expected tool round")
    };
    assert!(round.tool_calls.is_empty());
    assert_eq!(
        round.recoverable_tool_call_issues()[0]
            .metadata
            .arguments
            .get(),
        "{"
    );
    assert_eq!(round.usage, usage());
    round
        .commit_into(session.input_mut(), Vec::<lutum::ToolResult>::new())
        .unwrap();
    assert_eq!(session.list_turns().count(), 1);
}
