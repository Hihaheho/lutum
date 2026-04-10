use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::{StreamExt, executor::block_on};
use lutum::{
    EventHandler, FinishReason, HandlerContext, HandlerDirective, Lutum, MockLlmAdapter,
    MockTextScenario, ModelInputItem, RawJson, RawTextTurnEvent, Session, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, TextStepOutcomeWithTools, TextTurnEventWithTools,
    TextTurnStateWithTools, ToolHookOutcome, ToolMetadata, ToolResult, Toolset, Usage,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchResult {
    hits: Vec<String>,
}

#[lutum::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[lutum::tool_input(name = "search", output = SearchResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct SearchArgs {
    query: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum Tools {
    Weather(WeatherArgs),
    Search(SearchArgs),
}

#[test]
fn tool_call_hook_returns_unhandled_when_no_override_is_registered() {
    let call = Tools::parse_tool_call(ToolMetadata::new(
        "call-1",
        "weather",
        RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
    ))
    .unwrap();

    match block_on(call.hook(&ToolsHooks::new())) {
        ToolHookOutcome::Unhandled(ToolsCall::Weather(call)) => {
            assert_eq!(call.metadata.id.as_str(), "call-1");
            assert_eq!(call.input().city, "Tokyo");
        }
        ToolHookOutcome::Unhandled(other) => panic!("unexpected unhandled variant: {other:?}"),
        ToolHookOutcome::Handled(_) => panic!("hook should not have handled the call"),
    }
}

#[test]
fn tool_call_hook_preserves_metadata_input_and_output_when_handled() {
    let call = Tools::parse_tool_call(ToolMetadata::new(
        "call-2",
        "weather",
        RawJson::parse("{\"city\":\"Osaka\"}").unwrap(),
    ))
    .unwrap();
    let hooks =
        ToolsHooks::new().with_weather(|metadata: &lutum::ToolMetadata, input: &WeatherArgs| {
            let id = metadata.id.as_str().to_owned();
            let city = input.city.clone();
            async move {
                Some(WeatherResult {
                    forecast: format!("hooked:{id}:{city}"),
                })
            }
        });

    match block_on(call.hook(&hooks)) {
        ToolHookOutcome::Handled(ToolsHandled::Weather(handled)) => {
            assert_eq!(handled.metadata().id.as_str(), "call-2");
            assert_eq!(handled.input().city, "Osaka");
            assert_eq!(handled.output().forecast, "hooked:call-2:Osaka");
            assert_eq!(
                handled.clone().into_tool_result().unwrap(),
                ToolResult::new(
                    "call-2",
                    "weather",
                    RawJson::parse("{\"city\":\"Osaka\"}").unwrap(),
                    RawJson::parse("{\"forecast\":\"hooked:call-2:Osaka\"}").unwrap(),
                )
            );
        }
        ToolHookOutcome::Handled(other) => panic!("unexpected handled variant: {other:?}"),
        ToolHookOutcome::Unhandled(_) => panic!("hook should have handled the call"),
    }
}

#[test]
fn tool_round_commit_accepts_typed_handled_values() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-tool-hook".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::ToolCallChunk {
            id: "call-3".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Nagoya\"}".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-tool-hook".into()),
            finish_reason: FinishReason::ToolCall,
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
    let mut session = Session::new(ctx);
    session.push_user("Check weather.");
    let before_len = session.input().items().len();

    let outcome = block_on(async {
        session
            .text_turn()
            .tools::<Tools>()
            .available_tools(vec![ToolsSelector::Weather])
            .collect()
            .await
            .unwrap()
    });
    let hooks =
        ToolsHooks::new().with_weather(|_metadata: &lutum::ToolMetadata, input: &WeatherArgs| {
            let city = input.city.clone();
            async move {
                Some(WeatherResult {
                    forecast: format!("hooked:{city}"),
                })
            }
        });

    match outcome {
        TextStepOutcomeWithTools::NeedsTools(round) => {
            let handled = round
                .tool_calls
                .iter()
                .cloned()
                .map(|call| match block_on(call.hook(&hooks)) {
                    ToolHookOutcome::Handled(handled) => handled,
                    ToolHookOutcome::Unhandled(call) => {
                        panic!("tool hook unexpectedly returned unhandled: {call:?}")
                    }
                })
                .collect::<Vec<_>>();
            round.commit(&mut session, handled).unwrap();
        }
        TextStepOutcomeWithTools::Finished(_) => panic!("expected tool round"),
    }

    assert_eq!(session.input().items().len(), before_len + 2);
    let tool_result = session
        .input()
        .items()
        .iter()
        .find_map(|item| match item {
            ModelInputItem::ToolResult(result) => Some(result.clone()),
            _ => None,
        })
        .expect("tool result should be committed");
    assert_eq!(
        tool_result,
        ToolResult::new(
            "call-3",
            "weather",
            RawJson::parse("{\"city\":\"Nagoya\"}").unwrap(),
            RawJson::parse("{\"forecast\":\"hooked:Nagoya\"}").unwrap(),
        )
    );
}

/// Helper: make a mock adapter that emits a single disallowed (search) tool call.
fn make_disallowed_tool_adapter() -> MockLlmAdapter {
    MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("req-invalid".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "call-bad".into(),
            name: "search".into(),
            arguments_json_delta: "{\"query\":\"secret\"}".into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("req-invalid".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage::zero(),
        }),
    ]))
}

// Regression test for tool policy bypass:
// LLM が availability 制限外のツールを返した場合、tool_calls には届かず
// invalid_tool_calls に収集されることを確認する。
#[test]
fn invalid_tool_call_is_rejected_and_reported() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("req-bypass".into()),
            model: "gpt-4.1-mini".into(),
        }),
        // LLM が（ハルシネーション等で）許可されていない search を呼び出す
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "call-x".into(),
            name: "search".into(),
            arguments_json_delta: "{\"query\":\"secret data\"}".into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("req-bypass".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage::zero(),
        }),
    ]));

    let ctx = Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new(ctx);
    session.push_user("Search something.");

    // Weather のみ available に制限（Search の定義は LLM に送られない）
    let outcome = block_on(
        session
            .text_turn()
            .tools::<Tools>()
            .available_tools(vec![ToolsSelector::Weather])
            .collect(),
    )
    .unwrap();

    match outcome {
        TextStepOutcomeWithTools::NeedsTools(round) => {
            // 有効な tool call は届かない
            assert!(
                round.tool_calls.is_empty(),
                "no valid tool calls should reach user code, got: {:?}",
                round.tool_calls,
            );
            // 不正な tool call は invalid_tool_calls に収集される
            assert_eq!(round.invalid_tool_calls.len(), 1);
            assert_eq!(round.invalid_tool_calls[0].name.as_str(), "search");
        }
        TextStepOutcomeWithTools::Finished(_) => {
            panic!("expected NeedsTools (with invalid_tool_calls) but got Finished");
        }
    }
}

// Handler that records the names of InvalidToolCallChunk and InvalidToolCall events it sees.
struct RecordInvalidEvents(Arc<Mutex<Vec<String>>>);

#[async_trait]
impl EventHandler<TextTurnEventWithTools<Tools>, TextTurnStateWithTools<Tools>>
    for RecordInvalidEvents
{
    type Error = std::convert::Infallible;

    async fn on_event(
        &mut self,
        event: &TextTurnEventWithTools<Tools>,
        _cx: &HandlerContext<TextTurnStateWithTools<Tools>>,
    ) -> Result<HandlerDirective, Self::Error> {
        match event {
            TextTurnEventWithTools::InvalidToolCallChunk { name, .. } => {
                self.0.lock().unwrap().push(format!("chunk:{}", name));
            }
            TextTurnEventWithTools::InvalidToolCall(metadata) => {
                self.0.lock().unwrap().push(format!("call:{}", metadata.name));
            }
            _ => {}
        }
        Ok(HandlerDirective::Continue)
    }
}

// Verify that InvalidToolCallChunk and InvalidToolCall events reach the collect_with handler.
#[test]
fn invalid_tool_events_are_visible_in_collect_with_handler() {
    let ctx = Lutum::new(
        Arc::new(make_disallowed_tool_adapter()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let recorded: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let handler = RecordInvalidEvents(Arc::clone(&recorded));

    let pending = block_on(
        ctx.text_turn(vec![ModelInputItem::text(lutum::InputMessageRole::User, "go")].into())
            .tools::<Tools>()
            .available_tools(vec![ToolsSelector::Weather])
            .start(),
    )
    .unwrap();

    let staged = block_on(pending.collect_with(handler)).unwrap();
    staged.turn.discard();

    let seen = recorded.lock().unwrap();
    // Level 1: InvalidToolCallChunk fired for the disallowed chunk
    assert!(
        seen.contains(&"chunk:search".to_string()),
        "expected InvalidToolCallChunk for 'search', got: {seen:?}",
    );
    // Level 2: InvalidToolCall fired after full assembly
    assert!(
        seen.contains(&"call:search".to_string()),
        "expected InvalidToolCall for 'search', got: {seen:?}",
    );
}

// Verify that InvalidToolCallChunk and InvalidToolCall events appear in the raw event stream
// when consuming via into_stream() without collect/collect_with.
#[test]
fn invalid_tool_events_are_visible_in_raw_stream() {
    let ctx = Lutum::new(
        Arc::new(make_disallowed_tool_adapter()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );

    let pending = block_on(
        ctx.text_turn(vec![ModelInputItem::text(lutum::InputMessageRole::User, "go")].into())
            .tools::<Tools>()
            .available_tools(vec![ToolsSelector::Weather])
            .start(),
    )
    .unwrap();

    let stream = pending.into_stream();
    let events: Vec<_> = block_on(stream.collect());

    let has_invalid_chunk = events.iter().any(|r| {
        matches!(
            r,
            Ok(TextTurnEventWithTools::InvalidToolCallChunk { name, .. }) if name.as_str() == "search"
        )
    });
    let has_invalid_call = events.iter().any(|r| {
        matches!(
            r,
            Ok(TextTurnEventWithTools::InvalidToolCall(meta)) if meta.name.as_str() == "search"
        )
    });

    assert!(has_invalid_chunk, "expected InvalidToolCallChunk in stream, got: {events:?}");
    assert!(has_invalid_call, "expected InvalidToolCall in stream, got: {events:?}");
}
