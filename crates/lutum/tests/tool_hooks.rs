use std::sync::Arc;

use futures::executor::block_on;
use lutum::{
    FinishReason, MockLlmAdapter, MockTextScenario, ModelInputItem, RawJson, Session,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, TextStepOutcomeWithTools, ToolHookOutcome,
    ToolMetadata, ToolResult, Toolset, Usage,
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
    let hooks = ToolsHooks::new().with_weather(|metadata: &lutum::ToolMetadata, input: &WeatherArgs| {
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
            .allow_only(vec![ToolsSelector::Weather])
            .collect()
            .await
            .unwrap()
    });
    let hooks = ToolsHooks::new().with_weather(|_metadata: &lutum::ToolMetadata, input: &WeatherArgs| {
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
