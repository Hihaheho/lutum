use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::{StreamExt, executor::block_on};
use lutum::{
    EventHandler, FinishReason, HandlerContext, HandlerDirective, Lutum, MockLlmAdapter,
    MockTextScenario, ModelInputItem, RawJson, RawTextTurnEvent, Session, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, TextStepOutcomeWithTools, TextTurnEventWithTools,
    TextTurnStateWithTools, ToolDecision, ToolHookOutcome, ToolMetadata, ToolResult, ToolRoundPlan,
    Toolset, Usage,
};
use lutum_trace::FieldValue;
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

// ── impl_hook definitions ─────────────────────────────────────────────────────

/// Returns "hooked:{id}:{city}" — used to verify metadata and input are forwarded.
#[lutum::impl_hook(WeatherHook)]
async fn hooked_forecast(
    metadata: &lutum::ToolMetadata,
    input: WeatherArgs,
) -> ToolDecision<WeatherArgs, WeatherResult> {
    let id = metadata.id.as_str().to_owned();
    let city = input.city;
    ToolDecision::Complete(WeatherResult {
        forecast: format!("hooked:{id}:{city}"),
    })
}

/// Returns "hooked:{city}" — plain weather hook without metadata inspection.
#[lutum::impl_hook(WeatherHook)]
async fn weather_hook_plain(
    _metadata: &lutum::ToolMetadata,
    input: WeatherArgs,
) -> ToolDecision<WeatherArgs, WeatherResult> {
    let city = input.city;
    ToolDecision::Complete(WeatherResult {
        forecast: format!("hooked:{city}"),
    })
}

/// Returns "cached:{city}" — simulates a cache hit.
#[lutum::impl_hook(WeatherHook)]
async fn cached_weather_hook(
    _metadata: &lutum::ToolMetadata,
    input: WeatherArgs,
) -> ToolDecision<WeatherArgs, WeatherResult> {
    let city = input.city;
    ToolDecision::Complete(WeatherResult {
        forecast: format!("cached:{city}"),
    })
}

/// First-pass weather hook for multi-pass chaining test.
#[lutum::impl_hook(WeatherHook)]
async fn pass1_weather(
    _metadata: &lutum::ToolMetadata,
    input: WeatherArgs,
) -> ToolDecision<WeatherArgs, WeatherResult> {
    let city = input.city;
    ToolDecision::Complete(WeatherResult {
        forecast: format!("pass1:{city}"),
    })
}

/// Second-pass search hook for multi-pass chaining test.
#[lutum::impl_hook(SearchHook)]
async fn pass2_search(
    _metadata: &lutum::ToolMetadata,
    input: SearchArgs,
) -> ToolDecision<SearchArgs, SearchResult> {
    let q = input.query;
    ToolDecision::Complete(SearchResult {
        hits: vec![format!("pass2:{q}")],
    })
}

#[lutum::impl_hook(WeatherHook)]
async fn rewrite_weather(
    _metadata: &lutum::ToolMetadata,
    input: WeatherArgs,
) -> ToolDecision<WeatherArgs, WeatherResult> {
    ToolDecision::RunNormally(WeatherArgs {
        city: format!("rewritten:{}", input.city),
    })
}

#[lutum::impl_hook(SearchHook)]
async fn reject_search(
    _metadata: &lutum::ToolMetadata,
    input: SearchArgs,
) -> ToolDecision<SearchArgs, SearchResult> {
    ToolDecision::Reject(format!("blocked query: {}", input.query))
}

// ─────────────────────────────────────────────────────────────────────────────

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
        ToolHookOutcome::Rejected(_) => panic!("hook should not have rejected the call"),
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
    let hooks = ToolsHooks::new().with_weather_hook(HookedForecast);

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
        ToolHookOutcome::Rejected(_) => panic!("hook should not have rejected the call"),
    }
}

#[test]
fn tool_call_hook_can_rewrite_runtime_input_without_touching_metadata() {
    let call = Tools::parse_tool_call(ToolMetadata::new(
        "call-rewrite",
        "weather",
        RawJson::parse("{\"city\":\"Sapporo\"}").unwrap(),
    ))
    .unwrap();
    let hooks = ToolsHooks::new().with_weather_hook(RewriteWeather);

    match block_on(call.hook(&hooks)) {
        ToolHookOutcome::Unhandled(ToolsCall::Weather(call)) => {
            assert_eq!(call.input().city, "rewritten:Sapporo");
            assert_eq!(call.metadata.arguments.get(), "{\"city\":\"Sapporo\"}");
        }
        other => panic!("expected rewritten unhandled call, got: {other:?}"),
    }
}

#[test]
fn tool_call_hook_can_reject_with_reason() {
    let call = Tools::parse_tool_call(ToolMetadata::new(
        "call-reject",
        "search",
        RawJson::parse("{\"query\":\"secret\"}").unwrap(),
    ))
    .unwrap();
    let hooks = ToolsHooks::new().with_search_hook(RejectSearch);

    match block_on(call.hook(&hooks)) {
        ToolHookOutcome::Rejected(rejected) => {
            assert_eq!(rejected.source(), lutum::RejectedToolSource::Hook);
            assert_eq!(rejected.metadata().id.as_str(), "call-reject");
            assert_eq!(rejected.reason(), "blocked query: secret");
            match rejected.call() {
                Some(ToolsCall::Search(call)) => assert_eq!(call.input().query, "secret"),
                other => panic!("unexpected rejected call payload: {other:?}"),
            }
        }
        other => panic!("expected rejected hook outcome, got: {other:?}"),
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
    let hooks = ToolsHooks::new().with_weather_hook(WeatherHookPlain);

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
                    ToolHookOutcome::Rejected(call) => {
                        panic!("tool hook unexpectedly returned rejected: {call:?}")
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

#[test]
fn tool_round_plan_commit_preserves_original_arguments_after_rewrite() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(lutum::RawTextTurnEvent::Started {
            request_id: Some("req-rewrite".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(lutum::RawTextTurnEvent::ToolCallChunk {
            id: "call-rewrite".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Sapporo\"}".into(),
        }),
        Ok(lutum::RawTextTurnEvent::Completed {
            request_id: Some("req-rewrite".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage::zero(),
        }),
    ]));
    let ctx = Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new(ctx);
    session.push_user("rewrite weather input");

    let outcome = block_on(
        session
            .text_turn()
            .tools::<Tools>()
            .available_tools(vec![ToolsSelector::Weather])
            .collect(),
    )
    .unwrap();
    let round = match outcome {
        TextStepOutcomeWithTools::NeedsTools(round) => round,
        TextStepOutcomeWithTools::Finished(_) => panic!("expected tool round"),
    };

    let plan = block_on(round.apply_hooks(&ToolsHooks::new().with_weather_hook(RewriteWeather)));

    let pending_results: Vec<_> = plan
        .pending
        .iter()
        .map(|call| match call {
            ToolsCall::Weather(call) => {
                assert_eq!(call.input().city, "rewritten:Sapporo");
                call.clone()
                    .complete(WeatherResult {
                        forecast: format!("executed:{}", call.input().city),
                    })
                    .unwrap()
            }
            other => panic!("unexpected pending variant: {other:?}"),
        })
        .collect();

    plan.commit(&mut session, pending_results).unwrap();

    let tool_result = session
        .input()
        .items()
        .iter()
        .find_map(|item| match item {
            ModelInputItem::ToolResult(result) => Some(result.clone()),
            _ => None,
        })
        .expect("tool result should be committed");
    assert_eq!(tool_result.arguments.get(), "{\"city\":\"Sapporo\"}");
    let weather_result: WeatherResult = tool_result.result.deserialize().unwrap();
    assert_eq!(weather_result.forecast, "executed:rewritten:Sapporo");
}

fn make_two_tool_adapter() -> MockLlmAdapter {
    MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("req-two".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "call-w".into(),
            name: "weather".into(),
            arguments_json_delta: "{\"city\":\"Kyoto\"}".into(),
        }),
        Ok(RawTextTurnEvent::ToolCallChunk {
            id: "call-s".into(),
            name: "search".into(),
            arguments_json_delta: "{\"query\":\"ramen\"}".into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("req-two".into()),
            finish_reason: FinishReason::ToolCall,
            usage: Usage::zero(),
        }),
    ]))
}

// apply_hooks moves hook-handled calls to `handled`, rejected calls to `rejected`,
// and leaves the rest in `pending`.
#[test]
fn apply_hooks_splits_handled_and_pending() {
    let ctx = Lutum::new(
        Arc::new(make_two_tool_adapter()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new(ctx);
    session.push_user("go");

    let outcome = block_on(
        session
            .text_turn()
            .tools::<Tools>()
            .available_tools(vec![ToolsSelector::Weather, ToolsSelector::Search])
            .collect(),
    )
    .unwrap();

    let round = match outcome {
        TextStepOutcomeWithTools::NeedsTools(r) => r,
        TextStepOutcomeWithTools::Finished(_) => panic!("expected NeedsTools"),
    };

    // Hook only weather; search stays pending.
    let hooks = ToolsHooks::new().with_weather_hook(CachedWeatherHook);

    let plan: ToolRoundPlan<Tools> = block_on(round.apply_hooks(&hooks));

    assert_eq!(plan.handled.len(), 1, "weather should be handled");
    assert_eq!(plan.pending.len(), 1, "search should be pending");
    assert_eq!(plan.rejected.len(), 0);

    match &plan.handled[0] {
        ToolsHandled::Weather(h) => assert_eq!(h.output().forecast, "cached:Kyoto"),
        other => panic!("unexpected handled variant: {other:?}"),
    }
    match &plan.pending[0] {
        ToolsCall::Search(c) => assert_eq!(c.input().query, "ramen"),
        other => panic!("unexpected pending variant: {other:?}"),
    }
}

// apply_hooks called twice (multi-pass) narrows pending each time.
#[test]
fn apply_hooks_multi_pass_chaining_narrows_pending() {
    let ctx = Lutum::new(
        Arc::new(make_two_tool_adapter()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new(ctx);
    session.push_user("go");

    let outcome = block_on(
        session
            .text_turn()
            .tools::<Tools>()
            .available_tools(vec![ToolsSelector::Weather, ToolsSelector::Search])
            .collect(),
    )
    .unwrap();

    let round = match outcome {
        TextStepOutcomeWithTools::NeedsTools(r) => r,
        TextStepOutcomeWithTools::Finished(_) => panic!("expected NeedsTools"),
    };

    let weather_hooks = ToolsHooks::new().with_weather_hook(Pass1Weather);
    let search_hooks = ToolsHooks::new().with_search_hook(Pass2Search);

    let plan = block_on(async {
        round
            .apply_hooks(&weather_hooks)
            .await
            .apply_hooks(&search_hooks)
            .await
    });

    assert_eq!(
        plan.handled.len(),
        2,
        "both calls should be handled after two passes"
    );
    assert_eq!(plan.pending.len(), 0, "nothing should remain pending");
}

// ToolRoundPlan::commit combines handled + pending results in AssistantTurn order.
#[test]
fn tool_round_plan_commit_merges_handled_and_pending_results() {
    let ctx = Lutum::new(
        Arc::new(make_two_tool_adapter()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new(ctx);
    session.push_user("go");
    let before_len = session.input().items().len();

    let outcome = block_on(
        session
            .text_turn()
            .tools::<Tools>()
            .available_tools(vec![ToolsSelector::Weather, ToolsSelector::Search])
            .collect(),
    )
    .unwrap();

    let round = match outcome {
        TextStepOutcomeWithTools::NeedsTools(r) => r,
        TextStepOutcomeWithTools::Finished(_) => panic!("expected NeedsTools"),
    };

    let hooks = ToolsHooks::new().with_weather_hook(WeatherHookPlain);

    let plan = block_on(round.apply_hooks(&hooks));

    // Execute the remaining pending call (search).
    let pending_results: Vec<ToolResult> = plan
        .pending
        .iter()
        .map(|call| match call {
            ToolsCall::Search(c) => SearchArgs::tool_result(
                c.metadata.clone(),
                SearchResult {
                    hits: vec!["ramen-shop".into()],
                },
            )
            .unwrap(),
            other => panic!("unexpected: {other:?}"),
        })
        .collect();

    plan.commit(&mut session, pending_results).unwrap();

    // assistant turn + 2 tool results = 3 items added
    assert_eq!(session.input().items().len(), before_len + 3);

    let tool_results: Vec<_> = session
        .input()
        .items()
        .iter()
        .filter_map(|item| match item {
            ModelInputItem::ToolResult(r) => Some(r.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_results.len(), 2);

    let weather_result: WeatherResult = tool_results
        .iter()
        .find(|r| r.name.as_str() == "weather")
        .expect("weather result")
        .result
        .deserialize()
        .unwrap();
    assert_eq!(weather_result.forecast, "hooked:Kyoto");

    let search_result: SearchResult = tool_results
        .iter()
        .find(|r| r.name.as_str() == "search")
        .expect("search result")
        .result
        .deserialize()
        .unwrap();
    assert_eq!(search_result.hits, vec!["ramen-shop"]);
}

#[test]
fn tool_round_plan_commit_auto_commits_rejected_calls() {
    let ctx = Lutum::new(
        Arc::new(make_two_tool_adapter()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new(ctx);
    session.push_user("go");

    let outcome = block_on(
        session
            .text_turn()
            .tools::<Tools>()
            .available_tools(vec![ToolsSelector::Weather, ToolsSelector::Search])
            .collect(),
    )
    .unwrap();

    let round = match outcome {
        TextStepOutcomeWithTools::NeedsTools(r) => r,
        TextStepOutcomeWithTools::Finished(_) => panic!("expected NeedsTools"),
    };

    let hooks = ToolsHooks::new().with_search_hook(RejectSearch);
    let plan = block_on(round.apply_hooks(&hooks));

    assert_eq!(plan.pending.len(), 1);
    assert_eq!(plan.rejected.len(), 1);
    assert_eq!(plan.rejected[0].source(), lutum::RejectedToolSource::Hook);
    assert_eq!(plan.rejected[0].reason(), "blocked query: ramen");

    let pending_results: Vec<_> = plan
        .pending
        .iter()
        .map(|call| match call {
            ToolsCall::Weather(c) => c
                .clone()
                .complete(WeatherResult {
                    forecast: "sunny".into(),
                })
                .unwrap(),
            other => panic!("unexpected pending call: {other:?}"),
        })
        .collect();
    plan.commit(&mut session, pending_results).unwrap();

    let tool_results: Vec<_> = session
        .input()
        .items()
        .iter()
        .filter_map(|item| match item {
            ModelInputItem::ToolResult(result) => Some(result.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_results.len(), 2);

    let rejected_result = tool_results
        .iter()
        .find(|result| result.name.as_str() == "search")
        .expect("search rejection result");
    assert_eq!(
        rejected_result.rejection_reason().as_deref(),
        Some("blocked query: ramen")
    );
}

// apply_hooks also accepts a closure directly via the blanket ToolHooks impl.
#[test]
fn apply_hooks_accepts_closure_via_blanket_impl() {
    let ctx = Lutum::new(
        Arc::new(make_two_tool_adapter()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new(ctx);
    session.push_user("go");

    let outcome = block_on(
        session
            .text_turn()
            .tools::<Tools>()
            .available_tools(vec![ToolsSelector::Weather, ToolsSelector::Search])
            .collect(),
    )
    .unwrap();

    let round = match outcome {
        TextStepOutcomeWithTools::NeedsTools(r) => r,
        TextStepOutcomeWithTools::Finished(_) => panic!("expected NeedsTools"),
    };

    // Pass a bare closure — exercising the blanket impl on Fn(T::ToolCall) -> Fut.
    let hook = |call| async move {
        match call {
            ToolsCall::Weather(c) => {
                let output = WeatherResult {
                    forecast: format!("closure:{}", c.input().city),
                };
                ToolHookOutcome::Handled(ToolsHandled::Weather(c.handled(output)))
            }
            other => ToolHookOutcome::Unhandled(other),
        }
    };
    let plan = block_on(round.apply_hooks(&hook));

    assert_eq!(plan.handled.len(), 1);
    assert_eq!(plan.pending.len(), 1);
    match &plan.handled[0] {
        ToolsHandled::Weather(h) => assert!(h.output().forecast.starts_with("closure:")),
        other => panic!("unexpected: {other:?}"),
    }
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

#[test]
fn invalid_tool_call_commit_auto_synthesizes_rejection_result() {
    let ctx = Lutum::new(
        Arc::new(make_disallowed_tool_adapter()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let mut session = Session::new(ctx);
    session.push_user("Search something.");

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
            assert!(round.tool_calls.is_empty());
            round
                .commit(&mut session, Vec::<ToolResult>::new())
                .unwrap();
        }
        TextStepOutcomeWithTools::Finished(_) => panic!("expected tool round"),
    }

    let rejected_result = session
        .input()
        .items()
        .iter()
        .find_map(|item| match item {
            ModelInputItem::ToolResult(result) => Some(result.clone()),
            _ => None,
        })
        .expect("policy rejection should be committed");
    assert_eq!(
        rejected_result.result.deserialize::<String>().unwrap(),
        "__lutum_rejected__: tool `search` is not available in this round"
    );
    assert_eq!(
        rejected_result.rejection_reason().as_deref(),
        Some("tool `search` is not available in this round")
    );
}

#[test]
fn rejection_reason_helper_only_matches_reserved_prefix() {
    let rejected = ToolResult::new(
        "call-rejected",
        "search",
        RawJson::parse("{\"query\":\"secret\"}").unwrap(),
        RawJson::from_serializable(&"__lutum_rejected__: blocked query: secret".to_string())
            .unwrap(),
    );
    assert_eq!(
        rejected.rejection_reason().as_deref(),
        Some("blocked query: secret")
    );

    let normal = ToolResult::new(
        "call-normal",
        "search",
        RawJson::parse("{\"query\":\"secret\"}").unwrap(),
        RawJson::from_serializable(&"plain string output".to_string()).unwrap(),
    );
    assert_eq!(normal.rejection_reason(), None);
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
                self.0
                    .lock()
                    .unwrap()
                    .push(format!("call:{}", metadata.name));
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

    assert!(
        has_invalid_chunk,
        "expected InvalidToolCallChunk in stream, got: {events:?}"
    );
    assert!(
        has_invalid_call,
        "expected InvalidToolCall in stream, got: {events:?}"
    );
}

#[tokio::test]
async fn tool_hook_trace_records_rewrite_and_reject_decisions() {
    let call = Tools::parse_tool_call(ToolMetadata::new(
        "call-trace",
        "search",
        RawJson::parse("{\"query\":\"secret\"}").unwrap(),
    ))
    .unwrap();
    let hooks = ToolsHooks::new().with_search_hook(RejectSearch);

    let collected = lutum_trace::test::collect(async move {
        let _ = call.hook(&hooks).await;
    })
    .await;

    let event = collected
        .trace
        .events()
        .iter()
        .find(|event| event.message() == Some("tool hook decision"))
        .expect("tool hook decision event");
    assert_eq!(
        event.field("decision"),
        Some(&FieldValue::Str("reject".into()))
    );
    assert_eq!(
        event.field("reason"),
        Some(&FieldValue::Str("blocked query: secret".into()))
    );

    let rewrite_call = Tools::parse_tool_call(ToolMetadata::new(
        "call-trace-rewrite",
        "weather",
        RawJson::parse("{\"city\":\"Nagoya\"}").unwrap(),
    ))
    .unwrap();
    let rewrite_hooks = ToolsHooks::new().with_weather_hook(RewriteWeather);

    let collected = lutum_trace::test::collect(async move {
        let _ = rewrite_call.hook(&rewrite_hooks).await;
    })
    .await;

    let event = collected
        .trace
        .events()
        .iter()
        .find(|event| event.message() == Some("tool hook decision"))
        .expect("rewrite tool hook decision event");
    assert_eq!(
        event.field("decision"),
        Some(&FieldValue::Str("run_normally".into()))
    );
    assert_eq!(
        event.field("effective_input_json"),
        Some(&FieldValue::Str("{\"city\":\"rewritten:Nagoya\"}".into()))
    );
}
