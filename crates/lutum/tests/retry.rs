use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use futures::StreamExt;
use lutum::{
    AssistantTurnItem, BackoffPolicy, BudgetManager, CollectError, EventHandler, FinishReason,
    HandlerContext, HandlerDirective, InputMessageRole, Lutum, MockError, MockLlmAdapter,
    MockStructuredScenario, MockTextScenario, ModelInput, ModelInputItem, RequestBudget,
    RequestExtensions, RetryPolicy, SharedPoolBudgetManager, SharedPoolBudgetOptions,
    StructuredStepOutcomeWithTools, StructuredTurnOutcome, TextTurnEvent, TextTurnState, Usage,
    UsageEstimate,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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

fn retry_policy(max_attempts: u32) -> RetryPolicy {
    RetryPolicy {
        max_attempts,
        backoff: BackoffPolicy {
            initial_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            multiplier: 1.0,
            jitter_factor: 0.0,
        },
    }
}

fn budget() -> SharedPoolBudgetManager {
    SharedPoolBudgetManager::new(SharedPoolBudgetOptions {
        capacity_tokens: 100,
        capacity_cost_micros_usd: 1_000,
        stop_threshold_tokens: 0,
        stop_threshold_cost_micros_usd: 0,
    })
}

fn usage_estimate(tokens: u64) -> UsageEstimate {
    UsageEstimate {
        total_tokens: tokens,
        ..UsageEstimate::zero()
    }
}

fn usage(tokens: u64) -> Usage {
    Usage {
        total_tokens: tokens,
        ..Usage::zero()
    }
}

fn shared_pool_budget_error(err: &lutum::AgentError) -> &lutum::SharedPoolBudgetError {
    match err {
        lutum::AgentError::Budget(source) => source
            .downcast_ref::<lutum::SharedPoolBudgetError>()
            .expect("shared pool budget error source"),
        other => panic!("expected budget error, got {other}"),
    }
}

struct StopOnWillRetry;

#[async_trait]
impl EventHandler<TextTurnEvent, TextTurnState> for StopOnWillRetry {
    type Error = std::convert::Infallible;

    async fn on_event(
        &mut self,
        event: &TextTurnEvent,
        _cx: &HandlerContext<TextTurnState>,
    ) -> Result<HandlerDirective, Self::Error> {
        Ok(if matches!(event, TextTurnEvent::WillRetry { .. }) {
            HandlerDirective::Stop
        } else {
            HandlerDirective::Continue
        })
    }
}

#[tokio::test]
async fn pre_stream_failure_emits_will_retry_before_started() {
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(MockTextScenario::start_error(MockError::retryable_server(
            "boom",
        )))
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(lutum::mock::RawTextTurnEvent::Started {
                request_id: Some("req-2".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(lutum::mock::RawTextTurnEvent::Completed {
                request_id: Some("req-2".into()),
                finish_reason: FinishReason::Stop,
                usage: usage(3),
            }),
        ]));
    let ctx = Lutum::new(Arc::new(adapter), budget());
    let pending = ctx
        .text_turn(input())
        .retry_policy(retry_policy(2))
        .start()
        .await
        .unwrap();

    let events = pending.into_stream().collect::<Vec<_>>().await;

    assert!(matches!(
        events.as_slice(),
        [
            Ok(TextTurnEvent::WillRetry {
                attempt: 2,
                request_id: None,
                ..
            }),
            Ok(TextTurnEvent::Started { request_id, .. }),
            Ok(TextTurnEvent::Completed { request_id: completed_request_id, usage, .. }),
        ] if request_id.as_deref() == Some("req-2")
            && completed_request_id.as_deref() == Some("req-2")
            && usage.total_tokens == 3
    ));
}

#[tokio::test]
async fn collect_discards_failed_attempt_output_after_retry() {
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(lutum::mock::RawTextTurnEvent::Started {
                request_id: Some("req-1".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(lutum::mock::RawTextTurnEvent::TextDelta {
                delta: "bad".into(),
            }),
            Err(MockError::retryable_server("boom")),
        ]))
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(lutum::mock::RawTextTurnEvent::Started {
                request_id: Some("req-2".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(lutum::mock::RawTextTurnEvent::TextDelta {
                delta: "good".into(),
            }),
            Ok(lutum::mock::RawTextTurnEvent::Completed {
                request_id: Some("req-2".into()),
                finish_reason: FinishReason::Stop,
                usage: usage(6),
            }),
        ]));
    let ctx = Lutum::new(Arc::new(adapter), budget());

    let result = ctx
        .text_turn(input())
        .retry_policy(retry_policy(2))
        .ext(usage_estimate(4))
        .collect()
        .await
        .unwrap();

    assert_eq!(result.assistant_text(), "good");
    assert_eq!(result.request_id.as_deref(), Some("req-2"));
    assert_eq!(result.usage.total_tokens, 6);
    assert_eq!(result.cumulative_usage.total_tokens, 10);
}

#[tokio::test]
async fn collect_with_stop_on_will_retry_returns_original_execution_error() {
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(lutum::mock::RawTextTurnEvent::Started {
                request_id: Some("req-stop".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(lutum::mock::RawTextTurnEvent::TextDelta {
                delta: "bad".into(),
            }),
            Err(MockError::retryable_server("boom")),
        ]))
        .with_text_scenario(MockTextScenario::events(vec![Ok(
            lutum::mock::RawTextTurnEvent::Completed {
                request_id: Some("req-unused".into()),
                finish_reason: FinishReason::Stop,
                usage: usage(1),
            },
        )]));
    let ctx = Lutum::new(Arc::new(adapter), budget());
    let pending = ctx
        .text_turn(input())
        .retry_policy(retry_policy(2))
        .ext(usage_estimate(4))
        .start()
        .await
        .unwrap();

    let err = pending.collect_with(StopOnWillRetry).await.unwrap_err();

    match err {
        CollectError::Execution { source, partial } => {
            let failure = source.request_failure().expect("request failure");
            assert_eq!(failure.kind, lutum::RequestFailureKind::Server);
            assert_eq!(failure.status, Some(500));
            assert!(matches!(
                partial.assistant_turn.as_slice(),
                [AssistantTurnItem::Text(text)] if text == "bad"
            ));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn retry_after_hint_overrides_backoff_delay() {
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(MockTextScenario::start_error(
            MockError::retryable_server_after("boom", Duration::from_millis(1)),
        ))
        .with_text_scenario(MockTextScenario::events(vec![Ok(
            lutum::mock::RawTextTurnEvent::Completed {
                request_id: Some("req-ok".into()),
                finish_reason: FinishReason::Stop,
                usage: usage(1),
            },
        )]));
    let ctx = Lutum::new(Arc::new(adapter), budget());
    let pending = ctx
        .text_turn(input())
        .retry_policy(RetryPolicy {
            max_attempts: 2,
            backoff: BackoffPolicy {
                initial_delay: Duration::ZERO,
                max_delay: Duration::ZERO,
                multiplier: 1.0,
                jitter_factor: 0.0,
            },
        })
        .start()
        .await
        .unwrap();

    let events = pending.into_stream().collect::<Vec<_>>().await;

    assert!(matches!(
        events.first(),
        Some(Ok(TextTurnEvent::WillRetry { after, .. })) if *after == Duration::from_millis(1)
    ));
}

#[tokio::test]
async fn recover_usage_error_falls_back_to_estimate_for_retry_accounting() {
    let budget = budget();
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(lutum::mock::RawTextTurnEvent::Started {
                request_id: Some("req-recover".into()),
                model: "gpt-4.1".into(),
            }),
            Err(MockError::retryable_server("boom")),
        ]))
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(lutum::mock::RawTextTurnEvent::Started {
                request_id: Some("req-final".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(lutum::mock::RawTextTurnEvent::TextDelta { delta: "ok".into() }),
            Ok(lutum::mock::RawTextTurnEvent::Completed {
                request_id: Some("req-final".into()),
                finish_reason: FinishReason::Stop,
                usage: usage(3),
            }),
        ]))
        .with_recover_usage_error(
            lutum::OperationKind::TextTurn,
            "req-recover",
            MockError::Synthetic {
                message: "recovery failed".into(),
            },
        );
    let ctx = Lutum::new(Arc::new(adapter), budget.clone());

    let result = ctx
        .text_turn(input())
        .retry_policy(retry_policy(2))
        .ext(usage_estimate(11))
        .collect()
        .await
        .unwrap();

    assert_eq!(result.cumulative_usage.total_tokens, 14);
    assert_eq!(budget.remaining(&RequestExtensions::new()).tokens, 86);
}

#[tokio::test]
async fn structured_tool_and_output_state_resets_on_retry() {
    let adapter = MockLlmAdapter::new()
        .with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(lutum::mock::RawStructuredTurnEvent::Started {
                request_id: Some("req-structured-1".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(lutum::mock::RawStructuredTurnEvent::StructuredOutputChunk {
                json_delta: "{\"answer\":\"bad\"}".into(),
            }),
            Ok(lutum::mock::RawStructuredTurnEvent::ToolCallChunk {
                id: "call-1".into(),
                name: "weather".into(),
                arguments_json_delta: "{\"city\":\"Tokyo\"}".into(),
            }),
            Err(MockError::retryable_server("boom")),
        ]))
        .with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(lutum::mock::RawStructuredTurnEvent::Started {
                request_id: Some("req-structured-2".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(lutum::mock::RawStructuredTurnEvent::StructuredOutputChunk {
                json_delta: "{\"answer\":\"good\"}".into(),
            }),
            Ok(lutum::mock::RawStructuredTurnEvent::Completed {
                request_id: Some("req-structured-2".into()),
                finish_reason: FinishReason::Stop,
                usage: usage(5),
            }),
        ]));
    let ctx = Lutum::new(Arc::new(adapter), budget());

    let result = ctx
        .structured_turn::<Summary>(input())
        .tools::<Tools>()
        .available_tools(vec![ToolsSelector::Weather])
        .retry_policy(retry_policy(2))
        .ext(usage_estimate(4))
        .collect()
        .await
        .unwrap();

    let result = match result {
        StructuredStepOutcomeWithTools::Finished(result) => result,
        StructuredStepOutcomeWithTools::NeedsTools(_) => panic!("unexpected tool round"),
    };

    assert!(matches!(
        result.semantic,
        StructuredTurnOutcome::Structured(Summary { ref answer }) if answer == "good"
    ));
    assert!(result.tool_calls.is_empty());
    assert!(result.recoverable_tool_call_issues.is_empty());
    assert!(matches!(
        result.assistant_turn.items(),
        [AssistantTurnItem::Text(text)] if text == "{\"answer\":\"good\"}"
    ));
}

#[tokio::test]
async fn cumulative_request_budget_is_enforced_across_attempts() {
    let adapter = MockLlmAdapter::new()
        .with_text_scenario(MockTextScenario::start_error(MockError::retryable_server(
            "boom",
        )))
        .with_text_scenario(MockTextScenario::events(vec![
            Ok(lutum::mock::RawTextTurnEvent::Started {
                request_id: Some("req-budget".into()),
                model: "gpt-4.1".into(),
            }),
            Ok(lutum::mock::RawTextTurnEvent::Completed {
                request_id: Some("req-budget".into()),
                finish_reason: FinishReason::Stop,
                usage: usage(8),
            }),
        ]));
    let ctx = Lutum::new(Arc::new(adapter), budget());

    let err = ctx
        .text_turn(input())
        .budget(RequestBudget::from_tokens(10))
        .retry_policy(retry_policy(2))
        .ext(usage_estimate(4))
        .collect()
        .await
        .unwrap_err();

    match err {
        CollectError::Execution { source, partial } => {
            assert_eq!(partial.request_id.as_deref(), Some("req-budget"));
            assert!(matches!(
                shared_pool_budget_error(&source),
                lutum::SharedPoolBudgetError::RequestBudgetExceeded {
                    requested_tokens: 12,
                    budget_tokens: Some(10),
                    ..
                }
            ));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn default_retry_policy_is_disabled() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::start_error(
        MockError::retryable_server("boom"),
    ));
    let ctx = Lutum::new(Arc::new(adapter), budget());
    let pending = ctx.text_turn(input()).start().await.unwrap();

    let err = pending.collect().await.unwrap_err();

    match err {
        CollectError::Execution { source, .. } => {
            let failure = source.request_failure().expect("request failure");
            assert_eq!(failure.kind, lutum::RequestFailureKind::Server);
            assert_eq!(failure.status, Some(500));
        }
        other => panic!("unexpected error: {other:?}"),
    }
}
