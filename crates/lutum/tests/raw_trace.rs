use std::sync::Arc;

use async_trait::async_trait;
use futures::executor::block_on;

use lutum::{
    CollectError, CollectErrorKind, EventHandler, FinishReason, HandlerContext, HandlerDirective,
    InputMessageRole, Lutum, MockError, MockLlmAdapter, MockTextScenario, ModelInput,
    ModelInputItem, OperationKind, RawTelemetryConfig, RawTextTurnEvent, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, TextTurnEvent, TextTurnState, Usage,
};
use lutum_trace::RawTraceEntry;

fn input() -> ModelInput {
    ModelInput::from_items(vec![ModelInputItem::text(InputMessageRole::User, "hello")])
}

fn test_budget() -> SharedPoolBudgetManager {
    SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default())
}

struct FailOnTextDelta;

#[async_trait]
impl EventHandler<TextTurnEvent, TextTurnState> for FailOnTextDelta {
    type Error = MockError;

    async fn on_event(
        &mut self,
        event: &TextTurnEvent,
        _cx: &HandlerContext<TextTurnState>,
    ) -> Result<HandlerDirective, Self::Error> {
        if matches!(event, TextTurnEvent::TextDelta { .. }) {
            Err(MockError::Synthetic {
                message: "handler failed".into(),
            })
        } else {
            Ok(HandlerDirective::Continue)
        }
    }
}

fn assert_collect_error(
    entries: &[RawTraceEntry],
    expected_kind: CollectErrorKind,
    expected_request_id: Option<&str>,
) {
    assert_eq!(entries.len(), 1);
    match &entries[0] {
        RawTraceEntry::CollectError {
            operation_kind,
            request_id,
            kind,
            partial_summary,
            error,
        } => {
            assert_eq!(*operation_kind, OperationKind::TextTurn);
            assert_eq!(request_id.as_deref(), expected_request_id);
            assert_eq!(*kind, expected_kind);
            assert!(
                partial_summary.contains("request_id"),
                "partial summary should include request context: {partial_summary}"
            );
            assert!(
                !error.is_empty(),
                "raw collect error should carry an error string"
            );
        }
        other => panic!("expected collect error entry, got {other:?}"),
    }
}

#[test]
fn capture_raw_records_execution_collect_errors() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("req-execution".into()),
            model: "gpt-4.1".into(),
        }),
        Err(MockError::Synthetic {
            message: "backend failed".into(),
        }),
    ]));

    let collected = block_on(lutum_trace::test::collect_raw(async move {
        let ctx =
            Lutum::new(Arc::new(adapter), test_budget()).with_extension(RawTelemetryConfig::all());
        ctx.text_turn(input()).collect().await
    }));

    assert!(matches!(
        collected.output,
        Err(CollectError::Execution { .. })
    ));
    assert_collect_error(
        &collected.raw.entries,
        CollectErrorKind::Execution,
        Some("req-execution"),
    );
}

#[test]
fn capture_raw_records_pre_stream_execution_collect_errors() {
    let collected = block_on(lutum_trace::test::collect_raw(async move {
        let ctx = Lutum::new(Arc::new(MockLlmAdapter::new()), test_budget())
            .with_extension(RawTelemetryConfig::all());
        ctx.text_turn(input()).collect().await
    }));

    assert!(matches!(
        collected.output,
        Err(CollectError::Execution { .. })
    ));
    assert_collect_error(&collected.raw.entries, CollectErrorKind::Execution, None);
    match &collected.raw.entries[0] {
        RawTraceEntry::CollectError {
            partial_summary,
            error,
            ..
        } => {
            assert!(partial_summary.contains("stream_started=false"));
            assert!(error.contains("no mock text scenario configured"));
        }
        other => panic!("expected collect error entry, got {other:?}"),
    }
}

#[test]
fn capture_raw_records_reduction_collect_errors() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("req-reduction".into()),
            model: "gpt-4.1".into(),
        }),
        Ok(RawTextTurnEvent::Completed {
            request_id: Some("req-reduction".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 3,
                ..Usage::zero()
            },
        }),
    ]));

    let collected = block_on(lutum_trace::test::collect_raw(async move {
        let ctx =
            Lutum::new(Arc::new(adapter), test_budget()).with_extension(RawTelemetryConfig::all());
        ctx.text_turn(input()).collect().await
    }));

    assert!(matches!(
        collected.output,
        Err(CollectError::Reduction { .. })
    ));
    assert_collect_error(
        &collected.raw.entries,
        CollectErrorKind::Reduction,
        Some("req-reduction"),
    );
}

#[test]
fn capture_raw_records_handler_collect_errors() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(RawTextTurnEvent::Started {
            request_id: Some("req-handler".into()),
            model: "gpt-4.1".into(),
        }),
        Ok(RawTextTurnEvent::TextDelta {
            delta: "hello".into(),
        }),
    ]));

    let collected = block_on(lutum_trace::test::collect_raw(async move {
        let ctx =
            Lutum::new(Arc::new(adapter), test_budget()).with_extension(RawTelemetryConfig::all());
        let pending = ctx.text_turn(input()).start().await.unwrap();
        pending.collect_with(FailOnTextDelta).await
    }));

    assert!(matches!(
        collected.output,
        Err(CollectError::Handler { .. })
    ));
    assert_collect_error(
        &collected.raw.entries,
        CollectErrorKind::Handler,
        Some("req-handler"),
    );
}

#[test]
fn capture_raw_records_unexpected_eof_collect_errors() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![Ok(
        RawTextTurnEvent::Started {
            request_id: Some("req-eof".into()),
            model: "gpt-4.1".into(),
        },
    )]));

    let collected = block_on(lutum_trace::test::collect_raw(async move {
        let ctx =
            Lutum::new(Arc::new(adapter), test_budget()).with_extension(RawTelemetryConfig::all());
        ctx.text_turn(input()).collect().await
    }));

    assert!(matches!(
        collected.output,
        Err(CollectError::UnexpectedEof { .. })
    ));
    assert_collect_error(
        &collected.raw.entries,
        CollectErrorKind::UnexpectedEof,
        Some("req-eof"),
    );
}

#[test]
fn request_extensions_can_disable_lutum_default_raw_telemetry() {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![Ok(
        RawTextTurnEvent::Started {
            request_id: Some("req-disabled".into()),
            model: "gpt-4.1".into(),
        },
    )]));

    let collected = block_on(lutum_trace::test::collect_raw(async move {
        let ctx =
            Lutum::new(Arc::new(adapter), test_budget()).with_extension(RawTelemetryConfig::all());
        ctx.text_turn(input())
            .ext(RawTelemetryConfig::none())
            .collect()
            .await
    }));

    assert!(matches!(
        collected.output,
        Err(CollectError::UnexpectedEof { .. })
    ));
    assert!(collected.raw.entries.is_empty());
}
