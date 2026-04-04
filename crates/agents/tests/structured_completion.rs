use agents::{
    CollectError, FinishReason, MockLlmAdapter, MockStructuredCompletionScenario,
    RawStructuredCompletionEvent, RequestExtensions, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, StructuredCompletionReductionError, StructuredCompletionRequest,
    StructuredTurnOutcome, Usage, UsageEstimate,
};
use futures::executor::block_on;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Contact {
    email: String,
}

#[test]
fn structured_completion_collects_structured_output() {
    let adapter = MockLlmAdapter::new().with_structured_completion_scenario(
        MockStructuredCompletionScenario::events(vec![
            Ok(RawStructuredCompletionEvent::Started {
                request_id: Some("req-structured-completion".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(RawStructuredCompletionEvent::StructuredOutputChunk {
                json_delta: "{\"email\":\"user@example.com\"}".into(),
            }),
            Ok(RawStructuredCompletionEvent::Completed {
                request_id: Some("req-structured-completion".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 8,
                    ..Usage::zero()
                },
            }),
        ]),
    );
    let adapter = Arc::new(adapter);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = agents::Context::from_parts(adapter.clone(), adapter.clone(), adapter, budget);

    let result = block_on(async {
        ctx.structured_completion(
            RequestExtensions::new(),
            StructuredCompletionRequest::<Contact>::new(
                agents::ModelName::new("gpt-4.1-mini").unwrap(),
                "Extract the email address.",
            ),
            UsageEstimate::zero(),
        )
        .await
        .unwrap()
        .collect_noop()
        .await
        .unwrap()
    });

    assert!(matches!(
        result.semantic,
        StructuredTurnOutcome::Structured(Contact { ref email }) if email == "user@example.com"
    ));
}

#[test]
fn structured_completion_collects_refusal() {
    let adapter = MockLlmAdapter::new().with_structured_completion_scenario(
        MockStructuredCompletionScenario::events(vec![
            Ok(RawStructuredCompletionEvent::Started {
                request_id: Some("req-structured-refusal".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(RawStructuredCompletionEvent::RefusalDelta {
                delta: "cannot comply".into(),
            }),
            Ok(RawStructuredCompletionEvent::Completed {
                request_id: Some("req-structured-refusal".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 5,
                    ..Usage::zero()
                },
            }),
        ]),
    );
    let adapter = Arc::new(adapter);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = agents::Context::from_parts(adapter.clone(), adapter.clone(), adapter, budget);

    let result = block_on(async {
        ctx.structured_completion(
            RequestExtensions::new(),
            StructuredCompletionRequest::<Contact>::new(
                agents::ModelName::new("gpt-4.1-mini").unwrap(),
                "Extract the email address.",
            ),
            UsageEstimate::zero(),
        )
        .await
        .unwrap()
        .collect_noop()
        .await
        .unwrap()
    });

    assert_eq!(
        result.semantic,
        StructuredTurnOutcome::Refusal("cannot comply".into())
    );
}

#[test]
fn structured_completion_requires_semantic_output() {
    let adapter = MockLlmAdapter::new().with_structured_completion_scenario(
        MockStructuredCompletionScenario::events(vec![
            Ok(RawStructuredCompletionEvent::Started {
                request_id: Some("req-structured-empty".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(RawStructuredCompletionEvent::Completed {
                request_id: Some("req-structured-empty".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 3,
                    ..Usage::zero()
                },
            }),
        ]),
    );
    let adapter = Arc::new(adapter);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = agents::Context::from_parts(adapter.clone(), adapter.clone(), adapter, budget);

    let err = block_on(async {
        ctx.structured_completion(
            RequestExtensions::new(),
            StructuredCompletionRequest::<Contact>::new(
                agents::ModelName::new("gpt-4.1-mini").unwrap(),
                "Extract the email address.",
            ),
            UsageEstimate::zero(),
        )
        .await
        .unwrap()
        .collect_noop()
        .await
        .unwrap_err()
    });

    assert!(matches!(
        err,
        CollectError::Reduction {
            source: StructuredCompletionReductionError::MissingSemantic,
            ..
        }
    ));
}

#[test]
fn context_new_does_not_enable_structured_completion() {
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = agents::Context::new(Arc::new(MockLlmAdapter::new()), budget);

    let err = block_on(async {
        ctx.structured_completion(
            RequestExtensions::new(),
            StructuredCompletionRequest::<Contact>::new(
                agents::ModelName::new("gpt-4.1-mini").unwrap(),
                "Extract the email address.",
            ),
            UsageEstimate::zero(),
        )
        .await
    });

    let err = match err {
        Ok(_) => panic!("structured completion unexpectedly succeeded"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("completion adapter is not configured")
    );
}
