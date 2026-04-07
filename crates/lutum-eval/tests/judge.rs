use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use lutum::{
    FinishReason, Lutum, MockLlmAdapter, MockStructuredScenario, ModelInput,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, Usage,
};
use lutum_eval::{EvalExt, JudgeEval, JudgeEvalError, Score, maximize};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Verdict {
    score: u8,
    passed: bool,
}

#[derive(Debug)]
struct Draft {
    text: String,
}

fn make_context(adapter: MockLlmAdapter) -> Lutum {
    Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    )
}

#[tokio::test]
async fn judge_eval_returns_a_structured_report() {
    let adapter =
        MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(lutum::RawStructuredTurnEvent::Started {
                request_id: Some("judge-1".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(lutum::RawStructuredTurnEvent::StructuredOutputChunk {
                json_delta: "{\"score\":8,\"passed\":true}".into(),
            }),
            Ok(lutum::RawStructuredTurnEvent::Completed {
                request_id: Some("judge-1".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 6,
                    ..Usage::zero()
                },
            }),
        ]));
    let ctx = make_context(adapter);
    let saw_trace = Arc::new(AtomicBool::new(false));
    let trace_flag = saw_trace.clone();
    let eval = JudgeEval::<Draft, Verdict, _>::new(
        move |trace: &lutum_eval::TraceSnapshot, artifact: &Draft| {
            trace_flag.store(trace.has_event_message("judge-trace"), Ordering::SeqCst);
            ModelInput::new()
                .system("Judge the draft.")
                .user(format!("Draft: {}", artifact.text))
        },
    );
    let collected = lutum_trace::test::collect(async {
        tracing::info!("judge-trace");
        Draft {
            text: "ship it".into(),
        }
    })
    .await;

    let report = eval.run_collected(&ctx, &collected).await.unwrap();

    assert_eq!(
        report,
        Verdict {
            score: 8,
            passed: true,
        }
    );
    assert!(saw_trace.load(Ordering::SeqCst));
}

#[tokio::test]
async fn judge_eval_surfaces_structured_refusals() {
    let adapter =
        MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(lutum::RawStructuredTurnEvent::Started {
                request_id: Some("judge-refusal".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(lutum::RawStructuredTurnEvent::RefusalDelta {
                delta: "cannot comply".into(),
            }),
            Ok(lutum::RawStructuredTurnEvent::Completed {
                request_id: Some("judge-refusal".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 4,
                    ..Usage::zero()
                },
            }),
        ]));
    let ctx = make_context(adapter);
    let eval = JudgeEval::<Draft, Verdict, _>::new(
        |_trace: &lutum_eval::TraceSnapshot, artifact: &Draft| {
            ModelInput::new().user(format!("Draft: {}", artifact.text))
        },
    );
    let collected = lutum_trace::test::collect(async {
        Draft {
            text: "blocked".into(),
        }
    })
    .await;

    let error = eval.run_collected(&ctx, &collected).await.unwrap_err();

    assert!(matches!(
        error,
        JudgeEvalError::Refusal { reason } if reason == "cannot comply"
    ));
}

#[tokio::test]
async fn judge_reports_can_be_scored_by_an_objective() {
    let adapter =
        MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(lutum::RawStructuredTurnEvent::Started {
                request_id: Some("judge-score".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(lutum::RawStructuredTurnEvent::StructuredOutputChunk {
                json_delta: "{\"score\":6,\"passed\":true}".into(),
            }),
            Ok(lutum::RawStructuredTurnEvent::Completed {
                request_id: Some("judge-score".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 6,
                    ..Usage::zero()
                },
            }),
        ]));
    let ctx = make_context(adapter);
    let eval = JudgeEval::<Draft, Verdict, _>::new(
        |_trace: &lutum_eval::TraceSnapshot, artifact: &Draft| {
            ModelInput::new().user(format!("Draft: {}", artifact.text))
        },
    );
    let objective = maximize(|report: &Verdict| Score::new_clamped(report.score as f32 / 10.0));
    let collected = lutum_trace::test::collect(async {
        Draft {
            text: "scored".into(),
        }
    })
    .await;

    let scored = eval
        .scored_by(&objective)
        .run_collected(&ctx, &collected)
        .await
        .unwrap();

    assert_eq!(scored.score, Score::new_clamped(0.6));
    assert_eq!(scored.report.score, 6);
}
