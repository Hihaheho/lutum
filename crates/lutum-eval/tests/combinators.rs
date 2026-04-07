use async_trait::async_trait;
use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
use lutum_eval::{CombineError, Eval, EvalExt, PureEval, PureEvalExt};

#[derive(Debug)]
struct Draft {
    text: String,
}

#[derive(Debug)]
struct Envelope {
    draft: Draft,
}

fn envelope_draft(artifact: &Envelope) -> &Draft {
    &artifact.draft
}

struct WordCount;

impl PureEval for WordCount {
    type Artifact = Draft;
    type Report = usize;
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        _trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(artifact.text.split_whitespace().count())
    }
}

struct SawTraceMessage(&'static str);

impl PureEval for SawTraceMessage {
    type Artifact = Draft;
    type Report = bool;
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        trace: &lutum_eval::TraceSnapshot,
        _artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(trace.has_event_message(self.0))
    }
}

#[derive(Debug, Eq, PartialEq)]
enum FailError {
    MissingReport,
}

struct FailingEval;

#[async_trait]
impl Eval for FailingEval {
    type Artifact = Draft;
    type Report = usize;
    type Error = FailError;

    async fn evaluate(
        &self,
        _ctx: &Lutum,
        _trace: &lutum_eval::TraceSnapshot,
        _artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Err(FailError::MissingReport)
    }
}

fn make_context() -> Lutum {
    Lutum::new(
        std::sync::Arc::new(MockLlmAdapter::new()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    )
}

#[tokio::test]
async fn map_report_transforms_the_eval_output() {
    let collected = lutum_trace::test::collect(async {
        Draft {
            text: "hello world".into(),
        }
    })
    .await;

    let eval = WordCount.map_report(|count| count * 10);
    let report = eval.run_collected(&collected).unwrap();

    assert_eq!(report, 20);
}

#[tokio::test]
async fn contramap_artifact_projects_a_larger_input_shape() {
    let collected = lutum_trace::test::collect(async {
        Envelope {
            draft: Draft {
                text: "one two three".into(),
            },
        }
    })
    .await;

    let eval = WordCount.contramap_artifact::<Envelope, _>(envelope_draft);
    let report = eval.run_collected(&collected).unwrap();

    assert_eq!(report, 3);
}

#[tokio::test]
async fn combine_allows_custom_report_aggregation() {
    let eval = WordCount.combine(SawTraceMessage("combine-trace"), |count, saw_trace| {
        (count, saw_trace)
    });
    let collected = lutum_trace::test::collect(async {
        tracing::info!("combine-trace");
        Draft {
            text: "alpha beta".into(),
        }
    })
    .await;

    let report = eval.run_collected(&collected).unwrap();

    assert_eq!(report, (2, true));
}

#[tokio::test]
async fn map_error_rewrites_eval_failures() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async {
        Draft {
            text: "ignored".into(),
        }
    })
    .await;

    let eval = FailingEval.map_error(|error| match error {
        FailError::MissingReport => "rewritten",
    });
    let error = eval.run_collected(&ctx, &collected).await.unwrap_err();

    assert_eq!(error, "rewritten");
}

#[tokio::test]
async fn combine_surfaces_which_side_failed() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async {
        Draft {
            text: "ignored".into(),
        }
    })
    .await;
    let eval = FailingEval.combine(WordCount.lift(), |left, right| (left, right));

    let error = eval.run_collected(&ctx, &collected).await.unwrap_err();

    assert!(matches!(
        error,
        CombineError::Left(FailError::MissingReport)
    ));
}
