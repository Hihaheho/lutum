use async_trait::async_trait;
use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
use lutum_eval::{Eval, EvalExt, PureEval, PureEvalExt, Score, maximize};
use tracing::instrument::WithSubscriber as _;
use tracing_subscriber::layer::SubscriberExt as _;

#[derive(Debug, Eq, PartialEq)]
struct SampleArtifact {
    score: u32,
}

struct ArtifactPureEval;

impl PureEval for ArtifactPureEval {
    type Artifact = SampleArtifact;
    type Report = u32;
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        _trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(artifact.score)
    }
}

struct TracePureEval;

impl PureEval for TracePureEval {
    type Artifact = SampleArtifact;
    type Report = bool;
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        trace: &lutum_eval::TraceSnapshot,
        _artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(trace.has_event_message("trace-only"))
    }
}

struct CombinedPureEval;

impl PureEval for CombinedPureEval {
    type Artifact = SampleArtifact;
    type Report = u32;
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(artifact.score + u32::from(trace.has_event_message("inside pure live")))
    }
}

struct TuplePureEval;

impl PureEval for TuplePureEval {
    type Artifact = (u32, &'static str);
    type Report = (&'static str, u32);
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        _trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok((artifact.1, artifact.0))
    }
}

struct ResultPureEval;

impl PureEval for ResultPureEval {
    type Artifact = Result<u32, &'static str>;
    type Report = bool;
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        _trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(artifact.is_ok())
    }
}

struct ArtifactEval;

#[async_trait]
impl Eval for ArtifactEval {
    type Artifact = SampleArtifact;
    type Report = u32;
    type Error = core::convert::Infallible;

    async fn evaluate(
        &self,
        _ctx: &Lutum,
        _trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(artifact.score + 1)
    }
}

struct TraceEval;

#[async_trait]
impl Eval for TraceEval {
    type Artifact = SampleArtifact;
    type Report = bool;
    type Error = core::convert::Infallible;

    async fn evaluate(
        &self,
        _ctx: &Lutum,
        trace: &lutum_eval::TraceSnapshot,
        _artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(trace.has_event_message("metric-trace"))
    }
}

struct CombinedEval;

#[async_trait]
impl Eval for CombinedEval {
    type Artifact = SampleArtifact;
    type Report = u32;
    type Error = core::convert::Infallible;

    async fn evaluate(
        &self,
        _ctx: &Lutum,
        trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(artifact.score + u32::from(trace.has_event_message("inside eval live")) + 1)
    }
}

fn make_context() -> Lutum {
    Lutum::new(
        std::sync::Arc::new(MockLlmAdapter::new()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    )
}

#[tokio::test]
async fn evaluate_pure_collected_reads_user_defined_artifact() {
    let collected = lutum_trace::test::collect(async {
        tracing::info!(target: "lutum", "artifact eval");
        SampleArtifact { score: 7 }
    })
    .await;

    let report = ArtifactPureEval.run_collected(&collected).unwrap();

    assert_eq!(report, 7);
}

#[tokio::test]
async fn evaluate_pure_collected_can_read_trace_only_eval() {
    let collected = lutum_trace::test::collect(async {
        tracing::info!(target: "lutum", "trace-only");
        SampleArtifact { score: 0 }
    })
    .await;

    let report = TracePureEval.run_collected(&collected).unwrap();

    assert!(report);
}

#[tokio::test]
async fn pure_collected_results_can_be_reused_across_evals() {
    let collected = lutum_trace::test::collect(async {
        tracing::info!(target: "lutum", "trace-only");
        SampleArtifact { score: 11 }
    })
    .await;

    let artifact_report = ArtifactPureEval.run_collected(&collected).unwrap();
    let trace_report = TracePureEval.run_collected(&collected).unwrap();

    assert_eq!(artifact_report, 11);
    assert!(trace_report);
}

#[tokio::test]
async fn evaluate_pure_live_uses_future_output_as_artifact() {
    let collected = lutum_trace::test::collect(async {
        CombinedPureEval
            .run_future(async {
                tracing::info!(target: "lutum", "inside pure live");
                SampleArtifact { score: 5 }
            })
            .await
    })
    .await;

    assert_eq!(collected.output.unwrap(), 6);
}

#[tokio::test]
async fn tuple_artifacts_are_supported_for_pure_evals() {
    let collected = lutum_trace::test::collect(async { (3_u32, "ok") }).await;

    let report = TuplePureEval.run_collected(&collected).unwrap();

    assert_eq!(report, ("ok", 3));
}

#[tokio::test]
async fn result_artifacts_are_supported_for_pure_evals() {
    let collected = lutum_trace::test::collect(async { Ok::<u32, &'static str>(9) }).await;

    let report = ResultPureEval.run_collected(&collected).unwrap();

    assert!(report);
}

#[tokio::test]
async fn evaluate_collected_reads_user_defined_artifact() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async {
        tracing::info!(target: "lutum", "artifact eval async");
        SampleArtifact { score: 13 }
    })
    .await;

    let report = ArtifactEval.run_collected(&ctx, &collected).await.unwrap();

    assert_eq!(report, 14);
}

#[tokio::test]
async fn evaluate_collected_can_read_trace_only_eval() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async {
        tracing::info!(target: "lutum", "metric-trace");
        SampleArtifact { score: 0 }
    })
    .await;

    let report = TraceEval.run_collected(&ctx, &collected).await.unwrap();

    assert!(report);
}

#[tokio::test]
async fn collected_results_can_be_reused_across_evals() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async {
        tracing::info!(target: "lutum", "metric-trace");
        SampleArtifact { score: 17 }
    })
    .await;

    let artifact_report = ArtifactEval.run_collected(&ctx, &collected).await.unwrap();
    let trace_report = TraceEval.run_collected(&ctx, &collected).await.unwrap();

    assert_eq!(artifact_report, 18);
    assert!(trace_report);
}

#[tokio::test]
async fn evaluate_live_uses_future_output_as_artifact() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async {
        CombinedEval
            .run_future(&ctx, async {
                tracing::info!(target: "lutum", "inside eval live");
                SampleArtifact { score: 8 }
            })
            .await
    })
    .await;

    assert_eq!(collected.output.unwrap(), 10);
}

#[tokio::test]
async fn pure_eval_can_be_explicitly_lifted_into_eval() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async { SampleArtifact { score: 4 } }).await;

    let pure_report = ArtifactPureEval.run_collected(&collected).unwrap();
    let eval_report = ArtifactPureEval
        .lift()
        .run_collected(&ctx, &collected)
        .await
        .unwrap();

    assert_eq!(pure_report, 4);
    assert_eq!(eval_report, 4);
}

#[tokio::test]
async fn score_helpers_scalarize_reports_with_an_objective() {
    let ctx = make_context();
    let objective = maximize(|report: &u32| Score::new_clamped(*report as f32 / 10.0));
    let collected = lutum_trace::test::collect(async { SampleArtifact { score: 7 } }).await;

    let scored = ArtifactEval
        .scored_by(&objective)
        .run_collected(&ctx, &collected)
        .await
        .unwrap();

    assert_eq!(scored.score, Score::new_clamped(0.8));
    assert_eq!(scored.report, 8);
}

#[tokio::test]
async fn score_helpers_can_return_report_and_score_together() {
    let objective = maximize(|report: &u32| Score::new_clamped(*report as f32 / 10.0));

    let scored = ArtifactPureEval
        .scored_by(&objective)
        .run_future(async { SampleArtifact { score: 5 } })
        .with_subscriber(tracing_subscriber::registry().with(lutum_trace::layer()))
        .await
        .unwrap();

    assert_eq!(scored.report, 5);
    assert_eq!(scored.score, Score::new_clamped(0.5));
}
