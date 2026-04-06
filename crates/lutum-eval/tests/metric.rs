use async_trait::async_trait;
use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
use lutum_eval::{
    JudgeMetric, Metric, evaluate_collected, evaluate_live, judge_collected, judge_live,
};

#[derive(Debug, Eq, PartialEq)]
struct SampleArtifact {
    score: u32,
}

struct ArtifactMetric;

impl Metric for ArtifactMetric {
    type Artifact = SampleArtifact;
    type Score = u32;
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        _trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok(artifact.score)
    }
}

struct TraceMetric;

impl Metric for TraceMetric {
    type Artifact = SampleArtifact;
    type Score = bool;
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        trace: &lutum_eval::TraceSnapshot,
        _artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok(trace.has_event_message("trace-only"))
    }
}

struct CombinedMetric;

impl Metric for CombinedMetric {
    type Artifact = SampleArtifact;
    type Score = u32;
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok(artifact.score + u32::from(trace.has_event_message("inside live")))
    }
}

struct TupleMetric;

impl Metric for TupleMetric {
    type Artifact = (u32, &'static str);
    type Score = (&'static str, u32);
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        _trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok((artifact.1, artifact.0))
    }
}

struct ResultMetric;

impl Metric for ResultMetric {
    type Artifact = Result<u32, &'static str>;
    type Score = bool;
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        _trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok(artifact.is_ok())
    }
}

struct ArtifactJudge;

#[async_trait]
impl JudgeMetric for ArtifactJudge {
    type Artifact = SampleArtifact;
    type Score = u32;
    type Error = core::convert::Infallible;

    async fn judge(
        &self,
        _ctx: &Lutum,
        _trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok(artifact.score)
    }
}

struct TraceJudge;

#[async_trait]
impl JudgeMetric for TraceJudge {
    type Artifact = SampleArtifact;
    type Score = bool;
    type Error = core::convert::Infallible;

    async fn judge(
        &self,
        _ctx: &Lutum,
        trace: &lutum_eval::TraceSnapshot,
        _artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok(trace.has_event_message("judge-trace"))
    }
}

struct CombinedJudge;

#[async_trait]
impl JudgeMetric for CombinedJudge {
    type Artifact = SampleArtifact;
    type Score = u32;
    type Error = core::convert::Infallible;

    async fn judge(
        &self,
        _ctx: &Lutum,
        trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok(artifact.score + u32::from(trace.has_event_message("judge-live")))
    }
}

struct DualModeMetric;

impl Metric for DualModeMetric {
    type Artifact = SampleArtifact;
    type Score = u32;
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        _trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok(artifact.score)
    }
}

#[async_trait]
impl JudgeMetric for DualModeMetric {
    type Artifact = SampleArtifact;
    type Score = u32;
    type Error = core::convert::Infallible;

    async fn judge(
        &self,
        _ctx: &Lutum,
        _trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok(artifact.score + 1)
    }
}

fn make_context() -> Lutum {
    Lutum::new(
        std::sync::Arc::new(MockLlmAdapter::new()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    )
}

#[tokio::test]
async fn evaluate_collected_reads_user_defined_artifact() {
    let collected = lutum_trace::test::collect(async {
        tracing::info!("artifact metric");
        SampleArtifact { score: 7 }
    })
    .await;

    let score = evaluate_collected(&ArtifactMetric, &collected).unwrap();

    assert_eq!(score, 7);
}

#[tokio::test]
async fn evaluate_collected_can_read_trace_only_metric() {
    let collected = lutum_trace::test::collect(async {
        tracing::info!("trace-only");
        SampleArtifact { score: 0 }
    })
    .await;

    let score = evaluate_collected(&TraceMetric, &collected).unwrap();

    assert!(score);
}

#[tokio::test]
async fn collected_results_can_be_reused_across_metrics() {
    let collected = lutum_trace::test::collect(async {
        tracing::info!("trace-only");
        SampleArtifact { score: 11 }
    })
    .await;

    let artifact_score = evaluate_collected(&ArtifactMetric, &collected).unwrap();
    let trace_score = evaluate_collected(&TraceMetric, &collected).unwrap();

    assert_eq!(artifact_score, 11);
    assert!(trace_score);
}

#[tokio::test]
async fn evaluate_live_uses_future_output_as_artifact() {
    let collected = lutum_trace::test::collect(async {
        evaluate_live(&CombinedMetric, async {
            tracing::info!("inside live");
            SampleArtifact { score: 5 }
        })
        .await
    })
    .await;

    assert_eq!(collected.output.unwrap(), 6);
}

#[tokio::test]
async fn tuple_artifacts_are_supported() {
    let collected = lutum_trace::test::collect(async { (3_u32, "ok") }).await;

    let score = evaluate_collected(&TupleMetric, &collected).unwrap();

    assert_eq!(score, ("ok", 3));
}

#[tokio::test]
async fn result_artifacts_are_supported() {
    let collected = lutum_trace::test::collect(async { Ok::<u32, &'static str>(9) }).await;

    let score = evaluate_collected(&ResultMetric, &collected).unwrap();

    assert!(score);
}

#[tokio::test]
async fn judge_collected_reads_user_defined_artifact() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async {
        tracing::info!("judge artifact");
        SampleArtifact { score: 13 }
    })
    .await;

    let score = judge_collected(&ArtifactJudge, &ctx, &collected)
        .await
        .unwrap();

    assert_eq!(score, 13);
}

#[tokio::test]
async fn judge_collected_can_read_trace_only_judge() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async {
        tracing::info!("judge-trace");
        SampleArtifact { score: 0 }
    })
    .await;

    let score = judge_collected(&TraceJudge, &ctx, &collected)
        .await
        .unwrap();

    assert!(score);
}

#[tokio::test]
async fn collected_results_can_be_reused_across_judges() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async {
        tracing::info!("judge-trace");
        SampleArtifact { score: 17 }
    })
    .await;

    let artifact_score = judge_collected(&ArtifactJudge, &ctx, &collected)
        .await
        .unwrap();
    let trace_score = judge_collected(&TraceJudge, &ctx, &collected)
        .await
        .unwrap();

    assert_eq!(artifact_score, 17);
    assert!(trace_score);
}

#[tokio::test]
async fn judge_live_uses_future_output_as_artifact() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async {
        judge_live(&CombinedJudge, &ctx, async {
            tracing::info!("judge-live");
            SampleArtifact { score: 8 }
        })
        .await
    })
    .await;

    assert_eq!(collected.output.unwrap(), 9);
}

#[tokio::test]
async fn same_type_can_implement_metric_and_judge_metric() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async { SampleArtifact { score: 4 } }).await;

    let eval_score = evaluate_collected(&DualModeMetric, &collected).unwrap();
    let judge_score = judge_collected(&DualModeMetric, &ctx, &collected)
        .await
        .unwrap();

    assert_eq!(eval_score, 4);
    assert_eq!(judge_score, 5);
}
