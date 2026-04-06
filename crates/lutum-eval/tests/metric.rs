use async_trait::async_trait;
use lutum::{Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};
use lutum_eval::{
    Metric, PureMetric, evaluate_collected, evaluate_live, evaluate_pure_collected,
    evaluate_pure_live,
};

#[derive(Debug, Eq, PartialEq)]
struct SampleArtifact {
    score: u32,
}

struct ArtifactPureMetric;

impl PureMetric for ArtifactPureMetric {
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

struct TracePureMetric;

impl PureMetric for TracePureMetric {
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

struct CombinedPureMetric;

impl PureMetric for CombinedPureMetric {
    type Artifact = SampleArtifact;
    type Score = u32;
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok(artifact.score + u32::from(trace.has_event_message("inside pure live")))
    }
}

struct TuplePureMetric;

impl PureMetric for TuplePureMetric {
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

struct ResultPureMetric;

impl PureMetric for ResultPureMetric {
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

struct ArtifactMetric;

#[async_trait]
impl Metric for ArtifactMetric {
    type Artifact = SampleArtifact;
    type Score = u32;
    type Error = core::convert::Infallible;

    async fn evaluate(
        &self,
        _ctx: &Lutum,
        _trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok(artifact.score + 1)
    }
}

struct TraceMetric;

#[async_trait]
impl Metric for TraceMetric {
    type Artifact = SampleArtifact;
    type Score = bool;
    type Error = core::convert::Infallible;

    async fn evaluate(
        &self,
        _ctx: &Lutum,
        trace: &lutum_eval::TraceSnapshot,
        _artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok(trace.has_event_message("metric-trace"))
    }
}

struct CombinedMetric;

#[async_trait]
impl Metric for CombinedMetric {
    type Artifact = SampleArtifact;
    type Score = u32;
    type Error = core::convert::Infallible;

    async fn evaluate(
        &self,
        _ctx: &Lutum,
        trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok(artifact.score + u32::from(trace.has_event_message("inside metric live")) + 1)
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
        tracing::info!("artifact metric");
        SampleArtifact { score: 7 }
    })
    .await;

    let score = evaluate_pure_collected(&ArtifactPureMetric, &collected).unwrap();

    assert_eq!(score, 7);
}

#[tokio::test]
async fn evaluate_pure_collected_can_read_trace_only_metric() {
    let collected = lutum_trace::test::collect(async {
        tracing::info!("trace-only");
        SampleArtifact { score: 0 }
    })
    .await;

    let score = evaluate_pure_collected(&TracePureMetric, &collected).unwrap();

    assert!(score);
}

#[tokio::test]
async fn pure_collected_results_can_be_reused_across_metrics() {
    let collected = lutum_trace::test::collect(async {
        tracing::info!("trace-only");
        SampleArtifact { score: 11 }
    })
    .await;

    let artifact_score = evaluate_pure_collected(&ArtifactPureMetric, &collected).unwrap();
    let trace_score = evaluate_pure_collected(&TracePureMetric, &collected).unwrap();

    assert_eq!(artifact_score, 11);
    assert!(trace_score);
}

#[tokio::test]
async fn evaluate_pure_live_uses_future_output_as_artifact() {
    let collected = lutum_trace::test::collect(async {
        evaluate_pure_live(&CombinedPureMetric, async {
            tracing::info!("inside pure live");
            SampleArtifact { score: 5 }
        })
        .await
    })
    .await;

    assert_eq!(collected.output.unwrap(), 6);
}

#[tokio::test]
async fn tuple_artifacts_are_supported_for_pure_metrics() {
    let collected = lutum_trace::test::collect(async { (3_u32, "ok") }).await;

    let score = evaluate_pure_collected(&TuplePureMetric, &collected).unwrap();

    assert_eq!(score, ("ok", 3));
}

#[tokio::test]
async fn result_artifacts_are_supported_for_pure_metrics() {
    let collected = lutum_trace::test::collect(async { Ok::<u32, &'static str>(9) }).await;

    let score = evaluate_pure_collected(&ResultPureMetric, &collected).unwrap();

    assert!(score);
}

#[tokio::test]
async fn evaluate_collected_reads_user_defined_artifact() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async {
        tracing::info!("metric artifact");
        SampleArtifact { score: 13 }
    })
    .await;

    let score = evaluate_collected(&ArtifactMetric, &ctx, &collected)
        .await
        .unwrap();

    assert_eq!(score, 14);
}

#[tokio::test]
async fn evaluate_collected_can_read_trace_only_metric() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async {
        tracing::info!("metric-trace");
        SampleArtifact { score: 0 }
    })
    .await;

    let score = evaluate_collected(&TraceMetric, &ctx, &collected)
        .await
        .unwrap();

    assert!(score);
}

#[tokio::test]
async fn collected_results_can_be_reused_across_metrics() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async {
        tracing::info!("metric-trace");
        SampleArtifact { score: 17 }
    })
    .await;

    let artifact_score = evaluate_collected(&ArtifactMetric, &ctx, &collected)
        .await
        .unwrap();
    let trace_score = evaluate_collected(&TraceMetric, &ctx, &collected)
        .await
        .unwrap();

    assert_eq!(artifact_score, 18);
    assert!(trace_score);
}

#[tokio::test]
async fn evaluate_live_uses_future_output_as_artifact() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async {
        evaluate_live(&CombinedMetric, &ctx, async {
            tracing::info!("inside metric live");
            SampleArtifact { score: 8 }
        })
        .await
    })
    .await;

    assert_eq!(collected.output.unwrap(), 10);
}

#[tokio::test]
async fn pure_metric_is_lifted_into_metric() {
    let ctx = make_context();
    let collected = lutum_trace::test::collect(async { SampleArtifact { score: 4 } }).await;

    let pure_score = evaluate_pure_collected(&ArtifactPureMetric, &collected).unwrap();
    let metric_score = evaluate_collected(&ArtifactPureMetric, &ctx, &collected)
        .await
        .unwrap();

    assert_eq!(pure_score, 4);
    assert_eq!(metric_score, 4);
}
