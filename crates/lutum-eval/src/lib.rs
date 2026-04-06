use std::future::Future;

use async_trait::async_trait;

pub use lutum_trace::{Collected, EventRecord, FieldValue, SpanNode, TraceSnapshot};

/// Pure evaluation over a trace snapshot and a strongly typed artifact.
///
/// Metrics are intentionally synchronous and borrow their inputs so the same
/// collected result can be evaluated multiple times, both in live execution and
/// future replay runners.
pub trait Metric {
    type Artifact;
    type Score;
    type Error;

    fn evaluate(
        &self,
        trace: &TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error>;
}

/// Async evaluation over a trace snapshot and a strongly typed artifact with
/// access to a [`lutum::Lutum`].
///
/// Judge metrics are intended for model-based scoring and other evaluation
/// flows that need to execute through lutum at scoring time.
#[async_trait]
pub trait JudgeMetric {
    type Artifact;
    type Score;
    type Error;

    async fn judge(
        &self,
        ctx: &lutum::Lutum,
        trace: &TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error>;
}

/// Evaluate a metric against an existing [`Collected`] value.
pub fn evaluate_collected<M>(
    metric: &M,
    collected: &Collected<M::Artifact>,
) -> Result<M::Score, M::Error>
where
    M: Metric,
{
    metric.evaluate(&collected.trace, &collected.output)
}

/// Judge an existing [`Collected`] value with a [`JudgeMetric`].
pub async fn judge_collected<M>(
    metric: &M,
    ctx: &lutum::Lutum,
    collected: &Collected<M::Artifact>,
) -> Result<M::Score, M::Error>
where
    M: JudgeMetric,
{
    metric.judge(ctx, &collected.trace, &collected.output).await
}

/// Capture `future` with [`lutum_trace::capture`] and evaluate the resulting
/// trace/artifact pair with `metric`.
///
/// Like [`lutum_trace::capture`], this requires the active subscriber stack to
/// include [`lutum_trace::layer`]. If no layer is installed, the artifact is
/// still evaluated but the trace will be empty.
pub async fn evaluate_live<M, F>(metric: &M, future: F) -> Result<M::Score, M::Error>
where
    M: Metric,
    F: Future<Output = M::Artifact>,
{
    let collected = lutum_trace::capture(future).await;
    evaluate_collected(metric, &collected)
}

/// Capture `future` with [`lutum_trace::capture`] and judge the resulting
/// trace/artifact pair with `metric`.
pub async fn judge_live<M, F>(
    metric: &M,
    ctx: &lutum::Lutum,
    future: F,
) -> Result<M::Score, M::Error>
where
    M: JudgeMetric,
    F: Future<Output = M::Artifact>,
{
    let collected = lutum_trace::capture(future).await;
    judge_collected(metric, ctx, &collected).await
}
