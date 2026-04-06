use std::future::Future;

use async_trait::async_trait;

mod macros;
mod probe;

/// Re-exported so `register_probe_hook!` can use `$crate::paste::paste!` without
/// requiring users to add `paste` to their own `Cargo.toml`.
#[doc(hidden)]
pub use paste;

pub use crate::probe::{
    Probe, ProbeContext, ProbeDecision, ProbeDispatchError, ProbeDispatchFuture, ProbeDispatchHook,
    ProbeDispatcher, ProbeHandle, ProbeHookSlot, ProbeRunError, ProbeRuntime,
};
pub use lutum_trace::{
    Collected, EventRecord, FieldValue, SpanNode, TraceEvent, TraceSnapshot, TraceSpanId,
};

/// Pure evaluation over a trace snapshot and a strongly typed artifact.
///
/// Pure metrics are intentionally synchronous and borrow their inputs so the
/// same collected result can be evaluated multiple times, both in live
/// execution and future replay runners.
pub trait PureMetric {
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
/// `Metric` is the main scoring abstraction. [`PureMetric`] is the synchronous,
/// context-free subset and is lifted automatically into `Metric`.
#[async_trait]
pub trait Metric {
    type Artifact;
    type Score;
    type Error;

    async fn evaluate(
        &self,
        ctx: &lutum::Lutum,
        trace: &TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error>;
}

#[async_trait]
impl<T> Metric for T
where
    T: PureMetric + Send + Sync,
{
    type Artifact = T::Artifact;
    type Score = T::Score;
    type Error = T::Error;

    async fn evaluate(
        &self,
        _ctx: &lutum::Lutum,
        trace: &TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        PureMetric::evaluate(self, trace, artifact)
    }
}

/// Evaluate a pure metric against an existing [`Collected`] value.
pub fn evaluate_pure_collected<M>(
    metric: &M,
    collected: &Collected<M::Artifact>,
) -> Result<M::Score, M::Error>
where
    M: PureMetric,
{
    metric.evaluate(&collected.trace, &collected.output)
}

/// Evaluate a metric against an existing [`Collected`] value.
pub async fn evaluate_collected<M>(
    metric: &M,
    ctx: &lutum::Lutum,
    collected: &Collected<M::Artifact>,
) -> Result<M::Score, M::Error>
where
    M: Metric,
{
    metric
        .evaluate(ctx, &collected.trace, &collected.output)
        .await
}

/// Capture `future` with [`lutum_trace::capture`] and evaluate the resulting
/// trace/artifact pair with `metric`.
///
/// Like [`lutum_trace::capture`], this requires the active subscriber stack to
/// include [`lutum_trace::layer`]. If no layer is installed, the artifact is
/// still evaluated but the trace will be empty.
pub async fn evaluate_pure_live<M, F>(metric: &M, future: F) -> Result<M::Score, M::Error>
where
    M: PureMetric,
    F: Future<Output = M::Artifact>,
{
    let collected = lutum_trace::capture(future).await;
    evaluate_pure_collected(metric, &collected)
}

/// Capture `future` with [`lutum_trace::capture`] and evaluate the resulting
/// trace/artifact pair with `metric`.
pub async fn evaluate_live<M, F>(
    metric: &M,
    ctx: &lutum::Lutum,
    future: F,
) -> Result<M::Score, M::Error>
where
    M: Metric,
    F: Future<Output = M::Artifact>,
{
    let collected = lutum_trace::capture(future).await;
    evaluate_collected(metric, ctx, &collected).await
}
