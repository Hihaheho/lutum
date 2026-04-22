use async_trait::async_trait;
use thiserror::Error;

mod combinators;
mod judge;
mod objective;
mod probe;
mod record;
mod score;

pub use crate::probe::{
    Probe, ProbeDecision, ProbeDispatchError, ProbeDispatchFuture, ProbeHandle, ProbeRunError,
    ProbeRuntime, ProbeScoreError, ProbeScoredBy,
};
pub use combinators::{
    Combine, CombineError, ContramapArtifact, EvalExt, LiftPure, MapEvalError, MapReport,
    PureEvalExt, PureScoredBy, ScoredBy,
};
pub use judge::{JudgeEval, JudgeEvalError};
pub use lutum_trace::{
    Collected, CollectedRaw, EventRecord, FieldValue, RawTraceSnapshot, SpanNode, TraceEvent,
    TraceSnapshot, TraceSpanId,
};
pub use objective::{
    InvertObjective, MapObjectiveError, Maximize, Minimize, Objective, ObjectiveExt,
    PassFailObjective, maximize, minimize, pass_fail,
};
pub use record::{EvalRecord, RawEvalRecord};
pub use score::{Score, ScoreRangeError};

/// Pure evaluation over a trace snapshot and a strongly typed artifact.
///
/// Pure evals are intentionally synchronous and borrow their inputs so the same
/// collected result can be evaluated multiple times, both in live execution and
/// future replay runners.
pub trait PureEval {
    type Artifact;
    type Report;
    type Error;

    fn evaluate(
        &self,
        trace: &TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error>;
}

/// Async evaluation over a trace snapshot and a strongly typed artifact with
/// access to a [`lutum::Lutum`].
///
/// `Eval` is the main observation abstraction. [`PureEval`] is the
/// synchronous, context-free subset; use [`PureEvalExt::lift`] when you want
/// to run a pure eval through async `Eval` combinators.
#[async_trait]
pub trait Eval {
    type Artifact;
    type Report;
    type Error;

    async fn evaluate(
        &self,
        ctx: &lutum::Lutum,
        trace: &TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct Scored<R> {
    pub report: R,
    pub score: Score,
}

#[derive(Debug, Error)]
pub enum ScoreEvalError<EE, OE> {
    #[error("evaluation failed: {0}")]
    Eval(#[source] EE),
    #[error("objective failed: {0}")]
    Objective(#[source] OE),
}
