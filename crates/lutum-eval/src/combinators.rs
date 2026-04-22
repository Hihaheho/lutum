use std::{future::Future, marker::PhantomData};

use async_trait::async_trait;
use thiserror::Error;

use lutum_trace::RawTraceSnapshot;

use crate::{Collected, CollectedRaw, Eval, Objective, PureEval, ScoreEvalError, Scored};

type ScoredResult<R, EE, OE> = Result<Scored<R>, ScoreEvalError<EE, OE>>;

#[async_trait]
pub trait PureEvalExt: PureEval + Sized {
    fn lift(self) -> LiftPure<Self> {
        LiftPure { eval: self }
    }

    fn map_report<F>(self, map: F) -> MapReport<Self, F> {
        MapReport { eval: self, map }
    }

    fn map_error<F>(self, map: F) -> MapEvalError<Self, F> {
        MapEvalError { eval: self, map }
    }

    fn contramap_artifact<A, F>(self, project: F) -> ContramapArtifact<Self, F, A> {
        ContramapArtifact {
            eval: self,
            project,
            artifact: PhantomData,
        }
    }

    fn combine<E, F>(self, other: E, combine: F) -> Combine<Self, E, F>
    where
        E: PureEval<Artifact = Self::Artifact>,
    {
        Combine {
            left: self,
            right: other,
            combine,
        }
    }

    fn run_collected(
        &self,
        collected: &Collected<Self::Artifact>,
    ) -> Result<Self::Report, Self::Error> {
        self.evaluate(&collected.trace, &collected.output)
    }

    async fn run_future<F>(&self, future: F) -> Result<Self::Report, Self::Error>
    where
        F: Future<Output = Self::Artifact> + Send,
    {
        let collected = lutum_trace::capture(future).await;
        self.run_collected(&collected)
    }

    async fn run_future_raw<F>(
        &self,
        future: F,
    ) -> (Result<Self::Report, Self::Error>, RawTraceSnapshot)
    where
        F: Future<Output = Self::Artifact> + Send,
    {
        let CollectedRaw { output, trace, raw } = lutum_trace::capture_raw(future).await;
        let result = self.evaluate(&trace, &output);
        (result, raw)
    }

    fn scored_by<'a, O>(&'a self, objective: &'a O) -> PureScoredBy<'a, Self, O>
    where
        O: Objective<Self::Report>,
    {
        PureScoredBy {
            eval: self,
            objective,
        }
    }
}

#[async_trait]
impl<T> PureEvalExt for T where T: PureEval + Sized {}

pub struct LiftPure<E> {
    eval: E,
}

#[async_trait]
impl<E> Eval for LiftPure<E>
where
    E: PureEval + Send + Sync,
    E::Artifact: Sync,
{
    type Artifact = E::Artifact;
    type Report = E::Report;
    type Error = E::Error;

    async fn evaluate(
        &self,
        _ctx: &lutum::Lutum,
        trace: &crate::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        self.eval.evaluate(trace, artifact)
    }
}

#[async_trait]
pub trait EvalExt: Eval + Sized {
    fn map_report<F>(self, map: F) -> MapReport<Self, F> {
        MapReport { eval: self, map }
    }

    fn map_error<F>(self, map: F) -> MapEvalError<Self, F> {
        MapEvalError { eval: self, map }
    }

    fn contramap_artifact<A, F>(self, project: F) -> ContramapArtifact<Self, F, A> {
        ContramapArtifact {
            eval: self,
            project,
            artifact: PhantomData,
        }
    }

    fn combine<E, F>(self, other: E, combine: F) -> Combine<Self, E, F>
    where
        E: Eval<Artifact = Self::Artifact>,
    {
        Combine {
            left: self,
            right: other,
            combine,
        }
    }

    async fn run_collected(
        &self,
        ctx: &lutum::Lutum,
        collected: &Collected<Self::Artifact>,
    ) -> Result<Self::Report, Self::Error>
    where
        Self::Artifact: Sync,
    {
        self.evaluate(ctx, &collected.trace, &collected.output)
            .await
    }

    async fn run_future<F>(
        &self,
        ctx: &lutum::Lutum,
        future: F,
    ) -> Result<Self::Report, Self::Error>
    where
        Self::Artifact: Send + Sync,
        F: Future<Output = Self::Artifact> + Send,
    {
        let collected = lutum_trace::capture(future).await;
        self.run_collected(ctx, &collected).await
    }

    async fn run_future_raw<F>(
        &self,
        ctx: &lutum::Lutum,
        future: F,
    ) -> (Result<Self::Report, Self::Error>, RawTraceSnapshot)
    where
        Self::Artifact: Send + Sync,
        F: Future<Output = Self::Artifact> + Send,
    {
        let CollectedRaw { output, trace, raw } = lutum_trace::capture_raw(future).await;
        let result = self.evaluate(ctx, &trace, &output).await;
        (result, raw)
    }

    fn scored_by<'a, O>(&'a self, objective: &'a O) -> ScoredBy<'a, Self, O>
    where
        O: Objective<Self::Report>,
    {
        ScoredBy {
            eval: self,
            objective,
        }
    }
}

#[async_trait]
impl<T> EvalExt for T where T: Eval + Sized {}

pub struct PureScoredBy<'a, E, O> {
    eval: &'a E,
    objective: &'a O,
}

impl<'a, E, O> PureScoredBy<'a, E, O>
where
    E: PureEval,
    O: Objective<E::Report>,
{
    pub fn run_collected(
        &self,
        collected: &Collected<E::Artifact>,
    ) -> ScoredResult<E::Report, E::Error, O::Error> {
        let report = self
            .eval
            .run_collected(collected)
            .map_err(ScoreEvalError::Eval)?;
        let score = self
            .objective
            .score(&report)
            .map_err(ScoreEvalError::Objective)?;
        Ok(Scored { report, score })
    }

    pub async fn run_future<F>(&self, future: F) -> ScoredResult<E::Report, E::Error, O::Error>
    where
        F: Future<Output = E::Artifact>,
    {
        let collected = lutum_trace::capture(future).await;
        self.run_collected(&collected)
    }

    pub async fn run_future_raw<F>(
        &self,
        future: F,
    ) -> (ScoredResult<E::Report, E::Error, O::Error>, RawTraceSnapshot)
    where
        F: Future<Output = E::Artifact>,
    {
        let CollectedRaw { output, trace, raw } = lutum_trace::capture_raw(future).await;
        let collected = Collected { output, trace };
        let result = self.run_collected(&collected);
        (result, raw)
    }
}

pub struct ScoredBy<'a, E, O> {
    eval: &'a E,
    objective: &'a O,
}

impl<'a, E, O> ScoredBy<'a, E, O>
where
    E: Eval + Sync,
    O: Objective<E::Report>,
{
    pub async fn run_collected(
        &self,
        ctx: &lutum::Lutum,
        collected: &Collected<E::Artifact>,
    ) -> ScoredResult<E::Report, E::Error, O::Error>
    where
        E::Artifact: Sync,
    {
        let report = self
            .eval
            .run_collected(ctx, collected)
            .await
            .map_err(ScoreEvalError::Eval)?;
        let score = self
            .objective
            .score(&report)
            .map_err(ScoreEvalError::Objective)?;
        Ok(Scored { report, score })
    }

    pub async fn run_future<F>(
        &self,
        ctx: &lutum::Lutum,
        future: F,
    ) -> ScoredResult<E::Report, E::Error, O::Error>
    where
        E::Artifact: Sync,
        F: Future<Output = E::Artifact>,
    {
        let collected = lutum_trace::capture(future).await;
        self.run_collected(ctx, &collected).await
    }

    pub async fn run_future_raw<F>(
        &self,
        ctx: &lutum::Lutum,
        future: F,
    ) -> (ScoredResult<E::Report, E::Error, O::Error>, RawTraceSnapshot)
    where
        E::Artifact: Sync,
        F: Future<Output = E::Artifact>,
    {
        let CollectedRaw { output, trace, raw } = lutum_trace::capture_raw(future).await;
        let collected = Collected { output, trace };
        let result = self.run_collected(ctx, &collected).await;
        (result, raw)
    }
}

pub struct MapReport<E, F> {
    eval: E,
    map: F,
}

impl<E, F, R> PureEval for MapReport<E, F>
where
    E: PureEval,
    F: Fn(E::Report) -> R,
{
    type Artifact = E::Artifact;
    type Report = R;
    type Error = E::Error;

    fn evaluate(
        &self,
        trace: &crate::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        let report = self.eval.evaluate(trace, artifact)?;
        Ok((self.map)(report))
    }
}

#[async_trait]
impl<E, F, R> Eval for MapReport<E, F>
where
    E: Eval + Send + Sync,
    E::Artifact: Sync,
    F: Fn(E::Report) -> R + Send + Sync,
    R: Send,
{
    type Artifact = E::Artifact;
    type Report = R;
    type Error = E::Error;

    async fn evaluate(
        &self,
        ctx: &lutum::Lutum,
        trace: &crate::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        let report = self.eval.evaluate(ctx, trace, artifact).await?;
        Ok((self.map)(report))
    }
}

pub struct MapEvalError<E, F> {
    eval: E,
    map: F,
}

impl<E, F, EE> PureEval for MapEvalError<E, F>
where
    E: PureEval,
    F: Fn(E::Error) -> EE,
{
    type Artifact = E::Artifact;
    type Report = E::Report;
    type Error = EE;

    fn evaluate(
        &self,
        trace: &crate::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        self.eval.evaluate(trace, artifact).map_err(&self.map)
    }
}

#[async_trait]
impl<E, F, EE> Eval for MapEvalError<E, F>
where
    E: Eval + Send + Sync,
    E::Artifact: Sync,
    F: Fn(E::Error) -> EE + Send + Sync,
{
    type Artifact = E::Artifact;
    type Report = E::Report;
    type Error = EE;

    async fn evaluate(
        &self,
        ctx: &lutum::Lutum,
        trace: &crate::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        self.eval
            .evaluate(ctx, trace, artifact)
            .await
            .map_err(&self.map)
    }
}

pub struct ContramapArtifact<E, F, A> {
    eval: E,
    project: F,
    artifact: PhantomData<fn() -> A>,
}

impl<E, F, A> PureEval for ContramapArtifact<E, F, A>
where
    E: PureEval,
    F: for<'b> Fn(&'b A) -> &'b E::Artifact,
{
    type Artifact = A;
    type Report = E::Report;
    type Error = E::Error;

    fn evaluate(
        &self,
        trace: &crate::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        self.eval.evaluate(trace, (self.project)(artifact))
    }
}

#[async_trait]
impl<E, F, A> Eval for ContramapArtifact<E, F, A>
where
    E: Eval + Send + Sync,
    E::Artifact: Sync,
    A: Sync,
    F: Send + Sync + for<'b> Fn(&'b A) -> &'b E::Artifact,
{
    type Artifact = A;
    type Report = E::Report;
    type Error = E::Error;

    async fn evaluate(
        &self,
        ctx: &lutum::Lutum,
        trace: &crate::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        self.eval
            .evaluate(ctx, trace, (self.project)(artifact))
            .await
    }
}

pub struct Combine<L, R, F> {
    left: L,
    right: R,
    combine: F,
}

#[derive(Debug, Error)]
pub enum CombineError<LE, RE> {
    #[error("left eval failed: {0}")]
    Left(#[source] LE),
    #[error("right eval failed: {0}")]
    Right(#[source] RE),
}

impl<L, R, F, O> PureEval for Combine<L, R, F>
where
    L: PureEval,
    R: PureEval<Artifact = L::Artifact>,
    F: Fn(L::Report, R::Report) -> O,
{
    type Artifact = L::Artifact;
    type Report = O;
    type Error = CombineError<L::Error, R::Error>;

    fn evaluate(
        &self,
        trace: &crate::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        let left = self
            .left
            .evaluate(trace, artifact)
            .map_err(CombineError::Left)?;
        let right = self
            .right
            .evaluate(trace, artifact)
            .map_err(CombineError::Right)?;
        Ok((self.combine)(left, right))
    }
}

#[async_trait]
impl<L, R, F, O> Eval for Combine<L, R, F>
where
    L: Eval + Send + Sync,
    R: Eval<Artifact = L::Artifact> + Send + Sync,
    L::Artifact: Sync,
    L::Report: Send,
    F: Fn(L::Report, R::Report) -> O + Send + Sync,
    O: Send,
{
    type Artifact = L::Artifact;
    type Report = O;
    type Error = CombineError<L::Error, R::Error>;

    async fn evaluate(
        &self,
        ctx: &lutum::Lutum,
        trace: &crate::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        let left = self
            .left
            .evaluate(ctx, trace, artifact)
            .await
            .map_err(CombineError::Left)?;
        let right = self
            .right
            .evaluate(ctx, trace, artifact)
            .await
            .map_err(CombineError::Right)?;
        Ok((self.combine)(left, right))
    }
}
