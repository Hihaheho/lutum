use std::{future::Future, pin::Pin};

use async_trait::async_trait;
use lutum::Lutum;
use thiserror::Error;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};

use lutum_trace::{CollectedRaw, RawTraceSnapshot, TraceEvent, TraceSnapshot};

use crate::{Collected, EvalRecord, Eval, Objective, RawEvalRecord, Scored};

/// A mutable, live-only evaluator over a stream of trace events plus a final
/// trace/artifact pair.
///
/// Probes do not own their dispatcher. [`ProbeRuntime`] forwards trace events,
/// exposes a [`ProbeHandle`] for local hook composition, and calls
/// [`Probe::finalize`] after the traced future completes.
#[async_trait]
pub trait Probe: Send + 'static {
    type Report;
    type Artifact;
    type Error;

    async fn on_trace_event(
        &mut self,
        _ctx: &Lutum,
        _event: &TraceEvent,
    ) -> Result<ProbeDecision<Self::Report>, Self::Error> {
        Ok(ProbeDecision::Continue)
    }

    async fn finalize(
        &mut self,
        ctx: &Lutum,
        trace: &TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error>;
}

#[async_trait]
impl<T> Probe for T
where
    T: Eval + Send + Sync + 'static,
    T::Artifact: Sync,
{
    type Report = T::Report;
    type Artifact = T::Artifact;
    type Error = T::Error;

    async fn finalize(
        &mut self,
        ctx: &Lutum,
        trace: &TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Eval::evaluate(self, ctx, trace, artifact).await
    }
}

#[derive(Debug, PartialEq)]
pub enum ProbeDecision<R> {
    Continue,
    Complete(R),
}

type ProbeTaskFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
type ProbeTask<P> = Box<dyn for<'a> FnOnce(&'a mut P) -> ProbeTaskFuture<'a> + Send>;
pub type ProbeDispatchFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

enum ProbeMessage<P: Probe> {
    Dispatch(ProbeTask<P>),
    Trace {
        ctx: Lutum,
        event: TraceEvent,
    },
    Finalize {
        ctx: Lutum,
        trace: TraceSnapshot,
        artifact: P::Artifact,
        reply: oneshot::Sender<Result<P::Report, ProbeRunError<P::Error>>>,
    },
}

pub struct ProbeRuntime<P: Probe> {
    dispatcher: ProbeHandle<P>,
    task: JoinHandle<()>,
}

impl<P> ProbeRuntime<P>
where
    P: Probe,
    P::Artifact: Send + 'static,
    P::Error: Send + 'static,
    P::Report: Send + 'static,
{
    pub fn new(probe: P) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let dispatcher = ProbeHandle { tx };
        let task = tokio::spawn(run_probe(probe, rx));

        Self { dispatcher, task }
    }

    pub fn dispatcher(&self) -> ProbeHandle<P> {
        self.dispatcher.clone()
    }

    pub async fn run_future<F>(
        self,
        ctx: &Lutum,
        future: F,
    ) -> Result<P::Report, ProbeRunError<P::Error>>
    where
        F: Future<Output = P::Artifact>,
    {
        let dispatcher = self.dispatcher.clone();
        let ctx = ctx.clone();
        let event_ctx = ctx.clone();
        let collected = lutum_trace::capture_with_events(future, move |event| {
            let _ = dispatcher.send_trace(event_ctx.clone(), event);
        })
        .await;

        self.run_collected(&ctx, collected).await
    }

    pub async fn run_collected(
        self,
        ctx: &Lutum,
        collected: Collected<P::Artifact>,
    ) -> Result<P::Report, ProbeRunError<P::Error>> {
        self.run_parts(ctx, collected.trace, collected.output).await
    }

    pub fn scored_by<'a, O>(self, objective: &'a O) -> ProbeScoredBy<'a, P, O>
    where
        O: Objective<P::Report>,
    {
        ProbeScoredBy {
            runtime: self,
            objective,
        }
    }

    async fn run_future_with_trace<F>(
        self,
        ctx: &Lutum,
        future: F,
    ) -> (Result<P::Report, ProbeRunError<P::Error>>, TraceSnapshot)
    where
        F: Future<Output = P::Artifact>,
    {
        let dispatcher = self.dispatcher.clone();
        let event_ctx = ctx.clone();
        let Collected { output, trace } =
            lutum_trace::capture_with_events(future, move |event| {
                let _ = dispatcher.send_trace(event_ctx.clone(), event);
            })
            .await;
        let trace_clone = trace.clone();
        let result = self.run_parts(ctx, trace, output).await;
        (result, trace_clone)
    }

    async fn run_future_with_raw_trace<F>(
        self,
        ctx: &Lutum,
        future: F,
    ) -> (
        Result<P::Report, ProbeRunError<P::Error>>,
        TraceSnapshot,
        RawTraceSnapshot,
    )
    where
        F: Future<Output = P::Artifact>,
    {
        let dispatcher = self.dispatcher.clone();
        let event_ctx = ctx.clone();
        let CollectedRaw { output, trace, raw } =
            lutum_trace::capture_raw_with_events(future, move |event| {
                let _ = dispatcher.send_trace(event_ctx.clone(), event);
            })
            .await;
        let trace_clone = trace.clone();
        let result = self.run_parts(ctx, trace, output).await;
        (result, trace_clone, raw)
    }

    async fn run_parts(
        self,
        ctx: &Lutum,
        trace: TraceSnapshot,
        artifact: P::Artifact,
    ) -> Result<P::Report, ProbeRunError<P::Error>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let send_result = self.dispatcher.send_message(ProbeMessage::Finalize {
            ctx: ctx.clone(),
            trace,
            artifact,
            reply: reply_tx,
        });

        let response = if send_result.is_ok() {
            reply_rx.await.ok()
        } else {
            None
        };

        match self.task.await {
            Ok(()) => response.unwrap_or(Err(ProbeRunError::DispatcherClosed)),
            Err(source) => Err(ProbeRunError::Panicked { source }),
        }
    }
}

pub struct ProbeScoredBy<'a, P: Probe, O> {
    runtime: ProbeRuntime<P>,
    objective: &'a O,
}

impl<'a, P, O> ProbeScoredBy<'a, P, O>
where
    P: Probe,
    P::Artifact: Send + 'static,
    P::Error: Send + 'static,
    P::Report: Send + 'static,
    O: Objective<P::Report>,
{
    pub async fn run_future<F>(
        self,
        ctx: &Lutum,
        future: F,
    ) -> Result<Scored<P::Report>, ProbeScoreError<P::Error, O::Error>>
    where
        F: Future<Output = P::Artifact>,
    {
        let report = self
            .runtime
            .run_future(ctx, future)
            .await
            .map_err(ProbeScoreError::Probe)?;
        let score = self
            .objective
            .score(&report)
            .map_err(ProbeScoreError::Objective)?;
        Ok(Scored { report, score })
    }

    pub async fn run_collected(
        self,
        ctx: &Lutum,
        collected: Collected<P::Artifact>,
    ) -> Result<Scored<P::Report>, ProbeScoreError<P::Error, O::Error>> {
        let report = self
            .runtime
            .run_collected(ctx, collected)
            .await
            .map_err(ProbeScoreError::Probe)?;
        let score = self
            .objective
            .score(&report)
            .map_err(ProbeScoreError::Objective)?;
        Ok(Scored { report, score })
    }

    pub async fn run_future_record<F>(
        self,
        ctx: &Lutum,
        future: F,
    ) -> EvalRecord<P::Report, ProbeScoreError<P::Error, O::Error>>
    where
        F: Future<Output = P::Artifact>,
    {
        let (probe_result, trace) = self.runtime.run_future_with_trace(ctx, future).await;
        let result = probe_result.map_err(ProbeScoreError::Probe).and_then(|report| {
            self.objective
                .score(&report)
                .map_err(ProbeScoreError::Objective)
                .map(|score| Scored { report, score })
        });
        EvalRecord { result, trace }
    }

    pub async fn run_future_raw_record<F>(
        self,
        ctx: &Lutum,
        future: F,
    ) -> RawEvalRecord<P::Report, ProbeScoreError<P::Error, O::Error>>
    where
        F: Future<Output = P::Artifact>,
    {
        let (probe_result, trace, raw) =
            self.runtime.run_future_with_raw_trace(ctx, future).await;
        let result = probe_result.map_err(ProbeScoreError::Probe).and_then(|report| {
            self.objective
                .score(&report)
                .map_err(ProbeScoreError::Objective)
                .map(|score| Scored { report, score })
        });
        RawEvalRecord { result, trace, raw }
    }
}

pub struct ProbeHandle<P: Probe> {
    tx: mpsc::UnboundedSender<ProbeMessage<P>>,
}

impl<P: Probe> Clone for ProbeHandle<P> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl<P> ProbeHandle<P>
where
    P: Probe,
{
    pub async fn dispatch<R, F>(&self, f: F) -> Result<R, ProbeDispatchError>
    where
        R: Send + 'static,
        F: Send + 'static + for<'a> FnOnce(&'a mut P) -> ProbeDispatchFuture<'a, R>,
    {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.send_message(ProbeMessage::Dispatch(Box::new(move |probe| {
            Box::pin(async move {
                let result = f(probe).await;
                let _ = reply_tx.send(result);
            })
        })))?;

        reply_rx.await.map_err(|_| ProbeDispatchError)
    }

    fn send_trace(&self, ctx: Lutum, event: TraceEvent) -> Result<(), ProbeDispatchError> {
        self.send_message(ProbeMessage::Trace { ctx, event })
    }

    fn send_message(&self, message: ProbeMessage<P>) -> Result<(), ProbeDispatchError> {
        self.tx.send(message).map_err(|_| ProbeDispatchError)
    }
}

#[derive(Debug, Clone, Copy, Eq, Error, PartialEq)]
#[error("probe dispatcher closed")]
pub struct ProbeDispatchError;

#[derive(Debug, Error)]
pub enum ProbeRunError<E> {
    #[error("probe failed: {0}")]
    Probe(#[source] E),
    #[error("probe dispatcher closed")]
    DispatcherClosed,
    #[error("probe dispatcher task panicked: {source}")]
    Panicked {
        #[source]
        source: tokio::task::JoinError,
    },
}

#[derive(Debug, Error)]
pub enum ProbeScoreError<PE, OE> {
    #[error("probe failed: {0}")]
    Probe(#[source] ProbeRunError<PE>),
    #[error("objective failed: {0}")]
    Objective(#[source] OE),
}

async fn run_probe<P>(mut probe: P, mut rx: mpsc::UnboundedReceiver<ProbeMessage<P>>)
where
    P: Probe,
    P::Artifact: Send + 'static,
    P::Error: Send + 'static,
    P::Report: Send + 'static,
{
    let mut completed = None;
    let mut failure = None;

    while let Some(message) = rx.recv().await {
        match message {
            ProbeMessage::Dispatch(task) => task(&mut probe).await,
            ProbeMessage::Trace { ctx, event } => {
                if completed.is_some() || failure.is_some() {
                    continue;
                }

                match probe.on_trace_event(&ctx, &event).await {
                    Ok(ProbeDecision::Continue) => {}
                    Ok(ProbeDecision::Complete(report)) => completed = Some(report),
                    Err(source) => failure = Some(source),
                }
            }
            ProbeMessage::Finalize {
                ctx,
                trace,
                artifact,
                reply,
            } => {
                let result = if let Some(report) = completed.take() {
                    Ok(report)
                } else if let Some(source) = failure.take() {
                    Err(ProbeRunError::Probe(source))
                } else {
                    probe
                        .finalize(&ctx, &trace, &artifact)
                        .await
                        .map_err(ProbeRunError::Probe)
                };

                let _ = reply.send(result);
                break;
            }
        }
    }
}
