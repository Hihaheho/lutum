use std::{future::Future, pin::Pin};

use thiserror::Error;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};

use lutum::HookRegistry;
use lutum_trace::{TraceEvent, TraceSnapshot};

/// A mutable, live-only evaluator over a stream of trace events plus a final
/// trace/artifact pair.
///
/// Probes do not own their dispatcher. An external [`ProbeDispatcher`] builds
/// the hook registry, forwards trace events, and calls [`Probe::finalize`]
/// after the traced future completes.
pub trait Probe: Send + 'static {
    type Score;
    type Artifact;
    type Error;

    fn register_hooks(&self, _cx: &mut ProbeContext<'_, Self>)
    where
        Self: Sized,
    {
    }

    fn on_trace_event(
        &mut self,
        event: &TraceEvent,
    ) -> Result<ProbeDecision<Self::Score>, Self::Error>;

    fn finalize(
        &mut self,
        trace: &TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error>;
}

#[derive(Debug, PartialEq)]
pub enum ProbeDecision<S> {
    Continue,
    Complete(S),
}

pub struct ProbeContext<'a, P: Probe> {
    hooks: &'a mut HookRegistry,
    dispatcher: ProbeHandle<P>,
}

impl<'a, P: Probe> ProbeContext<'a, P> {
    /// Returns a cloneable handle to the probe's event loop.
    ///
    /// This is a lower-level escape hatch primarily used by [`ProbeHookSlot`]
    /// implementors. Prefer [`register_hook`][ProbeContext::register_hook] for
    /// the common case.
    pub fn dispatcher(&self) -> ProbeHandle<P> {
        self.dispatcher.clone()
    }

    pub fn hooks_mut(&mut self) -> &mut HookRegistry {
        self.hooks
    }

    /// Replace the registry with the result of `f`.
    ///
    /// This is convenient because generated `register_*` methods consume and
    /// return the registry by value.
    pub fn update_hooks(&mut self, f: impl FnOnce(HookRegistry) -> HookRegistry) {
        let hooks = std::mem::take(self.hooks);
        *self.hooks = f(hooks);
    }

    /// Register a hook slot so this probe receives its calls.
    ///
    /// `Slot` must implement [`ProbeHookSlot<P>`], which is available for every
    /// slot whose `Stateful*Hook` trait `P` implements.
    ///
    /// ```ignore
    /// fn register_hooks(&self, cx: &mut ProbeContext<'_, Self>) {
    ///     cx.register_hook::<RewriteNumber>();
    ///     cx.register_hook::<DecorateLabel>();
    /// }
    /// ```
    pub fn register_hook<Slot: ProbeHookSlot<P>>(&mut self) {
        Slot::register(self);
    }
}

/// Enables a hook slot to be driven by a probe.
///
/// Implement this for a slot marker type (e.g. `RewriteNumber`) alongside the
/// corresponding `XxxHook` impl for [`ProbeDispatchHook<P, Slot>`]. Together
/// they allow a probe to call `cx.register_hook::<Slot>()` in
/// [`Probe::register_hooks`] without ever seeing `ProbeHandle` or
/// `ProbeDispatcher`.
pub trait ProbeHookSlot<P: Probe>: Sized {
    fn register(cx: &mut ProbeContext<'_, P>);
}

/// Generic hook implementation that routes calls through the probe's event loop.
///
/// Used by [`ProbeHookSlot`] implementors. One instance is created per hook
/// slot that a probe handles. The relevant `XxxHook` trait must be implemented
/// for `ProbeDispatchHook<P, Slot>` alongside a [`ProbeHookSlot`] impl for the
/// slot marker type; both together enable `cx.register_hook::<Slot>()`.
pub struct ProbeDispatchHook<P: Probe, Slot> {
    pub dispatcher: ProbeHandle<P>,
    _slot: std::marker::PhantomData<fn() -> Slot>,
}

impl<P: Probe, Slot> ProbeDispatchHook<P, Slot> {
    pub fn new(dispatcher: ProbeHandle<P>) -> Self {
        Self {
            dispatcher,
            _slot: std::marker::PhantomData,
        }
    }
}

type ProbeTaskFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
type ProbeTask<P> = Box<dyn for<'a> FnOnce(&'a mut P) -> ProbeTaskFuture<'a> + Send>;
pub type ProbeDispatchFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

enum ProbeMessage<P: Probe> {
    Dispatch(ProbeTask<P>),
    Trace(TraceEvent),
    Finalize {
        trace: TraceSnapshot,
        artifact: P::Artifact,
        reply: oneshot::Sender<Result<P::Score, ProbeRunError<P::Error>>>,
    },
}

pub struct ProbeDispatcher<P: Probe> {
    hooks: HookRegistry,
    runtime: ProbeRuntime<P>,
}

impl<P> ProbeDispatcher<P>
where
    P: Probe,
    P::Artifact: Send + 'static,
    P::Error: Send + 'static,
    P::Score: Send + 'static,
{
    pub fn new(probe: P) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let dispatcher = ProbeHandle { tx };
        let mut hooks = HookRegistry::new();

        {
            let mut cx = ProbeContext {
                hooks: &mut hooks,
                dispatcher: dispatcher.clone(),
            };
            probe.register_hooks(&mut cx);
        }

        let task = tokio::spawn(run_probe(probe, rx));

        Self {
            hooks,
            runtime: ProbeRuntime { dispatcher, task },
        }
    }

    pub fn into_parts(self) -> (HookRegistry, ProbeRuntime<P>) {
        (self.hooks, self.runtime)
    }
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
    P::Score: Send + 'static,
{
    pub fn dispatcher(&self) -> ProbeHandle<P> {
        self.dispatcher.clone()
    }

    /// Capture `future` with live [`TraceEvent`] forwarding and finish the
    /// probe when the artifact is ready.
    ///
    /// Like [`lutum_trace::capture_with_events`], this requires the active
    /// subscriber stack to include [`lutum_trace::layer`]. If the layer is
    /// absent, no live events are forwarded and `trace` will be empty.
    pub async fn run_live<F>(self, future: F) -> Result<P::Score, ProbeRunError<P::Error>>
    where
        F: Future<Output = P::Artifact>,
    {
        let dispatcher = self.dispatcher.clone();
        let collected = lutum_trace::capture_with_events(future, move |event| {
            let _ = dispatcher.send_trace(event);
        })
        .await;

        self.finish(collected.trace, collected.output).await
    }

    pub async fn finish(
        self,
        trace: TraceSnapshot,
        artifact: P::Artifact,
    ) -> Result<P::Score, ProbeRunError<P::Error>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let send_result = self.dispatcher.send_message(ProbeMessage::Finalize {
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

    fn send_trace(&self, event: TraceEvent) -> Result<(), ProbeDispatchError> {
        self.send_message(ProbeMessage::Trace(event))
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
    #[error("probe failed")]
    Probe(E),
    #[error("probe dispatcher closed")]
    DispatcherClosed,
    #[error("probe dispatcher task panicked: {source}")]
    Panicked {
        #[source]
        source: tokio::task::JoinError,
    },
}

async fn run_probe<P>(mut probe: P, mut rx: mpsc::UnboundedReceiver<ProbeMessage<P>>)
where
    P: Probe,
    P::Artifact: Send + 'static,
    P::Error: Send + 'static,
    P::Score: Send + 'static,
{
    let mut completed = None;
    let mut failure = None;

    while let Some(message) = rx.recv().await {
        match message {
            ProbeMessage::Dispatch(task) => task(&mut probe).await,
            ProbeMessage::Trace(event) => {
                if completed.is_some() || failure.is_some() {
                    continue;
                }

                match probe.on_trace_event(&event) {
                    Ok(ProbeDecision::Continue) => {}
                    Ok(ProbeDecision::Complete(score)) => completed = Some(score),
                    Err(source) => failure = Some(source),
                }
            }
            ProbeMessage::Finalize {
                trace,
                artifact,
                reply,
            } => {
                let result = if let Some(score) = completed.take() {
                    Ok(score)
                } else if let Some(source) = failure.take() {
                    Err(ProbeRunError::Probe(source))
                } else {
                    probe
                        .finalize(&trace, &artifact)
                        .map_err(ProbeRunError::Probe)
                };

                let _ = reply.send(result);
                break;
            }
        }
    }
}
