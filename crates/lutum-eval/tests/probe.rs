use std::{
    convert::Infallible,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use async_trait::async_trait;
use lutum::{
    HookRegistry, Lutum, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions,
};
use lutum_eval::{
    Eval, ObjectiveExt, Probe, ProbeDecision, ProbeDispatcher, ProbeHandle, ProbeRunError,
    ProbeScoreError, Score, maximize,
};
use tracing::instrument::WithSubscriber as _;
use tracing_subscriber::layer::SubscriberExt as _;

#[lutum::def_hook(singleton)]
async fn rewrite_number(_llm: &Lutum, value: usize) -> usize {
    value
}

#[lutum::def_hook(singleton)]
async fn decorate_label(_llm: &Lutum, label: &str) -> String {
    label.to_string()
}

type Validation = Result<(), &'static str>;

#[lutum::def_hook(singleton)]
async fn validate_step(_llm: &Lutum, _step: &str) -> Validation {
    Ok(())
}

#[lutum::hooks]
struct ProbeHooks {
    number_hooks: RewriteNumber,
    label_hooks: DecorateLabel,
    validation_hooks: ValidateStep,
}

#[derive(Debug, Eq, PartialEq)]
struct ProbeArtifact {
    number: usize,
    label: String,
}

#[derive(Debug, Eq, PartialEq)]
enum TestProbeError {
    TraceFailed,
    FinalizeFailed,
}

#[derive(Default)]
struct TimelineProbe {
    timeline: Vec<String>,
}

#[async_trait]
impl Probe for TimelineProbe {
    type Report = Vec<String>;
    type Artifact = ProbeArtifact;
    type Error = Infallible;

    async fn on_trace_event(
        &mut self,
        _ctx: &Lutum,
        event: &lutum_eval::TraceEvent,
    ) -> Result<ProbeDecision<Self::Report>, Self::Error> {
        if let lutum_eval::TraceEvent::Event { record, .. } = event
            && let Some(message) = record.message()
        {
            self.timeline.push(format!("trace:{message}"));
        }

        Ok(ProbeDecision::Continue)
    }

    async fn finalize(
        &mut self,
        _ctx: &Lutum,
        trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        self.timeline.push(format!(
            "finalize:{}:{}:{}",
            artifact.number,
            artifact.label,
            trace.has_event_message("after-hook")
        ));
        Ok(self.timeline.clone())
    }
}

#[async_trait]
impl StatefulRewriteNumberHook for TimelineProbe {
    async fn call_mut(&mut self, _llm: &Lutum, args: RewriteNumberArgs) -> usize {
        self.timeline.push(format!("hook:number:{}", args.value));
        args.value + 1
    }
}

#[async_trait]
impl StatefulDecorateLabelHook for TimelineProbe {
    async fn call_mut(&mut self, _llm: &Lutum, args: DecorateLabelArgs) -> String {
        self.timeline.push(format!("hook:label:{}", args.label));
        format!("probe:{}", args.label)
    }
}

struct NoHooksProbe;

#[async_trait]
impl Probe for NoHooksProbe {
    type Report = (bool, usize);
    type Artifact = usize;
    type Error = Infallible;

    async fn on_trace_event(
        &mut self,
        _ctx: &Lutum,
        _event: &lutum_eval::TraceEvent,
    ) -> Result<ProbeDecision<Self::Report>, Self::Error> {
        Ok(ProbeDecision::Continue)
    }

    async fn finalize(
        &mut self,
        _ctx: &Lutum,
        trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok((trace.has_event_message("plain-trace"), *artifact))
    }
}

struct EarlyCompleteProbe {
    finalized: Arc<AtomicBool>,
}

#[async_trait]
impl Probe for EarlyCompleteProbe {
    type Report = usize;
    type Artifact = usize;
    type Error = Infallible;

    async fn on_trace_event(
        &mut self,
        _ctx: &Lutum,
        event: &lutum_eval::TraceEvent,
    ) -> Result<ProbeDecision<Self::Report>, Self::Error> {
        if let lutum_eval::TraceEvent::Event { record, .. } = event
            && record.message() == Some("stop-early")
        {
            return Ok(ProbeDecision::Complete(99));
        }

        Ok(ProbeDecision::Continue)
    }

    async fn finalize(
        &mut self,
        _ctx: &Lutum,
        _trace: &lutum_eval::TraceSnapshot,
        _artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        self.finalized.store(true, Ordering::SeqCst);
        Ok(0)
    }
}

struct TraceErrorProbe;

#[async_trait]
impl Probe for TraceErrorProbe {
    type Report = usize;
    type Artifact = usize;
    type Error = TestProbeError;

    async fn on_trace_event(
        &mut self,
        _ctx: &Lutum,
        event: &lutum_eval::TraceEvent,
    ) -> Result<ProbeDecision<Self::Report>, Self::Error> {
        if let lutum_eval::TraceEvent::Event { record, .. } = event
            && record.message() == Some("trace-error")
        {
            return Err(TestProbeError::TraceFailed);
        }

        Ok(ProbeDecision::Continue)
    }

    async fn finalize(
        &mut self,
        _ctx: &Lutum,
        _trace: &lutum_eval::TraceSnapshot,
        _artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(1)
    }
}

struct FinalizeErrorProbe;

#[async_trait]
impl Probe for FinalizeErrorProbe {
    type Report = usize;
    type Artifact = usize;
    type Error = TestProbeError;

    async fn on_trace_event(
        &mut self,
        _ctx: &Lutum,
        _event: &lutum_eval::TraceEvent,
    ) -> Result<ProbeDecision<Self::Report>, Self::Error> {
        Ok(ProbeDecision::Continue)
    }

    async fn finalize(
        &mut self,
        _ctx: &Lutum,
        _trace: &lutum_eval::TraceSnapshot,
        _artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Err(TestProbeError::FinalizeFailed)
    }
}

#[derive(Default)]
struct HookErrorProbe {
    hook_calls: usize,
}

#[async_trait]
impl Probe for HookErrorProbe {
    type Report = usize;
    type Artifact = ();
    type Error = Infallible;

    async fn on_trace_event(
        &mut self,
        _ctx: &Lutum,
        _event: &lutum_eval::TraceEvent,
    ) -> Result<ProbeDecision<Self::Report>, Self::Error> {
        Ok(ProbeDecision::Continue)
    }

    async fn finalize(
        &mut self,
        _ctx: &Lutum,
        _trace: &lutum_eval::TraceSnapshot,
        _artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(self.hook_calls)
    }
}

#[async_trait]
impl StatefulValidateStepHook for HookErrorProbe {
    async fn call_mut(&mut self, _llm: &Lutum, args: ValidateStepArgs) -> Validation {
        self.hook_calls += 1;
        if args.step == "blocked" {
            Err("blocked")
        } else {
            Ok(())
        }
    }
}

fn timeline_hooks(dispatcher: ProbeHandle<TimelineProbe>) -> ProbeHooks {
    ProbeHooks::new()
        .with_rewrite_number((
            std::marker::PhantomData::<fn() -> Lutum>,
            {
                let dispatcher = dispatcher.clone();
                move |ctx: Lutum, args: RewriteNumberArgs| {
                    let dispatcher = dispatcher.clone();
                    async move {
                        dispatcher
                            .dispatch(move |probe| {
                                Box::pin(async move {
                                    <TimelineProbe as StatefulRewriteNumberHook>::call_mut(
                                        probe, &ctx, args,
                                    )
                                    .await
                                })
                            })
                            .await
                            .expect("probe dispatcher alive")
                    }
                }
            },
        ))
        .with_decorate_label((
            std::marker::PhantomData::<fn() -> Lutum>,
            {
                let dispatcher = dispatcher.clone();
                move |ctx: Lutum, args: DecorateLabelArgs| {
                    let dispatcher = dispatcher.clone();
                    async move {
                        dispatcher
                            .dispatch(move |probe| {
                                Box::pin(async move {
                                    <TimelineProbe as StatefulDecorateLabelHook>::call_mut(
                                        probe, &ctx, args,
                                    )
                                    .await
                                })
                            })
                            .await
                            .expect("probe dispatcher alive")
                    }
                }
            },
        ))
}

fn hook_error_hooks(dispatcher: ProbeHandle<HookErrorProbe>) -> ProbeHooks {
    ProbeHooks::new().with_validate_step((
        std::marker::PhantomData::<fn() -> Lutum>,
        move |ctx: Lutum, args: ValidateStepArgs| {
            let dispatcher = dispatcher.clone();
            async move {
                dispatcher
                    .dispatch(move |probe| {
                        Box::pin(async move {
                            <HookErrorProbe as StatefulValidateStepHook>::call_mut(
                                probe, &ctx, args,
                            )
                            .await
                        })
                    })
                    .await
                    .expect("probe dispatcher alive")
            }
        },
    ))
}

fn make_lutum(hooks: HookRegistry) -> Lutum {
    Lutum::with_hooks(
        Arc::new(MockLlmAdapter::new()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        hooks,
    )
}

fn subscriber() -> impl tracing::Subscriber + Send + Sync {
    tracing_subscriber::registry().with(lutum_trace::layer())
}

struct FinalArtifactEval;

#[async_trait]
impl Eval for FinalArtifactEval {
    type Report = usize;
    type Artifact = usize;
    type Error = Infallible;

    async fn evaluate(
        &self,
        _ctx: &Lutum,
        trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(*artifact + usize::from(trace.has_event_message("eval-probe")))
    }
}

#[tokio::test]
async fn probe_without_hooks_can_finalize() {
    let llm = make_lutum(HookRegistry::new());
    let (_, runtime) = ProbeDispatcher::new(NoHooksProbe).into_parts();

    let report = runtime
        .run_future(
            &llm,
            async {
                tracing::info!("plain-trace");
                7usize
            }
            .with_subscriber(subscriber()),
        )
        .await
        .unwrap();

    assert_eq!(report, (true, 7));
}

#[tokio::test]
async fn probe_dispatcher_serializes_trace_events_and_hook_rpcs() {
    let (_, runtime) = ProbeDispatcher::new(TimelineProbe::default()).into_parts();
    let probe_hooks = timeline_hooks(runtime.dispatcher());
    let llm = make_lutum(HookRegistry::new());
    let runtime_llm = llm.clone();

    let report = runtime
        .run_future(
            &llm,
            async move {
                tracing::info!("before-hook");
                let number = probe_hooks.rewrite_number(&runtime_llm, 2).await;
                let label = probe_hooks.decorate_label(&runtime_llm, "seed").await;
                tracing::info!("after-hook");

                ProbeArtifact { number, label }
            }
            .with_subscriber(subscriber()),
        )
        .await
        .unwrap();

    assert_eq!(
        report,
        vec![
            "trace:before-hook".to_string(),
            "hook:number:2".to_string(),
            "hook:label:seed".to_string(),
            "trace:after-hook".to_string(),
            "finalize:3:probe:seed:true".to_string(),
        ]
    );
}

#[tokio::test]
async fn complete_short_circuits_finalize() {
    let llm = make_lutum(HookRegistry::new());
    let finalized = Arc::new(AtomicBool::new(false));
    let (_, runtime) = ProbeDispatcher::new(EarlyCompleteProbe {
        finalized: finalized.clone(),
    })
    .into_parts();

    let report = runtime
        .run_future(
            &llm,
            async {
                tracing::info!("stop-early");
                7usize
            }
            .with_subscriber(subscriber()),
        )
        .await
        .unwrap();

    assert_eq!(report, 99);
    assert!(!finalized.load(Ordering::SeqCst));
}

#[tokio::test]
async fn hook_proxy_can_return_probe_defined_errors() {
    let (_, runtime) = ProbeDispatcher::new(HookErrorProbe::default()).into_parts();
    let probe_hooks = hook_error_hooks(runtime.dispatcher());
    let llm = make_lutum(HookRegistry::new());
    let runtime_llm = llm.clone();

    let report = runtime
        .run_future(
            &llm,
            async move {
                let result = probe_hooks.validate_step(&runtime_llm, "blocked").await;
                assert_eq!(result, Err("blocked"));
            }
            .with_subscriber(subscriber()),
        )
        .await
        .unwrap();

    assert_eq!(report, 1);
}

#[tokio::test]
async fn trace_errors_are_returned_from_runtime() {
    let llm = make_lutum(HookRegistry::new());
    let (_, runtime) = ProbeDispatcher::new(TraceErrorProbe).into_parts();

    let err = runtime
        .run_future(
            &llm,
            async {
                tracing::info!("trace-error");
                1usize
            }
            .with_subscriber(subscriber()),
        )
        .await
        .unwrap_err();

    assert!(matches!(
        err,
        ProbeRunError::Probe(TestProbeError::TraceFailed)
    ));
}

#[tokio::test]
async fn finalize_errors_are_returned_from_runtime() {
    let llm = make_lutum(HookRegistry::new());
    let (_, runtime) = ProbeDispatcher::new(FinalizeErrorProbe).into_parts();

    let err = runtime
        .run_future(&llm, async { 1usize }.with_subscriber(subscriber()))
        .await
        .unwrap_err();

    assert!(matches!(
        err,
        ProbeRunError::Probe(TestProbeError::FinalizeFailed)
    ));
}

#[tokio::test]
async fn eval_can_run_through_probe_runtime() {
    let llm = make_lutum(HookRegistry::new());
    let (_, runtime) = ProbeDispatcher::new(FinalArtifactEval).into_parts();

    let report = runtime
        .run_future(
            &llm,
            async {
                tracing::info!("eval-probe");
                7usize
            }
            .with_subscriber(subscriber()),
        )
        .await
        .unwrap();

    assert_eq!(report, 8);
}

#[tokio::test]
async fn probe_runtime_can_score_reports_with_an_objective() {
    let llm = make_lutum(HookRegistry::new());
    let (_, runtime) = ProbeDispatcher::new(FinalArtifactEval).into_parts();
    let objective = maximize(|report: &usize| Score::new_clamped(*report as f32 / 10.0));

    let scored = runtime
        .scored_by(&objective)
        .run_future(
            &llm,
            async {
                tracing::info!("eval-probe");
                7usize
            }
            .with_subscriber(subscriber()),
        )
        .await
        .unwrap();

    assert_eq!(scored.score, Score::new_clamped(0.8));
    assert_eq!(scored.report, 8);
}

#[tokio::test]
async fn objective_failures_are_returned_from_probe_scoring() {
    let llm = make_lutum(HookRegistry::new());
    let (_, runtime) = ProbeDispatcher::new(FinalArtifactEval).into_parts();
    let objective = maximize(|_report: &usize| Score::new_clamped(1.0)).map_error(|_| "never");

    let scored = runtime
        .scored_by(&objective)
        .run_future(
            &llm,
            async {
                tracing::info!("eval-probe");
                7usize
            }
            .with_subscriber(subscriber()),
        )
        .await
        .unwrap();

    assert_eq!(scored.score, Score::pass());
    assert_eq!(scored.report, 8);

    let (_, runtime) = ProbeDispatcher::new(FinalArtifactEval).into_parts();
    struct FailingObjective;
    impl lutum_eval::Objective<usize> for FailingObjective {
        type Error = &'static str;

        fn score(&self, _report: &usize) -> Result<Score, Self::Error> {
            Err("bad objective")
        }
    }

    let err = runtime
        .scored_by(&FailingObjective)
        .run_future(&llm, async { 1usize }.with_subscriber(subscriber()))
        .await
        .unwrap_err();

    assert!(matches!(err, ProbeScoreError::Objective("bad objective")));
}
