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
use lutum_eval::register_probe_hook;
use lutum_eval::{Metric, Probe, ProbeContext, ProbeDecision, ProbeDispatcher, ProbeRunError};
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

// ---------------------------------------------------------------------------
// Probe implementations
// ---------------------------------------------------------------------------

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
    type Score = Vec<String>;
    type Artifact = ProbeArtifact;
    type Error = Infallible;

    fn register_hooks(&self, cx: &mut ProbeContext<'_, Self>) {
        register_probe_hook!(cx, RewriteNumber);
        register_probe_hook!(cx, DecorateLabel);
    }

    async fn on_trace_event(
        &mut self,
        _ctx: &Lutum,
        event: &lutum_eval::TraceEvent,
    ) -> Result<ProbeDecision<Self::Score>, Self::Error> {
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
    ) -> Result<Self::Score, Self::Error> {
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
    type Score = (bool, usize);
    type Artifact = usize;
    type Error = Infallible;

    async fn on_trace_event(
        &mut self,
        _ctx: &Lutum,
        _event: &lutum_eval::TraceEvent,
    ) -> Result<ProbeDecision<Self::Score>, Self::Error> {
        Ok(ProbeDecision::Continue)
    }

    async fn finalize(
        &mut self,
        _ctx: &Lutum,
        trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok((trace.has_event_message("plain-trace"), *artifact))
    }
}

struct EarlyCompleteProbe {
    finalized: Arc<AtomicBool>,
}

#[async_trait]
impl Probe for EarlyCompleteProbe {
    type Score = usize;
    type Artifact = usize;
    type Error = Infallible;

    async fn on_trace_event(
        &mut self,
        _ctx: &Lutum,
        event: &lutum_eval::TraceEvent,
    ) -> Result<ProbeDecision<Self::Score>, Self::Error> {
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
    ) -> Result<Self::Score, Self::Error> {
        self.finalized.store(true, Ordering::SeqCst);
        Ok(0)
    }
}

struct TraceErrorProbe;

#[async_trait]
impl Probe for TraceErrorProbe {
    type Score = usize;
    type Artifact = usize;
    type Error = TestProbeError;

    async fn on_trace_event(
        &mut self,
        _ctx: &Lutum,
        event: &lutum_eval::TraceEvent,
    ) -> Result<ProbeDecision<Self::Score>, Self::Error> {
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
    ) -> Result<Self::Score, Self::Error> {
        Ok(1)
    }
}

struct FinalizeErrorProbe;

#[async_trait]
impl Probe for FinalizeErrorProbe {
    type Score = usize;
    type Artifact = usize;
    type Error = TestProbeError;

    async fn on_trace_event(
        &mut self,
        _ctx: &Lutum,
        _event: &lutum_eval::TraceEvent,
    ) -> Result<ProbeDecision<Self::Score>, Self::Error> {
        Ok(ProbeDecision::Continue)
    }

    async fn finalize(
        &mut self,
        _ctx: &Lutum,
        _trace: &lutum_eval::TraceSnapshot,
        _artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Err(TestProbeError::FinalizeFailed)
    }
}

#[derive(Default)]
struct HookErrorProbe {
    hook_calls: usize,
}

#[async_trait]
impl Probe for HookErrorProbe {
    type Score = usize;
    type Artifact = ();
    type Error = Infallible;

    fn register_hooks(&self, cx: &mut ProbeContext<'_, Self>) {
        register_probe_hook!(cx, ValidateStep);
    }

    async fn on_trace_event(
        &mut self,
        _ctx: &Lutum,
        _event: &lutum_eval::TraceEvent,
    ) -> Result<ProbeDecision<Self::Score>, Self::Error> {
        Ok(ProbeDecision::Continue)
    }

    async fn finalize(
        &mut self,
        _ctx: &Lutum,
        _trace: &lutum_eval::TraceSnapshot,
        _artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
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

struct FinalArtifactMetric;

#[async_trait]
impl Metric for FinalArtifactMetric {
    type Score = usize;
    type Artifact = usize;
    type Error = Infallible;

    async fn evaluate(
        &self,
        _ctx: &Lutum,
        trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Score, Self::Error> {
        Ok(*artifact + usize::from(trace.has_event_message("metric-probe")))
    }
}

#[tokio::test]
async fn probe_without_hooks_can_finalize() {
    let llm = make_lutum(HookRegistry::new());
    let (_, runtime) = ProbeDispatcher::new(NoHooksProbe).into_parts();

    let score = runtime
        .run_live(
            &llm,
            async {
                tracing::info!("plain-trace");
                7usize
            }
            .with_subscriber(subscriber()),
        )
        .await
        .unwrap();

    assert_eq!(score, (true, 7));
}

#[tokio::test]
async fn probe_dispatcher_serializes_trace_events_and_hook_rpcs() {
    let (hooks, runtime) = ProbeDispatcher::new(TimelineProbe::default()).into_parts();
    let llm = make_lutum(hooks);
    let runtime_llm = llm.clone();

    let score = runtime
        .run_live(
            &llm,
            async move {
                tracing::info!("before-hook");
                let number = runtime_llm.rewrite_number(2).await;
                let label = runtime_llm.decorate_label("seed").await;
                tracing::info!("after-hook");

                ProbeArtifact { number, label }
            }
            .with_subscriber(subscriber()),
        )
        .await
        .unwrap();

    assert_eq!(
        score,
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

    let score = runtime
        .run_live(
            &llm,
            async {
                tracing::info!("stop-early");
                7usize
            }
            .with_subscriber(subscriber()),
        )
        .await
        .unwrap();

    assert_eq!(score, 99);
    assert!(!finalized.load(Ordering::SeqCst));
}

#[tokio::test]
async fn hook_proxy_can_return_probe_defined_errors() {
    let (hooks, runtime) = ProbeDispatcher::new(HookErrorProbe::default()).into_parts();
    let llm = make_lutum(hooks);
    let runtime_llm = llm.clone();

    let score = runtime
        .run_live(
            &llm,
            async move {
                let result = runtime_llm.validate_step("blocked").await;
                assert_eq!(result, Err("blocked"));
            }
            .with_subscriber(subscriber()),
        )
        .await
        .unwrap();

    assert_eq!(score, 1);
}

#[tokio::test]
async fn trace_errors_are_returned_from_runtime() {
    let llm = make_lutum(HookRegistry::new());
    let (_, runtime) = ProbeDispatcher::new(TraceErrorProbe).into_parts();

    let err = runtime
        .run_live(
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
        .run_live(&llm, async { 1usize }.with_subscriber(subscriber()))
        .await
        .unwrap_err();

    assert!(matches!(
        err,
        ProbeRunError::Probe(TestProbeError::FinalizeFailed)
    ));
}

#[tokio::test]
async fn metric_can_run_through_probe_runtime() {
    let llm = make_lutum(HookRegistry::new());
    let (_, runtime) = ProbeDispatcher::new(FinalArtifactMetric).into_parts();

    let score = runtime
        .run_live(
            &llm,
            async {
                tracing::info!("metric-probe");
                7usize
            }
            .with_subscriber(subscriber()),
        )
        .await
        .unwrap();

    assert_eq!(score, 8);
}
