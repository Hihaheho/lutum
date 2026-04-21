use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use futures::executor::block_on;
use lutum::{
    AdapterStructuredCompletionRequest, AdapterStructuredTurn, AdapterTextTurn, AgentError,
    CompletionAdapter, CompletionEventStream, CompletionRequest,
    ErasedStructuredCompletionEventStream, ErasedStructuredTurnEventStream,
    ErasedTextTurnEventStream, HookReentrancyError, InputMessageRole, Lutum, LutumHooks,
    MockLlmAdapter, ModelInput, ModelInputItem, OperationKind, RequestExtensions,
    ResolveUsageEstimate, SharedPoolBudgetManager, SharedPoolBudgetOptions, Stateful, TurnAdapter,
    Usage, UsageRecoveryAdapter, budget::UsageEstimate,
};
use lutum_trace::FieldValue;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[lutum::hooks]
trait TestHooks {
    #[hook(singleton)]
    async fn select_label(default: String) -> String {
        default
    }

    #[hook(always)]
    async fn format_label(label: &str) -> String {
        format!("default:{label}")
    }

    #[hook(fallback)]
    async fn choose_label(label: &str) -> String {
        format!("default:{label}")
    }

    #[hook(always, chain = lutum::ShortCircuit<String, String>)]
    async fn validate_chain_label(label: &str) -> Result<String, String> {
        Err(format!("default-blocked:{label}"))
    }

    #[hook(always, chain = lutum::ShortCircuit<String, String>)]
    async fn transform_chain_label(label: &str) -> Result<String, String> {
        Ok(format!("default:{label}"))
    }

    #[hook(fallback, chain = lutum::FirstSuccess<String>)]
    async fn choose_chain_label(label: &str) -> Option<String> {
        Some(format!("default:{label}"))
    }

    #[hook(fallback, chain = lutum::FirstSuccess<String>)]
    async fn choose_chain_default_after_hooks(label: &str) -> Option<String> {
        Some(format!("fallback-default:{label}"))
    }

    #[hook(singleton)]
    async fn next_counter(seed: usize) -> CounterResult {
        Ok(seed)
    }

    #[hook(singleton)]
    async fn describe_label(label: &str) -> String {
        label.to_string()
    }
}

// aggregate: each hook contributes independently (no `last`), outputs collected and reduced.
#[derive(Default)]
struct JoinStrings;

#[async_trait]
impl lutum::Aggregate<String> for JoinStrings {
    async fn call(&self, outputs: Vec<String>) -> String {
        outputs.join(", ")
    }
}

#[lutum::hooks]
trait AccumulateHooks {
    #[hook(always, aggregate = JoinStrings)]
    async fn accumulate_label(label: &str) -> String {
        format!("default:{label}")
    }

    #[hook(always, chain = IsShortCircuitString, aggregate = JoinStrings)]
    async fn accumulate_chain_label(label: &str) -> String {
        format!("default:{label}")
    }

    #[hook(always, finalize = WrapResult)]
    async fn finalized_label(label: &str) -> String {
        format!("default:{label}")
    }

    #[hook(always, chain = IsShortCircuitString, finalize = WrapResult)]
    async fn chain_finalized_label(label: &str) -> String {
        format!("default:{label}")
    }
}

#[lutum::hooks]
trait OutputIntoHooks {
    #[hook(always, aggregate = CollectIntoLabels, output = CollectedLabels)]
    async fn accumulate_label_into(label: &str) -> String {
        format!("default:{label}")
    }

    #[hook(fallback, aggregate = CollectIntoLabels, output = CollectedLabels)]
    async fn fallback_accumulate_label_into(label: &str) -> String {
        format!("fallback:{label}")
    }

    #[hook(fallback, chain = IsShortCircuitString, aggregate = CollectIntoLabels, output = CollectedLabels)]
    async fn fallback_accumulate_chain_label_into(label: &str) -> String {
        format!("fallback:{label}")
    }

    #[hook(always, chain = IsShortCircuitString, aggregate = CollectIntoLabels, output = CollectedLabels)]
    async fn accumulate_chain_label_into(label: &str) -> String {
        format!("default:{label}")
    }

    #[hook(always, finalize = WrapLabelInto, output = WrappedLabel)]
    async fn finalized_label_into(label: &str) -> String {
        format!("default:{label}")
    }
}

#[lutum::impl_hook(SelectLabel)]
async fn prefix_label(default: String) -> String {
    format!("hooked:{default}")
}

#[lutum::impl_hook(SelectLabel)]
async fn suffix_label(default: String) -> String {
    format!("{default}:suffix")
}

#[lutum::impl_hook(FormatLabel)]
async fn append_suffix(source_label: &str, last: Option<String>) -> String {
    let previous = last.expect("always hooks should receive the default result");
    assert_eq!(previous, format!("default:{source_label}"));
    format!("{previous}:hook")
}

#[lutum::impl_hook(ChooseLabel)]
async fn pick_registered_label(label: &str, last: Option<String>) -> String {
    assert!(last.is_none(), "fallback chains should start from None");
    format!("hook:{label}")
}

#[lutum::impl_hook(ValidateChainLabel)]
async fn append_chain_suffix(label: &str) -> Result<String, String> {
    Ok(format!("hooked:{label}"))
}

#[lutum::impl_hook(TransformChainLabel)]
async fn transform_chain_middle(label: &str) -> Result<String, String> {
    Ok(format!("mid:{label}"))
}

#[lutum::impl_hook(TransformChainLabel)]
async fn transform_chain_final(label: &str) -> Result<String, String> {
    Ok(format!("final:{label}"))
}

#[lutum::impl_hook(ChooseChainLabel)]
async fn choose_none(_label: &str) -> Option<String> {
    None
}

#[lutum::impl_hook(ChooseChainLabel)]
async fn choose_special(label: &str) -> Option<String> {
    Some(format!("hook:{label}"))
}

#[lutum::impl_hook(ChooseChainDefaultAfterHooks)]
async fn choose_none_again(_label: &str) -> Option<String> {
    None
}

#[lutum::impl_hook(AccumulateLabel)]
async fn accumulate_hook_a(label: &str) -> String {
    format!("hook-a:{label}")
}

#[lutum::impl_hook(AccumulateLabel)]
async fn accumulate_hook_b(label: &str) -> String {
    format!("hook-b:{label}")
}

// aggregate + chain: early exit during aggregation.
struct IsShortCircuitString;

impl Default for IsShortCircuitString {
    fn default() -> Self {
        Self
    }
}

#[async_trait]
impl lutum::Chain<String> for IsShortCircuitString {
    async fn call(&self, s: &String) -> std::ops::ControlFlow<()> {
        if s.starts_with("stop:") {
            std::ops::ControlFlow::Break(())
        } else {
            std::ops::ControlFlow::Continue(())
        }
    }
}

#[lutum::impl_hook(AccumulateChainLabel)]
async fn accumulate_chain_hook_stop(_label: &str) -> String {
    "stop:early".to_owned()
}

#[lutum::impl_hook(AccumulateChainLabel)]
async fn accumulate_chain_hook_unreachable(_label: &str) -> String {
    panic!("must not be called after stop")
}

// finalize: fold runs first, then finalize wraps the result.
#[derive(Default)]
struct WrapResult;

#[async_trait]
impl lutum::Finalize<String> for WrapResult {
    async fn call(&self, output: String) -> String {
        format!("[{output}]")
    }
}

#[lutum::impl_hook(FinalizedLabel)]
async fn finalized_append(label: &str, last: Option<String>) -> String {
    format!("{}+{label}", last.unwrap())
}

// chain + finalize: finalize captures early exits from chain dispatch.
#[lutum::impl_hook(ChainFinalizedLabel)]
async fn chain_finalized_stop(_label: &str) -> String {
    "stop:chain".to_owned()
}

#[lutum::impl_hook(ChainFinalizedLabel)]
async fn chain_finalized_unreachable(_label: &str) -> String {
    panic!("must not be called after stop")
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CollectedLabels(Vec<String>);

#[derive(Clone, Debug, Eq, PartialEq)]
struct WrappedLabel(String);

#[derive(Default)]
struct CollectIntoLabels;

#[async_trait]
impl lutum::AggregateInto<String, CollectedLabels> for CollectIntoLabels {
    async fn call(&self, outputs: Vec<String>) -> CollectedLabels {
        CollectedLabels(outputs)
    }
}

#[derive(Default)]
struct WrapLabelInto;

#[async_trait]
impl lutum::FinalizeInto<String, WrappedLabel> for WrapLabelInto {
    async fn call(&self, output: String) -> WrappedLabel {
        WrappedLabel(format!("[{output}]"))
    }
}

#[lutum::impl_hook(AccumulateLabelInto)]
async fn accumulate_into_hook_a(label: &str) -> String {
    format!("hook-a:{label}")
}

#[lutum::impl_hook(FallbackAccumulateLabelInto)]
async fn fallback_accumulate_into_hook(label: &str) -> String {
    format!("hook:{label}")
}

#[lutum::impl_hook(FallbackAccumulateChainLabelInto)]
async fn fallback_accumulate_chain_into_hook_a(label: &str) -> String {
    format!("hook-a:{label}")
}

#[lutum::impl_hook(FallbackAccumulateChainLabelInto)]
async fn fallback_accumulate_chain_into_hook_b(label: &str) -> String {
    format!("hook-b:{label}")
}

#[lutum::impl_hook(AccumulateChainLabelInto)]
async fn accumulate_chain_into_hook_stop(_label: &str) -> String {
    "stop:early".to_owned()
}

#[lutum::impl_hook(AccumulateChainLabelInto)]
async fn accumulate_chain_into_hook_unreachable(_label: &str) -> String {
    panic!("must not be called after stop")
}

#[lutum::impl_hook(FinalizedLabelInto)]
async fn finalized_into_append(label: &str, last: Option<String>) -> String {
    format!("{}+{label}", last.unwrap())
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CounterError {
    Reentered(HookReentrancyError),
}

type CounterResult = Result<usize, CounterError>;

struct CountingHook {
    next: usize,
}

#[async_trait]
impl StatefulNextCounter for CountingHook {
    fn on_reentrancy(err: HookReentrancyError) -> CounterResult {
        Err(CounterError::Reentered(err))
    }

    async fn call_mut(&mut self, seed: usize) -> CounterResult {
        let current = self.next.max(seed);
        self.next = current + 1;
        Ok(current)
    }
}

struct ReentrantCounter {
    hooks: Arc<OnceLock<TestHooks>>,
}

#[async_trait]
impl StatefulNextCounter for ReentrantCounter {
    fn on_reentrancy(err: HookReentrancyError) -> CounterResult {
        Err(CounterError::Reentered(err))
    }

    async fn call_mut(&mut self, seed: usize) -> CounterResult {
        if seed == 0 {
            Ok(0)
        } else {
            self.hooks
                .get()
                .expect("reentrant hook container must be initialized")
                .next_counter(seed - 1)
                .await
        }
    }
}

struct NestedLabelHook {
    hooks: TestHooks,
}

#[async_trait]
impl StatefulDescribeLabel for NestedLabelHook {
    async fn call_mut(&mut self, label: String) -> String {
        self.hooks.select_label(label).await
    }
}

fn test_context(hooks: LutumHooks) -> Lutum {
    Lutum::with_hooks(
        Arc::new(MockLlmAdapter::new()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        hooks,
    )
}

fn full_context(hooks: LutumHooks) -> Lutum {
    let adapter = Arc::new(NullAdapter);
    Lutum::from_parts_with_hooks(
        adapter.clone(),
        adapter.clone(),
        adapter,
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        hooks,
    )
}

fn input() -> ModelInput {
    ModelInput::from_items(vec![ModelInputItem::text(InputMessageRole::User, "hello")])
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Summary {
    answer: String,
}

struct NullAdapter;

#[async_trait]
impl TurnAdapter for NullAdapter {
    async fn text_turn(
        &self,
        _input: ModelInput,
        _turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        Ok(Box::pin(futures::stream::empty()) as ErasedTextTurnEventStream)
    }

    async fn structured_turn(
        &self,
        _input: ModelInput,
        _turn: AdapterStructuredTurn,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
        Ok(Box::pin(futures::stream::empty()) as ErasedStructuredTurnEventStream)
    }
}

#[async_trait]
impl CompletionAdapter for NullAdapter {
    async fn completion(
        &self,
        _request: CompletionRequest,
        _extensions: &RequestExtensions,
    ) -> Result<CompletionEventStream, AgentError> {
        Ok(Box::pin(futures::stream::empty()) as CompletionEventStream)
    }

    async fn structured_completion(
        &self,
        _request: AdapterStructuredCompletionRequest,
        _extensions: &RequestExtensions,
    ) -> Result<ErasedStructuredCompletionEventStream, AgentError> {
        Ok(Box::pin(futures::stream::empty()) as ErasedStructuredCompletionEventStream)
    }
}

#[async_trait]
impl UsageRecoveryAdapter for NullAdapter {
    async fn recover_usage(
        &self,
        _kind: OperationKind,
        _request_id: &str,
    ) -> Result<Option<Usage>, AgentError> {
        Ok(None)
    }
}

struct FixedEstimate {
    estimate: UsageEstimate,
}

#[async_trait]
impl ResolveUsageEstimate for FixedEstimate {
    async fn call(&self, _extensions: &RequestExtensions, _kind: OperationKind) -> UsageEstimate {
        self.estimate
    }
}

struct RecordOperationKinds {
    seen: Arc<Mutex<Vec<OperationKind>>>,
}

#[async_trait]
impl ResolveUsageEstimate for RecordOperationKinds {
    async fn call(&self, _extensions: &RequestExtensions, kind: OperationKind) -> UsageEstimate {
        self.seen.lock().unwrap().push(kind);
        UsageEstimate::zero()
    }
}

struct RecordExtensionEstimate {
    seen: Arc<Mutex<Vec<u64>>>,
}

#[async_trait]
impl ResolveUsageEstimate for RecordExtensionEstimate {
    async fn call(&self, extensions: &RequestExtensions, _kind: OperationKind) -> UsageEstimate {
        let total_tokens = extensions
            .get::<UsageEstimate>()
            .map(|estimate| estimate.total_tokens)
            .unwrap_or(0);
        self.seen.lock().unwrap().push(total_tokens);
        UsageEstimate::zero()
    }
}

#[test]
fn singleton_hook_uses_default_when_unregistered() {
    let hooks = TestHooks::new();

    let selected = block_on(hooks.select_label("base".into()));

    assert_eq!(selected, "base");
}

#[test]
fn singleton_hook_uses_registered_override() {
    let hooks = TestHooks::new().with_select_label(PrefixLabel);

    let selected = block_on(hooks.select_label("base".into()));

    assert_eq!(selected, "hooked:base");
}

#[tokio::test]
async fn singleton_hook_warns_and_uses_last_registered_override() {
    let collected = lutum_trace::test::collect(async {
        let hooks = TestHooks::new()
            .with_select_label(PrefixLabel)
            .with_select_label(SuffixLabel);

        hooks.select_label("base".into()).await
    })
    .await;

    assert_eq!(collected.output, "base:suffix");

    let warning = collected
        .trace
        .events()
        .iter()
        .find(|event| {
            event.level == "WARN"
                && event.message()
                    == Some("singleton hook registration overwritten; last registered hook wins")
        })
        .expect("expected singleton overwrite warning");

    assert_eq!(
        warning.field("slot"),
        Some(&FieldValue::Str("select_label".to_string()))
    );
}

#[test]
fn always_hook_uses_default_without_last_when_unregistered() {
    let hooks = TestHooks::new();

    let selected = block_on(hooks.format_label("base"));

    assert_eq!(selected, "default:base");
}

#[test]
fn always_hook_passes_default_result_to_registered_hook() {
    let hooks = TestHooks::new().with_format_label(AppendSuffix);

    let selected = block_on(hooks.format_label("base"));

    assert_eq!(selected, "default:base:hook");
}

#[test]
fn fallback_hook_uses_default_without_last_when_unregistered() {
    let hooks = TestHooks::new();

    let selected = block_on(hooks.choose_label("base"));

    assert_eq!(selected, "default:base");
}

#[test]
fn fallback_hook_starts_registered_chain_without_default_result() {
    let hooks = TestHooks::new().with_choose_label(PickRegisteredLabel);

    let selected = block_on(hooks.choose_label("base"));

    assert_eq!(selected, "hook:base");
}

#[test]
fn always_chain_short_circuit_stops_after_default_break() {
    let hooks = TestHooks::new().with_validate_chain_label(AppendChainSuffix);

    let result = block_on(hooks.validate_chain_label("base"));

    assert_eq!(result, Err("default-blocked:base".into()));
}

#[test]
fn always_chain_returns_last_hook_result_when_all_continue() {
    let hooks = TestHooks::new()
        .with_transform_chain_label(TransformChainMiddle)
        .with_transform_chain_label(TransformChainFinal);

    let result = block_on(hooks.transform_chain_label("base"));

    assert_eq!(result, Ok("final:base".into()));
}

#[test]
fn fallback_chain_first_success_stops_on_first_some() {
    let hooks = TestHooks::new()
        .with_choose_chain_label(ChooseNone)
        .with_choose_chain_label(ChooseSpecial);

    let result = block_on(hooks.choose_chain_label("base"));

    assert_eq!(result, Some("hook:base".into()));
}

#[test]
fn fallback_chain_runs_default_when_all_hooks_continue() {
    let hooks = TestHooks::new().with_choose_chain_default_after_hooks(ChooseNoneAgain);

    let result = block_on(hooks.choose_chain_default_after_hooks("base"));

    assert_eq!(result, Some("fallback-default:base".into()));
}

#[test]
fn stateful_hook_mutates_state_without_interior_mutability() {
    let hooks = TestHooks::new().with_next_counter(Stateful::new(CountingHook { next: 0 }));

    let first = block_on(hooks.next_counter(10));
    let second = block_on(hooks.next_counter(10));

    assert_eq!(first, Ok(10));
    assert_eq!(second, Ok(11));
}

#[test]
fn stateful_hook_reentrancy_can_return_a_typed_error() {
    let shared_hooks = Arc::new(OnceLock::new());
    let hooks = TestHooks::new().with_next_counter(Stateful::new(ReentrantCounter {
        hooks: Arc::clone(&shared_hooks),
    }));
    assert!(shared_hooks.set(hooks.clone()).is_ok());

    let result = block_on(hooks.next_counter(1));

    assert_eq!(
        result,
        Err(CounterError::Reentered(HookReentrancyError {
            slot: "next_counter",
            hook_type: std::any::type_name::<ReentrantCounter>(),
        }))
    );
}

#[test]
fn stateful_hook_can_call_other_hooks_without_registry_deadlock() {
    let hooks = TestHooks::new().with_select_label(PrefixLabel);
    let nested_hooks = hooks.clone();
    let hooks = hooks.with_describe_label(Stateful::new(NestedLabelHook {
        hooks: nested_hooks,
    }));

    let described = block_on(hooks.describe_label("base"));

    assert_eq!(described, "hooked:base");
}

#[test]
fn resolve_usage_estimate_defaults_to_zero() {
    let ctx = test_context(LutumHooks::new());

    let estimate =
        block_on(ctx.resolve_usage_estimate(&RequestExtensions::new(), OperationKind::TextTurn));

    assert_eq!(estimate, UsageEstimate::zero());
}

#[test]
fn resolve_usage_estimate_reads_request_extensions_by_default() {
    let ctx = test_context(LutumHooks::new());
    let mut extensions = RequestExtensions::new();
    extensions.insert(UsageEstimate {
        total_tokens: 42,
        ..UsageEstimate::zero()
    });

    let estimate = block_on(ctx.resolve_usage_estimate(&extensions, OperationKind::TextTurn));

    assert_eq!(
        estimate,
        UsageEstimate {
            total_tokens: 42,
            ..UsageEstimate::zero()
        }
    );
}

#[test]
fn resolve_usage_estimate_registered_override_wins_over_default_extensions_lookup() {
    let ctx = test_context(
        LutumHooks::new().with_resolve_usage_estimate(FixedEstimate {
            estimate: UsageEstimate {
                total_tokens: 7,
                ..UsageEstimate::zero()
            },
        }),
    );
    let mut extensions = RequestExtensions::new();
    extensions.insert(UsageEstimate {
        total_tokens: 42,
        ..UsageEstimate::zero()
    });

    let estimate = block_on(ctx.resolve_usage_estimate(&extensions, OperationKind::TextTurn));

    assert_eq!(
        estimate,
        UsageEstimate {
            total_tokens: 7,
            ..UsageEstimate::zero()
        }
    );
}

#[tokio::test]
async fn context_entrypoints_read_lutum_default_extensions_in_resolve_usage_estimate() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let ctx = full_context(LutumHooks::new().with_resolve_usage_estimate(
        RecordExtensionEstimate {
            seen: Arc::clone(&seen),
        },
    ))
    .with_extension(UsageEstimate {
        total_tokens: 42,
        ..UsageEstimate::zero()
    });

    let _pending = ctx.text_turn(input()).start().await.unwrap();

    assert_eq!(*seen.lock().unwrap(), vec![42]);
}

#[tokio::test]
async fn request_extensions_override_lutum_default_extensions_in_resolve_usage_estimate() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let ctx = full_context(LutumHooks::new().with_resolve_usage_estimate(
        RecordExtensionEstimate {
            seen: Arc::clone(&seen),
        },
    ))
    .with_extension(UsageEstimate {
        total_tokens: 42,
        ..UsageEstimate::zero()
    });

    let _pending = ctx
        .text_turn(input())
        .ext(UsageEstimate {
            total_tokens: 7,
            ..UsageEstimate::zero()
        })
        .start()
        .await
        .unwrap();

    assert_eq!(*seen.lock().unwrap(), vec![7]);
}

#[test]
fn context_entrypoints_pass_operation_kind_to_resolve_usage_estimate() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let ctx = full_context(
        LutumHooks::new().with_resolve_usage_estimate(RecordOperationKinds {
            seen: Arc::clone(&seen),
        }),
    );

    let _text = block_on(ctx.text_turn(input()).start()).unwrap();
    let _structured = block_on(ctx.structured_turn::<Summary>(input()).start()).unwrap();
    let _completion = block_on(ctx.completion("hello").start()).unwrap();
    let _structured_completion =
        block_on(ctx.structured_completion::<Summary>("hello").start()).unwrap();

    assert_eq!(
        *seen.lock().unwrap(),
        vec![
            OperationKind::TextTurn,
            OperationKind::StructuredTurn,
            OperationKind::Completion,
            OperationKind::StructuredCompletion,
        ]
    );
}

#[test]
fn accumulate_no_hooks_returns_default_only() {
    let hooks = AccumulateHooks::new();
    // No hooks: only default contributes → Vec with one entry, joined.
    let result = block_on(hooks.accumulate_label("x"));
    assert_eq!(result, "default:x");
}

#[test]
fn accumulate_with_hooks_collects_all_independently() {
    // Two hooks plus default: all contribute independently (no `last`).
    let hooks = AccumulateHooks::new()
        .with_accumulate_label(AccumulateHookA)
        .with_accumulate_label(AccumulateHookB);
    let result = block_on(hooks.accumulate_label("x"));
    assert_eq!(result, "default:x, hook-a:x, hook-b:x");
}

#[test]
fn accumulate_chain_stops_early_on_break() {
    // stop hook produces "stop:early" which triggers Break; unreachable hook never runs.
    let hooks = AccumulateHooks::new()
        .with_accumulate_chain_label(AccumulateChainHookStop)
        .with_accumulate_chain_label(AccumulateChainHookUnreachable);
    let result = block_on(hooks.accumulate_chain_label("x"));
    // default → Continue, stop → Break; aggregate collects [default:x, stop:early].
    assert_eq!(result, "default:x, stop:early");
}

#[test]
fn finalize_wraps_fold_result() {
    // One fold hook appends; finalize wraps the final result in brackets.
    let hooks = AccumulateHooks::new().with_finalized_label(FinalizedAppend);
    let result = block_on(hooks.finalized_label("x"));
    // fold: default="default:x", append gets last=Some("default:x") → "default:x+x"
    // finalize: "[default:x+x]"
    assert_eq!(result, "[default:x+x]");
}

#[test]
fn finalize_wraps_no_hooks_fold_result() {
    let hooks = AccumulateHooks::new();
    // No hooks: only default runs, finalize wraps it.
    let result = block_on(hooks.finalized_label("x"));
    assert_eq!(result, "[default:x]");
}

#[test]
fn chain_finalize_captures_early_exit() {
    // stop hook triggers Break; unreachable hook never runs.
    // finalize must still wrap even though dispatch returned early.
    let hooks = AccumulateHooks::new()
        .with_chain_finalized_label(ChainFinalizedStop)
        .with_chain_finalized_label(ChainFinalizedUnreachable);
    let result = block_on(hooks.chain_finalized_label("x"));
    // default → Continue, stop → "stop:chain" → Break (early return)
    // finalize wraps: "[stop:chain]"
    assert_eq!(result, "[stop:chain]");
}

#[test]
fn aggregate_output_override_returns_companion_output_type() {
    let hooks = OutputIntoHooks::new().with_accumulate_label_into(AccumulateIntoHookA);
    let result = block_on(hooks.accumulate_label_into("x"));

    assert_eq!(
        result,
        CollectedLabels(vec!["default:x".to_owned(), "hook-a:x".to_owned()])
    );
}

#[test]
fn fallback_aggregate_output_override_runs_companion_without_hooks() {
    let hooks = OutputIntoHooks::new();
    let result = block_on(hooks.fallback_accumulate_label_into("x"));

    assert_eq!(result, CollectedLabels(vec!["fallback:x".to_owned()]));
}

#[test]
fn fallback_chain_aggregate_excludes_default_when_hooks_are_registered() {
    let hooks = OutputIntoHooks::new()
        .with_fallback_accumulate_chain_label_into(FallbackAccumulateChainIntoHookA)
        .with_fallback_accumulate_chain_label_into(FallbackAccumulateChainIntoHookB);
    let result = block_on(hooks.fallback_accumulate_chain_label_into("x"));

    assert_eq!(
        result,
        CollectedLabels(vec!["hook-a:x".to_owned(), "hook-b:x".to_owned()])
    );
}

#[test]
fn finalize_output_override_returns_companion_output_type() {
    let hooks = OutputIntoHooks::new().with_finalized_label_into(FinalizedIntoAppend);
    let result = block_on(hooks.finalized_label_into("x"));

    assert_eq!(result, WrappedLabel("[default:x+x]".to_owned()));
}

#[test]
fn aggregate_output_override_preserves_chain_early_exit() {
    let hooks = OutputIntoHooks::new()
        .with_accumulate_chain_label_into(AccumulateChainIntoHookStop)
        .with_accumulate_chain_label_into(AccumulateChainIntoHookUnreachable);
    let result = block_on(hooks.accumulate_chain_label_into("x"));

    assert_eq!(
        result,
        CollectedLabels(vec!["default:x".to_owned(), "stop:early".to_owned()])
    );
}
