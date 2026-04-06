use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::executor::block_on;
use lutum::{
    AdapterStructuredCompletionRequest, AdapterStructuredTurn, AdapterTextTurn, AgentError,
    CompletionAdapter, CompletionEventStream, CompletionRequest,
    ErasedStructuredCompletionEventStream, ErasedStructuredTurnEventStream,
    ErasedTextTurnEventStream, HookReentrancyError, HookRegistry, InputMessageRole, Lutum,
    MockLlmAdapter, ModelInput, ModelInputItem, ModelName, OperationKind, RequestExtensions,
    ResolveUsageEstimateHook, ResolveUsageEstimateRegistryExt, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, Stateful, TurnAdapter, Usage, UsageRecoveryAdapter,
    budget::UsageEstimate, hooks::ResolveUsageEstimateLutumExt,
};
use lutum_trace::FieldValue;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[lutum::def_hook(singleton)]
async fn select_label(_ctx: &Lutum, default: String) -> String {
    default
}

#[lutum::hook(SelectLabel)]
async fn prefix_label(_ctx: &Lutum, default: String) -> String {
    format!("hooked:{default}")
}

#[lutum::hook(SelectLabel)]
async fn suffix_label(_ctx: &Lutum, default: String) -> String {
    format!("{default}:suffix")
}

#[lutum::def_hook(always)]
async fn format_label(_ctx: &Lutum, label: &str) -> String {
    format!("default:{label}")
}

#[lutum::hook(FormatLabel)]
async fn append_suffix(_ctx: &Lutum, _label: &str, last: Option<String>) -> String {
    let previous = last.expect("always hooks should receive the default result");
    format!("{previous}:hook")
}

#[lutum::def_hook(always)]
async fn legacy_format_label(_ctx: &Lutum, label: &str, _last: Option<String>) -> String {
    format!("legacy:{label}")
}

#[lutum::def_hook(fallback)]
async fn choose_label(_ctx: &Lutum, label: &str) -> String {
    format!("default:{label}")
}

#[lutum::hook(ChooseLabel)]
async fn pick_registered_label(_ctx: &Lutum, label: &str, last: Option<String>) -> String {
    assert!(last.is_none(), "fallback chains should start from None");
    format!("hook:{label}")
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CounterError {
    Reentered(HookReentrancyError),
}

type CounterResult = Result<usize, CounterError>;

#[lutum::def_hook(singleton)]
async fn next_counter(_ctx: &Lutum, seed: usize) -> CounterResult {
    Ok(seed)
}

struct CountingHook {
    next: usize,
}

#[async_trait]
impl StatefulNextCounterHook for CountingHook {
    fn on_reentrancy(err: HookReentrancyError) -> CounterResult {
        Err(CounterError::Reentered(err))
    }

    async fn call_mut(&mut self, _ctx: &Lutum, seed: usize) -> CounterResult {
        let current = self.next.max(seed);
        self.next = current + 1;
        Ok(current)
    }
}

struct ReentrantCounter;

#[async_trait]
impl StatefulNextCounterHook for ReentrantCounter {
    fn on_reentrancy(err: HookReentrancyError) -> CounterResult {
        Err(CounterError::Reentered(err))
    }

    async fn call_mut(&mut self, ctx: &Lutum, seed: usize) -> CounterResult {
        if seed == 0 {
            Ok(0)
        } else {
            ctx.next_counter(seed - 1).await
        }
    }
}

#[lutum::def_hook(singleton)]
async fn describe_label(_ctx: &Lutum, label: &str) -> String {
    label.to_string()
}

struct NestedLabelHook;

#[async_trait]
impl StatefulDescribeLabelHook for NestedLabelHook {
    async fn call_mut(&mut self, ctx: &Lutum, label: &str) -> String {
        ctx.select_label(label.to_string()).await
    }
}

fn test_context(hooks: HookRegistry) -> Lutum {
    Lutum::with_hooks(
        Arc::new(MockLlmAdapter::new()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        hooks,
    )
}

fn full_context(hooks: HookRegistry) -> Lutum {
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
        _hooks: &HookRegistry,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        Ok(Box::pin(futures::stream::empty()) as ErasedTextTurnEventStream)
    }

    async fn structured_turn(
        &self,
        _input: ModelInput,
        _turn: AdapterStructuredTurn,
        _hooks: &HookRegistry,
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
        _hooks: &HookRegistry,
    ) -> Result<CompletionEventStream, AgentError> {
        Ok(Box::pin(futures::stream::empty()) as CompletionEventStream)
    }

    async fn structured_completion(
        &self,
        _request: AdapterStructuredCompletionRequest,
        _extensions: &RequestExtensions,
        _hooks: &HookRegistry,
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
impl ResolveUsageEstimateHook for FixedEstimate {
    async fn call(
        &self,
        _ctx: &Lutum,
        _extensions: &RequestExtensions,
        _kind: OperationKind,
    ) -> UsageEstimate {
        self.estimate
    }
}

struct RecordOperationKinds {
    seen: Arc<Mutex<Vec<OperationKind>>>,
}

#[async_trait]
impl ResolveUsageEstimateHook for RecordOperationKinds {
    async fn call(
        &self,
        _ctx: &Lutum,
        _extensions: &RequestExtensions,
        kind: OperationKind,
    ) -> UsageEstimate {
        self.seen.lock().unwrap().push(kind);
        UsageEstimate::zero()
    }
}

#[test]
fn singleton_hook_uses_default_when_unregistered() {
    let ctx = test_context(HookRegistry::new());

    let selected = block_on(ctx.select_label("base".into()));

    assert_eq!(selected, "base");
}

#[test]
fn singleton_hook_uses_registered_override() {
    let ctx = test_context(HookRegistry::new().register_select_label(PrefixLabel));

    let selected = block_on(ctx.select_label("base".into()));

    assert_eq!(selected, "hooked:base");
}

#[test]
fn singleton_hook_warns_and_uses_last_registered_override() {
    let collected = block_on(lutum_trace::test::collect(async {
        let ctx = test_context(
            HookRegistry::new()
                .register_select_label(PrefixLabel)
                .register_select_label(SuffixLabel),
        );

        ctx.select_label("base".into()).await
    }));

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
    let ctx = test_context(HookRegistry::new());

    let selected = block_on(ctx.format_label("base"));

    assert_eq!(selected, "default:base");
}

#[test]
fn always_hook_passes_default_result_to_registered_hook() {
    let ctx = test_context(HookRegistry::new().register_format_label(AppendSuffix));

    let selected = block_on(ctx.format_label("base"));

    assert_eq!(selected, "default:base:hook");
}

#[test]
fn always_hook_keeps_supporting_default_definitions_with_last() {
    let ctx = test_context(HookRegistry::new());

    let selected = block_on(ctx.legacy_format_label("base"));

    assert_eq!(selected, "legacy:base");
}

#[test]
fn fallback_hook_uses_default_without_last_when_unregistered() {
    let ctx = test_context(HookRegistry::new());

    let selected = block_on(ctx.choose_label("base"));

    assert_eq!(selected, "default:base");
}

#[test]
fn fallback_hook_starts_registered_chain_without_default_result() {
    let ctx = test_context(HookRegistry::new().register_choose_label(PickRegisteredLabel));

    let selected = block_on(ctx.choose_label("base"));

    assert_eq!(selected, "hook:base");
}

#[test]
fn stateful_hook_mutates_state_without_interior_mutability() {
    let ctx = test_context(
        HookRegistry::new().register_next_counter(Stateful::new(CountingHook { next: 0 })),
    );

    let first = block_on(ctx.next_counter(10));
    let second = block_on(ctx.next_counter(10));

    assert_eq!(first, Ok(10));
    assert_eq!(second, Ok(11));
}

#[test]
fn stateful_hook_reentrancy_can_return_a_typed_error() {
    let ctx =
        test_context(HookRegistry::new().register_next_counter(Stateful::new(ReentrantCounter)));

    let result = block_on(ctx.next_counter(1));

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
    let ctx = test_context(
        HookRegistry::new()
            .register_select_label(PrefixLabel)
            .register_describe_label(Stateful::new(NestedLabelHook)),
    );

    let described = block_on(ctx.describe_label("base"));

    assert_eq!(described, "hooked:base");
}

#[test]
fn resolve_usage_estimate_defaults_to_zero() {
    let ctx = test_context(HookRegistry::new());

    let estimate =
        block_on(ctx.resolve_usage_estimate(&RequestExtensions::new(), OperationKind::TextTurn));

    assert_eq!(estimate, UsageEstimate::zero());
}

#[test]
fn resolve_usage_estimate_reads_request_extensions_by_default() {
    let ctx = test_context(HookRegistry::new());
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
        HookRegistry::new().register_resolve_usage_estimate(FixedEstimate {
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

#[test]
fn context_entrypoints_pass_operation_kind_to_resolve_usage_estimate() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let ctx = full_context(HookRegistry::new().register_resolve_usage_estimate(
        RecordOperationKinds {
            seen: Arc::clone(&seen),
        },
    ));

    let _text = block_on(ctx.text_turn(input()).start()).unwrap();
    let _structured = block_on(ctx.structured_turn::<Summary>(input()).start()).unwrap();
    let _completion = block_on(
        ctx.completion(ModelName::new("gpt-4.1-mini").unwrap(), "hello")
            .start(),
    )
    .unwrap();
    let _structured_completion = block_on(
        ctx.structured_completion::<Summary>(ModelName::new("gpt-4.1-mini").unwrap(), "hello")
            .start(),
    )
    .unwrap();

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
