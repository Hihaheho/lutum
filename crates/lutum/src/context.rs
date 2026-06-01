use std::{convert::Infallible, ops::Deref, sync::Arc, time::Duration};
#[cfg(not(target_family = "wasm"))]
use std::{
    sync::Mutex,
    task::{Context, Poll},
};

use async_stream::try_stream;
use futures::StreamExt;
use thiserror::Error;
use tracing::{Instrument, Span, field};

use lutum_protocol::{
    AgentError, AssistantInputItem, AssistantTurn, AssistantTurnItem, AssistantTurnView,
    CommittedTurn, NoTools, NoToolsContractViolation, RawJson, ToolCallId, ToolName,
    UncommittedAssistantTurn,
    budget::{BudgetLease, BudgetManager, Remaining, Usage, UsageEstimate},
    conversation::{MessageContent, ModelInput, ModelInputItem, ToolMetadata},
    error::RequestFailure,
    extensions::RequestExtensions,
    llm::{
        AdapterStructuredCompletionRequest, AdapterStructuredOutputSpec, AdapterStructuredTurn,
        AdapterTextTurn, AdapterToolChoice, AdapterToolDefinition, AdapterTurnConfig,
        CompletionAdapter, CompletionEvent, CompletionEventStream, CompletionOptions,
        CompletionRequest, ErasedStructuredCompletionEvent, ErasedStructuredCompletionEventStream,
        ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
        ErasedTextTurnEventStream, GenerationParams, MaxOutputTokens, OperationKind, RetryPolicy,
        StructuredCompletionEvent, StructuredCompletionEventStream, StructuredCompletionRequest,
        StructuredTurn as ProtocolStructuredTurn, StructuredTurnEvent, StructuredTurnEventStream,
        StructuredTurnEventStreamWithTools, StructuredTurnEventWithTools,
        TextTurn as ProtocolTextTurn, TextTurnEvent, TextTurnEventStream,
        TextTurnEventStreamWithTools, TextTurnEventWithTools, TokenCount, TokenCounter,
        TurnAdapter, TurnConfig, UsageRecoveryAdapter,
    },
    reducer::{
        CompletionReducer, CompletionReductionError, CompletionTurnResult, CompletionTurnState,
        StagedStructuredTurnResult, StagedStructuredTurnResultWithTools,
        StagedTextTurnOutcomeWithTools, StagedTextTurnResult, StagedTextTurnResultWithTools,
        StructuredCompletionReducer, StructuredCompletionReductionError,
        StructuredCompletionResult, StructuredCompletionState, StructuredTurnReducer,
        StructuredTurnReducerWithTools, StructuredTurnReductionError, StructuredTurnState,
        StructuredTurnStateWithTools, TextTurnReducer, TextTurnReducerWithTools,
        TextTurnReductionError, TextTurnState, TextTurnStateWithTools,
    },
    structured::StructuredOutput,
    telemetry::{
        CollectErrorKind, RawTelemetryConfig, emit_collect_error, raw_collect_errors_enabled,
    },
    toolset::{
        RecoverableToolCallIssue, RecoveredTextToolCalls, TextToolCallFallbackContext,
        TextToolCallFallbackParser, ToolAvailability, ToolCallFallbackError, ToolCallWrapper,
        ToolConstraints, ToolRequirement, ToolSelector, Toolset,
    },
};

use crate::hooks::{
    LutumHooksSet, LutumStreamEvent, MaybeSend, MaybeSync, ModelInputHookContext,
    StreamEventHookContext,
};

pub type LutumError = AgentError;

#[derive(Debug, Error)]
#[error("completion adapter is not configured; use Lutum::from_parts(...) to provide one")]
struct MissingCompletionAdapter;

#[derive(Clone, Default)]
struct UnsupportedCompletionAdapter;

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
impl CompletionAdapter for UnsupportedCompletionAdapter {
    async fn completion(
        &self,
        _request: CompletionRequest,
        _extensions: &RequestExtensions,
    ) -> Result<CompletionEventStream, AgentError> {
        Err(AgentError::other(MissingCompletionAdapter))
    }

    async fn structured_completion(
        &self,
        _request: AdapterStructuredCompletionRequest,
        _extensions: &RequestExtensions,
    ) -> Result<ErasedStructuredCompletionEventStream, AgentError> {
        Err(AgentError::other(MissingCompletionAdapter))
    }
}

#[derive(Clone)]
pub struct Lutum {
    budget: Arc<dyn BudgetManager>,
    turns: Arc<dyn TurnAdapter>,
    completion: Arc<dyn CompletionAdapter>,
    recovery: Option<Arc<dyn UsageRecoveryAdapter>>,
    token_counter: Option<Arc<dyn TokenCounter>>,
    hooks: Arc<LutumHooksSet<'static>>,
    default_extensions: Arc<RequestExtensions>,
}

impl Lutum {
    pub fn new<T>(adapter: Arc<T>, budget: impl BudgetManager + 'static) -> Self
    where
        T: TurnAdapter + 'static,
    {
        Self::with_hooks(adapter, budget, LutumHooksSet::new())
    }

    pub fn with_hooks<T>(
        adapter: Arc<T>,
        budget: impl BudgetManager + 'static,
        hooks: LutumHooksSet<'static>,
    ) -> Self
    where
        T: TurnAdapter + 'static,
    {
        Self {
            budget: Arc::new(budget),
            turns: adapter,
            completion: Arc::new(UnsupportedCompletionAdapter),
            recovery: None,
            token_counter: None,
            hooks: Arc::new(hooks),
            default_extensions: Arc::new(RequestExtensions::new()),
        }
    }

    pub fn from_parts(
        turns: Arc<dyn TurnAdapter>,
        completion: Arc<dyn CompletionAdapter>,
        budget: impl BudgetManager + 'static,
    ) -> Self {
        Self::from_parts_with_hooks(turns, completion, budget, LutumHooksSet::new())
    }

    pub fn from_parts_with_hooks(
        turns: Arc<dyn TurnAdapter>,
        completion: Arc<dyn CompletionAdapter>,
        budget: impl BudgetManager + 'static,
        hooks: LutumHooksSet<'static>,
    ) -> Self {
        Self {
            budget: Arc::new(budget),
            turns,
            completion,
            recovery: None,
            token_counter: None,
            hooks: Arc::new(hooks),
            default_extensions: Arc::new(RequestExtensions::new()),
        }
    }

    pub fn with_recovery(mut self, recovery: Arc<dyn UsageRecoveryAdapter>) -> Self {
        self.recovery = Some(recovery);
        self
    }

    pub fn with_token_counter<T>(mut self, token_counter: Arc<T>) -> Self
    where
        T: TokenCounter + 'static,
    {
        self.token_counter = Some(token_counter);
        self
    }

    pub fn budget(&self) -> &dyn BudgetManager {
        self.budget.as_ref()
    }

    pub fn with_extension<T>(self, extension: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        let mut extensions = RequestExtensions::new();
        extensions.insert(extension);
        self.with_extensions(extensions)
    }

    pub fn with_extensions(mut self, mut extensions: RequestExtensions) -> Self {
        extensions.push_fallback(Arc::clone(&self.default_extensions));
        self.default_extensions = Arc::new(extensions);
        self
    }

    pub fn with_retry_policy(self, retry_policy: RetryPolicy) -> Self {
        self.with_extension(retry_policy)
    }

    pub fn default_extensions(&self) -> &RequestExtensions {
        self.default_extensions.as_ref()
    }

    pub fn extend_hooks(&mut self, hooks: LutumHooksSet<'static>) -> &mut Self {
        Arc::make_mut(&mut self.hooks).extend(hooks);
        self
    }

    pub fn with_extended_hooks(mut self, hooks: LutumHooksSet<'static>) -> Self {
        self.extend_hooks(hooks);
        self
    }

    pub fn text_turn(&self, input: ModelInput) -> crate::builders::TextTurn<'_> {
        crate::builders::TextTurn::from_lutum(self, input)
    }

    pub fn structured_turn<O>(&self, input: ModelInput) -> crate::builders::StructuredTurn<'_, O>
    where
        O: StructuredOutput,
    {
        crate::builders::StructuredTurn::from_lutum(self, input)
    }

    pub fn completion(&self, prompt: impl Into<String>) -> crate::builders::Completion<'_> {
        crate::builders::Completion::new(self, prompt)
    }

    pub fn structured_completion<O>(
        &self,
        prompt: impl Into<String>,
    ) -> crate::builders::StructuredCompletion<'_, O>
    where
        O: StructuredOutput,
    {
        crate::builders::StructuredCompletion::new(self, prompt)
    }

    pub async fn resolve_usage_estimate(
        &self,
        extensions: &RequestExtensions,
        kind: OperationKind,
    ) -> UsageEstimate {
        self.hooks.resolve_usage_estimate(extensions, kind).await
    }

    async fn estimate_text_turn_usage(
        &self,
        extensions: &RequestExtensions,
        input: &ModelInput,
        turn: &AdapterTextTurn,
        max_output_tokens: Option<u32>,
    ) -> Result<UsageEstimate, AgentError> {
        let Some(counter) = self.token_counter.as_ref() else {
            return Ok(self
                .resolve_usage_estimate(extensions, OperationKind::TextTurn)
                .await);
        };
        match counter.count_text_turn(input, turn).await? {
            Some(count) => {
                let fallback = self
                    .resolve_usage_estimate(extensions, OperationKind::TextTurn)
                    .await;
                Ok(estimate_with_token_count(
                    fallback,
                    count.input_tokens,
                    max_output_tokens,
                ))
            }
            None => Ok(self
                .resolve_usage_estimate(extensions, OperationKind::TextTurn)
                .await),
        }
    }

    async fn estimate_structured_turn_usage(
        &self,
        extensions: &RequestExtensions,
        input: &ModelInput,
        turn: &AdapterStructuredTurn,
        max_output_tokens: Option<u32>,
    ) -> Result<UsageEstimate, AgentError> {
        let Some(counter) = self.token_counter.as_ref() else {
            return Ok(self
                .resolve_usage_estimate(extensions, OperationKind::StructuredTurn)
                .await);
        };
        match counter.count_structured_turn(input, turn).await? {
            Some(count) => {
                let fallback = self
                    .resolve_usage_estimate(extensions, OperationKind::StructuredTurn)
                    .await;
                Ok(estimate_with_token_count(
                    fallback,
                    count.input_tokens,
                    max_output_tokens,
                ))
            }
            None => Ok(self
                .resolve_usage_estimate(extensions, OperationKind::StructuredTurn)
                .await),
        }
    }

    async fn estimate_completion_usage(
        &self,
        extensions: &RequestExtensions,
        request: &CompletionRequest,
    ) -> Result<UsageEstimate, AgentError> {
        let Some(counter) = self.token_counter.as_ref() else {
            return Ok(self
                .resolve_usage_estimate(extensions, OperationKind::Completion)
                .await);
        };
        match counter.count_completion(request, extensions).await? {
            Some(count) => {
                let fallback = self
                    .resolve_usage_estimate(extensions, OperationKind::Completion)
                    .await;
                Ok(estimate_with_token_count(
                    fallback,
                    count.input_tokens,
                    request.options.max_output_tokens,
                ))
            }
            None => Ok(self
                .resolve_usage_estimate(extensions, OperationKind::Completion)
                .await),
        }
    }

    async fn estimate_structured_completion_usage(
        &self,
        extensions: &RequestExtensions,
        request: &AdapterStructuredCompletionRequest,
        max_output_tokens: Option<u32>,
    ) -> Result<UsageEstimate, AgentError> {
        let Some(counter) = self.token_counter.as_ref() else {
            return Ok(self
                .resolve_usage_estimate(extensions, OperationKind::StructuredCompletion)
                .await);
        };
        match counter
            .count_structured_completion(request, extensions)
            .await?
        {
            Some(count) => {
                let fallback = self
                    .resolve_usage_estimate(extensions, OperationKind::StructuredCompletion)
                    .await;
                Ok(estimate_with_token_count(
                    fallback,
                    count.input_tokens,
                    max_output_tokens,
                ))
            }
            None => Ok(self
                .resolve_usage_estimate(extensions, OperationKind::StructuredCompletion)
                .await),
        }
    }

    fn apply_default_extensions(&self, mut extensions: RequestExtensions) -> RequestExtensions {
        extensions.push_fallback(Arc::clone(&self.default_extensions));
        extensions
    }

    pub(crate) fn apply_max_output_tokens_extension(
        extensions: &RequestExtensions,
        generation: &mut GenerationParams,
    ) {
        if generation.max_output_tokens.is_none() {
            generation.max_output_tokens = extensions
                .get::<MaxOutputTokens>()
                .map(|max_output_tokens| max_output_tokens.get());
        }
    }

    pub(crate) fn apply_completion_max_output_tokens_extension(
        extensions: &RequestExtensions,
        options: &mut CompletionOptions,
    ) {
        if options.max_output_tokens.is_none() {
            options.max_output_tokens = extensions
                .get::<MaxOutputTokens>()
                .map(|max_output_tokens| max_output_tokens.get());
        }
    }

    pub(crate) fn raw_collect_errors_enabled(&self, extensions: &RequestExtensions) -> bool {
        if extensions.contains::<RawTelemetryConfig>() {
            raw_collect_errors_enabled(extensions)
        } else {
            raw_collect_errors_enabled(&self.default_extensions)
        }
    }

    async fn emit_model_input_hook(
        &self,
        span: &Span,
        extensions: &RequestExtensions,
        kind: OperationKind,
        input: &ModelInput,
    ) {
        let cx = ModelInputHookContext::new(extensions, kind, input);
        self.hooks
            .on_model_input(&cx)
            .instrument(span.clone())
            .await;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandlerDirective {
    Continue,
    Stop,
}

pub struct HandlerContext<'a, S> {
    extensions: &'a RequestExtensions,
    state: &'a S,
    remaining_budget: Remaining,
}

impl<'a, S> HandlerContext<'a, S> {
    pub fn extensions(&self) -> &RequestExtensions {
        self.extensions
    }

    pub fn state(&self) -> &S {
        self.state
    }

    pub fn remaining_budget(&self) -> Remaining {
        self.remaining_budget
    }
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
pub trait EventHandler<E, S>: MaybeSend {
    type Error;

    async fn on_event(
        &mut self,
        event: &E,
        cx: &HandlerContext<S>,
    ) -> Result<HandlerDirective, Self::Error>;
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
impl<E, S, F, Err> EventHandler<E, S> for F
where
    F: MaybeSend + for<'a> FnMut(&'a E, &'a HandlerContext<'a, S>) -> Result<HandlerDirective, Err>,
    E: MaybeSync,
    S: MaybeSync,
{
    type Error = Err;

    async fn on_event(
        &mut self,
        event: &E,
        cx: &HandlerContext<S>,
    ) -> Result<HandlerDirective, Self::Error> {
        (self)(event, cx)
    }
}

/// Recoverable collection errors exposed to [`TextToolEventHandler::on_error`].
///
/// Handler errors and budget-finalization errors are intentionally not routed
/// through this type; those errors are returned directly to avoid recursive
/// recovery paths.
#[derive(Debug)]
pub enum TextToolCollectError<'a> {
    /// The adapter or execution layer failed before a reducible event was available.
    Execution(&'a AgentError),
    /// The text+tools reducer rejected the current partial state.
    Reduction(&'a TextTurnReductionError),
    /// The stream ended without a `Completed` event.
    UnexpectedEof,
}

/// A user-synthesized text+tools turn returned from controlled collection.
///
/// Use [`SyntheticTextToolTurn::finished`] for a completed assistant turn and
/// [`SyntheticTextToolTurn::needs_tools`] when user code has recovered tool
/// calls from assistant text. Synthetic returns are still validated against the
/// active tool availability and requirement constraints.
#[derive(Clone, Debug)]
pub struct SyntheticTextToolTurn<T: Toolset> {
    /// Assistant turn items to replay into the transcript.
    pub assistant_turn: AssistantTurn,
    /// Decoded tool calls, if the synthetic outcome should enter a tool round.
    pub tool_calls: Vec<T::ToolCall>,
    /// Recoverable tool-call issues to attach to the staged result.
    pub recoverable_tool_call_issues: Vec<RecoverableToolCallIssue>,
    /// Optional suggestion that the caller should continue the model turn.
    pub continue_suggestion: Option<lutum_protocol::ContinueSuggestionReason>,
    /// Optional finish reason. Defaults to `ToolCall` when tool calls exist and
    /// `Stop` otherwise.
    pub finish_reason: Option<lutum_protocol::FinishReason>,
    /// Optional usage override. When absent, controlled collection uses provider
    /// usage already seen in the reducer, usage recovery, or the normal estimate.
    pub usage: Option<Usage>,
}

impl<T> SyntheticTextToolTurn<T>
where
    T: Toolset,
{
    /// Build a synthetic finished assistant turn.
    pub fn finished(assistant_turn: AssistantTurn) -> Self {
        Self {
            assistant_turn,
            tool_calls: Vec::new(),
            recoverable_tool_call_issues: Vec::new(),
            continue_suggestion: None,
            finish_reason: None,
            usage: None,
        }
    }

    /// Build a synthetic tool round from recovered text tool calls.
    pub fn needs_tools(recovered: RecoveredTextToolCalls<T>) -> Self {
        Self {
            assistant_turn: recovered.assistant_turn,
            tool_calls: recovered.tool_calls,
            recoverable_tool_call_issues: Vec::new(),
            continue_suggestion: None,
            finish_reason: None,
            usage: None,
        }
    }
}

/// Directive returned by [`TextToolEventHandler::on_event`].
#[derive(Debug)]
pub enum TextToolHandlerDirective<T: Toolset> {
    /// Continue normal stream collection.
    Continue,
    /// Stop collection and return `CollectError::Stopped`.
    Stop,
    /// Finish collection immediately with a synthetic text+tools outcome.
    Return(SyntheticTextToolTurn<T>),
}

/// Directive returned by [`TextToolEventHandler::on_error`].
#[derive(Debug)]
pub enum TextToolErrorDirective<T: Toolset> {
    /// Return the original collection error.
    Propagate,
    /// Treat the error as handled and return a synthetic text+tools outcome.
    Return(SyntheticTextToolTurn<T>),
}

/// Context passed to [`TextToolEventHandler`] callbacks.
///
/// The reducer is updated before handlers are called, so `state()` reflects the
/// event being observed. The validation helpers use the resolved tool
/// availability and requirement constraints for the current request.
pub struct TextToolHandlerContext<'a, T: Toolset> {
    extensions: &'a RequestExtensions,
    state: &'a TextTurnStateWithTools<T>,
    remaining_budget: Remaining,
    constraints: &'a ToolConstraints<T>,
    tool_definitions: &'a [AdapterToolDefinition],
}

impl<'a, T> TextToolHandlerContext<'a, T>
where
    T: Toolset,
{
    /// Request extensions active for this turn.
    pub fn extensions(&self) -> &RequestExtensions {
        self.extensions
    }

    /// Current reduced text+tools state.
    pub fn state(&self) -> &TextTurnStateWithTools<T> {
        self.state
    }

    /// Remaining request budget after current reservations.
    pub fn remaining_budget(&self) -> Remaining {
        self.remaining_budget
    }

    /// Resolved tool availability and requirement constraints.
    pub fn constraints(&self) -> &ToolConstraints<T> {
        self.constraints
    }

    /// Adapter-level tool definitions sent with this request.
    pub fn tool_definitions(&self) -> &[AdapterToolDefinition] {
        self.tool_definitions
    }

    /// Convert assistant items into decoded tool calls and validate them against
    /// the current tool constraints.
    pub fn recover_tool_calls_from_items(
        &self,
        items: Vec<AssistantTurnItem>,
    ) -> Result<RecoveredTextToolCalls<T>, ToolCallFallbackError> {
        let recovered = RecoveredTextToolCalls::<T>::from_items(items)?;
        self.validate_recovered_tool_calls(recovered)
    }

    /// Validate recovered calls against the current availability and required
    /// tool policy.
    pub fn validate_recovered_tool_calls(
        &self,
        recovered: RecoveredTextToolCalls<T>,
    ) -> Result<RecoveredTextToolCalls<T>, ToolCallFallbackError> {
        validate_recovered_text_tool_calls(
            recovered,
            &self.constraints.requirement,
            self.tool_definitions,
        )
    }

    /// Parse one JSON-derived tool call using the active availability and
    /// required-tool policy.
    pub fn parse_tool_call(
        &self,
        id: impl Into<ToolCallId>,
        name: impl Into<ToolName>,
        arguments: RawJson,
    ) -> Result<T::ToolCall, ToolCallFallbackError> {
        let metadata = ToolMetadata::new(id, name, arguments);
        let recovered = RecoveredTextToolCalls::from_parts(
            vec![AssistantTurnItem::ToolCall {
                id: metadata.id.clone(),
                name: metadata.name.clone(),
                arguments: metadata.arguments.clone(),
            }],
            vec![parse_validated_tool_call::<T>(
                metadata,
                &self.constraints.requirement,
                self.tool_definitions,
            )?],
        )?;
        let mut recovered = self.validate_recovered_tool_calls(recovered)?;
        recovered
            .tool_calls
            .pop()
            .ok_or(ToolCallFallbackError::NoToolCall)
    }
}

/// Advanced event handler for text turns with tools.
///
/// This is separate from [`EventHandler`] so existing `collect_with` behavior
/// remains unchanged. Use it with `collect_controlled_with` when the handler may
/// synthesize a turn/tool round or recover from collection errors.
#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
pub trait TextToolEventHandler<T: Toolset>: MaybeSend {
    /// User error returned from handler callbacks.
    type Error;

    /// Observe a reduced stream event and decide whether collection should
    /// continue, stop, or return a synthetic outcome.
    async fn on_event(
        &mut self,
        event: &TextTurnEventWithTools<T>,
        cx: &TextToolHandlerContext<T>,
    ) -> Result<TextToolHandlerDirective<T>, Self::Error>;

    /// Optionally recover from controlled collection errors.
    ///
    /// The default propagates the original error. This method is called for
    /// adapter execution failures, reducer errors such as
    /// `OutputLimitExceeded`, and unexpected EOF.
    async fn on_error(
        &mut self,
        _error: TextToolCollectError<'_>,
        _cx: &TextToolHandlerContext<T>,
    ) -> Result<TextToolErrorDirective<T>, Self::Error> {
        Ok(TextToolErrorDirective::Propagate)
    }
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
impl<T, F, Err> TextToolEventHandler<T> for F
where
    T: Toolset,
    F: MaybeSend
        + for<'a> FnMut(
            &'a TextTurnEventWithTools<T>,
            &'a TextToolHandlerContext<'a, T>,
        ) -> Result<TextToolHandlerDirective<T>, Err>,
{
    type Error = Err;

    async fn on_event(
        &mut self,
        event: &TextTurnEventWithTools<T>,
        cx: &TextToolHandlerContext<T>,
    ) -> Result<TextToolHandlerDirective<T>, Self::Error> {
        (self)(event, cx)
    }
}

#[derive(Debug, Error)]
pub enum CollectError<HandlerError, ReductionError, Partial> {
    #[error("execution error: {source}")]
    Execution {
        #[source]
        source: AgentError,
        partial: Partial,
    },
    #[error("handler error: {source}")]
    Handler {
        #[source]
        source: HandlerError,
        partial: Partial,
    },
    #[error("reduction error: {source}")]
    Reduction {
        #[source]
        source: ReductionError,
        partial: Partial,
    },
    #[error("collection stopped by handler")]
    Stopped { partial: Partial },
    #[error("stream ended before completion")]
    UnexpectedEof { partial: Partial },
}

struct OwnedLease {
    budget: Arc<dyn BudgetManager>,
    lease: Option<BudgetLease>,
}

#[cfg(not(target_family = "wasm"))]
struct SyncPinnedStream<Item> {
    inner: Mutex<core::pin::Pin<Box<dyn futures::Stream<Item = Item> + Send + 'static>>>,
}

#[cfg(not(target_family = "wasm"))]
impl<Item> futures::Stream for SyncPinnedStream<Item> {
    type Item = Item;

    fn poll_next(
        self: core::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let mut inner = self.inner.lock().expect("retry stream mutex poisoned");
        inner.as_mut().poll_next(cx)
    }
}

#[cfg(not(target_family = "wasm"))]
fn boxed_sync_stream<Item: 'static>(
    stream: impl futures::Stream<Item = Item> + Send + 'static,
) -> core::pin::Pin<Box<dyn futures::Stream<Item = Item> + Send + Sync + 'static>> {
    Box::pin(SyncPinnedStream {
        inner: Mutex::new(Box::pin(stream)),
    })
}

#[cfg(target_family = "wasm")]
fn boxed_sync_stream<Item: 'static>(
    stream: impl futures::Stream<Item = Item> + 'static,
) -> core::pin::Pin<Box<dyn futures::Stream<Item = Item> + 'static>> {
    Box::pin(stream)
}

impl Drop for OwnedLease {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take()
            && let Err(err) = self.budget.record_used(lease, Usage::zero())
        {
            tracing::error!(
                error = %err,
                "failed to finalize budget lease on drop; shared pool reservation may leak until the process restarts"
            );
        }
    }
}

pub struct PendingTextTurn {
    extensions: Arc<RequestExtensions>,
    owned_lease: OwnedLease,
    recovery: Option<Arc<dyn UsageRecoveryAdapter>>,
    turns: Arc<dyn TurnAdapter>,
    hooks: Arc<LutumHooksSet<'static>>,
    input: ModelInput,
    turn: AdapterTextTurn,
    estimate: UsageEstimate,
    retry_policy: RetryPolicy,
    span: Span,
    reducer: TextTurnReducer,
}

pub struct PendingTextTurnWithTools<T>
where
    T: Toolset,
{
    extensions: Arc<RequestExtensions>,
    owned_lease: OwnedLease,
    recovery: Option<Arc<dyn UsageRecoveryAdapter>>,
    turns: Arc<dyn TurnAdapter>,
    hooks: Arc<LutumHooksSet<'static>>,
    input: ModelInput,
    turn: AdapterTextTurn,
    tool_constraints: ToolConstraints<T>,
    tool_definitions: Vec<AdapterToolDefinition>,
    availability: ToolAvailability<T::Selector>,
    dynamic_names: Vec<String>,
    fallback_parser: Option<Arc<dyn TextToolCallFallbackParser<T>>>,
    estimate: UsageEstimate,
    retry_policy: RetryPolicy,
    span: Span,
    reducer: TextTurnReducerWithTools<T>,
}

pub struct PendingStructuredTurn<O>
where
    O: StructuredOutput,
{
    extensions: Arc<RequestExtensions>,
    owned_lease: OwnedLease,
    recovery: Option<Arc<dyn UsageRecoveryAdapter>>,
    turns: Arc<dyn TurnAdapter>,
    hooks: Arc<LutumHooksSet<'static>>,
    input: ModelInput,
    turn: AdapterStructuredTurn,
    estimate: UsageEstimate,
    retry_policy: RetryPolicy,
    span: Span,
    reducer: StructuredTurnReducer<O>,
}

pub struct PendingStructuredTurnWithTools<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    extensions: Arc<RequestExtensions>,
    owned_lease: OwnedLease,
    recovery: Option<Arc<dyn UsageRecoveryAdapter>>,
    turns: Arc<dyn TurnAdapter>,
    hooks: Arc<LutumHooksSet<'static>>,
    input: ModelInput,
    turn: AdapterStructuredTurn,
    availability: ToolAvailability<T::Selector>,
    dynamic_names: Vec<String>,
    estimate: UsageEstimate,
    retry_policy: RetryPolicy,
    span: Span,
    reducer: StructuredTurnReducerWithTools<T, O>,
}

#[derive(Clone, Debug)]
pub struct StructuredTurnPartial<O>
where
    O: StructuredOutput,
{
    pub state: StructuredTurnState<O>,
    pub committed_turn: Option<CommittedTurn>,
}

impl<O> StructuredTurnPartial<O>
where
    O: StructuredOutput,
{
    pub(crate) fn from_state(state: StructuredTurnState<O>) -> Self {
        let committed_turn = state.committed_turn.clone();
        Self {
            state,
            committed_turn,
        }
    }

    pub(crate) fn with_committed_turn(mut self, committed_turn: CommittedTurn) -> Self {
        self.committed_turn = Some(committed_turn);
        self
    }
}

impl<O> Deref for StructuredTurnPartial<O>
where
    O: StructuredOutput,
{
    type Target = StructuredTurnState<O>;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

#[derive(Clone, Debug)]
pub struct StructuredTurnPartialWithTools<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    pub state: StructuredTurnStateWithTools<T, O>,
    pub committed_turn: Option<CommittedTurn>,
}

impl<T, O> StructuredTurnPartialWithTools<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    pub(crate) fn from_state(state: StructuredTurnStateWithTools<T, O>) -> Self {
        let committed_turn = state.committed_turn.clone();
        Self {
            state,
            committed_turn,
        }
    }

    pub(crate) fn with_committed_turn(mut self, committed_turn: CommittedTurn) -> Self {
        self.committed_turn = Some(committed_turn);
        self
    }
}

impl<T, O> Deref for StructuredTurnPartialWithTools<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    type Target = StructuredTurnStateWithTools<T, O>;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

pub struct PendingCompletion {
    extensions: Arc<RequestExtensions>,
    owned_lease: OwnedLease,
    recovery: Option<Arc<dyn UsageRecoveryAdapter>>,
    completion: Arc<dyn CompletionAdapter>,
    hooks: Arc<LutumHooksSet<'static>>,
    request: CompletionRequest,
    estimate: UsageEstimate,
    retry_policy: RetryPolicy,
    span: Span,
    reducer: CompletionReducer,
}

pub struct PendingStructuredCompletion<O>
where
    O: StructuredOutput,
{
    extensions: Arc<RequestExtensions>,
    owned_lease: OwnedLease,
    recovery: Option<Arc<dyn UsageRecoveryAdapter>>,
    completion: Arc<dyn CompletionAdapter>,
    hooks: Arc<LutumHooksSet<'static>>,
    request: AdapterStructuredCompletionRequest,
    estimate: UsageEstimate,
    retry_policy: RetryPolicy,
    span: Span,
    reducer: StructuredCompletionReducer<O>,
}

impl Lutum {
    pub(crate) async fn count_text_turn_tokens<T>(
        &self,
        extensions: RequestExtensions,
        input: ModelInput,
        turn: &ProtocolTextTurn<T>,
        generation: GenerationParams,
    ) -> Result<Option<TokenCount>, LutumError>
    where
        T: Toolset,
    {
        input.validate()?;
        let extensions = Arc::new(self.apply_default_extensions(extensions));
        let mut generation = generation;
        Self::apply_max_output_tokens_extension(extensions.as_ref(), &mut generation);
        let turn = erase_text_turn_ref(turn, generation, Arc::clone(&extensions))?;
        let Some(counter) = self.token_counter.as_ref() else {
            return Ok(None);
        };
        counter.count_text_turn(&input, &turn).await
    }

    pub(crate) async fn count_structured_turn_tokens<T, O>(
        &self,
        extensions: RequestExtensions,
        input: ModelInput,
        turn: &ProtocolStructuredTurn<T, O>,
        generation: GenerationParams,
    ) -> Result<Option<TokenCount>, LutumError>
    where
        T: Toolset,
        O: StructuredOutput,
    {
        input.validate()?;
        let extensions = Arc::new(self.apply_default_extensions(extensions));
        let mut generation = generation;
        Self::apply_max_output_tokens_extension(extensions.as_ref(), &mut generation);
        let turn = erase_structured_turn_ref(turn, generation, Arc::clone(&extensions))?;
        let Some(counter) = self.token_counter.as_ref() else {
            return Ok(None);
        };
        counter.count_structured_turn(&input, &turn).await
    }

    pub(crate) async fn count_completion_tokens(
        &self,
        extensions: RequestExtensions,
        mut request: CompletionRequest,
    ) -> Result<Option<TokenCount>, LutumError> {
        let extensions = self.apply_default_extensions(extensions);
        Self::apply_completion_max_output_tokens_extension(&extensions, &mut request.options);
        let Some(counter) = self.token_counter.as_ref() else {
            return Ok(None);
        };
        counter.count_completion(&request, &extensions).await
    }

    pub(crate) async fn count_structured_completion_tokens<O>(
        &self,
        extensions: RequestExtensions,
        request: &StructuredCompletionRequest<O>,
    ) -> Result<Option<TokenCount>, LutumError>
    where
        O: StructuredOutput,
    {
        let extensions = self.apply_default_extensions(extensions);
        let mut request = erase_structured_completion_request_ref(request)?;
        Self::apply_max_output_tokens_extension(&extensions, &mut request.generation);
        let Some(counter) = self.token_counter.as_ref() else {
            return Ok(None);
        };
        counter
            .count_structured_completion(&request, &extensions)
            .await
    }

    pub(crate) async fn run_text_turn(
        &self,
        extensions: RequestExtensions,
        input: ModelInput,
        turn: ProtocolTextTurn<NoTools>,
    ) -> Result<PendingTextTurn, LutumError> {
        input.validate()?;
        let extensions = self.apply_default_extensions(extensions);
        let mut turn = turn;
        Self::apply_max_output_tokens_extension(&extensions, &mut turn.config.generation);
        let request_budget = turn.config.budget;
        let max_output_tokens = turn.config.generation.max_output_tokens;
        let extensions = Arc::new(extensions);
        let turn = erase_text_turn(turn, Arc::clone(&extensions))?;
        let estimate = self
            .estimate_text_turn_usage(extensions.as_ref(), &input, &turn, max_output_tokens)
            .await?;
        let lease = self
            .budget
            .reserve(extensions.as_ref(), &estimate, request_budget)?;
        let retry_policy = extensions.get::<RetryPolicy>().cloned().unwrap_or_default();
        let span = turn_span("text_turn", estimate);
        self.emit_model_input_hook(&span, extensions.as_ref(), OperationKind::TextTurn, &input)
            .await;
        log_input_transcript(&span, &input);
        Ok(PendingTextTurn {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: self.recovery.clone(),
            turns: Arc::clone(&self.turns),
            hooks: Arc::clone(&self.hooks),
            input,
            turn,
            estimate,
            retry_policy,
            span,
            reducer: TextTurnReducer::new(),
        })
    }

    pub(crate) async fn run_text_turn_with_tools<T>(
        &self,
        extensions: RequestExtensions,
        input: ModelInput,
        turn: ProtocolTextTurn<T>,
        fallback_parser: Option<Arc<dyn TextToolCallFallbackParser<T>>>,
    ) -> Result<PendingTextTurnWithTools<T>, LutumError>
    where
        T: Toolset,
    {
        input.validate()?;
        let extensions = self.apply_default_extensions(extensions);
        let mut turn = turn;
        Self::apply_max_output_tokens_extension(&extensions, &mut turn.config.generation);
        let request_budget = turn.config.budget;
        let max_output_tokens = turn.config.generation.max_output_tokens;
        // Extract availability before erase_text_turn consumes the turn config.
        let availability = turn.config.tools.available.clone();
        let tool_constraints = ToolConstraints {
            available: turn.config.tools.available.clone(),
            requirement: turn.config.tools.requirement.clone(),
            description_overrides: turn.config.tools.description_overrides.clone(),
            dynamic_tools: turn.config.tools.dynamic_tools.clone(),
        };
        let dynamic_names = turn
            .config
            .tools
            .dynamic_tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<Vec<_>>();
        let extensions = Arc::new(extensions);
        let turn = erase_text_turn(turn, Arc::clone(&extensions))?;
        let tool_definitions = turn.config.tools.clone();
        let estimate = self
            .estimate_text_turn_usage(extensions.as_ref(), &input, &turn, max_output_tokens)
            .await?;
        let lease = self
            .budget
            .reserve(extensions.as_ref(), &estimate, request_budget)?;
        let retry_policy = extensions.get::<RetryPolicy>().cloned().unwrap_or_default();
        let span = turn_span("text_turn", estimate);
        self.emit_model_input_hook(&span, extensions.as_ref(), OperationKind::TextTurn, &input)
            .await;
        log_input_transcript(&span, &input);
        Ok(PendingTextTurnWithTools {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: self.recovery.clone(),
            turns: Arc::clone(&self.turns),
            hooks: Arc::clone(&self.hooks),
            input,
            turn,
            tool_constraints,
            tool_definitions,
            availability,
            dynamic_names,
            fallback_parser,
            estimate,
            retry_policy,
            span,
            reducer: TextTurnReducerWithTools::new(),
        })
    }

    pub(crate) async fn run_structured_turn<O>(
        &self,
        extensions: RequestExtensions,
        input: ModelInput,
        turn: ProtocolStructuredTurn<NoTools, O>,
    ) -> Result<PendingStructuredTurn<O>, LutumError>
    where
        O: StructuredOutput,
    {
        input.validate()?;
        let extensions = self.apply_default_extensions(extensions);
        let mut turn = turn;
        Self::apply_max_output_tokens_extension(&extensions, &mut turn.config.generation);
        let request_budget = turn.config.budget;
        let max_output_tokens = turn.config.generation.max_output_tokens;
        let extensions = Arc::new(extensions);
        let turn = erase_structured_turn(turn, Arc::clone(&extensions))?;
        let estimate = self
            .estimate_structured_turn_usage(extensions.as_ref(), &input, &turn, max_output_tokens)
            .await?;
        let lease = self
            .budget
            .reserve(extensions.as_ref(), &estimate, request_budget)?;
        let retry_policy = extensions.get::<RetryPolicy>().cloned().unwrap_or_default();
        let span = turn_span("structured_turn", estimate);
        self.emit_model_input_hook(
            &span,
            extensions.as_ref(),
            OperationKind::StructuredTurn,
            &input,
        )
        .await;
        log_input_transcript(&span, &input);
        Ok(PendingStructuredTurn {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: self.recovery.clone(),
            turns: Arc::clone(&self.turns),
            hooks: Arc::clone(&self.hooks),
            input,
            turn,
            estimate,
            retry_policy,
            span,
            reducer: StructuredTurnReducer::new(),
        })
    }

    pub(crate) async fn run_structured_turn_with_tools<T, O>(
        &self,
        extensions: RequestExtensions,
        input: ModelInput,
        turn: ProtocolStructuredTurn<T, O>,
    ) -> Result<PendingStructuredTurnWithTools<T, O>, LutumError>
    where
        T: Toolset,
        O: StructuredOutput,
    {
        input.validate()?;
        let extensions = self.apply_default_extensions(extensions);
        let mut turn = turn;
        Self::apply_max_output_tokens_extension(&extensions, &mut turn.config.generation);
        let request_budget = turn.config.budget;
        let max_output_tokens = turn.config.generation.max_output_tokens;
        // Extract availability before erase_structured_turn consumes the turn config.
        let availability = turn.config.tools.available.clone();
        let dynamic_names = turn
            .config
            .tools
            .dynamic_tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<Vec<_>>();
        let extensions = Arc::new(extensions);
        let turn = erase_structured_turn(turn, Arc::clone(&extensions))?;
        let estimate = self
            .estimate_structured_turn_usage(extensions.as_ref(), &input, &turn, max_output_tokens)
            .await?;
        let lease = self
            .budget
            .reserve(extensions.as_ref(), &estimate, request_budget)?;
        let retry_policy = extensions.get::<RetryPolicy>().cloned().unwrap_or_default();
        let span = turn_span("structured_turn", estimate);
        self.emit_model_input_hook(
            &span,
            extensions.as_ref(),
            OperationKind::StructuredTurn,
            &input,
        )
        .await;
        log_input_transcript(&span, &input);
        Ok(PendingStructuredTurnWithTools {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: self.recovery.clone(),
            turns: Arc::clone(&self.turns),
            hooks: Arc::clone(&self.hooks),
            input,
            turn,
            availability,
            dynamic_names,
            estimate,
            retry_policy,
            span,
            reducer: StructuredTurnReducerWithTools::new(),
        })
    }

    pub(crate) async fn run_completion(
        &self,
        extensions: RequestExtensions,
        mut request: CompletionRequest,
    ) -> Result<PendingCompletion, LutumError> {
        let extensions = self.apply_default_extensions(extensions);
        Self::apply_completion_max_output_tokens_extension(&extensions, &mut request.options);
        let estimate = self
            .estimate_completion_usage(&extensions, &request)
            .await?;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, request.budget)?;
        let retry_policy = extensions.get::<RetryPolicy>().cloned().unwrap_or_default();
        let extensions = Arc::new(extensions);
        let span = turn_span("completion", estimate);
        Ok(PendingCompletion {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: self.recovery.clone(),
            completion: Arc::clone(&self.completion),
            hooks: Arc::clone(&self.hooks),
            request,
            estimate,
            retry_policy,
            span,
            reducer: CompletionReducer::new(),
        })
    }

    pub(crate) async fn run_structured_completion<O>(
        &self,
        extensions: RequestExtensions,
        mut request: StructuredCompletionRequest<O>,
    ) -> Result<PendingStructuredCompletion<O>, LutumError>
    where
        O: StructuredOutput,
    {
        let extensions = self.apply_default_extensions(extensions);
        Self::apply_max_output_tokens_extension(&extensions, &mut request.generation);
        let request_budget = request.budget;
        let max_output_tokens = request.generation.max_output_tokens;
        let request = erase_structured_completion_request(request)?;
        let estimate = self
            .estimate_structured_completion_usage(&extensions, &request, max_output_tokens)
            .await?;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, request_budget)?;
        let retry_policy = extensions.get::<RetryPolicy>().cloned().unwrap_or_default();
        let extensions = Arc::new(extensions);
        let span = turn_span("structured_completion", estimate);
        Ok(PendingStructuredCompletion {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: self.recovery.clone(),
            completion: Arc::clone(&self.completion),
            hooks: Arc::clone(&self.hooks),
            request,
            estimate,
            retry_policy,
            span,
            reducer: StructuredCompletionReducer::new(),
        })
    }
}

impl PendingTextTurn {
    async fn start_attempt(&self) -> Result<TextTurnEventStream, AgentError> {
        let stream = self
            .turns
            .text_turn(self.input.clone(), self.turn.clone())
            .await?;
        let stream = observe_text_stream(
            stream,
            Arc::clone(&self.hooks),
            Arc::clone(&self.extensions),
        );
        Ok(map_text_stream(stream))
    }

    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> TextTurnEventStream {
        let Self {
            recovery,
            turns,
            hooks,
            extensions,
            input,
            turn,
            estimate,
            retry_policy,
            span,
            ..
        } = self;
        boxed_sync_stream(try_stream! {
            let mut attempt = 1_u32;
            let mut cumulative_usage = Usage::zero();

            'attempts: loop {
                let stream = turns.text_turn(input.clone(), turn.clone()).await;
                let mut stream = match stream {
                    Ok(stream) => map_text_stream(observe_text_stream(
                        stream,
                        Arc::clone(&hooks),
                        Arc::clone(&extensions),
                    )),
                    Err(source) => {
                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&retry_policy, attempt, &source)
                        {
                            let accounted_usage =
                                recover_or_estimate_usage(recovery.as_deref(), OperationKind::TextTurn, None, estimate).await;
                            cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
                            yield TextTurnEvent::WillRetry {
                                attempt: next_attempt,
                                after,
                                kind,
                                status,
                                request_id: None,
                                accounted_usage,
                                cumulative_usage,
                            };
                            tokio::time::sleep(after).await;
                            attempt = next_attempt;
                            continue 'attempts;
                        }
                        Err(source)?;
                        break;
                    }
                };

                let mut request_id = None;
                while let Some(item) = stream.next().instrument(span.clone()).await {
                    match item {
                        Ok(event) => {
                            match &event {
                                TextTurnEvent::Started {
                                    request_id: event_request_id,
                                    ..
                                } => request_id = event_request_id.clone(),
                                TextTurnEvent::Completed {
                                    request_id: event_request_id,
                                    ..
                                } => {
                                    if let Some(event_request_id) = event_request_id.clone() {
                                        request_id = Some(event_request_id);
                                    }
                                }
                                _ => {}
                            }
                            yield event;
                        }
                        Err(source) => {
                            if let Some((next_attempt, after, status, kind)) =
                                maybe_retry_plan(&retry_policy, attempt, &source)
                            {
                                let accounted_usage = recover_or_estimate_usage(
                                    recovery.as_deref(),
                                    OperationKind::TextTurn,
                                    request_id.as_deref(),
                                    estimate,
                                )
                                .await;
                                cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
                                yield TextTurnEvent::WillRetry {
                                    attempt: next_attempt,
                                    after,
                                    kind,
                                    status,
                                    request_id,
                                    accounted_usage,
                                    cumulative_usage,
                                };
                                tokio::time::sleep(after).await;
                                attempt = next_attempt;
                                continue 'attempts;
                            }
                            Err(source)?;
                            break 'attempts;
                        }
                    }
                }

                break;
            }
        })
    }

    pub async fn collect_with<H>(
        mut self,
        mut handler: H,
    ) -> Result<StagedTextTurnResult, CollectError<H::Error, TextTurnReductionError, TextTurnState>>
    where
        H: EventHandler<TextTurnEvent, TextTurnState>,
    {
        let mut attempt = 1_u32;
        let mut cumulative_usage = Usage::zero();

        'attempts: loop {
            let mut stream = match self.start_attempt().await {
                Ok(stream) => stream,
                Err(source) => {
                    let partial = self.reducer.state().clone();
                    let accounted_usage = recover_or_estimate_usage(
                        self.recovery.as_deref(),
                        OperationKind::TextTurn,
                        self.reducer.state().request_id.as_deref(),
                        self.estimate,
                    )
                    .await;
                    let next_cumulative_usage = cumulative_usage.saturating_add(accounted_usage);

                    if let Some((next_attempt, after, status, kind)) =
                        maybe_retry_plan(&self.retry_policy, attempt, &source)
                    {
                        let retry_event = TextTurnEvent::WillRetry {
                            attempt: next_attempt,
                            after,
                            kind,
                            status,
                            request_id: self.reducer.state().request_id.clone(),
                            accounted_usage,
                            cumulative_usage: next_cumulative_usage,
                        };
                        match self.call_handler(&mut handler, &retry_event).await {
                            Ok(HandlerDirective::Continue) => {
                                cumulative_usage = next_cumulative_usage;
                                self.reducer.reset_for_retry();
                                tokio::time::sleep(after).await;
                                attempt = next_attempt;
                                continue 'attempts;
                            }
                            Ok(HandlerDirective::Stop) => {
                                if let Err(finalize_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    partial.request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state(&partial),
                                        finalize_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: finalize_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::TextTurn,
                                    partial.request_id.as_deref(),
                                    CollectErrorKind::Execution,
                                    summarize_text_state(&partial),
                                    source.to_string(),
                                );
                                return Err(CollectError::Execution { source, partial });
                            }
                            Err(handler_source) => {
                                if let Err(finalize_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    partial.request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state(&partial),
                                        finalize_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: finalize_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::TextTurn,
                                    partial.request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_text_state(&partial),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&handler_source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source: handler_source,
                                    partial: self.reducer.into_state(),
                                });
                            }
                        }
                    }

                    if let Err(finalize_source) = finalize_budget_cumulative(
                        &mut self.owned_lease,
                        &self.span,
                        partial.request_id.as_deref(),
                        next_cumulative_usage,
                    ) {
                        emit_raw_collect_error(
                            self.extensions.as_ref(),
                            OperationKind::TextTurn,
                            partial.request_id.as_deref(),
                            CollectErrorKind::Execution,
                            summarize_text_state(&partial),
                            finalize_source.to_string(),
                        );
                        return Err(CollectError::Execution {
                            source: finalize_source,
                            partial,
                        });
                    }
                    emit_raw_collect_error(
                        self.extensions.as_ref(),
                        OperationKind::TextTurn,
                        partial.request_id.as_deref(),
                        CollectErrorKind::Execution,
                        summarize_text_state(&partial),
                        source.to_string(),
                    );
                    return Err(CollectError::Execution {
                        source,
                        partial: self.reducer.into_state(),
                    });
                }
            };

            while let Some(item) = stream.next().instrument(self.span.clone()).await {
                match item {
                    Ok(event) => {
                        if let Err(source) = self.reducer.apply(&event) {
                            emit_raw_collect_error(
                                self.extensions.as_ref(),
                                OperationKind::TextTurn,
                                self.reducer.state().request_id.as_deref(),
                                CollectErrorKind::Reduction,
                                summarize_text_state(self.reducer.state()),
                                source.to_string(),
                            );
                            return Err(CollectError::Reduction {
                                source,
                                partial: self.reducer.state().clone(),
                            });
                        }
                        record_request_id(&self.span, self.reducer.state().request_id.as_deref());
                        if let TextTurnEvent::Completed { committed_turn, .. } = &event {
                            log_output_turn(&self.span, committed_turn);
                        }
                        if let Some(usage) = completed_usage_from_text(&event) {
                            let next_cumulative_usage = cumulative_usage.saturating_add(usage);
                            if let Err(source) = finalize_budget_cumulative(
                                &mut self.owned_lease,
                                &self.span,
                                self.reducer.state().request_id.as_deref(),
                                next_cumulative_usage,
                            ) {
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::TextTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Execution,
                                    summarize_text_state(self.reducer.state()),
                                    source.to_string(),
                                );
                                return Err(CollectError::Execution {
                                    source,
                                    partial: self.reducer.state().clone(),
                                });
                            }
                            if let Err(source) = self.call_handler(&mut handler, &event).await {
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::TextTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_text_state(self.reducer.state()),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source,
                                    partial: self.reducer.state().clone(),
                                });
                            }
                            let partial = self.reducer.state().clone();
                            return match self.reducer.into_result() {
                                Ok(mut result) => {
                                    result.cumulative_usage = next_cumulative_usage;
                                    Ok(result)
                                }
                                Err(source) => {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Reduction,
                                        summarize_text_state(&partial),
                                        source.to_string(),
                                    );
                                    Err(CollectError::Reduction { source, partial })
                                }
                            };
                        }

                        match self.call_handler(&mut handler, &event).await {
                            Ok(HandlerDirective::Continue) => {}
                            Ok(HandlerDirective::Stop) => {
                                let partial = self.reducer.state().clone();
                                let accounted_usage = recover_or_estimate_usage(
                                    self.recovery.as_deref(),
                                    OperationKind::TextTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    self.estimate,
                                )
                                .await;
                                let next_cumulative_usage =
                                    cumulative_usage.saturating_add(accounted_usage);
                                if let Err(source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    self.reducer.state().request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        self.reducer.state().request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state(&partial),
                                        source.to_string(),
                                    );
                                    return Err(CollectError::Execution { source, partial });
                                }
                                return Err(CollectError::Stopped {
                                    partial: self.reducer.into_state(),
                                });
                            }
                            Err(source) => {
                                let partial = self.reducer.state().clone();
                                let accounted_usage = recover_or_estimate_usage(
                                    self.recovery.as_deref(),
                                    OperationKind::TextTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    self.estimate,
                                )
                                .await;
                                let next_cumulative_usage =
                                    cumulative_usage.saturating_add(accounted_usage);
                                if let Err(execution_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    self.reducer.state().request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        self.reducer.state().request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state(&partial),
                                        execution_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: execution_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::TextTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_text_state(self.reducer.state()),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source,
                                    partial: self.reducer.into_state(),
                                });
                            }
                        }
                    }
                    Err(source) => {
                        let partial = self.reducer.state().clone();
                        let accounted_usage = recover_or_estimate_usage(
                            self.recovery.as_deref(),
                            OperationKind::TextTurn,
                            self.reducer.state().request_id.as_deref(),
                            self.estimate,
                        )
                        .await;
                        let next_cumulative_usage =
                            cumulative_usage.saturating_add(accounted_usage);

                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&self.retry_policy, attempt, &source)
                        {
                            let retry_event = TextTurnEvent::WillRetry {
                                attempt: next_attempt,
                                after,
                                kind,
                                status,
                                request_id: self.reducer.state().request_id.clone(),
                                accounted_usage,
                                cumulative_usage: next_cumulative_usage,
                            };
                            match self.call_handler(&mut handler, &retry_event).await {
                                Ok(HandlerDirective::Continue) => {
                                    cumulative_usage = next_cumulative_usage;
                                    self.reducer.reset_for_retry();
                                    tokio::time::sleep(after).await;
                                    attempt = next_attempt;
                                    continue 'attempts;
                                }
                                Ok(HandlerDirective::Stop) => {
                                    if let Err(finalize_source) = finalize_budget_cumulative(
                                        &mut self.owned_lease,
                                        &self.span,
                                        partial.request_id.as_deref(),
                                        next_cumulative_usage,
                                    ) {
                                        emit_raw_collect_error(
                                            self.extensions.as_ref(),
                                            OperationKind::TextTurn,
                                            partial.request_id.as_deref(),
                                            CollectErrorKind::Execution,
                                            summarize_text_state(&partial),
                                            finalize_source.to_string(),
                                        );
                                        return Err(CollectError::Execution {
                                            source: finalize_source,
                                            partial,
                                        });
                                    }
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state(&partial),
                                        source.to_string(),
                                    );
                                    return Err(CollectError::Execution { source, partial });
                                }
                                Err(handler_source) => {
                                    if let Err(finalize_source) = finalize_budget_cumulative(
                                        &mut self.owned_lease,
                                        &self.span,
                                        partial.request_id.as_deref(),
                                        next_cumulative_usage,
                                    ) {
                                        emit_raw_collect_error(
                                            self.extensions.as_ref(),
                                            OperationKind::TextTurn,
                                            partial.request_id.as_deref(),
                                            CollectErrorKind::Execution,
                                            summarize_text_state(&partial),
                                            finalize_source.to_string(),
                                        );
                                        return Err(CollectError::Execution {
                                            source: finalize_source,
                                            partial,
                                        });
                                    }
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Handler,
                                        summarize_text_state(&partial),
                                        format!(
                                            "handler error type={}",
                                            std::any::type_name_of_val(&handler_source)
                                        ),
                                    );
                                    return Err(CollectError::Handler {
                                        source: handler_source,
                                        partial: self.reducer.into_state(),
                                    });
                                }
                            }
                        }

                        if let Err(execution_source) = finalize_budget_cumulative(
                            &mut self.owned_lease,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            next_cumulative_usage,
                        ) {
                            emit_raw_collect_error(
                                self.extensions.as_ref(),
                                OperationKind::TextTurn,
                                self.reducer.state().request_id.as_deref(),
                                CollectErrorKind::Execution,
                                summarize_text_state(&partial),
                                execution_source.to_string(),
                            );
                            return Err(CollectError::Execution {
                                source: execution_source,
                                partial,
                            });
                        }
                        emit_raw_collect_error(
                            self.extensions.as_ref(),
                            OperationKind::TextTurn,
                            self.reducer.state().request_id.as_deref(),
                            CollectErrorKind::Execution,
                            summarize_text_state(&partial),
                            source.to_string(),
                        );
                        return Err(CollectError::Execution {
                            source,
                            partial: self.reducer.into_state(),
                        });
                    }
                }
            }

            let partial = self.reducer.state().clone();
            let accounted_usage = recover_or_estimate_usage(
                self.recovery.as_deref(),
                OperationKind::TextTurn,
                self.reducer.state().request_id.as_deref(),
                self.estimate,
            )
            .await;
            let next_cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
            if let Err(source) = finalize_budget_cumulative(
                &mut self.owned_lease,
                &self.span,
                self.reducer.state().request_id.as_deref(),
                next_cumulative_usage,
            ) {
                emit_raw_collect_error(
                    self.extensions.as_ref(),
                    OperationKind::TextTurn,
                    self.reducer.state().request_id.as_deref(),
                    CollectErrorKind::Execution,
                    summarize_text_state(&partial),
                    source.to_string(),
                );
                return Err(CollectError::Execution { source, partial });
            }
            emit_raw_collect_error(
                self.extensions.as_ref(),
                OperationKind::TextTurn,
                self.reducer.state().request_id.as_deref(),
                CollectErrorKind::UnexpectedEof,
                summarize_text_state(self.reducer.state()),
                "stream ended before completion".to_string(),
            );
            return Err(CollectError::UnexpectedEof {
                partial: self.reducer.into_state(),
            });
        }
    }

    pub async fn collect(
        self,
    ) -> Result<StagedTextTurnResult, CollectError<Infallible, TextTurnReductionError, TextTurnState>>
    {
        self.collect_with(NoopHandler).await
    }

    async fn call_handler<H>(
        &self,
        handler: &mut H,
        event: &TextTurnEvent,
    ) -> Result<HandlerDirective, H::Error>
    where
        H: EventHandler<TextTurnEvent, TextTurnState>,
    {
        let cx = HandlerContext {
            extensions: self.extensions.as_ref(),
            state: self.reducer.state(),
            remaining_budget: self.owned_lease.budget.remaining(self.extensions.as_ref()),
        };
        handler.on_event(event, &cx).await
    }
}

impl<T> PendingTextTurnWithTools<T>
where
    T: Toolset,
{
    async fn start_attempt(&self) -> Result<TextTurnEventStreamWithTools<T>, AgentError> {
        let stream = self
            .turns
            .text_turn(self.input.clone(), self.turn.clone())
            .await?;
        let stream = observe_text_stream(
            stream,
            Arc::clone(&self.hooks),
            Arc::clone(&self.extensions),
        );
        Ok(map_text_stream_with_tools::<T>(
            stream,
            self.availability.clone(),
            self.dynamic_names.clone(),
        ))
    }

    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> TextTurnEventStreamWithTools<T> {
        let Self {
            recovery,
            turns,
            hooks,
            extensions,
            input,
            turn,
            availability,
            dynamic_names,
            estimate,
            retry_policy,
            span,
            ..
        } = self;
        boxed_sync_stream(try_stream! {
            let mut attempt = 1_u32;
            let mut cumulative_usage = Usage::zero();

            'attempts: loop {
                let stream = turns.text_turn(input.clone(), turn.clone()).await;
                let mut stream = match stream {
                    Ok(stream) => map_text_stream_with_tools::<T>(
                        observe_text_stream(
                            stream,
                            Arc::clone(&hooks),
                            Arc::clone(&extensions),
                        ),
                        availability.clone(),
                        dynamic_names.clone(),
                    ),
                    Err(source) => {
                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&retry_policy, attempt, &source)
                        {
                            let accounted_usage =
                                recover_or_estimate_usage(recovery.as_deref(), OperationKind::TextTurn, None, estimate).await;
                            cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
                            yield TextTurnEventWithTools::WillRetry {
                                attempt: next_attempt,
                                after,
                                kind,
                                status,
                                request_id: None,
                                accounted_usage,
                                cumulative_usage,
                            };
                            tokio::time::sleep(after).await;
                            attempt = next_attempt;
                            continue 'attempts;
                        }
                        Err(source)?;
                        break;
                    }
                };

                let mut request_id = None;
                while let Some(item) = stream.next().instrument(span.clone()).await {
                    match item {
                        Ok(event) => {
                            match &event {
                                TextTurnEventWithTools::Started {
                                    request_id: event_request_id,
                                    ..
                                } => request_id = event_request_id.clone(),
                                TextTurnEventWithTools::Completed {
                                    request_id: event_request_id,
                                    ..
                                } => {
                                    if let Some(event_request_id) = event_request_id.clone() {
                                        request_id = Some(event_request_id);
                                    }
                                }
                                _ => {}
                            }
                            yield event;
                        }
                        Err(source) => {
                            if let Some((next_attempt, after, status, kind)) =
                                maybe_retry_plan(&retry_policy, attempt, &source)
                            {
                                let accounted_usage = recover_or_estimate_usage(
                                    recovery.as_deref(),
                                    OperationKind::TextTurn,
                                    request_id.as_deref(),
                                    estimate,
                                )
                                .await;
                                cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
                                yield TextTurnEventWithTools::WillRetry {
                                    attempt: next_attempt,
                                    after,
                                    kind,
                                    status,
                                    request_id,
                                    accounted_usage,
                                    cumulative_usage,
                                };
                                tokio::time::sleep(after).await;
                                attempt = next_attempt;
                                continue 'attempts;
                            }
                            Err(source)?;
                            break 'attempts;
                        }
                    }
                }

                break;
            }
        })
    }

    pub async fn collect_with<H>(
        mut self,
        mut handler: H,
    ) -> Result<
        StagedTextTurnOutcomeWithTools<T>,
        CollectError<H::Error, TextTurnReductionError, TextTurnStateWithTools<T>>,
    >
    where
        H: EventHandler<TextTurnEventWithTools<T>, TextTurnStateWithTools<T>>,
    {
        let mut attempt = 1_u32;
        let mut cumulative_usage = Usage::zero();

        'attempts: loop {
            let mut stream = match self.start_attempt().await {
                Ok(stream) => stream,
                Err(source) => {
                    let partial = self.reducer.state().clone();
                    let accounted_usage = recover_or_estimate_usage(
                        self.recovery.as_deref(),
                        OperationKind::TextTurn,
                        self.reducer.state().request_id.as_deref(),
                        self.estimate,
                    )
                    .await;
                    let next_cumulative_usage = cumulative_usage.saturating_add(accounted_usage);

                    if let Some((next_attempt, after, status, kind)) =
                        maybe_retry_plan(&self.retry_policy, attempt, &source)
                    {
                        let retry_event = TextTurnEventWithTools::WillRetry {
                            attempt: next_attempt,
                            after,
                            kind,
                            status,
                            request_id: self.reducer.state().request_id.clone(),
                            accounted_usage,
                            cumulative_usage: next_cumulative_usage,
                        };
                        match self.call_handler(&mut handler, &retry_event).await {
                            Ok(HandlerDirective::Continue) => {
                                cumulative_usage = next_cumulative_usage;
                                self.reducer.reset_for_retry();
                                tokio::time::sleep(after).await;
                                attempt = next_attempt;
                                continue 'attempts;
                            }
                            Ok(HandlerDirective::Stop) => {
                                if let Err(finalize_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    partial.request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state_with_tools(&partial),
                                        finalize_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: finalize_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::TextTurn,
                                    partial.request_id.as_deref(),
                                    CollectErrorKind::Execution,
                                    summarize_text_state_with_tools(&partial),
                                    source.to_string(),
                                );
                                return Err(CollectError::Execution { source, partial });
                            }
                            Err(handler_source) => {
                                if let Err(finalize_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    partial.request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state_with_tools(&partial),
                                        finalize_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: finalize_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::TextTurn,
                                    partial.request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_text_state_with_tools(&partial),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&handler_source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source: handler_source,
                                    partial: self.reducer.into_state(),
                                });
                            }
                        }
                    }

                    if let Err(finalize_source) = finalize_budget_cumulative(
                        &mut self.owned_lease,
                        &self.span,
                        partial.request_id.as_deref(),
                        next_cumulative_usage,
                    ) {
                        emit_raw_collect_error(
                            self.extensions.as_ref(),
                            OperationKind::TextTurn,
                            partial.request_id.as_deref(),
                            CollectErrorKind::Execution,
                            summarize_text_state_with_tools(&partial),
                            finalize_source.to_string(),
                        );
                        return Err(CollectError::Execution {
                            source: finalize_source,
                            partial,
                        });
                    }
                    emit_raw_collect_error(
                        self.extensions.as_ref(),
                        OperationKind::TextTurn,
                        partial.request_id.as_deref(),
                        CollectErrorKind::Execution,
                        summarize_text_state_with_tools(&partial),
                        source.to_string(),
                    );
                    return Err(CollectError::Execution {
                        source,
                        partial: self.reducer.into_state(),
                    });
                }
            };

            while let Some(item) = stream.next().instrument(self.span.clone()).await {
                match item {
                    Ok(event) => {
                        if let Err(source) = self.reducer.apply(&event) {
                            emit_raw_collect_error(
                                self.extensions.as_ref(),
                                OperationKind::TextTurn,
                                self.reducer.state().request_id.as_deref(),
                                CollectErrorKind::Reduction,
                                summarize_text_state_with_tools(self.reducer.state()),
                                source.to_string(),
                            );
                            return Err(CollectError::Reduction {
                                source,
                                partial: self.reducer.state().clone(),
                            });
                        }
                        record_request_id(&self.span, self.reducer.state().request_id.as_deref());
                        if let TextTurnEventWithTools::Completed { committed_turn, .. } = &event {
                            log_output_turn(&self.span, committed_turn);
                        }
                        if let Some(usage) = completed_usage_from_text_with_tools(&event) {
                            let next_cumulative_usage = cumulative_usage.saturating_add(usage);
                            if let Err(source) = finalize_budget_cumulative(
                                &mut self.owned_lease,
                                &self.span,
                                self.reducer.state().request_id.as_deref(),
                                next_cumulative_usage,
                            ) {
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::TextTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Execution,
                                    summarize_text_state_with_tools(self.reducer.state()),
                                    source.to_string(),
                                );
                                return Err(CollectError::Execution {
                                    source,
                                    partial: self.reducer.state().clone(),
                                });
                            }
                            if let Err(source) = self.call_handler(&mut handler, &event).await {
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::TextTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_text_state_with_tools(self.reducer.state()),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source,
                                    partial: self.reducer.state().clone(),
                                });
                            }
                            let partial = self.reducer.state().clone();
                            return match self.reducer.into_result() {
                                Ok(mut outcome) => {
                                    match &mut outcome {
                                        StagedTextTurnOutcomeWithTools::Turn(result) => {
                                            result.cumulative_usage = next_cumulative_usage;
                                        }
                                        StagedTextTurnOutcomeWithTools::FinishedNoOutput(
                                            result,
                                        ) => {
                                            result.cumulative_usage = next_cumulative_usage;
                                        }
                                    }
                                    match recover_required_text_tool_outcome(
                                        outcome,
                                        &partial,
                                        self.fallback_parser.as_deref(),
                                        &self.tool_constraints,
                                        &self.tool_definitions,
                                        self.extensions.as_ref(),
                                    ) {
                                        Ok(outcome) => Ok(outcome),
                                        Err(source) => {
                                            emit_raw_collect_error(
                                                self.extensions.as_ref(),
                                                OperationKind::TextTurn,
                                                partial.request_id.as_deref(),
                                                CollectErrorKind::Reduction,
                                                summarize_text_state_with_tools(&partial),
                                                source.to_string(),
                                            );
                                            Err(CollectError::Reduction { source, partial })
                                        }
                                    }
                                }
                                Err(source) => {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Reduction,
                                        summarize_text_state_with_tools(&partial),
                                        source.to_string(),
                                    );
                                    Err(CollectError::Reduction { source, partial })
                                }
                            };
                        }

                        match self.call_handler(&mut handler, &event).await {
                            Ok(HandlerDirective::Continue) => {}
                            Ok(HandlerDirective::Stop) => {
                                let partial = self.reducer.state().clone();
                                let accounted_usage = recover_or_estimate_usage(
                                    self.recovery.as_deref(),
                                    OperationKind::TextTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    self.estimate,
                                )
                                .await;
                                let next_cumulative_usage =
                                    cumulative_usage.saturating_add(accounted_usage);
                                if let Err(source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    self.reducer.state().request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        self.reducer.state().request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state_with_tools(&partial),
                                        source.to_string(),
                                    );
                                    return Err(CollectError::Execution { source, partial });
                                }
                                return Err(CollectError::Stopped {
                                    partial: self.reducer.into_state(),
                                });
                            }
                            Err(source) => {
                                let partial = self.reducer.state().clone();
                                let accounted_usage = recover_or_estimate_usage(
                                    self.recovery.as_deref(),
                                    OperationKind::TextTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    self.estimate,
                                )
                                .await;
                                let next_cumulative_usage =
                                    cumulative_usage.saturating_add(accounted_usage);
                                if let Err(execution_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    self.reducer.state().request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        self.reducer.state().request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state_with_tools(&partial),
                                        execution_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: execution_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::TextTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_text_state_with_tools(self.reducer.state()),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source,
                                    partial: self.reducer.into_state(),
                                });
                            }
                        }
                    }
                    Err(source) => {
                        let partial = self.reducer.state().clone();
                        let accounted_usage = recover_or_estimate_usage(
                            self.recovery.as_deref(),
                            OperationKind::TextTurn,
                            self.reducer.state().request_id.as_deref(),
                            self.estimate,
                        )
                        .await;
                        let next_cumulative_usage =
                            cumulative_usage.saturating_add(accounted_usage);

                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&self.retry_policy, attempt, &source)
                        {
                            let retry_event = TextTurnEventWithTools::WillRetry {
                                attempt: next_attempt,
                                after,
                                kind,
                                status,
                                request_id: self.reducer.state().request_id.clone(),
                                accounted_usage,
                                cumulative_usage: next_cumulative_usage,
                            };
                            match self.call_handler(&mut handler, &retry_event).await {
                                Ok(HandlerDirective::Continue) => {
                                    cumulative_usage = next_cumulative_usage;
                                    self.reducer.reset_for_retry();
                                    tokio::time::sleep(after).await;
                                    attempt = next_attempt;
                                    continue 'attempts;
                                }
                                Ok(HandlerDirective::Stop) => {
                                    if let Err(finalize_source) = finalize_budget_cumulative(
                                        &mut self.owned_lease,
                                        &self.span,
                                        partial.request_id.as_deref(),
                                        next_cumulative_usage,
                                    ) {
                                        emit_raw_collect_error(
                                            self.extensions.as_ref(),
                                            OperationKind::TextTurn,
                                            partial.request_id.as_deref(),
                                            CollectErrorKind::Execution,
                                            summarize_text_state_with_tools(&partial),
                                            finalize_source.to_string(),
                                        );
                                        return Err(CollectError::Execution {
                                            source: finalize_source,
                                            partial,
                                        });
                                    }
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state_with_tools(&partial),
                                        source.to_string(),
                                    );
                                    return Err(CollectError::Execution { source, partial });
                                }
                                Err(handler_source) => {
                                    if let Err(finalize_source) = finalize_budget_cumulative(
                                        &mut self.owned_lease,
                                        &self.span,
                                        partial.request_id.as_deref(),
                                        next_cumulative_usage,
                                    ) {
                                        emit_raw_collect_error(
                                            self.extensions.as_ref(),
                                            OperationKind::TextTurn,
                                            partial.request_id.as_deref(),
                                            CollectErrorKind::Execution,
                                            summarize_text_state_with_tools(&partial),
                                            finalize_source.to_string(),
                                        );
                                        return Err(CollectError::Execution {
                                            source: finalize_source,
                                            partial,
                                        });
                                    }
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Handler,
                                        summarize_text_state_with_tools(&partial),
                                        format!(
                                            "handler error type={}",
                                            std::any::type_name_of_val(&handler_source)
                                        ),
                                    );
                                    return Err(CollectError::Handler {
                                        source: handler_source,
                                        partial: self.reducer.into_state(),
                                    });
                                }
                            }
                        }

                        if let Err(execution_source) = finalize_budget_cumulative(
                            &mut self.owned_lease,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            next_cumulative_usage,
                        ) {
                            emit_raw_collect_error(
                                self.extensions.as_ref(),
                                OperationKind::TextTurn,
                                self.reducer.state().request_id.as_deref(),
                                CollectErrorKind::Execution,
                                summarize_text_state_with_tools(&partial),
                                execution_source.to_string(),
                            );
                            return Err(CollectError::Execution {
                                source: execution_source,
                                partial,
                            });
                        }
                        emit_raw_collect_error(
                            self.extensions.as_ref(),
                            OperationKind::TextTurn,
                            self.reducer.state().request_id.as_deref(),
                            CollectErrorKind::Execution,
                            summarize_text_state_with_tools(&partial),
                            source.to_string(),
                        );
                        return Err(CollectError::Execution {
                            source,
                            partial: self.reducer.into_state(),
                        });
                    }
                }
            }

            let partial = self.reducer.state().clone();
            let accounted_usage = recover_or_estimate_usage(
                self.recovery.as_deref(),
                OperationKind::TextTurn,
                self.reducer.state().request_id.as_deref(),
                self.estimate,
            )
            .await;
            let next_cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
            if let Err(source) = finalize_budget_cumulative(
                &mut self.owned_lease,
                &self.span,
                self.reducer.state().request_id.as_deref(),
                next_cumulative_usage,
            ) {
                emit_raw_collect_error(
                    self.extensions.as_ref(),
                    OperationKind::TextTurn,
                    self.reducer.state().request_id.as_deref(),
                    CollectErrorKind::Execution,
                    summarize_text_state_with_tools(&partial),
                    source.to_string(),
                );
                return Err(CollectError::Execution { source, partial });
            }
            emit_raw_collect_error(
                self.extensions.as_ref(),
                OperationKind::TextTurn,
                self.reducer.state().request_id.as_deref(),
                CollectErrorKind::UnexpectedEof,
                summarize_text_state_with_tools(self.reducer.state()),
                "stream ended before completion".to_string(),
            );
            return Err(CollectError::UnexpectedEof {
                partial: self.reducer.into_state(),
            });
        }
    }

    pub async fn collect(
        self,
    ) -> Result<
        StagedTextTurnOutcomeWithTools<T>,
        CollectError<Infallible, TextTurnReductionError, TextTurnStateWithTools<T>>,
    > {
        self.collect_with(NoopHandler).await
    }

    /// Collect this pending text+tools stream with a handler that can return a
    /// synthetic turn/tool outcome or recover from controlled collection errors.
    ///
    /// The reducer is applied before the handler observes each event, matching
    /// the existing `collect_with` event ordering.
    pub async fn collect_controlled_with<H>(
        mut self,
        mut handler: H,
    ) -> Result<
        StagedTextTurnOutcomeWithTools<T>,
        CollectError<H::Error, TextTurnReductionError, TextTurnStateWithTools<T>>,
    >
    where
        H: TextToolEventHandler<T>,
    {
        let mut attempt = 1_u32;
        let mut cumulative_usage = Usage::zero();

        'attempts: loop {
            let mut stream = match self.start_attempt().await {
                Ok(stream) => stream,
                Err(source) => {
                    let partial = self.reducer.state().clone();
                    let accounted_usage = recover_or_estimate_usage(
                        self.recovery.as_deref(),
                        OperationKind::TextTurn,
                        self.reducer.state().request_id.as_deref(),
                        self.estimate,
                    )
                    .await;
                    let next_cumulative_usage = cumulative_usage.saturating_add(accounted_usage);

                    if let Some((next_attempt, after, status, kind)) =
                        maybe_retry_plan(&self.retry_policy, attempt, &source)
                    {
                        let retry_event = TextTurnEventWithTools::WillRetry {
                            attempt: next_attempt,
                            after,
                            kind,
                            status,
                            request_id: self.reducer.state().request_id.clone(),
                            accounted_usage,
                            cumulative_usage: next_cumulative_usage,
                        };
                        match self
                            .call_controlled_handler(&mut handler, &retry_event)
                            .await
                        {
                            Ok(TextToolHandlerDirective::Continue) => {
                                cumulative_usage = next_cumulative_usage;
                                self.reducer.reset_for_retry();
                                tokio::time::sleep(after).await;
                                attempt = next_attempt;
                                continue 'attempts;
                            }
                            Ok(TextToolHandlerDirective::Stop) => {
                                if let Err(finalize_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    partial.request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state_with_tools(&partial),
                                        finalize_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: finalize_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::TextTurn,
                                    partial.request_id.as_deref(),
                                    CollectErrorKind::Execution,
                                    summarize_text_state_with_tools(&partial),
                                    source.to_string(),
                                );
                                return Err(CollectError::Execution { source, partial });
                            }
                            Ok(TextToolHandlerDirective::Return(synthetic)) => {
                                return self
                                    .finish_controlled_synthetic(synthetic, cumulative_usage)
                                    .await;
                            }
                            Err(handler_source) => {
                                if let Err(finalize_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    partial.request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state_with_tools(&partial),
                                        finalize_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: finalize_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::TextTurn,
                                    partial.request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_text_state_with_tools(&partial),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&handler_source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source: handler_source,
                                    partial: self.reducer.into_state(),
                                });
                            }
                        }
                    }

                    if let Some(outcome) = self
                        .try_controlled_execution_error(
                            &mut handler,
                            &source,
                            cumulative_usage,
                            next_cumulative_usage,
                            &partial,
                        )
                        .await?
                    {
                        return Ok(outcome);
                    }
                    emit_raw_collect_error(
                        self.extensions.as_ref(),
                        OperationKind::TextTurn,
                        partial.request_id.as_deref(),
                        CollectErrorKind::Execution,
                        summarize_text_state_with_tools(&partial),
                        source.to_string(),
                    );
                    return Err(CollectError::Execution {
                        source,
                        partial: self.reducer.into_state(),
                    });
                }
            };

            while let Some(item) = stream.next().instrument(self.span.clone()).await {
                match item {
                    Ok(event) => {
                        if let Err(source) = self.reducer.apply(&event) {
                            let partial = self.reducer.state().clone();
                            let accounted_usage = recover_or_estimate_usage(
                                self.recovery.as_deref(),
                                OperationKind::TextTurn,
                                self.reducer.state().request_id.as_deref(),
                                self.estimate,
                            )
                            .await;
                            let next_cumulative_usage =
                                cumulative_usage.saturating_add(accounted_usage);
                            if let Some(outcome) = self
                                .try_controlled_reduction_error(
                                    &mut handler,
                                    &source,
                                    cumulative_usage,
                                    next_cumulative_usage,
                                    &partial,
                                )
                                .await?
                            {
                                return Ok(outcome);
                            }
                            emit_raw_collect_error(
                                self.extensions.as_ref(),
                                OperationKind::TextTurn,
                                self.reducer.state().request_id.as_deref(),
                                CollectErrorKind::Reduction,
                                summarize_text_state_with_tools(self.reducer.state()),
                                source.to_string(),
                            );
                            return Err(CollectError::Reduction { source, partial });
                        }
                        record_request_id(&self.span, self.reducer.state().request_id.as_deref());
                        if let TextTurnEventWithTools::Completed { committed_turn, .. } = &event {
                            log_output_turn(&self.span, committed_turn);
                        }
                        if let Some(usage) = completed_usage_from_text_with_tools(&event) {
                            let next_cumulative_usage = cumulative_usage.saturating_add(usage);
                            if let Err(source) = finalize_budget_cumulative(
                                &mut self.owned_lease,
                                &self.span,
                                self.reducer.state().request_id.as_deref(),
                                next_cumulative_usage,
                            ) {
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::TextTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Execution,
                                    summarize_text_state_with_tools(self.reducer.state()),
                                    source.to_string(),
                                );
                                return Err(CollectError::Execution {
                                    source,
                                    partial: self.reducer.state().clone(),
                                });
                            }
                            let partial = self.reducer.state().clone();
                            match self.call_controlled_handler(&mut handler, &event).await {
                                Ok(TextToolHandlerDirective::Continue) => {}
                                Ok(TextToolHandlerDirective::Stop) => {
                                    return Err(CollectError::Stopped { partial });
                                }
                                Ok(TextToolHandlerDirective::Return(synthetic)) => {
                                    return self
                                        .finish_controlled_synthetic(synthetic, cumulative_usage)
                                        .await;
                                }
                                Err(source) => {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        self.reducer.state().request_id.as_deref(),
                                        CollectErrorKind::Handler,
                                        summarize_text_state_with_tools(self.reducer.state()),
                                        format!(
                                            "handler error type={}",
                                            std::any::type_name_of_val(&source)
                                        ),
                                    );
                                    return Err(CollectError::Handler { source, partial });
                                }
                            }
                            return match partial.clone().finish() {
                                Ok(mut outcome) => {
                                    match &mut outcome {
                                        StagedTextTurnOutcomeWithTools::Turn(result) => {
                                            result.cumulative_usage = next_cumulative_usage;
                                        }
                                        StagedTextTurnOutcomeWithTools::FinishedNoOutput(
                                            result,
                                        ) => {
                                            result.cumulative_usage = next_cumulative_usage;
                                        }
                                    }
                                    match recover_required_text_tool_outcome(
                                        outcome,
                                        &partial,
                                        self.fallback_parser.as_deref(),
                                        &self.tool_constraints,
                                        &self.tool_definitions,
                                        self.extensions.as_ref(),
                                    ) {
                                        Ok(outcome) => Ok(outcome),
                                        Err(source) => {
                                            if let Some(outcome) = self
                                                .try_controlled_reduction_error(
                                                    &mut handler,
                                                    &source,
                                                    cumulative_usage,
                                                    next_cumulative_usage,
                                                    &partial,
                                                )
                                                .await?
                                            {
                                                return Ok(outcome);
                                            }
                                            emit_raw_collect_error(
                                                self.extensions.as_ref(),
                                                OperationKind::TextTurn,
                                                partial.request_id.as_deref(),
                                                CollectErrorKind::Reduction,
                                                summarize_text_state_with_tools(&partial),
                                                source.to_string(),
                                            );
                                            Err(CollectError::Reduction { source, partial })
                                        }
                                    }
                                }
                                Err(source) => {
                                    if let Some(outcome) = self
                                        .try_controlled_reduction_error(
                                            &mut handler,
                                            &source,
                                            cumulative_usage,
                                            next_cumulative_usage,
                                            &partial,
                                        )
                                        .await?
                                    {
                                        return Ok(outcome);
                                    }
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Reduction,
                                        summarize_text_state_with_tools(&partial),
                                        source.to_string(),
                                    );
                                    Err(CollectError::Reduction { source, partial })
                                }
                            };
                        }

                        match self.call_controlled_handler(&mut handler, &event).await {
                            Ok(TextToolHandlerDirective::Continue) => {}
                            Ok(TextToolHandlerDirective::Stop) => {
                                let partial = self.reducer.state().clone();
                                let accounted_usage = recover_or_estimate_usage(
                                    self.recovery.as_deref(),
                                    OperationKind::TextTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    self.estimate,
                                )
                                .await;
                                let next_cumulative_usage =
                                    cumulative_usage.saturating_add(accounted_usage);
                                if let Err(source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    self.reducer.state().request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        self.reducer.state().request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state_with_tools(&partial),
                                        source.to_string(),
                                    );
                                    return Err(CollectError::Execution { source, partial });
                                }
                                return Err(CollectError::Stopped {
                                    partial: self.reducer.into_state(),
                                });
                            }
                            Ok(TextToolHandlerDirective::Return(synthetic)) => {
                                return self
                                    .finish_controlled_synthetic(synthetic, cumulative_usage)
                                    .await;
                            }
                            Err(source) => {
                                let partial = self.reducer.state().clone();
                                let accounted_usage = recover_or_estimate_usage(
                                    self.recovery.as_deref(),
                                    OperationKind::TextTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    self.estimate,
                                )
                                .await;
                                let next_cumulative_usage =
                                    cumulative_usage.saturating_add(accounted_usage);
                                if let Err(execution_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    self.reducer.state().request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        self.reducer.state().request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state_with_tools(&partial),
                                        execution_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: execution_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::TextTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_text_state_with_tools(self.reducer.state()),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source,
                                    partial: self.reducer.into_state(),
                                });
                            }
                        }
                    }
                    Err(source) => {
                        let partial = self.reducer.state().clone();
                        let accounted_usage = recover_or_estimate_usage(
                            self.recovery.as_deref(),
                            OperationKind::TextTurn,
                            self.reducer.state().request_id.as_deref(),
                            self.estimate,
                        )
                        .await;
                        let next_cumulative_usage =
                            cumulative_usage.saturating_add(accounted_usage);

                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&self.retry_policy, attempt, &source)
                        {
                            let retry_event = TextTurnEventWithTools::WillRetry {
                                attempt: next_attempt,
                                after,
                                kind,
                                status,
                                request_id: self.reducer.state().request_id.clone(),
                                accounted_usage,
                                cumulative_usage: next_cumulative_usage,
                            };
                            match self
                                .call_controlled_handler(&mut handler, &retry_event)
                                .await
                            {
                                Ok(TextToolHandlerDirective::Continue) => {
                                    cumulative_usage = next_cumulative_usage;
                                    self.reducer.reset_for_retry();
                                    tokio::time::sleep(after).await;
                                    attempt = next_attempt;
                                    continue 'attempts;
                                }
                                Ok(TextToolHandlerDirective::Stop) => {
                                    if let Err(finalize_source) = finalize_budget_cumulative(
                                        &mut self.owned_lease,
                                        &self.span,
                                        partial.request_id.as_deref(),
                                        next_cumulative_usage,
                                    ) {
                                        emit_raw_collect_error(
                                            self.extensions.as_ref(),
                                            OperationKind::TextTurn,
                                            partial.request_id.as_deref(),
                                            CollectErrorKind::Execution,
                                            summarize_text_state_with_tools(&partial),
                                            finalize_source.to_string(),
                                        );
                                        return Err(CollectError::Execution {
                                            source: finalize_source,
                                            partial,
                                        });
                                    }
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_text_state_with_tools(&partial),
                                        source.to_string(),
                                    );
                                    return Err(CollectError::Execution { source, partial });
                                }
                                Ok(TextToolHandlerDirective::Return(synthetic)) => {
                                    return self
                                        .finish_controlled_synthetic(synthetic, cumulative_usage)
                                        .await;
                                }
                                Err(handler_source) => {
                                    if let Err(finalize_source) = finalize_budget_cumulative(
                                        &mut self.owned_lease,
                                        &self.span,
                                        partial.request_id.as_deref(),
                                        next_cumulative_usage,
                                    ) {
                                        emit_raw_collect_error(
                                            self.extensions.as_ref(),
                                            OperationKind::TextTurn,
                                            partial.request_id.as_deref(),
                                            CollectErrorKind::Execution,
                                            summarize_text_state_with_tools(&partial),
                                            finalize_source.to_string(),
                                        );
                                        return Err(CollectError::Execution {
                                            source: finalize_source,
                                            partial,
                                        });
                                    }
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::TextTurn,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Handler,
                                        summarize_text_state_with_tools(&partial),
                                        format!(
                                            "handler error type={}",
                                            std::any::type_name_of_val(&handler_source)
                                        ),
                                    );
                                    return Err(CollectError::Handler {
                                        source: handler_source,
                                        partial: self.reducer.into_state(),
                                    });
                                }
                            }
                        }

                        if let Some(outcome) = self
                            .try_controlled_execution_error(
                                &mut handler,
                                &source,
                                cumulative_usage,
                                next_cumulative_usage,
                                &partial,
                            )
                            .await?
                        {
                            return Ok(outcome);
                        }
                        emit_raw_collect_error(
                            self.extensions.as_ref(),
                            OperationKind::TextTurn,
                            self.reducer.state().request_id.as_deref(),
                            CollectErrorKind::Execution,
                            summarize_text_state_with_tools(&partial),
                            source.to_string(),
                        );
                        return Err(CollectError::Execution {
                            source,
                            partial: self.reducer.into_state(),
                        });
                    }
                }
            }

            let partial = self.reducer.state().clone();
            let accounted_usage = recover_or_estimate_usage(
                self.recovery.as_deref(),
                OperationKind::TextTurn,
                self.reducer.state().request_id.as_deref(),
                self.estimate,
            )
            .await;
            let next_cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
            if let Some(outcome) = self
                .try_controlled_unexpected_eof(
                    &mut handler,
                    cumulative_usage,
                    next_cumulative_usage,
                    &partial,
                )
                .await?
            {
                return Ok(outcome);
            }
            emit_raw_collect_error(
                self.extensions.as_ref(),
                OperationKind::TextTurn,
                self.reducer.state().request_id.as_deref(),
                CollectErrorKind::UnexpectedEof,
                summarize_text_state_with_tools(self.reducer.state()),
                "stream ended before completion".to_string(),
            );
            return Err(CollectError::UnexpectedEof {
                partial: self.reducer.into_state(),
            });
        }
    }

    async fn call_handler<H>(
        &self,
        handler: &mut H,
        event: &TextTurnEventWithTools<T>,
    ) -> Result<HandlerDirective, H::Error>
    where
        H: EventHandler<TextTurnEventWithTools<T>, TextTurnStateWithTools<T>>,
    {
        let cx = HandlerContext {
            extensions: self.extensions.as_ref(),
            state: self.reducer.state(),
            remaining_budget: self.owned_lease.budget.remaining(self.extensions.as_ref()),
        };
        handler.on_event(event, &cx).await
    }

    async fn call_controlled_handler<H>(
        &self,
        handler: &mut H,
        event: &TextTurnEventWithTools<T>,
    ) -> Result<TextToolHandlerDirective<T>, H::Error>
    where
        H: TextToolEventHandler<T>,
    {
        let cx = self.controlled_handler_context();
        handler.on_event(event, &cx).await
    }

    async fn call_controlled_error_handler<H>(
        &self,
        handler: &mut H,
        error: TextToolCollectError<'_>,
    ) -> Result<TextToolErrorDirective<T>, H::Error>
    where
        H: TextToolEventHandler<T>,
    {
        let cx = self.controlled_handler_context();
        handler.on_error(error, &cx).await
    }

    fn controlled_handler_context(&self) -> TextToolHandlerContext<'_, T> {
        TextToolHandlerContext {
            extensions: self.extensions.as_ref(),
            state: self.reducer.state(),
            remaining_budget: self.owned_lease.budget.remaining(self.extensions.as_ref()),
            constraints: &self.tool_constraints,
            tool_definitions: &self.tool_definitions,
        }
    }

    async fn finish_controlled_synthetic<HandlerError>(
        &mut self,
        synthetic: SyntheticTextToolTurn<T>,
        cumulative_usage: Usage,
    ) -> Result<
        StagedTextTurnOutcomeWithTools<T>,
        CollectError<HandlerError, TextTurnReductionError, TextTurnStateWithTools<T>>,
    > {
        let partial = self.reducer.state().clone();
        let (outcome, next_cumulative_usage) = match self
            .synthesize_text_tool_outcome(synthetic, cumulative_usage)
            .await
        {
            Ok(outcome) => outcome,
            Err(source) => {
                emit_raw_collect_error(
                    self.extensions.as_ref(),
                    OperationKind::TextTurn,
                    partial.request_id.as_deref(),
                    CollectErrorKind::Reduction,
                    summarize_text_state_with_tools(&partial),
                    source.to_string(),
                );
                return Err(CollectError::Reduction { source, partial });
            }
        };
        let outcome = match recover_required_text_tool_outcome(
            outcome,
            &partial,
            self.fallback_parser.as_deref(),
            &self.tool_constraints,
            &self.tool_definitions,
            self.extensions.as_ref(),
        ) {
            Ok(outcome) => outcome,
            Err(source) => {
                if let Err(finalize_source) = finalize_budget_cumulative(
                    &mut self.owned_lease,
                    &self.span,
                    partial.request_id.as_deref(),
                    next_cumulative_usage,
                ) {
                    emit_raw_collect_error(
                        self.extensions.as_ref(),
                        OperationKind::TextTurn,
                        partial.request_id.as_deref(),
                        CollectErrorKind::Execution,
                        summarize_text_state_with_tools(&partial),
                        finalize_source.to_string(),
                    );
                    return Err(CollectError::Execution {
                        source: finalize_source,
                        partial,
                    });
                }
                emit_raw_collect_error(
                    self.extensions.as_ref(),
                    OperationKind::TextTurn,
                    partial.request_id.as_deref(),
                    CollectErrorKind::Reduction,
                    summarize_text_state_with_tools(&partial),
                    source.to_string(),
                );
                return Err(CollectError::Reduction { source, partial });
            }
        };
        if let Err(source) = finalize_budget_cumulative(
            &mut self.owned_lease,
            &self.span,
            self.reducer.state().request_id.as_deref(),
            next_cumulative_usage,
        ) {
            emit_raw_collect_error(
                self.extensions.as_ref(),
                OperationKind::TextTurn,
                self.reducer.state().request_id.as_deref(),
                CollectErrorKind::Execution,
                summarize_text_state_with_tools(&partial),
                source.to_string(),
            );
            return Err(CollectError::Execution { source, partial });
        }
        Ok(outcome)
    }

    async fn synthesize_text_tool_outcome(
        &self,
        mut synthetic: SyntheticTextToolTurn<T>,
        cumulative_usage: Usage,
    ) -> Result<(StagedTextTurnOutcomeWithTools<T>, Usage), TextTurnReductionError> {
        let state = self.reducer.state();
        let assistant_tool_calls = synthetic
            .assistant_turn
            .items()
            .iter()
            .filter(|item| matches!(item, AssistantTurnItem::ToolCall { .. }))
            .count();
        if !synthetic.tool_calls.is_empty() {
            let recovered = RecoveredTextToolCalls::<T>::new(
                synthetic.assistant_turn.clone(),
                synthetic.tool_calls.clone(),
            );
            let recovered = validate_recovered_text_tool_calls(
                recovered,
                &self.tool_constraints.requirement,
                &self.tool_definitions,
            )
            .map_err(|source| tool_fallback_error_from_state::<T>(state, source))?;
            synthetic.assistant_turn = recovered.assistant_turn;
            synthetic.tool_calls = recovered.tool_calls;
        } else if assistant_tool_calls != 0 && synthetic.recoverable_tool_call_issues.is_empty() {
            return Err(tool_fallback_error_from_state::<T>(
                state,
                ToolCallFallbackError::ToolCallCountMismatch {
                    assistant_tool_calls,
                    tool_calls: 0,
                },
            ));
        }

        let usage = if let Some(usage) = synthetic.usage {
            usage
        } else if let Some(usage) = state.usage {
            usage
        } else {
            recover_or_estimate_usage(
                self.recovery.as_deref(),
                OperationKind::TextTurn,
                state.request_id.as_deref(),
                self.estimate,
            )
            .await
        };
        let cumulative_usage = cumulative_usage.saturating_add(usage);
        let finish_reason = synthetic
            .finish_reason
            .or_else(|| state.finish_reason.clone())
            .unwrap_or_else(|| {
                if !synthetic.tool_calls.is_empty()
                    || !synthetic.recoverable_tool_call_issues.is_empty()
                {
                    lutum_protocol::FinishReason::ToolCall
                } else {
                    lutum_protocol::FinishReason::Stop
                }
            });
        let committed_turn = Arc::new(AssistantTurnView::from_items(
            synthetic.assistant_turn.items(),
        )) as CommittedTurn;
        let turn = UncommittedAssistantTurn::new(synthetic.assistant_turn, committed_turn);
        Ok((
            StagedTextTurnOutcomeWithTools::Turn(StagedTextTurnResultWithTools {
                request_id: state.request_id.clone(),
                model: state.model.clone(),
                turn,
                tool_calls: synthetic.tool_calls,
                recoverable_tool_call_issues: synthetic.recoverable_tool_call_issues,
                continue_suggestion: synthetic.continue_suggestion,
                finish_reason,
                usage,
                cumulative_usage,
            }),
            cumulative_usage,
        ))
    }

    async fn try_controlled_execution_error<H>(
        &mut self,
        handler: &mut H,
        source: &AgentError,
        cumulative_usage: Usage,
        next_cumulative_usage: Usage,
        partial: &TextTurnStateWithTools<T>,
    ) -> Result<
        Option<StagedTextTurnOutcomeWithTools<T>>,
        CollectError<H::Error, TextTurnReductionError, TextTurnStateWithTools<T>>,
    >
    where
        H: TextToolEventHandler<T>,
    {
        match self
            .call_controlled_error_handler(handler, TextToolCollectError::Execution(source))
            .await
        {
            Ok(TextToolErrorDirective::Propagate) => {
                if let Err(finalize_source) = finalize_budget_cumulative(
                    &mut self.owned_lease,
                    &self.span,
                    partial.request_id.as_deref(),
                    next_cumulative_usage,
                ) {
                    emit_raw_collect_error(
                        self.extensions.as_ref(),
                        OperationKind::TextTurn,
                        partial.request_id.as_deref(),
                        CollectErrorKind::Execution,
                        summarize_text_state_with_tools(partial),
                        finalize_source.to_string(),
                    );
                    return Err(CollectError::Execution {
                        source: finalize_source,
                        partial: partial.clone(),
                    });
                }
                Ok(None)
            }
            Ok(TextToolErrorDirective::Return(synthetic)) => self
                .finish_controlled_synthetic(synthetic, cumulative_usage)
                .await
                .map(Some),
            Err(handler_source) => {
                if let Err(finalize_source) = finalize_budget_cumulative(
                    &mut self.owned_lease,
                    &self.span,
                    partial.request_id.as_deref(),
                    next_cumulative_usage,
                ) {
                    emit_raw_collect_error(
                        self.extensions.as_ref(),
                        OperationKind::TextTurn,
                        partial.request_id.as_deref(),
                        CollectErrorKind::Execution,
                        summarize_text_state_with_tools(partial),
                        finalize_source.to_string(),
                    );
                    return Err(CollectError::Execution {
                        source: finalize_source,
                        partial: partial.clone(),
                    });
                }
                emit_raw_collect_error(
                    self.extensions.as_ref(),
                    OperationKind::TextTurn,
                    partial.request_id.as_deref(),
                    CollectErrorKind::Handler,
                    summarize_text_state_with_tools(partial),
                    format!(
                        "handler error type={}",
                        std::any::type_name_of_val(&handler_source)
                    ),
                );
                Err(CollectError::Handler {
                    source: handler_source,
                    partial: self.reducer.state().clone(),
                })
            }
        }
    }

    async fn try_controlled_reduction_error<H>(
        &mut self,
        handler: &mut H,
        source: &TextTurnReductionError,
        cumulative_usage: Usage,
        next_cumulative_usage: Usage,
        partial: &TextTurnStateWithTools<T>,
    ) -> Result<
        Option<StagedTextTurnOutcomeWithTools<T>>,
        CollectError<H::Error, TextTurnReductionError, TextTurnStateWithTools<T>>,
    >
    where
        H: TextToolEventHandler<T>,
    {
        match self
            .call_controlled_error_handler(handler, TextToolCollectError::Reduction(source))
            .await
        {
            Ok(TextToolErrorDirective::Propagate) => {
                if let Err(finalize_source) = finalize_budget_cumulative(
                    &mut self.owned_lease,
                    &self.span,
                    partial.request_id.as_deref(),
                    next_cumulative_usage,
                ) {
                    emit_raw_collect_error(
                        self.extensions.as_ref(),
                        OperationKind::TextTurn,
                        partial.request_id.as_deref(),
                        CollectErrorKind::Execution,
                        summarize_text_state_with_tools(partial),
                        finalize_source.to_string(),
                    );
                    return Err(CollectError::Execution {
                        source: finalize_source,
                        partial: partial.clone(),
                    });
                }
                Ok(None)
            }
            Ok(TextToolErrorDirective::Return(synthetic)) => self
                .finish_controlled_synthetic(synthetic, cumulative_usage)
                .await
                .map(Some),
            Err(handler_source) => {
                if let Err(finalize_source) = finalize_budget_cumulative(
                    &mut self.owned_lease,
                    &self.span,
                    partial.request_id.as_deref(),
                    next_cumulative_usage,
                ) {
                    emit_raw_collect_error(
                        self.extensions.as_ref(),
                        OperationKind::TextTurn,
                        partial.request_id.as_deref(),
                        CollectErrorKind::Execution,
                        summarize_text_state_with_tools(partial),
                        finalize_source.to_string(),
                    );
                    return Err(CollectError::Execution {
                        source: finalize_source,
                        partial: partial.clone(),
                    });
                }
                emit_raw_collect_error(
                    self.extensions.as_ref(),
                    OperationKind::TextTurn,
                    partial.request_id.as_deref(),
                    CollectErrorKind::Handler,
                    summarize_text_state_with_tools(partial),
                    format!(
                        "handler error type={}",
                        std::any::type_name_of_val(&handler_source)
                    ),
                );
                Err(CollectError::Handler {
                    source: handler_source,
                    partial: self.reducer.state().clone(),
                })
            }
        }
    }

    async fn try_controlled_unexpected_eof<H>(
        &mut self,
        handler: &mut H,
        cumulative_usage: Usage,
        next_cumulative_usage: Usage,
        partial: &TextTurnStateWithTools<T>,
    ) -> Result<
        Option<StagedTextTurnOutcomeWithTools<T>>,
        CollectError<H::Error, TextTurnReductionError, TextTurnStateWithTools<T>>,
    >
    where
        H: TextToolEventHandler<T>,
    {
        match self
            .call_controlled_error_handler(handler, TextToolCollectError::UnexpectedEof)
            .await
        {
            Ok(TextToolErrorDirective::Propagate) => {
                if let Err(finalize_source) = finalize_budget_cumulative(
                    &mut self.owned_lease,
                    &self.span,
                    partial.request_id.as_deref(),
                    next_cumulative_usage,
                ) {
                    emit_raw_collect_error(
                        self.extensions.as_ref(),
                        OperationKind::TextTurn,
                        partial.request_id.as_deref(),
                        CollectErrorKind::Execution,
                        summarize_text_state_with_tools(partial),
                        finalize_source.to_string(),
                    );
                    return Err(CollectError::Execution {
                        source: finalize_source,
                        partial: partial.clone(),
                    });
                }
                Ok(None)
            }
            Ok(TextToolErrorDirective::Return(synthetic)) => self
                .finish_controlled_synthetic(synthetic, cumulative_usage)
                .await
                .map(Some),
            Err(handler_source) => {
                if let Err(finalize_source) = finalize_budget_cumulative(
                    &mut self.owned_lease,
                    &self.span,
                    partial.request_id.as_deref(),
                    next_cumulative_usage,
                ) {
                    emit_raw_collect_error(
                        self.extensions.as_ref(),
                        OperationKind::TextTurn,
                        partial.request_id.as_deref(),
                        CollectErrorKind::Execution,
                        summarize_text_state_with_tools(partial),
                        finalize_source.to_string(),
                    );
                    return Err(CollectError::Execution {
                        source: finalize_source,
                        partial: partial.clone(),
                    });
                }
                emit_raw_collect_error(
                    self.extensions.as_ref(),
                    OperationKind::TextTurn,
                    partial.request_id.as_deref(),
                    CollectErrorKind::Handler,
                    summarize_text_state_with_tools(partial),
                    format!(
                        "handler error type={}",
                        std::any::type_name_of_val(&handler_source)
                    ),
                );
                Err(CollectError::Handler {
                    source: handler_source,
                    partial: self.reducer.state().clone(),
                })
            }
        }
    }
}

fn recover_required_text_tool_outcome<T>(
    outcome: StagedTextTurnOutcomeWithTools<T>,
    partial: &TextTurnStateWithTools<T>,
    fallback_parser: Option<&dyn TextToolCallFallbackParser<T>>,
    constraints: &ToolConstraints<T>,
    tool_definitions: &[AdapterToolDefinition],
    extensions: &RequestExtensions,
) -> Result<StagedTextTurnOutcomeWithTools<T>, TextTurnReductionError>
where
    T: Toolset,
{
    if matches!(constraints.requirement, ToolRequirement::Optional) {
        return Ok(outcome);
    }

    match outcome {
        StagedTextTurnOutcomeWithTools::FinishedNoOutput(result) => {
            Err(unmet_tool_requirement_error::<T>(
                result.model,
                result.request_id,
                &constraints.requirement,
                partial.event_count,
            ))
        }
        StagedTextTurnOutcomeWithTools::Turn(staged)
            if !staged.tool_calls.is_empty() || !staged.recoverable_tool_call_issues.is_empty() =>
        {
            Ok(StagedTextTurnOutcomeWithTools::Turn(staged))
        }
        StagedTextTurnOutcomeWithTools::Turn(staged) => recover_required_text_tool_turn(
            staged,
            partial,
            fallback_parser,
            constraints,
            tool_definitions,
            extensions,
        )
        .map(StagedTextTurnOutcomeWithTools::Turn),
    }
}

fn recover_required_text_tool_turn<T>(
    staged: StagedTextTurnResultWithTools<T>,
    partial: &TextTurnStateWithTools<T>,
    fallback_parser: Option<&dyn TextToolCallFallbackParser<T>>,
    constraints: &ToolConstraints<T>,
    tool_definitions: &[AdapterToolDefinition],
    extensions: &RequestExtensions,
) -> Result<StagedTextTurnResultWithTools<T>, TextTurnReductionError>
where
    T: Toolset,
{
    let Some(fallback_parser) = fallback_parser else {
        return Err(unmet_tool_requirement_error::<T>(
            staged.model,
            staged.request_id,
            &constraints.requirement,
            partial.event_count,
        ));
    };

    let cx = TextToolCallFallbackContext {
        assistant_turn: staged.turn.assistant_turn(),
        constraints,
        tool_definitions,
        requirement: &constraints.requirement,
        request_id: staged.request_id.as_deref(),
        model: staged.model.as_str(),
        finish_reason: staged.finish_reason.clone(),
        usage: staged.usage,
        event_count: partial.event_count,
        extensions,
    };
    let recovered = fallback_parser
        .parse_fallback_tool_calls(&cx)
        .map_err(|source| tool_fallback_error(&staged, partial.event_count, source))?
        .ok_or_else(|| {
            tool_fallback_error(
                &staged,
                partial.event_count,
                ToolCallFallbackError::NoToolCall,
            )
        })?;
    let recovered =
        validate_recovered_text_tool_calls(recovered, &constraints.requirement, tool_definitions)
            .map_err(|source| tool_fallback_error(&staged, partial.event_count, source))?;

    let committed_turn = Arc::new(AssistantTurnView::from_items(
        recovered.assistant_turn.items(),
    )) as CommittedTurn;
    let turn = UncommittedAssistantTurn::new(recovered.assistant_turn, committed_turn);

    Ok(StagedTextTurnResultWithTools {
        request_id: staged.request_id,
        model: staged.model,
        turn,
        tool_calls: recovered.tool_calls,
        recoverable_tool_call_issues: staged.recoverable_tool_call_issues,
        continue_suggestion: staged.continue_suggestion,
        finish_reason: staged.finish_reason,
        usage: staged.usage,
        cumulative_usage: staged.cumulative_usage,
    })
}

fn validate_recovered_text_tool_calls<T>(
    recovered: RecoveredTextToolCalls<T>,
    requirement: &ToolRequirement<T::Selector>,
    tool_definitions: &[AdapterToolDefinition],
) -> Result<RecoveredTextToolCalls<T>, ToolCallFallbackError>
where
    T: Toolset,
{
    let assistant_tool_calls = recovered
        .assistant_turn
        .items()
        .iter()
        .filter_map(assistant_item_tool_metadata)
        .collect::<Vec<_>>();

    if assistant_tool_calls.is_empty() {
        return Err(ToolCallFallbackError::NoToolCall);
    }
    if assistant_tool_calls.len() != recovered.tool_calls.len() {
        return Err(ToolCallFallbackError::ToolCallCountMismatch {
            assistant_tool_calls: assistant_tool_calls.len(),
            tool_calls: recovered.tool_calls.len(),
        });
    }

    for (index, (metadata, tool_call)) in assistant_tool_calls
        .iter()
        .zip(recovered.tool_calls.iter())
        .enumerate()
    {
        if !tool_definitions
            .iter()
            .any(|definition| definition.name == metadata.name.as_str())
        {
            return Err(ToolCallFallbackError::UnavailableTool {
                name: metadata.name.as_str().to_string(),
            });
        }

        if let ToolRequirement::Specific(selector) = requirement {
            let expected = selector.name();
            if metadata.name.as_str() != expected {
                return Err(ToolCallFallbackError::WrongRequiredTool {
                    expected: expected.to_string(),
                    actual: metadata.name.as_str().to_string(),
                });
            }
        }

        let typed_metadata = tool_call.metadata();
        if typed_metadata.id != metadata.id
            || typed_metadata.name != metadata.name
            || typed_metadata.arguments != metadata.arguments
        {
            return Err(ToolCallFallbackError::MismatchedTypedCall { index });
        }
    }

    Ok(recovered)
}

fn parse_validated_tool_call<T>(
    metadata: ToolMetadata,
    requirement: &ToolRequirement<T::Selector>,
    tool_definitions: &[AdapterToolDefinition],
) -> Result<T::ToolCall, ToolCallFallbackError>
where
    T: Toolset,
{
    let name = metadata.name.as_str().to_string();
    if !tool_definitions
        .iter()
        .any(|definition| definition.name == metadata.name.as_str())
    {
        return Err(ToolCallFallbackError::UnavailableTool { name });
    }

    if let ToolRequirement::Specific(selector) = requirement {
        let expected = selector.name();
        if metadata.name.as_str() != expected {
            return Err(ToolCallFallbackError::WrongRequiredTool {
                expected: expected.to_string(),
                actual: metadata.name.as_str().to_string(),
            });
        }
    }

    T::parse_tool_call(metadata).map_err(|source| ToolCallFallbackError::ToolCallParse {
        name,
        message: source.to_string(),
    })
}

fn assistant_item_tool_metadata(item: &AssistantTurnItem) -> Option<ToolMetadata> {
    let AssistantTurnItem::ToolCall {
        id,
        name,
        arguments,
    } = item
    else {
        return None;
    };
    Some(ToolMetadata::new(
        id.clone(),
        name.clone(),
        arguments.clone(),
    ))
}

fn unmet_tool_requirement_error<T>(
    model: String,
    request_id: Option<String>,
    requirement: &ToolRequirement<T::Selector>,
    event_count: u32,
) -> TextTurnReductionError
where
    T: Toolset,
{
    TextTurnReductionError::UnmetToolRequirement {
        model,
        request_id,
        requirement: tool_requirement_label::<T>(requirement),
        event_count,
    }
}

fn tool_fallback_error<T>(
    staged: &StagedTextTurnResultWithTools<T>,
    event_count: u32,
    source: ToolCallFallbackError,
) -> TextTurnReductionError
where
    T: Toolset,
{
    TextTurnReductionError::ToolCallFallback {
        model: staged.model.clone(),
        request_id: staged.request_id.clone(),
        event_count,
        source,
    }
}

fn tool_fallback_error_from_state<T>(
    state: &TextTurnStateWithTools<T>,
    source: ToolCallFallbackError,
) -> TextTurnReductionError
where
    T: Toolset,
{
    TextTurnReductionError::ToolCallFallback {
        model: state.model.clone(),
        request_id: state.request_id.clone(),
        event_count: state.event_count,
        source,
    }
}

fn tool_requirement_label<T>(requirement: &ToolRequirement<T::Selector>) -> String
where
    T: Toolset,
{
    match requirement {
        ToolRequirement::Optional => "optional".to_string(),
        ToolRequirement::AtLeastOne => "at_least_one".to_string(),
        ToolRequirement::Specific(selector) => format!("specific:{}", selector.name()),
    }
}

impl<O> PendingStructuredTurn<O>
where
    O: StructuredOutput,
{
    async fn start_attempt(&self) -> Result<StructuredTurnEventStream<O>, AgentError> {
        let stream = self
            .turns
            .structured_turn(self.input.clone(), self.turn.clone())
            .await?;
        let stream = observe_structured_stream(
            stream,
            Arc::clone(&self.hooks),
            Arc::clone(&self.extensions),
        );
        Ok(map_structured_stream::<O>(stream))
    }

    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> StructuredTurnEventStream<O> {
        let Self {
            recovery,
            turns,
            hooks,
            extensions,
            input,
            turn,
            estimate,
            retry_policy,
            span,
            ..
        } = self;
        boxed_sync_stream(try_stream! {
            let mut attempt = 1_u32;
            let mut cumulative_usage = Usage::zero();

            'attempts: loop {
                let stream = turns.structured_turn(input.clone(), turn.clone()).await;
                let mut stream = match stream {
                    Ok(stream) => map_structured_stream::<O>(observe_structured_stream(
                        stream,
                        Arc::clone(&hooks),
                        Arc::clone(&extensions),
                    )),
                    Err(source) => {
                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&retry_policy, attempt, &source)
                        {
                            let accounted_usage =
                                recover_or_estimate_usage(recovery.as_deref(), OperationKind::StructuredTurn, None, estimate).await;
                            cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
                            yield StructuredTurnEvent::WillRetry {
                                attempt: next_attempt,
                                after,
                                kind,
                                status,
                                request_id: None,
                                accounted_usage,
                                cumulative_usage,
                            };
                            tokio::time::sleep(after).await;
                            attempt = next_attempt;
                            continue 'attempts;
                        }
                        Err(source)?;
                        break;
                    }
                };

                let mut request_id = None;
                while let Some(item) = stream.next().instrument(span.clone()).await {
                    match item {
                        Ok(event) => {
                            match &event {
                                StructuredTurnEvent::Started { request_id: event_request_id, .. } => {
                                    request_id = event_request_id.clone();
                                }
                                StructuredTurnEvent::Completed { request_id: event_request_id, .. } => {
                                    if let Some(event_request_id) = event_request_id.clone() {
                                        request_id = Some(event_request_id);
                                    }
                                }
                                _ => {}
                            }
                            yield event;
                        }
                        Err(source) => {
                            if let Some((next_attempt, after, status, kind)) =
                                maybe_retry_plan(&retry_policy, attempt, &source)
                            {
                                let accounted_usage = recover_or_estimate_usage(
                                    recovery.as_deref(),
                                    OperationKind::StructuredTurn,
                                    request_id.as_deref(),
                                    estimate,
                                )
                                .await;
                                cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
                                yield StructuredTurnEvent::WillRetry {
                                    attempt: next_attempt,
                                    after,
                                    kind,
                                    status,
                                    request_id,
                                    accounted_usage,
                                    cumulative_usage,
                                };
                                tokio::time::sleep(after).await;
                                attempt = next_attempt;
                                continue 'attempts;
                            }
                            Err(source)?;
                            break 'attempts;
                        }
                    }
                }

                break;
            }
        })
    }

    pub async fn collect_with<H>(
        mut self,
        mut handler: H,
    ) -> Result<
        StagedStructuredTurnResult<O>,
        CollectError<H::Error, StructuredTurnReductionError, StructuredTurnPartial<O>>,
    >
    where
        H: EventHandler<StructuredTurnEvent<O>, StructuredTurnState<O>>,
    {
        let mut attempt = 1_u32;
        let mut cumulative_usage = Usage::zero();

        'attempts: loop {
            let mut stream = match self.start_attempt().await {
                Ok(stream) => stream,
                Err(source) => {
                    let partial = StructuredTurnPartial::from_state(self.reducer.state().clone());
                    let accounted_usage = recover_or_estimate_usage(
                        self.recovery.as_deref(),
                        OperationKind::StructuredTurn,
                        self.reducer.state().request_id.as_deref(),
                        self.estimate,
                    )
                    .await;
                    let next_cumulative_usage = cumulative_usage.saturating_add(accounted_usage);

                    if let Some((next_attempt, after, status, kind)) =
                        maybe_retry_plan(&self.retry_policy, attempt, &source)
                    {
                        let retry_event = StructuredTurnEvent::WillRetry {
                            attempt: next_attempt,
                            after,
                            kind,
                            status,
                            request_id: partial.state.request_id.clone(),
                            accounted_usage,
                            cumulative_usage: next_cumulative_usage,
                        };
                        match self.call_handler(&mut handler, &retry_event).await {
                            Ok(HandlerDirective::Continue) => {
                                cumulative_usage = next_cumulative_usage;
                                self.reducer.reset_for_retry();
                                tokio::time::sleep(after).await;
                                attempt = next_attempt;
                                continue 'attempts;
                            }
                            Ok(HandlerDirective::Stop) => {
                                if let Err(finalize_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    partial.state.request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::StructuredTurn,
                                        partial.state.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_partial(&partial),
                                        finalize_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: finalize_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::StructuredTurn,
                                    partial.state.request_id.as_deref(),
                                    CollectErrorKind::Execution,
                                    summarize_structured_partial(&partial),
                                    source.to_string(),
                                );
                                return Err(CollectError::Execution { source, partial });
                            }
                            Err(handler_source) => {
                                if let Err(finalize_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    partial.state.request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::StructuredTurn,
                                        partial.state.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_partial(&partial),
                                        finalize_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: finalize_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::StructuredTurn,
                                    partial.state.request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_structured_partial(&partial),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&handler_source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source: handler_source,
                                    partial: StructuredTurnPartial::from_state(
                                        self.reducer.into_state(),
                                    ),
                                });
                            }
                        }
                    }

                    if let Err(finalize_source) = finalize_budget_cumulative(
                        &mut self.owned_lease,
                        &self.span,
                        partial.state.request_id.as_deref(),
                        next_cumulative_usage,
                    ) {
                        emit_raw_collect_error(
                            self.extensions.as_ref(),
                            OperationKind::StructuredTurn,
                            partial.state.request_id.as_deref(),
                            CollectErrorKind::Execution,
                            summarize_structured_partial(&partial),
                            finalize_source.to_string(),
                        );
                        return Err(CollectError::Execution {
                            source: finalize_source,
                            partial,
                        });
                    }
                    emit_raw_collect_error(
                        self.extensions.as_ref(),
                        OperationKind::StructuredTurn,
                        partial.state.request_id.as_deref(),
                        CollectErrorKind::Execution,
                        summarize_structured_partial(&partial),
                        source.to_string(),
                    );
                    return Err(CollectError::Execution {
                        source,
                        partial: StructuredTurnPartial::from_state(self.reducer.into_state()),
                    });
                }
            };

            while let Some(item) = stream.next().instrument(self.span.clone()).await {
                match item {
                    Ok(event) => {
                        if let Err(source) = self.reducer.apply(&event) {
                            let partial =
                                StructuredTurnPartial::from_state(self.reducer.state().clone());
                            emit_raw_collect_error(
                                self.extensions.as_ref(),
                                OperationKind::StructuredTurn,
                                partial.state.request_id.as_deref(),
                                CollectErrorKind::Reduction,
                                summarize_structured_partial(&partial),
                                source.to_string(),
                            );
                            return Err(CollectError::Reduction { source, partial });
                        }
                        record_request_id(&self.span, self.reducer.state().request_id.as_deref());
                        if let Some(usage) = completed_usage_from_structured(&event) {
                            let next_cumulative_usage = cumulative_usage.saturating_add(usage);
                            if let Err(source) = finalize_budget_cumulative(
                                &mut self.owned_lease,
                                &self.span,
                                self.reducer.state().request_id.as_deref(),
                                next_cumulative_usage,
                            ) {
                                let partial =
                                    StructuredTurnPartial::from_state(self.reducer.state().clone());
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::StructuredTurn,
                                    partial.state.request_id.as_deref(),
                                    CollectErrorKind::Execution,
                                    summarize_structured_partial(&partial),
                                    source.to_string(),
                                );
                                return Err(CollectError::Execution { source, partial });
                            }
                            if let Err(source) = self.call_handler(&mut handler, &event).await {
                                let partial =
                                    StructuredTurnPartial::from_state(self.reducer.state().clone());
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::StructuredTurn,
                                    partial.state.request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_structured_partial(&partial),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&source)
                                    ),
                                );
                                return Err(CollectError::Handler { source, partial });
                            }
                            let partial =
                                StructuredTurnPartial::from_state(self.reducer.state().clone());
                            return match self.reducer.into_result() {
                                Ok(mut result) => {
                                    result.cumulative_usage = next_cumulative_usage;
                                    Ok(result)
                                }
                                Err((source, committed_turn)) => {
                                    let partial = if let Some(committed_turn) = committed_turn {
                                        partial.with_committed_turn(committed_turn)
                                    } else {
                                        partial
                                    };
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::StructuredTurn,
                                        partial.state.request_id.as_deref(),
                                        CollectErrorKind::Reduction,
                                        summarize_structured_partial(&partial),
                                        source.to_string(),
                                    );
                                    Err(CollectError::Reduction { source, partial })
                                }
                            };
                        }

                        match self.call_handler(&mut handler, &event).await {
                            Ok(HandlerDirective::Continue) => {}
                            Ok(HandlerDirective::Stop) => {
                                let partial =
                                    StructuredTurnPartial::from_state(self.reducer.state().clone());
                                let accounted_usage = recover_or_estimate_usage(
                                    self.recovery.as_deref(),
                                    OperationKind::StructuredTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    self.estimate,
                                )
                                .await;
                                let next_cumulative_usage =
                                    cumulative_usage.saturating_add(accounted_usage);
                                if let Err(source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    self.reducer.state().request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::StructuredTurn,
                                        partial.state.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_partial(&partial),
                                        source.to_string(),
                                    );
                                    return Err(CollectError::Execution { source, partial });
                                }
                                return Err(CollectError::Stopped {
                                    partial: StructuredTurnPartial::from_state(
                                        self.reducer.into_state(),
                                    ),
                                });
                            }
                            Err(source) => {
                                let partial =
                                    StructuredTurnPartial::from_state(self.reducer.state().clone());
                                let accounted_usage = recover_or_estimate_usage(
                                    self.recovery.as_deref(),
                                    OperationKind::StructuredTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    self.estimate,
                                )
                                .await;
                                let next_cumulative_usage =
                                    cumulative_usage.saturating_add(accounted_usage);
                                if let Err(execution_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    self.reducer.state().request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::StructuredTurn,
                                        partial.state.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_partial(&partial),
                                        execution_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: execution_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::StructuredTurn,
                                    partial.state.request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_structured_partial(&partial),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source,
                                    partial: StructuredTurnPartial::from_state(
                                        self.reducer.into_state(),
                                    ),
                                });
                            }
                        }
                    }
                    Err(source) => {
                        let partial =
                            StructuredTurnPartial::from_state(self.reducer.state().clone());
                        let accounted_usage = recover_or_estimate_usage(
                            self.recovery.as_deref(),
                            OperationKind::StructuredTurn,
                            self.reducer.state().request_id.as_deref(),
                            self.estimate,
                        )
                        .await;
                        let next_cumulative_usage =
                            cumulative_usage.saturating_add(accounted_usage);

                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&self.retry_policy, attempt, &source)
                        {
                            let retry_event = StructuredTurnEvent::WillRetry {
                                attempt: next_attempt,
                                after,
                                kind,
                                status,
                                request_id: partial.state.request_id.clone(),
                                accounted_usage,
                                cumulative_usage: next_cumulative_usage,
                            };
                            match self.call_handler(&mut handler, &retry_event).await {
                                Ok(HandlerDirective::Continue) => {
                                    cumulative_usage = next_cumulative_usage;
                                    self.reducer.reset_for_retry();
                                    tokio::time::sleep(after).await;
                                    attempt = next_attempt;
                                    continue 'attempts;
                                }
                                Ok(HandlerDirective::Stop) => {
                                    if let Err(finalize_source) = finalize_budget_cumulative(
                                        &mut self.owned_lease,
                                        &self.span,
                                        partial.state.request_id.as_deref(),
                                        next_cumulative_usage,
                                    ) {
                                        emit_raw_collect_error(
                                            self.extensions.as_ref(),
                                            OperationKind::StructuredTurn,
                                            partial.state.request_id.as_deref(),
                                            CollectErrorKind::Execution,
                                            summarize_structured_partial(&partial),
                                            finalize_source.to_string(),
                                        );
                                        return Err(CollectError::Execution {
                                            source: finalize_source,
                                            partial,
                                        });
                                    }
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::StructuredTurn,
                                        partial.state.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_partial(&partial),
                                        source.to_string(),
                                    );
                                    return Err(CollectError::Execution { source, partial });
                                }
                                Err(handler_source) => {
                                    if let Err(finalize_source) = finalize_budget_cumulative(
                                        &mut self.owned_lease,
                                        &self.span,
                                        partial.state.request_id.as_deref(),
                                        next_cumulative_usage,
                                    ) {
                                        emit_raw_collect_error(
                                            self.extensions.as_ref(),
                                            OperationKind::StructuredTurn,
                                            partial.state.request_id.as_deref(),
                                            CollectErrorKind::Execution,
                                            summarize_structured_partial(&partial),
                                            finalize_source.to_string(),
                                        );
                                        return Err(CollectError::Execution {
                                            source: finalize_source,
                                            partial,
                                        });
                                    }
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::StructuredTurn,
                                        partial.state.request_id.as_deref(),
                                        CollectErrorKind::Handler,
                                        summarize_structured_partial(&partial),
                                        format!(
                                            "handler error type={}",
                                            std::any::type_name_of_val(&handler_source)
                                        ),
                                    );
                                    return Err(CollectError::Handler {
                                        source: handler_source,
                                        partial: StructuredTurnPartial::from_state(
                                            self.reducer.into_state(),
                                        ),
                                    });
                                }
                            }
                        }

                        if let Err(execution_source) = finalize_budget_cumulative(
                            &mut self.owned_lease,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            next_cumulative_usage,
                        ) {
                            emit_raw_collect_error(
                                self.extensions.as_ref(),
                                OperationKind::StructuredTurn,
                                partial.state.request_id.as_deref(),
                                CollectErrorKind::Execution,
                                summarize_structured_partial(&partial),
                                execution_source.to_string(),
                            );
                            return Err(CollectError::Execution {
                                source: execution_source,
                                partial,
                            });
                        }
                        emit_raw_collect_error(
                            self.extensions.as_ref(),
                            OperationKind::StructuredTurn,
                            partial.state.request_id.as_deref(),
                            CollectErrorKind::Execution,
                            summarize_structured_partial(&partial),
                            source.to_string(),
                        );
                        return Err(CollectError::Execution {
                            source,
                            partial: StructuredTurnPartial::from_state(self.reducer.into_state()),
                        });
                    }
                }
            }

            let partial = StructuredTurnPartial::from_state(self.reducer.state().clone());
            let accounted_usage = recover_or_estimate_usage(
                self.recovery.as_deref(),
                OperationKind::StructuredTurn,
                self.reducer.state().request_id.as_deref(),
                self.estimate,
            )
            .await;
            let next_cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
            if let Err(source) = finalize_budget_cumulative(
                &mut self.owned_lease,
                &self.span,
                self.reducer.state().request_id.as_deref(),
                next_cumulative_usage,
            ) {
                emit_raw_collect_error(
                    self.extensions.as_ref(),
                    OperationKind::StructuredTurn,
                    partial.state.request_id.as_deref(),
                    CollectErrorKind::Execution,
                    summarize_structured_partial(&partial),
                    source.to_string(),
                );
                return Err(CollectError::Execution { source, partial });
            }
            emit_raw_collect_error(
                self.extensions.as_ref(),
                OperationKind::StructuredTurn,
                partial.state.request_id.as_deref(),
                CollectErrorKind::UnexpectedEof,
                summarize_structured_partial(&partial),
                "stream ended before completion".to_string(),
            );
            return Err(CollectError::UnexpectedEof {
                partial: StructuredTurnPartial::from_state(self.reducer.into_state()),
            });
        }
    }

    pub async fn collect(
        self,
    ) -> Result<
        StagedStructuredTurnResult<O>,
        CollectError<Infallible, StructuredTurnReductionError, StructuredTurnPartial<O>>,
    > {
        self.collect_with(NoopHandler).await
    }

    async fn call_handler<H>(
        &self,
        handler: &mut H,
        event: &StructuredTurnEvent<O>,
    ) -> Result<HandlerDirective, H::Error>
    where
        H: EventHandler<StructuredTurnEvent<O>, StructuredTurnState<O>>,
    {
        let cx = HandlerContext {
            extensions: self.extensions.as_ref(),
            state: self.reducer.state(),
            remaining_budget: self.owned_lease.budget.remaining(self.extensions.as_ref()),
        };
        handler.on_event(event, &cx).await
    }
}

impl<T, O> PendingStructuredTurnWithTools<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    async fn start_attempt(&self) -> Result<StructuredTurnEventStreamWithTools<T, O>, AgentError> {
        let stream = self
            .turns
            .structured_turn(self.input.clone(), self.turn.clone())
            .await?;
        let stream = observe_structured_stream(
            stream,
            Arc::clone(&self.hooks),
            Arc::clone(&self.extensions),
        );
        Ok(map_structured_stream_with_tools::<T, O>(
            stream,
            self.availability.clone(),
            self.dynamic_names.clone(),
        ))
    }

    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> StructuredTurnEventStreamWithTools<T, O> {
        let Self {
            recovery,
            turns,
            hooks,
            extensions,
            input,
            turn,
            availability,
            dynamic_names,
            estimate,
            retry_policy,
            span,
            ..
        } = self;
        boxed_sync_stream(try_stream! {
            let mut attempt = 1_u32;
            let mut cumulative_usage = Usage::zero();

            'attempts: loop {
                let stream = turns.structured_turn(input.clone(), turn.clone()).await;
                let mut stream = match stream {
                    Ok(stream) => map_structured_stream_with_tools::<T, O>(
                        observe_structured_stream(
                            stream,
                            Arc::clone(&hooks),
                            Arc::clone(&extensions),
                        ),
                        availability.clone(),
                        dynamic_names.clone(),
                    ),
                    Err(source) => {
                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&retry_policy, attempt, &source)
                        {
                            let accounted_usage =
                                recover_or_estimate_usage(recovery.as_deref(), OperationKind::StructuredTurn, None, estimate).await;
                            cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
                            yield StructuredTurnEventWithTools::WillRetry {
                                attempt: next_attempt,
                                after,
                                kind,
                                status,
                                request_id: None,
                                accounted_usage,
                                cumulative_usage,
                            };
                            tokio::time::sleep(after).await;
                            attempt = next_attempt;
                            continue 'attempts;
                        }
                        Err(source)?;
                        break;
                    }
                };

                let mut request_id = None;
                while let Some(item) = stream.next().instrument(span.clone()).await {
                    match item {
                        Ok(event) => {
                            match &event {
                                StructuredTurnEventWithTools::Started {
                                    request_id: event_request_id,
                                    ..
                                } => request_id = event_request_id.clone(),
                                StructuredTurnEventWithTools::Completed {
                                    request_id: event_request_id,
                                    ..
                                } => {
                                    if let Some(event_request_id) = event_request_id.clone() {
                                        request_id = Some(event_request_id);
                                    }
                                }
                                _ => {}
                            }
                            yield event;
                        }
                        Err(source) => {
                            if let Some((next_attempt, after, status, kind)) =
                                maybe_retry_plan(&retry_policy, attempt, &source)
                            {
                                let accounted_usage = recover_or_estimate_usage(
                                    recovery.as_deref(),
                                    OperationKind::StructuredTurn,
                                    request_id.as_deref(),
                                    estimate,
                                )
                                .await;
                                cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
                                yield StructuredTurnEventWithTools::WillRetry {
                                    attempt: next_attempt,
                                    after,
                                    kind,
                                    status,
                                    request_id,
                                    accounted_usage,
                                    cumulative_usage,
                                };
                                tokio::time::sleep(after).await;
                                attempt = next_attempt;
                                continue 'attempts;
                            }
                            Err(source)?;
                            break 'attempts;
                        }
                    }
                }

                break;
            }
        })
    }

    pub async fn collect_with<H>(
        mut self,
        mut handler: H,
    ) -> Result<
        StagedStructuredTurnResultWithTools<T, O>,
        CollectError<H::Error, StructuredTurnReductionError, StructuredTurnPartialWithTools<T, O>>,
    >
    where
        H: EventHandler<StructuredTurnEventWithTools<T, O>, StructuredTurnStateWithTools<T, O>>,
    {
        let mut attempt = 1_u32;
        let mut cumulative_usage = Usage::zero();

        'attempts: loop {
            let mut stream = match self.start_attempt().await {
                Ok(stream) => stream,
                Err(source) => {
                    let partial =
                        StructuredTurnPartialWithTools::from_state(self.reducer.state().clone());
                    let accounted_usage = recover_or_estimate_usage(
                        self.recovery.as_deref(),
                        OperationKind::StructuredTurn,
                        self.reducer.state().request_id.as_deref(),
                        self.estimate,
                    )
                    .await;
                    let next_cumulative_usage = cumulative_usage.saturating_add(accounted_usage);

                    if let Some((next_attempt, after, status, kind)) =
                        maybe_retry_plan(&self.retry_policy, attempt, &source)
                    {
                        let retry_event = StructuredTurnEventWithTools::WillRetry {
                            attempt: next_attempt,
                            after,
                            kind,
                            status,
                            request_id: partial.state.request_id.clone(),
                            accounted_usage,
                            cumulative_usage: next_cumulative_usage,
                        };
                        match self.call_handler(&mut handler, &retry_event).await {
                            Ok(HandlerDirective::Continue) => {
                                cumulative_usage = next_cumulative_usage;
                                self.reducer.reset_for_retry();
                                tokio::time::sleep(after).await;
                                attempt = next_attempt;
                                continue 'attempts;
                            }
                            Ok(HandlerDirective::Stop) => {
                                if let Err(finalize_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    partial.state.request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::StructuredTurn,
                                        partial.state.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_partial_with_tools(&partial),
                                        finalize_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: finalize_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::StructuredTurn,
                                    partial.state.request_id.as_deref(),
                                    CollectErrorKind::Execution,
                                    summarize_structured_partial_with_tools(&partial),
                                    source.to_string(),
                                );
                                return Err(CollectError::Execution { source, partial });
                            }
                            Err(handler_source) => {
                                if let Err(finalize_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    partial.state.request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::StructuredTurn,
                                        partial.state.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_partial_with_tools(&partial),
                                        finalize_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: finalize_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::StructuredTurn,
                                    partial.state.request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_structured_partial_with_tools(&partial),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&handler_source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source: handler_source,
                                    partial: StructuredTurnPartialWithTools::from_state(
                                        self.reducer.into_state(),
                                    ),
                                });
                            }
                        }
                    }

                    if let Err(finalize_source) = finalize_budget_cumulative(
                        &mut self.owned_lease,
                        &self.span,
                        partial.state.request_id.as_deref(),
                        next_cumulative_usage,
                    ) {
                        emit_raw_collect_error(
                            self.extensions.as_ref(),
                            OperationKind::StructuredTurn,
                            partial.state.request_id.as_deref(),
                            CollectErrorKind::Execution,
                            summarize_structured_partial_with_tools(&partial),
                            finalize_source.to_string(),
                        );
                        return Err(CollectError::Execution {
                            source: finalize_source,
                            partial,
                        });
                    }
                    emit_raw_collect_error(
                        self.extensions.as_ref(),
                        OperationKind::StructuredTurn,
                        partial.state.request_id.as_deref(),
                        CollectErrorKind::Execution,
                        summarize_structured_partial_with_tools(&partial),
                        source.to_string(),
                    );
                    return Err(CollectError::Execution {
                        source,
                        partial: StructuredTurnPartialWithTools::from_state(
                            self.reducer.into_state(),
                        ),
                    });
                }
            };

            while let Some(item) = stream.next().instrument(self.span.clone()).await {
                match item {
                    Ok(event) => {
                        if let Err(source) = self.reducer.apply(&event) {
                            let partial = StructuredTurnPartialWithTools::from_state(
                                self.reducer.state().clone(),
                            );
                            emit_raw_collect_error(
                                self.extensions.as_ref(),
                                OperationKind::StructuredTurn,
                                partial.state.request_id.as_deref(),
                                CollectErrorKind::Reduction,
                                summarize_structured_partial_with_tools(&partial),
                                source.to_string(),
                            );
                            return Err(CollectError::Reduction { source, partial });
                        }
                        record_request_id(&self.span, self.reducer.state().request_id.as_deref());
                        if let Some(usage) = completed_usage_from_structured_with_tools(&event) {
                            let next_cumulative_usage = cumulative_usage.saturating_add(usage);
                            if let Err(source) = finalize_budget_cumulative(
                                &mut self.owned_lease,
                                &self.span,
                                self.reducer.state().request_id.as_deref(),
                                next_cumulative_usage,
                            ) {
                                let partial = StructuredTurnPartialWithTools::from_state(
                                    self.reducer.state().clone(),
                                );
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::StructuredTurn,
                                    partial.state.request_id.as_deref(),
                                    CollectErrorKind::Execution,
                                    summarize_structured_partial_with_tools(&partial),
                                    source.to_string(),
                                );
                                return Err(CollectError::Execution { source, partial });
                            }
                            if let Err(source) = self.call_handler(&mut handler, &event).await {
                                let partial = StructuredTurnPartialWithTools::from_state(
                                    self.reducer.state().clone(),
                                );
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::StructuredTurn,
                                    partial.state.request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_structured_partial_with_tools(&partial),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&source)
                                    ),
                                );
                                return Err(CollectError::Handler { source, partial });
                            }
                            let partial = StructuredTurnPartialWithTools::from_state(
                                self.reducer.state().clone(),
                            );
                            return match self.reducer.into_result() {
                                Ok(mut result) => {
                                    result.cumulative_usage = next_cumulative_usage;
                                    Ok(result)
                                }
                                Err((source, committed_turn)) => {
                                    let partial = if let Some(committed_turn) = committed_turn {
                                        partial.with_committed_turn(committed_turn)
                                    } else {
                                        partial
                                    };
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::StructuredTurn,
                                        partial.state.request_id.as_deref(),
                                        CollectErrorKind::Reduction,
                                        summarize_structured_partial_with_tools(&partial),
                                        source.to_string(),
                                    );
                                    Err(CollectError::Reduction { source, partial })
                                }
                            };
                        }

                        match self.call_handler(&mut handler, &event).await {
                            Ok(HandlerDirective::Continue) => {}
                            Ok(HandlerDirective::Stop) => {
                                let partial = StructuredTurnPartialWithTools::from_state(
                                    self.reducer.state().clone(),
                                );
                                let accounted_usage = recover_or_estimate_usage(
                                    self.recovery.as_deref(),
                                    OperationKind::StructuredTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    self.estimate,
                                )
                                .await;
                                let next_cumulative_usage =
                                    cumulative_usage.saturating_add(accounted_usage);
                                if let Err(source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    self.reducer.state().request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::StructuredTurn,
                                        partial.state.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_partial_with_tools(&partial),
                                        source.to_string(),
                                    );
                                    return Err(CollectError::Execution { source, partial });
                                }
                                return Err(CollectError::Stopped {
                                    partial: StructuredTurnPartialWithTools::from_state(
                                        self.reducer.into_state(),
                                    ),
                                });
                            }
                            Err(source) => {
                                let partial = StructuredTurnPartialWithTools::from_state(
                                    self.reducer.state().clone(),
                                );
                                let accounted_usage = recover_or_estimate_usage(
                                    self.recovery.as_deref(),
                                    OperationKind::StructuredTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    self.estimate,
                                )
                                .await;
                                let next_cumulative_usage =
                                    cumulative_usage.saturating_add(accounted_usage);
                                if let Err(execution_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    self.reducer.state().request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::StructuredTurn,
                                        partial.state.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_partial_with_tools(&partial),
                                        execution_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: execution_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    self.extensions.as_ref(),
                                    OperationKind::StructuredTurn,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_structured_partial_with_tools(&partial),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source,
                                    partial: StructuredTurnPartialWithTools::from_state(
                                        self.reducer.into_state(),
                                    ),
                                });
                            }
                        }
                    }
                    Err(source) => {
                        let partial = StructuredTurnPartialWithTools::from_state(
                            self.reducer.state().clone(),
                        );
                        let accounted_usage = recover_or_estimate_usage(
                            self.recovery.as_deref(),
                            OperationKind::StructuredTurn,
                            self.reducer.state().request_id.as_deref(),
                            self.estimate,
                        )
                        .await;
                        let next_cumulative_usage =
                            cumulative_usage.saturating_add(accounted_usage);

                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&self.retry_policy, attempt, &source)
                        {
                            let retry_event = StructuredTurnEventWithTools::WillRetry {
                                attempt: next_attempt,
                                after,
                                kind,
                                status,
                                request_id: partial.state.request_id.clone(),
                                accounted_usage,
                                cumulative_usage: next_cumulative_usage,
                            };
                            match self.call_handler(&mut handler, &retry_event).await {
                                Ok(HandlerDirective::Continue) => {
                                    cumulative_usage = next_cumulative_usage;
                                    self.reducer.reset_for_retry();
                                    tokio::time::sleep(after).await;
                                    attempt = next_attempt;
                                    continue 'attempts;
                                }
                                Ok(HandlerDirective::Stop) => {
                                    if let Err(finalize_source) = finalize_budget_cumulative(
                                        &mut self.owned_lease,
                                        &self.span,
                                        partial.state.request_id.as_deref(),
                                        next_cumulative_usage,
                                    ) {
                                        emit_raw_collect_error(
                                            self.extensions.as_ref(),
                                            OperationKind::StructuredTurn,
                                            partial.state.request_id.as_deref(),
                                            CollectErrorKind::Execution,
                                            summarize_structured_partial_with_tools(&partial),
                                            finalize_source.to_string(),
                                        );
                                        return Err(CollectError::Execution {
                                            source: finalize_source,
                                            partial,
                                        });
                                    }
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::StructuredTurn,
                                        partial.state.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_partial_with_tools(&partial),
                                        source.to_string(),
                                    );
                                    return Err(CollectError::Execution { source, partial });
                                }
                                Err(handler_source) => {
                                    if let Err(finalize_source) = finalize_budget_cumulative(
                                        &mut self.owned_lease,
                                        &self.span,
                                        partial.state.request_id.as_deref(),
                                        next_cumulative_usage,
                                    ) {
                                        emit_raw_collect_error(
                                            self.extensions.as_ref(),
                                            OperationKind::StructuredTurn,
                                            partial.state.request_id.as_deref(),
                                            CollectErrorKind::Execution,
                                            summarize_structured_partial_with_tools(&partial),
                                            finalize_source.to_string(),
                                        );
                                        return Err(CollectError::Execution {
                                            source: finalize_source,
                                            partial,
                                        });
                                    }
                                    emit_raw_collect_error(
                                        self.extensions.as_ref(),
                                        OperationKind::StructuredTurn,
                                        partial.state.request_id.as_deref(),
                                        CollectErrorKind::Handler,
                                        summarize_structured_partial_with_tools(&partial),
                                        format!(
                                            "handler error type={}",
                                            std::any::type_name_of_val(&handler_source)
                                        ),
                                    );
                                    return Err(CollectError::Handler {
                                        source: handler_source,
                                        partial: StructuredTurnPartialWithTools::from_state(
                                            self.reducer.into_state(),
                                        ),
                                    });
                                }
                            }
                        }

                        if let Err(execution_source) = finalize_budget_cumulative(
                            &mut self.owned_lease,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            next_cumulative_usage,
                        ) {
                            emit_raw_collect_error(
                                self.extensions.as_ref(),
                                OperationKind::StructuredTurn,
                                partial.state.request_id.as_deref(),
                                CollectErrorKind::Execution,
                                summarize_structured_partial_with_tools(&partial),
                                execution_source.to_string(),
                            );
                            return Err(CollectError::Execution {
                                source: execution_source,
                                partial,
                            });
                        }
                        emit_raw_collect_error(
                            self.extensions.as_ref(),
                            OperationKind::StructuredTurn,
                            partial.state.request_id.as_deref(),
                            CollectErrorKind::Execution,
                            summarize_structured_partial_with_tools(&partial),
                            source.to_string(),
                        );
                        return Err(CollectError::Execution {
                            source,
                            partial: StructuredTurnPartialWithTools::from_state(
                                self.reducer.into_state(),
                            ),
                        });
                    }
                }
            }

            let partial = StructuredTurnPartialWithTools::from_state(self.reducer.state().clone());
            let accounted_usage = recover_or_estimate_usage(
                self.recovery.as_deref(),
                OperationKind::StructuredTurn,
                self.reducer.state().request_id.as_deref(),
                self.estimate,
            )
            .await;
            let next_cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
            if let Err(source) = finalize_budget_cumulative(
                &mut self.owned_lease,
                &self.span,
                self.reducer.state().request_id.as_deref(),
                next_cumulative_usage,
            ) {
                emit_raw_collect_error(
                    self.extensions.as_ref(),
                    OperationKind::StructuredTurn,
                    partial.state.request_id.as_deref(),
                    CollectErrorKind::Execution,
                    summarize_structured_partial_with_tools(&partial),
                    source.to_string(),
                );
                return Err(CollectError::Execution { source, partial });
            }
            emit_raw_collect_error(
                self.extensions.as_ref(),
                OperationKind::StructuredTurn,
                partial.state.request_id.as_deref(),
                CollectErrorKind::UnexpectedEof,
                summarize_structured_partial_with_tools(&partial),
                "stream ended before completion".to_string(),
            );
            return Err(CollectError::UnexpectedEof {
                partial: StructuredTurnPartialWithTools::from_state(self.reducer.into_state()),
            });
        }
    }

    pub async fn collect(
        self,
    ) -> Result<
        StagedStructuredTurnResultWithTools<T, O>,
        CollectError<
            Infallible,
            StructuredTurnReductionError,
            StructuredTurnPartialWithTools<T, O>,
        >,
    > {
        self.collect_with(NoopHandler).await
    }

    async fn call_handler<H>(
        &self,
        handler: &mut H,
        event: &StructuredTurnEventWithTools<T, O>,
    ) -> Result<HandlerDirective, H::Error>
    where
        H: EventHandler<StructuredTurnEventWithTools<T, O>, StructuredTurnStateWithTools<T, O>>,
    {
        let cx = HandlerContext {
            extensions: self.extensions.as_ref(),
            state: self.reducer.state(),
            remaining_budget: self.owned_lease.budget.remaining(self.extensions.as_ref()),
        };
        handler.on_event(event, &cx).await
    }
}

impl PendingCompletion {
    async fn start_attempt(&self) -> Result<CompletionEventStream, AgentError> {
        let stream = self
            .completion
            .completion(self.request.clone(), self.extensions.as_ref())
            .await?;
        Ok(observe_completion_stream(
            stream,
            Arc::clone(&self.hooks),
            Arc::clone(&self.extensions),
        ))
    }

    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> CompletionEventStream {
        let Self {
            recovery,
            completion,
            hooks,
            request,
            estimate,
            retry_policy,
            extensions,
            span,
            ..
        } = self;
        boxed_sync_stream(try_stream! {
            let mut attempt = 1_u32;
            let mut cumulative_usage = Usage::zero();

            'attempts: loop {
                let stream = completion.completion(request.clone(), extensions.as_ref()).await;
                let mut stream = match stream {
                    Ok(stream) => observe_completion_stream(
                        stream,
                        Arc::clone(&hooks),
                        Arc::clone(&extensions),
                    ),
                    Err(source) => {
                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&retry_policy, attempt, &source)
                        {
                            let accounted_usage =
                                recover_or_estimate_usage(recovery.as_deref(), OperationKind::Completion, None, estimate).await;
                            cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
                            yield CompletionEvent::WillRetry {
                                attempt: next_attempt,
                                after,
                                kind,
                                status,
                                request_id: None,
                                accounted_usage,
                                cumulative_usage,
                            };
                            tokio::time::sleep(after).await;
                            attempt = next_attempt;
                            continue 'attempts;
                        }
                        Err(source)?;
                        break;
                    }
                };

                let mut request_id = None;
                while let Some(item) = stream.next().instrument(span.clone()).await {
                    match item {
                        Ok(event) => {
                            match &event {
                                CompletionEvent::Started {
                                    request_id: event_request_id,
                                    ..
                                } => request_id = event_request_id.clone(),
                                CompletionEvent::Completed {
                                    request_id: event_request_id,
                                    ..
                                } => {
                                    if let Some(event_request_id) = event_request_id.clone() {
                                        request_id = Some(event_request_id);
                                    }
                                }
                                _ => {}
                            }
                            yield event;
                        }
                        Err(source) => {
                            if let Some((next_attempt, after, status, kind)) =
                                maybe_retry_plan(&retry_policy, attempt, &source)
                            {
                                let accounted_usage = recover_or_estimate_usage(
                                    recovery.as_deref(),
                                    OperationKind::Completion,
                                    request_id.as_deref(),
                                    estimate,
                                )
                                .await;
                                cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
                                yield CompletionEvent::WillRetry {
                                    attempt: next_attempt,
                                    after,
                                    kind,
                                    status,
                                    request_id,
                                    accounted_usage,
                                    cumulative_usage,
                                };
                                tokio::time::sleep(after).await;
                                attempt = next_attempt;
                                continue 'attempts;
                            }
                            Err(source)?;
                            break 'attempts;
                        }
                    }
                }

                break;
            }
        })
    }

    pub async fn collect_with<H>(
        mut self,
        mut handler: H,
    ) -> Result<
        CompletionTurnResult,
        CollectError<H::Error, CompletionReductionError, CompletionTurnState>,
    >
    where
        H: EventHandler<CompletionEvent, CompletionTurnState>,
    {
        let mut attempt = 1_u32;
        let mut cumulative_usage = Usage::zero();

        'attempts: loop {
            let mut stream = match self.start_attempt().await {
                Ok(stream) => stream,
                Err(source) => {
                    let partial = self.reducer.state().clone();
                    let accounted_usage = recover_or_estimate_usage(
                        self.recovery.as_deref(),
                        OperationKind::Completion,
                        self.reducer.state().request_id.as_deref(),
                        self.estimate,
                    )
                    .await;
                    let next_cumulative_usage = cumulative_usage.saturating_add(accounted_usage);

                    if let Some((next_attempt, after, status, kind)) =
                        maybe_retry_plan(&self.retry_policy, attempt, &source)
                    {
                        let retry_event = CompletionEvent::WillRetry {
                            attempt: next_attempt,
                            after,
                            kind,
                            status,
                            request_id: self.reducer.state().request_id.clone(),
                            accounted_usage,
                            cumulative_usage: next_cumulative_usage,
                        };
                        match self.call_handler(&mut handler, &retry_event).await {
                            Ok(HandlerDirective::Continue) => {
                                cumulative_usage = next_cumulative_usage;
                                self.reducer.reset_for_retry();
                                tokio::time::sleep(after).await;
                                attempt = next_attempt;
                                continue 'attempts;
                            }
                            Ok(HandlerDirective::Stop) => {
                                if let Err(finalize_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    partial.request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        &self.extensions,
                                        OperationKind::Completion,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_completion_state(&partial),
                                        finalize_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: finalize_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    &self.extensions,
                                    OperationKind::Completion,
                                    partial.request_id.as_deref(),
                                    CollectErrorKind::Execution,
                                    summarize_completion_state(&partial),
                                    source.to_string(),
                                );
                                return Err(CollectError::Execution { source, partial });
                            }
                            Err(handler_source) => {
                                if let Err(finalize_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    partial.request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        &self.extensions,
                                        OperationKind::Completion,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_completion_state(&partial),
                                        finalize_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: finalize_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    &self.extensions,
                                    OperationKind::Completion,
                                    partial.request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_completion_state(&partial),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&handler_source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source: handler_source,
                                    partial: self.reducer.into_state(),
                                });
                            }
                        }
                    }

                    if let Err(finalize_source) = finalize_budget_cumulative(
                        &mut self.owned_lease,
                        &self.span,
                        partial.request_id.as_deref(),
                        next_cumulative_usage,
                    ) {
                        emit_raw_collect_error(
                            &self.extensions,
                            OperationKind::Completion,
                            partial.request_id.as_deref(),
                            CollectErrorKind::Execution,
                            summarize_completion_state(&partial),
                            finalize_source.to_string(),
                        );
                        return Err(CollectError::Execution {
                            source: finalize_source,
                            partial,
                        });
                    }
                    emit_raw_collect_error(
                        &self.extensions,
                        OperationKind::Completion,
                        partial.request_id.as_deref(),
                        CollectErrorKind::Execution,
                        summarize_completion_state(&partial),
                        source.to_string(),
                    );
                    return Err(CollectError::Execution {
                        source,
                        partial: self.reducer.into_state(),
                    });
                }
            };

            while let Some(item) = stream.next().instrument(self.span.clone()).await {
                match item {
                    Ok(event) => {
                        if let Err(source) = self.reducer.apply(&event) {
                            emit_raw_collect_error(
                                &self.extensions,
                                OperationKind::Completion,
                                self.reducer.state().request_id.as_deref(),
                                CollectErrorKind::Reduction,
                                summarize_completion_state(self.reducer.state()),
                                source.to_string(),
                            );
                            return Err(CollectError::Reduction {
                                source,
                                partial: self.reducer.state().clone(),
                            });
                        }
                        record_request_id(&self.span, self.reducer.state().request_id.as_deref());
                        if let Some(usage) = completed_usage_from_completion(&event) {
                            let next_cumulative_usage = cumulative_usage.saturating_add(usage);
                            if let Err(source) = finalize_budget_cumulative(
                                &mut self.owned_lease,
                                &self.span,
                                self.reducer.state().request_id.as_deref(),
                                next_cumulative_usage,
                            ) {
                                emit_raw_collect_error(
                                    &self.extensions,
                                    OperationKind::Completion,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Execution,
                                    summarize_completion_state(self.reducer.state()),
                                    source.to_string(),
                                );
                                return Err(CollectError::Execution {
                                    source,
                                    partial: self.reducer.state().clone(),
                                });
                            }
                            if let Err(source) = self.call_handler(&mut handler, &event).await {
                                emit_raw_collect_error(
                                    &self.extensions,
                                    OperationKind::Completion,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_completion_state(self.reducer.state()),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source,
                                    partial: self.reducer.state().clone(),
                                });
                            }
                            let partial = self.reducer.state().clone();
                            return match self.reducer.into_result() {
                                Ok(mut result) => {
                                    result.cumulative_usage = next_cumulative_usage;
                                    Ok(result)
                                }
                                Err(source) => {
                                    emit_raw_collect_error(
                                        &self.extensions,
                                        OperationKind::Completion,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Reduction,
                                        summarize_completion_state(&partial),
                                        source.to_string(),
                                    );
                                    Err(CollectError::Reduction { source, partial })
                                }
                            };
                        }

                        match self.call_handler(&mut handler, &event).await {
                            Ok(HandlerDirective::Continue) => {}
                            Ok(HandlerDirective::Stop) => {
                                let partial = self.reducer.state().clone();
                                let accounted_usage = recover_or_estimate_usage(
                                    self.recovery.as_deref(),
                                    OperationKind::Completion,
                                    self.reducer.state().request_id.as_deref(),
                                    self.estimate,
                                )
                                .await;
                                let next_cumulative_usage =
                                    cumulative_usage.saturating_add(accounted_usage);
                                if let Err(source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    self.reducer.state().request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        &self.extensions,
                                        OperationKind::Completion,
                                        self.reducer.state().request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_completion_state(&partial),
                                        source.to_string(),
                                    );
                                    return Err(CollectError::Execution { source, partial });
                                }
                                return Err(CollectError::Stopped {
                                    partial: self.reducer.into_state(),
                                });
                            }
                            Err(source) => {
                                let partial = self.reducer.state().clone();
                                let accounted_usage = recover_or_estimate_usage(
                                    self.recovery.as_deref(),
                                    OperationKind::Completion,
                                    self.reducer.state().request_id.as_deref(),
                                    self.estimate,
                                )
                                .await;
                                let next_cumulative_usage =
                                    cumulative_usage.saturating_add(accounted_usage);
                                if let Err(execution_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    self.reducer.state().request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        &self.extensions,
                                        OperationKind::Completion,
                                        self.reducer.state().request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_completion_state(&partial),
                                        execution_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: execution_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    &self.extensions,
                                    OperationKind::Completion,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_completion_state(self.reducer.state()),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source,
                                    partial: self.reducer.into_state(),
                                });
                            }
                        }
                    }
                    Err(source) => {
                        let partial = self.reducer.state().clone();
                        let accounted_usage = recover_or_estimate_usage(
                            self.recovery.as_deref(),
                            OperationKind::Completion,
                            self.reducer.state().request_id.as_deref(),
                            self.estimate,
                        )
                        .await;
                        let next_cumulative_usage =
                            cumulative_usage.saturating_add(accounted_usage);

                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&self.retry_policy, attempt, &source)
                        {
                            let retry_event = CompletionEvent::WillRetry {
                                attempt: next_attempt,
                                after,
                                kind,
                                status,
                                request_id: self.reducer.state().request_id.clone(),
                                accounted_usage,
                                cumulative_usage: next_cumulative_usage,
                            };
                            match self.call_handler(&mut handler, &retry_event).await {
                                Ok(HandlerDirective::Continue) => {
                                    cumulative_usage = next_cumulative_usage;
                                    self.reducer.reset_for_retry();
                                    tokio::time::sleep(after).await;
                                    attempt = next_attempt;
                                    continue 'attempts;
                                }
                                Ok(HandlerDirective::Stop) => {
                                    if let Err(finalize_source) = finalize_budget_cumulative(
                                        &mut self.owned_lease,
                                        &self.span,
                                        partial.request_id.as_deref(),
                                        next_cumulative_usage,
                                    ) {
                                        emit_raw_collect_error(
                                            &self.extensions,
                                            OperationKind::Completion,
                                            partial.request_id.as_deref(),
                                            CollectErrorKind::Execution,
                                            summarize_completion_state(&partial),
                                            finalize_source.to_string(),
                                        );
                                        return Err(CollectError::Execution {
                                            source: finalize_source,
                                            partial,
                                        });
                                    }
                                    emit_raw_collect_error(
                                        &self.extensions,
                                        OperationKind::Completion,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_completion_state(&partial),
                                        source.to_string(),
                                    );
                                    return Err(CollectError::Execution { source, partial });
                                }
                                Err(handler_source) => {
                                    if let Err(finalize_source) = finalize_budget_cumulative(
                                        &mut self.owned_lease,
                                        &self.span,
                                        partial.request_id.as_deref(),
                                        next_cumulative_usage,
                                    ) {
                                        emit_raw_collect_error(
                                            &self.extensions,
                                            OperationKind::Completion,
                                            partial.request_id.as_deref(),
                                            CollectErrorKind::Execution,
                                            summarize_completion_state(&partial),
                                            finalize_source.to_string(),
                                        );
                                        return Err(CollectError::Execution {
                                            source: finalize_source,
                                            partial,
                                        });
                                    }
                                    emit_raw_collect_error(
                                        &self.extensions,
                                        OperationKind::Completion,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Handler,
                                        summarize_completion_state(&partial),
                                        format!(
                                            "handler error type={}",
                                            std::any::type_name_of_val(&handler_source)
                                        ),
                                    );
                                    return Err(CollectError::Handler {
                                        source: handler_source,
                                        partial: self.reducer.into_state(),
                                    });
                                }
                            }
                        }

                        if let Err(execution_source) = finalize_budget_cumulative(
                            &mut self.owned_lease,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            next_cumulative_usage,
                        ) {
                            emit_raw_collect_error(
                                &self.extensions,
                                OperationKind::Completion,
                                self.reducer.state().request_id.as_deref(),
                                CollectErrorKind::Execution,
                                summarize_completion_state(&partial),
                                execution_source.to_string(),
                            );
                            return Err(CollectError::Execution {
                                source: execution_source,
                                partial,
                            });
                        }
                        emit_raw_collect_error(
                            &self.extensions,
                            OperationKind::Completion,
                            self.reducer.state().request_id.as_deref(),
                            CollectErrorKind::Execution,
                            summarize_completion_state(&partial),
                            source.to_string(),
                        );
                        return Err(CollectError::Execution {
                            source,
                            partial: self.reducer.into_state(),
                        });
                    }
                }
            }

            let partial = self.reducer.state().clone();
            let accounted_usage = recover_or_estimate_usage(
                self.recovery.as_deref(),
                OperationKind::Completion,
                self.reducer.state().request_id.as_deref(),
                self.estimate,
            )
            .await;
            let next_cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
            if let Err(source) = finalize_budget_cumulative(
                &mut self.owned_lease,
                &self.span,
                self.reducer.state().request_id.as_deref(),
                next_cumulative_usage,
            ) {
                emit_raw_collect_error(
                    &self.extensions,
                    OperationKind::Completion,
                    self.reducer.state().request_id.as_deref(),
                    CollectErrorKind::Execution,
                    summarize_completion_state(&partial),
                    source.to_string(),
                );
                return Err(CollectError::Execution { source, partial });
            }
            emit_raw_collect_error(
                &self.extensions,
                OperationKind::Completion,
                self.reducer.state().request_id.as_deref(),
                CollectErrorKind::UnexpectedEof,
                summarize_completion_state(self.reducer.state()),
                "stream ended before completion".to_string(),
            );
            return Err(CollectError::UnexpectedEof {
                partial: self.reducer.into_state(),
            });
        }
    }

    pub async fn collect(
        self,
    ) -> Result<
        CompletionTurnResult,
        CollectError<Infallible, CompletionReductionError, CompletionTurnState>,
    > {
        self.collect_with(NoopHandler).await
    }

    async fn call_handler<H>(
        &self,
        handler: &mut H,
        event: &CompletionEvent,
    ) -> Result<HandlerDirective, H::Error>
    where
        H: EventHandler<CompletionEvent, CompletionTurnState>,
    {
        let cx = HandlerContext {
            extensions: &self.extensions,
            state: self.reducer.state(),
            remaining_budget: self.owned_lease.budget.remaining(&self.extensions),
        };
        handler.on_event(event, &cx).await
    }
}

impl<O> PendingStructuredCompletion<O>
where
    O: StructuredOutput,
{
    async fn start_attempt(&self) -> Result<StructuredCompletionEventStream<O>, AgentError> {
        let stream = self
            .completion
            .structured_completion(self.request.clone(), self.extensions.as_ref())
            .await?;
        let stream = observe_structured_completion_stream(
            stream,
            Arc::clone(&self.hooks),
            Arc::clone(&self.extensions),
        );
        Ok(map_structured_completion_stream::<O>(stream))
    }

    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> StructuredCompletionEventStream<O> {
        let Self {
            recovery,
            completion,
            hooks,
            request,
            estimate,
            retry_policy,
            extensions,
            span,
            ..
        } = self;
        boxed_sync_stream(try_stream! {
            let mut attempt = 1_u32;
            let mut cumulative_usage = Usage::zero();

            'attempts: loop {
                let stream = completion
                    .structured_completion(request.clone(), extensions.as_ref())
                    .await;
                let mut stream = match stream {
                    Ok(stream) => map_structured_completion_stream::<O>(
                        observe_structured_completion_stream(
                            stream,
                            Arc::clone(&hooks),
                            Arc::clone(&extensions),
                        ),
                    ),
                    Err(source) => {
                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&retry_policy, attempt, &source)
                        {
                            let accounted_usage =
                                recover_or_estimate_usage(recovery.as_deref(), OperationKind::StructuredCompletion, None, estimate).await;
                            cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
                            yield StructuredCompletionEvent::WillRetry {
                                attempt: next_attempt,
                                after,
                                kind,
                                status,
                                request_id: None,
                                accounted_usage,
                                cumulative_usage,
                            };
                            tokio::time::sleep(after).await;
                            attempt = next_attempt;
                            continue 'attempts;
                        }
                        Err(source)?;
                        break;
                    }
                };

                let mut request_id = None;
                while let Some(item) = stream.next().instrument(span.clone()).await {
                    match item {
                        Ok(event) => {
                            match &event {
                                StructuredCompletionEvent::Started {
                                    request_id: event_request_id,
                                    ..
                                } => request_id = event_request_id.clone(),
                                StructuredCompletionEvent::Completed {
                                    request_id: event_request_id,
                                    ..
                                } => {
                                    if let Some(event_request_id) = event_request_id.clone() {
                                        request_id = Some(event_request_id);
                                    }
                                }
                                _ => {}
                            }
                            yield event;
                        }
                        Err(source) => {
                            if let Some((next_attempt, after, status, kind)) =
                                maybe_retry_plan(&retry_policy, attempt, &source)
                            {
                                let accounted_usage = recover_or_estimate_usage(
                                    recovery.as_deref(),
                                    OperationKind::StructuredCompletion,
                                    request_id.as_deref(),
                                    estimate,
                                )
                                .await;
                                cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
                                yield StructuredCompletionEvent::WillRetry {
                                    attempt: next_attempt,
                                    after,
                                    kind,
                                    status,
                                    request_id,
                                    accounted_usage,
                                    cumulative_usage,
                                };
                                tokio::time::sleep(after).await;
                                attempt = next_attempt;
                                continue 'attempts;
                            }
                            Err(source)?;
                            break 'attempts;
                        }
                    }
                }

                break;
            }
        })
    }

    pub async fn collect_with<H>(
        mut self,
        mut handler: H,
    ) -> Result<
        StructuredCompletionResult<O>,
        CollectError<H::Error, StructuredCompletionReductionError, StructuredCompletionState<O>>,
    >
    where
        H: EventHandler<StructuredCompletionEvent<O>, StructuredCompletionState<O>>,
    {
        let mut attempt = 1_u32;
        let mut cumulative_usage = Usage::zero();

        'attempts: loop {
            let mut stream = match self.start_attempt().await {
                Ok(stream) => stream,
                Err(source) => {
                    let partial = self.reducer.state().clone();
                    let accounted_usage = recover_or_estimate_usage(
                        self.recovery.as_deref(),
                        OperationKind::StructuredCompletion,
                        self.reducer.state().request_id.as_deref(),
                        self.estimate,
                    )
                    .await;
                    let next_cumulative_usage = cumulative_usage.saturating_add(accounted_usage);

                    if let Some((next_attempt, after, status, kind)) =
                        maybe_retry_plan(&self.retry_policy, attempt, &source)
                    {
                        let retry_event = StructuredCompletionEvent::WillRetry {
                            attempt: next_attempt,
                            after,
                            kind,
                            status,
                            request_id: self.reducer.state().request_id.clone(),
                            accounted_usage,
                            cumulative_usage: next_cumulative_usage,
                        };
                        match self.call_handler(&mut handler, &retry_event).await {
                            Ok(HandlerDirective::Continue) => {
                                cumulative_usage = next_cumulative_usage;
                                self.reducer.reset_for_retry();
                                tokio::time::sleep(after).await;
                                attempt = next_attempt;
                                continue 'attempts;
                            }
                            Ok(HandlerDirective::Stop) => {
                                if let Err(finalize_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    partial.request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        &self.extensions,
                                        OperationKind::StructuredCompletion,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_completion_state(&partial),
                                        finalize_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: finalize_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    &self.extensions,
                                    OperationKind::StructuredCompletion,
                                    partial.request_id.as_deref(),
                                    CollectErrorKind::Execution,
                                    summarize_structured_completion_state(&partial),
                                    source.to_string(),
                                );
                                return Err(CollectError::Execution { source, partial });
                            }
                            Err(handler_source) => {
                                if let Err(finalize_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    partial.request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        &self.extensions,
                                        OperationKind::StructuredCompletion,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_completion_state(&partial),
                                        finalize_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: finalize_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    &self.extensions,
                                    OperationKind::StructuredCompletion,
                                    partial.request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_structured_completion_state(&partial),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&handler_source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source: handler_source,
                                    partial: self.reducer.into_state(),
                                });
                            }
                        }
                    }

                    if let Err(finalize_source) = finalize_budget_cumulative(
                        &mut self.owned_lease,
                        &self.span,
                        partial.request_id.as_deref(),
                        next_cumulative_usage,
                    ) {
                        emit_raw_collect_error(
                            &self.extensions,
                            OperationKind::StructuredCompletion,
                            partial.request_id.as_deref(),
                            CollectErrorKind::Execution,
                            summarize_structured_completion_state(&partial),
                            finalize_source.to_string(),
                        );
                        return Err(CollectError::Execution {
                            source: finalize_source,
                            partial,
                        });
                    }
                    emit_raw_collect_error(
                        &self.extensions,
                        OperationKind::StructuredCompletion,
                        partial.request_id.as_deref(),
                        CollectErrorKind::Execution,
                        summarize_structured_completion_state(&partial),
                        source.to_string(),
                    );
                    return Err(CollectError::Execution {
                        source,
                        partial: self.reducer.into_state(),
                    });
                }
            };

            while let Some(item) = stream.next().instrument(self.span.clone()).await {
                match item {
                    Ok(event) => {
                        if let Err(source) = self.reducer.apply(&event) {
                            emit_raw_collect_error(
                                &self.extensions,
                                OperationKind::StructuredCompletion,
                                self.reducer.state().request_id.as_deref(),
                                CollectErrorKind::Reduction,
                                summarize_structured_completion_state(self.reducer.state()),
                                source.to_string(),
                            );
                            return Err(CollectError::Reduction {
                                source,
                                partial: self.reducer.state().clone(),
                            });
                        }
                        record_request_id(&self.span, self.reducer.state().request_id.as_deref());
                        if let Some(usage) = completed_usage_from_structured_completion(&event) {
                            let next_cumulative_usage = cumulative_usage.saturating_add(usage);
                            if let Err(source) = finalize_budget_cumulative(
                                &mut self.owned_lease,
                                &self.span,
                                self.reducer.state().request_id.as_deref(),
                                next_cumulative_usage,
                            ) {
                                emit_raw_collect_error(
                                    &self.extensions,
                                    OperationKind::StructuredCompletion,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Execution,
                                    summarize_structured_completion_state(self.reducer.state()),
                                    source.to_string(),
                                );
                                return Err(CollectError::Execution {
                                    source,
                                    partial: self.reducer.state().clone(),
                                });
                            }
                            if let Err(source) = self.call_handler(&mut handler, &event).await {
                                emit_raw_collect_error(
                                    &self.extensions,
                                    OperationKind::StructuredCompletion,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_structured_completion_state(self.reducer.state()),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source,
                                    partial: self.reducer.state().clone(),
                                });
                            }
                            let partial = self.reducer.state().clone();
                            return match self.reducer.into_result() {
                                Ok(mut result) => {
                                    result.cumulative_usage = next_cumulative_usage;
                                    Ok(result)
                                }
                                Err(source) => {
                                    emit_raw_collect_error(
                                        &self.extensions,
                                        OperationKind::StructuredCompletion,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Reduction,
                                        summarize_structured_completion_state(&partial),
                                        source.to_string(),
                                    );
                                    Err(CollectError::Reduction { source, partial })
                                }
                            };
                        }

                        match self.call_handler(&mut handler, &event).await {
                            Ok(HandlerDirective::Continue) => {}
                            Ok(HandlerDirective::Stop) => {
                                let partial = self.reducer.state().clone();
                                let accounted_usage = recover_or_estimate_usage(
                                    self.recovery.as_deref(),
                                    OperationKind::StructuredCompletion,
                                    self.reducer.state().request_id.as_deref(),
                                    self.estimate,
                                )
                                .await;
                                let next_cumulative_usage =
                                    cumulative_usage.saturating_add(accounted_usage);
                                if let Err(source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    self.reducer.state().request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        &self.extensions,
                                        OperationKind::StructuredCompletion,
                                        self.reducer.state().request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_completion_state(&partial),
                                        source.to_string(),
                                    );
                                    return Err(CollectError::Execution { source, partial });
                                }
                                return Err(CollectError::Stopped {
                                    partial: self.reducer.into_state(),
                                });
                            }
                            Err(source) => {
                                let partial = self.reducer.state().clone();
                                let accounted_usage = recover_or_estimate_usage(
                                    self.recovery.as_deref(),
                                    OperationKind::StructuredCompletion,
                                    self.reducer.state().request_id.as_deref(),
                                    self.estimate,
                                )
                                .await;
                                let next_cumulative_usage =
                                    cumulative_usage.saturating_add(accounted_usage);
                                if let Err(execution_source) = finalize_budget_cumulative(
                                    &mut self.owned_lease,
                                    &self.span,
                                    self.reducer.state().request_id.as_deref(),
                                    next_cumulative_usage,
                                ) {
                                    emit_raw_collect_error(
                                        &self.extensions,
                                        OperationKind::StructuredCompletion,
                                        self.reducer.state().request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_completion_state(&partial),
                                        execution_source.to_string(),
                                    );
                                    return Err(CollectError::Execution {
                                        source: execution_source,
                                        partial,
                                    });
                                }
                                emit_raw_collect_error(
                                    &self.extensions,
                                    OperationKind::StructuredCompletion,
                                    self.reducer.state().request_id.as_deref(),
                                    CollectErrorKind::Handler,
                                    summarize_structured_completion_state(self.reducer.state()),
                                    format!(
                                        "handler error type={}",
                                        std::any::type_name_of_val(&source)
                                    ),
                                );
                                return Err(CollectError::Handler {
                                    source,
                                    partial: self.reducer.into_state(),
                                });
                            }
                        }
                    }
                    Err(source) => {
                        let partial = self.reducer.state().clone();
                        let accounted_usage = recover_or_estimate_usage(
                            self.recovery.as_deref(),
                            OperationKind::StructuredCompletion,
                            self.reducer.state().request_id.as_deref(),
                            self.estimate,
                        )
                        .await;
                        let next_cumulative_usage =
                            cumulative_usage.saturating_add(accounted_usage);

                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&self.retry_policy, attempt, &source)
                        {
                            let retry_event = StructuredCompletionEvent::WillRetry {
                                attempt: next_attempt,
                                after,
                                kind,
                                status,
                                request_id: self.reducer.state().request_id.clone(),
                                accounted_usage,
                                cumulative_usage: next_cumulative_usage,
                            };
                            match self.call_handler(&mut handler, &retry_event).await {
                                Ok(HandlerDirective::Continue) => {
                                    cumulative_usage = next_cumulative_usage;
                                    self.reducer.reset_for_retry();
                                    tokio::time::sleep(after).await;
                                    attempt = next_attempt;
                                    continue 'attempts;
                                }
                                Ok(HandlerDirective::Stop) => {
                                    if let Err(finalize_source) = finalize_budget_cumulative(
                                        &mut self.owned_lease,
                                        &self.span,
                                        partial.request_id.as_deref(),
                                        next_cumulative_usage,
                                    ) {
                                        emit_raw_collect_error(
                                            &self.extensions,
                                            OperationKind::StructuredCompletion,
                                            partial.request_id.as_deref(),
                                            CollectErrorKind::Execution,
                                            summarize_structured_completion_state(&partial),
                                            finalize_source.to_string(),
                                        );
                                        return Err(CollectError::Execution {
                                            source: finalize_source,
                                            partial,
                                        });
                                    }
                                    emit_raw_collect_error(
                                        &self.extensions,
                                        OperationKind::StructuredCompletion,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Execution,
                                        summarize_structured_completion_state(&partial),
                                        source.to_string(),
                                    );
                                    return Err(CollectError::Execution { source, partial });
                                }
                                Err(handler_source) => {
                                    if let Err(finalize_source) = finalize_budget_cumulative(
                                        &mut self.owned_lease,
                                        &self.span,
                                        partial.request_id.as_deref(),
                                        next_cumulative_usage,
                                    ) {
                                        emit_raw_collect_error(
                                            &self.extensions,
                                            OperationKind::StructuredCompletion,
                                            partial.request_id.as_deref(),
                                            CollectErrorKind::Execution,
                                            summarize_structured_completion_state(&partial),
                                            finalize_source.to_string(),
                                        );
                                        return Err(CollectError::Execution {
                                            source: finalize_source,
                                            partial,
                                        });
                                    }
                                    emit_raw_collect_error(
                                        &self.extensions,
                                        OperationKind::StructuredCompletion,
                                        partial.request_id.as_deref(),
                                        CollectErrorKind::Handler,
                                        summarize_structured_completion_state(&partial),
                                        format!(
                                            "handler error type={}",
                                            std::any::type_name_of_val(&handler_source)
                                        ),
                                    );
                                    return Err(CollectError::Handler {
                                        source: handler_source,
                                        partial: self.reducer.into_state(),
                                    });
                                }
                            }
                        }

                        if let Err(execution_source) = finalize_budget_cumulative(
                            &mut self.owned_lease,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            next_cumulative_usage,
                        ) {
                            emit_raw_collect_error(
                                &self.extensions,
                                OperationKind::StructuredCompletion,
                                self.reducer.state().request_id.as_deref(),
                                CollectErrorKind::Execution,
                                summarize_structured_completion_state(&partial),
                                execution_source.to_string(),
                            );
                            return Err(CollectError::Execution {
                                source: execution_source,
                                partial,
                            });
                        }
                        emit_raw_collect_error(
                            &self.extensions,
                            OperationKind::StructuredCompletion,
                            self.reducer.state().request_id.as_deref(),
                            CollectErrorKind::Execution,
                            summarize_structured_completion_state(&partial),
                            source.to_string(),
                        );
                        return Err(CollectError::Execution {
                            source,
                            partial: self.reducer.into_state(),
                        });
                    }
                }
            }

            let partial = self.reducer.state().clone();
            let accounted_usage = recover_or_estimate_usage(
                self.recovery.as_deref(),
                OperationKind::StructuredCompletion,
                self.reducer.state().request_id.as_deref(),
                self.estimate,
            )
            .await;
            let next_cumulative_usage = cumulative_usage.saturating_add(accounted_usage);
            if let Err(source) = finalize_budget_cumulative(
                &mut self.owned_lease,
                &self.span,
                self.reducer.state().request_id.as_deref(),
                next_cumulative_usage,
            ) {
                emit_raw_collect_error(
                    &self.extensions,
                    OperationKind::StructuredCompletion,
                    self.reducer.state().request_id.as_deref(),
                    CollectErrorKind::Execution,
                    summarize_structured_completion_state(&partial),
                    source.to_string(),
                );
                return Err(CollectError::Execution { source, partial });
            }
            emit_raw_collect_error(
                &self.extensions,
                OperationKind::StructuredCompletion,
                self.reducer.state().request_id.as_deref(),
                CollectErrorKind::UnexpectedEof,
                summarize_structured_completion_state(self.reducer.state()),
                "stream ended before completion".to_string(),
            );
            return Err(CollectError::UnexpectedEof {
                partial: self.reducer.into_state(),
            });
        }
    }

    pub async fn collect(
        self,
    ) -> Result<
        StructuredCompletionResult<O>,
        CollectError<Infallible, StructuredCompletionReductionError, StructuredCompletionState<O>>,
    > {
        self.collect_with(NoopHandler).await
    }

    async fn call_handler<H>(
        &self,
        handler: &mut H,
        event: &StructuredCompletionEvent<O>,
    ) -> Result<HandlerDirective, H::Error>
    where
        H: EventHandler<StructuredCompletionEvent<O>, StructuredCompletionState<O>>,
    {
        let cx = HandlerContext {
            extensions: &self.extensions,
            state: self.reducer.state(),
            remaining_budget: self.owned_lease.budget.remaining(&self.extensions),
        };
        handler.on_event(event, &cx).await
    }
}

struct NoopHandler;

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
impl<E, S> EventHandler<E, S> for NoopHandler
where
    E: MaybeSend + MaybeSync + 'static,
    S: MaybeSend + MaybeSync + 'static,
{
    type Error = Infallible;

    async fn on_event(
        &mut self,
        _event: &E,
        _cx: &HandlerContext<S>,
    ) -> Result<HandlerDirective, Self::Error> {
        Ok(HandlerDirective::Continue)
    }
}

fn erase_text_turn<T>(
    turn: ProtocolTextTurn<T>,
    extensions: Arc<RequestExtensions>,
) -> Result<AdapterTextTurn, AgentError>
where
    T: Toolset,
{
    Ok(AdapterTextTurn {
        config: erase_turn_config(turn.config)?,
        extensions,
    })
}

fn erase_text_turn_ref<T>(
    turn: &ProtocolTextTurn<T>,
    generation: GenerationParams,
    extensions: Arc<RequestExtensions>,
) -> Result<AdapterTextTurn, AgentError>
where
    T: Toolset,
{
    Ok(AdapterTextTurn {
        config: erase_turn_config_ref(&turn.config, generation)?,
        extensions,
    })
}

fn erase_structured_turn<T, O>(
    turn: ProtocolStructuredTurn<T, O>,
    extensions: Arc<RequestExtensions>,
) -> Result<AdapterStructuredTurn, AgentError>
where
    T: Toolset,
    O: StructuredOutput,
{
    let output = erase_structured_output_spec(turn.output)?;
    Ok(AdapterStructuredTurn {
        config: erase_turn_config(turn.config)?,
        extensions,
        output,
    })
}

fn erase_structured_turn_ref<T, O>(
    turn: &ProtocolStructuredTurn<T, O>,
    generation: GenerationParams,
    extensions: Arc<RequestExtensions>,
) -> Result<AdapterStructuredTurn, AgentError>
where
    T: Toolset,
    O: StructuredOutput,
{
    let output = erase_structured_output_spec_ref(&turn.output)?;
    Ok(AdapterStructuredTurn {
        config: erase_turn_config_ref(&turn.config, generation)?,
        extensions,
        output,
    })
}

fn erase_structured_completion_request<O>(
    request: StructuredCompletionRequest<O>,
) -> Result<AdapterStructuredCompletionRequest, AgentError>
where
    O: StructuredOutput,
{
    let output = erase_structured_output_spec(request.output)?;
    Ok(AdapterStructuredCompletionRequest {
        system: request.system,
        prompt: request.prompt,
        generation: request.generation,
        output,
    })
}

fn erase_structured_completion_request_ref<O>(
    request: &StructuredCompletionRequest<O>,
) -> Result<AdapterStructuredCompletionRequest, AgentError>
where
    O: StructuredOutput,
{
    let output = erase_structured_output_spec_ref(&request.output)?;
    Ok(AdapterStructuredCompletionRequest {
        system: request.system.clone(),
        prompt: request.prompt.clone(),
        generation: request.generation.clone(),
        output,
    })
}

fn erase_structured_output_spec<O>(
    output: lutum_protocol::StructuredOutputSpec<O>,
) -> Result<AdapterStructuredOutputSpec, AgentError>
where
    O: StructuredOutput,
{
    Ok(AdapterStructuredOutputSpec {
        schema_name: output
            .schema_name
            .unwrap_or_else(|| <O as StructuredOutput>::schema_name().into_owned()),
        schema: match output.schema {
            Some(schema) => schema,
            None => serde_json::to_value(<O as StructuredOutput>::json_schema())?,
        },
    })
}

fn erase_structured_output_spec_ref<O>(
    output: &lutum_protocol::StructuredOutputSpec<O>,
) -> Result<AdapterStructuredOutputSpec, AgentError>
where
    O: StructuredOutput,
{
    Ok(AdapterStructuredOutputSpec {
        schema_name: output
            .schema_name
            .clone()
            .unwrap_or_else(|| <O as StructuredOutput>::schema_name().into_owned()),
        schema: match output.schema.as_ref() {
            Some(schema) => schema.clone(),
            None => serde_json::to_value(<O as StructuredOutput>::json_schema())?,
        },
    })
}

fn erase_turn_config<T>(config: TurnConfig<T>) -> Result<AdapterTurnConfig, AgentError>
where
    T: Toolset,
{
    erase_turn_config_ref(&config, config.generation.clone())
}

fn erase_turn_config_ref<T>(
    config: &TurnConfig<T>,
    generation: GenerationParams,
) -> Result<AdapterTurnConfig, AgentError>
where
    T: Toolset,
{
    let ToolConstraints {
        available,
        requirement,
        description_overrides,
        dynamic_tools,
    } = &config.tools;

    if !dynamic_tools.is_empty() && !T::has_dynamic_slot() {
        return Err(AgentError::InvalidToolConstraints {
            tool: dynamic_tools
                .first()
                .map(|tool| tool.name.clone())
                .unwrap_or_else(|| "(dynamic tools)".to_string()),
        });
    }

    let mut dynamic_name_set = std::collections::HashSet::new();
    for tool in dynamic_tools {
        if !dynamic_name_set.insert(tool.name.as_str()) {
            return Err(AgentError::InvalidToolConstraints {
                tool: tool.name.clone(),
            });
        }
        if T::Selector::try_from_name(tool.name.as_str()).is_some() {
            return Err(AgentError::InvalidToolConstraints {
                tool: tool.name.clone(),
            });
        }
    }

    // Validate: require_tool(x) must be in the available set when availability is restricted.
    if let ToolRequirement::Specific(selector) = requirement {
        let in_available = match available {
            ToolAvailability::All => true,
            ToolAvailability::Default => T::default_selectors().contains(selector),
            ToolAvailability::Only(only) => only.contains(selector),
            ToolAvailability::DefaultPlus(extra) => {
                T::default_selectors().contains(selector) || extra.contains(selector)
            }
        };
        if !in_available {
            return Err(AgentError::InvalidToolConstraints {
                tool: selector.name().to_string(),
            });
        }
    }

    let tool_defs = match available {
        ToolAvailability::All => T::definitions().iter().collect::<Vec<_>>(),
        ToolAvailability::Default => T::definitions_for(T::default_selectors()),
        ToolAvailability::Only(selectors) => T::definitions_for(selectors.iter().copied()),
        ToolAvailability::DefaultPlus(extra) => {
            let mut selectors = T::default_selectors();
            for s in extra {
                if !selectors.contains(s) {
                    selectors.push(*s);
                }
            }
            T::definitions_for(selectors)
        }
    };

    // Build a last-write-wins override map from selector name → description.
    let mut override_map: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for (sel, desc) in description_overrides {
        override_map.insert(sel.name(), desc.as_str());
    }

    let mut tools = tool_defs
        .into_iter()
        .map(|tool| {
            let description = override_map
                .get(tool.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| tool.description.to_string());
            let mut input_schema = serde_json::to_value(tool.input_schema())?;
            // Some providers (e.g. Azure OpenAI) reject object schemas that lack a
            // "properties" field, which schemars emits for no-argument tools.
            if input_schema.get("type") == Some(&serde_json::Value::String("object".into()))
                && input_schema.get("properties").is_none()
            {
                input_schema
                    .as_object_mut()
                    .unwrap()
                    .insert("properties".into(), serde_json::json!({}));
            }
            Ok(AdapterToolDefinition {
                name: tool.name.to_string(),
                description,
                input_schema,
            })
        })
        .collect::<Result<Vec<_>, serde_json::Error>>()?;

    for tool in dynamic_tools {
        tools.push(AdapterToolDefinition {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.input_schema.clone(),
        });
    }

    // When the resolved tool list is empty, tools are effectively disabled.
    // AtLeastOne with no available tools is a constraint violation.
    if tools.is_empty() {
        if let ToolRequirement::AtLeastOne = requirement {
            return Err(AgentError::InvalidToolConstraints {
                tool: "(none available)".to_string(),
            });
        }
        return Ok(AdapterTurnConfig {
            generation,
            tools: vec![],
            tool_choice: AdapterToolChoice::None,
        });
    }

    let tool_choice = match requirement {
        ToolRequirement::Optional => AdapterToolChoice::Auto,
        ToolRequirement::AtLeastOne => AdapterToolChoice::Required,
        ToolRequirement::Specific(selector) => {
            AdapterToolChoice::Specific(selector.name().to_string())
        }
    };

    Ok(AdapterTurnConfig {
        generation,
        tools,
        tool_choice,
    })
}

fn map_text_stream(stream: ErasedTextTurnEventStream) -> TextTurnEventStream {
    Box::pin(stream.map(|item| item.and_then(map_text_event)))
}

fn observe_text_stream(
    mut stream: ErasedTextTurnEventStream,
    hooks: Arc<LutumHooksSet<'static>>,
    extensions: Arc<RequestExtensions>,
) -> ErasedTextTurnEventStream {
    boxed_sync_stream(try_stream! {
        while let Some(item) = stream.next().await {
            let event = item?;
            let cx = StreamEventHookContext::new(
                extensions.as_ref(),
                OperationKind::TextTurn,
                LutumStreamEvent::TextTurn(&event),
            );
            hooks.on_stream_event(&cx).await;
            yield event;
        }
    })
}

fn map_text_stream_with_tools<T>(
    stream: ErasedTextTurnEventStream,
    availability: ToolAvailability<T::Selector>,
    dynamic_names: Vec<String>,
) -> TextTurnEventStreamWithTools<T>
where
    T: Toolset,
{
    Box::pin(stream.map(move |item| {
        item.and_then(|event| map_text_event_with_tools::<T>(event, &availability, &dynamic_names))
    }))
}

fn map_structured_stream<O>(stream: ErasedStructuredTurnEventStream) -> StructuredTurnEventStream<O>
where
    O: StructuredOutput,
{
    Box::pin(stream.map(|item| item.and_then(map_structured_event::<O>)))
}

fn observe_structured_stream(
    mut stream: ErasedStructuredTurnEventStream,
    hooks: Arc<LutumHooksSet<'static>>,
    extensions: Arc<RequestExtensions>,
) -> ErasedStructuredTurnEventStream {
    boxed_sync_stream(try_stream! {
        while let Some(item) = stream.next().await {
            let event = item?;
            let cx = StreamEventHookContext::new(
                extensions.as_ref(),
                OperationKind::StructuredTurn,
                LutumStreamEvent::StructuredTurn(&event),
            );
            hooks.on_stream_event(&cx).await;
            yield event;
        }
    })
}

fn map_structured_stream_with_tools<T, O>(
    stream: ErasedStructuredTurnEventStream,
    availability: ToolAvailability<T::Selector>,
    dynamic_names: Vec<String>,
) -> StructuredTurnEventStreamWithTools<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    Box::pin(stream.map(move |item| {
        item.and_then(|event| {
            map_structured_event_with_tools::<T, O>(event, &availability, &dynamic_names)
        })
    }))
}

fn map_structured_completion_stream<O>(
    stream: ErasedStructuredCompletionEventStream,
) -> StructuredCompletionEventStream<O>
where
    O: StructuredOutput,
{
    Box::pin(stream.map(|item| item.and_then(map_structured_completion_event::<O>)))
}

fn observe_completion_stream(
    mut stream: CompletionEventStream,
    hooks: Arc<LutumHooksSet<'static>>,
    extensions: Arc<RequestExtensions>,
) -> CompletionEventStream {
    boxed_sync_stream(try_stream! {
        while let Some(item) = stream.next().await {
            let event = item?;
            let cx = StreamEventHookContext::new(
                extensions.as_ref(),
                OperationKind::Completion,
                LutumStreamEvent::Completion(&event),
            );
            hooks.on_stream_event(&cx).await;
            yield event;
        }
    })
}

fn observe_structured_completion_stream(
    mut stream: ErasedStructuredCompletionEventStream,
    hooks: Arc<LutumHooksSet<'static>>,
    extensions: Arc<RequestExtensions>,
) -> ErasedStructuredCompletionEventStream {
    boxed_sync_stream(try_stream! {
        while let Some(item) = stream.next().await {
            let event = item?;
            let cx = StreamEventHookContext::new(
                extensions.as_ref(),
                OperationKind::StructuredCompletion,
                LutumStreamEvent::StructuredCompletion(&event),
            );
            hooks.on_stream_event(&cx).await;
            yield event;
        }
    })
}

fn map_text_event(event: ErasedTextTurnEvent) -> Result<TextTurnEvent, AgentError> {
    match event {
        ErasedTextTurnEvent::Started { request_id, model } => {
            Ok(TextTurnEvent::Started { request_id, model })
        }
        ErasedTextTurnEvent::TextDelta { delta } => Ok(TextTurnEvent::TextDelta { delta }),
        ErasedTextTurnEvent::ReasoningDelta { delta } => {
            Ok(TextTurnEvent::ReasoningDelta { delta })
        }
        ErasedTextTurnEvent::RefusalDelta { delta } => Ok(TextTurnEvent::RefusalDelta { delta }),
        ErasedTextTurnEvent::ToolCallChunk { .. } => {
            Err(NoToolsContractViolation::TextTurnToolCallChunk.into())
        }
        ErasedTextTurnEvent::ToolCallReady(_) => {
            Err(NoToolsContractViolation::TextTurnToolCallReady.into())
        }
        ErasedTextTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            committed_turn,
        } => {
            if finish_reason == lutum_protocol::FinishReason::ToolCall {
                Err(NoToolsContractViolation::TextTurnFinishReasonToolCall.into())
            } else {
                Ok(TextTurnEvent::Completed {
                    request_id,
                    finish_reason,
                    usage,
                    committed_turn,
                })
            }
        }
    }
}

fn is_tool_name_allowed<T: Toolset>(
    name: &str,
    availability: &ToolAvailability<T::Selector>,
    dynamic_names: &[String],
) -> bool {
    if dynamic_names
        .iter()
        .any(|dynamic_name| dynamic_name == name)
    {
        return true;
    }
    // This is only the outer availability-policy gate for the current round.
    // With `ToolAvailability::All`, names are not restricted here, so a name may still fail the
    // inner toolset parse step and become `UnknownTool` instead of `NotAvailable`. `NoTools`
    // intentionally lands on that inner path.
    match availability {
        ToolAvailability::All => true,
        ToolAvailability::Default => T::default_selectors().iter().any(|s| s.name() == name),
        ToolAvailability::Only(selectors) => selectors.iter().any(|s| s.name() == name),
        ToolAvailability::DefaultPlus(extra) => {
            T::default_selectors().iter().any(|s| s.name() == name)
                || extra.iter().any(|s| s.name() == name)
        }
    }
}

fn is_registered_dynamic_tool_name(name: &str, dynamic_names: &[String]) -> bool {
    dynamic_names
        .iter()
        .any(|dynamic_name| dynamic_name == name)
}

fn is_static_tool_name<T: Toolset>(name: &str) -> bool {
    T::Selector::try_from_name(name).is_some()
}

enum ReadyToolNameClass {
    Available,
    NotAvailable,
    Unknown,
}

fn classify_ready_tool_name<T: Toolset>(
    name: &str,
    availability: &ToolAvailability<T::Selector>,
    dynamic_names: &[String],
) -> ReadyToolNameClass {
    if T::has_dynamic_slot()
        && !is_registered_dynamic_tool_name(name, dynamic_names)
        && !is_static_tool_name::<T>(name)
        && matches!(availability, ToolAvailability::All)
    {
        return ReadyToolNameClass::Unknown;
    }

    if is_tool_name_allowed::<T>(name, availability, dynamic_names) {
        ReadyToolNameClass::Available
    } else {
        ReadyToolNameClass::NotAvailable
    }
}

fn map_text_event_with_tools<T>(
    event: ErasedTextTurnEvent,
    availability: &ToolAvailability<T::Selector>,
    dynamic_names: &[String],
) -> Result<TextTurnEventWithTools<T>, AgentError>
where
    T: Toolset,
{
    match event {
        ErasedTextTurnEvent::Started { request_id, model } => {
            Ok(TextTurnEventWithTools::Started { request_id, model })
        }
        ErasedTextTurnEvent::TextDelta { delta } => Ok(TextTurnEventWithTools::TextDelta { delta }),
        ErasedTextTurnEvent::ReasoningDelta { delta } => {
            Ok(TextTurnEventWithTools::ReasoningDelta { delta })
        }
        ErasedTextTurnEvent::RefusalDelta { delta } => {
            Ok(TextTurnEventWithTools::RefusalDelta { delta })
        }
        // Level 1 validation: check tool name at stream-event level before deserialization.
        ErasedTextTurnEvent::ToolCallChunk {
            id,
            name,
            arguments_json_delta,
        } => {
            if is_tool_name_allowed::<T>(name.as_str(), availability, dynamic_names) {
                Ok(TextTurnEventWithTools::ToolCallChunk {
                    id,
                    name,
                    arguments_json_delta,
                })
            } else {
                Ok(TextTurnEventWithTools::InvalidToolCallChunk {
                    id,
                    name,
                    arguments_json_delta,
                })
            }
        }
        // Level 2 validation: check tool name after assembly, before parse_tool_call.
        ErasedTextTurnEvent::ToolCallReady(metadata) => {
            let tool_name = metadata.name.as_str();
            match classify_ready_tool_name::<T>(tool_name, availability, dynamic_names) {
                ReadyToolNameClass::Unknown => {
                    let original_metadata = metadata.clone();
                    Ok(TextTurnEventWithTools::ToolCallIssue(
                        RecoverableToolCallIssue::from_tool_call_error(
                            original_metadata,
                            lutum_protocol::ToolCallError::UnknownTool {
                                name: tool_name.to_string(),
                            },
                        ),
                    ))
                }
                ReadyToolNameClass::Available => {
                    let original_metadata = metadata.clone();
                    match T::parse_tool_call(metadata) {
                        Ok(tool_call) => Ok(TextTurnEventWithTools::ToolCallReady(tool_call)),
                        // All current toolset parse errors are model-authored and recoverable here.
                        Err(error) => Ok(TextTurnEventWithTools::ToolCallIssue(
                            RecoverableToolCallIssue::from_tool_call_error(
                                original_metadata,
                                error,
                            ),
                        )),
                    }
                }
                ReadyToolNameClass::NotAvailable => Ok(TextTurnEventWithTools::ToolCallIssue(
                    RecoverableToolCallIssue::not_available(metadata),
                )),
            }
        }
        ErasedTextTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            committed_turn,
        } => Ok(TextTurnEventWithTools::Completed {
            request_id,
            finish_reason,
            usage,
            committed_turn,
        }),
    }
}

fn map_structured_event<O>(
    event: ErasedStructuredTurnEvent,
) -> Result<StructuredTurnEvent<O>, AgentError>
where
    O: StructuredOutput,
{
    match event {
        ErasedStructuredTurnEvent::Started { request_id, model } => {
            Ok(StructuredTurnEvent::Started { request_id, model })
        }
        ErasedStructuredTurnEvent::StructuredOutputChunk { json_delta } => {
            Ok(StructuredTurnEvent::StructuredOutputChunk { json_delta })
        }
        ErasedStructuredTurnEvent::StructuredOutputReady(raw) => {
            Ok(StructuredTurnEvent::StructuredOutputReady(
                raw.deserialize().map_err(AgentError::structured_output)?,
            ))
        }
        ErasedStructuredTurnEvent::ReasoningDelta { delta } => {
            Ok(StructuredTurnEvent::ReasoningDelta { delta })
        }
        ErasedStructuredTurnEvent::RefusalDelta { delta } => {
            Ok(StructuredTurnEvent::RefusalDelta { delta })
        }
        ErasedStructuredTurnEvent::ToolCallChunk { .. } => {
            Err(NoToolsContractViolation::StructuredTurnToolCallChunk.into())
        }
        ErasedStructuredTurnEvent::ToolCallReady(_) => {
            Err(NoToolsContractViolation::StructuredTurnToolCallReady.into())
        }
        ErasedStructuredTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            committed_turn,
        } => {
            if finish_reason == lutum_protocol::FinishReason::ToolCall {
                Err(NoToolsContractViolation::StructuredTurnFinishReasonToolCall.into())
            } else {
                Ok(StructuredTurnEvent::Completed {
                    request_id,
                    finish_reason,
                    usage,
                    committed_turn,
                })
            }
        }
    }
}

fn map_structured_event_with_tools<T, O>(
    event: ErasedStructuredTurnEvent,
    availability: &ToolAvailability<T::Selector>,
    dynamic_names: &[String],
) -> Result<StructuredTurnEventWithTools<T, O>, AgentError>
where
    T: Toolset,
    O: StructuredOutput,
{
    match event {
        ErasedStructuredTurnEvent::Started { request_id, model } => {
            Ok(StructuredTurnEventWithTools::Started { request_id, model })
        }
        ErasedStructuredTurnEvent::StructuredOutputChunk { json_delta } => {
            Ok(StructuredTurnEventWithTools::StructuredOutputChunk { json_delta })
        }
        ErasedStructuredTurnEvent::StructuredOutputReady(raw) => {
            Ok(StructuredTurnEventWithTools::StructuredOutputReady(
                raw.deserialize().map_err(AgentError::structured_output)?,
            ))
        }
        ErasedStructuredTurnEvent::ReasoningDelta { delta } => {
            Ok(StructuredTurnEventWithTools::ReasoningDelta { delta })
        }
        ErasedStructuredTurnEvent::RefusalDelta { delta } => {
            Ok(StructuredTurnEventWithTools::RefusalDelta { delta })
        }
        // Level 1 validation: check tool name at stream-event level before deserialization.
        ErasedStructuredTurnEvent::ToolCallChunk {
            id,
            name,
            arguments_json_delta,
        } => {
            if is_tool_name_allowed::<T>(name.as_str(), availability, dynamic_names) {
                Ok(StructuredTurnEventWithTools::ToolCallChunk {
                    id,
                    name,
                    arguments_json_delta,
                })
            } else {
                Ok(StructuredTurnEventWithTools::InvalidToolCallChunk {
                    id,
                    name,
                    arguments_json_delta,
                })
            }
        }
        // Level 2 validation: check tool name after assembly, before parse_tool_call.
        ErasedStructuredTurnEvent::ToolCallReady(metadata) => {
            let tool_name = metadata.name.as_str();
            match classify_ready_tool_name::<T>(tool_name, availability, dynamic_names) {
                ReadyToolNameClass::Unknown => {
                    let original_metadata = metadata.clone();
                    Ok(StructuredTurnEventWithTools::ToolCallIssue(
                        RecoverableToolCallIssue::from_tool_call_error(
                            original_metadata,
                            lutum_protocol::ToolCallError::UnknownTool {
                                name: tool_name.to_string(),
                            },
                        ),
                    ))
                }
                ReadyToolNameClass::Available => {
                    let original_metadata = metadata.clone();
                    match T::parse_tool_call(metadata) {
                        Ok(tool_call) => Ok(StructuredTurnEventWithTools::ToolCallReady(tool_call)),
                        // All current toolset parse errors are model-authored and recoverable here.
                        Err(error) => Ok(StructuredTurnEventWithTools::ToolCallIssue(
                            RecoverableToolCallIssue::from_tool_call_error(
                                original_metadata,
                                error,
                            ),
                        )),
                    }
                }
                ReadyToolNameClass::NotAvailable => {
                    Ok(StructuredTurnEventWithTools::ToolCallIssue(
                        RecoverableToolCallIssue::not_available(metadata),
                    ))
                }
            }
        }
        ErasedStructuredTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            committed_turn,
        } => Ok(StructuredTurnEventWithTools::Completed {
            request_id,
            finish_reason,
            usage,
            committed_turn,
        }),
    }
}

fn map_structured_completion_event<O>(
    event: ErasedStructuredCompletionEvent,
) -> Result<StructuredCompletionEvent<O>, AgentError>
where
    O: StructuredOutput,
{
    match event {
        ErasedStructuredCompletionEvent::Started { request_id, model } => {
            Ok(StructuredCompletionEvent::Started { request_id, model })
        }
        ErasedStructuredCompletionEvent::StructuredOutputChunk { json_delta } => {
            Ok(StructuredCompletionEvent::StructuredOutputChunk { json_delta })
        }
        ErasedStructuredCompletionEvent::StructuredOutputReady(raw) => {
            Ok(StructuredCompletionEvent::StructuredOutputReady(
                raw.deserialize().map_err(AgentError::structured_output)?,
            ))
        }
        ErasedStructuredCompletionEvent::ReasoningDelta { delta } => {
            Ok(StructuredCompletionEvent::ReasoningDelta { delta })
        }
        ErasedStructuredCompletionEvent::RefusalDelta { delta } => {
            Ok(StructuredCompletionEvent::RefusalDelta { delta })
        }
        ErasedStructuredCompletionEvent::Completed {
            request_id,
            finish_reason,
            usage,
        } => Ok(StructuredCompletionEvent::Completed {
            request_id,
            finish_reason,
            usage,
        }),
    }
}

fn finalize_budget(
    owned_lease: &mut OwnedLease,
    span: &Span,
    request_id: Option<&str>,
    usage: Usage,
) -> Result<(), AgentError> {
    if let Some(request_id) = request_id {
        span.record("request_id", field::display(request_id));
    }
    record_budget_usage(owned_lease, usage)
}

fn finalize_budget_cumulative(
    owned_lease: &mut OwnedLease,
    span: &Span,
    request_id: Option<&str>,
    cumulative_usage: Usage,
) -> Result<(), AgentError> {
    finalize_budget(owned_lease, span, request_id, cumulative_usage)
}

fn record_budget_usage(owned_lease: &mut OwnedLease, usage: Usage) -> Result<(), AgentError> {
    if let Some(lease) = owned_lease.lease.as_ref().cloned() {
        owned_lease.budget.record_used(lease, usage)?;
        owned_lease.lease = None;
    }
    Ok(())
}

fn turn_span(kind: &'static str, estimate: UsageEstimate) -> Span {
    tracing::info_span!(
        target: "lutum",
        "llm_turn",
        kind = %kind,
        model = field::Empty,
        request_id = field::Empty,
        estimate_tokens = estimate.total_tokens,
        estimate_cost_micros_usd = estimate.cost_micros_usd,
        finish_reason = field::Empty
    )
}

fn estimate_with_token_count(
    mut fallback: UsageEstimate,
    input_tokens: u64,
    max_output_tokens: Option<u32>,
) -> UsageEstimate {
    let estimated_non_input_tokens = fallback
        .total_tokens
        .saturating_sub(fallback.input_tokens)
        .max(fallback.output_tokens);
    let output_tokens =
        estimated_non_input_tokens.max(max_output_tokens.map(u64::from).unwrap_or_default());
    fallback.input_tokens = input_tokens;
    fallback.output_tokens = output_tokens;
    fallback.total_tokens = input_tokens.saturating_add(output_tokens);
    fallback
}

fn emit_raw_collect_error(
    extensions: &RequestExtensions,
    operation_kind: OperationKind,
    request_id: Option<&str>,
    kind: CollectErrorKind,
    partial_summary: String,
    error: String,
) {
    emit_collect_error(
        extensions,
        operation_kind,
        request_id,
        kind,
        &partial_summary,
        &error,
    );
}

fn summarize_text_state(state: &TextTurnState) -> String {
    format!(
        "request_id={:?}, model={}, assistant_items={}, finish_reason={:?}, usage_present={}, committed_turn={}",
        state.request_id,
        state.model,
        state.assistant_turn.len(),
        state.finish_reason,
        state.usage.is_some(),
        state.committed_turn.is_some(),
    )
}

fn summarize_text_state_with_tools<T>(state: &TextTurnStateWithTools<T>) -> String
where
    T: Toolset,
{
    format!(
        "request_id={:?}, model={}, assistant_items={}, tool_calls={}, issues={}, continue_suggestion={:?}, finish_reason={:?}, usage_present={}, committed_turn={}",
        state.request_id,
        state.model,
        state.assistant_turn.len(),
        state.tool_calls.len(),
        state.recoverable_tool_call_issues.len(),
        state.continue_suggestion,
        state.finish_reason,
        state.usage.is_some(),
        state.committed_turn.is_some(),
    )
}

fn summarize_structured_partial<O>(partial: &StructuredTurnPartial<O>) -> String
where
    O: StructuredOutput,
{
    format!(
        "request_id={:?}, model={}, assistant_items={}, structured_present={}, refusal_present={}, finish_reason={:?}, usage_present={}, committed_turn={}",
        partial.state.request_id,
        partial.state.model,
        partial.state.assistant_turn.len(),
        partial.state.structured.is_some(),
        partial.state.refusal.is_some(),
        partial.state.finish_reason,
        partial.state.usage.is_some(),
        partial.committed_turn.is_some(),
    )
}

fn summarize_structured_partial_with_tools<T, O>(
    partial: &StructuredTurnPartialWithTools<T, O>,
) -> String
where
    T: Toolset,
    O: StructuredOutput,
{
    format!(
        "request_id={:?}, model={}, assistant_items={}, tool_calls={}, issues={}, continue_suggestion={:?}, structured_present={}, refusal_present={}, finish_reason={:?}, usage_present={}, committed_turn={}",
        partial.state.request_id,
        partial.state.model,
        partial.state.assistant_turn.len(),
        partial.state.tool_calls.len(),
        partial.state.recoverable_tool_call_issues.len(),
        partial.state.continue_suggestion,
        partial.state.structured.is_some(),
        partial.state.refusal.is_some(),
        partial.state.finish_reason,
        partial.state.usage.is_some(),
        partial.committed_turn.is_some(),
    )
}

fn summarize_completion_state(state: &CompletionTurnState) -> String {
    format!(
        "request_id={:?}, model={}, text_len={}, finish_reason={:?}, usage_present={}",
        state.request_id,
        state.model,
        state.text.len(),
        state.finish_reason,
        state.usage.is_some(),
    )
}

fn summarize_structured_completion_state<O>(state: &StructuredCompletionState<O>) -> String
where
    O: StructuredOutput,
{
    format!(
        "request_id={:?}, model={}, structured_present={}, refusal_present={}, finish_reason={:?}, usage_present={}",
        state.request_id,
        state.model,
        state.structured.is_some(),
        state.refusal.is_some(),
        state.finish_reason,
        state.usage.is_some(),
    )
}

fn format_turn_items(iter: lutum_protocol::transcript::TurnItemIter<'_>, buf: &mut String) {
    use std::fmt::Write as _;
    for item in iter {
        if let Some(t) = item.as_text() {
            buf.push_str(t);
            buf.push('\n');
        }
        if let Some(t) = item.as_reasoning() {
            buf.push_str("<reasoning>");
            buf.push_str(t);
            buf.push_str("</reasoning>\n");
        }
        if let Some(tc) = item.as_tool_call() {
            writeln!(
                buf,
                "<tool_call name={}>{}</tool_call>",
                tc.name,
                tc.arguments.get()
            )
            .unwrap();
        }
    }
}

fn log_input_transcript(span: &Span, input: &ModelInput) {
    if !tracing::enabled!(target: "lutum", tracing::Level::DEBUG) {
        return;
    }
    use lutum_protocol::transcript::TurnItemIter;
    use std::fmt::Write as _;
    let mut buf = String::new();
    for item in input.items() {
        match item {
            ModelInputItem::Message { role, content } => {
                writeln!(buf, "[{role:?}]").unwrap();
                for c in content.iter() {
                    match c {
                        MessageContent::Text(t) => buf.push_str(t),
                        MessageContent::Image(_) => buf.push_str("[image]"),
                    }
                }
                buf.push('\n');
            }
            ModelInputItem::Assistant(a) => {
                buf.push_str("[assistant]\n");
                match a {
                    AssistantInputItem::Text(t) => {
                        buf.push_str(t);
                        buf.push('\n');
                    }
                    AssistantInputItem::Image(_) => {
                        buf.push_str("[image]\n");
                    }
                    AssistantInputItem::Reasoning(t) => {
                        buf.push_str("<reasoning>");
                        buf.push_str(t);
                        buf.push_str("</reasoning>\n");
                    }
                    AssistantInputItem::Refusal(t) => {
                        buf.push_str("<refusal>");
                        buf.push_str(t);
                        buf.push_str("</refusal>\n");
                    }
                }
            }
            ModelInputItem::ToolResult(tr) => {
                write!(buf, "[tool_result name={}]\n{}\n", tr.name, tr.result.get()).unwrap();
            }
            ModelInputItem::Turn(committed) => {
                writeln!(buf, "[{:?}]", committed.role()).unwrap();
                format_turn_items(TurnItemIter::new(committed.as_ref()), &mut buf);
            }
        }
        buf.push('\n');
    }
    span.in_scope(|| {
        tracing::event!(
            target: "lutum",
            tracing::Level::DEBUG,
            transcript = %buf,
            "llm_input_transcript"
        );
    });
}

fn log_output_turn(span: &Span, committed: &CommittedTurn) {
    if !tracing::enabled!(target: "lutum", tracing::Level::DEBUG) {
        return;
    }
    use lutum_protocol::transcript::TurnItemIter;
    let mut buf = String::new();
    format_turn_items(TurnItemIter::new(committed.as_ref()), &mut buf);
    span.in_scope(|| {
        tracing::event!(
            target: "lutum",
            tracing::Level::DEBUG,
            output = %buf,
            "llm_output"
        );
    });
}

fn record_request_id(span: &Span, request_id: Option<&str>) {
    if let Some(request_id) = request_id {
        span.record("request_id", field::display(request_id));
    }
}

fn retry_delay_for(
    failure: &RequestFailure,
    retry_policy: &RetryPolicy,
    next_attempt: u32,
) -> Duration {
    failure
        .retry_after
        .unwrap_or_else(|| retry_policy.backoff.delay_for_retry(next_attempt))
}

async fn recover_or_estimate_usage(
    recovery: Option<&dyn UsageRecoveryAdapter>,
    kind: OperationKind,
    request_id: Option<&str>,
    estimate: UsageEstimate,
) -> Usage {
    if let (Some(recovery), Some(request_id)) = (recovery, request_id) {
        match recovery.recover_usage(kind, request_id).await {
            Ok(Some(usage)) => usage,
            Ok(None) => Usage::from_estimate(estimate),
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    kind = ?kind,
                    request_id,
                    "failed to recover usage for retry accounting; falling back to reserved estimate"
                );
                Usage::from_estimate(estimate)
            }
        }
    } else {
        Usage::from_estimate(estimate)
    }
}

fn maybe_retry_plan(
    retry_policy: &RetryPolicy,
    current_attempt: u32,
    source: &AgentError,
) -> Option<(
    u32,
    Duration,
    Option<u16>,
    lutum_protocol::RequestFailureKind,
)> {
    let failure = source.request_failure()?;
    let next_attempt = current_attempt.saturating_add(1);
    retry_policy
        .allows_retry(current_attempt, failure.kind)
        .then(|| {
            (
                next_attempt,
                retry_delay_for(failure, retry_policy, next_attempt),
                failure.status,
                failure.kind,
            )
        })
}

fn completed_usage_from_text(event: &TextTurnEvent) -> Option<Usage> {
    match event {
        TextTurnEvent::Completed { usage, .. } => Some(*usage),
        _ => None,
    }
}

fn completed_usage_from_text_with_tools<T>(event: &TextTurnEventWithTools<T>) -> Option<Usage>
where
    T: Toolset,
{
    match event {
        TextTurnEventWithTools::Completed { usage, .. } => Some(*usage),
        _ => None,
    }
}

fn completed_usage_from_structured<O>(event: &StructuredTurnEvent<O>) -> Option<Usage>
where
    O: StructuredOutput,
{
    match event {
        StructuredTurnEvent::Completed { usage, .. } => Some(*usage),
        _ => None,
    }
}

fn completed_usage_from_structured_with_tools<T, O>(
    event: &StructuredTurnEventWithTools<T, O>,
) -> Option<Usage>
where
    T: Toolset,
    O: StructuredOutput,
{
    match event {
        StructuredTurnEventWithTools::Completed { usage, .. } => Some(*usage),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_tools_on_non_dynamic_toolset_are_rejected() {
        let mut config = TurnConfig::<NoTools>::new();
        config
            .tools
            .dynamic_tools
            .push(lutum_protocol::DynamicTool::new(
                "runtime",
                "Runtime tool",
                serde_json::json!({"type": "object"}),
            ));

        let err = erase_turn_config::<NoTools>(config).unwrap_err();
        assert!(matches!(
            err,
            AgentError::InvalidToolConstraints { ref tool } if tool == "runtime"
        ));
    }
}

fn completed_usage_from_completion(event: &CompletionEvent) -> Option<Usage> {
    match event {
        CompletionEvent::Completed { usage, .. } => Some(*usage),
        _ => None,
    }
}

fn completed_usage_from_structured_completion<O>(
    event: &StructuredCompletionEvent<O>,
) -> Option<Usage>
where
    O: StructuredOutput,
{
    match event {
        StructuredCompletionEvent::Completed { usage, .. } => Some(*usage),
        _ => None,
    }
}

#[test]
#[cfg(not(target_family = "wasm"))]
fn test_pending_turns_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PendingTextTurn>();
    assert_send_sync::<PendingTextTurnWithTools<lutum_protocol::toolset::NoTools>>();
    assert_send_sync::<PendingStructuredTurn<()>>();
    assert_send_sync::<PendingStructuredTurnWithTools<lutum_protocol::toolset::NoTools, ()>>();
    assert_send_sync::<PendingStructuredCompletion<()>>();
    assert_send_sync::<PendingCompletion>();
}

#[test]
fn record_budget_usage_keeps_lease_after_request_budget_exceeded() {
    use lutum_protocol::budget::{
        RequestBudget, SharedPoolBudgetError, SharedPoolBudgetManager, SharedPoolBudgetOptions,
    };

    let budget = Arc::new(SharedPoolBudgetManager::new(SharedPoolBudgetOptions {
        capacity_tokens: 100,
        capacity_cost_micros_usd: 1_000,
        stop_threshold_tokens: 0,
        stop_threshold_cost_micros_usd: 0,
    }));
    let extensions = RequestExtensions::new();
    let lease = budget
        .reserve(
            &extensions,
            &UsageEstimate {
                total_tokens: 8,
                cost_micros_usd: 80,
                ..UsageEstimate::zero()
            },
            RequestBudget::from_tokens(10),
        )
        .unwrap();
    let mut owned_lease = OwnedLease {
        budget: budget.clone(),
        lease: Some(lease),
    };

    let err = record_budget_usage(
        &mut owned_lease,
        Usage {
            total_tokens: 12,
            cost_micros_usd: 120,
            ..Usage::zero()
        },
    )
    .unwrap_err();

    assert!(matches!(
        err,
        AgentError::Budget(ref source)
            if source
                .downcast_ref::<SharedPoolBudgetError>()
                .is_some_and(|err| matches!(err, SharedPoolBudgetError::RequestBudgetExceeded { .. })),
    ));
    assert!(owned_lease.lease.is_some());

    record_budget_usage(&mut owned_lease, Usage::zero()).unwrap();
    assert!(owned_lease.lease.is_none());
    assert_eq!(budget.remaining(&extensions).tokens, 100);
    assert_eq!(budget.remaining(&extensions).cost_micros_usd, 1_000);
}
