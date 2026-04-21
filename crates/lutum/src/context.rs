use std::{
    convert::Infallible,
    ops::Deref,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Duration,
};

use async_stream::try_stream;
use futures::StreamExt;
use thiserror::Error;
use tracing::{Instrument, Span, field};

use lutum_protocol::{
    AgentError, AssistantInputItem, CommittedTurn, NoTools, NoToolsContractViolation,
    budget::{BudgetLease, BudgetManager, Remaining, Usage, UsageEstimate},
    conversation::{MessageContent, ModelInput, ModelInputItem},
    error::RequestFailure,
    extensions::RequestExtensions,
    llm::{
        AdapterStructuredCompletionRequest, AdapterStructuredOutputSpec, AdapterStructuredTurn,
        AdapterTextTurn, AdapterToolChoice, AdapterToolDefinition, AdapterTurnConfig,
        CompletionAdapter, CompletionEvent, CompletionEventStream, CompletionRequest,
        ErasedStructuredCompletionEvent, ErasedStructuredCompletionEventStream,
        ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
        ErasedTextTurnEventStream, OperationKind, RetryPolicy, StructuredCompletionEvent,
        StructuredCompletionEventStream, StructuredCompletionRequest,
        StructuredTurn as ProtocolStructuredTurn, StructuredTurnEvent, StructuredTurnEventStream,
        StructuredTurnEventStreamWithTools, StructuredTurnEventWithTools,
        TextTurn as ProtocolTextTurn, TextTurnEvent, TextTurnEventStream,
        TextTurnEventStreamWithTools, TextTurnEventWithTools, TurnAdapter, TurnConfig,
        UsageRecoveryAdapter,
    },
    reducer::{
        CompletionReducer, CompletionReductionError, CompletionTurnResult, CompletionTurnState,
        StagedStructuredTurnResult, StagedStructuredTurnResultWithTools, StagedTextTurnResult,
        StagedTextTurnResultWithTools, StructuredCompletionReducer,
        StructuredCompletionReductionError, StructuredCompletionResult, StructuredCompletionState,
        StructuredTurnReducer, StructuredTurnReducerWithTools, StructuredTurnReductionError,
        StructuredTurnState, StructuredTurnStateWithTools, TextTurnReducer,
        TextTurnReducerWithTools, TextTurnReductionError, TextTurnState, TextTurnStateWithTools,
    },
    structured::StructuredOutput,
    telemetry::{
        CollectErrorKind, RawTelemetryConfig, emit_collect_error, raw_collect_errors_enabled,
    },
    toolset::{
        RecoverableToolCallIssue, ToolAvailability, ToolConstraints, ToolRequirement, ToolSelector,
        Toolset,
    },
};

use crate::hooks::LutumHooks;

pub type LutumError = AgentError;

#[derive(Debug, Error)]
#[error("completion adapter is not configured; use Lutum::from_parts(...) to provide one")]
struct MissingCompletionAdapter;

#[derive(Clone, Default)]
struct UnsupportedCompletionAdapter;

#[async_trait::async_trait]
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
    recovery: Arc<dyn UsageRecoveryAdapter>,
    hooks: Arc<LutumHooks>,
    default_extensions: Arc<RequestExtensions>,
}

impl Lutum {
    pub fn new<T>(adapter: Arc<T>, budget: impl BudgetManager + 'static) -> Self
    where
        T: TurnAdapter + UsageRecoveryAdapter + 'static,
    {
        Self::with_hooks(adapter, budget, LutumHooks::new())
    }

    pub fn with_hooks<T>(
        adapter: Arc<T>,
        budget: impl BudgetManager + 'static,
        hooks: LutumHooks,
    ) -> Self
    where
        T: TurnAdapter + UsageRecoveryAdapter + 'static,
    {
        Self {
            budget: Arc::new(budget),
            turns: adapter.clone(),
            completion: Arc::new(UnsupportedCompletionAdapter),
            recovery: adapter,
            hooks: Arc::new(hooks),
            default_extensions: Arc::new(RequestExtensions::new()),
        }
    }

    pub fn from_parts(
        turns: Arc<dyn TurnAdapter>,
        completion: Arc<dyn CompletionAdapter>,
        recovery: Arc<dyn UsageRecoveryAdapter>,
        budget: impl BudgetManager + 'static,
    ) -> Self {
        Self::from_parts_with_hooks(turns, completion, recovery, budget, LutumHooks::new())
    }

    pub fn from_parts_with_hooks(
        turns: Arc<dyn TurnAdapter>,
        completion: Arc<dyn CompletionAdapter>,
        recovery: Arc<dyn UsageRecoveryAdapter>,
        budget: impl BudgetManager + 'static,
        hooks: LutumHooks,
    ) -> Self {
        Self {
            budget: Arc::new(budget),
            turns,
            completion,
            recovery,
            hooks: Arc::new(hooks),
            default_extensions: Arc::new(RequestExtensions::new()),
        }
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

    fn apply_default_extensions(&self, mut extensions: RequestExtensions) -> RequestExtensions {
        extensions.push_fallback(Arc::clone(&self.default_extensions));
        extensions
    }

    pub(crate) fn raw_collect_errors_enabled(&self, extensions: &RequestExtensions) -> bool {
        if extensions.contains::<RawTelemetryConfig>() {
            raw_collect_errors_enabled(extensions)
        } else {
            raw_collect_errors_enabled(&self.default_extensions)
        }
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

#[async_trait::async_trait]
pub trait EventHandler<E, S>: Send {
    type Error;

    async fn on_event(
        &mut self,
        event: &E,
        cx: &HandlerContext<S>,
    ) -> Result<HandlerDirective, Self::Error>;
}

#[async_trait::async_trait]
impl<E, S, F, Err> EventHandler<E, S> for F
where
    F: Send + for<'a> FnMut(&'a E, &'a HandlerContext<'a, S>) -> Result<HandlerDirective, Err>,
    E: Sync,
    S: Sync,
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

struct SyncPinnedStream<Item> {
    inner: Mutex<core::pin::Pin<Box<dyn futures::Stream<Item = Item> + Send + 'static>>>,
}

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

fn boxed_sync_stream<Item: 'static>(
    stream: impl futures::Stream<Item = Item> + Send + 'static,
) -> core::pin::Pin<Box<dyn futures::Stream<Item = Item> + Send + Sync + 'static>> {
    Box::pin(SyncPinnedStream {
        inner: Mutex::new(Box::pin(stream)),
    })
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
    recovery: Arc<dyn UsageRecoveryAdapter>,
    turns: Arc<dyn TurnAdapter>,
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
    recovery: Arc<dyn UsageRecoveryAdapter>,
    turns: Arc<dyn TurnAdapter>,
    input: ModelInput,
    turn: AdapterTextTurn,
    availability: ToolAvailability<T::Selector>,
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
    recovery: Arc<dyn UsageRecoveryAdapter>,
    turns: Arc<dyn TurnAdapter>,
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
    recovery: Arc<dyn UsageRecoveryAdapter>,
    turns: Arc<dyn TurnAdapter>,
    input: ModelInput,
    turn: AdapterStructuredTurn,
    availability: ToolAvailability<T::Selector>,
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
    recovery: Arc<dyn UsageRecoveryAdapter>,
    completion: Arc<dyn CompletionAdapter>,
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
    recovery: Arc<dyn UsageRecoveryAdapter>,
    completion: Arc<dyn CompletionAdapter>,
    request: AdapterStructuredCompletionRequest,
    estimate: UsageEstimate,
    retry_policy: RetryPolicy,
    span: Span,
    reducer: StructuredCompletionReducer<O>,
}

impl Lutum {
    pub(crate) async fn run_text_turn(
        &self,
        extensions: RequestExtensions,
        input: ModelInput,
        turn: ProtocolTextTurn<NoTools>,
    ) -> Result<PendingTextTurn, LutumError> {
        input.validate()?;
        let extensions = self.apply_default_extensions(extensions);
        let estimate = self
            .resolve_usage_estimate(&extensions, OperationKind::TextTurn)
            .await;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, turn.config.budget)?;
        let extensions = Arc::new(extensions);
        let retry_policy = extensions.get::<RetryPolicy>().cloned().unwrap_or_default();
        let span = turn_span("text_turn", estimate);
        log_input_transcript(&span, &input);
        let turn = erase_text_turn(turn, Arc::clone(&extensions))?;
        Ok(PendingTextTurn {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: Arc::clone(&self.recovery),
            turns: Arc::clone(&self.turns),
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
    ) -> Result<PendingTextTurnWithTools<T>, LutumError>
    where
        T: Toolset,
    {
        input.validate()?;
        let extensions = self.apply_default_extensions(extensions);
        let estimate = self
            .resolve_usage_estimate(&extensions, OperationKind::TextTurn)
            .await;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, turn.config.budget)?;
        // Extract availability before erase_text_turn consumes the turn config.
        let availability = turn.config.tools.available.clone();
        let extensions = Arc::new(extensions);
        let retry_policy = extensions.get::<RetryPolicy>().cloned().unwrap_or_default();
        let span = turn_span("text_turn", estimate);
        log_input_transcript(&span, &input);
        let turn = erase_text_turn(turn, Arc::clone(&extensions))?;
        Ok(PendingTextTurnWithTools {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: Arc::clone(&self.recovery),
            turns: Arc::clone(&self.turns),
            input,
            turn,
            availability,
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
        let estimate = self
            .resolve_usage_estimate(&extensions, OperationKind::StructuredTurn)
            .await;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, turn.config.budget)?;
        let extensions = Arc::new(extensions);
        let retry_policy = extensions.get::<RetryPolicy>().cloned().unwrap_or_default();
        let span = turn_span("structured_turn", estimate);
        log_input_transcript(&span, &input);
        let turn = erase_structured_turn(turn, Arc::clone(&extensions))?;
        Ok(PendingStructuredTurn {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: Arc::clone(&self.recovery),
            turns: Arc::clone(&self.turns),
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
        let estimate = self
            .resolve_usage_estimate(&extensions, OperationKind::StructuredTurn)
            .await;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, turn.config.budget)?;
        // Extract availability before erase_structured_turn consumes the turn config.
        let availability = turn.config.tools.available.clone();
        let extensions = Arc::new(extensions);
        let retry_policy = extensions.get::<RetryPolicy>().cloned().unwrap_or_default();
        let span = turn_span("structured_turn", estimate);
        log_input_transcript(&span, &input);
        let turn = erase_structured_turn(turn, Arc::clone(&extensions))?;
        Ok(PendingStructuredTurnWithTools {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: Arc::clone(&self.recovery),
            turns: Arc::clone(&self.turns),
            input,
            turn,
            availability,
            estimate,
            retry_policy,
            span,
            reducer: StructuredTurnReducerWithTools::new(),
        })
    }

    pub(crate) async fn run_completion(
        &self,
        extensions: RequestExtensions,
        request: CompletionRequest,
    ) -> Result<PendingCompletion, LutumError> {
        let extensions = self.apply_default_extensions(extensions);
        let estimate = self
            .resolve_usage_estimate(&extensions, OperationKind::Completion)
            .await;
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
            recovery: Arc::clone(&self.recovery),
            completion: Arc::clone(&self.completion),
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
        request: StructuredCompletionRequest<O>,
    ) -> Result<PendingStructuredCompletion<O>, LutumError>
    where
        O: StructuredOutput,
    {
        let extensions = self.apply_default_extensions(extensions);
        let estimate = self
            .resolve_usage_estimate(&extensions, OperationKind::StructuredCompletion)
            .await;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, request.budget)?;
        let retry_policy = extensions.get::<RetryPolicy>().cloned().unwrap_or_default();
        let extensions = Arc::new(extensions);
        let span = turn_span("structured_completion", estimate);
        Ok(PendingStructuredCompletion {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: Arc::clone(&self.recovery),
            completion: Arc::clone(&self.completion),
            request: erase_structured_completion_request(request)?,
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
        Ok(map_text_stream(stream))
    }

    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> TextTurnEventStream {
        let Self {
            recovery,
            turns,
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
                    Ok(stream) => map_text_stream(stream),
                    Err(source) => {
                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&retry_policy, attempt, &source)
                        {
                            let accounted_usage =
                                recover_or_estimate_usage(&*recovery, OperationKind::TextTurn, None, estimate).await;
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
                                    &*recovery,
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
                        &*self.recovery,
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
                                    &*self.recovery,
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
                                    &*self.recovery,
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
                            &*self.recovery,
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
                &*self.recovery,
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
        Ok(map_text_stream_with_tools::<T>(
            stream,
            self.availability.clone(),
        ))
    }

    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> TextTurnEventStreamWithTools<T> {
        let Self {
            recovery,
            turns,
            input,
            turn,
            availability,
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
                    Ok(stream) => map_text_stream_with_tools::<T>(stream, availability.clone()),
                    Err(source) => {
                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&retry_policy, attempt, &source)
                        {
                            let accounted_usage =
                                recover_or_estimate_usage(&*recovery, OperationKind::TextTurn, None, estimate).await;
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
                                    &*recovery,
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
        StagedTextTurnResultWithTools<T>,
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
                        &*self.recovery,
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
                                    &*self.recovery,
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
                                    &*self.recovery,
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
                            &*self.recovery,
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
                &*self.recovery,
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
        StagedTextTurnResultWithTools<T>,
        CollectError<Infallible, TextTurnReductionError, TextTurnStateWithTools<T>>,
    > {
        self.collect_with(NoopHandler).await
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
        Ok(map_structured_stream::<O>(stream))
    }

    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> StructuredTurnEventStream<O> {
        let Self {
            recovery,
            turns,
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
                    Ok(stream) => map_structured_stream::<O>(stream),
                    Err(source) => {
                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&retry_policy, attempt, &source)
                        {
                            let accounted_usage =
                                recover_or_estimate_usage(&*recovery, OperationKind::StructuredTurn, None, estimate).await;
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
                                    &*recovery,
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
                        &*self.recovery,
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
                                    &*self.recovery,
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
                                    &*self.recovery,
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
                            &*self.recovery,
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
                &*self.recovery,
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
        Ok(map_structured_stream_with_tools::<T, O>(
            stream,
            self.availability.clone(),
        ))
    }

    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> StructuredTurnEventStreamWithTools<T, O> {
        let Self {
            recovery,
            turns,
            input,
            turn,
            availability,
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
                    Ok(stream) => map_structured_stream_with_tools::<T, O>(stream, availability.clone()),
                    Err(source) => {
                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&retry_policy, attempt, &source)
                        {
                            let accounted_usage =
                                recover_or_estimate_usage(&*recovery, OperationKind::StructuredTurn, None, estimate).await;
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
                                    &*recovery,
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
                        &*self.recovery,
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
                                    &*self.recovery,
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
                                    &*self.recovery,
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
                            &*self.recovery,
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
                &*self.recovery,
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
        self.completion
            .completion(self.request.clone(), self.extensions.as_ref())
            .await
    }

    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> CompletionEventStream {
        let Self {
            recovery,
            completion,
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
                    Ok(stream) => stream,
                    Err(source) => {
                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&retry_policy, attempt, &source)
                        {
                            let accounted_usage =
                                recover_or_estimate_usage(&*recovery, OperationKind::Completion, None, estimate).await;
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
                                    &*recovery,
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
                        &*self.recovery,
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
                                    &*self.recovery,
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
                                    &*self.recovery,
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
                            &*self.recovery,
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
                &*self.recovery,
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
        Ok(map_structured_completion_stream::<O>(stream))
    }

    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> StructuredCompletionEventStream<O> {
        let Self {
            recovery,
            completion,
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
                    Ok(stream) => map_structured_completion_stream::<O>(stream),
                    Err(source) => {
                        if let Some((next_attempt, after, status, kind)) =
                            maybe_retry_plan(&retry_policy, attempt, &source)
                        {
                            let accounted_usage =
                                recover_or_estimate_usage(&*recovery, OperationKind::StructuredCompletion, None, estimate).await;
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
                                    &*recovery,
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
                        &*self.recovery,
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
                                    &*self.recovery,
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
                                    &*self.recovery,
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
                            &*self.recovery,
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
                &*self.recovery,
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

#[async_trait::async_trait]
impl<E, S> EventHandler<E, S> for NoopHandler
where
    E: Send + Sync + 'static,
    S: Send + Sync + 'static,
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

fn erase_structured_turn<T, O>(
    turn: ProtocolStructuredTurn<T, O>,
    extensions: Arc<RequestExtensions>,
) -> Result<AdapterStructuredTurn, AgentError>
where
    T: Toolset,
    O: StructuredOutput,
{
    Ok(AdapterStructuredTurn {
        config: erase_turn_config(turn.config)?,
        extensions,
        output: AdapterStructuredOutputSpec {
            schema_name: <O as StructuredOutput>::schema_name().into_owned(),
            schema: serde_json::to_value(<O as StructuredOutput>::json_schema())?,
        },
    })
}

fn erase_structured_completion_request<O>(
    request: StructuredCompletionRequest<O>,
) -> Result<AdapterStructuredCompletionRequest, AgentError>
where
    O: StructuredOutput,
{
    Ok(AdapterStructuredCompletionRequest {
        system: request.system,
        prompt: request.prompt,
        generation: request.generation,
        output: AdapterStructuredOutputSpec {
            schema_name: <O as StructuredOutput>::schema_name().into_owned(),
            schema: serde_json::to_value(<O as StructuredOutput>::json_schema())?,
        },
    })
}

fn erase_turn_config<T>(config: TurnConfig<T>) -> Result<AdapterTurnConfig, AgentError>
where
    T: Toolset,
{
    let ToolConstraints {
        available,
        requirement,
        description_overrides,
    } = config.tools;

    // Validate: require_tool(x) must be in the available set when availability is restricted.
    if let ToolRequirement::Specific(ref selector) = requirement {
        let in_available = match &available {
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

    let tool_defs = match &available {
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

    // When the resolved tool list is empty, tools are effectively disabled.
    // AtLeastOne with no available tools is a constraint violation.
    if tool_defs.is_empty() {
        if let ToolRequirement::AtLeastOne = requirement {
            return Err(AgentError::InvalidToolConstraints {
                tool: "(none available)".to_string(),
            });
        }
        return Ok(AdapterTurnConfig {
            generation: config.generation,
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

    // Build a last-write-wins override map from selector name → description.
    let mut override_map: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for (sel, desc) in &description_overrides {
        override_map.insert(sel.name(), desc.as_str());
    }

    let tools = tool_defs
        .into_iter()
        .map(|tool| {
            let description = override_map
                .get(tool.name)
                .map(|s| s.to_string())
                .unwrap_or_else(|| tool.description.to_string());
            Ok(AdapterToolDefinition {
                name: tool.name.to_string(),
                description,
                input_schema: serde_json::to_value(tool.input_schema())?,
            })
        })
        .collect::<Result<Vec<_>, serde_json::Error>>()?;

    Ok(AdapterTurnConfig {
        generation: config.generation,
        tools,
        tool_choice,
    })
}

fn map_text_stream(stream: ErasedTextTurnEventStream) -> TextTurnEventStream {
    Box::pin(stream.map(|item| item.and_then(map_text_event)))
}

fn map_text_stream_with_tools<T>(
    stream: ErasedTextTurnEventStream,
    availability: ToolAvailability<T::Selector>,
) -> TextTurnEventStreamWithTools<T>
where
    T: Toolset,
{
    Box::pin(stream.map(move |item| {
        item.and_then(|event| map_text_event_with_tools::<T>(event, &availability))
    }))
}

fn map_structured_stream<O>(stream: ErasedStructuredTurnEventStream) -> StructuredTurnEventStream<O>
where
    O: StructuredOutput,
{
    Box::pin(stream.map(|item| item.and_then(map_structured_event::<O>)))
}

fn map_structured_stream_with_tools<T, O>(
    stream: ErasedStructuredTurnEventStream,
    availability: ToolAvailability<T::Selector>,
) -> StructuredTurnEventStreamWithTools<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    Box::pin(stream.map(move |item| {
        item.and_then(|event| map_structured_event_with_tools::<T, O>(event, &availability))
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
) -> bool {
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

fn map_text_event_with_tools<T>(
    event: ErasedTextTurnEvent,
    availability: &ToolAvailability<T::Selector>,
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
            if is_tool_name_allowed::<T>(name.as_str(), availability) {
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
            if is_tool_name_allowed::<T>(metadata.name.as_str(), availability) {
                let original_metadata = metadata.clone();
                match T::parse_tool_call(metadata) {
                    Ok(tool_call) => Ok(TextTurnEventWithTools::ToolCallReady(tool_call)),
                    // All current toolset parse errors are model-authored and recoverable here.
                    Err(error) => Ok(TextTurnEventWithTools::ToolCallIssue(
                        RecoverableToolCallIssue::from_tool_call_error(original_metadata, error),
                    )),
                }
            } else {
                Ok(TextTurnEventWithTools::ToolCallIssue(
                    RecoverableToolCallIssue::not_available(metadata),
                ))
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
            if is_tool_name_allowed::<T>(name.as_str(), availability) {
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
            if is_tool_name_allowed::<T>(metadata.name.as_str(), availability) {
                let original_metadata = metadata.clone();
                match T::parse_tool_call(metadata) {
                    Ok(tool_call) => Ok(StructuredTurnEventWithTools::ToolCallReady(tool_call)),
                    // All current toolset parse errors are model-authored and recoverable here.
                    Err(error) => Ok(StructuredTurnEventWithTools::ToolCallIssue(
                        RecoverableToolCallIssue::from_tool_call_error(original_metadata, error),
                    )),
                }
            } else {
                Ok(StructuredTurnEventWithTools::ToolCallIssue(
                    RecoverableToolCallIssue::not_available(metadata),
                ))
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
    recovery: &dyn UsageRecoveryAdapter,
    kind: OperationKind,
    request_id: Option<&str>,
    estimate: UsageEstimate,
) -> Usage {
    if let Some(request_id) = request_id {
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
