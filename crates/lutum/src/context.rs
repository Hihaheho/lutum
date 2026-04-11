use std::{convert::Infallible, ops::Deref, sync::Arc};

use futures::StreamExt;
use thiserror::Error;
use tracing::{Instrument, Span, field};

use lutum_protocol::{
    AgentError, CommittedTurn, NoTools, NoToolsContractViolation,
    budget::{BudgetLease, BudgetManager, Remaining, Usage, UsageEstimate},
    conversation::ModelInput,
    extensions::RequestExtensions,
    llm::{
        AdapterStructuredCompletionRequest, AdapterStructuredOutputSpec, AdapterStructuredTurn,
        AdapterTextTurn, AdapterToolChoice, AdapterToolDefinition, AdapterTurnConfig,
        CompletionAdapter, CompletionEvent, CompletionEventStream, CompletionRequest,
        ErasedStructuredCompletionEvent, ErasedStructuredCompletionEventStream,
        ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream, ErasedTextTurnEvent,
        ErasedTextTurnEventStream, OperationKind, StructuredCompletionEvent,
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
    toolset::{ToolAvailability, ToolConstraints, ToolRequirement, ToolSelector, Toolset},
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
        }
    }

    pub fn budget(&self) -> &dyn BudgetManager {
        self.budget.as_ref()
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
    span: Span,
    stream: TextTurnEventStream,
    reducer: TextTurnReducer,
}

pub struct PendingTextTurnWithTools<T>
where
    T: Toolset,
{
    extensions: Arc<RequestExtensions>,
    owned_lease: OwnedLease,
    recovery: Arc<dyn UsageRecoveryAdapter>,
    span: Span,
    stream: TextTurnEventStreamWithTools<T>,
    reducer: TextTurnReducerWithTools<T>,
}

pub struct PendingStructuredTurn<O>
where
    O: StructuredOutput,
{
    extensions: Arc<RequestExtensions>,
    owned_lease: OwnedLease,
    recovery: Arc<dyn UsageRecoveryAdapter>,
    span: Span,
    stream: StructuredTurnEventStream<O>,
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
    span: Span,
    stream: StructuredTurnEventStreamWithTools<T, O>,
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
    extensions: RequestExtensions,
    owned_lease: OwnedLease,
    recovery: Arc<dyn UsageRecoveryAdapter>,
    span: Span,
    stream: CompletionEventStream,
    reducer: CompletionReducer,
}

pub struct PendingStructuredCompletion<O>
where
    O: StructuredOutput,
{
    extensions: RequestExtensions,
    owned_lease: OwnedLease,
    recovery: Arc<dyn UsageRecoveryAdapter>,
    span: Span,
    stream: StructuredCompletionEventStream<O>,
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
        let estimate = self
            .resolve_usage_estimate(&extensions, OperationKind::TextTurn)
            .await;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, turn.config.budget)?;
        let extensions = Arc::new(extensions);
        let span = turn_span("text_turn", estimate);
        let stream = match self
            .turns
            .text_turn(input, erase_text_turn(turn, Arc::clone(&extensions))?)
            .await
        {
            Ok(stream) => stream,
            Err(source) => {
                self.budget.record_used(lease, Usage::zero())?;
                return Err(source);
            }
        };
        Ok(PendingTextTurn {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: Arc::clone(&self.recovery),
            span,
            stream: map_text_stream(stream),
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
        let estimate = self
            .resolve_usage_estimate(&extensions, OperationKind::TextTurn)
            .await;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, turn.config.budget)?;
        // Extract availability before erase_text_turn consumes the turn config.
        let availability = turn.config.tools.available.clone();
        let extensions = Arc::new(extensions);
        let span = turn_span("text_turn", estimate);
        let stream = match self
            .turns
            .text_turn(input, erase_text_turn(turn, Arc::clone(&extensions))?)
            .await
        {
            Ok(stream) => stream,
            Err(source) => {
                self.budget.record_used(lease, Usage::zero())?;
                return Err(source);
            }
        };
        Ok(PendingTextTurnWithTools {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: Arc::clone(&self.recovery),
            span,
            stream: map_text_stream_with_tools::<T>(stream, availability),
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
        let estimate = self
            .resolve_usage_estimate(&extensions, OperationKind::StructuredTurn)
            .await;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, turn.config.budget)?;
        let extensions = Arc::new(extensions);
        let span = turn_span("structured_turn", estimate);
        let stream = match self
            .turns
            .structured_turn(input, erase_structured_turn(turn, Arc::clone(&extensions))?)
            .await
        {
            Ok(stream) => stream,
            Err(source) => {
                self.budget.record_used(lease, Usage::zero())?;
                return Err(source);
            }
        };
        Ok(PendingStructuredTurn {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: Arc::clone(&self.recovery),
            span,
            stream: map_structured_stream::<O>(stream),
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
        let estimate = self
            .resolve_usage_estimate(&extensions, OperationKind::StructuredTurn)
            .await;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, turn.config.budget)?;
        // Extract availability before erase_structured_turn consumes the turn config.
        let availability = turn.config.tools.available.clone();
        let extensions = Arc::new(extensions);
        let span = turn_span("structured_turn", estimate);
        let stream = match self
            .turns
            .structured_turn(input, erase_structured_turn(turn, Arc::clone(&extensions))?)
            .await
        {
            Ok(stream) => stream,
            Err(source) => {
                self.budget.record_used(lease, Usage::zero())?;
                return Err(source);
            }
        };
        Ok(PendingStructuredTurnWithTools {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: Arc::clone(&self.recovery),
            span,
            stream: map_structured_stream_with_tools::<T, O>(stream, availability),
            reducer: StructuredTurnReducerWithTools::new(),
        })
    }

    pub(crate) async fn run_completion(
        &self,
        extensions: RequestExtensions,
        request: CompletionRequest,
    ) -> Result<PendingCompletion, LutumError> {
        let estimate = self
            .resolve_usage_estimate(&extensions, OperationKind::Completion)
            .await;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, request.budget)?;
        let span = turn_span("completion", estimate);
        let stream = match self.completion.completion(request, &extensions).await {
            Ok(stream) => stream,
            Err(source) => {
                self.budget.record_used(lease, Usage::zero())?;
                return Err(source);
            }
        };
        Ok(PendingCompletion {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: Arc::clone(&self.recovery),
            span,
            stream,
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
        let estimate = self
            .resolve_usage_estimate(&extensions, OperationKind::StructuredCompletion)
            .await;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, request.budget)?;
        let span = turn_span("structured_completion", estimate);
        let stream = match self
            .completion
            .structured_completion(erase_structured_completion_request(request)?, &extensions)
            .await
        {
            Ok(stream) => stream,
            Err(source) => {
                self.budget.record_used(lease, Usage::zero())?;
                return Err(source);
            }
        };
        Ok(PendingStructuredCompletion {
            extensions,
            owned_lease: OwnedLease {
                budget: Arc::clone(&self.budget),
                lease: Some(lease),
            },
            recovery: Arc::clone(&self.recovery),
            span,
            stream: map_structured_completion_stream::<O>(stream),
            reducer: StructuredCompletionReducer::new(),
        })
    }
}

impl PendingTextTurn {
    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> TextTurnEventStream {
        let Self { stream, .. } = self;
        stream
    }

    pub async fn collect_with<H>(
        mut self,
        mut handler: H,
    ) -> Result<StagedTextTurnResult, CollectError<H::Error, TextTurnReductionError, TextTurnState>>
    where
        H: EventHandler<TextTurnEvent, TextTurnState>,
    {
        while let Some(item) = self.stream.next().instrument(self.span.clone()).await {
            match item {
                Ok(event) => {
                    if let Err(source) = self.reducer.apply(&event) {
                        return Err(CollectError::Reduction {
                            source,
                            partial: self.reducer.state().clone(),
                        });
                    }
                    record_request_id(&self.span, self.reducer.state().request_id.as_deref());
                    if let Some(usage) = completed_usage_from_text(&event) {
                        if let Err(source) = finalize_budget(
                            &mut self.owned_lease,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            usage,
                        ) {
                            return Err(CollectError::Execution {
                                source,
                                partial: self.reducer.state().clone(),
                            });
                        }
                        if let Err(source) = self.call_handler(&mut handler, &event).await {
                            return Err(CollectError::Handler {
                                source,
                                partial: self.reducer.state().clone(),
                            });
                        }
                        let partial = self.reducer.state().clone();
                        return self
                            .reducer
                            .into_result()
                            .map_err(|source| CollectError::Reduction { source, partial });
                    }

                    match self.call_handler(&mut handler, &event).await {
                        Ok(HandlerDirective::Continue) => {}
                        Ok(HandlerDirective::Stop) => {
                            let partial = self.reducer.state().clone();
                            if let Err(source) = recover_or_release_budget(
                                &mut self.owned_lease,
                                &*self.recovery,
                                OperationKind::TextTurn,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Execution { source, partial });
                            }
                            return Err(CollectError::Stopped {
                                partial: self.reducer.into_state(),
                            });
                        }
                        Err(source) => {
                            let partial = self.reducer.state().clone();
                            if let Err(execution_source) = recover_or_release_budget(
                                &mut self.owned_lease,
                                &*self.recovery,
                                OperationKind::TextTurn,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Execution {
                                    source: execution_source,
                                    partial,
                                });
                            }
                            return Err(CollectError::Handler {
                                source,
                                partial: self.reducer.into_state(),
                            });
                        }
                    }
                }
                Err(source) => {
                    let partial = self.reducer.state().clone();
                    if let Err(execution_source) = recover_or_release_budget(
                        &mut self.owned_lease,
                        &*self.recovery,
                        OperationKind::TextTurn,
                        self.reducer.state().request_id.as_deref(),
                    )
                    .await
                    {
                        return Err(CollectError::Execution {
                            source: execution_source,
                            partial,
                        });
                    }
                    return Err(CollectError::Execution {
                        source,
                        partial: self.reducer.into_state(),
                    });
                }
            }
        }

        let partial = self.reducer.state().clone();
        if let Err(source) = recover_or_release_budget(
            &mut self.owned_lease,
            &*self.recovery,
            OperationKind::TextTurn,
            self.reducer.state().request_id.as_deref(),
        )
        .await
        {
            return Err(CollectError::Execution { source, partial });
        }
        Err(CollectError::UnexpectedEof {
            partial: self.reducer.into_state(),
        })
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
    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> TextTurnEventStreamWithTools<T> {
        let Self { stream, .. } = self;
        stream
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
        while let Some(item) = self.stream.next().instrument(self.span.clone()).await {
            match item {
                Ok(event) => {
                    if let Err(source) = self.reducer.apply(&event) {
                        return Err(CollectError::Reduction {
                            source,
                            partial: self.reducer.state().clone(),
                        });
                    }
                    record_request_id(&self.span, self.reducer.state().request_id.as_deref());
                    if let Some(usage) = completed_usage_from_text_with_tools(&event) {
                        if let Err(source) = finalize_budget(
                            &mut self.owned_lease,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            usage,
                        ) {
                            return Err(CollectError::Execution {
                                source,
                                partial: self.reducer.state().clone(),
                            });
                        }
                        if let Err(source) = self.call_handler(&mut handler, &event).await {
                            return Err(CollectError::Handler {
                                source,
                                partial: self.reducer.state().clone(),
                            });
                        }
                        let partial = self.reducer.state().clone();
                        return self
                            .reducer
                            .into_result()
                            .map_err(|source| CollectError::Reduction { source, partial });
                    }

                    match self.call_handler(&mut handler, &event).await {
                        Ok(HandlerDirective::Continue) => {}
                        Ok(HandlerDirective::Stop) => {
                            let partial = self.reducer.state().clone();
                            if let Err(source) = recover_or_release_budget(
                                &mut self.owned_lease,
                                &*self.recovery,
                                OperationKind::TextTurn,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Execution { source, partial });
                            }
                            return Err(CollectError::Stopped {
                                partial: self.reducer.into_state(),
                            });
                        }
                        Err(source) => {
                            let partial = self.reducer.state().clone();
                            if let Err(execution_source) = recover_or_release_budget(
                                &mut self.owned_lease,
                                &*self.recovery,
                                OperationKind::TextTurn,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Execution {
                                    source: execution_source,
                                    partial,
                                });
                            }
                            return Err(CollectError::Handler {
                                source,
                                partial: self.reducer.into_state(),
                            });
                        }
                    }
                }
                Err(source) => {
                    let partial = self.reducer.state().clone();
                    if let Err(execution_source) = recover_or_release_budget(
                        &mut self.owned_lease,
                        &*self.recovery,
                        OperationKind::TextTurn,
                        self.reducer.state().request_id.as_deref(),
                    )
                    .await
                    {
                        return Err(CollectError::Execution {
                            source: execution_source,
                            partial,
                        });
                    }
                    return Err(CollectError::Execution {
                        source,
                        partial: self.reducer.into_state(),
                    });
                }
            }
        }

        let partial = self.reducer.state().clone();
        if let Err(source) = recover_or_release_budget(
            &mut self.owned_lease,
            &*self.recovery,
            OperationKind::TextTurn,
            self.reducer.state().request_id.as_deref(),
        )
        .await
        {
            return Err(CollectError::Execution { source, partial });
        }
        Err(CollectError::UnexpectedEof {
            partial: self.reducer.into_state(),
        })
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
    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> StructuredTurnEventStream<O> {
        let Self { stream, .. } = self;
        stream
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
        while let Some(item) = self.stream.next().instrument(self.span.clone()).await {
            match item {
                Ok(event) => {
                    if let Err(source) = self.reducer.apply(&event) {
                        return Err(CollectError::Reduction {
                            source,
                            partial: StructuredTurnPartial::from_state(
                                self.reducer.state().clone(),
                            ),
                        });
                    }
                    record_request_id(&self.span, self.reducer.state().request_id.as_deref());
                    if let Some(usage) = completed_usage_from_structured(&event) {
                        if let Err(source) = finalize_budget(
                            &mut self.owned_lease,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            usage,
                        ) {
                            return Err(CollectError::Execution {
                                source,
                                partial: StructuredTurnPartial::from_state(
                                    self.reducer.state().clone(),
                                ),
                            });
                        }
                        if let Err(source) = self.call_handler(&mut handler, &event).await {
                            return Err(CollectError::Handler {
                                source,
                                partial: StructuredTurnPartial::from_state(
                                    self.reducer.state().clone(),
                                ),
                            });
                        }
                        let partial =
                            StructuredTurnPartial::from_state(self.reducer.state().clone());
                        return self
                            .reducer
                            .into_result()
                            .map_err(|(source, committed_turn)| {
                                let partial = if let Some(committed_turn) = committed_turn {
                                    partial.with_committed_turn(committed_turn)
                                } else {
                                    partial
                                };
                                CollectError::Reduction { source, partial }
                            });
                    }

                    match self.call_handler(&mut handler, &event).await {
                        Ok(HandlerDirective::Continue) => {}
                        Ok(HandlerDirective::Stop) => {
                            let partial =
                                StructuredTurnPartial::from_state(self.reducer.state().clone());
                            if let Err(source) = recover_or_release_budget(
                                &mut self.owned_lease,
                                &*self.recovery,
                                OperationKind::StructuredTurn,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
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
                            if let Err(execution_source) = recover_or_release_budget(
                                &mut self.owned_lease,
                                &*self.recovery,
                                OperationKind::StructuredTurn,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Execution {
                                    source: execution_source,
                                    partial,
                                });
                            }
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
                    let partial = StructuredTurnPartial::from_state(self.reducer.state().clone());
                    if let Err(execution_source) = recover_or_release_budget(
                        &mut self.owned_lease,
                        &*self.recovery,
                        OperationKind::StructuredTurn,
                        self.reducer.state().request_id.as_deref(),
                    )
                    .await
                    {
                        return Err(CollectError::Execution {
                            source: execution_source,
                            partial,
                        });
                    }
                    return Err(CollectError::Execution {
                        source,
                        partial: StructuredTurnPartial::from_state(self.reducer.into_state()),
                    });
                }
            }
        }

        let partial = StructuredTurnPartial::from_state(self.reducer.state().clone());
        if let Err(source) = recover_or_release_budget(
            &mut self.owned_lease,
            &*self.recovery,
            OperationKind::StructuredTurn,
            self.reducer.state().request_id.as_deref(),
        )
        .await
        {
            return Err(CollectError::Execution { source, partial });
        }
        Err(CollectError::UnexpectedEof {
            partial: StructuredTurnPartial::from_state(self.reducer.into_state()),
        })
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
    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> StructuredTurnEventStreamWithTools<T, O> {
        let Self { stream, .. } = self;
        stream
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
        while let Some(item) = self.stream.next().instrument(self.span.clone()).await {
            match item {
                Ok(event) => {
                    if let Err(source) = self.reducer.apply(&event) {
                        return Err(CollectError::Reduction {
                            source,
                            partial: StructuredTurnPartialWithTools::from_state(
                                self.reducer.state().clone(),
                            ),
                        });
                    }
                    record_request_id(&self.span, self.reducer.state().request_id.as_deref());
                    if let Some(usage) = completed_usage_from_structured_with_tools(&event) {
                        if let Err(source) = finalize_budget(
                            &mut self.owned_lease,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            usage,
                        ) {
                            return Err(CollectError::Execution {
                                source,
                                partial: StructuredTurnPartialWithTools::from_state(
                                    self.reducer.state().clone(),
                                ),
                            });
                        }
                        if let Err(source) = self.call_handler(&mut handler, &event).await {
                            return Err(CollectError::Handler {
                                source,
                                partial: StructuredTurnPartialWithTools::from_state(
                                    self.reducer.state().clone(),
                                ),
                            });
                        }
                        let partial = StructuredTurnPartialWithTools::from_state(
                            self.reducer.state().clone(),
                        );
                        return self
                            .reducer
                            .into_result()
                            .map_err(|(source, committed_turn)| {
                                let partial = if let Some(committed_turn) = committed_turn {
                                    partial.with_committed_turn(committed_turn)
                                } else {
                                    partial
                                };
                                CollectError::Reduction { source, partial }
                            });
                    }

                    match self.call_handler(&mut handler, &event).await {
                        Ok(HandlerDirective::Continue) => {}
                        Ok(HandlerDirective::Stop) => {
                            let partial = StructuredTurnPartialWithTools::from_state(
                                self.reducer.state().clone(),
                            );
                            if let Err(source) = recover_or_release_budget(
                                &mut self.owned_lease,
                                &*self.recovery,
                                OperationKind::StructuredTurn,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
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
                            if let Err(execution_source) = recover_or_release_budget(
                                &mut self.owned_lease,
                                &*self.recovery,
                                OperationKind::StructuredTurn,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Execution {
                                    source: execution_source,
                                    partial,
                                });
                            }
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
                    let partial =
                        StructuredTurnPartialWithTools::from_state(self.reducer.state().clone());
                    if let Err(execution_source) = recover_or_release_budget(
                        &mut self.owned_lease,
                        &*self.recovery,
                        OperationKind::StructuredTurn,
                        self.reducer.state().request_id.as_deref(),
                    )
                    .await
                    {
                        return Err(CollectError::Execution {
                            source: execution_source,
                            partial,
                        });
                    }
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
        if let Err(source) = recover_or_release_budget(
            &mut self.owned_lease,
            &*self.recovery,
            OperationKind::StructuredTurn,
            self.reducer.state().request_id.as_deref(),
        )
        .await
        {
            return Err(CollectError::Execution { source, partial });
        }
        Err(CollectError::UnexpectedEof {
            partial: StructuredTurnPartialWithTools::from_state(self.reducer.into_state()),
        })
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
    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> CompletionEventStream {
        let Self { stream, .. } = self;
        stream
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
        while let Some(item) = self.stream.next().instrument(self.span.clone()).await {
            match item {
                Ok(event) => {
                    if let Err(source) = self.reducer.apply(&event) {
                        return Err(CollectError::Reduction {
                            source,
                            partial: self.reducer.state().clone(),
                        });
                    }
                    record_request_id(&self.span, self.reducer.state().request_id.as_deref());
                    if let Some(usage) = completed_usage_from_completion(&event) {
                        if let Err(source) = finalize_budget(
                            &mut self.owned_lease,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            usage,
                        ) {
                            return Err(CollectError::Execution {
                                source,
                                partial: self.reducer.state().clone(),
                            });
                        }
                        if let Err(source) = self.call_handler(&mut handler, &event).await {
                            return Err(CollectError::Handler {
                                source,
                                partial: self.reducer.state().clone(),
                            });
                        }
                        let partial = self.reducer.state().clone();
                        return self
                            .reducer
                            .into_result()
                            .map_err(|source| CollectError::Reduction { source, partial });
                    }

                    match self.call_handler(&mut handler, &event).await {
                        Ok(HandlerDirective::Continue) => {}
                        Ok(HandlerDirective::Stop) => {
                            let partial = self.reducer.state().clone();
                            if let Err(source) = recover_or_release_budget(
                                &mut self.owned_lease,
                                &*self.recovery,
                                OperationKind::Completion,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Execution { source, partial });
                            }
                            return Err(CollectError::Stopped {
                                partial: self.reducer.into_state(),
                            });
                        }
                        Err(source) => {
                            let partial = self.reducer.state().clone();
                            if let Err(execution_source) = recover_or_release_budget(
                                &mut self.owned_lease,
                                &*self.recovery,
                                OperationKind::Completion,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Execution {
                                    source: execution_source,
                                    partial,
                                });
                            }
                            return Err(CollectError::Handler {
                                source,
                                partial: self.reducer.into_state(),
                            });
                        }
                    }
                }
                Err(source) => {
                    let partial = self.reducer.state().clone();
                    if let Err(execution_source) = recover_or_release_budget(
                        &mut self.owned_lease,
                        &*self.recovery,
                        OperationKind::Completion,
                        self.reducer.state().request_id.as_deref(),
                    )
                    .await
                    {
                        return Err(CollectError::Execution {
                            source: execution_source,
                            partial,
                        });
                    }
                    return Err(CollectError::Execution {
                        source,
                        partial: self.reducer.into_state(),
                    });
                }
            }
        }

        let partial = self.reducer.state().clone();
        if let Err(source) = recover_or_release_budget(
            &mut self.owned_lease,
            &*self.recovery,
            OperationKind::Completion,
            self.reducer.state().request_id.as_deref(),
        )
        .await
        {
            return Err(CollectError::Execution { source, partial });
        }
        Err(CollectError::UnexpectedEof {
            partial: self.reducer.into_state(),
        })
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
    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> StructuredCompletionEventStream<O> {
        let Self { stream, .. } = self;
        stream
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
        while let Some(item) = self.stream.next().instrument(self.span.clone()).await {
            match item {
                Ok(event) => {
                    if let Err(source) = self.reducer.apply(&event) {
                        return Err(CollectError::Reduction {
                            source,
                            partial: self.reducer.state().clone(),
                        });
                    }
                    record_request_id(&self.span, self.reducer.state().request_id.as_deref());
                    if let Some(usage) = completed_usage_from_structured_completion(&event) {
                        if let Err(source) = finalize_budget(
                            &mut self.owned_lease,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            usage,
                        ) {
                            return Err(CollectError::Execution {
                                source,
                                partial: self.reducer.state().clone(),
                            });
                        }
                        if let Err(source) = self.call_handler(&mut handler, &event).await {
                            return Err(CollectError::Handler {
                                source,
                                partial: self.reducer.state().clone(),
                            });
                        }
                        let partial = self.reducer.state().clone();
                        return self
                            .reducer
                            .into_result()
                            .map_err(|source| CollectError::Reduction { source, partial });
                    }

                    match self.call_handler(&mut handler, &event).await {
                        Ok(HandlerDirective::Continue) => {}
                        Ok(HandlerDirective::Stop) => {
                            let partial = self.reducer.state().clone();
                            if let Err(source) = recover_or_release_budget(
                                &mut self.owned_lease,
                                &*self.recovery,
                                OperationKind::StructuredCompletion,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Execution { source, partial });
                            }
                            return Err(CollectError::Stopped {
                                partial: self.reducer.into_state(),
                            });
                        }
                        Err(source) => {
                            let partial = self.reducer.state().clone();
                            if let Err(execution_source) = recover_or_release_budget(
                                &mut self.owned_lease,
                                &*self.recovery,
                                OperationKind::StructuredCompletion,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Execution {
                                    source: execution_source,
                                    partial,
                                });
                            }
                            return Err(CollectError::Handler {
                                source,
                                partial: self.reducer.into_state(),
                            });
                        }
                    }
                }
                Err(source) => {
                    let partial = self.reducer.state().clone();
                    if let Err(execution_source) = recover_or_release_budget(
                        &mut self.owned_lease,
                        &*self.recovery,
                        OperationKind::StructuredCompletion,
                        self.reducer.state().request_id.as_deref(),
                    )
                    .await
                    {
                        return Err(CollectError::Execution {
                            source: execution_source,
                            partial,
                        });
                    }
                    return Err(CollectError::Execution {
                        source,
                        partial: self.reducer.into_state(),
                    });
                }
            }
        }

        let partial = self.reducer.state().clone();
        if let Err(source) = recover_or_release_budget(
            &mut self.owned_lease,
            &*self.recovery,
            OperationKind::StructuredCompletion,
            self.reducer.state().request_id.as_deref(),
        )
        .await
        {
            return Err(CollectError::Execution { source, partial });
        }
        Err(CollectError::UnexpectedEof {
            partial: self.reducer.into_state(),
        })
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
    if let ToolRequirement::Specific(ref selector) = requirement
        && let ToolAvailability::Only(ref only) = available
        && !only.contains(selector)
    {
        return Err(AgentError::InvalidToolConstraints {
            tool: selector.name().to_string(),
        });
    }

    let tool_defs = match &available {
        ToolAvailability::All => T::definitions().iter().collect::<Vec<_>>(),
        ToolAvailability::Only(selectors) => T::definitions_for(selectors.iter().copied()),
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
    match availability {
        ToolAvailability::All => true,
        ToolAvailability::Only(selectors) => selectors.iter().any(|s| s.name() == name),
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
                let tool_call = T::parse_tool_call(metadata)?;
                Ok(TextTurnEventWithTools::ToolCallReady(tool_call))
            } else {
                Ok(TextTurnEventWithTools::InvalidToolCall(metadata))
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
                let tool_call = T::parse_tool_call(metadata)?;
                Ok(StructuredTurnEventWithTools::ToolCallReady(tool_call))
            } else {
                Ok(StructuredTurnEventWithTools::InvalidToolCall(metadata))
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

fn record_budget_usage(owned_lease: &mut OwnedLease, usage: Usage) -> Result<(), AgentError> {
    if let Some(lease) = owned_lease.lease.as_ref().cloned() {
        owned_lease.budget.record_used(lease, usage)?;
        owned_lease.lease = None;
    }
    Ok(())
}

async fn recover_or_release_budget(
    owned_lease: &mut OwnedLease,
    recovery: &dyn UsageRecoveryAdapter,
    kind: OperationKind,
    request_id: Option<&str>,
) -> Result<(), AgentError> {
    let recovered_usage = if let Some(request_id) = request_id {
        match recovery.recover_usage(kind, request_id).await {
            Ok(Some(usage)) => usage,
            Ok(None) => Usage::zero(),
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    kind = ?kind,
                    request_id,
                    "failed to recover usage; releasing reserved budget with zero usage"
                );
                Usage::zero()
            }
        }
    } else {
        Usage::zero()
    };

    record_budget_usage(owned_lease, recovered_usage)
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

fn record_request_id(span: &Span, request_id: Option<&str>) {
    if let Some(request_id) = request_id {
        span.record("request_id", field::display(request_id));
    }
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
