use std::{convert::Infallible, ops::Deref, sync::Arc};

use futures::StreamExt;
use thiserror::Error;
use tracing::{Instrument, Span, field};

use lutum_protocol::{
    AgentError, CommittedTurn,
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
        StructuredCompletionEventStream, StructuredCompletionRequest, StructuredTurn,
        StructuredTurnEvent, StructuredTurnEventStream, TextTurn, TextTurnEvent,
        TextTurnEventStream, TurnAdapter, TurnConfig, UsageRecoveryAdapter,
    },
    reducer::{
        CompletionReducer, CompletionReductionError, CompletionTurnResult, CompletionTurnState,
        StructuredCompletionReducer, StructuredCompletionReductionError,
        StructuredCompletionResult, StructuredCompletionState, StructuredTurnReducer,
        StructuredTurnReductionError, StructuredTurnResult, StructuredTurnState, TextTurnReducer,
        TextTurnReductionError, TextTurnResult, TextTurnState,
    },
    structured::StructuredOutput,
    toolset::{ToolSelector, Toolset},
};

use lutum_protocol::hooks::HookRegistry;

pub type ContextError = AgentError;

#[derive(Debug, Error)]
#[error("completion adapter is not configured; use Context::from_parts(...) to provide one")]
struct MissingCompletionAdapter;

#[derive(Clone, Default)]
struct UnsupportedCompletionAdapter;

#[async_trait::async_trait]
impl CompletionAdapter for UnsupportedCompletionAdapter {
    async fn completion(
        &self,
        _request: CompletionRequest,
        _extensions: &RequestExtensions,
        _hooks: &HookRegistry,
    ) -> Result<CompletionEventStream, AgentError> {
        Err(AgentError::other(MissingCompletionAdapter))
    }

    async fn structured_completion(
        &self,
        _request: AdapterStructuredCompletionRequest,
        _extensions: &RequestExtensions,
        _hooks: &HookRegistry,
    ) -> Result<ErasedStructuredCompletionEventStream, AgentError> {
        Err(AgentError::other(MissingCompletionAdapter))
    }
}

#[derive(Clone)]
pub struct Context {
    budget: Arc<dyn BudgetManager>,
    turns: Arc<dyn TurnAdapter>,
    completion: Arc<dyn CompletionAdapter>,
    recovery: Arc<dyn UsageRecoveryAdapter>,
    hooks: Arc<HookRegistry>,
}

impl Context {
    pub fn new<T>(adapter: Arc<T>, budget: impl BudgetManager + 'static) -> Self
    where
        T: TurnAdapter + UsageRecoveryAdapter + 'static,
    {
        Self::with_hooks(adapter, budget, HookRegistry::new())
    }

    pub fn with_hooks<T>(
        adapter: Arc<T>,
        budget: impl BudgetManager + 'static,
        hooks: HookRegistry,
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
        Self::from_parts_with_hooks(turns, completion, recovery, budget, HookRegistry::new())
    }

    pub fn from_parts_with_hooks(
        turns: Arc<dyn TurnAdapter>,
        completion: Arc<dyn CompletionAdapter>,
        recovery: Arc<dyn UsageRecoveryAdapter>,
        budget: impl BudgetManager + 'static,
        hooks: HookRegistry,
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

    pub fn hooks(&self) -> &HookRegistry {
        self.hooks.as_ref()
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
        if let Some(lease) = self.lease.take() {
            let _ = self.budget.record_used(lease, Usage::zero());
        }
    }
}

pub struct PendingTextTurn<T>
where
    T: Toolset,
{
    extensions: Arc<RequestExtensions>,
    owned_lease: OwnedLease,
    recovery: Arc<dyn UsageRecoveryAdapter>,
    span: Span,
    stream: TextTurnEventStream<T>,
    reducer: TextTurnReducer<T>,
}

pub struct PendingStructuredTurn<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    extensions: Arc<RequestExtensions>,
    owned_lease: OwnedLease,
    recovery: Arc<dyn UsageRecoveryAdapter>,
    span: Span,
    stream: StructuredTurnEventStream<T, O>,
    reducer: StructuredTurnReducer<T, O>,
}

#[derive(Clone, Debug)]
pub struct StructuredTurnPartial<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    pub state: StructuredTurnState<T, O>,
    pub committed_turn: Option<CommittedTurn>,
}

impl<T, O> StructuredTurnPartial<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    fn from_state(state: StructuredTurnState<T, O>) -> Self {
        let committed_turn = state.committed_turn.clone();
        Self {
            state,
            committed_turn,
        }
    }

    fn with_committed_turn(mut self, committed_turn: CommittedTurn) -> Self {
        self.committed_turn = Some(committed_turn);
        self
    }
}

impl<T, O> Deref for StructuredTurnPartial<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    type Target = StructuredTurnState<T, O>;

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

impl Context {
    pub async fn text_turn<T>(
        &self,
        extensions: RequestExtensions,
        input: ModelInput,
        turn: TextTurn<T>,
        estimate: UsageEstimate,
    ) -> Result<PendingTextTurn<T>, ContextError>
    where
        T: Toolset,
    {
        input.validate()?;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, turn.config.budget)?;
        let extensions = Arc::new(extensions);
        let span = turn_span("text_turn", estimate);
        let stream = match self
            .turns
            .text_turn(
                input,
                erase_text_turn(turn, Arc::clone(&extensions))?,
                self.hooks(),
            )
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
            stream: map_text_stream::<T>(stream),
            reducer: TextTurnReducer::new(),
        })
    }

    pub async fn structured_turn<T, O>(
        &self,
        extensions: RequestExtensions,
        input: ModelInput,
        turn: StructuredTurn<T, O>,
        estimate: UsageEstimate,
    ) -> Result<PendingStructuredTurn<T, O>, ContextError>
    where
        T: Toolset,
        O: StructuredOutput,
    {
        input.validate()?;
        let lease = self
            .budget
            .reserve(&extensions, &estimate, turn.config.budget)?;
        let extensions = Arc::new(extensions);
        let span = turn_span("structured_turn", estimate);
        let stream = match self
            .turns
            .structured_turn(
                input,
                erase_structured_turn(turn, Arc::clone(&extensions))?,
                self.hooks(),
            )
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
            stream: map_structured_stream::<T, O>(stream),
            reducer: StructuredTurnReducer::new(),
        })
    }

    pub async fn completion(
        &self,
        extensions: RequestExtensions,
        request: CompletionRequest,
        estimate: UsageEstimate,
    ) -> Result<PendingCompletion, ContextError> {
        let lease = self
            .budget
            .reserve(&extensions, &estimate, request.budget)?;
        let span = turn_span("completion", estimate);
        let stream = match self
            .completion
            .completion(request, &extensions, self.hooks())
            .await
        {
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

    pub async fn structured_completion<O>(
        &self,
        extensions: RequestExtensions,
        request: StructuredCompletionRequest<O>,
        estimate: UsageEstimate,
    ) -> Result<PendingStructuredCompletion<O>, ContextError>
    where
        O: StructuredOutput,
    {
        let lease = self
            .budget
            .reserve(&extensions, &estimate, request.budget)?;
        let span = turn_span("structured_completion", estimate);
        let stream = match self
            .completion
            .structured_completion(
                erase_structured_completion_request(request)?,
                &extensions,
                self.hooks(),
            )
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

impl<T> PendingTextTurn<T>
where
    T: Toolset,
{
    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> TextTurnEventStream<T> {
        let Self { stream, .. } = self;
        stream
    }

    pub async fn collect<H>(
        mut self,
        mut handler: H,
    ) -> Result<TextTurnResult<T>, CollectError<H::Error, TextTurnReductionError, TextTurnState<T>>>
    where
        H: EventHandler<TextTurnEvent<T>, TextTurnState<T>>,
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

    pub async fn collect_noop(
        self,
    ) -> Result<TextTurnResult<T>, CollectError<Infallible, TextTurnReductionError, TextTurnState<T>>>
    {
        self.collect(NoopHandler).await
    }

    async fn call_handler<H>(
        &self,
        handler: &mut H,
        event: &TextTurnEvent<T>,
    ) -> Result<HandlerDirective, H::Error>
    where
        H: EventHandler<TextTurnEvent<T>, TextTurnState<T>>,
    {
        let cx = HandlerContext {
            extensions: self.extensions.as_ref(),
            state: self.reducer.state(),
            remaining_budget: self.owned_lease.budget.remaining(self.extensions.as_ref()),
        };
        handler.on_event(event, &cx).await
    }
}

impl<T, O> PendingStructuredTurn<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    /// Returns the raw typed event stream.
    ///
    /// Releasing this wrapper commits zero usage and frees any reserved budget.
    pub fn into_stream(self) -> StructuredTurnEventStream<T, O> {
        let Self { stream, .. } = self;
        stream
    }

    pub async fn collect<H>(
        mut self,
        mut handler: H,
    ) -> Result<
        StructuredTurnResult<T, O>,
        CollectError<H::Error, StructuredTurnReductionError, StructuredTurnPartial<T, O>>,
    >
    where
        H: EventHandler<StructuredTurnEvent<T, O>, StructuredTurnState<T, O>>,
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

    pub async fn collect_noop(
        self,
    ) -> Result<
        StructuredTurnResult<T, O>,
        CollectError<Infallible, StructuredTurnReductionError, StructuredTurnPartial<T, O>>,
    > {
        self.collect(NoopHandler).await
    }

    async fn call_handler<H>(
        &self,
        handler: &mut H,
        event: &StructuredTurnEvent<T, O>,
    ) -> Result<HandlerDirective, H::Error>
    where
        H: EventHandler<StructuredTurnEvent<T, O>, StructuredTurnState<T, O>>,
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
    pub async fn collect<H>(
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

    pub async fn collect_noop(
        self,
    ) -> Result<
        CompletionTurnResult,
        CollectError<Infallible, CompletionReductionError, CompletionTurnState>,
    > {
        self.collect(NoopHandler).await
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
    pub async fn collect<H>(
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

    pub async fn collect_noop(
        self,
    ) -> Result<
        StructuredCompletionResult<O>,
        CollectError<Infallible, StructuredCompletionReductionError, StructuredCompletionState<O>>,
    > {
        self.collect(NoopHandler).await
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
    turn: TextTurn<T>,
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
    turn: StructuredTurn<T, O>,
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
        model: request.model,
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
    let selected = config.tools.selected();
    let selected_names = selected.as_ref().map(|selectors| {
        selectors
            .iter()
            .map(|selector| selector.name())
            .collect::<Vec<_>>()
    });
    let tool_choice = if config.tools.requires_tools() {
        if let Some(names) = selected_names.as_ref() {
            if names.len() == 1 {
                AdapterToolChoice::Specific(names[0].to_string())
            } else {
                AdapterToolChoice::Required
            }
        } else {
            AdapterToolChoice::Required
        }
    } else if config.tools.uses_tools() {
        AdapterToolChoice::Auto
    } else {
        AdapterToolChoice::None
    };
    let tool_defs = selected
        .map(|selectors| T::definitions_for(selectors.iter().copied()))
        .unwrap_or_else(|| T::definitions().iter().collect::<Vec<_>>());
    let tools = tool_defs
        .into_iter()
        .map(|tool| {
            Ok(AdapterToolDefinition {
                name: tool.name.to_string(),
                description: tool.description.to_string(),
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

fn map_text_stream<T>(stream: ErasedTextTurnEventStream) -> TextTurnEventStream<T>
where
    T: Toolset,
{
    Box::pin(stream.map(|item| item.and_then(map_text_event::<T>)))
}

fn map_structured_stream<T, O>(
    stream: ErasedStructuredTurnEventStream,
) -> StructuredTurnEventStream<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    Box::pin(stream.map(|item| item.and_then(map_structured_event::<T, O>)))
}

fn map_structured_completion_stream<O>(
    stream: ErasedStructuredCompletionEventStream,
) -> StructuredCompletionEventStream<O>
where
    O: StructuredOutput,
{
    Box::pin(stream.map(|item| item.and_then(map_structured_completion_event::<O>)))
}

fn map_text_event<T>(event: ErasedTextTurnEvent) -> Result<TextTurnEvent<T>, AgentError>
where
    T: Toolset,
{
    match event {
        ErasedTextTurnEvent::Started { request_id, model } => {
            Ok(TextTurnEvent::Started { request_id, model })
        }
        ErasedTextTurnEvent::TextDelta { delta } => Ok(TextTurnEvent::TextDelta { delta }),
        ErasedTextTurnEvent::ReasoningDelta { delta } => {
            Ok(TextTurnEvent::ReasoningDelta { delta })
        }
        ErasedTextTurnEvent::RefusalDelta { delta } => Ok(TextTurnEvent::RefusalDelta { delta }),
        ErasedTextTurnEvent::ToolCallChunk {
            id,
            name,
            arguments_json_delta,
        } => Ok(TextTurnEvent::ToolCallChunk {
            id,
            name,
            arguments_json_delta,
        }),
        ErasedTextTurnEvent::ToolCallReady(metadata) => {
            let tool_call = T::parse_tool_call(metadata)?;
            Ok(TextTurnEvent::ToolCallReady(tool_call))
        }
        ErasedTextTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            committed_turn,
        } => Ok(TextTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            committed_turn,
        }),
    }
}

fn map_structured_event<T, O>(
    event: ErasedStructuredTurnEvent,
) -> Result<StructuredTurnEvent<T, O>, AgentError>
where
    T: Toolset,
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
        ErasedStructuredTurnEvent::ToolCallChunk {
            id,
            name,
            arguments_json_delta,
        } => Ok(StructuredTurnEvent::ToolCallChunk {
            id,
            name,
            arguments_json_delta,
        }),
        ErasedStructuredTurnEvent::ToolCallReady(metadata) => {
            let tool_call = T::parse_tool_call(metadata)?;
            Ok(StructuredTurnEvent::ToolCallReady(tool_call))
        }
        ErasedStructuredTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
            committed_turn,
        } => Ok(StructuredTurnEvent::Completed {
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
    if let Some(lease) = owned_lease.lease.take() {
        owned_lease.budget.record_used(lease, usage)?;
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
        recovery
            .recover_usage(kind, request_id)
            .await
            .ok()
            .flatten()
            .unwrap_or_else(Usage::zero)
    } else {
        Usage::zero()
    };

    record_budget_usage(owned_lease, recovered_usage)
}

fn turn_span(kind: &'static str, estimate: UsageEstimate) -> Span {
    tracing::info_span!(
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

fn completed_usage_from_text<T>(event: &TextTurnEvent<T>) -> Option<Usage>
where
    T: Toolset,
{
    match event {
        TextTurnEvent::Completed { usage, .. } => Some(*usage),
        _ => None,
    }
}

fn completed_usage_from_structured<T, O>(event: &StructuredTurnEvent<T, O>) -> Option<Usage>
where
    T: Toolset,
    O: StructuredOutput,
{
    match event {
        StructuredTurnEvent::Completed { usage, .. } => Some(*usage),
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
    assert_send_sync::<PendingTextTurn<lutum_protocol::toolset::NoTools>>();
    assert_send_sync::<PendingStructuredTurn<lutum_protocol::toolset::NoTools, ()>>();
    assert_send_sync::<PendingStructuredCompletion<()>>();
    assert_send_sync::<PendingCompletion>();
}
