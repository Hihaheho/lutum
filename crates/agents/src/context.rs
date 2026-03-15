use std::{convert::Infallible, sync::Arc};

use futures::StreamExt;
use thiserror::Error;
use tracing::{Instrument, Span, field};

use agents_protocol::{
    budget::{BudgetLease, BudgetManager, Remaining, Usage, UsageEstimate},
    conversation::{ModelInput, ModelInputValidationError},
    llm::{
        CompletionEvent, CompletionEventStream, CompletionRequest, LlmAdapter, StreamKind,
        StructuredTurn, StructuredTurnEvent, StructuredTurnEventStream, TextTurn, TextTurnEvent,
        TextTurnEventStream,
    },
    marker::Marker,
    reducer::{
        CompletionReducer, CompletionReductionError, CompletionTurnResult, CompletionTurnState,
        StructuredTurnReducer, StructuredTurnReductionError, StructuredTurnResult,
        StructuredTurnState, TextTurnReducer, TextTurnReductionError, TextTurnResult,
        TextTurnState,
    },
    structured::StructuredOutput,
    toolset::Toolset,
};

#[derive(Clone)]
pub struct Context<M, B, L> {
    budget: Arc<B>,
    adapter: Arc<L>,
    _marker: std::marker::PhantomData<fn() -> M>,
}

impl<M, B, L> Context<M, B, L> {
    pub fn new(budget: B, adapter: L) -> Self {
        Self {
            budget: Arc::new(budget),
            adapter: Arc::new(adapter),
            _marker: std::marker::PhantomData,
        }
    }

    pub fn budget(&self) -> &B {
        self.budget.as_ref()
    }

    pub fn adapter(&self) -> &L {
        self.adapter.as_ref()
    }
}

#[derive(Debug, Error)]
pub enum ContextError<BudgetError, AdapterError> {
    #[error("invalid model input: {0}")]
    InvalidModelInput(#[from] ModelInputValidationError),
    #[error("budget error: {0}")]
    Budget(#[source] BudgetError),
    #[error("adapter error: {0}")]
    Adapter(#[source] AdapterError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HandlerDirective {
    Continue,
    Stop,
}

pub struct HandlerContext<'a, M, S> {
    marker: &'a M,
    state: &'a S,
    remaining_budget: Remaining,
}

impl<'a, M, S> HandlerContext<'a, M, S> {
    pub fn marker(&self) -> &M {
        self.marker
    }

    pub fn state(&self) -> &S {
        self.state
    }

    pub fn remaining_budget(&self) -> Remaining {
        self.remaining_budget
    }
}

#[async_trait::async_trait]
pub trait EventHandler<E, M, S>: Send {
    type Error;

    async fn on_event(
        &mut self,
        event: &E,
        cx: &HandlerContext<M, S>,
    ) -> Result<HandlerDirective, Self::Error>;
}

#[async_trait::async_trait]
impl<E, M, S, F, Err> EventHandler<E, M, S> for F
where
    F: Send + for<'a> FnMut(&'a E, &'a HandlerContext<'a, M, S>) -> Result<HandlerDirective, Err>,
    E: Sync,
    M: Sync,
    S: Sync,
{
    type Error = Err;

    async fn on_event(
        &mut self,
        event: &E,
        cx: &HandlerContext<M, S>,
    ) -> Result<HandlerDirective, Self::Error> {
        (self)(event, cx)
    }
}

#[derive(Debug, Error)]
pub enum CollectError<BudgetError, AdapterError, HandlerError, ReductionError, Partial> {
    #[error("budget error: {source}")]
    Budget {
        #[source]
        source: BudgetError,
        partial: Partial,
    },
    #[error("adapter error: {source}")]
    Adapter {
        #[source]
        source: AdapterError,
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

pub struct PendingTextTurn<M, B, L, T>
where
    T: Toolset,
    L: LlmAdapter,
{
    marker: M,
    budget: Arc<B>,
    adapter: Arc<L>,
    span: Span,
    lease: Option<BudgetLease<M>>,
    stream: TextTurnEventStream<T, L::Error>,
    reducer: TextTurnReducer<T>,
}

pub struct PendingStructuredTurn<M, B, L, T, O>
where
    T: Toolset,
    O: StructuredOutput,
    L: LlmAdapter,
{
    marker: M,
    budget: Arc<B>,
    adapter: Arc<L>,
    span: Span,
    lease: Option<BudgetLease<M>>,
    stream: StructuredTurnEventStream<T, O, L::Error>,
    reducer: StructuredTurnReducer<T, O>,
}

pub struct PendingCompletion<M, B, L>
where
    L: LlmAdapter,
{
    marker: M,
    budget: Arc<B>,
    adapter: Arc<L>,
    span: Span,
    lease: Option<BudgetLease<M>>,
    stream: CompletionEventStream<L::Error>,
    reducer: CompletionReducer,
}

impl<M, B, L> Context<M, B, L>
where
    M: Marker,
    B: BudgetManager<M>,
    L: LlmAdapter,
{
    pub async fn responses_text<T>(
        &self,
        marker: M,
        input: ModelInput,
        turn: TextTurn<T>,
        estimate: UsageEstimate,
    ) -> Result<PendingTextTurn<M, B, L, T>, ContextError<B::Error, L::Error>>
    where
        T: Toolset,
    {
        input.validate()?;
        let lease = self
            .budget
            .reserve(&marker, &estimate, turn.config.budget)
            .map_err(ContextError::Budget)?;
        let span = turn_span(
            marker.span_name().into_owned(),
            "responses_text",
            turn.config.model.as_ref(),
            estimate,
        );
        let stream = self
            .adapter
            .responses_text(input, turn)
            .await
            .map_err(ContextError::Adapter)?;
        Ok(PendingTextTurn {
            marker,
            budget: Arc::clone(&self.budget),
            adapter: Arc::clone(&self.adapter),
            span,
            lease: Some(lease),
            stream,
            reducer: TextTurnReducer::new(),
        })
    }

    pub async fn responses_structured<T, O>(
        &self,
        marker: M,
        input: ModelInput,
        turn: StructuredTurn<T, O>,
        estimate: UsageEstimate,
    ) -> Result<PendingStructuredTurn<M, B, L, T, O>, ContextError<B::Error, L::Error>>
    where
        T: Toolset,
        O: StructuredOutput,
    {
        input.validate()?;
        let lease = self
            .budget
            .reserve(&marker, &estimate, turn.config.budget)
            .map_err(ContextError::Budget)?;
        let span = turn_span(
            marker.span_name().into_owned(),
            "responses_structured",
            turn.config.model.as_ref(),
            estimate,
        );
        let stream = self
            .adapter
            .responses_structured(input, turn)
            .await
            .map_err(ContextError::Adapter)?;
        Ok(PendingStructuredTurn {
            marker,
            budget: Arc::clone(&self.budget),
            adapter: Arc::clone(&self.adapter),
            span,
            lease: Some(lease),
            stream,
            reducer: StructuredTurnReducer::new(),
        })
    }

    pub async fn completion(
        &self,
        marker: M,
        request: CompletionRequest,
        estimate: UsageEstimate,
    ) -> Result<PendingCompletion<M, B, L>, ContextError<B::Error, L::Error>> {
        let lease = self
            .budget
            .reserve(&marker, &estimate, request.budget)
            .map_err(ContextError::Budget)?;
        let span = turn_span(
            marker.span_name().into_owned(),
            "completion",
            request.model.as_ref(),
            estimate,
        );
        let stream = self
            .adapter
            .completion(request)
            .await
            .map_err(ContextError::Adapter)?;
        Ok(PendingCompletion {
            marker,
            budget: Arc::clone(&self.budget),
            adapter: Arc::clone(&self.adapter),
            span,
            lease: Some(lease),
            stream,
            reducer: CompletionReducer::new(),
        })
    }
}

impl<M, B, L, T> PendingTextTurn<M, B, L, T>
where
    M: Marker,
    B: BudgetManager<M>,
    L: LlmAdapter,
    T: Toolset,
{
    pub async fn collect<H>(
        mut self,
        mut handler: H,
    ) -> Result<
        TextTurnResult<T>,
        CollectError<B::Error, L::Error, H::Error, TextTurnReductionError, TextTurnState<T>>,
    >
    where
        H: EventHandler<TextTurnEvent<T>, M, TextTurnState<T>>,
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
                            &mut self.lease,
                            &*self.budget,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            usage,
                        ) {
                            return Err(CollectError::Budget {
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
                                &mut self.lease,
                                &*self.budget,
                                &*self.adapter,
                                StreamKind::ResponsesText,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Budget { source, partial });
                            }
                            return Err(CollectError::Stopped {
                                partial: self.reducer.into_state(),
                            });
                        }
                        Err(source) => {
                            let partial = self.reducer.state().clone();
                            if let Err(budget_source) = recover_or_release_budget(
                                &mut self.lease,
                                &*self.budget,
                                &*self.adapter,
                                StreamKind::ResponsesText,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Budget {
                                    source: budget_source,
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
                    if let Err(budget_source) = recover_or_release_budget(
                        &mut self.lease,
                        &*self.budget,
                        &*self.adapter,
                        StreamKind::ResponsesText,
                        self.reducer.state().request_id.as_deref(),
                    )
                    .await
                    {
                        return Err(CollectError::Budget {
                            source: budget_source,
                            partial,
                        });
                    }
                    return Err(CollectError::Adapter {
                        source,
                        partial: self.reducer.into_state(),
                    });
                }
            }
        }

        let partial = self.reducer.state().clone();
        if let Err(source) = recover_or_release_budget(
            &mut self.lease,
            &*self.budget,
            &*self.adapter,
            StreamKind::ResponsesText,
            self.reducer.state().request_id.as_deref(),
        )
        .await
        {
            return Err(CollectError::Budget { source, partial });
        }
        Err(CollectError::UnexpectedEof {
            partial: self.reducer.into_state(),
        })
    }

    pub async fn collect_noop(
        self,
    ) -> Result<
        TextTurnResult<T>,
        CollectError<B::Error, L::Error, Infallible, TextTurnReductionError, TextTurnState<T>>,
    > {
        self.collect(NoopHandler).await
    }

    async fn call_handler<H>(
        &self,
        handler: &mut H,
        event: &TextTurnEvent<T>,
    ) -> Result<HandlerDirective, H::Error>
    where
        H: EventHandler<TextTurnEvent<T>, M, TextTurnState<T>>,
    {
        let cx = HandlerContext {
            marker: &self.marker,
            state: self.reducer.state(),
            remaining_budget: self.budget.remaining(&self.marker),
        };
        handler.on_event(event, &cx).await
    }
}

impl<M, B, L, T, O> PendingStructuredTurn<M, B, L, T, O>
where
    M: Marker,
    B: BudgetManager<M>,
    L: LlmAdapter,
    T: Toolset,
    O: StructuredOutput,
{
    pub async fn collect<H>(
        mut self,
        mut handler: H,
    ) -> Result<
        StructuredTurnResult<T, O>,
        CollectError<
            B::Error,
            L::Error,
            H::Error,
            StructuredTurnReductionError,
            StructuredTurnState<T, O>,
        >,
    >
    where
        H: EventHandler<StructuredTurnEvent<T, O>, M, StructuredTurnState<T, O>>,
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
                    if let Some(usage) = completed_usage_from_structured(&event) {
                        if let Err(source) = finalize_budget(
                            &mut self.lease,
                            &*self.budget,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            usage,
                        ) {
                            return Err(CollectError::Budget {
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
                                &mut self.lease,
                                &*self.budget,
                                &*self.adapter,
                                StreamKind::ResponsesStructured,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Budget { source, partial });
                            }
                            return Err(CollectError::Stopped {
                                partial: self.reducer.into_state(),
                            });
                        }
                        Err(source) => {
                            let partial = self.reducer.state().clone();
                            if let Err(budget_source) = recover_or_release_budget(
                                &mut self.lease,
                                &*self.budget,
                                &*self.adapter,
                                StreamKind::ResponsesStructured,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Budget {
                                    source: budget_source,
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
                    if let Err(budget_source) = recover_or_release_budget(
                        &mut self.lease,
                        &*self.budget,
                        &*self.adapter,
                        StreamKind::ResponsesStructured,
                        self.reducer.state().request_id.as_deref(),
                    )
                    .await
                    {
                        return Err(CollectError::Budget {
                            source: budget_source,
                            partial,
                        });
                    }
                    return Err(CollectError::Adapter {
                        source,
                        partial: self.reducer.into_state(),
                    });
                }
            }
        }

        let partial = self.reducer.state().clone();
        if let Err(source) = recover_or_release_budget(
            &mut self.lease,
            &*self.budget,
            &*self.adapter,
            StreamKind::ResponsesStructured,
            self.reducer.state().request_id.as_deref(),
        )
        .await
        {
            return Err(CollectError::Budget { source, partial });
        }
        Err(CollectError::UnexpectedEof {
            partial: self.reducer.into_state(),
        })
    }

    pub async fn collect_noop(
        self,
    ) -> Result<
        StructuredTurnResult<T, O>,
        CollectError<
            B::Error,
            L::Error,
            Infallible,
            StructuredTurnReductionError,
            StructuredTurnState<T, O>,
        >,
    > {
        self.collect(NoopHandler).await
    }

    async fn call_handler<H>(
        &self,
        handler: &mut H,
        event: &StructuredTurnEvent<T, O>,
    ) -> Result<HandlerDirective, H::Error>
    where
        H: EventHandler<StructuredTurnEvent<T, O>, M, StructuredTurnState<T, O>>,
    {
        let cx = HandlerContext {
            marker: &self.marker,
            state: self.reducer.state(),
            remaining_budget: self.budget.remaining(&self.marker),
        };
        handler.on_event(event, &cx).await
    }
}

impl<M, B, L> PendingCompletion<M, B, L>
where
    M: Marker,
    B: BudgetManager<M>,
    L: LlmAdapter,
{
    pub async fn collect<H>(
        mut self,
        mut handler: H,
    ) -> Result<
        CompletionTurnResult,
        CollectError<B::Error, L::Error, H::Error, CompletionReductionError, CompletionTurnState>,
    >
    where
        H: EventHandler<CompletionEvent, M, CompletionTurnState>,
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
                            &mut self.lease,
                            &*self.budget,
                            &self.span,
                            self.reducer.state().request_id.as_deref(),
                            usage,
                        ) {
                            return Err(CollectError::Budget {
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
                                &mut self.lease,
                                &*self.budget,
                                &*self.adapter,
                                StreamKind::Completion,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Budget { source, partial });
                            }
                            return Err(CollectError::Stopped {
                                partial: self.reducer.into_state(),
                            });
                        }
                        Err(source) => {
                            let partial = self.reducer.state().clone();
                            if let Err(budget_source) = recover_or_release_budget(
                                &mut self.lease,
                                &*self.budget,
                                &*self.adapter,
                                StreamKind::Completion,
                                self.reducer.state().request_id.as_deref(),
                            )
                            .await
                            {
                                return Err(CollectError::Budget {
                                    source: budget_source,
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
                    if let Err(budget_source) = recover_or_release_budget(
                        &mut self.lease,
                        &*self.budget,
                        &*self.adapter,
                        StreamKind::Completion,
                        self.reducer.state().request_id.as_deref(),
                    )
                    .await
                    {
                        return Err(CollectError::Budget {
                            source: budget_source,
                            partial,
                        });
                    }
                    return Err(CollectError::Adapter {
                        source,
                        partial: self.reducer.into_state(),
                    });
                }
            }
        }

        let partial = self.reducer.state().clone();
        if let Err(source) = recover_or_release_budget(
            &mut self.lease,
            &*self.budget,
            &*self.adapter,
            StreamKind::Completion,
            self.reducer.state().request_id.as_deref(),
        )
        .await
        {
            return Err(CollectError::Budget { source, partial });
        }
        Err(CollectError::UnexpectedEof {
            partial: self.reducer.into_state(),
        })
    }

    pub async fn collect_noop(
        self,
    ) -> Result<
        CompletionTurnResult,
        CollectError<B::Error, L::Error, Infallible, CompletionReductionError, CompletionTurnState>,
    > {
        self.collect(NoopHandler).await
    }

    async fn call_handler<H>(
        &self,
        handler: &mut H,
        event: &CompletionEvent,
    ) -> Result<HandlerDirective, H::Error>
    where
        H: EventHandler<CompletionEvent, M, CompletionTurnState>,
    {
        let cx = HandlerContext {
            marker: &self.marker,
            state: self.reducer.state(),
            remaining_budget: self.budget.remaining(&self.marker),
        };
        handler.on_event(event, &cx).await
    }
}

struct NoopHandler;

#[async_trait::async_trait]
impl<E, M, S> EventHandler<E, M, S> for NoopHandler
where
    E: Send + Sync + 'static,
    M: Send + Sync + 'static,
    S: Send + Sync + 'static,
{
    type Error = Infallible;

    async fn on_event(
        &mut self,
        _event: &E,
        _cx: &HandlerContext<M, S>,
    ) -> Result<HandlerDirective, Self::Error> {
        Ok(HandlerDirective::Continue)
    }
}

fn finalize_budget<M, B>(
    lease: &mut Option<BudgetLease<M>>,
    budget: &B,
    span: &Span,
    request_id: Option<&str>,
    usage: Usage,
) -> Result<(), B::Error>
where
    B: BudgetManager<M>,
{
    if let Some(request_id) = request_id {
        span.record("request_id", field::display(request_id));
    }
    record_budget_usage(lease, budget, usage)
}

fn record_budget_usage<M, B>(
    lease: &mut Option<BudgetLease<M>>,
    budget: &B,
    usage: Usage,
) -> Result<(), B::Error>
where
    B: BudgetManager<M>,
{
    if let Some(lease) = lease.take() {
        budget.record_used(lease, usage)?;
    }
    Ok(())
}

async fn recover_or_release_budget<M, B, L>(
    lease: &mut Option<BudgetLease<M>>,
    budget: &B,
    adapter: &L,
    kind: StreamKind,
    request_id: Option<&str>,
) -> Result<(), B::Error>
where
    B: BudgetManager<M>,
    L: LlmAdapter,
{
    let recovered_usage = if let Some(request_id) = request_id {
        adapter
            .recover_usage(kind, request_id)
            .await
            .ok()
            .flatten()
            .unwrap_or_else(Usage::zero)
    } else {
        Usage::zero()
    };

    record_budget_usage(lease, budget, recovered_usage)
}

fn turn_span(marker: String, kind: &'static str, model: &str, estimate: UsageEstimate) -> Span {
    tracing::info_span!(
        "llm_turn",
        marker = %marker,
        kind = %kind,
        model = %model,
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
