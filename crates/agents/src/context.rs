use std::{convert::Infallible, sync::Arc};

use futures::StreamExt;
use thiserror::Error;
use tracing::{Instrument, Span, field};

use agents_protocol::{
    AgentError,
    budget::{BudgetLease, BudgetManager, Remaining, Usage, UsageEstimate},
    conversation::ModelInput,
    llm::{
        AdapterStructuredOutputSpec, AdapterStructuredTurn, AdapterTextTurn, AdapterToolChoice,
        AdapterToolDefinition, AdapterTurnConfig, CompletionEvent, CompletionEventStream,
        CompletionRequest, ErasedStructuredTurnEvent, ErasedStructuredTurnEventStream,
        ErasedTextTurnEvent, ErasedTextTurnEventStream, LlmAdapter, StreamKind, StructuredTurn,
        StructuredTurnEvent, StructuredTurnEventStream, TextTurn, TextTurnEvent,
        TextTurnEventStream, TurnConfig,
    },
    marker::Marker,
    reducer::{
        CompletionReducer, CompletionReductionError, CompletionTurnResult, CompletionTurnState,
        StructuredTurnReducer, StructuredTurnReductionError, StructuredTurnResult,
        StructuredTurnState, TextTurnReducer, TextTurnReductionError, TextTurnResult,
        TextTurnState,
    },
    structured::StructuredOutput,
    toolset::{ToolSelector, Toolset},
};

pub type ContextError = AgentError;

#[derive(Clone)]
pub struct Context<M> {
    budget: Arc<dyn BudgetManager<M>>,
    adapter: Arc<dyn LlmAdapter>,
}

impl<M> Context<M> {
    pub fn new<B, L>(budget: B, adapter: L) -> Self
    where
        B: BudgetManager<M>,
        L: LlmAdapter,
    {
        Self {
            budget: Arc::new(budget),
            adapter: Arc::new(adapter),
        }
    }

    pub fn budget(&self) -> &dyn BudgetManager<M> {
        self.budget.as_ref()
    }

    pub fn adapter(&self) -> &dyn LlmAdapter {
        self.adapter.as_ref()
    }
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

pub struct PendingTextTurn<M, T>
where
    T: Toolset,
{
    marker: M,
    budget: Arc<dyn BudgetManager<M>>,
    adapter: Arc<dyn LlmAdapter>,
    span: Span,
    lease: Option<BudgetLease<M>>,
    stream: TextTurnEventStream<T>,
    reducer: TextTurnReducer<T>,
}

pub struct PendingStructuredTurn<M, T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    marker: M,
    budget: Arc<dyn BudgetManager<M>>,
    adapter: Arc<dyn LlmAdapter>,
    span: Span,
    lease: Option<BudgetLease<M>>,
    stream: StructuredTurnEventStream<T, O>,
    reducer: StructuredTurnReducer<T, O>,
}

pub struct PendingCompletion<M> {
    marker: M,
    budget: Arc<dyn BudgetManager<M>>,
    adapter: Arc<dyn LlmAdapter>,
    span: Span,
    lease: Option<BudgetLease<M>>,
    stream: CompletionEventStream,
    reducer: CompletionReducer,
}

impl<M> Context<M>
where
    M: Marker,
{
    pub async fn responses_text<T>(
        &self,
        marker: M,
        input: ModelInput,
        turn: TextTurn<T>,
        estimate: UsageEstimate,
    ) -> Result<PendingTextTurn<M, T>, ContextError>
    where
        T: Toolset,
    {
        input.validate()?;
        let lease = self
            .budget
            .reserve(&marker, &estimate, turn.config.budget)?;
        let span = turn_span(
            marker.span_name().into_owned(),
            "responses_text",
            turn.config.model.as_ref(),
            estimate,
        );
        let stream = self
            .adapter
            .responses_text(input, erase_text_turn(turn)?)
            .await?;
        Ok(PendingTextTurn {
            marker,
            budget: Arc::clone(&self.budget),
            adapter: Arc::clone(&self.adapter),
            span,
            lease: Some(lease),
            stream: map_text_stream::<T>(stream),
            reducer: TextTurnReducer::new(),
        })
    }

    pub async fn responses_structured<T, O>(
        &self,
        marker: M,
        input: ModelInput,
        turn: StructuredTurn<T, O>,
        estimate: UsageEstimate,
    ) -> Result<PendingStructuredTurn<M, T, O>, ContextError>
    where
        T: Toolset,
        O: StructuredOutput,
    {
        input.validate()?;
        let lease = self
            .budget
            .reserve(&marker, &estimate, turn.config.budget)?;
        let span = turn_span(
            marker.span_name().into_owned(),
            "responses_structured",
            turn.config.model.as_ref(),
            estimate,
        );
        let stream = self
            .adapter
            .responses_structured(input, erase_structured_turn(turn)?)
            .await?;
        Ok(PendingStructuredTurn {
            marker,
            budget: Arc::clone(&self.budget),
            adapter: Arc::clone(&self.adapter),
            span,
            lease: Some(lease),
            stream: map_structured_stream::<T, O>(stream),
            reducer: StructuredTurnReducer::new(),
        })
    }

    pub async fn completion(
        &self,
        marker: M,
        request: CompletionRequest,
        estimate: UsageEstimate,
    ) -> Result<PendingCompletion<M>, ContextError> {
        let lease = self.budget.reserve(&marker, &estimate, request.budget)?;
        let span = turn_span(
            marker.span_name().into_owned(),
            "completion",
            request.model.as_ref(),
            estimate,
        );
        let stream = self.adapter.completion(request).await?;
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

impl<M, T> PendingTextTurn<M, T>
where
    M: Marker,
    T: Toolset,
{
    pub async fn collect<H>(
        mut self,
        mut handler: H,
    ) -> Result<TextTurnResult<T>, CollectError<H::Error, TextTurnReductionError, TextTurnState<T>>>
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
                                &mut self.lease,
                                &*self.budget,
                                &*self.adapter,
                                StreamKind::ResponsesText,
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
                                &mut self.lease,
                                &*self.budget,
                                &*self.adapter,
                                StreamKind::ResponsesText,
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
                        &mut self.lease,
                        &*self.budget,
                        &*self.adapter,
                        StreamKind::ResponsesText,
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
            &mut self.lease,
            &*self.budget,
            &*self.adapter,
            StreamKind::ResponsesText,
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

impl<M, T, O> PendingStructuredTurn<M, T, O>
where
    M: Marker,
    T: Toolset,
    O: StructuredOutput,
{
    pub async fn collect<H>(
        mut self,
        mut handler: H,
    ) -> Result<
        StructuredTurnResult<T, O>,
        CollectError<H::Error, StructuredTurnReductionError, StructuredTurnState<T, O>>,
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
                                &mut self.lease,
                                &*self.budget,
                                &*self.adapter,
                                StreamKind::ResponsesStructured,
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
                                &mut self.lease,
                                &*self.budget,
                                &*self.adapter,
                                StreamKind::ResponsesStructured,
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
                        &mut self.lease,
                        &*self.budget,
                        &*self.adapter,
                        StreamKind::ResponsesStructured,
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
            &mut self.lease,
            &*self.budget,
            &*self.adapter,
            StreamKind::ResponsesStructured,
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
        StructuredTurnResult<T, O>,
        CollectError<Infallible, StructuredTurnReductionError, StructuredTurnState<T, O>>,
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

impl<M> PendingCompletion<M>
where
    M: Marker,
{
    pub async fn collect<H>(
        mut self,
        mut handler: H,
    ) -> Result<
        CompletionTurnResult,
        CollectError<H::Error, CompletionReductionError, CompletionTurnState>,
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
                                &mut self.lease,
                                &*self.budget,
                                &*self.adapter,
                                StreamKind::Completion,
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
                                &mut self.lease,
                                &*self.budget,
                                &*self.adapter,
                                StreamKind::Completion,
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
                        &mut self.lease,
                        &*self.budget,
                        &*self.adapter,
                        StreamKind::Completion,
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
            &mut self.lease,
            &*self.budget,
            &*self.adapter,
            StreamKind::Completion,
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

fn erase_text_turn<T>(turn: TextTurn<T>) -> Result<AdapterTextTurn, AgentError>
where
    T: Toolset,
{
    Ok(AdapterTextTurn {
        config: erase_turn_config(turn.config)?,
    })
}

fn erase_structured_turn<T, O>(
    turn: StructuredTurn<T, O>,
) -> Result<AdapterStructuredTurn, AgentError>
where
    T: Toolset,
    O: StructuredOutput,
{
    Ok(AdapterStructuredTurn {
        config: erase_turn_config(turn.config)?,
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
    let tool_choice = if config.tools.requires_tools() {
        AdapterToolChoice::Required
    } else if config.tools.uses_tools() {
        AdapterToolChoice::Auto
    } else {
        AdapterToolChoice::None
    };
    let selected = config.tools.selected().map(|selectors| {
        selectors
            .iter()
            .map(|selector| selector.name())
            .collect::<Vec<_>>()
    });
    let tools = T::definitions()
        .iter()
        .filter(|tool| {
            selected
                .as_ref()
                .is_none_or(|names: &Vec<&'static str>| names.contains(&tool.name))
        })
        .map(|tool| {
            Ok(AdapterToolDefinition {
                name: tool.name.to_string(),
                description: tool.description.to_string(),
                input_schema: serde_json::to_value(tool.input_schema())?,
            })
        })
        .collect::<Result<Vec<_>, serde_json::Error>>()?;

    Ok(AdapterTurnConfig {
        model: config.model,
        generation: config.generation,
        reasoning: config.reasoning,
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
            let name = metadata.name.as_str().to_string();
            let arguments_json = metadata.arguments.get().to_string();
            let tool_call = T::parse_tool_call(metadata, &name, &arguments_json)?;
            Ok(TextTurnEvent::ToolCallReady(tool_call))
        }
        ErasedTextTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
        } => Ok(TextTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
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
            let name = metadata.name.as_str().to_string();
            let arguments_json = metadata.arguments.get().to_string();
            let tool_call = T::parse_tool_call(metadata, &name, &arguments_json)?;
            Ok(StructuredTurnEvent::ToolCallReady(tool_call))
        }
        ErasedStructuredTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
        } => Ok(StructuredTurnEvent::Completed {
            request_id,
            finish_reason,
            usage,
        }),
    }
}

fn finalize_budget<M: 'static>(
    lease: &mut Option<BudgetLease<M>>,
    budget: &dyn BudgetManager<M>,
    span: &Span,
    request_id: Option<&str>,
    usage: Usage,
) -> Result<(), AgentError> {
    if let Some(request_id) = request_id {
        span.record("request_id", field::display(request_id));
    }
    record_budget_usage(lease, budget, usage)
}

fn record_budget_usage<M: 'static>(
    lease: &mut Option<BudgetLease<M>>,
    budget: &dyn BudgetManager<M>,
    usage: Usage,
) -> Result<(), AgentError> {
    if let Some(lease) = lease.take() {
        budget.record_used(lease, usage)?;
    }
    Ok(())
}

async fn recover_or_release_budget<M: 'static>(
    lease: &mut Option<BudgetLease<M>>,
    budget: &dyn BudgetManager<M>,
    adapter: &dyn LlmAdapter,
    kind: StreamKind,
    request_id: Option<&str>,
) -> Result<(), AgentError> {
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
