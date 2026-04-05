use std::convert::Infallible;

use lutum_protocol::{
    NoTools, RequestBudget, RequestExtensions,
    conversation::ModelInput,
    llm::{
        CompletionEventStream, CompletionOptions, CompletionRequest, GenerationParams, ModelName,
        StructuredCompletionEventStream, StructuredCompletionRequest,
        StructuredTurn as ProtocolStructuredTurn,
        StructuredTurnEventStream as ProtocolStructuredTurnEventStream, Temperature,
        TextTurn as ProtocolTextTurn, TextTurnEventStream as ProtocolTextTurnEventStream,
        TurnConfig,
    },
    reducer::{
        CompletionReductionError, CompletionTurnResult, CompletionTurnState,
        StructuredCompletionReductionError, StructuredCompletionResult, StructuredCompletionState,
        StructuredTurnReductionError, StructuredTurnResult as StructuredTurnCollectedResult,
        StructuredTurnState as StructuredTurnCollectedState, StructuredTurnStateWithTools,
        TextTurnReductionError, TextTurnResult as TextTurnCollectedResult,
        TextTurnState as TextTurnCollectedState, TextTurnStateWithTools,
    },
    structured::StructuredOutput,
    toolset::{ToolPolicy, Toolset},
};

use crate::{
    CollectError, Context, ContextError, EventHandler, PendingCompletion,
    PendingStructuredCompletion, PendingStructuredTurn, PendingStructuredTurnWithTools,
    PendingTextTurn, PendingTextTurnWithTools, Session, StructuredStepOutcomeWithTools,
    TextStepOutcomeWithTools,
    context::{StructuredTurnPartial, StructuredTurnPartialWithTools},
};

enum TurnTarget<'a> {
    Context { ctx: &'a Context, input: ModelInput },
    Session(&'a Session),
}

impl<'a> TurnTarget<'a> {
    fn context(&self) -> &Context {
        match self {
            Self::Context { ctx, .. } => ctx,
            Self::Session(session) => session.context(),
        }
    }

    fn input(&self) -> ModelInput {
        match self {
            Self::Context { input, .. } => input.clone(),
            Self::Session(session) => session.snapshot_input(),
        }
    }

    fn apply_defaults<T>(&self, turn: &mut TurnConfig<T>)
    where
        T: Toolset,
    {
        if let Self::Session(session) = self {
            session.apply_defaults(turn);
        }
    }
}

pub struct TextTurn<'a> {
    target: TurnTarget<'a>,
    extensions: RequestExtensions,
    turn: ProtocolTextTurn<NoTools>,
}

impl<'a> TextTurn<'a> {
    pub(crate) fn from_context(ctx: &'a Context, input: ModelInput) -> Self {
        Self {
            target: TurnTarget::Context { ctx, input },
            extensions: RequestExtensions::new(),
            turn: ProtocolTextTurn::new(),
        }
    }

    pub(crate) fn from_session(session: &'a Session) -> Self {
        Self {
            target: TurnTarget::Session(session),
            extensions: RequestExtensions::new(),
            turn: ProtocolTextTurn::new(),
        }
    }

    pub fn ext<T>(mut self, extension: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        self.extensions.insert(extension);
        self
    }

    pub fn extensions(mut self, extensions: RequestExtensions) -> Self {
        self.extensions.extend(extensions);
        self
    }

    pub fn temperature(mut self, temperature: Temperature) -> Self {
        self.turn.config.generation.temperature = Some(temperature);
        self
    }

    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.turn.config.generation.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.turn.config.generation.seed = Some(seed);
        self
    }

    pub fn budget(mut self, budget: RequestBudget) -> Self {
        self.turn.config.budget = budget;
        self
    }

    pub fn generation_config(mut self, generation: GenerationParams) -> Self {
        self.turn.config.generation = generation;
        self
    }

    pub fn tools<T>(self) -> TextTurnWithTools<'a, T>
    where
        T: Toolset,
    {
        let TextTurn {
            target,
            extensions,
            turn,
        } = self;
        let turn = ProtocolTextTurn {
            config: TurnConfig {
                generation: turn.config.generation,
                tools: ToolPolicy::Disabled,
                budget: turn.config.budget,
            },
        };
        TextTurnWithTools {
            target,
            extensions,
            turn,
        }
    }

    pub async fn start(self) -> Result<PendingTextTurn, ContextError> {
        let mut turn = self.turn;
        self.target.apply_defaults(&mut turn.config);
        self.target
            .context()
            .run_text_turn(self.extensions, self.target.input(), turn)
            .await
    }

    pub async fn stream(self) -> Result<ProtocolTextTurnEventStream, ContextError> {
        Ok(self.start().await?.into_stream())
    }

    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<
        TextTurnCollectedResult,
        CollectError<H::Error, TextTurnReductionError, TextTurnCollectedState>,
    >
    where
        H: EventHandler<lutum_protocol::TextTurnEvent, TextTurnCollectedState>,
    {
        match self.start().await {
            Ok(pending) => pending.collect_with(handler).await,
            Err(source) => Err(CollectError::Execution {
                source,
                partial: TextTurnCollectedState::default(),
            }),
        }
    }

    pub async fn collect(
        self,
    ) -> Result<
        TextTurnCollectedResult,
        CollectError<Infallible, TextTurnReductionError, TextTurnCollectedState>,
    > {
        match self.start().await {
            Ok(pending) => pending.collect().await,
            Err(source) => Err(CollectError::Execution {
                source,
                partial: TextTurnCollectedState::default(),
            }),
        }
    }
}

pub struct TextTurnWithTools<'a, T>
where
    T: Toolset,
{
    target: TurnTarget<'a>,
    extensions: RequestExtensions,
    turn: ProtocolTextTurn<T>,
}

impl<'a, T> TextTurnWithTools<'a, T>
where
    T: Toolset,
{
    pub fn ext<E>(mut self, extension: E) -> Self
    where
        E: Send + Sync + 'static,
    {
        self.extensions.insert(extension);
        self
    }

    pub fn extensions(mut self, extensions: RequestExtensions) -> Self {
        self.extensions.extend(extensions);
        self
    }

    pub fn temperature(mut self, temperature: Temperature) -> Self {
        self.turn.config.generation.temperature = Some(temperature);
        self
    }

    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.turn.config.generation.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.turn.config.generation.seed = Some(seed);
        self
    }

    pub fn budget(mut self, budget: RequestBudget) -> Self {
        self.turn.config.budget = budget;
        self
    }

    pub fn generation_config(mut self, generation: GenerationParams) -> Self {
        self.turn.config.generation = generation;
        self
    }

    pub fn allow_all(mut self) -> Self {
        self.turn.config.tools = ToolPolicy::AllowAll;
        self
    }

    pub fn allow_only(
        mut self,
        selectors: impl IntoIterator<Item = T::Selector>,
    ) -> Self {
        self.turn.config.tools = ToolPolicy::allow_only(selectors);
        self
    }

    pub fn require_all(mut self) -> Self {
        self.turn.config.tools = ToolPolicy::RequireAll;
        self
    }

    pub fn require_only(
        mut self,
        selectors: impl IntoIterator<Item = T::Selector>,
    ) -> Self {
        self.turn.config.tools = ToolPolicy::require_only(selectors);
        self
    }

    pub async fn start(self) -> Result<PendingTextTurnWithTools<T>, ContextError> {
        let mut turn = self.turn;
        self.target.apply_defaults(&mut turn.config);
        self.target
            .context()
            .run_text_turn_with_tools(self.extensions, self.target.input(), turn)
            .await
    }

    pub async fn stream(self) -> Result<lutum_protocol::TextTurnEventStreamWithTools<T>, ContextError> {
        Ok(self.start().await?.into_stream())
    }

    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<
        TextStepOutcomeWithTools<T>,
        CollectError<H::Error, TextTurnReductionError, TextTurnStateWithTools<T>>,
    >
    where
        H: EventHandler<lutum_protocol::TextTurnEventWithTools<T>, TextTurnStateWithTools<T>>,
    {
        match self.start().await {
            Ok(pending) => {
                let result = pending.collect_with(handler).await?;
                Ok(TextStepOutcomeWithTools::from_result(result))
            }
            Err(source) => Err(CollectError::Execution {
                source,
                partial: TextTurnStateWithTools::default(),
            }),
        }
    }

    pub async fn collect(
        self,
    ) -> Result<
        TextStepOutcomeWithTools<T>,
        CollectError<Infallible, TextTurnReductionError, TextTurnStateWithTools<T>>,
    > {
        match self.start().await {
            Ok(pending) => {
                let result = pending.collect().await?;
                Ok(TextStepOutcomeWithTools::from_result(result))
            }
            Err(source) => Err(CollectError::Execution {
                source,
                partial: TextTurnStateWithTools::default(),
            }),
        }
    }
}

pub struct StructuredTurn<'a, O>
where
    O: StructuredOutput,
{
    target: TurnTarget<'a>,
    extensions: RequestExtensions,
    turn: ProtocolStructuredTurn<NoTools, O>,
}

impl<'a, O> StructuredTurn<'a, O>
where
    O: StructuredOutput,
{
    pub(crate) fn from_context(ctx: &'a Context, input: ModelInput) -> Self {
        Self {
            target: TurnTarget::Context { ctx, input },
            extensions: RequestExtensions::new(),
            turn: ProtocolStructuredTurn::new(),
        }
    }

    pub(crate) fn from_session(session: &'a Session) -> Self {
        Self {
            target: TurnTarget::Session(session),
            extensions: RequestExtensions::new(),
            turn: ProtocolStructuredTurn::new(),
        }
    }

    pub fn ext<T>(mut self, extension: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        self.extensions.insert(extension);
        self
    }

    pub fn extensions(mut self, extensions: RequestExtensions) -> Self {
        self.extensions.extend(extensions);
        self
    }

    pub fn temperature(mut self, temperature: Temperature) -> Self {
        self.turn.config.generation.temperature = Some(temperature);
        self
    }

    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.turn.config.generation.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.turn.config.generation.seed = Some(seed);
        self
    }

    pub fn budget(mut self, budget: RequestBudget) -> Self {
        self.turn.config.budget = budget;
        self
    }

    pub fn generation_config(mut self, generation: GenerationParams) -> Self {
        self.turn.config.generation = generation;
        self
    }

    pub fn tools<T>(self) -> StructuredTurnWithTools<'a, T, O>
    where
        T: Toolset,
    {
        let StructuredTurn {
            target,
            extensions,
            turn,
        } = self;
        let turn = ProtocolStructuredTurn {
            config: TurnConfig {
                generation: turn.config.generation,
                tools: ToolPolicy::Disabled,
                budget: turn.config.budget,
            },
            output: turn.output,
        };
        StructuredTurnWithTools {
            target,
            extensions,
            turn,
        }
    }

    pub async fn start(self) -> Result<PendingStructuredTurn<O>, ContextError> {
        let mut turn = self.turn;
        self.target.apply_defaults(&mut turn.config);
        self.target
            .context()
            .run_structured_turn(self.extensions, self.target.input(), turn)
            .await
    }

    pub async fn stream(self) -> Result<ProtocolStructuredTurnEventStream<O>, ContextError> {
        Ok(self.start().await?.into_stream())
    }

    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<
        StructuredTurnCollectedResult<O>,
        CollectError<H::Error, StructuredTurnReductionError, StructuredTurnPartial<O>>,
    >
    where
        H: EventHandler<lutum_protocol::StructuredTurnEvent<O>, StructuredTurnCollectedState<O>>,
    {
        match self.start().await {
            Ok(pending) => pending.collect_with(handler).await,
            Err(source) => Err(CollectError::Execution {
                source,
                partial: StructuredTurnPartial::from_state(StructuredTurnCollectedState::default()),
            }),
        }
    }

    pub async fn collect(
        self,
    ) -> Result<
        StructuredTurnCollectedResult<O>,
        CollectError<Infallible, StructuredTurnReductionError, StructuredTurnPartial<O>>,
    > {
        match self.start().await {
            Ok(pending) => pending.collect().await,
            Err(source) => Err(CollectError::Execution {
                source,
                partial: StructuredTurnPartial::from_state(StructuredTurnCollectedState::default()),
            }),
        }
    }
}

pub struct StructuredTurnWithTools<'a, T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    target: TurnTarget<'a>,
    extensions: RequestExtensions,
    turn: ProtocolStructuredTurn<T, O>,
}

impl<'a, T, O> StructuredTurnWithTools<'a, T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    pub fn ext<E>(mut self, extension: E) -> Self
    where
        E: Send + Sync + 'static,
    {
        self.extensions.insert(extension);
        self
    }

    pub fn extensions(mut self, extensions: RequestExtensions) -> Self {
        self.extensions.extend(extensions);
        self
    }

    pub fn temperature(mut self, temperature: Temperature) -> Self {
        self.turn.config.generation.temperature = Some(temperature);
        self
    }

    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.turn.config.generation.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.turn.config.generation.seed = Some(seed);
        self
    }

    pub fn budget(mut self, budget: RequestBudget) -> Self {
        self.turn.config.budget = budget;
        self
    }

    pub fn generation_config(mut self, generation: GenerationParams) -> Self {
        self.turn.config.generation = generation;
        self
    }

    pub fn allow_all(mut self) -> Self {
        self.turn.config.tools = ToolPolicy::AllowAll;
        self
    }

    pub fn allow_only(
        mut self,
        selectors: impl IntoIterator<Item = T::Selector>,
    ) -> Self {
        self.turn.config.tools = ToolPolicy::allow_only(selectors);
        self
    }

    pub fn require_all(mut self) -> Self {
        self.turn.config.tools = ToolPolicy::RequireAll;
        self
    }

    pub fn require_only(
        mut self,
        selectors: impl IntoIterator<Item = T::Selector>,
    ) -> Self {
        self.turn.config.tools = ToolPolicy::require_only(selectors);
        self
    }

    pub async fn start(self) -> Result<PendingStructuredTurnWithTools<T, O>, ContextError> {
        let mut turn = self.turn;
        self.target.apply_defaults(&mut turn.config);
        self.target
            .context()
            .run_structured_turn_with_tools(self.extensions, self.target.input(), turn)
            .await
    }

    pub async fn stream(
        self,
    ) -> Result<lutum_protocol::StructuredTurnEventStreamWithTools<T, O>, ContextError> {
        Ok(self.start().await?.into_stream())
    }

    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<
        StructuredStepOutcomeWithTools<T, O>,
        CollectError<H::Error, StructuredTurnReductionError, StructuredTurnPartialWithTools<T, O>>,
    >
    where
        H: EventHandler<
            lutum_protocol::StructuredTurnEventWithTools<T, O>,
            StructuredTurnStateWithTools<T, O>,
        >,
    {
        match self.start().await {
            Err(source) => Err(CollectError::Execution {
                source,
                partial: StructuredTurnPartialWithTools::from_state(
                    StructuredTurnStateWithTools::default(),
                ),
            }),
            Ok(pending) => match pending.collect_with(handler).await {
                Ok(result) => Ok(StructuredStepOutcomeWithTools::from_result(result)),
                Err(CollectError::Reduction {
                    source: StructuredTurnReductionError::MissingSemantic,
                    partial,
                }) => {
                    let partial_for_outcome = StructuredTurnPartialWithTools {
                        state: partial.state.clone(),
                        committed_turn: partial.committed_turn.clone(),
                    };
                    if let Some(outcome) =
                        StructuredStepOutcomeWithTools::from_partial(partial_for_outcome)
                    {
                        Ok(outcome)
                    } else {
                        Err(CollectError::Reduction {
                            source: StructuredTurnReductionError::MissingSemantic,
                            partial,
                        })
                    }
                }
                Err(err) => Err(err),
            },
        }
    }

    pub async fn collect(
        self,
    ) -> Result<
        StructuredStepOutcomeWithTools<T, O>,
        CollectError<
            Infallible,
            StructuredTurnReductionError,
            StructuredTurnPartialWithTools<T, O>,
        >,
    > {
        match self.start().await {
            Err(source) => Err(CollectError::Execution {
                source,
                partial: StructuredTurnPartialWithTools::from_state(
                    StructuredTurnStateWithTools::default(),
                ),
            }),
            Ok(pending) => match pending.collect().await {
                Ok(result) => Ok(StructuredStepOutcomeWithTools::from_result(result)),
                Err(CollectError::Reduction {
                    source: StructuredTurnReductionError::MissingSemantic,
                    partial,
                }) => {
                    let partial_for_outcome = StructuredTurnPartialWithTools {
                        state: partial.state.clone(),
                        committed_turn: partial.committed_turn.clone(),
                    };
                    if let Some(outcome) =
                        StructuredStepOutcomeWithTools::from_partial(partial_for_outcome)
                    {
                        Ok(outcome)
                    } else {
                        Err(CollectError::Reduction {
                            source: StructuredTurnReductionError::MissingSemantic,
                            partial,
                        })
                    }
                }
                Err(err) => Err(err),
            },
        }
    }
}

pub struct Completion<'a> {
    ctx: &'a Context,
    extensions: RequestExtensions,
    request: CompletionRequest,
}

impl<'a> Completion<'a> {
    pub(crate) fn new(ctx: &'a Context, model: ModelName, prompt: impl Into<String>) -> Self {
        Self {
            ctx,
            extensions: RequestExtensions::new(),
            request: CompletionRequest::new(model, prompt),
        }
    }

    pub fn ext<T>(mut self, extension: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        self.extensions.insert(extension);
        self
    }

    pub fn extensions(mut self, extensions: RequestExtensions) -> Self {
        self.extensions.extend(extensions);
        self
    }

    pub fn temperature(mut self, temperature: Temperature) -> Self {
        self.request.options.temperature = Some(temperature);
        self
    }

    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.request.options.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub fn completion_options(mut self, options: CompletionOptions) -> Self {
        self.request.options = options;
        self
    }

    pub fn budget(mut self, budget: RequestBudget) -> Self {
        self.request.budget = budget;
        self
    }

    pub async fn start(self) -> Result<PendingCompletion, ContextError> {
        self.ctx.run_completion(self.extensions, self.request).await
    }

    pub async fn stream(self) -> Result<CompletionEventStream, ContextError> {
        Ok(self.start().await?.into_stream())
    }

    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<
        CompletionTurnResult,
        CollectError<H::Error, CompletionReductionError, CompletionTurnState>,
    >
    where
        H: EventHandler<lutum_protocol::CompletionEvent, CompletionTurnState>,
    {
        match self.start().await {
            Ok(pending) => pending.collect_with(handler).await,
            Err(source) => Err(CollectError::Execution {
                source,
                partial: CompletionTurnState::default(),
            }),
        }
    }

    pub async fn collect(
        self,
    ) -> Result<
        CompletionTurnResult,
        CollectError<Infallible, CompletionReductionError, CompletionTurnState>,
    > {
        match self.start().await {
            Ok(pending) => pending.collect().await,
            Err(source) => Err(CollectError::Execution {
                source,
                partial: CompletionTurnState::default(),
            }),
        }
    }
}

pub struct StructuredCompletion<'a, O>
where
    O: StructuredOutput,
{
    ctx: &'a Context,
    extensions: RequestExtensions,
    request: StructuredCompletionRequest<O>,
}

impl<'a, O> StructuredCompletion<'a, O>
where
    O: StructuredOutput,
{
    pub(crate) fn new(ctx: &'a Context, model: ModelName, prompt: impl Into<String>) -> Self {
        Self {
            ctx,
            extensions: RequestExtensions::new(),
            request: StructuredCompletionRequest::new(model, prompt),
        }
    }

    pub fn ext<T>(mut self, extension: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        self.extensions.insert(extension);
        self
    }

    pub fn extensions(mut self, extensions: RequestExtensions) -> Self {
        self.extensions.extend(extensions);
        self
    }

    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.request.system = Some(system.into());
        self
    }

    pub fn temperature(mut self, temperature: Temperature) -> Self {
        self.request.generation.temperature = Some(temperature);
        self
    }

    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.request.generation.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.request.generation.seed = Some(seed);
        self
    }

    pub fn budget(mut self, budget: RequestBudget) -> Self {
        self.request.budget = budget;
        self
    }

    pub fn generation_config(mut self, generation: GenerationParams) -> Self {
        self.request.generation = generation;
        self
    }

    pub async fn start(self) -> Result<PendingStructuredCompletion<O>, ContextError> {
        self.ctx
            .run_structured_completion(self.extensions, self.request)
            .await
    }

    pub async fn stream(self) -> Result<StructuredCompletionEventStream<O>, ContextError> {
        Ok(self.start().await?.into_stream())
    }

    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<
        StructuredCompletionResult<O>,
        CollectError<
            H::Error,
            StructuredCompletionReductionError,
            StructuredCompletionState<O>,
        >,
    >
    where
        H: EventHandler<lutum_protocol::StructuredCompletionEvent<O>, StructuredCompletionState<O>>,
    {
        match self.start().await {
            Ok(pending) => pending.collect_with(handler).await,
            Err(source) => Err(CollectError::Execution {
                source,
                partial: StructuredCompletionState::default(),
            }),
        }
    }

    pub async fn collect(
        self,
    ) -> Result<
        StructuredCompletionResult<O>,
        CollectError<
            Infallible,
            StructuredCompletionReductionError,
            StructuredCompletionState<O>,
        >,
    > {
        match self.start().await {
            Ok(pending) => pending.collect().await,
            Err(source) => Err(CollectError::Execution {
                source,
                partial: StructuredCompletionState::default(),
            }),
        }
    }
}
