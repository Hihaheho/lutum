use std::convert::Infallible;

use lutum_protocol::{
    AssistantTurn, NoTools, RequestBudget, RequestExtensions, UncommittedAssistantTurn,
    conversation::ModelInput,
    llm::{
        CompletionEventStream, CompletionOptions, CompletionRequest, GenerationParams,
        StructuredCompletionEventStream, StructuredCompletionRequest,
        StructuredTurn as ProtocolStructuredTurn,
        StructuredTurnEventStream as ProtocolStructuredTurnEventStream, Temperature,
        TextTurn as ProtocolTextTurn, TextTurnEventStream as ProtocolTextTurnEventStream,
        TurnConfig,
    },
    reducer::{
        CompletionReductionError, CompletionTurnResult, CompletionTurnState,
        StagedStructuredTurnResult, StagedTextTurnResult, StructuredCompletionReductionError,
        StructuredCompletionResult, StructuredCompletionState, StructuredTurnReductionError,
        StructuredTurnState as StructuredTurnCollectedState, StructuredTurnStateWithTools,
        TextTurnReductionError, TextTurnState as TextTurnCollectedState, TextTurnStateWithTools,
    },
    structured::StructuredOutput,
    toolset::{ToolAvailability, ToolConstraints, ToolRequirement, Toolset},
};

use crate::{
    CollectError, EventHandler, Lutum, LutumError, PendingCompletion, PendingStructuredCompletion,
    PendingStructuredTurn, PendingStructuredTurnWithTools, PendingTextTurn,
    PendingTextTurnWithTools, Session, StructuredStepOutcomeWithTools, TextStepOutcomeWithTools,
    context::{StructuredTurnPartial, StructuredTurnPartialWithTools},
};

enum TurnTarget<'a> {
    Lutum { lutum: &'a Lutum, input: ModelInput },
    Session { session: &'a mut Session },
}

impl<'a> TurnTarget<'a> {
    fn lutum_owned(&self) -> Lutum {
        match self {
            Self::Lutum { lutum, .. } => (*lutum).clone(),
            Self::Session { session } => session.lutum().clone(),
        }
    }

    fn input(&mut self) -> ModelInput {
        match self {
            Self::Lutum { input, .. } => input.clone(),
            Self::Session { session } => session.snapshot_input(),
        }
    }

    fn apply_defaults<T>(&self, turn: &mut TurnConfig<T>)
    where
        T: Toolset,
    {
        if let Self::Session { session } = self {
            session.apply_defaults(turn);
        }
    }

    /// Commit to the session if this is a session target; otherwise discard.
    fn commit_staged(self, turn: UncommittedAssistantTurn) {
        match self {
            Self::Lutum { .. } => turn.discard(),
            Self::Session { session } => turn.commit_into(session.input_mut()),
        }
    }
}

pub struct TextTurn<'a> {
    target: TurnTarget<'a>,
    extensions: RequestExtensions,
    turn: ProtocolTextTurn<NoTools>,
}

impl<'a> TextTurn<'a> {
    pub(crate) fn from_lutum(lutum: &'a Lutum, input: ModelInput) -> Self {
        Self {
            target: TurnTarget::Lutum { lutum, input },
            extensions: RequestExtensions::new(),
            turn: ProtocolTextTurn::new(),
        }
    }

    pub(crate) fn from_session(session: &'a mut Session) -> Self {
        Self {
            target: TurnTarget::Session { session },
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
                tools: ToolConstraints::default(),
                budget: turn.config.budget,
            },
        };
        TextTurnWithTools {
            target,
            extensions,
            turn,
        }
    }

    pub async fn start(self) -> Result<PendingTextTurn, LutumError> {
        let TextTurn {
            mut target,
            extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input();
        lutum.run_text_turn(extensions, input, turn).await
    }

    pub async fn stream(self) -> Result<ProtocolTextTurnEventStream, LutumError> {
        Ok(self.start().await?.into_stream())
    }

    /// Collect the turn with a custom event handler. Always returns a staged result
    /// (never auto-commits). Use [`collect`] for auto-commit or
    /// [`collect_staged`] for staged without a handler.
    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<
        StagedTextTurnResult,
        CollectError<H::Error, TextTurnReductionError, TextTurnCollectedState>,
    >
    where
        H: EventHandler<lutum_protocol::TextTurnEvent, TextTurnCollectedState>,
    {
        let TextTurn {
            mut target,
            extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input();
        drop(target);
        match lutum.run_text_turn(extensions, input, turn).await {
            Ok(pending) => pending.collect_with(handler).await,
            Err(source) => Err(CollectError::Execution {
                source,
                partial: TextTurnCollectedState::default(),
            }),
        }
    }

    /// Collect without auto-committing. Returns a staged result with an
    /// [`UncommittedAssistantTurn`] that you can commit later.
    pub async fn collect_staged(
        self,
    ) -> Result<
        StagedTextTurnResult,
        CollectError<Infallible, TextTurnReductionError, TextTurnCollectedState>,
    > {
        let TextTurn {
            mut target,
            extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input();
        drop(target);
        match lutum.run_text_turn(extensions, input, turn).await {
            Ok(pending) => pending.collect().await,
            Err(source) => Err(CollectError::Execution {
                source,
                partial: TextTurnCollectedState::default(),
            }),
        }
    }

    /// Collect and auto-commit to the session (if session-originated). Returns the
    /// committed result directly; use [`collect_staged`] to opt out of auto-commit.
    pub async fn collect(
        self,
    ) -> Result<
        lutum_protocol::TextTurnResult,
        CollectError<Infallible, TextTurnReductionError, TextTurnCollectedState>,
    > {
        let TextTurn {
            mut target,
            extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input();
        let staged = match lutum.run_text_turn(extensions, input, turn).await {
            Ok(pending) => match pending.collect().await {
                Ok(s) => s,
                Err(e) => return Err(e),
            },
            Err(source) => {
                return Err(CollectError::Execution {
                    source,
                    partial: TextTurnCollectedState::default(),
                });
            }
        };
        let assistant_turn = staged.turn.assistant_turn().clone();
        target.commit_staged(staged.turn);
        Ok(lutum_protocol::TextTurnResult {
            request_id: staged.request_id,
            model: staged.model,
            assistant_turn,
            finish_reason: staged.finish_reason,
            usage: staged.usage,
        })
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

    pub fn available_tools(mut self, selectors: impl IntoIterator<Item = T::Selector>) -> Self {
        self.turn.config.tools.available = ToolAvailability::Only(selectors.into_iter().collect());
        self
    }

    pub fn require_any_tool(mut self) -> Self {
        self.turn.config.tools.requirement = ToolRequirement::AtLeastOne;
        self
    }

    pub fn require_tool(mut self, selector: T::Selector) -> Self {
        self.turn.config.tools.requirement = ToolRequirement::Specific(selector);
        self
    }

    /// Override the description for a single tool at this turn site. Useful for
    /// injecting live state into tool descriptions (e.g. "calls remaining: 2").
    pub fn describe_tool(mut self, selector: T::Selector, description: impl Into<String>) -> Self {
        self.turn
            .config
            .tools
            .description_overrides
            .push((selector, description.into()));
        self
    }

    /// Bulk-apply description overrides. Pairs well with
    /// `{Name}Hooks::description_overrides().await` for eval-driven description probing.
    pub fn describe_many_tools(
        mut self,
        overrides: impl IntoIterator<Item = (T::Selector, String)>,
    ) -> Self {
        self.turn
            .config
            .tools
            .description_overrides
            .extend(overrides);
        self
    }

    pub async fn start(self) -> Result<PendingTextTurnWithTools<T>, LutumError> {
        let TextTurnWithTools {
            mut target,
            extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input();
        lutum
            .run_text_turn_with_tools(extensions, input, turn)
            .await
    }

    pub async fn stream(
        self,
    ) -> Result<lutum_protocol::TextTurnEventStreamWithTools<T>, LutumError> {
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
        let TextTurnWithTools {
            mut target,
            extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input();
        let staged = match lutum
            .run_text_turn_with_tools(extensions, input, turn)
            .await
        {
            Ok(pending) => match pending.collect_with(handler).await {
                Ok(s) => s,
                Err(e) => return Err(e),
            },
            Err(source) => {
                return Err(CollectError::Execution {
                    source,
                    partial: TextTurnStateWithTools::default(),
                });
            }
        };
        let outcome = match target {
            TurnTarget::Session { session } => {
                TextStepOutcomeWithTools::from_staged(staged, Some(session.input_mut()))
            }
            TurnTarget::Lutum { .. } => TextStepOutcomeWithTools::from_staged(staged, None),
        };
        Ok(outcome)
    }

    pub async fn collect(
        self,
    ) -> Result<
        TextStepOutcomeWithTools<T>,
        CollectError<Infallible, TextTurnReductionError, TextTurnStateWithTools<T>>,
    > {
        let TextTurnWithTools {
            mut target,
            extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input();
        let staged = match lutum
            .run_text_turn_with_tools(extensions, input, turn)
            .await
        {
            Ok(pending) => match pending.collect().await {
                Ok(s) => s,
                Err(e) => return Err(e),
            },
            Err(source) => {
                return Err(CollectError::Execution {
                    source,
                    partial: TextTurnStateWithTools::default(),
                });
            }
        };
        let outcome = match target {
            TurnTarget::Session { session } => {
                TextStepOutcomeWithTools::from_staged(staged, Some(session.input_mut()))
            }
            TurnTarget::Lutum { .. } => TextStepOutcomeWithTools::from_staged(staged, None),
        };
        Ok(outcome)
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
    pub(crate) fn from_lutum(lutum: &'a Lutum, input: ModelInput) -> Self {
        Self {
            target: TurnTarget::Lutum { lutum, input },
            extensions: RequestExtensions::new(),
            turn: ProtocolStructuredTurn::new(),
        }
    }

    pub(crate) fn from_session(session: &'a mut Session) -> Self {
        Self {
            target: TurnTarget::Session { session },
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
                tools: ToolConstraints::default(),
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

    pub async fn start(self) -> Result<PendingStructuredTurn<O>, LutumError> {
        let StructuredTurn {
            mut target,
            extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input();
        lutum.run_structured_turn(extensions, input, turn).await
    }

    pub async fn stream(self) -> Result<ProtocolStructuredTurnEventStream<O>, LutumError> {
        Ok(self.start().await?.into_stream())
    }

    /// Collect the turn with a custom event handler. Always returns a staged result
    /// (never auto-commits). Use [`collect`] for auto-commit or
    /// [`collect_staged`] for staged without a handler.
    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<
        StagedStructuredTurnResult<O>,
        CollectError<H::Error, StructuredTurnReductionError, StructuredTurnPartial<O>>,
    >
    where
        H: EventHandler<lutum_protocol::StructuredTurnEvent<O>, StructuredTurnCollectedState<O>>,
    {
        let StructuredTurn {
            mut target,
            extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input();
        drop(target);
        match lutum.run_structured_turn(extensions, input, turn).await {
            Ok(pending) => pending.collect_with(handler).await,
            Err(source) => Err(CollectError::Execution {
                source,
                partial: StructuredTurnPartial::from_state(StructuredTurnCollectedState::default()),
            }),
        }
    }

    /// Collect without auto-committing. Returns a staged result with an
    /// [`UncommittedAssistantTurn`] that you can commit later.
    pub async fn collect_staged(
        self,
    ) -> Result<
        StagedStructuredTurnResult<O>,
        CollectError<Infallible, StructuredTurnReductionError, StructuredTurnPartial<O>>,
    > {
        let StructuredTurn {
            mut target,
            extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input();
        drop(target);
        match lutum.run_structured_turn(extensions, input, turn).await {
            Ok(pending) => pending.collect().await,
            Err(source) => Err(CollectError::Execution {
                source,
                partial: StructuredTurnPartial::from_state(StructuredTurnCollectedState::default()),
            }),
        }
    }

    /// Collect and auto-commit to the session (if session-originated). Returns the
    /// committed result directly; use [`collect_staged`] to opt out of auto-commit.
    pub async fn collect(
        self,
    ) -> Result<
        lutum_protocol::StructuredTurnResult<O>,
        CollectError<Infallible, StructuredTurnReductionError, StructuredTurnPartial<O>>,
    > {
        let StructuredTurn {
            mut target,
            extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input();
        let staged = match lutum.run_structured_turn(extensions, input, turn).await {
            Ok(pending) => match pending.collect().await {
                Ok(s) => s,
                Err(e) => return Err(e),
            },
            Err(source) => {
                return Err(CollectError::Execution {
                    source,
                    partial: StructuredTurnPartial::from_state(
                        StructuredTurnCollectedState::default(),
                    ),
                });
            }
        };
        let assistant_turn = staged.turn.assistant_turn().clone();
        target.commit_staged(staged.turn);
        Ok(lutum_protocol::StructuredTurnResult {
            request_id: staged.request_id,
            model: staged.model,
            assistant_turn,
            semantic: staged.semantic,
            finish_reason: staged.finish_reason,
            usage: staged.usage,
        })
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

    pub fn available_tools(mut self, selectors: impl IntoIterator<Item = T::Selector>) -> Self {
        self.turn.config.tools.available = ToolAvailability::Only(selectors.into_iter().collect());
        self
    }

    pub fn require_any_tool(mut self) -> Self {
        self.turn.config.tools.requirement = ToolRequirement::AtLeastOne;
        self
    }

    pub fn require_tool(mut self, selector: T::Selector) -> Self {
        self.turn.config.tools.requirement = ToolRequirement::Specific(selector);
        self
    }

    /// Override the description for a single tool at this turn site. Useful for
    /// injecting live state into tool descriptions (e.g. "calls remaining: 2").
    pub fn describe_tool(mut self, selector: T::Selector, description: impl Into<String>) -> Self {
        self.turn
            .config
            .tools
            .description_overrides
            .push((selector, description.into()));
        self
    }

    /// Bulk-apply description overrides. Pairs well with
    /// `{Name}Hooks::description_overrides().await` for eval-driven description probing.
    pub fn describe_many_tools(
        mut self,
        overrides: impl IntoIterator<Item = (T::Selector, String)>,
    ) -> Self {
        self.turn
            .config
            .tools
            .description_overrides
            .extend(overrides);
        self
    }

    pub async fn start(self) -> Result<PendingStructuredTurnWithTools<T, O>, LutumError> {
        let StructuredTurnWithTools {
            mut target,
            extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input();
        lutum
            .run_structured_turn_with_tools(extensions, input, turn)
            .await
    }

    pub async fn stream(
        self,
    ) -> Result<lutum_protocol::StructuredTurnEventStreamWithTools<T, O>, LutumError> {
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
        let StructuredTurnWithTools {
            mut target,
            extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input();
        let pending = match lutum
            .run_structured_turn_with_tools(extensions, input, turn)
            .await
        {
            Ok(p) => p,
            Err(source) => {
                return Err(CollectError::Execution {
                    source,
                    partial: StructuredTurnPartialWithTools::from_state(
                        StructuredTurnStateWithTools::default(),
                    ),
                });
            }
        };
        match pending.collect_with(handler).await {
            Ok(staged) => {
                let outcome = match target {
                    TurnTarget::Session { session } => StructuredStepOutcomeWithTools::from_staged(
                        staged,
                        Some(session.input_mut()),
                    ),
                    TurnTarget::Lutum { .. } => {
                        StructuredStepOutcomeWithTools::from_staged(staged, None)
                    }
                };
                Ok(outcome)
            }
            Err(CollectError::Reduction {
                source: StructuredTurnReductionError::MissingSemantic,
                partial,
            }) => {
                // The model used tool calls without structured output — recover as NeedsTools.
                if !partial.state.tool_calls.is_empty()
                    && let (
                        Some(committed_turn),
                        Some(finish_reason),
                        Some(usage),
                        Ok(assistant_turn),
                    ) = (
                        partial.committed_turn.clone(),
                        partial.state.finish_reason.clone(),
                        partial.state.usage,
                        AssistantTurn::from_items(partial.state.assistant_turn.clone()),
                    )
                {
                    let tool_calls = partial.state.tool_calls.clone();
                    let outcome = StructuredStepOutcomeWithTools::from_partial(
                        assistant_turn,
                        committed_turn,
                        tool_calls,
                        partial.state.request_id.clone(),
                        partial.state.model.clone(),
                        finish_reason,
                        usage,
                    );
                    return Ok(outcome);
                }
                Err(CollectError::Reduction {
                    source: StructuredTurnReductionError::MissingSemantic,
                    partial,
                })
            }
            Err(err) => Err(err),
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
        let StructuredTurnWithTools {
            mut target,
            extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input();
        let pending = match lutum
            .run_structured_turn_with_tools(extensions, input, turn)
            .await
        {
            Ok(p) => p,
            Err(source) => {
                return Err(CollectError::Execution {
                    source,
                    partial: StructuredTurnPartialWithTools::from_state(
                        StructuredTurnStateWithTools::default(),
                    ),
                });
            }
        };
        match pending.collect().await {
            Ok(staged) => {
                let outcome = match target {
                    TurnTarget::Session { session } => StructuredStepOutcomeWithTools::from_staged(
                        staged,
                        Some(session.input_mut()),
                    ),
                    TurnTarget::Lutum { .. } => {
                        StructuredStepOutcomeWithTools::from_staged(staged, None)
                    }
                };
                Ok(outcome)
            }
            Err(CollectError::Reduction {
                source: StructuredTurnReductionError::MissingSemantic,
                partial,
            }) => {
                // The model used tool calls without structured output — recover as NeedsTools.
                if !partial.state.tool_calls.is_empty()
                    && let (
                        Some(committed_turn),
                        Some(finish_reason),
                        Some(usage),
                        Ok(assistant_turn),
                    ) = (
                        partial.committed_turn.clone(),
                        partial.state.finish_reason.clone(),
                        partial.state.usage,
                        AssistantTurn::from_items(partial.state.assistant_turn.clone()),
                    )
                {
                    let tool_calls = partial.state.tool_calls.clone();
                    let outcome = StructuredStepOutcomeWithTools::from_partial(
                        assistant_turn,
                        committed_turn,
                        tool_calls,
                        partial.state.request_id.clone(),
                        partial.state.model.clone(),
                        finish_reason,
                        usage,
                    );
                    return Ok(outcome);
                }
                Err(CollectError::Reduction {
                    source: StructuredTurnReductionError::MissingSemantic,
                    partial,
                })
            }
            Err(err) => Err(err),
        }
    }
}

pub struct Completion<'a> {
    lutum: &'a Lutum,
    extensions: RequestExtensions,
    request: CompletionRequest,
}

impl<'a> Completion<'a> {
    pub(crate) fn new(lutum: &'a Lutum, prompt: impl Into<String>) -> Self {
        Self {
            lutum,
            extensions: RequestExtensions::new(),
            request: CompletionRequest::new(prompt),
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

    pub async fn start(self) -> Result<PendingCompletion, LutumError> {
        self.lutum
            .run_completion(self.extensions, self.request)
            .await
    }

    pub async fn stream(self) -> Result<CompletionEventStream, LutumError> {
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
    lutum: &'a Lutum,
    extensions: RequestExtensions,
    request: StructuredCompletionRequest<O>,
}

impl<'a, O> StructuredCompletion<'a, O>
where
    O: StructuredOutput,
{
    pub(crate) fn new(lutum: &'a Lutum, prompt: impl Into<String>) -> Self {
        Self {
            lutum,
            extensions: RequestExtensions::new(),
            request: StructuredCompletionRequest::new(prompt),
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

    pub async fn start(self) -> Result<PendingStructuredCompletion<O>, LutumError> {
        self.lutum
            .run_structured_completion(self.extensions, self.request)
            .await
    }

    pub async fn stream(self) -> Result<StructuredCompletionEventStream<O>, LutumError> {
        Ok(self.start().await?.into_stream())
    }

    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<
        StructuredCompletionResult<O>,
        CollectError<H::Error, StructuredCompletionReductionError, StructuredCompletionState<O>>,
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
        CollectError<Infallible, StructuredCompletionReductionError, StructuredCompletionState<O>>,
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
