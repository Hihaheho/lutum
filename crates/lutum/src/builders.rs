use std::sync::Arc;

use lutum_protocol::{
    AssistantTurn, CollectErrorKind, NoTools, OperationKind, RequestBudget, RequestExtensions,
    UncommittedAssistantTurn,
    conversation::ModelInput,
    emit_collect_error_enabled,
    llm::{
        CompletionEventStream, CompletionOptions, CompletionRequest, GenerationParams, RetryPolicy,
        StructuredCompletionEventStream, StructuredCompletionRequest,
        StructuredTurn as ProtocolStructuredTurn,
        StructuredTurnEventStream as ProtocolStructuredTurnEventStream, Temperature,
        TextTurn as ProtocolTextTurn, TextTurnEventStream as ProtocolTextTurnEventStream,
        TokenCount, TurnConfig,
    },
    reducer::{
        CompletionReductionError, CompletionTurnResult, CompletionTurnState,
        StagedStructuredTurnResult, StagedTextTurnResult, StructuredCompletionReductionError,
        StructuredCompletionResult, StructuredCompletionState, StructuredTurnReductionError,
        StructuredTurnState as StructuredTurnCollectedState, StructuredTurnStateWithTools,
        TextTurnReductionError, TextTurnState as TextTurnCollectedState, TextTurnStateWithTools,
    },
    structured::StructuredOutput,
    toolset::{
        DynamicTool, HasDynamicSlot, TextToolCallFallbackParser, ToolAvailability, ToolConstraints,
        ToolRequirement, Toolset,
    },
};

use crate::{
    CollectError, EventHandler, EventHandlers, Lutum, LutumError, PendingCompletion,
    PendingStructuredCompletion, PendingStructuredTurn, PendingStructuredTurnWithTools,
    PendingTextTurn, PendingTextTurnWithTools, Session, StagedTextStepOutcomeWithTools,
    StructuredStepOutcomeWithTools, TextStepOutcomeWithTools, TextToolEventHandler,
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
            Self::Session { .. } => unreachable!("session builders receive Lutum at execution"),
        }
    }

    fn apply_session_extensions(&self, extensions: &mut RequestExtensions) {
        if let Self::Session { session, .. } = self {
            extensions.push_fallback(Arc::new(session.extensions().clone()));
        }
    }

    fn input(&mut self, extensions: &mut RequestExtensions) -> ModelInput {
        match self {
            Self::Lutum { input, .. } => input.clone(),
            Self::Session { session, .. } => {
                let (input, ephemeral_indices) = session.snapshot_input_with_ephemeral_indices();
                if !ephemeral_indices.is_empty() {
                    extensions.insert(ephemeral_indices);
                }
                input
            }
        }
    }

    fn preview_input(&self, extensions: &mut RequestExtensions) -> ModelInput {
        match self {
            Self::Lutum { input, .. } => input.clone(),
            Self::Session { session, .. } => {
                let (input, ephemeral_indices) = session.preview_input_with_ephemeral_indices();
                if !ephemeral_indices.is_empty() {
                    extensions.insert(ephemeral_indices);
                }
                input
            }
        }
    }

    fn apply_defaults<T>(&self, extensions: &mut RequestExtensions, turn: &mut TurnConfig<T>)
    where
        T: Toolset,
    {
        self.apply_session_extensions(extensions);
        Lutum::apply_max_output_tokens_extension(extensions, &mut turn.generation);
        if let Self::Session { session, .. } = self {
            session.apply_defaults(turn);
        }
    }

    fn generation_with_defaults<T>(
        &self,
        extensions: &mut RequestExtensions,
        config: &TurnConfig<T>,
    ) -> GenerationParams
    where
        T: Toolset,
    {
        self.apply_session_extensions(extensions);
        let mut generation = config.generation.clone();
        Lutum::apply_max_output_tokens_extension(extensions, &mut generation);
        if let Self::Session { session, .. } = self {
            let defaults = session.defaults();
            if generation.temperature.is_none() {
                generation.temperature = defaults.generation.temperature;
            }
            if generation.max_output_tokens.is_none() {
                generation.max_output_tokens = defaults.generation.max_output_tokens;
            }
            if generation.seed.is_none() {
                generation.seed = defaults.generation.seed;
            }
        }
        generation
    }

    /// Commit to the session if this is a session target; otherwise discard.
    fn commit_staged(self, turn: UncommittedAssistantTurn) {
        match self {
            Self::Lutum { .. } => turn.discard(),
            Self::Session { session, .. } => turn.commit_into(session.input_mut()),
        }
    }
}

fn emit_pre_stream_collect_error(
    enabled: bool,
    operation_kind: OperationKind,
    error: &impl std::fmt::Display,
) {
    emit_collect_error_enabled(
        enabled,
        operation_kind,
        None,
        CollectErrorKind::Execution,
        "request_id=None, stream_started=false",
        &error.to_string(),
    );
}

pub struct TextTurn<'a> {
    target: TurnTarget<'a>,
    extensions: RequestExtensions,
    turn: ProtocolTextTurn<NoTools>,
    event_handlers: EventHandlers<'a, lutum_protocol::TextTurnEvent, TextTurnCollectedState>,
}

impl<'a> TextTurn<'a> {
    pub(crate) fn from_lutum(lutum: &'a Lutum, input: ModelInput) -> Self {
        Self {
            target: TurnTarget::Lutum { lutum, input },
            extensions: RequestExtensions::new(),
            turn: ProtocolTextTurn::new(),
            event_handlers: EventHandlers::new(),
        }
    }

    pub(crate) fn from_session(session: &'a mut Session) -> SessionTextTurn<'a> {
        SessionTextTurn {
            inner: Self {
                target: TurnTarget::Session { session },
                extensions: RequestExtensions::new(),
                turn: ProtocolTextTurn::new(),
                event_handlers: EventHandlers::new(),
            },
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

    pub fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.extensions.insert(retry_policy);
        self
    }

    /// Register a per-turn event handler for this no-tools text turn.
    ///
    /// Handlers run after the reducer has applied each event, in registration
    /// order. After registering a no-tools text handler, use `collect`,
    /// `collect_staged`, or `collect_with` to run the turn.
    pub fn on_event<H>(mut self, handler: H) -> Self
    where
        H: EventHandler<lutum_protocol::TextTurnEvent, TextTurnCollectedState> + 'a,
    {
        self.event_handlers.push(handler);
        self
    }

    /// Enable tools for this text turn.
    ///
    /// # Panics
    ///
    /// Panics if `on_event` was already called on the no-tools text builder.
    /// Register event handlers after `tools::<T>()` so they use the
    /// tool-enabled event and state types.
    #[track_caller]
    pub fn tools<T>(self) -> TextTurnWithTools<'a, T>
    where
        T: Toolset,
    {
        let TextTurn {
            target,
            extensions,
            turn,
            event_handlers,
        } = self;
        assert!(
            event_handlers.is_empty(),
            "text event handlers must be registered after `.tools::<T>()` because tool-enabled turns use a different event type"
        );
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
            fallback_parser: None,
            event_handlers: EventHandlers::new(),
        }
    }

    /// Start the turn and return the pending stream collector.
    ///
    /// # Panics
    ///
    /// Panics if `on_event` was already called. Builder-registered handlers are
    /// consumed by `collect`, `collect_staged`, and `collect_with`.
    pub async fn start(self) -> Result<PendingTextTurn, LutumError> {
        let TextTurn {
            mut target,
            mut extensions,
            mut turn,
            event_handlers,
        } = self;
        assert!(
            event_handlers.is_empty(),
            "`on_event` handlers are used by collection methods; call `collect`, `collect_staged`, or `collect_with` instead of `start` after registering handlers"
        );
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        lutum.run_text_turn(extensions, input, turn).await
    }

    /// Start the turn and return the raw event stream.
    ///
    /// # Panics
    ///
    /// Panics if `on_event` was already called. Builder-registered handlers are
    /// only used by collection methods.
    pub async fn stream(self) -> Result<ProtocolTextTurnEventStream, LutumError> {
        Ok(self.start().await?.into_stream())
    }

    /// Count input tokens for this turn without sending a generation request.
    ///
    /// Returns `Ok(None)` when no token counter is attached to `Lutum`, or when
    /// the configured adapter surface does not support exact token counting.
    pub async fn count_tokens(&self) -> Result<Option<TokenCount>, LutumError> {
        let mut extensions = self.extensions.clone();
        let generation = self
            .target
            .generation_with_defaults(&mut extensions, &self.turn.config);
        let lutum = self.target.lutum_owned();
        let input = self.target.preview_input(&mut extensions);
        lutum
            .count_text_turn_tokens(extensions, input, &self.turn, generation)
            .await
    }

    /// Collect the turn with a custom event handler. Always returns a staged result
    /// (never auto-commits). Use [`collect`] for auto-commit or
    /// [`collect_staged`] for staged without a handler.
    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<StagedTextTurnResult, CollectError<TextTurnReductionError, TextTurnCollectedState>>
    where
        H: EventHandler<lutum_protocol::TextTurnEvent, TextTurnCollectedState>,
    {
        let TextTurn {
            mut target,
            mut extensions,
            mut turn,
            mut event_handlers,
        } = self;
        event_handlers.push(handler);
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        drop(target);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        match lutum.run_text_turn(extensions, input, turn).await {
            Ok(pending) => pending.collect_with(event_handlers).await,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: TextTurnCollectedState::default(),
                })
            }
        }
    }

    /// Collect without auto-committing. Returns a staged result with an
    /// [`UncommittedAssistantTurn`] that you can commit later.
    pub async fn collect_staged(
        self,
    ) -> Result<StagedTextTurnResult, CollectError<TextTurnReductionError, TextTurnCollectedState>>
    {
        let TextTurn {
            mut target,
            mut extensions,
            mut turn,
            event_handlers,
        } = self;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        drop(target);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        match lutum.run_text_turn(extensions, input, turn).await {
            Ok(pending) => pending.collect_with(event_handlers).await,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: TextTurnCollectedState::default(),
                })
            }
        }
    }

    /// Collect and return the completed result directly; use [`collect_staged`]
    /// to keep the uncommitted assistant turn.
    pub async fn collect(
        self,
    ) -> Result<
        lutum_protocol::TextTurnResult,
        CollectError<TextTurnReductionError, TextTurnCollectedState>,
    > {
        let TextTurn {
            mut target,
            mut extensions,
            mut turn,
            event_handlers,
        } = self;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        let staged = match lutum.run_text_turn(extensions, input, turn).await {
            Ok(pending) => match pending.collect_with(event_handlers).await {
                Ok(s) => s,
                Err(e) => return Err(e),
            },
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
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
            cumulative_usage: staged.cumulative_usage,
        })
    }
}

pub struct SessionTextTurn<'a> {
    inner: TextTurn<'a>,
}

impl<'a> SessionTextTurn<'a> {
    pub fn ext<T>(self, extension: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        Self {
            inner: self.inner.ext(extension),
        }
    }

    pub fn extensions(self, extensions: RequestExtensions) -> Self {
        Self {
            inner: self.inner.extensions(extensions),
        }
    }

    pub fn temperature(self, temperature: Temperature) -> Self {
        Self {
            inner: self.inner.temperature(temperature),
        }
    }

    pub fn max_output_tokens(self, max_output_tokens: u32) -> Self {
        Self {
            inner: self.inner.max_output_tokens(max_output_tokens),
        }
    }

    pub fn seed(self, seed: u64) -> Self {
        Self {
            inner: self.inner.seed(seed),
        }
    }

    pub fn budget(self, budget: RequestBudget) -> Self {
        Self {
            inner: self.inner.budget(budget),
        }
    }

    pub fn generation_config(self, generation: GenerationParams) -> Self {
        Self {
            inner: self.inner.generation_config(generation),
        }
    }

    pub fn retry_policy(self, retry_policy: RetryPolicy) -> Self {
        Self {
            inner: self.inner.retry_policy(retry_policy),
        }
    }

    /// Register a per-turn event handler for this no-tools text turn.
    ///
    /// Handlers run after the reducer has applied each event, in registration
    /// order. After registering a no-tools text handler, use `collect`,
    /// `collect_staged`, or `collect_with` to run the turn.
    pub fn on_event<H>(self, handler: H) -> Self
    where
        H: EventHandler<lutum_protocol::TextTurnEvent, TextTurnCollectedState> + 'a,
    {
        Self {
            inner: self.inner.on_event(handler),
        }
    }

    /// Enable tools for this text turn.
    ///
    /// # Panics
    ///
    /// Panics if `on_event` was already called on the no-tools text builder.
    /// Register event handlers after `tools::<T>()` so they use the
    /// tool-enabled event and state types.
    #[track_caller]
    pub fn tools<T>(self) -> SessionTextTurnWithTools<'a, T>
    where
        T: Toolset,
    {
        let TextTurn {
            target,
            extensions,
            turn,
            event_handlers,
        } = self.inner;
        assert!(
            event_handlers.is_empty(),
            "text event handlers must be registered after `.tools::<T>()` because tool-enabled turns use a different event type"
        );
        let turn = ProtocolTextTurn {
            config: TurnConfig {
                generation: turn.config.generation,
                tools: ToolConstraints::default(),
                budget: turn.config.budget,
            },
        };
        SessionTextTurnWithTools {
            inner: TextTurnWithTools {
                target,
                extensions,
                turn,
                fallback_parser: None,
                event_handlers: EventHandlers::new(),
            },
        }
    }

    /// Start the turn and return the pending stream collector.
    ///
    /// # Panics
    ///
    /// Panics if `on_event` was already called. Builder-registered handlers are
    /// consumed by `collect`, `collect_staged`, and `collect_with`.
    pub async fn start(self, lutum: &Lutum) -> Result<PendingTextTurn, LutumError> {
        let TextTurn {
            mut target,
            mut extensions,
            mut turn,
            event_handlers,
        } = self.inner;
        assert!(
            event_handlers.is_empty(),
            "`on_event` handlers are used by collection methods; call `collect`, `collect_staged`, or `collect_with` instead of `start` after registering handlers"
        );
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        lutum.run_text_turn(extensions, input, turn).await
    }

    /// Start the turn and return the raw event stream.
    ///
    /// # Panics
    ///
    /// Panics if `on_event` was already called. Builder-registered handlers are
    /// only used by collection methods.
    pub async fn stream(self, lutum: &Lutum) -> Result<ProtocolTextTurnEventStream, LutumError> {
        Ok(self.start(lutum).await?.into_stream())
    }

    /// Count input tokens for this turn without sending a generation request.
    ///
    /// Counting snapshots the current session input but does not commit, strip
    /// ephemeral items, or otherwise mutate the session.
    pub async fn count_tokens(&self, lutum: &Lutum) -> Result<Option<TokenCount>, LutumError> {
        let mut extensions = self.inner.extensions.clone();
        let generation = self
            .inner
            .target
            .generation_with_defaults(&mut extensions, &self.inner.turn.config);
        let input = self.inner.target.preview_input(&mut extensions);
        lutum
            .count_text_turn_tokens(extensions, input, &self.inner.turn, generation)
            .await
    }

    /// Collect the turn with a custom event handler. Always returns a staged result
    /// (never auto-commits). Use [`collect`] for auto-commit or
    /// [`collect_staged`] for staged without a handler.
    pub async fn collect_with<H>(
        self,
        lutum: &Lutum,
        handler: H,
    ) -> Result<StagedTextTurnResult, CollectError<TextTurnReductionError, TextTurnCollectedState>>
    where
        H: EventHandler<lutum_protocol::TextTurnEvent, TextTurnCollectedState>,
    {
        let TextTurn {
            mut target,
            mut extensions,
            mut turn,
            mut event_handlers,
        } = self.inner;
        event_handlers.push(handler);
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        drop(target);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        match lutum.run_text_turn(extensions, input, turn).await {
            Ok(pending) => pending.collect_with(event_handlers).await,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: TextTurnCollectedState::default(),
                })
            }
        }
    }

    /// Collect without auto-committing. Returns a staged result with an
    /// [`UncommittedAssistantTurn`] that you can commit later.
    pub async fn collect_staged(
        self,
        lutum: &Lutum,
    ) -> Result<StagedTextTurnResult, CollectError<TextTurnReductionError, TextTurnCollectedState>>
    {
        let TextTurn {
            mut target,
            mut extensions,
            mut turn,
            event_handlers,
        } = self.inner;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        drop(target);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        match lutum.run_text_turn(extensions, input, turn).await {
            Ok(pending) => pending.collect_with(event_handlers).await,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: TextTurnCollectedState::default(),
                })
            }
        }
    }

    /// Collect and auto-commit to the session. Returns the committed result directly;
    /// use [`collect_staged`] to opt out of auto-commit.
    pub async fn collect(
        self,
        lutum: &Lutum,
    ) -> Result<
        lutum_protocol::TextTurnResult,
        CollectError<TextTurnReductionError, TextTurnCollectedState>,
    > {
        let TextTurn {
            mut target,
            mut extensions,
            mut turn,
            event_handlers,
        } = self.inner;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        let staged = match lutum.run_text_turn(extensions, input, turn).await {
            Ok(pending) => match pending.collect_with(event_handlers).await {
                Ok(s) => s,
                Err(e) => return Err(e),
            },
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
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
            cumulative_usage: staged.cumulative_usage,
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
    fallback_parser: Option<Arc<dyn TextToolCallFallbackParser<T>>>,
    event_handlers:
        EventHandlers<'a, lutum_protocol::TextTurnEventWithTools<T>, TextTurnStateWithTools<T>>,
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

    pub fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.extensions.insert(retry_policy);
        self
    }

    /// Register a normal per-turn event handler for this tool-enabled text turn.
    ///
    /// Handlers run after the reducer has applied each event, in registration
    /// order. Use `collect_controlled_with` instead when the handler needs the
    /// controlled text+tools directives.
    pub fn on_event<H>(mut self, handler: H) -> Self
    where
        H: EventHandler<lutum_protocol::TextTurnEventWithTools<T>, TextTurnStateWithTools<T>> + 'a,
    {
        self.event_handlers.push(handler);
        self
    }

    pub fn available_tools(mut self, selectors: impl IntoIterator<Item = T::Selector>) -> Self {
        self.turn.config.tools.available = ToolAvailability::Only(selectors.into_iter().collect());
        self
    }

    /// Expose the default-on toolset *plus* the listed selectors on this turn.
    /// This is the typical way to temporarily re-enable variants marked
    /// `#[tool(off)]` / `#[toolset(off)]` (e.g. a loaded "skill") without
    /// having to enumerate the rest of the default set.
    pub fn available_tools_default_plus(
        mut self,
        selectors: impl IntoIterator<Item = T::Selector>,
    ) -> Self {
        self.turn.config.tools.available =
            ToolAvailability::DefaultPlus(selectors.into_iter().collect());
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

    /// Register an opt-in parser for backends that emit required tool calls as
    /// assistant text instead of native tool-call events.
    pub fn recover_tool_calls_with<P>(mut self, parser: P) -> Self
    where
        P: TextToolCallFallbackParser<T> + 'static,
    {
        self.fallback_parser = Some(Arc::new(parser));
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

    /// Start the turn and return the pending stream collector.
    ///
    /// # Panics
    ///
    /// Panics if `on_event` was already called. Builder-registered handlers are
    /// consumed by `collect`, `collect_staged`, and `collect_with`.
    pub async fn start(self) -> Result<PendingTextTurnWithTools<T>, LutumError> {
        let TextTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
            fallback_parser,
            event_handlers,
        } = self;
        assert!(
            event_handlers.is_empty(),
            "`on_event` handlers are used by collection methods; call `collect`, `collect_staged`, or `collect_with` instead of `start` after registering handlers"
        );
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        lutum
            .run_text_turn_with_tools(extensions, input, turn, fallback_parser)
            .await
    }

    /// Start the turn and return the raw event stream.
    ///
    /// # Panics
    ///
    /// Panics if `on_event` was already called. Builder-registered handlers are
    /// only used by collection methods.
    pub async fn stream(
        self,
    ) -> Result<lutum_protocol::TextTurnEventStreamWithTools<T>, LutumError> {
        Ok(self.start().await?.into_stream())
    }

    /// Count input tokens for this tool-capable turn without sending a generation request.
    ///
    /// Returns `Ok(None)` when no token counter is attached to `Lutum`, or when
    /// the configured adapter surface does not support exact token counting.
    pub async fn count_tokens(&self) -> Result<Option<TokenCount>, LutumError> {
        let mut extensions = self.extensions.clone();
        let generation = self
            .target
            .generation_with_defaults(&mut extensions, &self.turn.config);
        let lutum = self.target.lutum_owned();
        let input = self.target.preview_input(&mut extensions);
        lutum
            .count_text_turn_tokens(extensions, input, &self.turn, generation)
            .await
    }

    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<
        TextStepOutcomeWithTools<T>,
        CollectError<TextTurnReductionError, TextTurnStateWithTools<T>>,
    >
    where
        H: EventHandler<lutum_protocol::TextTurnEventWithTools<T>, TextTurnStateWithTools<T>>,
    {
        let TextTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
            fallback_parser,
            mut event_handlers,
        } = self;
        event_handlers.push(handler);
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        let staged = match lutum
            .run_text_turn_with_tools(extensions, input, turn, fallback_parser)
            .await
        {
            Ok(pending) => match pending.collect_with(event_handlers).await {
                Ok(s) => s,
                Err(e) => return Err(e),
            },
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
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

    /// Collect a text+tools turn with an advanced handler that can synthesize
    /// a finished turn or tool round.
    ///
    /// This is the text+tools-only control path. It keeps `collect_with`
    /// source-compatible while allowing handlers to return early or recover
    /// from collection errors through `TextToolEventHandler::on_error`.
    ///
    /// # Panics
    ///
    /// Panics if normal `on_event` handlers are already registered on the
    /// builder. Use `collect_with` for normal handlers, or pass controlled
    /// handlers directly to this method.
    pub async fn collect_controlled_with<H>(
        self,
        handler: H,
    ) -> Result<
        TextStepOutcomeWithTools<T>,
        CollectError<TextTurnReductionError, TextTurnStateWithTools<T>>,
    >
    where
        H: TextToolEventHandler<T>,
    {
        let TextTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
            fallback_parser,
            event_handlers,
        } = self;
        assert!(
            event_handlers.is_empty(),
            "use `collect_with` for normal text+tools event handlers; `collect_controlled_with` accepts `TextToolEventHandler` handlers"
        );
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        let staged = match lutum
            .run_text_turn_with_tools(extensions, input, turn, fallback_parser)
            .await
        {
            Ok(pending) => match pending.collect_controlled_with(handler).await {
                Ok(s) => s,
                Err(e) => return Err(e),
            },
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
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

    /// Collect without auto-committing. This returns the same staged tool outcome
    /// shape used internally by `collect()`, so callers can inspect, discard, or
    /// explicitly commit a finished tool-enabled turn.
    pub async fn collect_staged_with<H>(
        self,
        handler: H,
    ) -> Result<
        StagedTextStepOutcomeWithTools<T>,
        CollectError<TextTurnReductionError, TextTurnStateWithTools<T>>,
    >
    where
        H: EventHandler<lutum_protocol::TextTurnEventWithTools<T>, TextTurnStateWithTools<T>>,
    {
        let TextTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
            fallback_parser,
            mut event_handlers,
        } = self;
        event_handlers.push(handler);
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        drop(target);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        match lutum
            .run_text_turn_with_tools(extensions, input, turn, fallback_parser)
            .await
        {
            Ok(pending) => pending
                .collect_with(event_handlers)
                .await
                .map(StagedTextStepOutcomeWithTools::from_staged),
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: TextTurnStateWithTools::default(),
                })
            }
        }
    }

    /// Collect without auto-committing, using the controlled text+tools handler
    /// path.
    ///
    /// Finished turns are returned as staged outcomes. Tool rounds remain
    /// uncommitted until the caller commits the round.
    ///
    /// # Panics
    ///
    /// Panics if normal `on_event` handlers are already registered on the
    /// builder. Use `collect_staged_with` for normal handlers, or pass
    /// controlled handlers directly to this method.
    pub async fn collect_staged_controlled_with<H>(
        self,
        handler: H,
    ) -> Result<
        StagedTextStepOutcomeWithTools<T>,
        CollectError<TextTurnReductionError, TextTurnStateWithTools<T>>,
    >
    where
        H: TextToolEventHandler<T>,
    {
        let TextTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
            fallback_parser,
            event_handlers,
        } = self;
        assert!(
            event_handlers.is_empty(),
            "use `collect_staged_with` for normal text+tools event handlers; `collect_staged_controlled_with` accepts `TextToolEventHandler` handlers"
        );
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        drop(target);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        match lutum
            .run_text_turn_with_tools(extensions, input, turn, fallback_parser)
            .await
        {
            Ok(pending) => pending
                .collect_controlled_with(handler)
                .await
                .map(StagedTextStepOutcomeWithTools::from_staged),
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: TextTurnStateWithTools::default(),
                })
            }
        }
    }

    /// Collect without auto-committing. Returns a staged outcome with either an
    /// uncommitted finished turn, an uncommitted tool round, or no-output metadata.
    pub async fn collect_staged(
        self,
    ) -> Result<
        StagedTextStepOutcomeWithTools<T>,
        CollectError<TextTurnReductionError, TextTurnStateWithTools<T>>,
    > {
        let TextTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
            fallback_parser,
            event_handlers,
        } = self;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        drop(target);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        match lutum
            .run_text_turn_with_tools(extensions, input, turn, fallback_parser)
            .await
        {
            Ok(pending) => pending
                .collect_with(event_handlers)
                .await
                .map(StagedTextStepOutcomeWithTools::from_staged),
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: TextTurnStateWithTools::default(),
                })
            }
        }
    }

    pub async fn collect(
        self,
    ) -> Result<
        TextStepOutcomeWithTools<T>,
        CollectError<TextTurnReductionError, TextTurnStateWithTools<T>>,
    > {
        let TextTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
            fallback_parser,
            event_handlers,
        } = self;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        let staged = match lutum
            .run_text_turn_with_tools(extensions, input, turn, fallback_parser)
            .await
        {
            Ok(pending) => match pending.collect_with(event_handlers).await {
                Ok(s) => s,
                Err(e) => return Err(e),
            },
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
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

pub struct SessionTextTurnWithTools<'a, T>
where
    T: Toolset,
{
    inner: TextTurnWithTools<'a, T>,
}

impl<'a, T> SessionTextTurnWithTools<'a, T>
where
    T: Toolset,
{
    pub fn ext<E>(self, extension: E) -> Self
    where
        E: Send + Sync + 'static,
    {
        Self {
            inner: self.inner.ext(extension),
        }
    }

    pub fn extensions(self, extensions: RequestExtensions) -> Self {
        Self {
            inner: self.inner.extensions(extensions),
        }
    }

    pub fn temperature(self, temperature: Temperature) -> Self {
        Self {
            inner: self.inner.temperature(temperature),
        }
    }

    pub fn max_output_tokens(self, max_output_tokens: u32) -> Self {
        Self {
            inner: self.inner.max_output_tokens(max_output_tokens),
        }
    }

    pub fn seed(self, seed: u64) -> Self {
        Self {
            inner: self.inner.seed(seed),
        }
    }

    pub fn budget(self, budget: RequestBudget) -> Self {
        Self {
            inner: self.inner.budget(budget),
        }
    }

    pub fn generation_config(self, generation: GenerationParams) -> Self {
        Self {
            inner: self.inner.generation_config(generation),
        }
    }

    pub fn retry_policy(self, retry_policy: RetryPolicy) -> Self {
        Self {
            inner: self.inner.retry_policy(retry_policy),
        }
    }

    /// Register a normal per-turn event handler for this tool-enabled text turn.
    ///
    /// Handlers run after the reducer has applied each event, in registration
    /// order. Use `collect_controlled_with` instead when the handler needs the
    /// controlled text+tools directives.
    pub fn on_event<H>(self, handler: H) -> Self
    where
        H: EventHandler<lutum_protocol::TextTurnEventWithTools<T>, TextTurnStateWithTools<T>> + 'a,
    {
        Self {
            inner: self.inner.on_event(handler),
        }
    }

    pub fn available_tools(self, selectors: impl IntoIterator<Item = T::Selector>) -> Self {
        Self {
            inner: self.inner.available_tools(selectors),
        }
    }

    pub fn available_tools_default_plus(
        self,
        selectors: impl IntoIterator<Item = T::Selector>,
    ) -> Self {
        Self {
            inner: self.inner.available_tools_default_plus(selectors),
        }
    }

    pub fn require_any_tool(self) -> Self {
        Self {
            inner: self.inner.require_any_tool(),
        }
    }

    pub fn require_tool(self, selector: T::Selector) -> Self {
        Self {
            inner: self.inner.require_tool(selector),
        }
    }

    pub fn recover_tool_calls_with<P>(self, parser: P) -> Self
    where
        P: TextToolCallFallbackParser<T> + 'static,
    {
        Self {
            inner: self.inner.recover_tool_calls_with(parser),
        }
    }

    pub fn describe_tool(self, selector: T::Selector, description: impl Into<String>) -> Self {
        Self {
            inner: self.inner.describe_tool(selector, description),
        }
    }

    pub fn describe_many_tools(
        self,
        overrides: impl IntoIterator<Item = (T::Selector, String)>,
    ) -> Self {
        Self {
            inner: self.inner.describe_many_tools(overrides),
        }
    }

    /// Start the turn and return the pending stream collector.
    ///
    /// # Panics
    ///
    /// Panics if `on_event` was already called. Builder-registered handlers are
    /// consumed by `collect`, `collect_staged`, and `collect_with`.
    pub async fn start(self, lutum: &Lutum) -> Result<PendingTextTurnWithTools<T>, LutumError> {
        let TextTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
            fallback_parser,
            event_handlers,
        } = self.inner;
        assert!(
            event_handlers.is_empty(),
            "`on_event` handlers are used by collection methods; call `collect`, `collect_staged`, or `collect_with` instead of `start` after registering handlers"
        );
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        lutum
            .run_text_turn_with_tools(extensions, input, turn, fallback_parser)
            .await
    }

    /// Start the turn and return the raw event stream.
    ///
    /// # Panics
    ///
    /// Panics if `on_event` was already called. Builder-registered handlers are
    /// only used by collection methods.
    pub async fn stream(
        self,
        lutum: &Lutum,
    ) -> Result<lutum_protocol::TextTurnEventStreamWithTools<T>, LutumError> {
        Ok(self.start(lutum).await?.into_stream())
    }

    /// Count input tokens for this tool-capable turn without sending a generation request.
    pub async fn count_tokens(&self, lutum: &Lutum) -> Result<Option<TokenCount>, LutumError> {
        let mut extensions = self.inner.extensions.clone();
        let generation = self
            .inner
            .target
            .generation_with_defaults(&mut extensions, &self.inner.turn.config);
        let input = self.inner.target.preview_input(&mut extensions);
        lutum
            .count_text_turn_tokens(extensions, input, &self.inner.turn, generation)
            .await
    }

    pub async fn collect_with<H>(
        self,
        lutum: &Lutum,
        handler: H,
    ) -> Result<
        TextStepOutcomeWithTools<T>,
        CollectError<TextTurnReductionError, TextTurnStateWithTools<T>>,
    >
    where
        H: EventHandler<lutum_protocol::TextTurnEventWithTools<T>, TextTurnStateWithTools<T>>,
    {
        let TextTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
            fallback_parser,
            mut event_handlers,
        } = self.inner;
        event_handlers.push(handler);
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        let staged = match lutum
            .run_text_turn_with_tools(extensions, input, turn, fallback_parser)
            .await
        {
            Ok(pending) => match pending.collect_with(event_handlers).await {
                Ok(s) => s,
                Err(e) => return Err(e),
            },
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
                return Err(CollectError::Execution {
                    source,
                    partial: TextTurnStateWithTools::default(),
                });
            }
        };
        Ok(match target {
            TurnTarget::Session { session } => {
                TextStepOutcomeWithTools::from_staged(staged, Some(session.input_mut()))
            }
            TurnTarget::Lutum { .. } => TextStepOutcomeWithTools::from_staged(staged, None),
        })
    }

    /// Collect a text+tools turn with an advanced handler that can synthesize
    /// a finished turn or tool round.
    ///
    /// # Panics
    ///
    /// Panics if normal `on_event` handlers are already registered on the
    /// builder. Use `collect_with` for normal handlers, or pass controlled
    /// handlers directly to this method.
    pub async fn collect_controlled_with<H>(
        self,
        lutum: &Lutum,
        handler: H,
    ) -> Result<
        TextStepOutcomeWithTools<T>,
        CollectError<TextTurnReductionError, TextTurnStateWithTools<T>>,
    >
    where
        H: TextToolEventHandler<T>,
    {
        let TextTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
            fallback_parser,
            event_handlers,
        } = self.inner;
        assert!(
            event_handlers.is_empty(),
            "use `collect_with` for normal text+tools event handlers; `collect_controlled_with` accepts `TextToolEventHandler` handlers"
        );
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        let staged = match lutum
            .run_text_turn_with_tools(extensions, input, turn, fallback_parser)
            .await
        {
            Ok(pending) => match pending.collect_controlled_with(handler).await {
                Ok(s) => s,
                Err(e) => return Err(e),
            },
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
                return Err(CollectError::Execution {
                    source,
                    partial: TextTurnStateWithTools::default(),
                });
            }
        };
        Ok(match target {
            TurnTarget::Session { session } => {
                TextStepOutcomeWithTools::from_staged(staged, Some(session.input_mut()))
            }
            TurnTarget::Lutum { .. } => TextStepOutcomeWithTools::from_staged(staged, None),
        })
    }

    pub async fn collect_staged_with<H>(
        self,
        lutum: &Lutum,
        handler: H,
    ) -> Result<
        StagedTextStepOutcomeWithTools<T>,
        CollectError<TextTurnReductionError, TextTurnStateWithTools<T>>,
    >
    where
        H: EventHandler<lutum_protocol::TextTurnEventWithTools<T>, TextTurnStateWithTools<T>>,
    {
        let TextTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
            fallback_parser,
            mut event_handlers,
        } = self.inner;
        event_handlers.push(handler);
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        drop(target);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        match lutum
            .run_text_turn_with_tools(extensions, input, turn, fallback_parser)
            .await
        {
            Ok(pending) => pending
                .collect_with(event_handlers)
                .await
                .map(StagedTextStepOutcomeWithTools::from_staged),
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: TextTurnStateWithTools::default(),
                })
            }
        }
    }

    /// Collect without auto-committing, using the controlled text+tools handler
    /// path.
    ///
    /// # Panics
    ///
    /// Panics if normal `on_event` handlers are already registered on the
    /// builder. Use `collect_staged_with` for normal handlers, or pass
    /// controlled handlers directly to this method.
    pub async fn collect_staged_controlled_with<H>(
        self,
        lutum: &Lutum,
        handler: H,
    ) -> Result<
        StagedTextStepOutcomeWithTools<T>,
        CollectError<TextTurnReductionError, TextTurnStateWithTools<T>>,
    >
    where
        H: TextToolEventHandler<T>,
    {
        let TextTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
            fallback_parser,
            event_handlers,
        } = self.inner;
        assert!(
            event_handlers.is_empty(),
            "use `collect_staged_with` for normal text+tools event handlers; `collect_staged_controlled_with` accepts `TextToolEventHandler` handlers"
        );
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        drop(target);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        match lutum
            .run_text_turn_with_tools(extensions, input, turn, fallback_parser)
            .await
        {
            Ok(pending) => pending
                .collect_controlled_with(handler)
                .await
                .map(StagedTextStepOutcomeWithTools::from_staged),
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: TextTurnStateWithTools::default(),
                })
            }
        }
    }

    pub async fn collect_staged(
        self,
        lutum: &Lutum,
    ) -> Result<
        StagedTextStepOutcomeWithTools<T>,
        CollectError<TextTurnReductionError, TextTurnStateWithTools<T>>,
    > {
        let TextTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
            fallback_parser,
            event_handlers,
        } = self.inner;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        drop(target);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        match lutum
            .run_text_turn_with_tools(extensions, input, turn, fallback_parser)
            .await
        {
            Ok(pending) => pending
                .collect_with(event_handlers)
                .await
                .map(StagedTextStepOutcomeWithTools::from_staged),
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: TextTurnStateWithTools::default(),
                })
            }
        }
    }

    pub async fn collect(
        self,
        lutum: &Lutum,
    ) -> Result<
        TextStepOutcomeWithTools<T>,
        CollectError<TextTurnReductionError, TextTurnStateWithTools<T>>,
    > {
        let TextTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
            fallback_parser,
            event_handlers,
        } = self.inner;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        let staged = match lutum
            .run_text_turn_with_tools(extensions, input, turn, fallback_parser)
            .await
        {
            Ok(pending) => match pending.collect_with(event_handlers).await {
                Ok(s) => s,
                Err(e) => return Err(e),
            },
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::TextTurn,
                    &source,
                );
                return Err(CollectError::Execution {
                    source,
                    partial: TextTurnStateWithTools::default(),
                });
            }
        };
        Ok(match target {
            TurnTarget::Session { session } => {
                TextStepOutcomeWithTools::from_staged(staged, Some(session.input_mut()))
            }
            TurnTarget::Lutum { .. } => TextStepOutcomeWithTools::from_staged(staged, None),
        })
    }
}

impl<'a, T> TextTurnWithTools<'a, T>
where
    T: Toolset + HasDynamicSlot,
{
    /// Register runtime-defined tools for this turn.
    ///
    /// Dynamic tools are not persisted on the session. Each turn that wants
    /// them must register them explicitly. Dynamic tools cannot be targeted by
    /// `require_tool`; use `require_any_tool` to require one tool call among
    /// the static and dynamic tools available on this turn.
    pub fn with_dynamic_tools(mut self, tools: impl IntoIterator<Item = DynamicTool>) -> Self {
        self.turn.config.tools.dynamic_tools.extend(tools);
        self
    }
}

impl<'a, T> SessionTextTurnWithTools<'a, T>
where
    T: Toolset + HasDynamicSlot,
{
    pub fn with_dynamic_tools(self, tools: impl IntoIterator<Item = DynamicTool>) -> Self {
        Self {
            inner: self.inner.with_dynamic_tools(tools),
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
    pub(crate) fn from_lutum(lutum: &'a Lutum, input: ModelInput) -> Self {
        Self {
            target: TurnTarget::Lutum { lutum, input },
            extensions: RequestExtensions::new(),
            turn: ProtocolStructuredTurn::new(),
        }
    }

    pub(crate) fn from_session(session: &'a mut Session) -> SessionStructuredTurn<'a, O> {
        SessionStructuredTurn {
            inner: Self {
                target: TurnTarget::Session { session },
                extensions: RequestExtensions::new(),
                turn: ProtocolStructuredTurn::new(),
            },
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

    /// Override the JSON Schema sent to the model for this structured output.
    ///
    /// The response is still deserialized as `O`. Use `serde_json::Value` as
    /// `O` when both the schema and decoded shape are runtime-defined.
    pub fn output_schema(
        mut self,
        schema_name: impl Into<String>,
        schema: impl Into<serde_json::Value>,
    ) -> Self {
        self.turn.output = self.turn.output.with_json_schema(schema_name, schema);
        self
    }

    pub fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.extensions.insert(retry_policy);
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
            mut extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        lutum.run_structured_turn(extensions, input, turn).await
    }

    pub async fn stream(self) -> Result<ProtocolStructuredTurnEventStream<O>, LutumError> {
        Ok(self.start().await?.into_stream())
    }

    /// Count input tokens for this structured turn without sending a generation request.
    ///
    /// Returns `Ok(None)` when no token counter is attached to `Lutum`, or when
    /// the configured adapter surface does not support exact token counting.
    pub async fn count_tokens(&self) -> Result<Option<TokenCount>, LutumError> {
        let mut extensions = self.extensions.clone();
        let generation = self
            .target
            .generation_with_defaults(&mut extensions, &self.turn.config);
        let lutum = self.target.lutum_owned();
        let input = self.target.preview_input(&mut extensions);
        lutum
            .count_structured_turn_tokens(extensions, input, &self.turn, generation)
            .await
    }

    /// Collect the turn with a custom event handler. Always returns a staged result
    /// (never auto-commits). Use [`collect`] for auto-commit or
    /// [`collect_staged`] for staged without a handler.
    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<
        StagedStructuredTurnResult<O>,
        CollectError<StructuredTurnReductionError, StructuredTurnPartial<O>>,
    >
    where
        H: EventHandler<lutum_protocol::StructuredTurnEvent<O>, StructuredTurnCollectedState<O>>,
    {
        let StructuredTurn {
            mut target,
            mut extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        drop(target);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        match lutum.run_structured_turn(extensions, input, turn).await {
            Ok(pending) => pending.collect_with(handler).await,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::StructuredTurn,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: StructuredTurnPartial::from_state(
                        StructuredTurnCollectedState::default(),
                    ),
                })
            }
        }
    }

    /// Collect without auto-committing. Returns a staged result with an
    /// [`UncommittedAssistantTurn`] that you can commit later.
    pub async fn collect_staged(
        self,
    ) -> Result<
        StagedStructuredTurnResult<O>,
        CollectError<StructuredTurnReductionError, StructuredTurnPartial<O>>,
    > {
        let StructuredTurn {
            mut target,
            mut extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        drop(target);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        match lutum.run_structured_turn(extensions, input, turn).await {
            Ok(pending) => pending.collect().await,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::StructuredTurn,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: StructuredTurnPartial::from_state(
                        StructuredTurnCollectedState::default(),
                    ),
                })
            }
        }
    }

    /// Collect and return the completed result directly; use [`collect_staged`]
    /// to keep the uncommitted assistant turn.
    pub async fn collect(
        self,
    ) -> Result<
        lutum_protocol::StructuredTurnResult<O>,
        CollectError<StructuredTurnReductionError, StructuredTurnPartial<O>>,
    > {
        let StructuredTurn {
            mut target,
            mut extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        let staged = match lutum.run_structured_turn(extensions, input, turn).await {
            Ok(pending) => match pending.collect().await {
                Ok(s) => s,
                Err(e) => return Err(e),
            },
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::StructuredTurn,
                    &source,
                );
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
            cumulative_usage: staged.cumulative_usage,
        })
    }
}

pub struct SessionStructuredTurn<'a, O>
where
    O: StructuredOutput,
{
    inner: StructuredTurn<'a, O>,
}

impl<'a, O> SessionStructuredTurn<'a, O>
where
    O: StructuredOutput,
{
    pub fn ext<T>(self, extension: T) -> Self
    where
        T: Send + Sync + 'static,
    {
        Self {
            inner: self.inner.ext(extension),
        }
    }

    pub fn extensions(self, extensions: RequestExtensions) -> Self {
        Self {
            inner: self.inner.extensions(extensions),
        }
    }

    pub fn temperature(self, temperature: Temperature) -> Self {
        Self {
            inner: self.inner.temperature(temperature),
        }
    }

    pub fn max_output_tokens(self, max_output_tokens: u32) -> Self {
        Self {
            inner: self.inner.max_output_tokens(max_output_tokens),
        }
    }

    pub fn seed(self, seed: u64) -> Self {
        Self {
            inner: self.inner.seed(seed),
        }
    }

    pub fn budget(self, budget: RequestBudget) -> Self {
        Self {
            inner: self.inner.budget(budget),
        }
    }

    pub fn generation_config(self, generation: GenerationParams) -> Self {
        Self {
            inner: self.inner.generation_config(generation),
        }
    }

    pub fn output_schema(
        self,
        schema_name: impl Into<String>,
        schema: impl Into<serde_json::Value>,
    ) -> Self {
        Self {
            inner: self.inner.output_schema(schema_name, schema),
        }
    }

    pub fn retry_policy(self, retry_policy: RetryPolicy) -> Self {
        Self {
            inner: self.inner.retry_policy(retry_policy),
        }
    }

    pub fn tools<T>(self) -> SessionStructuredTurnWithTools<'a, T, O>
    where
        T: Toolset,
    {
        let StructuredTurn {
            target,
            extensions,
            turn,
        } = self.inner;
        let turn = ProtocolStructuredTurn {
            config: TurnConfig {
                generation: turn.config.generation,
                tools: ToolConstraints::default(),
                budget: turn.config.budget,
            },
            output: turn.output,
        };
        SessionStructuredTurnWithTools {
            inner: StructuredTurnWithTools {
                target,
                extensions,
                turn,
            },
        }
    }

    pub async fn start(self, lutum: &Lutum) -> Result<PendingStructuredTurn<O>, LutumError> {
        let StructuredTurn {
            mut target,
            mut extensions,
            mut turn,
        } = self.inner;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        lutum.run_structured_turn(extensions, input, turn).await
    }

    pub async fn stream(
        self,
        lutum: &Lutum,
    ) -> Result<ProtocolStructuredTurnEventStream<O>, LutumError> {
        Ok(self.start(lutum).await?.into_stream())
    }

    /// Count input tokens for this structured turn without sending a generation request.
    pub async fn count_tokens(&self, lutum: &Lutum) -> Result<Option<TokenCount>, LutumError> {
        let mut extensions = self.inner.extensions.clone();
        let generation = self
            .inner
            .target
            .generation_with_defaults(&mut extensions, &self.inner.turn.config);
        let input = self.inner.target.preview_input(&mut extensions);
        lutum
            .count_structured_turn_tokens(extensions, input, &self.inner.turn, generation)
            .await
    }

    pub async fn collect_with<H>(
        self,
        lutum: &Lutum,
        handler: H,
    ) -> Result<
        StagedStructuredTurnResult<O>,
        CollectError<StructuredTurnReductionError, StructuredTurnPartial<O>>,
    >
    where
        H: EventHandler<lutum_protocol::StructuredTurnEvent<O>, StructuredTurnCollectedState<O>>,
    {
        let StructuredTurn {
            mut target,
            mut extensions,
            mut turn,
        } = self.inner;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        drop(target);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        match lutum.run_structured_turn(extensions, input, turn).await {
            Ok(pending) => pending.collect_with(handler).await,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::StructuredTurn,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: StructuredTurnPartial::from_state(
                        StructuredTurnCollectedState::default(),
                    ),
                })
            }
        }
    }

    pub async fn collect_staged(
        self,
        lutum: &Lutum,
    ) -> Result<
        StagedStructuredTurnResult<O>,
        CollectError<StructuredTurnReductionError, StructuredTurnPartial<O>>,
    > {
        let StructuredTurn {
            mut target,
            mut extensions,
            mut turn,
        } = self.inner;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        drop(target);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        match lutum.run_structured_turn(extensions, input, turn).await {
            Ok(pending) => pending.collect().await,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::StructuredTurn,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: StructuredTurnPartial::from_state(
                        StructuredTurnCollectedState::default(),
                    ),
                })
            }
        }
    }

    pub async fn collect(
        self,
        lutum: &Lutum,
    ) -> Result<
        lutum_protocol::StructuredTurnResult<O>,
        CollectError<StructuredTurnReductionError, StructuredTurnPartial<O>>,
    > {
        let StructuredTurn {
            mut target,
            mut extensions,
            mut turn,
        } = self.inner;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        let staged = match lutum.run_structured_turn(extensions, input, turn).await {
            Ok(pending) => match pending.collect().await {
                Ok(s) => s,
                Err(e) => return Err(e),
            },
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::StructuredTurn,
                    &source,
                );
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
            cumulative_usage: staged.cumulative_usage,
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

    /// Override the JSON Schema sent to the model for this structured output.
    ///
    /// The response is still deserialized as `O`. Use `serde_json::Value` as
    /// `O` when both the schema and decoded shape are runtime-defined.
    pub fn output_schema(
        mut self,
        schema_name: impl Into<String>,
        schema: impl Into<serde_json::Value>,
    ) -> Self {
        self.turn.output = self.turn.output.with_json_schema(schema_name, schema);
        self
    }

    pub fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.extensions.insert(retry_policy);
        self
    }

    pub fn available_tools(mut self, selectors: impl IntoIterator<Item = T::Selector>) -> Self {
        self.turn.config.tools.available = ToolAvailability::Only(selectors.into_iter().collect());
        self
    }

    /// Expose the default-on toolset *plus* the listed selectors on this turn.
    /// This is the typical way to temporarily re-enable variants marked
    /// `#[tool(off)]` / `#[toolset(off)]` (e.g. a loaded "skill") without
    /// having to enumerate the rest of the default set.
    pub fn available_tools_default_plus(
        mut self,
        selectors: impl IntoIterator<Item = T::Selector>,
    ) -> Self {
        self.turn.config.tools.available =
            ToolAvailability::DefaultPlus(selectors.into_iter().collect());
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
            mut extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        lutum
            .run_structured_turn_with_tools(extensions, input, turn)
            .await
    }

    pub async fn stream(
        self,
    ) -> Result<lutum_protocol::StructuredTurnEventStreamWithTools<T, O>, LutumError> {
        Ok(self.start().await?.into_stream())
    }

    /// Count input tokens for this tool-capable structured turn without sending a generation request.
    ///
    /// Returns `Ok(None)` when no token counter is attached to `Lutum`, or when
    /// the configured adapter surface does not support exact token counting.
    pub async fn count_tokens(&self) -> Result<Option<TokenCount>, LutumError> {
        let mut extensions = self.extensions.clone();
        let generation = self
            .target
            .generation_with_defaults(&mut extensions, &self.turn.config);
        let lutum = self.target.lutum_owned();
        let input = self.target.preview_input(&mut extensions);
        lutum
            .count_structured_turn_tokens(extensions, input, &self.turn, generation)
            .await
    }

    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<
        StructuredStepOutcomeWithTools<T, O>,
        CollectError<StructuredTurnReductionError, StructuredTurnPartialWithTools<T, O>>,
    >
    where
        H: EventHandler<
                lutum_protocol::StructuredTurnEventWithTools<T, O>,
                StructuredTurnStateWithTools<T, O>,
            >,
    {
        let StructuredTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        let pending = match lutum
            .run_structured_turn_with_tools(extensions, input, turn)
            .await
        {
            Ok(p) => p,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::StructuredTurn,
                    &source,
                );
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
                if (!partial.state.tool_calls.is_empty()
                    || !partial.state.recoverable_tool_call_issues.is_empty())
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
                    let recoverable_tool_call_issues =
                        partial.state.recoverable_tool_call_issues.clone();
                    let outcome = StructuredStepOutcomeWithTools::from_partial(
                        assistant_turn,
                        committed_turn,
                        tool_calls,
                        recoverable_tool_call_issues,
                        partial.state.continue_suggestion,
                        partial.state.request_id.clone(),
                        partial.state.model.clone(),
                        finish_reason,
                        usage,
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
        CollectError<StructuredTurnReductionError, StructuredTurnPartialWithTools<T, O>>,
    > {
        let StructuredTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
        } = self;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let lutum = target.lutum_owned();
        let input = target.input(&mut extensions);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        let pending = match lutum
            .run_structured_turn_with_tools(extensions, input, turn)
            .await
        {
            Ok(p) => p,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::StructuredTurn,
                    &source,
                );
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
                if (!partial.state.tool_calls.is_empty()
                    || !partial.state.recoverable_tool_call_issues.is_empty())
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
                    let recoverable_tool_call_issues =
                        partial.state.recoverable_tool_call_issues.clone();
                    let outcome = StructuredStepOutcomeWithTools::from_partial(
                        assistant_turn,
                        committed_turn,
                        tool_calls,
                        recoverable_tool_call_issues,
                        partial.state.continue_suggestion,
                        partial.state.request_id.clone(),
                        partial.state.model.clone(),
                        finish_reason,
                        usage,
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

pub struct SessionStructuredTurnWithTools<'a, T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    inner: StructuredTurnWithTools<'a, T, O>,
}

impl<'a, T, O> SessionStructuredTurnWithTools<'a, T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    pub fn ext<E>(self, extension: E) -> Self
    where
        E: Send + Sync + 'static,
    {
        Self {
            inner: self.inner.ext(extension),
        }
    }

    pub fn extensions(self, extensions: RequestExtensions) -> Self {
        Self {
            inner: self.inner.extensions(extensions),
        }
    }

    pub fn temperature(self, temperature: Temperature) -> Self {
        Self {
            inner: self.inner.temperature(temperature),
        }
    }

    pub fn max_output_tokens(self, max_output_tokens: u32) -> Self {
        Self {
            inner: self.inner.max_output_tokens(max_output_tokens),
        }
    }

    pub fn seed(self, seed: u64) -> Self {
        Self {
            inner: self.inner.seed(seed),
        }
    }

    pub fn budget(self, budget: RequestBudget) -> Self {
        Self {
            inner: self.inner.budget(budget),
        }
    }

    pub fn generation_config(self, generation: GenerationParams) -> Self {
        Self {
            inner: self.inner.generation_config(generation),
        }
    }

    pub fn output_schema(
        self,
        schema_name: impl Into<String>,
        schema: impl Into<serde_json::Value>,
    ) -> Self {
        Self {
            inner: self.inner.output_schema(schema_name, schema),
        }
    }

    pub fn retry_policy(self, retry_policy: RetryPolicy) -> Self {
        Self {
            inner: self.inner.retry_policy(retry_policy),
        }
    }

    pub fn available_tools(self, selectors: impl IntoIterator<Item = T::Selector>) -> Self {
        Self {
            inner: self.inner.available_tools(selectors),
        }
    }

    pub fn available_tools_default_plus(
        self,
        selectors: impl IntoIterator<Item = T::Selector>,
    ) -> Self {
        Self {
            inner: self.inner.available_tools_default_plus(selectors),
        }
    }

    pub fn require_any_tool(self) -> Self {
        Self {
            inner: self.inner.require_any_tool(),
        }
    }

    pub fn require_tool(self, selector: T::Selector) -> Self {
        Self {
            inner: self.inner.require_tool(selector),
        }
    }

    pub fn describe_tool(self, selector: T::Selector, description: impl Into<String>) -> Self {
        Self {
            inner: self.inner.describe_tool(selector, description),
        }
    }

    pub fn describe_many_tools(
        self,
        overrides: impl IntoIterator<Item = (T::Selector, String)>,
    ) -> Self {
        Self {
            inner: self.inner.describe_many_tools(overrides),
        }
    }

    pub async fn start(
        self,
        lutum: &Lutum,
    ) -> Result<PendingStructuredTurnWithTools<T, O>, LutumError> {
        let StructuredTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
        } = self.inner;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        lutum
            .run_structured_turn_with_tools(extensions, input, turn)
            .await
    }

    pub async fn stream(
        self,
        lutum: &Lutum,
    ) -> Result<lutum_protocol::StructuredTurnEventStreamWithTools<T, O>, LutumError> {
        Ok(self.start(lutum).await?.into_stream())
    }

    pub async fn count_tokens(&self, lutum: &Lutum) -> Result<Option<TokenCount>, LutumError> {
        let mut extensions = self.inner.extensions.clone();
        let generation = self
            .inner
            .target
            .generation_with_defaults(&mut extensions, &self.inner.turn.config);
        let input = self.inner.target.preview_input(&mut extensions);
        lutum
            .count_structured_turn_tokens(extensions, input, &self.inner.turn, generation)
            .await
    }

    pub async fn collect_with<H>(
        self,
        lutum: &Lutum,
        handler: H,
    ) -> Result<
        StructuredStepOutcomeWithTools<T, O>,
        CollectError<StructuredTurnReductionError, StructuredTurnPartialWithTools<T, O>>,
    >
    where
        H: EventHandler<
                lutum_protocol::StructuredTurnEventWithTools<T, O>,
                StructuredTurnStateWithTools<T, O>,
            >,
    {
        let StructuredTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
        } = self.inner;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        let pending = match lutum
            .run_structured_turn_with_tools(extensions, input, turn)
            .await
        {
            Ok(p) => p,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::StructuredTurn,
                    &source,
                );
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
                if (!partial.state.tool_calls.is_empty()
                    || !partial.state.recoverable_tool_call_issues.is_empty())
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
                    let recoverable_tool_call_issues =
                        partial.state.recoverable_tool_call_issues.clone();
                    let outcome = StructuredStepOutcomeWithTools::from_partial(
                        assistant_turn,
                        committed_turn,
                        tool_calls,
                        recoverable_tool_call_issues,
                        partial.state.continue_suggestion,
                        partial.state.request_id.clone(),
                        partial.state.model.clone(),
                        finish_reason,
                        usage,
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
        lutum: &Lutum,
    ) -> Result<
        StructuredStepOutcomeWithTools<T, O>,
        CollectError<StructuredTurnReductionError, StructuredTurnPartialWithTools<T, O>>,
    > {
        let StructuredTurnWithTools {
            mut target,
            mut extensions,
            mut turn,
        } = self.inner;
        target.apply_defaults(&mut extensions, &mut turn.config);
        let input = target.input(&mut extensions);
        let raw_collect_errors_enabled = lutum.raw_collect_errors_enabled(&extensions);
        let pending = match lutum
            .run_structured_turn_with_tools(extensions, input, turn)
            .await
        {
            Ok(p) => p,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::StructuredTurn,
                    &source,
                );
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
                if (!partial.state.tool_calls.is_empty()
                    || !partial.state.recoverable_tool_call_issues.is_empty())
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
                    let recoverable_tool_call_issues =
                        partial.state.recoverable_tool_call_issues.clone();
                    let outcome = StructuredStepOutcomeWithTools::from_partial(
                        assistant_turn,
                        committed_turn,
                        tool_calls,
                        recoverable_tool_call_issues,
                        partial.state.continue_suggestion,
                        partial.state.request_id.clone(),
                        partial.state.model.clone(),
                        finish_reason,
                        usage,
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

impl<'a, T, O> StructuredTurnWithTools<'a, T, O>
where
    T: Toolset + HasDynamicSlot,
    O: StructuredOutput,
{
    /// Register runtime-defined tools for this turn.
    ///
    /// Dynamic tools are not persisted on the session. Each turn that wants
    /// them must register them explicitly. Dynamic tools cannot be targeted by
    /// `require_tool`; use `require_any_tool` to require one tool call among
    /// the static and dynamic tools available on this turn.
    pub fn with_dynamic_tools(mut self, tools: impl IntoIterator<Item = DynamicTool>) -> Self {
        self.turn.config.tools.dynamic_tools.extend(tools);
        self
    }
}

impl<'a, T, O> SessionStructuredTurnWithTools<'a, T, O>
where
    T: Toolset + HasDynamicSlot,
    O: StructuredOutput,
{
    pub fn with_dynamic_tools(self, tools: impl IntoIterator<Item = DynamicTool>) -> Self {
        Self {
            inner: self.inner.with_dynamic_tools(tools),
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

    pub fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.extensions.insert(retry_policy);
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

    /// Count input tokens for this completion without sending a generation request.
    ///
    /// Returns `Ok(None)` when no token counter is attached to `Lutum`, or when
    /// the configured adapter surface does not support exact token counting.
    pub async fn count_tokens(&self) -> Result<Option<TokenCount>, LutumError> {
        self.lutum
            .count_completion_tokens(self.extensions.clone(), self.request.clone())
            .await
    }

    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<CompletionTurnResult, CollectError<CompletionReductionError, CompletionTurnState>>
    where
        H: EventHandler<lutum_protocol::CompletionEvent, CompletionTurnState>,
    {
        let raw_collect_errors_enabled = self.lutum.raw_collect_errors_enabled(&self.extensions);
        match self.start().await {
            Ok(pending) => pending.collect_with(handler).await,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::Completion,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: CompletionTurnState::default(),
                })
            }
        }
    }

    pub async fn collect(
        self,
    ) -> Result<CompletionTurnResult, CollectError<CompletionReductionError, CompletionTurnState>>
    {
        let raw_collect_errors_enabled = self.lutum.raw_collect_errors_enabled(&self.extensions);
        match self.start().await {
            Ok(pending) => pending.collect().await,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::Completion,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: CompletionTurnState::default(),
                })
            }
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

    /// Override the JSON Schema sent to the model for this structured output.
    ///
    /// The response is still deserialized as `O`. Use `serde_json::Value` as
    /// `O` when both the schema and decoded shape are runtime-defined.
    pub fn output_schema(
        mut self,
        schema_name: impl Into<String>,
        schema: impl Into<serde_json::Value>,
    ) -> Self {
        self.request.output = self.request.output.with_json_schema(schema_name, schema);
        self
    }

    pub fn retry_policy(mut self, retry_policy: RetryPolicy) -> Self {
        self.extensions.insert(retry_policy);
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

    /// Count input tokens for this structured completion without sending a generation request.
    ///
    /// Returns `Ok(None)` when no token counter is attached to `Lutum`, or when
    /// the configured adapter surface does not support exact token counting.
    pub async fn count_tokens(&self) -> Result<Option<TokenCount>, LutumError> {
        self.lutum
            .count_structured_completion_tokens(self.extensions.clone(), &self.request)
            .await
    }

    pub async fn collect_with<H>(
        self,
        handler: H,
    ) -> Result<
        StructuredCompletionResult<O>,
        CollectError<StructuredCompletionReductionError, StructuredCompletionState<O>>,
    >
    where
        H: EventHandler<lutum_protocol::StructuredCompletionEvent<O>, StructuredCompletionState<O>>,
    {
        let raw_collect_errors_enabled = self.lutum.raw_collect_errors_enabled(&self.extensions);
        match self.start().await {
            Ok(pending) => pending.collect_with(handler).await,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::StructuredCompletion,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: StructuredCompletionState::default(),
                })
            }
        }
    }

    pub async fn collect(
        self,
    ) -> Result<
        StructuredCompletionResult<O>,
        CollectError<StructuredCompletionReductionError, StructuredCompletionState<O>>,
    > {
        let raw_collect_errors_enabled = self.lutum.raw_collect_errors_enabled(&self.extensions);
        match self.start().await {
            Ok(pending) => pending.collect().await,
            Err(source) => {
                emit_pre_stream_collect_error(
                    raw_collect_errors_enabled,
                    OperationKind::StructuredCompletion,
                    &source,
                );
                Err(CollectError::Execution {
                    source,
                    partial: StructuredCompletionState::default(),
                })
            }
        }
    }
}
