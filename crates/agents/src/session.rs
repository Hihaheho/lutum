use std::{convert::Infallible, sync::Arc};

use agents_protocol::{
    AssistantTurn, AssistantTurnInputError, AssistantTurnView, CommittedTurn, FinishReason,
    GenerationParams, InputMessageRole, ModelInput, ModelInputItem, ModelName, ReasoningParams,
    RequestBudget, RequestExtensions, StructuredTurn, TextTurn, ToolUse, Toolset, TurnConfig,
    TurnView, UsageEstimate,
    budget::Usage,
    reducer::{
        StructuredTurnReductionError, StructuredTurnResult, StructuredTurnState, TextTurnResult,
    },
    structured::StructuredOutput,
};

use crate::{
    CollectError, Context, ContextError, EventHandler, PendingStructuredTurn, PendingTextTurn,
    context::{
        PendingStructuredTurn as CorePendingStructuredTurn, PendingTextTurn as CorePendingTextTurn,
    },
};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionDefaults {
    pub model: Option<ModelName>,
    pub generation: GenerationParams,
    pub reasoning: ReasoningParams,
    pub budget: RequestBudget,
}

impl SessionDefaults {
    fn apply<T>(&self, turn: &mut TurnConfig<T>)
    where
        T: Toolset,
    {
        if turn.generation.temperature.is_none() {
            turn.generation.temperature = self.generation.temperature;
        }
        if turn.generation.max_output_tokens.is_none() {
            turn.generation.max_output_tokens = self.generation.max_output_tokens;
        }
        if turn.reasoning.effort.is_none() {
            turn.reasoning.effort = self.reasoning.effort;
        }
        if turn.reasoning.summary.is_none() {
            turn.reasoning.summary = self.reasoning.summary;
        }
        if turn.budget == RequestBudget::unlimited() && self.budget != RequestBudget::unlimited() {
            turn.budget = self.budget;
        }
    }
}

#[derive(Clone)]
pub struct Session {
    ctx: Context,
    input: ModelInput,
    defaults: SessionDefaults,
    /// Committed turns in the order they were added.
    ///
    /// Each element is an `Arc` so that cloned sessions share the same turn
    /// objects cheaply and future commits on one branch do not affect the other.
    turns: Vec<CommittedTurn>,
}

impl Session {
    pub fn new(ctx: Context) -> Self {
        Self {
            ctx,
            input: ModelInput::new(),
            defaults: SessionDefaults::default(),
            turns: Vec::new(),
        }
    }

    /// Restores a session from a previously captured [`ModelInput`] snapshot.
    ///
    /// This only restores request input. Committed turns are not encoded in the
    /// snapshot, so a restored session starts with an empty
    /// [`list_turns`](Self::list_turns) view.
    pub fn from_snapshot(ctx: Context, input: ModelInput) -> Self {
        Self {
            ctx,
            input,
            defaults: SessionDefaults::default(),
            turns: Vec::new(),
        }
    }

    pub fn with_defaults(mut self, defaults: SessionDefaults) -> Self {
        self.defaults = defaults;
        self
    }

    pub fn defaults(&self) -> &SessionDefaults {
        &self.defaults
    }

    pub fn context(&self) -> &Context {
        &self.ctx
    }

    pub fn input(&self) -> &ModelInput {
        &self.input
    }

    pub fn input_mut(&mut self) -> &mut ModelInput {
        &mut self.input
    }

    pub fn into_input(self) -> ModelInput {
        self.input
    }

    /// Returns a copy of the current [`ModelInput`].
    ///
    /// This snapshot captures only request input and does not include the
    /// committed-turn list maintained by [`list_turns`](Self::list_turns).
    pub fn snapshot(&self) -> ModelInput {
        self.input.clone()
    }

    pub fn text_turn<T>(&self) -> Option<TextTurn<T>>
    where
        T: Toolset,
    {
        let model = self.defaults.model.clone()?;
        let mut turn = TextTurn::new(model);
        self.defaults.apply(&mut turn.config);
        Some(turn)
    }

    pub fn structured_turn<T, O>(&self) -> Option<StructuredTurn<T, O>>
    where
        T: Toolset,
        O: StructuredOutput,
    {
        let model = self.defaults.model.clone()?;
        let mut turn = StructuredTurn::new(model);
        self.defaults.apply(&mut turn.config);
        Some(turn)
    }

    pub fn push_system(&mut self, text: impl Into<String>) {
        self.input
            .push(ModelInputItem::text(InputMessageRole::System, text));
    }

    pub fn push_developer(&mut self, text: impl Into<String>) {
        self.input
            .push(ModelInputItem::text(InputMessageRole::Developer, text));
    }

    pub fn push_user(&mut self, text: impl Into<String>) {
        self.input
            .push(ModelInputItem::text(InputMessageRole::User, text));
    }

    pub fn append_assistant_turn(
        &mut self,
        turn: AssistantTurn,
        tool_uses: impl IntoIterator<Item = ToolUse>,
    ) -> Result<(), AssistantTurnInputError> {
        self.input.append_assistant_turn(turn, tool_uses)
    }

    pub async fn prepare_text<T>(
        &self,
        extensions: RequestExtensions,
        mut turn: TextTurn<T>,
        estimate: UsageEstimate,
    ) -> Result<SessionPendingText<T>, ContextError>
    where
        T: Toolset,
    {
        self.defaults.apply(&mut turn.config);
        let pending = self
            .ctx
            .responses_text(extensions, self.input.clone(), turn, estimate)
            .await?;
        Ok(SessionPendingText { pending })
    }

    pub async fn prepare_structured<T, O>(
        &self,
        extensions: RequestExtensions,
        mut turn: StructuredTurn<T, O>,
        estimate: UsageEstimate,
    ) -> Result<SessionPendingStructured<T, O>, ContextError>
    where
        T: Toolset,
        O: StructuredOutput,
    {
        self.defaults.apply(&mut turn.config);
        let pending = self
            .ctx
            .responses_structured(extensions, self.input.clone(), turn, estimate)
            .await?;
        Ok(SessionPendingStructured { pending })
    }

    /// Returns an iterator over committed assistant turns in the order they
    /// were committed.
    ///
    /// **Scope:** this iterator covers only turns that were explicitly stored
    /// via `commit_text`, `commit_structured`, or `commit_tool_round`.  User,
    /// system, and developer messages pushed via `push_*` are part of the
    /// request input (`ModelInput`) but are not "committed turns" in the
    /// design's sense — they are not produced by the model and do not go
    /// through the commit path.
    ///
    /// **Interim storage:** committed turns are currently stored using the
    /// core-provided [`AssistantTurnView`] fallback, which preserves all
    /// item data.  Once adapters expose their own exact committed-turn types
    /// through the commit path, callers will see richer provider-specific
    /// views here.  See [`OpenAiCommittedTurn`](agents_openai::OpenAiCommittedTurn)
    /// for the intended adapter-side ownership pattern.
    pub fn list_turns(&self) -> impl Iterator<Item = &dyn TurnView> {
        self.turns.iter().map(|t| t.as_ref() as &dyn TurnView)
    }

    /// Commit a completed text turn into the session transcript.
    ///
    /// Appends the assistant output to `ModelInput` and records a
    /// [`TurnView`] entry in the committed-turns list so `list_turns()` can
    /// surface it.
    ///
    /// **Note:** the current implementation stores an [`AssistantTurnView`]
    /// (core-provided) as the committed turn.  Full exact-turn storage —
    /// where the adapter supplies its own provider-native type — is a
    /// planned follow-up that requires threading an opaque payload through
    /// the result types.
    pub fn commit_text<T>(
        &mut self,
        result: TextTurnResult<T>,
    ) -> Result<(), AssistantTurnInputError>
    where
        T: Toolset,
    {
        let view = AssistantTurnView::from_items(result.assistant_turn.items());
        self.append_assistant_turn(result.assistant_turn, std::iter::empty::<ToolUse>())?;
        self.turns.push(Arc::new(view));
        Ok(())
    }

    /// Commit a completed structured turn into the session transcript.
    ///
    /// See [`commit_text`](Self::commit_text) for notes on the interim
    /// [`AssistantTurnView`] storage.
    pub fn commit_structured<T, O>(
        &mut self,
        result: StructuredTurnResult<T, O>,
    ) -> Result<(), AssistantTurnInputError>
    where
        T: Toolset,
        O: StructuredOutput,
    {
        let view = AssistantTurnView::from_items(result.assistant_turn.items());
        self.append_assistant_turn(result.assistant_turn, std::iter::empty::<ToolUse>())?;
        self.turns.push(Arc::new(view));
        Ok(())
    }

    /// Commit a completed tool-call round into the session transcript.
    ///
    /// Records the assistant output (tool calls) as a committed turn.
    /// Tool results are stored in `ModelInput` (paired with tool calls) but
    /// are not separately recorded as committed turns — they are part of the
    /// same round's assistant output.
    ///
    /// See [`commit_text`](Self::commit_text) for notes on interim storage.
    pub fn commit_tool_round<T>(
        &mut self,
        round: ToolRound<T>,
        tool_uses: impl IntoIterator<Item = ToolUse>,
    ) -> Result<(), AssistantTurnInputError>
    where
        T: Toolset,
    {
        let view = AssistantTurnView::from_items(round.assistant_turn.items());
        self.append_assistant_turn(round.assistant_turn, tool_uses)?;
        self.turns.push(Arc::new(view));
        Ok(())
    }
}

pub struct SessionPendingText<T>
where
    T: Toolset,
{
    pending: CorePendingTextTurn<T>,
}

impl<T> SessionPendingText<T>
where
    T: Toolset,
{
    pub fn into_pending(self) -> PendingTextTurn<T> {
        self.pending
    }

    pub async fn collect<H>(
        self,
        handler: H,
    ) -> Result<
        TextStepOutcome<T>,
        CollectError<H::Error, crate::TextTurnReductionError, crate::TextTurnState<T>>,
    >
    where
        H: EventHandler<crate::TextTurnEvent<T>, crate::TextTurnState<T>>,
    {
        let result = self.pending.collect(handler).await?;
        Ok(TextStepOutcome::from_result(result))
    }

    pub async fn collect_noop(
        self,
    ) -> Result<
        TextStepOutcome<T>,
        CollectError<Infallible, crate::TextTurnReductionError, crate::TextTurnState<T>>,
    > {
        let result = self.pending.collect_noop().await?;
        Ok(TextStepOutcome::from_result(result))
    }
}

fn map_structured_result<T, O, HE>(
    raw: Result<
        StructuredTurnResult<T, O>,
        CollectError<HE, StructuredTurnReductionError, crate::StructuredTurnState<T, O>>,
    >,
) -> Result<
    StructuredStepOutcome<T, O>,
    CollectError<HE, StructuredTurnReductionError, crate::StructuredTurnState<T, O>>,
>
where
    T: Toolset,
    O: StructuredOutput,
{
    match raw {
        Ok(result) => Ok(StructuredStepOutcome::from_result(result)),
        Err(CollectError::Reduction {
            source: StructuredTurnReductionError::MissingSemantic,
            partial,
        }) => {
            if let Some(outcome) = StructuredStepOutcome::from_partial(partial.clone()) {
                Ok(outcome)
            } else {
                Err(CollectError::Reduction {
                    source: StructuredTurnReductionError::MissingSemantic,
                    partial,
                })
            }
        }
        Err(err) => Err(err),
    }
}

pub struct SessionPendingStructured<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    pending: CorePendingStructuredTurn<T, O>,
}

impl<T, O> SessionPendingStructured<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    pub fn into_pending(self) -> PendingStructuredTurn<T, O> {
        self.pending
    }

    pub async fn collect<H>(
        self,
        handler: H,
    ) -> Result<
        StructuredStepOutcome<T, O>,
        CollectError<
            H::Error,
            crate::StructuredTurnReductionError,
            crate::StructuredTurnState<T, O>,
        >,
    >
    where
        H: EventHandler<crate::StructuredTurnEvent<T, O>, crate::StructuredTurnState<T, O>>,
    {
        map_structured_result(self.pending.collect(handler).await)
    }

    pub async fn collect_noop(
        self,
    ) -> Result<
        StructuredStepOutcome<T, O>,
        CollectError<
            Infallible,
            crate::StructuredTurnReductionError,
            crate::StructuredTurnState<T, O>,
        >,
    > {
        map_structured_result(self.pending.collect_noop().await)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ToolRound<T: Toolset> {
    pub request_id: Option<String>,
    pub model: String,
    pub assistant_turn: AssistantTurn,
    pub tool_calls: Vec<T::ToolCall>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

impl<T> ToolRound<T>
where
    T: Toolset,
{
    pub fn into_input_items(
        self,
        tool_uses: impl IntoIterator<Item = ToolUse>,
    ) -> Result<Vec<ModelInputItem>, AssistantTurnInputError> {
        self.assistant_turn.into_input_items(tool_uses)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextStepOutcome<T: Toolset> {
    Finished(TextTurnResult<T>),
    NeedsToolResults(ToolRound<T>),
}

impl<T> TextStepOutcome<T>
where
    T: Toolset,
{
    fn from_result(result: TextTurnResult<T>) -> Self {
        if result.finish_reason == FinishReason::ToolCall && !result.tool_calls.is_empty() {
            Self::NeedsToolResults(ToolRound {
                request_id: result.request_id,
                model: result.model,
                assistant_turn: result.assistant_turn,
                tool_calls: result.tool_calls,
                finish_reason: result.finish_reason,
                usage: result.usage,
            })
        } else {
            Self::Finished(result)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StructuredStepOutcome<T: Toolset, O: StructuredOutput> {
    Finished(StructuredTurnResult<T, O>),
    NeedsToolResults(ToolRound<T>),
}

impl<T, O> StructuredStepOutcome<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    fn from_result(result: StructuredTurnResult<T, O>) -> Self {
        if result.finish_reason == FinishReason::ToolCall && !result.tool_calls.is_empty() {
            Self::NeedsToolResults(ToolRound {
                request_id: result.request_id,
                model: result.model,
                assistant_turn: result.assistant_turn,
                tool_calls: result.tool_calls,
                finish_reason: result.finish_reason,
                usage: result.usage,
            })
        } else {
            Self::Finished(result)
        }
    }

    fn from_partial(partial: StructuredTurnState<T, O>) -> Option<Self> {
        if partial.finish_reason != Some(FinishReason::ToolCall) || partial.tool_calls.is_empty() {
            return None;
        }
        let assistant_turn = AssistantTurn::from_items(partial.assistant_turn).ok()?;
        let usage = partial.usage?;
        Some(Self::NeedsToolResults(ToolRound {
            request_id: partial.request_id,
            model: partial.model,
            assistant_turn,
            tool_calls: partial.tool_calls,
            finish_reason: FinishReason::ToolCall,
            usage,
        }))
    }
}
