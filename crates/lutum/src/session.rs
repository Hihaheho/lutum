use std::collections::BTreeMap;
use std::convert::Infallible;

use lutum_protocol::{
    AssistantTurn, AssistantTurnInputError, AssistantTurnItem, CommittedTurn, FinishReason,
    GenerationParams, InputMessageRole, ModelInput, ModelInputItem, RequestBudget,
    RequestExtensions, StructuredTurn, StructuredTurnEventStream, TextTurn, TextTurnEventStream,
    ToolUse, Toolset, TurnConfig, TurnView, UsageEstimate,
    budget::Usage,
    reducer::{
        StructuredTurnReductionError, StructuredTurnResult, StructuredTurnState, TextTurnResult,
    },
    structured::StructuredOutput,
};
use thiserror::Error;

use crate::{
    CollectError, Context, ContextError, EventHandler, PendingStructuredTurn, PendingTextTurn,
    context::{
        PendingStructuredTurn as CorePendingStructuredTurn, PendingTextTurn as CorePendingTextTurn,
        StructuredTurnPartial as CoreStructuredTurnPartial,
    },
};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionDefaults {
    pub generation: GenerationParams,
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
        if turn.generation.seed.is_none() {
            turn.generation.seed = self.generation.seed;
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
}

impl Session {
    pub fn new(ctx: Context) -> Self {
        Self {
            ctx,
            input: ModelInput::new(),
            defaults: SessionDefaults::default(),
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

    pub fn text_turn<T>(&self) -> TextTurn<T>
    where
        T: Toolset,
    {
        let mut turn = TextTurn::new();
        self.defaults.apply(&mut turn.config);
        turn
    }

    pub fn structured_turn<T, O>(&self) -> StructuredTurn<T, O>
    where
        T: Toolset,
        O: StructuredOutput,
    {
        let mut turn = StructuredTurn::new();
        self.defaults.apply(&mut turn.config);
        turn
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
            .text_turn(extensions, self.input.clone(), turn, estimate)
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
            .structured_turn(extensions, self.input.clone(), turn, estimate)
            .await?;
        Ok(SessionPendingStructured { pending })
    }

    /// Returns committed turns stored directly in the ordered `ModelInput`.
    pub fn list_turns(&self) -> impl Iterator<Item = &dyn TurnView> {
        self.input.items().iter().filter_map(|item| match item {
            ModelInputItem::Turn(turn) => Some(turn.as_ref() as &dyn TurnView),
            _ => None,
        })
    }

    /// Commit a completed text turn into the session transcript.
    pub fn commit_text<T>(&mut self, result: TextTurnResult<T>)
    where
        T: Toolset,
    {
        self.input.push(ModelInputItem::Turn(result.committed_turn));
    }

    /// Commit a completed structured turn into the session transcript.
    pub fn commit_structured<T, O>(&mut self, result: StructuredTurnResult<T, O>)
    where
        T: Toolset,
        O: StructuredOutput,
    {
        self.input.push(ModelInputItem::Turn(result.committed_turn));
    }

    /// Commit a completed tool-call round into the session transcript.
    pub fn commit_tool_round<T>(
        &mut self,
        round: ToolRound<T>,
        tool_uses: impl IntoIterator<Item = ToolUse>,
    ) -> Result<(), AssistantTurnInputError>
    where
        T: Toolset,
    {
        let ordered_tool_uses = validate_and_order_tool_uses(&round, tool_uses)?;
        self.input.push(ModelInputItem::Turn(round.committed_turn));
        for tool_use in ordered_tool_uses {
            self.input.push(ModelInputItem::ToolUse(tool_use));
        }
        Ok(())
    }
}

fn validate_and_order_tool_uses<T>(
    round: &ToolRound<T>,
    tool_uses: impl IntoIterator<Item = ToolUse>,
) -> Result<Vec<ToolUse>, AssistantTurnInputError>
where
    T: Toolset,
{
    let mut tool_use_map = BTreeMap::new();
    for tool_use in tool_uses {
        let duplicate_id = tool_use.id.clone();
        if tool_use_map
            .insert(duplicate_id.clone(), tool_use)
            .is_some()
        {
            return Err(AssistantTurnInputError::DuplicateToolUse { id: duplicate_id });
        }
    }

    let mut ordered = Vec::new();
    for item in round.assistant_turn.items() {
        let AssistantTurnItem::ToolCall {
            id,
            name,
            arguments,
        } = item
        else {
            continue;
        };

        let Some(tool_use) = tool_use_map.remove(id) else {
            return Err(AssistantTurnInputError::MissingToolUse { id: id.clone() });
        };
        if tool_use.name != *name {
            return Err(AssistantTurnInputError::MismatchedToolName {
                id: id.clone(),
                expected: name.clone(),
                actual: tool_use.name,
            });
        }
        if tool_use.arguments != *arguments {
            return Err(AssistantTurnInputError::MismatchedToolArguments {
                id: id.clone(),
                expected: arguments.clone(),
                actual: tool_use.arguments,
            });
        }
        ordered.push(tool_use);
    }

    if let Some((id, _)) = tool_use_map.into_iter().next() {
        return Err(AssistantTurnInputError::ExtraToolUse { id });
    }

    Ok(ordered)
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

    pub fn into_stream(self) -> TextTurnEventStream<T> {
        self.pending.into_stream()
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

#[allow(clippy::result_large_err, clippy::type_complexity)]
fn map_structured_result<T, O, HE>(
    raw: Result<
        StructuredTurnResult<T, O>,
        CollectError<HE, StructuredTurnReductionError, CoreStructuredTurnPartial<T, O>>,
    >,
) -> Result<
    StructuredStepOutcome<T, O>,
    CollectError<HE, StructuredTurnReductionError, CoreStructuredTurnPartial<T, O>>,
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
            let partial_for_outcome = CoreStructuredTurnPartial {
                state: partial.state.clone(),
                committed_turn: partial.committed_turn.clone(),
            };
            if let Some(outcome) = StructuredStepOutcome::from_partial(partial_for_outcome) {
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

    pub fn into_stream(self) -> StructuredTurnEventStream<T, O> {
        self.pending.into_stream()
    }

    pub async fn collect<H>(
        self,
        handler: H,
    ) -> Result<
        StructuredStepOutcome<T, O>,
        CollectError<
            H::Error,
            crate::StructuredTurnReductionError,
            CoreStructuredTurnPartial<T, O>,
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
            CoreStructuredTurnPartial<T, O>,
        >,
    > {
        map_structured_result(self.pending.collect_noop().await)
    }
}

#[derive(Clone, Debug)]
pub struct ToolRound<T: Toolset> {
    pub request_id: Option<String>,
    pub model: String,
    pub assistant_turn: AssistantTurn,
    pub tool_calls: Vec<T::ToolCall>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub committed_turn: CommittedTurn,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ToolRoundArityError {
    #[error("expected exactly one tool call, got {actual}")]
    ExpectedOne { actual: usize },
    #[error("expected at most one tool call, got {actual}")]
    ExpectedAtMostOne { actual: usize },
}

impl<T> ToolRound<T>
where
    T: Toolset,
{
    pub fn tool_count(&self) -> usize {
        self.tool_calls.len()
    }

    pub fn expect_one(self) -> Result<T::ToolCall, ToolRoundArityError> {
        let actual = self.tool_calls.len();
        if actual != 1 {
            return Err(ToolRoundArityError::ExpectedOne { actual });
        }
        Ok(self
            .tool_calls
            .into_iter()
            .next()
            .expect("length checked above"))
    }

    pub fn expect_at_most_one(self) -> Result<Option<T::ToolCall>, ToolRoundArityError> {
        let actual = self.tool_calls.len();
        if actual > 1 {
            return Err(ToolRoundArityError::ExpectedAtMostOne { actual });
        }
        Ok(self.tool_calls.into_iter().next())
    }

    pub fn into_tool_calls(self) -> Vec<T::ToolCall> {
        self.tool_calls
    }
}

#[derive(Clone, Debug)]
pub enum TextStepOutcome<T: Toolset> {
    Finished(TextTurnResult<T>),
    NeedsToolResults(ToolRound<T>),
}

impl<T> TextStepOutcome<T>
where
    T: Toolset,
{
    fn from_result(result: TextTurnResult<T>) -> Self {
        let TextTurnResult {
            request_id,
            model,
            assistant_turn,
            tool_calls,
            finish_reason,
            usage,
            committed_turn,
        } = result;
        if finish_reason == FinishReason::ToolCall && !tool_calls.is_empty() {
            Self::NeedsToolResults(ToolRound {
                request_id,
                model,
                assistant_turn,
                tool_calls,
                finish_reason,
                usage,
                committed_turn,
            })
        } else {
            Self::Finished(TextTurnResult {
                request_id,
                model,
                assistant_turn,
                tool_calls,
                finish_reason,
                usage,
                committed_turn,
            })
        }
    }
}

#[derive(Clone, Debug)]
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
        let StructuredTurnResult {
            request_id,
            model,
            assistant_turn,
            tool_calls,
            semantic,
            finish_reason,
            usage,
            committed_turn,
        } = result;
        if finish_reason == FinishReason::ToolCall && !tool_calls.is_empty() {
            Self::NeedsToolResults(ToolRound {
                request_id,
                model,
                assistant_turn,
                tool_calls,
                finish_reason,
                usage,
                committed_turn,
            })
        } else {
            Self::Finished(StructuredTurnResult {
                request_id,
                model,
                assistant_turn,
                tool_calls,
                semantic,
                finish_reason,
                usage,
                committed_turn,
            })
        }
    }

    fn from_partial(partial: CoreStructuredTurnPartial<T, O>) -> Option<Self> {
        let CoreStructuredTurnPartial {
            state,
            committed_turn,
        } = partial;
        if state.finish_reason != Some(FinishReason::ToolCall) || state.tool_calls.is_empty() {
            return None;
        }
        let committed_turn = committed_turn?;
        let StructuredTurnState {
            request_id,
            model,
            assistant_turn,
            tool_calls,
            usage,
            ..
        } = state;
        let assistant_turn = AssistantTurn::from_items(assistant_turn).ok()?;
        let usage = usage?;
        Some(Self::NeedsToolResults(ToolRound {
            request_id,
            model,
            assistant_turn,
            tool_calls,
            finish_reason: FinishReason::ToolCall,
            usage,
            committed_turn,
        }))
    }
}
