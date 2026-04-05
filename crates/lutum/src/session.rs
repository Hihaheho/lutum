use std::collections::BTreeMap;

use lutum_protocol::{
    AssistantTurn, AssistantTurnInputError, AssistantTurnItem, CommittedTurn, FinishReason,
    GenerationParams, InputMessageRole, ModelInput, ModelInputItem, RequestBudget, ToolUse,
    Toolset, TurnConfig, TurnView,
    budget::Usage,
    reducer::{
        StructuredTurnResult as StructuredTurnResultNoTools, StructuredTurnResultWithTools,
        StructuredTurnStateWithTools, TextTurnResult as TextTurnResultNoTools,
        TextTurnResultWithTools,
    },
    structured::StructuredOutput,
};
use thiserror::Error;

use crate::{
    Context,
    builders::{StructuredTurn, TextTurn},
    context::StructuredTurnPartialWithTools,
};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionDefaults {
    pub generation: GenerationParams,
    pub budget: RequestBudget,
}

impl SessionDefaults {
    pub(crate) fn apply<T>(&self, turn: &mut TurnConfig<T>)
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

    pub fn text_turn(&self) -> TextTurn<'_> {
        TextTurn::from_session(self)
    }

    pub fn structured_turn<O>(&self) -> StructuredTurn<'_, O>
    where
        O: StructuredOutput,
    {
        StructuredTurn::from_session(self)
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

    pub(crate) fn snapshot_input(&self) -> ModelInput {
        self.input.clone()
    }

    pub(crate) fn apply_defaults<T>(&self, turn: &mut TurnConfig<T>)
    where
        T: Toolset,
    {
        self.defaults.apply(turn);
    }

    /// Returns committed turns stored directly in the ordered `ModelInput`.
    pub fn list_turns(&self) -> impl Iterator<Item = &dyn TurnView> {
        self.input.items().iter().filter_map(|item| match item {
            ModelInputItem::Turn(turn) => Some(turn.as_ref() as &dyn TurnView),
            _ => None,
        })
    }

    /// Commit a completed no-tools text turn into the session transcript.
    pub fn commit_text(&mut self, result: TextTurnResultNoTools) {
        self.input.push(ModelInputItem::Turn(result.committed_turn));
    }

    /// Commit a completed tool-enabled text turn into the session transcript.
    pub fn commit_text_with_tools<T>(&mut self, result: TextTurnResultWithTools<T>)
    where
        T: Toolset,
    {
        self.input.push(ModelInputItem::Turn(result.committed_turn));
    }

    /// Commit a completed no-tools structured turn into the session transcript.
    pub fn commit_structured<O>(&mut self, result: StructuredTurnResultNoTools<O>)
    where
        O: StructuredOutput,
    {
        self.input.push(ModelInputItem::Turn(result.committed_turn));
    }

    /// Commit a completed tool-enabled structured turn into the session transcript.
    pub fn commit_structured_with_tools<T, O>(&mut self, result: StructuredTurnResultWithTools<T, O>)
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
pub enum TextStepOutcomeWithTools<T: Toolset> {
    Finished(TextTurnResultWithTools<T>),
    NeedsToolResults(ToolRound<T>),
}

impl<T> TextStepOutcomeWithTools<T>
where
    T: Toolset,
{
    pub(crate) fn from_result(result: TextTurnResultWithTools<T>) -> Self {
        let TextTurnResultWithTools {
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
            Self::Finished(TextTurnResultWithTools {
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
pub enum StructuredStepOutcomeWithTools<T: Toolset, O: StructuredOutput> {
    Finished(StructuredTurnResultWithTools<T, O>),
    NeedsToolResults(ToolRound<T>),
}

impl<T, O> StructuredStepOutcomeWithTools<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    pub(crate) fn from_result(result: StructuredTurnResultWithTools<T, O>) -> Self {
        let StructuredTurnResultWithTools {
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
            Self::Finished(StructuredTurnResultWithTools {
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

    pub(crate) fn from_partial(partial: StructuredTurnPartialWithTools<T, O>) -> Option<Self> {
        let StructuredTurnPartialWithTools {
            state,
            committed_turn,
        } = partial;
        if state.finish_reason != Some(FinishReason::ToolCall) || state.tool_calls.is_empty() {
            return None;
        }
        let committed_turn = committed_turn?;
        let StructuredTurnStateWithTools {
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
