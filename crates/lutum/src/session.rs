use std::collections::BTreeMap;

use lutum_protocol::{
    AssistantTurn, AssistantTurnInputError, AssistantTurnItem, CommittedTurn, FinishReason,
    GenerationParams, InputMessageRole, IntoToolResult, ModelInput, ModelInputItem, RequestBudget,
    ToolMetadata, ToolResult, ToolResultError, Toolset, TurnConfig, TurnView,
    UncommittedAssistantTurn,
    budget::Usage,
    reducer::{
        StagedStructuredTurnResultWithTools, StagedTextTurnResultWithTools,
        StructuredTurnResultWithTools, TextTurnResultWithTools,
    },
    structured::StructuredOutput,
};
use thiserror::Error;

use crate::{
    Lutum,
    agent_loop::AgentLoop,
    builders::{StructuredTurn, TextTurn},
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
    lutum: Lutum,
    input: ModelInput,
    defaults: SessionDefaults,
}

impl Session {
    pub fn new(lutum: Lutum) -> Self {
        Self {
            lutum,
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

    pub fn lutum(&self) -> &Lutum {
        &self.lutum
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

    /// Create a text turn builder. Calling `collect()` on the builder will auto-commit the turn
    /// to this session. Use `collect_staged()` to opt out of auto-commit.
    pub fn text_turn(&mut self) -> TextTurn<'_> {
        TextTurn::from_session(self)
    }

    /// Create an [`AgentLoop`] builder for running a tool-calling agentic loop on this session.
    ///
    /// The loop drives the model through tool calls until it produces a text-only
    /// response or the round limit is reached.
    pub fn agent_loop<T: Toolset>(&mut self) -> AgentLoop<'_, T> {
        AgentLoop::new(self)
    }

    /// Create a structured turn builder. Calling `collect()` on the builder will auto-commit the
    /// turn to this session. Use `collect_staged()` to opt out of auto-commit.
    pub fn structured_turn<O>(&mut self) -> StructuredTurn<'_, O>
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
}

/// Extension trait that adds `commit(&mut Session)` to [`UncommittedAssistantTurn`].
///
/// Import this trait to use `turn.commit(&mut session)` syntax.
pub trait CommitTurn {
    fn commit(self, session: &mut Session);
}

impl CommitTurn for UncommittedAssistantTurn {
    fn commit(self, session: &mut Session) {
        self.commit_into(session.input_mut());
    }
}

/// An assistant turn round that needs tool results before it can be committed.
///
/// The assistant turn is NOT committed to the session yet. After executing the tool calls,
/// commit everything atomically with [`UncommittedToolRound::commit`].
#[derive(Debug)]
#[must_use = "call .commit() to commit the turn and tool results, or .discard() to opt out"]
pub struct UncommittedToolRound<T: Toolset> {
    pub request_id: Option<String>,
    pub model: String,
    pub tool_calls: Vec<T::ToolCall>,
    /// Tool calls that were rejected because the tool name was not in the availability set.
    /// These are NOT executed. Inspect them to decide how to respond (e.g., return an error
    /// to the model, retry with corrected constraints, or ignore).
    pub invalid_tool_calls: Vec<ToolMetadata>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    turn: UncommittedAssistantTurn,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ToolRoundArityError {
    #[error("expected exactly one tool call, got {actual}")]
    ExpectedOne { actual: usize },
    #[error("expected at most one tool call, got {actual}")]
    ExpectedAtMostOne { actual: usize },
}

#[derive(Debug, Error)]
pub enum ToolRoundCommitError {
    #[error("failed to build tool result: {0}")]
    ToolResult(#[from] ToolResultError),
    #[error("tool round validation failed: {0}")]
    AssistantTurn(#[from] AssistantTurnInputError),
}

impl<T> UncommittedToolRound<T>
where
    T: Toolset,
{
    pub fn tool_count(&self) -> usize {
        self.tool_calls.len()
    }

    /// Validate that there is exactly one tool call. Non-consuming.
    pub fn expect_one(&self) -> Result<&T::ToolCall, ToolRoundArityError> {
        match self.tool_calls.as_slice() {
            [one] => Ok(one),
            calls => Err(ToolRoundArityError::ExpectedOne {
                actual: calls.len(),
            }),
        }
    }

    /// Validate that there is at most one tool call. Non-consuming.
    pub fn expect_at_most_one(&self) -> Result<Option<&T::ToolCall>, ToolRoundArityError> {
        match self.tool_calls.as_slice() {
            [] => Ok(None),
            [one] => Ok(Some(one)),
            calls => Err(ToolRoundArityError::ExpectedAtMostOne {
                actual: calls.len(),
            }),
        }
    }

    /// Validate `tool_results`, then commit the assistant turn and all tool results to the session.
    ///
    /// Returns an error if the tool results don't match the assistant turn's tool calls (missing,
    /// extra, or mismatched id/name/arguments).
    pub fn commit<I, R>(
        self,
        session: &mut Session,
        tool_results: I,
    ) -> Result<(), ToolRoundCommitError>
    where
        T: Toolset,
        I: IntoIterator<Item = R>,
        R: IntoToolResult,
    {
        let tool_results = tool_results
            .into_iter()
            .map(IntoToolResult::into_tool_result)
            .collect::<Result<Vec<_>, _>>()?;
        let ordered_tool_results = validate_and_order_tool_results(&self.turn, tool_results)?;
        self.turn.commit_into(session.input_mut());
        for tool_result in ordered_tool_results {
            session.input.push(ModelInputItem::ToolResult(tool_result));
        }
        Ok(())
    }

    /// Explicitly discard this round without committing.
    pub fn discard(self) {}
}

fn validate_and_order_tool_results(
    assistant_turn: &AssistantTurn,
    tool_results: impl IntoIterator<Item = ToolResult>,
) -> Result<Vec<ToolResult>, AssistantTurnInputError> {
    let mut tool_result_map = BTreeMap::new();
    for tool_result in tool_results {
        let duplicate_id = tool_result.id.clone();
        if tool_result_map
            .insert(duplicate_id.clone(), tool_result)
            .is_some()
        {
            return Err(AssistantTurnInputError::DuplicateToolResult { id: duplicate_id });
        }
    }

    let mut ordered = Vec::new();
    for item in assistant_turn.items() {
        let AssistantTurnItem::ToolCall {
            id,
            name,
            arguments,
        } = item
        else {
            continue;
        };

        let Some(tool_result) = tool_result_map.remove(id) else {
            return Err(AssistantTurnInputError::MissingToolResult { id: id.clone() });
        };
        if tool_result.name != *name {
            return Err(AssistantTurnInputError::MismatchedToolName {
                id: id.clone(),
                expected: name.clone(),
                actual: tool_result.name,
            });
        }
        if tool_result.arguments != *arguments {
            return Err(AssistantTurnInputError::MismatchedToolArguments {
                id: id.clone(),
                expected: arguments.clone(),
                actual: tool_result.arguments,
            });
        }
        ordered.push(tool_result);
    }

    if let Some((id, _)) = tool_result_map.into_iter().next() {
        return Err(AssistantTurnInputError::ExtraToolResult { id });
    }

    Ok(ordered)
}

#[derive(Debug)]
pub enum TextStepOutcomeWithTools<T: Toolset> {
    /// The model finished without requesting tool calls. The assistant turn has already been
    /// committed to the session.
    Finished(TextTurnResultWithTools<T>),
    /// The model requested tool calls. Execute them, then call `round.commit(&mut session, uses)`.
    NeedsTools(UncommittedToolRound<T>),
}

impl<T> TextStepOutcomeWithTools<T>
where
    T: Toolset,
{
    pub(crate) fn from_staged(
        staged: StagedTextTurnResultWithTools<T>,
        session: Option<&mut ModelInput>,
    ) -> Self {
        if staged.finish_reason == FinishReason::ToolCall
            && (!staged.tool_calls.is_empty() || !staged.invalid_tool_calls.is_empty())
        {
            let StagedTextTurnResultWithTools {
                request_id,
                model,
                turn,
                tool_calls,
                invalid_tool_calls,
                finish_reason,
                usage,
            } = staged;
            Self::NeedsTools(UncommittedToolRound {
                request_id,
                model,
                turn,
                tool_calls,
                invalid_tool_calls,
                finish_reason,
                usage,
            })
        } else {
            let StagedTextTurnResultWithTools {
                request_id,
                model,
                turn,
                tool_calls,
                invalid_tool_calls,
                finish_reason,
                usage,
            } = staged;
            let assistant_turn = turn.assistant_turn().clone();
            if let Some(input) = session {
                turn.commit_into(input);
            } else {
                turn.discard();
            }
            Self::Finished(TextTurnResultWithTools {
                request_id,
                model,
                assistant_turn,
                tool_calls,
                invalid_tool_calls,
                finish_reason,
                usage,
            })
        }
    }
}

#[derive(Debug)]
pub enum StructuredStepOutcomeWithTools<T: Toolset, O: StructuredOutput> {
    /// The model finished without requesting tool calls. The assistant turn has already been
    /// committed to the session.
    Finished(StructuredTurnResultWithTools<T, O>),
    /// The model requested tool calls. Execute them, then call `round.commit(&mut session, uses)`.
    NeedsTools(UncommittedToolRound<T>),
}

impl<T, O> StructuredStepOutcomeWithTools<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    pub(crate) fn from_staged(
        staged: StagedStructuredTurnResultWithTools<T, O>,
        session: Option<&mut ModelInput>,
    ) -> Self {
        if staged.finish_reason == FinishReason::ToolCall
            && (!staged.tool_calls.is_empty() || !staged.invalid_tool_calls.is_empty())
        {
            let StagedStructuredTurnResultWithTools {
                request_id,
                model,
                turn,
                tool_calls,
                invalid_tool_calls,
                semantic: _,
                finish_reason,
                usage,
            } = staged;
            Self::NeedsTools(UncommittedToolRound {
                request_id,
                model,
                turn,
                tool_calls,
                invalid_tool_calls,
                finish_reason,
                usage,
            })
        } else {
            let StagedStructuredTurnResultWithTools {
                request_id,
                model,
                turn,
                tool_calls,
                invalid_tool_calls,
                semantic,
                finish_reason,
                usage,
            } = staged;
            let assistant_turn = turn.assistant_turn().clone();
            if let Some(input) = session {
                turn.commit_into(input);
            } else {
                turn.discard();
            }
            Self::Finished(StructuredTurnResultWithTools {
                request_id,
                model,
                assistant_turn,
                tool_calls,
                invalid_tool_calls,
                semantic,
                finish_reason,
                usage,
            })
        }
    }

    pub(crate) fn from_partial(
        assistant_turn: AssistantTurn,
        committed_turn: CommittedTurn,
        tool_calls: Vec<T::ToolCall>,
        request_id: Option<String>,
        model: String,
        finish_reason: FinishReason,
        usage: Usage,
    ) -> Self {
        Self::NeedsTools(UncommittedToolRound {
            request_id,
            model,
            turn: UncommittedAssistantTurn::new(assistant_turn, committed_turn),
            tool_calls,
            invalid_tool_calls: Vec::new(),
            finish_reason,
            usage,
        })
    }
}
