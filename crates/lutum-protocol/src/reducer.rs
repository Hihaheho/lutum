use std::sync::Arc;

use thiserror::Error;

use crate::{
    budget::Usage,
    conversation::{
        AssistantTurn, AssistantTurnItem, RawJson, ToolCallId, ToolMetadata,
        UncommittedAssistantTurn,
    },
    llm::{
        CompletionEvent, FinishReason, StructuredCompletionEvent, StructuredTurnEvent,
        StructuredTurnEventWithTools, TextTurnEvent, TextTurnEventWithTools,
    },
    structured::StructuredOutput,
    toolset::{ContinueSuggestionReason, RecoverableToolCallIssue, ToolCallWrapper, Toolset},
    transcript::CommittedTurn,
};

#[derive(Debug, Default)]
pub struct TextTurnState {
    pub request_id: Option<String>,
    pub model: String,
    pub assistant_turn: Vec<AssistantTurnItem>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    pub committed_turn: Option<CommittedTurn>,
    pub event_count: u32,
}

impl Clone for TextTurnState {
    fn clone(&self) -> Self {
        Self {
            request_id: self.request_id.clone(),
            model: self.model.clone(),
            assistant_turn: self.assistant_turn.clone(),
            finish_reason: self.finish_reason.clone(),
            usage: self.usage,
            committed_turn: self.committed_turn.clone(),
            event_count: self.event_count,
        }
    }
}

impl PartialEq for TextTurnState {
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
            && self.model == other.model
            && self.assistant_turn == other.assistant_turn
            && self.finish_reason == other.finish_reason
            && self.usage == other.usage
            && committed_turn_option_eq(&self.committed_turn, &other.committed_turn)
    }
}

impl Eq for TextTurnState {}

impl TextTurnState {
    pub fn assistant_text(&self) -> String {
        assistant_text(&self.assistant_turn)
    }

    /// Apply a streaming event to advance this turn state.
    ///
    /// Returns an error if the turn has already completed.
    pub fn apply(&mut self, event: &TextTurnEvent) -> Result<(), TextTurnReductionError> {
        if self.finish_reason.is_some() {
            return Err(TextTurnReductionError::AlreadyCompleted);
        }
        self.event_count += 1;

        match event {
            TextTurnEvent::Started { request_id, model } => {
                self.request_id = request_id.clone();
                self.model = model.clone();
            }
            TextTurnEvent::TextDelta { delta } => {
                push_or_extend_text(&mut self.assistant_turn, delta);
            }
            TextTurnEvent::ReasoningDelta { delta } => {
                push_or_extend_reasoning(&mut self.assistant_turn, delta);
            }
            TextTurnEvent::RefusalDelta { delta } => {
                push_or_extend_refusal(&mut self.assistant_turn, delta);
            }
            TextTurnEvent::Completed {
                request_id,
                finish_reason,
                usage,
                committed_turn,
            } => {
                if let Some(request_id) = request_id.clone() {
                    self.request_id = Some(request_id);
                }
                self.finish_reason = Some(finish_reason.clone());
                self.usage = Some(*usage);
                self.committed_turn = Some(committed_turn.clone());
            }
        }

        Ok(())
    }

    /// Finalize the accumulated state into a completed turn result.
    ///
    /// Returns [`TextTurnReductionError::Incomplete`] if the turn has not yet
    /// received a `Completed` event, and
    /// [`TextTurnReductionError::EmptyAssistantOutput`] if the completed turn
    /// produced no assistant items.
    pub fn finish(self) -> Result<StagedTextTurnResult, TextTurnReductionError> {
        // Check completion first so callers get Incomplete (not EmptyAssistantOutput)
        // when finish() is called on a fresh or mid-stream state.
        let finish_reason = self
            .finish_reason
            .ok_or(TextTurnReductionError::Incomplete)?;
        let usage = self.usage.ok_or(TextTurnReductionError::Incomplete)?;
        let committed_turn = self
            .committed_turn
            .ok_or(TextTurnReductionError::Incomplete)?;
        let assistant_turn = AssistantTurn::from_items(self.assistant_turn).map_err(|_| {
            TextTurnReductionError::EmptyAssistantOutput {
                model: self.model.clone(),
                request_id: self.request_id.clone(),
                finish_reason: finish_reason.clone(),
                event_count: self.event_count,
            }
        })?;
        Ok(StagedTextTurnResult {
            request_id: self.request_id,
            model: self.model,
            turn: UncommittedAssistantTurn::new(assistant_turn, committed_turn),
            finish_reason,
            usage,
        })
    }
}

/// Result of a completed no-tools text turn that has been auto-committed to the session.
///
/// Returned by `TextTurn::collect()` on session-originated builders. The turn is already in
/// the session transcript; only the content is accessible here.
#[derive(Clone, Debug)]
pub struct TextTurnResult {
    pub request_id: Option<String>,
    pub model: String,
    pub assistant_turn: AssistantTurn,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

impl TextTurnResult {
    pub fn assistant_text(&self) -> String {
        self.assistant_turn.assistant_text()
    }
}

/// Staged (not yet committed) result of a no-tools text turn.
///
/// Returned by `TextTurn::collect_staged()` and by `PendingTextTurn::collect()`.
/// Call `turn.commit_into()` or use the `CommitTurn` extension trait to commit.
#[derive(Debug)]
#[must_use = "call turn.commit_into() / CommitTurn::commit() to commit, or turn.discard() to opt out"]
pub struct StagedTextTurnResult {
    pub request_id: Option<String>,
    pub model: String,
    pub turn: UncommittedAssistantTurn,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

impl StagedTextTurnResult {
    pub fn assistant_text(&self) -> String {
        self.turn.assistant_text()
    }
}

#[derive(Debug)]
pub struct TextTurnStateWithTools<T: Toolset> {
    pub request_id: Option<String>,
    pub model: String,
    pub assistant_turn: Vec<AssistantTurnItem>,
    pub tool_calls: Vec<T::ToolCall>,
    pub recoverable_tool_call_issues: Vec<RecoverableToolCallIssue>,
    pub continue_suggestion: Option<ContinueSuggestionReason>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    pub committed_turn: Option<CommittedTurn>,
    pub event_count: u32,
}

impl<T> Default for TextTurnStateWithTools<T>
where
    T: Toolset,
{
    fn default() -> Self {
        Self {
            request_id: None,
            model: String::new(),
            assistant_turn: Vec::new(),
            tool_calls: Vec::new(),
            recoverable_tool_call_issues: Vec::new(),
            continue_suggestion: None,
            finish_reason: None,
            usage: None,
            committed_turn: None,
            event_count: 0,
        }
    }
}

impl<T> Clone for TextTurnStateWithTools<T>
where
    T: Toolset,
{
    fn clone(&self) -> Self {
        Self {
            request_id: self.request_id.clone(),
            model: self.model.clone(),
            assistant_turn: self.assistant_turn.clone(),
            tool_calls: self.tool_calls.clone(),
            recoverable_tool_call_issues: self.recoverable_tool_call_issues.clone(),
            continue_suggestion: self.continue_suggestion,
            finish_reason: self.finish_reason.clone(),
            usage: self.usage,
            committed_turn: self.committed_turn.clone(),
            event_count: self.event_count,
        }
    }
}

impl<T> PartialEq for TextTurnStateWithTools<T>
where
    T: Toolset,
    T::ToolCall: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
            && self.model == other.model
            && self.assistant_turn == other.assistant_turn
            && self.tool_calls == other.tool_calls
            && self.recoverable_tool_call_issues == other.recoverable_tool_call_issues
            && self.continue_suggestion == other.continue_suggestion
            && self.finish_reason == other.finish_reason
            && self.usage == other.usage
            && committed_turn_option_eq(&self.committed_turn, &other.committed_turn)
    }
}

impl<T> Eq for TextTurnStateWithTools<T>
where
    T: Toolset,
    T::ToolCall: Eq,
{
}

impl<T> TextTurnStateWithTools<T>
where
    T: Toolset,
{
    pub fn apply(
        &mut self,
        event: &TextTurnEventWithTools<T>,
    ) -> Result<(), TextTurnReductionError> {
        if self.finish_reason.is_some() {
            return Err(TextTurnReductionError::AlreadyCompleted);
        }
        self.event_count += 1;

        match event {
            TextTurnEventWithTools::Started { request_id, model } => {
                self.request_id = request_id.clone();
                self.model = model.clone();
            }
            TextTurnEventWithTools::TextDelta { delta } => {
                push_or_extend_text(&mut self.assistant_turn, delta);
            }
            TextTurnEventWithTools::ReasoningDelta { delta } => {
                push_or_extend_reasoning(&mut self.assistant_turn, delta);
            }
            TextTurnEventWithTools::RefusalDelta { delta } => {
                push_or_extend_refusal(&mut self.assistant_turn, delta);
            }
            TextTurnEventWithTools::ToolCallChunk { .. } => {}
            TextTurnEventWithTools::ToolCallReady(tool_call) => {
                push_tool_call(&mut self.assistant_turn, &mut self.tool_calls, tool_call);
            }
            TextTurnEventWithTools::InvalidToolCallChunk { .. } => {}
            TextTurnEventWithTools::ToolCallIssue(issue) => {
                // Add to assistant_turn so the committed turn stays consistent for commit
                // validation, but do NOT add to tool_calls — collect in recoverable issues.
                push_untyped_tool_call(&mut self.assistant_turn, &issue.metadata);
                self.recoverable_tool_call_issues.push(issue.clone());
                self.continue_suggestion = Some(ContinueSuggestionReason::RecoverableToolCallIssue);
            }
            TextTurnEventWithTools::Completed {
                request_id,
                finish_reason,
                usage,
                committed_turn,
            } => {
                if let Some(request_id) = request_id.clone() {
                    self.request_id = Some(request_id);
                }
                self.finish_reason = Some(finish_reason.clone());
                self.usage = Some(*usage);
                self.committed_turn = Some(committed_turn.clone());
            }
        }

        Ok(())
    }

    pub fn finish(self) -> Result<StagedTextTurnResultWithTools<T>, TextTurnReductionError> {
        let finish_reason = self
            .finish_reason
            .ok_or(TextTurnReductionError::Incomplete)?;
        let usage = self.usage.ok_or(TextTurnReductionError::Incomplete)?;
        let committed_turn = self
            .committed_turn
            .ok_or(TextTurnReductionError::Incomplete)?;
        let assistant_turn = AssistantTurn::from_items(self.assistant_turn).map_err(|_| {
            TextTurnReductionError::EmptyAssistantOutput {
                model: self.model.clone(),
                request_id: self.request_id.clone(),
                finish_reason: finish_reason.clone(),
                event_count: self.event_count,
            }
        })?;
        Ok(StagedTextTurnResultWithTools {
            request_id: self.request_id,
            model: self.model,
            turn: UncommittedAssistantTurn::new(assistant_turn, committed_turn),
            tool_calls: self.tool_calls,
            recoverable_tool_call_issues: self.recoverable_tool_call_issues,
            continue_suggestion: self.continue_suggestion,
            finish_reason,
            usage,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StructuredTurnOutcome<O> {
    Structured(O),
    Refusal(String),
}

/// Result of a completed tool-enabled text turn that has been auto-committed to the session.
#[derive(Clone, Debug)]
pub struct TextTurnResultWithTools<T: Toolset> {
    pub request_id: Option<String>,
    pub model: String,
    pub assistant_turn: AssistantTurn,
    pub tool_calls: Vec<T::ToolCall>,
    /// Tool calls that were present in the transcript but could not be executed normally.
    /// Availability failures and typed-parse failures are both normalized here.
    pub recoverable_tool_call_issues: Vec<RecoverableToolCallIssue>,
    pub continue_suggestion: Option<ContinueSuggestionReason>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

impl<T> TextTurnResultWithTools<T>
where
    T: Toolset,
{
    pub fn assistant_text(&self) -> String {
        self.assistant_turn.assistant_text()
    }
}

/// Staged (not yet committed) result of a tool-enabled text turn.
#[derive(Debug)]
#[must_use = "call turn.commit_into() / CommitTurn::commit() to commit, or turn.discard() to opt out"]
pub struct StagedTextTurnResultWithTools<T: Toolset> {
    pub request_id: Option<String>,
    pub model: String,
    pub turn: UncommittedAssistantTurn,
    pub tool_calls: Vec<T::ToolCall>,
    pub recoverable_tool_call_issues: Vec<RecoverableToolCallIssue>,
    pub continue_suggestion: Option<ContinueSuggestionReason>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

impl<T> StagedTextTurnResultWithTools<T>
where
    T: Toolset,
{
    pub fn assistant_text(&self) -> String {
        self.turn.assistant_text()
    }
}

#[derive(Debug)]
pub struct StructuredTurnState<O: StructuredOutput> {
    pub request_id: Option<String>,
    pub model: String,
    pub assistant_turn: Vec<AssistantTurnItem>,
    pub structured: Option<O>,
    pub refusal: Option<String>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    pub committed_turn: Option<CommittedTurn>,
    pub event_count: u32,
}

impl<O> Default for StructuredTurnState<O>
where
    O: StructuredOutput,
{
    fn default() -> Self {
        Self {
            request_id: None,
            model: String::new(),
            assistant_turn: Vec::new(),
            structured: None,
            refusal: None,
            finish_reason: None,
            usage: None,
            committed_turn: None,
            event_count: 0,
        }
    }
}

impl<O> Clone for StructuredTurnState<O>
where
    O: StructuredOutput,
{
    fn clone(&self) -> Self {
        Self {
            request_id: self.request_id.clone(),
            model: self.model.clone(),
            assistant_turn: self.assistant_turn.clone(),
            structured: self.structured.clone(),
            refusal: self.refusal.clone(),
            finish_reason: self.finish_reason.clone(),
            usage: self.usage,
            committed_turn: self.committed_turn.clone(),
            event_count: self.event_count,
        }
    }
}

impl<O> PartialEq for StructuredTurnState<O>
where
    O: StructuredOutput + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
            && self.model == other.model
            && self.assistant_turn == other.assistant_turn
            && self.structured == other.structured
            && self.refusal == other.refusal
            && self.finish_reason == other.finish_reason
            && self.usage == other.usage
            && committed_turn_option_eq(&self.committed_turn, &other.committed_turn)
    }
}

impl<O> Eq for StructuredTurnState<O> where O: StructuredOutput + Eq {}

impl<O> StructuredTurnState<O>
where
    O: StructuredOutput,
{
    pub fn apply(
        &mut self,
        event: &StructuredTurnEvent<O>,
    ) -> Result<(), StructuredTurnReductionError> {
        if self.finish_reason.is_some() {
            return Err(StructuredTurnReductionError::AlreadyCompleted);
        }
        self.event_count += 1;

        match event {
            StructuredTurnEvent::Started { request_id, model } => {
                self.request_id = request_id.clone();
                self.model = model.clone();
            }
            StructuredTurnEvent::StructuredOutputChunk { json_delta } => {
                push_or_extend_text(&mut self.assistant_turn, json_delta);
            }
            StructuredTurnEvent::StructuredOutputReady(value) => {
                if self.structured.is_some() {
                    return Err(StructuredTurnReductionError::DuplicateStructuredOutput);
                }
                self.structured = Some(value.clone());
            }
            StructuredTurnEvent::ReasoningDelta { delta } => {
                push_or_extend_reasoning(&mut self.assistant_turn, delta);
            }
            StructuredTurnEvent::RefusalDelta { delta } => {
                push_or_extend_refusal(&mut self.assistant_turn, delta);
                if let Some(existing) = self.refusal.as_mut() {
                    existing.push_str(delta);
                } else {
                    self.refusal = Some(delta.clone());
                }
            }
            StructuredTurnEvent::Completed {
                request_id,
                finish_reason,
                usage,
                committed_turn,
            } => {
                if let Some(request_id) = request_id.clone() {
                    self.request_id = Some(request_id);
                }
                self.finish_reason = Some(finish_reason.clone());
                self.usage = Some(*usage);
                self.committed_turn = Some(committed_turn.clone());
            }
        }

        Ok(())
    }

    pub fn finish(self) -> Result<StagedStructuredTurnResult<O>, StructuredTurnReductionError> {
        let finish_reason = self
            .finish_reason
            .ok_or(StructuredTurnReductionError::Incomplete)?;
        let usage = self.usage.ok_or(StructuredTurnReductionError::Incomplete)?;
        let committed_turn = self
            .committed_turn
            .ok_or(StructuredTurnReductionError::Incomplete)?;
        let assistant_turn = AssistantTurn::from_items(self.assistant_turn).map_err(|_| {
            StructuredTurnReductionError::EmptyAssistantOutput {
                model: self.model.clone(),
                request_id: self.request_id.clone(),
                finish_reason: finish_reason.clone(),
                event_count: self.event_count,
            }
        })?;
        let semantic = match (self.structured, self.refusal) {
            (Some(value), None) => StructuredTurnOutcome::Structured(value),
            (None, Some(refusal)) => StructuredTurnOutcome::Refusal(refusal),
            (None, None) => return Err(StructuredTurnReductionError::MissingSemantic),
            (Some(_), Some(_)) => return Err(StructuredTurnReductionError::ConflictingSemantic),
        };

        Ok(StagedStructuredTurnResult {
            request_id: self.request_id,
            model: self.model,
            turn: UncommittedAssistantTurn::new(assistant_turn, committed_turn),
            semantic,
            finish_reason,
            usage,
        })
    }
}

#[derive(Debug)]
pub struct StructuredTurnStateWithTools<T: Toolset, O: StructuredOutput> {
    pub request_id: Option<String>,
    pub model: String,
    pub assistant_turn: Vec<AssistantTurnItem>,
    pub tool_calls: Vec<T::ToolCall>,
    pub recoverable_tool_call_issues: Vec<RecoverableToolCallIssue>,
    pub continue_suggestion: Option<ContinueSuggestionReason>,
    pub structured: Option<O>,
    pub refusal: Option<String>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    pub committed_turn: Option<CommittedTurn>,
    pub event_count: u32,
}

impl<T, O> Default for StructuredTurnStateWithTools<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    fn default() -> Self {
        Self {
            request_id: None,
            model: String::new(),
            assistant_turn: Vec::new(),
            tool_calls: Vec::new(),
            recoverable_tool_call_issues: Vec::new(),
            continue_suggestion: None,
            structured: None,
            refusal: None,
            finish_reason: None,
            usage: None,
            committed_turn: None,
            event_count: 0,
        }
    }
}

impl<T, O> Clone for StructuredTurnStateWithTools<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    fn clone(&self) -> Self {
        Self {
            request_id: self.request_id.clone(),
            model: self.model.clone(),
            assistant_turn: self.assistant_turn.clone(),
            tool_calls: self.tool_calls.clone(),
            recoverable_tool_call_issues: self.recoverable_tool_call_issues.clone(),
            continue_suggestion: self.continue_suggestion,
            structured: self.structured.clone(),
            refusal: self.refusal.clone(),
            finish_reason: self.finish_reason.clone(),
            usage: self.usage,
            committed_turn: self.committed_turn.clone(),
            event_count: self.event_count,
        }
    }
}

impl<T, O> PartialEq for StructuredTurnStateWithTools<T, O>
where
    T: Toolset,
    T::ToolCall: PartialEq,
    O: StructuredOutput + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
            && self.model == other.model
            && self.assistant_turn == other.assistant_turn
            && self.tool_calls == other.tool_calls
            && self.recoverable_tool_call_issues == other.recoverable_tool_call_issues
            && self.continue_suggestion == other.continue_suggestion
            && self.structured == other.structured
            && self.refusal == other.refusal
            && self.finish_reason == other.finish_reason
            && self.usage == other.usage
            && committed_turn_option_eq(&self.committed_turn, &other.committed_turn)
    }
}

impl<T, O> Eq for StructuredTurnStateWithTools<T, O>
where
    T: Toolset,
    T::ToolCall: Eq,
    O: StructuredOutput + Eq,
{
}

impl<T, O> StructuredTurnStateWithTools<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    pub fn apply(
        &mut self,
        event: &StructuredTurnEventWithTools<T, O>,
    ) -> Result<(), StructuredTurnReductionError> {
        if self.finish_reason.is_some() {
            return Err(StructuredTurnReductionError::AlreadyCompleted);
        }
        self.event_count += 1;

        match event {
            StructuredTurnEventWithTools::Started { request_id, model } => {
                self.request_id = request_id.clone();
                self.model = model.clone();
            }
            StructuredTurnEventWithTools::StructuredOutputChunk { json_delta } => {
                push_or_extend_text(&mut self.assistant_turn, json_delta);
            }
            StructuredTurnEventWithTools::StructuredOutputReady(value) => {
                if self.structured.is_some() {
                    return Err(StructuredTurnReductionError::DuplicateStructuredOutput);
                }
                self.structured = Some(value.clone());
            }
            StructuredTurnEventWithTools::ReasoningDelta { delta } => {
                push_or_extend_reasoning(&mut self.assistant_turn, delta);
            }
            StructuredTurnEventWithTools::RefusalDelta { delta } => {
                push_or_extend_refusal(&mut self.assistant_turn, delta);
                if let Some(existing) = self.refusal.as_mut() {
                    existing.push_str(delta);
                } else {
                    self.refusal = Some(delta.clone());
                }
            }
            StructuredTurnEventWithTools::ToolCallChunk { .. } => {}
            StructuredTurnEventWithTools::ToolCallReady(tool_call) => {
                push_tool_call(&mut self.assistant_turn, &mut self.tool_calls, tool_call);
            }
            StructuredTurnEventWithTools::InvalidToolCallChunk { .. } => {}
            StructuredTurnEventWithTools::ToolCallIssue(issue) => {
                push_untyped_tool_call(&mut self.assistant_turn, &issue.metadata);
                self.recoverable_tool_call_issues.push(issue.clone());
                self.continue_suggestion = Some(ContinueSuggestionReason::RecoverableToolCallIssue);
            }
            StructuredTurnEventWithTools::Completed {
                request_id,
                finish_reason,
                usage,
                committed_turn,
            } => {
                if let Some(request_id) = request_id.clone() {
                    self.request_id = Some(request_id);
                }
                self.finish_reason = Some(finish_reason.clone());
                self.usage = Some(*usage);
                self.committed_turn = Some(committed_turn.clone());
            }
        }

        Ok(())
    }

    pub fn finish(
        self,
    ) -> Result<StagedStructuredTurnResultWithTools<T, O>, StructuredTurnReductionError> {
        let finish_reason = self
            .finish_reason
            .ok_or(StructuredTurnReductionError::Incomplete)?;
        let usage = self.usage.ok_or(StructuredTurnReductionError::Incomplete)?;
        let committed_turn = self
            .committed_turn
            .ok_or(StructuredTurnReductionError::Incomplete)?;
        let assistant_turn = AssistantTurn::from_items(self.assistant_turn).map_err(|_| {
            StructuredTurnReductionError::EmptyAssistantOutput {
                model: self.model.clone(),
                request_id: self.request_id.clone(),
                finish_reason: finish_reason.clone(),
                event_count: self.event_count,
            }
        })?;
        let semantic = match (self.structured, self.refusal) {
            (Some(value), None) => StructuredTurnOutcome::Structured(value),
            (None, Some(refusal)) => StructuredTurnOutcome::Refusal(refusal),
            (None, None) => return Err(StructuredTurnReductionError::MissingSemantic),
            (Some(_), Some(_)) => return Err(StructuredTurnReductionError::ConflictingSemantic),
        };

        Ok(StagedStructuredTurnResultWithTools {
            request_id: self.request_id,
            model: self.model,
            turn: UncommittedAssistantTurn::new(assistant_turn, committed_turn),
            tool_calls: self.tool_calls,
            recoverable_tool_call_issues: self.recoverable_tool_call_issues,
            continue_suggestion: self.continue_suggestion,
            semantic,
            finish_reason,
            usage,
        })
    }
}

/// Result of a completed no-tools structured turn that has been auto-committed to the session.
#[derive(Clone, Debug)]
pub struct StructuredTurnResult<O: StructuredOutput> {
    pub request_id: Option<String>,
    pub model: String,
    pub assistant_turn: AssistantTurn,
    pub semantic: StructuredTurnOutcome<O>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

/// Staged (not yet committed) result of a no-tools structured turn.
#[derive(Debug)]
#[must_use = "call turn.commit_into() / CommitTurn::commit() to commit, or turn.discard() to opt out"]
pub struct StagedStructuredTurnResult<O: StructuredOutput> {
    pub request_id: Option<String>,
    pub model: String,
    pub turn: UncommittedAssistantTurn,
    pub semantic: StructuredTurnOutcome<O>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

/// Result of a completed tool-enabled structured turn that has been auto-committed to the session.
#[derive(Clone, Debug)]
pub struct StructuredTurnResultWithTools<T: Toolset, O: StructuredOutput> {
    pub request_id: Option<String>,
    pub model: String,
    pub assistant_turn: AssistantTurn,
    pub tool_calls: Vec<T::ToolCall>,
    /// Tool calls that were present in the transcript but could not be executed normally.
    /// Availability failures and typed-parse failures are both normalized here.
    pub recoverable_tool_call_issues: Vec<RecoverableToolCallIssue>,
    pub continue_suggestion: Option<ContinueSuggestionReason>,
    pub semantic: StructuredTurnOutcome<O>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

/// Staged (not yet committed) result of a tool-enabled structured turn.
#[derive(Debug)]
#[must_use = "call turn.commit_into() / CommitTurn::commit() to commit, or turn.discard() to opt out"]
pub struct StagedStructuredTurnResultWithTools<T: Toolset, O: StructuredOutput> {
    pub request_id: Option<String>,
    pub model: String,
    pub turn: UncommittedAssistantTurn,
    pub tool_calls: Vec<T::ToolCall>,
    pub recoverable_tool_call_issues: Vec<RecoverableToolCallIssue>,
    pub continue_suggestion: Option<ContinueSuggestionReason>,
    pub semantic: StructuredTurnOutcome<O>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct CompletionTurnState {
    pub request_id: Option<String>,
    pub model: String,
    pub text: String,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
}

impl CompletionTurnState {
    /// Apply a streaming event to advance this turn state.
    ///
    /// Returns an error if the turn has already completed.
    pub fn apply(&mut self, event: &CompletionEvent) -> Result<(), CompletionReductionError> {
        if self.finish_reason.is_some() {
            return Err(CompletionReductionError::AlreadyCompleted);
        }

        match event {
            CompletionEvent::Started { request_id, model } => {
                self.request_id = request_id.clone();
                self.model = model.clone();
            }
            CompletionEvent::TextDelta(delta) => {
                self.text.push_str(delta);
            }
            CompletionEvent::Completed {
                request_id,
                finish_reason,
                usage,
            } => {
                if let Some(request_id) = request_id.clone() {
                    self.request_id = Some(request_id);
                }
                self.finish_reason = Some(finish_reason.clone());
                self.usage = Some(*usage);
            }
        }

        Ok(())
    }

    /// Finalize the accumulated state into a completed turn result.
    pub fn finish(self) -> Result<CompletionTurnResult, CompletionReductionError> {
        Ok(CompletionTurnResult {
            request_id: self.request_id,
            model: self.model,
            text: self.text,
            finish_reason: self
                .finish_reason
                .ok_or(CompletionReductionError::Incomplete)?,
            usage: self.usage.ok_or(CompletionReductionError::Incomplete)?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletionTurnResult {
    pub request_id: Option<String>,
    pub model: String,
    pub text: String,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

#[derive(Clone, Debug)]
pub struct StructuredCompletionState<O: StructuredOutput> {
    pub request_id: Option<String>,
    pub model: String,
    pub structured: Option<O>,
    pub refusal: Option<String>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
}

impl<O> Default for StructuredCompletionState<O>
where
    O: StructuredOutput,
{
    fn default() -> Self {
        Self {
            request_id: None,
            model: String::new(),
            structured: None,
            refusal: None,
            finish_reason: None,
            usage: None,
        }
    }
}

impl<O> StructuredCompletionState<O>
where
    O: StructuredOutput,
{
    pub fn apply(
        &mut self,
        event: &StructuredCompletionEvent<O>,
    ) -> Result<(), StructuredCompletionReductionError> {
        if self.finish_reason.is_some() {
            return Err(StructuredCompletionReductionError::AlreadyCompleted);
        }

        match event {
            StructuredCompletionEvent::Started { request_id, model } => {
                self.request_id = request_id.clone();
                self.model = model.clone();
            }
            StructuredCompletionEvent::StructuredOutputChunk { .. } => {}
            StructuredCompletionEvent::StructuredOutputReady(value) => {
                if self.structured.is_some() {
                    return Err(StructuredCompletionReductionError::DuplicateStructuredOutput);
                }
                self.structured = Some(value.clone());
            }
            StructuredCompletionEvent::ReasoningDelta { .. } => {}
            StructuredCompletionEvent::RefusalDelta { delta } => match self.refusal.as_mut() {
                Some(existing) => existing.push_str(delta),
                None => self.refusal = Some(delta.clone()),
            },
            StructuredCompletionEvent::Completed {
                request_id,
                finish_reason,
                usage,
            } => {
                if let Some(request_id) = request_id.clone() {
                    self.request_id = Some(request_id);
                }
                self.finish_reason = Some(finish_reason.clone());
                self.usage = Some(*usage);
            }
        }

        Ok(())
    }

    pub fn finish(
        self,
    ) -> Result<StructuredCompletionResult<O>, StructuredCompletionReductionError> {
        let semantic = match (self.structured, self.refusal) {
            (Some(value), None) => StructuredTurnOutcome::Structured(value),
            (None, Some(refusal)) => StructuredTurnOutcome::Refusal(refusal),
            (None, None) => return Err(StructuredCompletionReductionError::MissingSemantic),
            (Some(_), Some(_)) => {
                return Err(StructuredCompletionReductionError::ConflictingSemantic);
            }
        };

        Ok(StructuredCompletionResult {
            request_id: self.request_id,
            model: self.model,
            semantic,
            finish_reason: self
                .finish_reason
                .ok_or(StructuredCompletionReductionError::Incomplete)?,
            usage: self
                .usage
                .ok_or(StructuredCompletionReductionError::Incomplete)?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructuredCompletionResult<O: StructuredOutput> {
    pub request_id: Option<String>,
    pub model: String,
    pub semantic: StructuredTurnOutcome<O>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum TextTurnReductionError {
    #[error("turn already completed")]
    AlreadyCompleted,
    #[error(
        "completed turn produced no assistant items (model={model}, request_id={request_id:?}, finish_reason={finish_reason:?}, event_count={event_count})"
    )]
    EmptyAssistantOutput {
        model: String,
        request_id: Option<String>,
        finish_reason: FinishReason,
        event_count: u32,
    },
    #[error("turn has not completed yet")]
    Incomplete,
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum StructuredTurnReductionError {
    #[error("turn already completed")]
    AlreadyCompleted,
    #[error("structured output appeared more than once")]
    DuplicateStructuredOutput,
    #[error(
        "completed turn produced no assistant items (model={model}, request_id={request_id:?}, finish_reason={finish_reason:?}, event_count={event_count})"
    )]
    EmptyAssistantOutput {
        model: String,
        request_id: Option<String>,
        finish_reason: FinishReason,
        event_count: u32,
    },
    #[error("turn has not completed yet")]
    Incomplete,
    #[error("turn completed without structured output or refusal")]
    MissingSemantic,
    #[error("turn completed with both structured output and refusal")]
    ConflictingSemantic,
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum CompletionReductionError {
    #[error("turn already completed")]
    AlreadyCompleted,
    #[error("turn has not completed yet")]
    Incomplete,
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum StructuredCompletionReductionError {
    #[error("turn already completed")]
    AlreadyCompleted,
    #[error("structured output appeared more than once")]
    DuplicateStructuredOutput,
    #[error("turn has not completed yet")]
    Incomplete,
    #[error("turn completed without structured output or refusal")]
    MissingSemantic,
    #[error("turn completed with both structured output and refusal")]
    ConflictingSemantic,
}

pub struct TextTurnReducer {
    state: TextTurnState,
}

impl Default for TextTurnReducer {
    fn default() -> Self {
        Self::new()
    }
}

impl TextTurnReducer {
    pub fn new() -> Self {
        Self {
            state: TextTurnState::default(),
        }
    }

    pub fn state(&self) -> &TextTurnState {
        &self.state
    }

    pub fn into_state(self) -> TextTurnState {
        self.state
    }

    pub fn apply(&mut self, event: &TextTurnEvent) -> Result<(), TextTurnReductionError> {
        self.state.apply(event)
    }

    pub fn into_result(self) -> Result<StagedTextTurnResult, TextTurnReductionError> {
        self.state.finish()
    }
}

pub struct TextTurnReducerWithTools<T: Toolset> {
    state: TextTurnStateWithTools<T>,
}

impl<T> Default for TextTurnReducerWithTools<T>
where
    T: Toolset,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TextTurnReducerWithTools<T>
where
    T: Toolset,
{
    pub fn new() -> Self {
        Self {
            state: TextTurnStateWithTools::default(),
        }
    }

    pub fn state(&self) -> &TextTurnStateWithTools<T> {
        &self.state
    }

    pub fn into_state(self) -> TextTurnStateWithTools<T> {
        self.state
    }

    pub fn apply(
        &mut self,
        event: &TextTurnEventWithTools<T>,
    ) -> Result<(), TextTurnReductionError> {
        self.state.apply(event)
    }

    pub fn into_result(self) -> Result<StagedTextTurnResultWithTools<T>, TextTurnReductionError> {
        self.state.finish()
    }
}

pub struct StructuredTurnReducer<O: StructuredOutput> {
    state: StructuredTurnState<O>,
}

impl<O> Default for StructuredTurnReducer<O>
where
    O: StructuredOutput,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<O> StructuredTurnReducer<O>
where
    O: StructuredOutput,
{
    pub fn new() -> Self {
        Self {
            state: StructuredTurnState::default(),
        }
    }

    pub fn state(&self) -> &StructuredTurnState<O> {
        &self.state
    }

    pub fn into_state(self) -> StructuredTurnState<O> {
        self.state
    }

    pub fn apply(
        &mut self,
        event: &StructuredTurnEvent<O>,
    ) -> Result<(), StructuredTurnReductionError> {
        self.state.apply(event)
    }

    pub fn into_result(
        self,
    ) -> Result<StagedStructuredTurnResult<O>, (StructuredTurnReductionError, Option<CommittedTurn>)>
    {
        let committed_turn = self.state.committed_turn.clone();
        self.state
            .finish()
            .map_err(|source| (source, committed_turn))
    }
}

pub struct StructuredTurnReducerWithTools<T: Toolset, O: StructuredOutput> {
    state: StructuredTurnStateWithTools<T, O>,
}

impl<T, O> Default for StructuredTurnReducerWithTools<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, O> StructuredTurnReducerWithTools<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    pub fn new() -> Self {
        Self {
            state: StructuredTurnStateWithTools::default(),
        }
    }

    pub fn state(&self) -> &StructuredTurnStateWithTools<T, O> {
        &self.state
    }

    pub fn into_state(self) -> StructuredTurnStateWithTools<T, O> {
        self.state
    }

    pub fn apply(
        &mut self,
        event: &StructuredTurnEventWithTools<T, O>,
    ) -> Result<(), StructuredTurnReductionError> {
        self.state.apply(event)
    }

    pub fn into_result(
        self,
    ) -> Result<
        StagedStructuredTurnResultWithTools<T, O>,
        (StructuredTurnReductionError, Option<CommittedTurn>),
    > {
        let committed_turn = self.state.committed_turn.clone();
        self.state
            .finish()
            .map_err(|source| (source, committed_turn))
    }
}

pub struct CompletionReducer {
    state: CompletionTurnState,
}

impl Default for CompletionReducer {
    fn default() -> Self {
        Self::new()
    }
}

impl CompletionReducer {
    pub fn new() -> Self {
        Self {
            state: CompletionTurnState::default(),
        }
    }

    pub fn state(&self) -> &CompletionTurnState {
        &self.state
    }

    pub fn into_state(self) -> CompletionTurnState {
        self.state
    }

    pub fn apply(&mut self, event: &CompletionEvent) -> Result<(), CompletionReductionError> {
        self.state.apply(event)
    }

    pub fn into_result(self) -> Result<CompletionTurnResult, CompletionReductionError> {
        self.state.finish()
    }
}

pub struct StructuredCompletionReducer<O: StructuredOutput> {
    state: StructuredCompletionState<O>,
}

impl<O> Default for StructuredCompletionReducer<O>
where
    O: StructuredOutput,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<O> StructuredCompletionReducer<O>
where
    O: StructuredOutput,
{
    pub fn new() -> Self {
        Self {
            state: StructuredCompletionState::default(),
        }
    }

    pub fn state(&self) -> &StructuredCompletionState<O> {
        &self.state
    }

    pub fn into_state(self) -> StructuredCompletionState<O> {
        self.state
    }

    pub fn apply(
        &mut self,
        event: &StructuredCompletionEvent<O>,
    ) -> Result<(), StructuredCompletionReductionError> {
        self.state.apply(event)
    }

    pub fn into_result(
        self,
    ) -> Result<StructuredCompletionResult<O>, StructuredCompletionReductionError> {
        self.state.finish()
    }
}

fn push_or_extend_text(items: &mut Vec<AssistantTurnItem>, delta: &str) {
    if delta.is_empty() {
        return;
    }
    match items.last_mut() {
        Some(AssistantTurnItem::Text(existing)) => existing.push_str(delta),
        _ => items.push(AssistantTurnItem::Text(delta.to_string())),
    }
}

fn push_or_extend_reasoning(items: &mut Vec<AssistantTurnItem>, delta: &str) {
    if delta.is_empty() {
        return;
    }
    match items.last_mut() {
        Some(AssistantTurnItem::Reasoning(existing)) => existing.push_str(delta),
        _ => items.push(AssistantTurnItem::Reasoning(delta.to_string())),
    }
}

fn push_or_extend_refusal(items: &mut Vec<AssistantTurnItem>, delta: &str) {
    if delta.is_empty() {
        return;
    }
    match items.last_mut() {
        Some(AssistantTurnItem::Refusal(existing)) => existing.push_str(delta),
        _ => items.push(AssistantTurnItem::Refusal(delta.to_string())),
    }
}

fn push_tool_call<T>(assistant: &mut Vec<AssistantTurnItem>, tool_calls: &mut Vec<T>, tool_call: &T)
where
    T: ToolCallWrapper + Clone,
{
    if tool_calls
        .iter()
        .any(|existing| existing.metadata().id == tool_call.metadata().id)
    {
        return;
    }

    let metadata = tool_call.metadata();
    assistant.push(AssistantTurnItem::ToolCall {
        id: metadata.id.clone(),
        name: metadata.name.clone(),
        arguments: metadata.arguments.clone(),
    });
    tool_calls.push(tool_call.clone());
}

fn push_untyped_tool_call(assistant: &mut Vec<AssistantTurnItem>, metadata: &ToolMetadata) {
    assistant.push(AssistantTurnItem::ToolCall {
        id: metadata.id.clone(),
        name: metadata.name.clone(),
        arguments: metadata.arguments.clone(),
    });
}

fn assistant_text(items: &[AssistantTurnItem]) -> String {
    let mut text = String::new();
    for item in items {
        if let AssistantTurnItem::Text(delta) = item {
            text.push_str(delta);
        }
    }
    text
}

fn committed_turn_option_eq(lhs: &Option<CommittedTurn>, rhs: &Option<CommittedTurn>) -> bool {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Arc::ptr_eq(lhs, rhs),
        (None, None) => true,
        _ => false,
    }
}

pub fn assistant_json(items: &[AssistantTurnItem]) -> Option<Result<RawJson, serde_json::Error>> {
    let text = assistant_text(items);
    if text.is_empty() {
        None
    } else {
        Some(RawJson::parse(text))
    }
}

pub fn find_tool_call_arguments<'a>(
    items: &'a [AssistantTurnItem],
    id: &ToolCallId,
) -> Option<&'a RawJson> {
    items.iter().find_map(|item| match item {
        AssistantTurnItem::ToolCall {
            id: candidate,
            arguments,
            ..
        } if candidate == id => Some(arguments),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::{
        ToolCallError, ToolDef, ToolMetadata, ToolName,
        toolset::{ToolCallWrapper, ToolInput, ToolSelector},
        transcript::AssistantTurnView,
    };

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct WeatherArgs {
        city: String,
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct WeatherResult {
        forecast: String,
    }

    impl ToolInput for WeatherArgs {
        type Output = WeatherResult;

        const NAME: &'static str = "weather";
        const DESCRIPTION: &'static str = "Get weather";
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct WeatherArgsCall {
        metadata: ToolMetadata,
        input: WeatherArgs,
    }

    impl ToolCallWrapper for WeatherArgsCall {
        fn metadata(&self) -> &ToolMetadata {
            &self.metadata
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum CallsCall {
        Weather(WeatherArgsCall),
    }

    impl ToolCallWrapper for CallsCall {
        fn metadata(&self) -> &ToolMetadata {
            match self {
                Self::Weather(call) => &call.metadata,
            }
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    struct Tools;

    #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize, JsonSchema)]
    enum ToolsSelector {
        Weather,
    }

    impl ToolSelector<Tools> for ToolsSelector {
        fn name(self) -> &'static str {
            match self {
                Self::Weather => "weather",
            }
        }

        fn definition(self) -> &'static ToolDef {
            &Tools::definitions()[match self {
                Self::Weather => 0,
            }]
        }

        fn all() -> &'static [Self] {
            &[Self::Weather]
        }

        fn try_from_name(name: &str) -> Option<Self> {
            match name {
                "weather" => Some(Self::Weather),
                _ => None,
            }
        }
    }

    impl Toolset for Tools {
        type ToolCall = CallsCall;
        type Selector = ToolsSelector;

        fn definitions() -> &'static [ToolDef] {
            fn weather_args_schema() -> schemars::Schema {
                schemars::schema_for!(WeatherArgs)
            }

            static DEFS: [ToolDef; 1] =
                [ToolDef::new("weather", "Get weather", weather_args_schema)];
            &DEFS
        }

        fn parse_tool_call(metadata: ToolMetadata) -> Result<Self::ToolCall, ToolCallError> {
            match metadata.name.as_str() {
                "weather" => serde_json::from_str(metadata.arguments.get())
                    .map(|input| CallsCall::Weather(WeatherArgsCall { metadata, input }))
                    .map_err(|source| ToolCallError::Deserialize {
                        name: "weather".to_string(),
                        source,
                    }),
                _ => Err(ToolCallError::UnknownTool {
                    name: metadata.name.as_str().to_string(),
                }),
            }
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct Summary {
        answer: String,
    }

    #[test]
    fn text_reducer_returns_assistant_turn() {
        let mut reducer = TextTurnReducerWithTools::<Tools>::new();
        reducer
            .apply(&TextTurnEventWithTools::Started {
                request_id: Some("req-1".into()),
                model: "gpt-4.1".into(),
            })
            .unwrap();
        reducer
            .apply(&TextTurnEventWithTools::TextDelta {
                delta: "checking ".into(),
            })
            .unwrap();
        reducer
            .apply(&TextTurnEventWithTools::ToolCallReady(CallsCall::Weather(
                WeatherArgsCall {
                    metadata: ToolMetadata::new(
                        ToolCallId::from("call-1"),
                        ToolName::from("weather"),
                        RawJson::parse("{\"city\":\"Tokyo\"}").unwrap(),
                    ),
                    input: WeatherArgs {
                        city: "Tokyo".into(),
                    },
                },
            )))
            .unwrap();
        reducer
            .apply(&TextTurnEventWithTools::Completed {
                request_id: Some("req-1".into()),
                finish_reason: FinishReason::ToolCall,
                usage: Usage {
                    total_tokens: 12,
                    ..Usage::zero()
                },
                committed_turn: Arc::new(AssistantTurnView::from_items(&[])),
            })
            .unwrap();

        let result = reducer.into_result().unwrap();
        assert_eq!(result.turn.items().len(), 2);
        assert_eq!(result.turn.assistant_text(), "checking ");
    }

    #[test]
    fn structured_reducer_distinguishes_structured_and_refusal() {
        let mut reducer = StructuredTurnReducer::<Summary>::new();
        reducer
            .apply(&StructuredTurnEvent::Started {
                request_id: Some("req-2".into()),
                model: "gpt-4.1".into(),
            })
            .unwrap();
        reducer
            .apply(&StructuredTurnEvent::RefusalDelta { delta: "no".into() })
            .unwrap();
        reducer
            .apply(&StructuredTurnEvent::Completed {
                request_id: Some("req-2".into()),
                finish_reason: FinishReason::ContentFilter,
                usage: Usage {
                    total_tokens: 7,
                    ..Usage::zero()
                },
                committed_turn: Arc::new(AssistantTurnView::from_items(&[])),
            })
            .unwrap();

        let result = reducer.into_result().unwrap();
        assert_eq!(
            result.semantic,
            StructuredTurnOutcome::Refusal(String::from("no"))
        );
    }
}
