use std::sync::Arc;

use thiserror::Error;

use crate::{
    budget::Usage,
    conversation::{AssistantTurn, AssistantTurnItem, RawJson, ToolCallId},
    llm::{CompletionEvent, FinishReason, StructuredTurnEvent, TextTurnEvent},
    structured::StructuredOutput,
    toolset::{ToolCallWrapper, Toolset},
    transcript::CommittedTurn,
};

#[derive(Debug)]
pub struct TextTurnState<T: Toolset> {
    pub request_id: Option<String>,
    pub model: String,
    pub assistant_turn: Vec<AssistantTurnItem>,
    pub tool_calls: Vec<T::ToolCall>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    pub committed_turn: Option<CommittedTurn>,
}

impl<T> Default for TextTurnState<T>
where
    T: Toolset,
{
    fn default() -> Self {
        Self {
            request_id: None,
            model: String::new(),
            assistant_turn: Vec::new(),
            tool_calls: Vec::new(),
            finish_reason: None,
            usage: None,
            committed_turn: None,
        }
    }
}

impl<T> Clone for TextTurnState<T>
where
    T: Toolset,
{
    fn clone(&self) -> Self {
        Self {
            request_id: self.request_id.clone(),
            model: self.model.clone(),
            assistant_turn: self.assistant_turn.clone(),
            tool_calls: self.tool_calls.clone(),
            finish_reason: self.finish_reason.clone(),
            usage: self.usage,
            committed_turn: self.committed_turn.clone(),
        }
    }
}

impl<T> PartialEq for TextTurnState<T>
where
    T: Toolset,
    T::ToolCall: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
            && self.model == other.model
            && self.assistant_turn == other.assistant_turn
            && self.tool_calls == other.tool_calls
            && self.finish_reason == other.finish_reason
            && self.usage == other.usage
            && committed_turn_option_eq(&self.committed_turn, &other.committed_turn)
    }
}

impl<T> Eq for TextTurnState<T>
where
    T: Toolset,
    T::ToolCall: Eq,
{
}

impl<T> TextTurnState<T>
where
    T: Toolset,
{
    pub fn assistant_text(&self) -> String {
        assistant_text(&self.assistant_turn)
    }

    /// Apply a streaming event to advance this turn state.
    ///
    /// Returns an error if the turn has already completed.
    pub fn apply(&mut self, event: &TextTurnEvent<T>) -> Result<(), TextTurnReductionError> {
        if self.finish_reason.is_some() {
            return Err(TextTurnReductionError::AlreadyCompleted);
        }

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
            TextTurnEvent::ToolCallChunk { .. } => {}
            TextTurnEvent::ToolCallReady(tool_call) => {
                push_tool_call(&mut self.assistant_turn, &mut self.tool_calls, tool_call);
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
    pub fn finish(self) -> Result<TextTurnResult<T>, TextTurnReductionError> {
        // Check completion first so callers get Incomplete (not EmptyAssistantOutput)
        // when finish() is called on a fresh or mid-stream state.
        let finish_reason = self
            .finish_reason
            .ok_or(TextTurnReductionError::Incomplete)?;
        let usage = self.usage.ok_or(TextTurnReductionError::Incomplete)?;
        let committed_turn = self
            .committed_turn
            .ok_or(TextTurnReductionError::Incomplete)?;
        let assistant_turn = AssistantTurn::from_items(self.assistant_turn)
            .map_err(|_| TextTurnReductionError::EmptyAssistantOutput)?;
        Ok(TextTurnResult {
            request_id: self.request_id,
            model: self.model,
            assistant_turn,
            tool_calls: self.tool_calls,
            finish_reason,
            usage,
            committed_turn,
        })
    }
}

#[derive(Debug)]
pub struct TextTurnResult<T: Toolset> {
    pub request_id: Option<String>,
    pub model: String,
    pub assistant_turn: AssistantTurn,
    pub tool_calls: Vec<T::ToolCall>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub committed_turn: CommittedTurn,
}

impl<T> TextTurnResult<T>
where
    T: Toolset,
{
    pub fn assistant_text(&self) -> String {
        self.assistant_turn.assistant_text()
    }
}

impl<T> Clone for TextTurnResult<T>
where
    T: Toolset,
{
    fn clone(&self) -> Self {
        Self {
            request_id: self.request_id.clone(),
            model: self.model.clone(),
            assistant_turn: self.assistant_turn.clone(),
            tool_calls: self.tool_calls.clone(),
            finish_reason: self.finish_reason.clone(),
            usage: self.usage,
            committed_turn: self.committed_turn.clone(),
        }
    }
}

#[derive(Debug)]
pub struct StructuredTurnState<T: Toolset, O: StructuredOutput> {
    pub request_id: Option<String>,
    pub model: String,
    pub assistant_turn: Vec<AssistantTurnItem>,
    pub tool_calls: Vec<T::ToolCall>,
    pub structured: Option<O>,
    pub refusal: Option<String>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    pub committed_turn: Option<CommittedTurn>,
}

impl<T, O> Default for StructuredTurnState<T, O>
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
            structured: None,
            refusal: None,
            finish_reason: None,
            usage: None,
            committed_turn: None,
        }
    }
}

impl<T, O> Clone for StructuredTurnState<T, O>
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
            structured: self.structured.clone(),
            refusal: self.refusal.clone(),
            finish_reason: self.finish_reason.clone(),
            usage: self.usage,
            committed_turn: self.committed_turn.clone(),
        }
    }
}

impl<T, O> PartialEq for StructuredTurnState<T, O>
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
            && self.structured == other.structured
            && self.refusal == other.refusal
            && self.finish_reason == other.finish_reason
            && self.usage == other.usage
            && committed_turn_option_eq(&self.committed_turn, &other.committed_turn)
    }
}

impl<T, O> Eq for StructuredTurnState<T, O>
where
    T: Toolset,
    T::ToolCall: Eq,
    O: StructuredOutput + Eq,
{
}

impl<T, O> StructuredTurnState<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    /// Apply a streaming event to advance this turn state.
    ///
    /// Returns an error if the turn has already completed.
    pub fn apply(
        &mut self,
        event: &StructuredTurnEvent<T, O>,
    ) -> Result<(), StructuredTurnReductionError> {
        if self.finish_reason.is_some() {
            return Err(StructuredTurnReductionError::AlreadyCompleted);
        }

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
            StructuredTurnEvent::ToolCallChunk { .. } => {}
            StructuredTurnEvent::ToolCallReady(tool_call) => {
                push_tool_call(&mut self.assistant_turn, &mut self.tool_calls, tool_call);
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

    /// Finalize the accumulated state into a completed turn result.
    ///
    /// Returns [`StructuredTurnReductionError::Incomplete`] if the turn has
    /// not yet received a `Completed` event, and
    /// [`StructuredTurnReductionError::EmptyAssistantOutput`] if the completed
    /// turn produced no assistant items.
    pub fn finish(self) -> Result<StructuredTurnResult<T, O>, StructuredTurnReductionError> {
        // Check completion first so callers get Incomplete (not EmptyAssistantOutput)
        // when finish() is called on a fresh or mid-stream state.
        let finish_reason = self
            .finish_reason
            .ok_or(StructuredTurnReductionError::Incomplete)?;
        let usage = self.usage.ok_or(StructuredTurnReductionError::Incomplete)?;
        let committed_turn = self
            .committed_turn
            .ok_or(StructuredTurnReductionError::Incomplete)?;
        let assistant_turn = AssistantTurn::from_items(self.assistant_turn)
            .map_err(|_| StructuredTurnReductionError::EmptyAssistantOutput)?;
        let semantic = match (self.structured, self.refusal) {
            (Some(value), None) => StructuredTurnOutcome::Structured(value),
            (None, Some(refusal)) => StructuredTurnOutcome::Refusal(refusal),
            (None, None) => return Err(StructuredTurnReductionError::MissingSemantic),
            (Some(_), Some(_)) => return Err(StructuredTurnReductionError::ConflictingSemantic),
        };

        Ok(StructuredTurnResult {
            request_id: self.request_id,
            model: self.model,
            assistant_turn,
            tool_calls: self.tool_calls,
            semantic,
            finish_reason,
            usage,
            committed_turn,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StructuredTurnOutcome<O> {
    Structured(O),
    Refusal(String),
}

#[derive(Debug)]
pub struct StructuredTurnResult<T: Toolset, O: StructuredOutput> {
    pub request_id: Option<String>,
    pub model: String,
    pub assistant_turn: AssistantTurn,
    pub tool_calls: Vec<T::ToolCall>,
    pub semantic: StructuredTurnOutcome<O>,
    pub finish_reason: FinishReason,
    pub usage: Usage,
    pub committed_turn: CommittedTurn,
}

impl<T, O> Clone for StructuredTurnResult<T, O>
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
            semantic: self.semantic.clone(),
            finish_reason: self.finish_reason.clone(),
            usage: self.usage,
            committed_turn: self.committed_turn.clone(),
        }
    }
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

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum TextTurnReductionError {
    #[error("turn already completed")]
    AlreadyCompleted,
    #[error("completed turn produced no assistant items")]
    EmptyAssistantOutput,
    #[error("turn has not completed yet")]
    Incomplete,
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum StructuredTurnReductionError {
    #[error("turn already completed")]
    AlreadyCompleted,
    #[error("structured output appeared more than once")]
    DuplicateStructuredOutput,
    #[error("completed turn produced no assistant items")]
    EmptyAssistantOutput,
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

pub struct TextTurnReducer<T: Toolset> {
    state: TextTurnState<T>,
}

impl<T> Default for TextTurnReducer<T>
where
    T: Toolset,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TextTurnReducer<T>
where
    T: Toolset,
{
    pub fn new() -> Self {
        Self {
            state: TextTurnState::default(),
        }
    }

    pub fn state(&self) -> &TextTurnState<T> {
        &self.state
    }

    pub fn into_state(self) -> TextTurnState<T> {
        self.state
    }

    pub fn apply(&mut self, event: &TextTurnEvent<T>) -> Result<(), TextTurnReductionError> {
        self.state.apply(event)
    }

    pub fn into_result(self) -> Result<TextTurnResult<T>, TextTurnReductionError> {
        self.state.finish()
    }
}

pub struct StructuredTurnReducer<T: Toolset, O: StructuredOutput> {
    state: StructuredTurnState<T, O>,
}

impl<T, O> Default for StructuredTurnReducer<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, O> StructuredTurnReducer<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    pub fn new() -> Self {
        Self {
            state: StructuredTurnState::default(),
        }
    }

    pub fn state(&self) -> &StructuredTurnState<T, O> {
        &self.state
    }

    pub fn into_state(self) -> StructuredTurnState<T, O> {
        self.state
    }

    pub fn apply(
        &mut self,
        event: &StructuredTurnEvent<T, O>,
    ) -> Result<(), StructuredTurnReductionError> {
        self.state.apply(event)
    }

    pub fn into_result(
        self,
    ) -> Result<StructuredTurnResult<T, O>, (StructuredTurnReductionError, Option<CommittedTurn>)>
    {
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

        fn parse_tool_call(
            metadata: ToolMetadata,
            name: &str,
            arguments_json: &str,
        ) -> Result<Self::ToolCall, ToolCallError> {
            match name {
                "weather" => serde_json::from_str(arguments_json)
                    .map(|input| CallsCall::Weather(WeatherArgsCall { metadata, input }))
                    .map_err(|source| ToolCallError::Deserialize {
                        name: name.to_string(),
                        source,
                    }),
                _ => Err(ToolCallError::UnknownTool {
                    name: name.to_string(),
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
        let mut reducer = TextTurnReducer::<Tools>::new();
        reducer
            .apply(&TextTurnEvent::Started {
                request_id: Some("req-1".into()),
                model: "gpt-4.1".into(),
            })
            .unwrap();
        reducer
            .apply(&TextTurnEvent::TextDelta {
                delta: "checking ".into(),
            })
            .unwrap();
        reducer
            .apply(&TextTurnEvent::ToolCallReady(CallsCall::Weather(
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
            .apply(&TextTurnEvent::Completed {
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
        assert_eq!(result.assistant_turn.items().len(), 2);
        assert_eq!(result.assistant_turn.assistant_text(), "checking ");
    }

    #[test]
    fn structured_reducer_distinguishes_structured_and_refusal() {
        let mut reducer = StructuredTurnReducer::<Tools, Summary>::new();
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
