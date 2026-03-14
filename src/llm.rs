use std::{marker::PhantomData, pin::Pin};

use bon::Builder;
use futures::Stream;

use crate::{
    budget::{RequestBudget, Usage},
    conversation::{ModelInput, RawJson, ToolCallId, ToolName},
    structured::StructuredOutput,
    toolset::{ToolMode, Toolset},
};

pub type TextTurnEventStream<T, E> =
    Pin<Box<dyn Stream<Item = Result<TextTurnEvent<T>, E>> + Send + 'static>>;
pub type StructuredTurnEventStream<T, O, E> =
    Pin<Box<dyn Stream<Item = Result<StructuredTurnEvent<T, O>, E>> + Send + 'static>>;
pub type CompletionEventStream<E> =
    Pin<Box<dyn Stream<Item = Result<CompletionEvent, E>> + Send + 'static>>;

#[derive(Builder, Clone, Debug, Default, PartialEq)]
pub struct ResponsesOptions {
    pub temperature: Option<Temperature>,
    pub max_output_tokens: Option<u32>,
    pub reasoning: Option<ReasoningConfig>,
    pub thinking_budget: Option<ThinkingBudget>,
}

impl ResponsesOptions {
    pub fn effective_reasoning(&self) -> Option<ReasoningConfig> {
        match (&self.reasoning, self.thinking_budget) {
            (Some(reasoning), Some(thinking_budget)) => {
                let mut reasoning = reasoning.clone();
                reasoning.effort = thinking_budget;
                Some(reasoning)
            }
            (Some(reasoning), None) => Some(reasoning.clone()),
            (None, Some(thinking_budget)) => Some(ReasoningConfig {
                effort: thinking_budget,
                ..ReasoningConfig::default()
            }),
            (None, None) => None,
        }
    }
}

#[derive(Builder, Clone, Debug, PartialEq)]
pub struct TextTurnRequest<T: Toolset> {
    #[builder(into)]
    pub model: String,
    #[builder(default)]
    pub options: ResponsesOptions,
    #[builder(default)]
    pub tool_mode: ToolMode<T>,
    #[builder(default = RequestBudget::unlimited())]
    pub budget: RequestBudget,
}

impl<T> TextTurnRequest<T>
where
    T: Toolset,
{
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            options: ResponsesOptions::default(),
            tool_mode: ToolMode::<T>::Disabled,
            budget: RequestBudget::unlimited(),
        }
    }

    pub fn with_options(mut self, options: ResponsesOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_tool_mode(mut self, tool_mode: ToolMode<T>) -> Self {
        self.tool_mode = tool_mode;
        self
    }

    pub fn with_budget(mut self, budget: RequestBudget) -> Self {
        self.budget = budget;
        self
    }
}

#[derive(Builder, Clone, Debug, PartialEq)]
pub struct StructuredTurnRequest<T: Toolset, O: StructuredOutput> {
    #[builder(into)]
    pub model: String,
    #[builder(default)]
    pub options: ResponsesOptions,
    #[builder(default)]
    pub tool_mode: ToolMode<T>,
    #[builder(default = RequestBudget::unlimited())]
    pub budget: RequestBudget,
    #[builder(skip = PhantomData)]
    _marker: PhantomData<fn() -> O>,
}

impl<T, O> StructuredTurnRequest<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            options: ResponsesOptions::default(),
            tool_mode: ToolMode::<T>::Disabled,
            budget: RequestBudget::unlimited(),
            _marker: PhantomData,
        }
    }

    pub fn with_options(mut self, options: ResponsesOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_tool_mode(mut self, tool_mode: ToolMode<T>) -> Self {
        self.tool_mode = tool_mode;
        self
    }

    pub fn with_budget(mut self, budget: RequestBudget) -> Self {
        self.budget = budget;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Temperature(f32);

impl Temperature {
    pub const MIN: f32 = 0.0;
    pub const MAX: f32 = 2.0;

    pub fn new(value: f32) -> Result<Self, TemperatureError> {
        if !value.is_finite() {
            return Err(TemperatureError::NonFinite);
        }
        if !(Self::MIN..=Self::MAX).contains(&value) {
            return Err(TemperatureError::OutOfRange { value });
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for Temperature {
    type Error = TemperatureError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq)]
pub enum TemperatureError {
    #[error("temperature must be finite")]
    NonFinite,
    #[error("temperature {value} must be in the range [0.0, 2.0]")]
    OutOfRange { value: f32 },
}

#[derive(Builder, Clone, Debug, Default, PartialEq, Eq)]
pub struct ReasoningConfig {
    #[builder(default)]
    pub effort: ReasoningEffort,
    #[builder(default)]
    pub summary: ReasoningSummary,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ThinkingBudget {
    Low,
    #[default]
    Medium,
    High,
}

pub type ReasoningEffort = ThinkingBudget;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReasoningSummary {
    #[default]
    Auto,
    Concise,
    Detailed,
}

#[derive(Builder, Clone, Debug, PartialEq)]
pub struct CompletionRequest {
    #[builder(into)]
    pub model: String,
    #[builder(into)]
    pub prompt: String,
    #[builder(default)]
    pub options: CompletionOptions,
    #[builder(default = RequestBudget::unlimited())]
    pub budget: RequestBudget,
}

impl CompletionRequest {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            options: CompletionOptions::default(),
            budget: RequestBudget::unlimited(),
        }
    }

    pub fn with_options(mut self, options: CompletionOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_budget(mut self, budget: RequestBudget) -> Self {
        self.budget = budget;
        self
    }
}

#[derive(Builder, Clone, Debug, Default, PartialEq)]
pub struct CompletionOptions {
    pub temperature: Option<Temperature>,
    pub max_output_tokens: Option<u32>,
    #[builder(default)]
    pub stop: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedToolInvocation<C> {
    pub id: ToolCallId,
    pub name: ToolName,
    pub call: C,
    pub arguments: RawJson,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextTurnEvent<T: Toolset> {
    Started {
        request_id: Option<String>,
        model: String,
    },
    TextDelta {
        delta: String,
    },
    ReasoningDelta {
        delta: String,
    },
    RefusalDelta {
        delta: String,
    },
    ToolCallChunk {
        id: ToolCallId,
        name: ToolName,
        arguments_json_delta: String,
    },
    ToolCallReady(TypedToolInvocation<T::Call>),
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructuredTurnEvent<T: Toolset, O: StructuredOutput> {
    Started {
        request_id: Option<String>,
        model: String,
    },
    StructuredOutputChunk {
        json_delta: String,
    },
    StructuredOutputReady(O),
    ReasoningDelta {
        delta: String,
    },
    RefusalDelta {
        delta: String,
    },
    ToolCallChunk {
        id: ToolCallId,
        name: ToolName,
        arguments_json_delta: String,
    },
    ToolCallReady(TypedToolInvocation<T::Call>),
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompletionEvent {
    Started {
        request_id: Option<String>,
        model: String,
    },
    TextDelta(String),
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    ContentFilter,
    Unknown(String),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum StreamKind {
    ResponsesText,
    ResponsesStructured,
    Completion,
}

#[async_trait::async_trait]
pub trait LlmAdapter: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    async fn responses_text<T>(
        &self,
        input: ModelInput,
        turn: TextTurnRequest<T>,
    ) -> Result<TextTurnEventStream<T, Self::Error>, Self::Error>
    where
        T: Toolset;

    async fn responses_structured<T, O>(
        &self,
        input: ModelInput,
        turn: StructuredTurnRequest<T, O>,
    ) -> Result<StructuredTurnEventStream<T, O, Self::Error>, Self::Error>
    where
        T: Toolset,
        O: StructuredOutput;

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionEventStream<Self::Error>, Self::Error>;

    async fn recover_usage(
        &self,
        kind: StreamKind,
        request_id: &str,
    ) -> Result<Option<Usage>, Self::Error>;
}
