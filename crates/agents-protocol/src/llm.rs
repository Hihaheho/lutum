use std::{fmt, marker::PhantomData, pin::Pin};

use bon::Builder;
use futures::Stream;

use crate::{
    AgentError,
    budget::{RequestBudget, Usage},
    conversation::{ModelInput, RawJson, ToolMetadata},
    structured::StructuredOutput,
    toolset::{NoTools, ToolPolicy, Toolset},
};

pub type TextTurnEventStream<T, E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<TextTurnEvent<T>, E>> + Send + 'static>>;
pub type StructuredTurnEventStream<T, O, E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<StructuredTurnEvent<T, O>, E>> + Send + 'static>>;
pub type CompletionEventStream<E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<CompletionEvent, E>> + Send + 'static>>;
pub type ErasedTextTurnEventStream<E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<ErasedTextTurnEvent, E>> + Send + 'static>>;
pub type ErasedStructuredTurnEventStream<E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<ErasedStructuredTurnEvent, E>> + Send + 'static>>;

#[derive(
    Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(try_from = "String", into = "String")]
pub struct ModelName(String);

impl ModelName {
    pub fn new(model: impl Into<String>) -> Result<Self, ModelNameError> {
        let model = model.into();
        if model.trim().is_empty() {
            return Err(ModelNameError::Empty);
        }
        Ok(Self(model))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }
}

impl AsRef<str> for ModelName {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for ModelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<ModelName> for String {
    fn from(value: ModelName) -> Self {
        value.into_string()
    }
}

impl TryFrom<String> for ModelName {
    type Error = ModelNameError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<&str> for ModelName {
    type Error = ModelNameError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum ModelNameError {
    #[error("model must not be empty")]
    Empty,
}

#[derive(Builder, Clone, Debug, Default, PartialEq)]
#[builder(builder_type(name = GenerationParamsBuilder))]
pub struct GenerationParams {
    pub temperature: Option<Temperature>,
    pub max_output_tokens: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReasoningEffort {
    Low,
    #[default]
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReasoningSummary {
    #[default]
    Auto,
    Concise,
    Detailed,
}

#[derive(Builder, Clone, Debug, Default, PartialEq, Eq)]
#[builder(builder_type(name = ReasoningParamsBuilder))]
pub struct ReasoningParams {
    pub effort: Option<ReasoningEffort>,
    pub summary: Option<ReasoningSummary>,
}

#[derive(Builder, Clone, Debug, PartialEq)]
#[builder(builder_type(name = TurnConfigBuilder))]
pub struct TurnConfig<T: Toolset = NoTools> {
    pub model: ModelName,
    #[builder(default)]
    pub generation: GenerationParams,
    #[builder(default)]
    pub reasoning: ReasoningParams,
    #[builder(default)]
    pub tools: ToolPolicy<T>,
    #[builder(default = RequestBudget::unlimited())]
    pub budget: RequestBudget,
}

impl<T> TurnConfig<T>
where
    T: Toolset,
{
    pub fn new(model: ModelName) -> Self {
        Self {
            model,
            generation: GenerationParams::default(),
            reasoning: ReasoningParams::default(),
            tools: ToolPolicy::Disabled,
            budget: RequestBudget::unlimited(),
        }
    }
}

#[derive(Builder, Clone, Debug, PartialEq)]
#[builder(builder_type(name = TextTurnBuilder))]
pub struct TextTurn<T: Toolset = NoTools> {
    pub config: TurnConfig<T>,
}

impl<T> TextTurn<T>
where
    T: Toolset,
{
    pub fn new(model: ModelName) -> Self {
        Self {
            config: TurnConfig::new(model),
        }
    }
}

#[derive(Builder, Clone, Debug, PartialEq)]
#[builder(builder_type(name = StructuredOutputSpecBuilder))]
pub struct StructuredOutputSpec<O: StructuredOutput> {
    #[builder(skip = PhantomData)]
    _marker: PhantomData<fn() -> O>,
}

impl<O> Default for StructuredOutputSpec<O>
where
    O: StructuredOutput,
{
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

#[derive(Builder, Clone, Debug, PartialEq)]
#[builder(builder_type(name = StructuredTurnBuilder))]
pub struct StructuredTurn<T: Toolset, O: StructuredOutput> {
    pub config: TurnConfig<T>,
    #[builder(default)]
    pub output: StructuredOutputSpec<O>,
}

impl<T, O> StructuredTurn<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    pub fn new(model: ModelName) -> Self {
        Self {
            config: TurnConfig::new(model),
            output: StructuredOutputSpec::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterToolChoice {
    None,
    Auto,
    Required,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdapterTurnConfig {
    pub model: ModelName,
    pub generation: GenerationParams,
    pub reasoning: ReasoningParams,
    pub tools: Vec<AdapterToolDefinition>,
    pub tool_choice: AdapterToolChoice,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdapterTextTurn {
    pub config: AdapterTurnConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdapterStructuredOutputSpec {
    pub schema_name: String,
    pub schema: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdapterStructuredTurn {
    pub config: AdapterTurnConfig,
    pub output: AdapterStructuredOutputSpec,
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

#[derive(Builder, Clone, Debug, PartialEq)]
#[builder(builder_type(name = CompletionRequestBuilder))]
pub struct CompletionRequest {
    pub model: ModelName,
    #[builder(into)]
    pub prompt: String,
    #[builder(default)]
    pub options: CompletionOptions,
    #[builder(default = RequestBudget::unlimited())]
    pub budget: RequestBudget,
}

impl CompletionRequest {
    pub fn new(model: ModelName, prompt: impl Into<String>) -> Self {
        Self {
            model,
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
#[builder(builder_type(name = CompletionOptionsBuilder))]
pub struct CompletionOptions {
    pub temperature: Option<Temperature>,
    pub max_output_tokens: Option<u32>,
    #[builder(default)]
    pub stop: Vec<String>,
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
        id: crate::conversation::ToolCallId,
        name: crate::conversation::ToolName,
        arguments_json_delta: String,
    },
    ToolCallReady(T::ToolCall),
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
        id: crate::conversation::ToolCallId,
        name: crate::conversation::ToolName,
        arguments_json_delta: String,
    },
    ToolCallReady(T::ToolCall),
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ErasedTextTurnEvent {
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
        id: crate::conversation::ToolCallId,
        name: crate::conversation::ToolName,
        arguments_json_delta: String,
    },
    ToolCallReady(ToolMetadata),
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ErasedStructuredTurnEvent {
    Started {
        request_id: Option<String>,
        model: String,
    },
    StructuredOutputChunk {
        json_delta: String,
    },
    StructuredOutputReady(RawJson),
    ReasoningDelta {
        delta: String,
    },
    RefusalDelta {
        delta: String,
    },
    ToolCallChunk {
        id: crate::conversation::ToolCallId,
        name: crate::conversation::ToolName,
        arguments_json_delta: String,
    },
    ToolCallReady(ToolMetadata),
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
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
    async fn responses_text(
        &self,
        input: ModelInput,
        turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError>;

    async fn responses_structured(
        &self,
        input: ModelInput,
        turn: AdapterStructuredTurn,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError>;

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionEventStream, AgentError>;

    async fn recover_usage(
        &self,
        kind: StreamKind,
        request_id: &str,
    ) -> Result<Option<Usage>, AgentError>;
}

#[async_trait::async_trait]
impl<T> LlmAdapter for std::sync::Arc<T>
where
    T: LlmAdapter + ?Sized,
{
    async fn responses_text(
        &self,
        input: ModelInput,
        turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        (**self).responses_text(input, turn).await
    }

    async fn responses_structured(
        &self,
        input: ModelInput,
        turn: AdapterStructuredTurn,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
        (**self).responses_structured(input, turn).await
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionEventStream, AgentError> {
        (**self).completion(request).await
    }

    async fn recover_usage(
        &self,
        kind: StreamKind,
        request_id: &str,
    ) -> Result<Option<Usage>, AgentError> {
        (**self).recover_usage(kind, request_id).await
    }
}
