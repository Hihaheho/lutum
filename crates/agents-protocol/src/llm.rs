use std::{borrow::Cow, fmt, marker::PhantomData, pin::Pin, sync::Arc};

use bon::Builder;
use futures::Stream;

use crate::{
    AgentError,
    budget::{RequestBudget, Usage},
    conversation::{ModelInput, RawJson, ToolMetadata},
    structured::StructuredOutput,
    toolset::{NoTools, ToolPolicy, Toolset},
    transcript::CommittedTurn,
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

#[derive(Builder, Clone, Debug, PartialEq)]
#[builder(builder_type(name = TurnConfigBuilder))]
pub struct TurnConfig<T: Toolset = NoTools> {
    pub model: ModelName,
    #[builder(default)]
    pub generation: GenerationParams,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterToolChoice {
    None,
    Auto,
    Required,
    Specific(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdapterTurnConfig {
    pub model: ModelName,
    pub generation: GenerationParams,
    pub tools: Vec<AdapterToolDefinition>,
    pub tool_choice: AdapterToolChoice,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ModelSelection {
    /// Borrowed model names must be truly `'static`; runtime-derived values
    /// should use `Cow::Owned`.
    pub primary: Option<Cow<'static, str>>,
    /// Borrowed model names must be truly `'static`; runtime-derived values
    /// should use `Cow::Owned`.
    pub fallbacks: Option<Vec<Cow<'static, str>>>,
}

pub trait ModelSelector: Send + Sync {
    fn select_model(&self, extensions: &crate::extensions::RequestExtensions) -> ModelSelection;
}

#[derive(Clone)]
pub struct AdapterTextTurn {
    pub config: AdapterTurnConfig,
    pub extensions: Arc<crate::extensions::RequestExtensions>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AdapterStructuredOutputSpec {
    pub schema_name: String,
    pub schema: serde_json::Value,
}

#[derive(Clone)]
pub struct AdapterStructuredTurn {
    pub config: AdapterTurnConfig,
    pub extensions: Arc<crate::extensions::RequestExtensions>,
    pub output: AdapterStructuredOutputSpec,
}

impl fmt::Debug for AdapterTextTurn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdapterTextTurn")
            .field("config", &self.config)
            .field("extensions", &"<opaque>")
            .finish()
    }
}

impl PartialEq for AdapterTextTurn {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config && Arc::ptr_eq(&self.extensions, &other.extensions)
    }
}

impl fmt::Debug for AdapterStructuredTurn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdapterStructuredTurn")
            .field("config", &self.config)
            .field("extensions", &"<opaque>")
            .field("output", &self.output)
            .finish()
    }
}

impl PartialEq for AdapterStructuredTurn {
    fn eq(&self, other: &Self) -> bool {
        self.config == other.config
            && Arc::ptr_eq(&self.extensions, &other.extensions)
            && self.output == other.output
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

#[derive(Clone, Debug)]
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
        committed_turn: CommittedTurn,
    },
}

#[derive(Clone, Debug)]
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
        committed_turn: CommittedTurn,
    },
}

#[derive(Clone, Debug)]
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
        committed_turn: CommittedTurn,
    },
}

#[derive(Clone, Debug)]
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
        committed_turn: CommittedTurn,
    },
}

impl<T> PartialEq for TextTurnEvent<T>
where
    T: Toolset,
    T::ToolCall: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Started {
                    request_id: lhs_request_id,
                    model: lhs_model,
                },
                Self::Started {
                    request_id: rhs_request_id,
                    model: rhs_model,
                },
            ) => lhs_request_id == rhs_request_id && lhs_model == rhs_model,
            (Self::TextDelta { delta: lhs }, Self::TextDelta { delta: rhs }) => lhs == rhs,
            (Self::ReasoningDelta { delta: lhs }, Self::ReasoningDelta { delta: rhs }) => {
                lhs == rhs
            }
            (Self::RefusalDelta { delta: lhs }, Self::RefusalDelta { delta: rhs }) => lhs == rhs,
            (
                Self::ToolCallChunk {
                    id: lhs_id,
                    name: lhs_name,
                    arguments_json_delta: lhs_delta,
                },
                Self::ToolCallChunk {
                    id: rhs_id,
                    name: rhs_name,
                    arguments_json_delta: rhs_delta,
                },
            ) => lhs_id == rhs_id && lhs_name == rhs_name && lhs_delta == rhs_delta,
            (Self::ToolCallReady(lhs), Self::ToolCallReady(rhs)) => lhs == rhs,
            (
                Self::Completed {
                    request_id: lhs_request_id,
                    finish_reason: lhs_finish_reason,
                    usage: lhs_usage,
                    committed_turn: lhs_committed_turn,
                },
                Self::Completed {
                    request_id: rhs_request_id,
                    finish_reason: rhs_finish_reason,
                    usage: rhs_usage,
                    committed_turn: rhs_committed_turn,
                },
            ) => {
                lhs_request_id == rhs_request_id
                    && lhs_finish_reason == rhs_finish_reason
                    && lhs_usage == rhs_usage
                    && Arc::ptr_eq(lhs_committed_turn, rhs_committed_turn)
            }
            _ => false,
        }
    }
}

impl<T, O> PartialEq for StructuredTurnEvent<T, O>
where
    T: Toolset,
    T::ToolCall: PartialEq,
    O: StructuredOutput + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Started {
                    request_id: lhs_request_id,
                    model: lhs_model,
                },
                Self::Started {
                    request_id: rhs_request_id,
                    model: rhs_model,
                },
            ) => lhs_request_id == rhs_request_id && lhs_model == rhs_model,
            (
                Self::StructuredOutputChunk { json_delta: lhs },
                Self::StructuredOutputChunk { json_delta: rhs },
            ) => lhs == rhs,
            (Self::StructuredOutputReady(lhs), Self::StructuredOutputReady(rhs)) => lhs == rhs,
            (Self::ReasoningDelta { delta: lhs }, Self::ReasoningDelta { delta: rhs }) => {
                lhs == rhs
            }
            (Self::RefusalDelta { delta: lhs }, Self::RefusalDelta { delta: rhs }) => lhs == rhs,
            (
                Self::ToolCallChunk {
                    id: lhs_id,
                    name: lhs_name,
                    arguments_json_delta: lhs_delta,
                },
                Self::ToolCallChunk {
                    id: rhs_id,
                    name: rhs_name,
                    arguments_json_delta: rhs_delta,
                },
            ) => lhs_id == rhs_id && lhs_name == rhs_name && lhs_delta == rhs_delta,
            (Self::ToolCallReady(lhs), Self::ToolCallReady(rhs)) => lhs == rhs,
            (
                Self::Completed {
                    request_id: lhs_request_id,
                    finish_reason: lhs_finish_reason,
                    usage: lhs_usage,
                    committed_turn: lhs_committed_turn,
                },
                Self::Completed {
                    request_id: rhs_request_id,
                    finish_reason: rhs_finish_reason,
                    usage: rhs_usage,
                    committed_turn: rhs_committed_turn,
                },
            ) => {
                lhs_request_id == rhs_request_id
                    && lhs_finish_reason == rhs_finish_reason
                    && lhs_usage == rhs_usage
                    && Arc::ptr_eq(lhs_committed_turn, rhs_committed_turn)
            }
            _ => false,
        }
    }
}

impl PartialEq for ErasedTextTurnEvent {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Started {
                    request_id: lhs_request_id,
                    model: lhs_model,
                },
                Self::Started {
                    request_id: rhs_request_id,
                    model: rhs_model,
                },
            ) => lhs_request_id == rhs_request_id && lhs_model == rhs_model,
            (Self::TextDelta { delta: lhs }, Self::TextDelta { delta: rhs }) => lhs == rhs,
            (Self::ReasoningDelta { delta: lhs }, Self::ReasoningDelta { delta: rhs }) => {
                lhs == rhs
            }
            (Self::RefusalDelta { delta: lhs }, Self::RefusalDelta { delta: rhs }) => lhs == rhs,
            (
                Self::ToolCallChunk {
                    id: lhs_id,
                    name: lhs_name,
                    arguments_json_delta: lhs_delta,
                },
                Self::ToolCallChunk {
                    id: rhs_id,
                    name: rhs_name,
                    arguments_json_delta: rhs_delta,
                },
            ) => lhs_id == rhs_id && lhs_name == rhs_name && lhs_delta == rhs_delta,
            (Self::ToolCallReady(lhs), Self::ToolCallReady(rhs)) => lhs == rhs,
            (
                Self::Completed {
                    request_id: lhs_request_id,
                    finish_reason: lhs_finish_reason,
                    usage: lhs_usage,
                    ..
                },
                Self::Completed {
                    request_id: rhs_request_id,
                    finish_reason: rhs_finish_reason,
                    usage: rhs_usage,
                    ..
                },
            ) => {
                lhs_request_id == rhs_request_id
                    && lhs_finish_reason == rhs_finish_reason
                    && lhs_usage == rhs_usage
            }
            _ => false,
        }
    }
}

impl PartialEq for ErasedStructuredTurnEvent {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Started {
                    request_id: lhs_request_id,
                    model: lhs_model,
                },
                Self::Started {
                    request_id: rhs_request_id,
                    model: rhs_model,
                },
            ) => lhs_request_id == rhs_request_id && lhs_model == rhs_model,
            (
                Self::StructuredOutputChunk { json_delta: lhs },
                Self::StructuredOutputChunk { json_delta: rhs },
            ) => lhs == rhs,
            (Self::StructuredOutputReady(lhs), Self::StructuredOutputReady(rhs)) => lhs == rhs,
            (Self::ReasoningDelta { delta: lhs }, Self::ReasoningDelta { delta: rhs }) => {
                lhs == rhs
            }
            (Self::RefusalDelta { delta: lhs }, Self::RefusalDelta { delta: rhs }) => lhs == rhs,
            (
                Self::ToolCallChunk {
                    id: lhs_id,
                    name: lhs_name,
                    arguments_json_delta: lhs_delta,
                },
                Self::ToolCallChunk {
                    id: rhs_id,
                    name: rhs_name,
                    arguments_json_delta: rhs_delta,
                },
            ) => lhs_id == rhs_id && lhs_name == rhs_name && lhs_delta == rhs_delta,
            (Self::ToolCallReady(lhs), Self::ToolCallReady(rhs)) => lhs == rhs,
            (
                Self::Completed {
                    request_id: lhs_request_id,
                    finish_reason: lhs_finish_reason,
                    usage: lhs_usage,
                    ..
                },
                Self::Completed {
                    request_id: rhs_request_id,
                    finish_reason: rhs_finish_reason,
                    usage: rhs_usage,
                    ..
                },
            ) => {
                lhs_request_id == rhs_request_id
                    && lhs_finish_reason == rhs_finish_reason
                    && lhs_usage == rhs_usage
            }
            _ => false,
        }
    }
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

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    ContentFilter,
    Unknown(String),
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum OperationKind {
    TextTurn,
    StructuredTurn,
    Completion,
}

#[async_trait::async_trait]
pub trait TurnAdapter: Send + Sync + 'static {
    async fn text_turn(
        &self,
        input: ModelInput,
        turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError>;

    async fn structured_turn(
        &self,
        input: ModelInput,
        turn: AdapterStructuredTurn,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError>;
}

#[async_trait::async_trait]
pub trait CompletionAdapter: Send + Sync + 'static {
    async fn completion(
        &self,
        request: CompletionRequest,
        extensions: &crate::extensions::RequestExtensions,
    ) -> Result<CompletionEventStream, AgentError>;
}

#[async_trait::async_trait]
pub trait UsageRecoveryAdapter: Send + Sync + 'static {
    async fn recover_usage(
        &self,
        kind: OperationKind,
        request_id: &str,
    ) -> Result<Option<Usage>, AgentError>;
}

#[async_trait::async_trait]
impl<T> TurnAdapter for Arc<T>
where
    T: TurnAdapter + ?Sized,
{
    async fn text_turn(
        &self,
        input: ModelInput,
        turn: AdapterTextTurn,
    ) -> Result<ErasedTextTurnEventStream, AgentError> {
        (**self).text_turn(input, turn).await
    }

    async fn structured_turn(
        &self,
        input: ModelInput,
        turn: AdapterStructuredTurn,
    ) -> Result<ErasedStructuredTurnEventStream, AgentError> {
        (**self).structured_turn(input, turn).await
    }
}

#[async_trait::async_trait]
impl<T> CompletionAdapter for Arc<T>
where
    T: CompletionAdapter + ?Sized,
{
    async fn completion(
        &self,
        request: CompletionRequest,
        extensions: &crate::extensions::RequestExtensions,
    ) -> Result<CompletionEventStream, AgentError> {
        (**self).completion(request, extensions).await
    }
}

#[async_trait::async_trait]
impl<T> UsageRecoveryAdapter for Arc<T>
where
    T: UsageRecoveryAdapter + ?Sized,
{
    async fn recover_usage(
        &self,
        kind: OperationKind,
        request_id: &str,
    ) -> Result<Option<Usage>, AgentError> {
        (**self).recover_usage(kind, request_id).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::transcript::AssistantTurnView;

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct Summary {
        answer: String,
    }

    #[test]
    fn text_completed_equality_checks_committed_turn_identity() {
        let shared_turn = Arc::new(AssistantTurnView::from_items(&[]));
        let different_turn = Arc::new(AssistantTurnView::from_items(&[]));

        let lhs = TextTurnEvent::<NoTools>::Completed {
            request_id: Some("req-1".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
            committed_turn: shared_turn.clone(),
        };
        let same = TextTurnEvent::<NoTools>::Completed {
            request_id: Some("req-1".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
            committed_turn: shared_turn,
        };
        let different = TextTurnEvent::<NoTools>::Completed {
            request_id: Some("req-1".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
            committed_turn: different_turn,
        };

        assert_eq!(lhs, same);
        assert_ne!(lhs, different);
    }

    #[test]
    fn structured_completed_equality_checks_committed_turn_identity() {
        let shared_turn = Arc::new(AssistantTurnView::from_items(&[]));
        let different_turn = Arc::new(AssistantTurnView::from_items(&[]));

        let lhs = StructuredTurnEvent::<NoTools, Summary>::Completed {
            request_id: Some("req-2".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
            committed_turn: shared_turn.clone(),
        };
        let same = StructuredTurnEvent::<NoTools, Summary>::Completed {
            request_id: Some("req-2".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
            committed_turn: shared_turn,
        };
        let different = StructuredTurnEvent::<NoTools, Summary>::Completed {
            request_id: Some("req-2".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
            committed_turn: different_turn,
        };

        assert_eq!(lhs, same);
        assert_ne!(lhs, different);
    }
}
