use std::{fmt, marker::PhantomData, pin::Pin, sync::Arc, time::Duration};

use bon::Builder;
use futures::Stream;

use crate::{
    AgentError,
    budget::{RequestBudget, Usage},
    conversation::{ModelInput, RawJson, ToolMetadata},
    error::RequestFailureKind,
    extensions::RequestExtensions,
    structured::StructuredOutput,
    toolset::{NoTools, RecoverableToolCallIssue, ToolConstraints, Toolset},
    transcript::CommittedTurn,
};

#[cfg(not(target_family = "wasm"))]
pub type TextTurnEventStream<E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<TextTurnEvent, E>> + Send + Sync + 'static>>;
#[cfg(target_family = "wasm")]
pub type TextTurnEventStream<E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<TextTurnEvent, E>> + 'static>>;
#[cfg(not(target_family = "wasm"))]
pub type TextTurnEventStreamWithTools<T, E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<TextTurnEventWithTools<T>, E>> + Send + Sync + 'static>>;
#[cfg(target_family = "wasm")]
pub type TextTurnEventStreamWithTools<T, E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<TextTurnEventWithTools<T>, E>> + 'static>>;
#[cfg(not(target_family = "wasm"))]
pub type StructuredTurnEventStream<O, E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<StructuredTurnEvent<O>, E>> + Send + Sync + 'static>>;
#[cfg(target_family = "wasm")]
pub type StructuredTurnEventStream<O, E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<StructuredTurnEvent<O>, E>> + 'static>>;
#[cfg(not(target_family = "wasm"))]
pub type StructuredTurnEventStreamWithTools<T, O, E = AgentError> = Pin<
    Box<dyn Stream<Item = Result<StructuredTurnEventWithTools<T, O>, E>> + Send + Sync + 'static>,
>;
#[cfg(target_family = "wasm")]
pub type StructuredTurnEventStreamWithTools<T, O, E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<StructuredTurnEventWithTools<T, O>, E>> + 'static>>;
#[cfg(not(target_family = "wasm"))]
pub type StructuredCompletionEventStream<O, E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<StructuredCompletionEvent<O>, E>> + Send + Sync + 'static>>;
#[cfg(target_family = "wasm")]
pub type StructuredCompletionEventStream<O, E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<StructuredCompletionEvent<O>, E>> + 'static>>;
#[cfg(not(target_family = "wasm"))]
pub type CompletionEventStream<E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<CompletionEvent, E>> + Send + Sync + 'static>>;
#[cfg(target_family = "wasm")]
pub type CompletionEventStream<E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<CompletionEvent, E>> + 'static>>;
#[cfg(not(target_family = "wasm"))]
pub type ErasedTextTurnEventStream<E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<ErasedTextTurnEvent, E>> + Send + Sync + 'static>>;
#[cfg(target_family = "wasm")]
pub type ErasedTextTurnEventStream<E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<ErasedTextTurnEvent, E>> + 'static>>;
#[cfg(not(target_family = "wasm"))]
pub type ErasedStructuredTurnEventStream<E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<ErasedStructuredTurnEvent, E>> + Send + Sync + 'static>>;
#[cfg(target_family = "wasm")]
pub type ErasedStructuredTurnEventStream<E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<ErasedStructuredTurnEvent, E>> + 'static>>;
#[cfg(not(target_family = "wasm"))]
pub type ErasedStructuredCompletionEventStream<E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<ErasedStructuredCompletionEvent, E>> + Send + Sync + 'static>>;
#[cfg(target_family = "wasm")]
pub type ErasedStructuredCompletionEventStream<E = AgentError> =
    Pin<Box<dyn Stream<Item = Result<ErasedStructuredCompletionEvent, E>> + 'static>>;

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
    pub top_p: Option<TopP>,
    pub frequency_penalty: Option<FrequencyPenalty>,
    pub presence_penalty: Option<PresencePenalty>,
    pub max_output_tokens: Option<MaxOutputTokens>,
    pub seed: Option<Seed>,
    pub stop_sequences: Option<StopSequences>,
}

impl GenerationParams {
    pub fn resolved_with_extensions(mut self, extensions: &RequestExtensions) -> Self {
        self.resolve_with_extensions(extensions);
        self
    }

    pub fn resolve_with_extensions(&mut self, extensions: &RequestExtensions) {
        resolve_generation_param::<Temperature>(self, extensions);
        resolve_generation_param::<TopP>(self, extensions);
        resolve_generation_param::<FrequencyPenalty>(self, extensions);
        resolve_generation_param::<PresencePenalty>(self, extensions);
        resolve_generation_param::<MaxOutputTokens>(self, extensions);
        resolve_generation_param::<Seed>(self, extensions);
        resolve_generation_param::<StopSequences>(self, extensions);
    }
}

fn resolve_generation_param<T>(generation: &mut GenerationParams, extensions: &RequestExtensions)
where
    T: GenerationParamValue,
{
    if T::get(generation).is_some() {
        return;
    }
    match extensions.get::<GenerationSetting<T>>() {
        Some(GenerationSetting::Set(value)) => T::set(generation, value.clone()),
        Some(GenerationSetting::Clear) | None => {}
    }
}

mod generation_param_seal {
    pub trait Sealed {}
}

pub trait GenerationParamValue:
    generation_param_seal::Sealed + Clone + Send + Sync + 'static
{
    fn get(generation: &GenerationParams) -> Option<Self>;
    fn set(generation: &mut GenerationParams, value: Self);
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenerationSetting<T> {
    Set(T),
    Clear,
}

impl<T> GenerationSetting<T> {
    pub fn set(value: T) -> Self {
        Self::Set(value)
    }

    pub fn clear() -> Self {
        Self::Clear
    }
}

impl<T> From<T> for GenerationSetting<T> {
    fn from(value: T) -> Self {
        Self::Set(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxOutputTokens(u32);

impl MaxOutputTokens {
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for MaxOutputTokens {
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

impl generation_param_seal::Sealed for MaxOutputTokens {}

impl GenerationParamValue for MaxOutputTokens {
    fn get(generation: &GenerationParams) -> Option<Self> {
        generation.max_output_tokens
    }

    fn set(generation: &mut GenerationParams, value: Self) {
        generation.max_output_tokens = Some(value);
    }
}

#[derive(Builder, Clone, Debug, PartialEq)]
#[builder(builder_type(name = TurnConfigBuilder))]
pub struct TurnConfig<T: Toolset = NoTools> {
    #[builder(default)]
    pub generation: GenerationParams,
    #[builder(default)]
    pub tools: ToolConstraints<T>,
    #[builder(default = RequestBudget::unlimited())]
    pub budget: RequestBudget,
}

impl<T> TurnConfig<T>
where
    T: Toolset,
{
    pub fn new() -> Self {
        Self {
            generation: GenerationParams::default(),
            tools: ToolConstraints::default(),
            budget: RequestBudget::unlimited(),
        }
    }
}

impl<T> Default for TurnConfig<T>
where
    T: Toolset,
{
    fn default() -> Self {
        Self::new()
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
    pub fn new() -> Self {
        Self {
            config: TurnConfig::new(),
        }
    }
}

impl<T> Default for TextTurn<T>
where
    T: Toolset,
{
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Builder, Clone, Debug, PartialEq)]
#[builder(builder_type(name = StructuredOutputSpecBuilder))]
pub struct StructuredOutputSpec<O: StructuredOutput> {
    #[builder(skip = PhantomData)]
    _marker: PhantomData<fn() -> O>,
    pub schema_name: Option<String>,
    pub schema: Option<serde_json::Value>,
}

impl<O> StructuredOutputSpec<O>
where
    O: StructuredOutput,
{
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the JSON Schema sent to the model for this structured output.
    ///
    /// The model response is still deserialized as `O`. Use `serde_json::Value`
    /// as `O` when both the schema and the decoded shape are runtime-defined.
    pub fn with_json_schema(
        mut self,
        schema_name: impl Into<String>,
        schema: impl Into<serde_json::Value>,
    ) -> Self {
        self.schema_name = Some(schema_name.into());
        self.schema = Some(schema.into());
        self
    }
}

impl<O> Default for StructuredOutputSpec<O>
where
    O: StructuredOutput,
{
    fn default() -> Self {
        Self {
            _marker: PhantomData,
            schema_name: None,
            schema: None,
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
    pub fn new() -> Self {
        Self {
            config: TurnConfig::new(),
            output: StructuredOutputSpec::default(),
        }
    }

    pub fn with_output_schema(
        mut self,
        schema_name: impl Into<String>,
        schema: impl Into<serde_json::Value>,
    ) -> Self {
        self.output = self.output.with_json_schema(schema_name, schema);
        self
    }
}

impl<T, O> Default for StructuredTurn<T, O>
where
    T: Toolset,
    O: StructuredOutput,
{
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Builder, Clone, Debug, PartialEq)]
#[builder(builder_type(name = StructuredCompletionRequestBuilder))]
pub struct StructuredCompletionRequest<O: StructuredOutput> {
    pub system: Option<String>,
    #[builder(into)]
    pub prompt: String,
    #[builder(default)]
    pub generation: GenerationParams,
    #[builder(default = RequestBudget::unlimited())]
    pub budget: RequestBudget,
    #[builder(default)]
    pub output: StructuredOutputSpec<O>,
}

impl<O> StructuredCompletionRequest<O>
where
    O: StructuredOutput,
{
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            system: None,
            prompt: prompt.into(),
            generation: GenerationParams::default(),
            budget: RequestBudget::unlimited(),
            output: StructuredOutputSpec::default(),
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn with_output_schema(
        mut self,
        schema_name: impl Into<String>,
        schema: impl Into<serde_json::Value>,
    ) -> Self {
        self.output = self.output.with_json_schema(schema_name, schema);
        self
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
    pub generation: GenerationParams,
    pub tools: Vec<AdapterToolDefinition>,
    pub tool_choice: AdapterToolChoice,
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

#[derive(Clone, Debug, PartialEq)]
pub struct AdapterStructuredCompletionRequest {
    pub system: Option<String>,
    pub prompt: String,
    pub generation: GenerationParams,
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

impl GenerationParamValue for Temperature {
    fn get(generation: &GenerationParams) -> Option<Self> {
        generation.temperature
    }

    fn set(generation: &mut GenerationParams, value: Self) {
        generation.temperature = Some(value);
    }
}

impl generation_param_seal::Sealed for Temperature {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TopP(f32);

impl TopP {
    pub const MIN: f32 = 0.0;
    pub const MAX: f32 = 1.0;

    pub fn new(value: f32) -> Result<Self, TopPError> {
        if !value.is_finite() {
            return Err(TopPError::NonFinite);
        }
        if !(Self::MIN..=Self::MAX).contains(&value) {
            return Err(TopPError::OutOfRange { value });
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for TopP {
    type Error = TopPError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq)]
pub enum TopPError {
    #[error("top_p must be finite")]
    NonFinite,
    #[error("top_p {value} must be in the range [0.0, 1.0]")]
    OutOfRange { value: f32 },
}

impl GenerationParamValue for TopP {
    fn get(generation: &GenerationParams) -> Option<Self> {
        generation.top_p
    }

    fn set(generation: &mut GenerationParams, value: Self) {
        generation.top_p = Some(value);
    }
}

impl generation_param_seal::Sealed for TopP {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrequencyPenalty(f32);

impl FrequencyPenalty {
    pub const MIN: f32 = -2.0;
    pub const MAX: f32 = 2.0;

    pub fn new(value: f32) -> Result<Self, PenaltyError> {
        if !value.is_finite() {
            return Err(PenaltyError::NonFinite);
        }
        if !(Self::MIN..=Self::MAX).contains(&value) {
            return Err(PenaltyError::OutOfRange { value });
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for FrequencyPenalty {
    type Error = PenaltyError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl GenerationParamValue for FrequencyPenalty {
    fn get(generation: &GenerationParams) -> Option<Self> {
        generation.frequency_penalty
    }

    fn set(generation: &mut GenerationParams, value: Self) {
        generation.frequency_penalty = Some(value);
    }
}

impl generation_param_seal::Sealed for FrequencyPenalty {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PresencePenalty(f32);

impl PresencePenalty {
    pub const MIN: f32 = -2.0;
    pub const MAX: f32 = 2.0;

    pub fn new(value: f32) -> Result<Self, PenaltyError> {
        if !value.is_finite() {
            return Err(PenaltyError::NonFinite);
        }
        if !(Self::MIN..=Self::MAX).contains(&value) {
            return Err(PenaltyError::OutOfRange { value });
        }
        Ok(Self(value))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for PresencePenalty {
    type Error = PenaltyError;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl GenerationParamValue for PresencePenalty {
    fn get(generation: &GenerationParams) -> Option<Self> {
        generation.presence_penalty
    }

    fn set(generation: &mut GenerationParams, value: Self) {
        generation.presence_penalty = Some(value);
    }
}

impl generation_param_seal::Sealed for PresencePenalty {}

#[derive(Clone, Debug, thiserror::Error, PartialEq)]
pub enum PenaltyError {
    #[error("penalty must be finite")]
    NonFinite,
    #[error("penalty {value} must be in the range [-2.0, 2.0]")]
    OutOfRange { value: f32 },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Seed(u64);

impl Seed {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub fn get(self) -> u64 {
        self.0
    }
}

impl From<u64> for Seed {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl GenerationParamValue for Seed {
    fn get(generation: &GenerationParams) -> Option<Self> {
        generation.seed
    }

    fn set(generation: &mut GenerationParams, value: Self) {
        generation.seed = Some(value);
    }
}

impl generation_param_seal::Sealed for Seed {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StopSequences(Vec<String>);

impl StopSequences {
    pub fn new(sequences: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self(sequences.into_iter().map(Into::into).collect())
    }

    pub fn one(sequence: impl Into<String>) -> Self {
        Self(vec![sequence.into()])
    }

    pub fn as_slice(&self) -> &[String] {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn into_vec(self) -> Vec<String> {
        self.0
    }
}

impl<S> From<Vec<S>> for StopSequences
where
    S: Into<String>,
{
    fn from(value: Vec<S>) -> Self {
        Self::new(value)
    }
}

impl<const N: usize> From<[&str; N]> for StopSequences {
    fn from(value: [&str; N]) -> Self {
        Self::new(value)
    }
}

impl GenerationParamValue for StopSequences {
    fn get(generation: &GenerationParams) -> Option<Self> {
        generation.stop_sequences.clone()
    }

    fn set(generation: &mut GenerationParams, value: Self) {
        generation.stop_sequences = Some(value);
    }
}

impl generation_param_seal::Sealed for StopSequences {}

#[derive(Builder, Clone, Debug, PartialEq)]
#[builder(builder_type(name = CompletionRequestBuilder))]
pub struct CompletionRequest {
    #[builder(into)]
    pub prompt: String,
    #[builder(default)]
    pub options: CompletionOptions,
    #[builder(default = RequestBudget::unlimited())]
    pub budget: RequestBudget,
}

impl CompletionRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
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
    pub top_p: Option<TopP>,
    pub frequency_penalty: Option<FrequencyPenalty>,
    pub presence_penalty: Option<PresencePenalty>,
    pub max_output_tokens: Option<MaxOutputTokens>,
    pub seed: Option<Seed>,
    pub stop_sequences: Option<StopSequences>,
}

impl CompletionOptions {
    pub fn resolved_with_extensions(mut self, extensions: &RequestExtensions) -> Self {
        self.resolve_with_extensions(extensions);
        self
    }

    pub fn resolve_with_extensions(&mut self, extensions: &RequestExtensions) {
        let mut generation = GenerationParams {
            temperature: self.temperature,
            top_p: self.top_p,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            max_output_tokens: self.max_output_tokens,
            seed: self.seed,
            stop_sequences: self.stop_sequences.clone(),
        }
        .resolved_with_extensions(extensions);
        self.temperature = generation.temperature.take();
        self.top_p = generation.top_p.take();
        self.frequency_penalty = generation.frequency_penalty.take();
        self.presence_penalty = generation.presence_penalty.take();
        self.max_output_tokens = generation.max_output_tokens.take();
        self.seed = generation.seed.take();
        self.stop_sequences = generation.stop_sequences.take();
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BackoffPolicy {
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub multiplier: f64,
    pub jitter_factor: f64,
}

impl Default for BackoffPolicy {
    fn default() -> Self {
        Self {
            initial_delay: Duration::from_millis(250),
            max_delay: Duration::from_secs(8),
            multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}

impl BackoffPolicy {
    pub fn delay_for_retry(&self, next_attempt: u32) -> Duration {
        let retry_index = next_attempt.saturating_sub(2) as i32;
        let base_secs = (self.initial_delay.as_secs_f64()
            * self.multiplier.max(1.0).powi(retry_index))
        .min(self.max_delay.as_secs_f64());
        let jitter = if self.jitter_factor <= 0.0 {
            0.0
        } else {
            let mut seed = next_attempt as u64;
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let normalized = (seed as f64 / u64::MAX as f64) * 2.0 - 1.0;
            normalized * self.jitter_factor
        };
        Duration::from_secs_f64(
            (base_secs * (1.0 + jitter)).clamp(0.0, self.max_delay.as_secs_f64()),
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff: BackoffPolicy,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 1,
            backoff: BackoffPolicy::default(),
        }
    }
}

impl RetryPolicy {
    pub const fn disabled() -> Self {
        Self {
            max_attempts: 1,
            backoff: BackoffPolicy {
                initial_delay: Duration::from_millis(250),
                max_delay: Duration::from_secs(8),
                multiplier: 2.0,
                jitter_factor: 0.1,
            },
        }
    }

    pub fn allows_retry(&self, current_attempt: u32, kind: RequestFailureKind) -> bool {
        kind == RequestFailureKind::Server && current_attempt < self.max_attempts
    }
}

#[derive(Clone, Debug)]
pub enum TextTurnEvent {
    Started {
        request_id: Option<String>,
        model: String,
    },
    WillRetry {
        attempt: u32,
        after: Duration,
        kind: RequestFailureKind,
        status: Option<u16>,
        request_id: Option<String>,
        accounted_usage: Usage,
        cumulative_usage: Usage,
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
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
        committed_turn: CommittedTurn,
    },
}

#[derive(Clone, Debug)]
pub enum TextTurnEventWithTools<T: Toolset> {
    Started {
        request_id: Option<String>,
        model: String,
    },
    WillRetry {
        attempt: u32,
        after: Duration,
        kind: RequestFailureKind,
        status: Option<u16>,
        request_id: Option<String>,
        accounted_usage: Usage,
        cumulative_usage: Usage,
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
    /// A tool call chunk that was rejected because the tool name is not in the available set.
    /// Emitted at stream-event level (Level 1 validation) before deserialization.
    InvalidToolCallChunk {
        id: crate::conversation::ToolCallId,
        name: crate::conversation::ToolName,
        arguments_json_delta: String,
    },
    /// A fully assembled tool call issue, emitted after assembly. Availability-policy failures
    /// and parse failures are both normalized into this recoverable issue path.
    ToolCallIssue(RecoverableToolCallIssue),
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
        committed_turn: CommittedTurn,
    },
}

#[derive(Clone, Debug)]
pub enum StructuredTurnEvent<O: StructuredOutput> {
    Started {
        request_id: Option<String>,
        model: String,
    },
    WillRetry {
        attempt: u32,
        after: Duration,
        kind: RequestFailureKind,
        status: Option<u16>,
        request_id: Option<String>,
        accounted_usage: Usage,
        cumulative_usage: Usage,
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
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
        committed_turn: CommittedTurn,
    },
}

#[derive(Clone, Debug)]
pub enum StructuredTurnEventWithTools<T: Toolset, O: StructuredOutput> {
    Started {
        request_id: Option<String>,
        model: String,
    },
    WillRetry {
        attempt: u32,
        after: Duration,
        kind: RequestFailureKind,
        status: Option<u16>,
        request_id: Option<String>,
        accounted_usage: Usage,
        cumulative_usage: Usage,
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
    /// A tool call chunk that was rejected because the tool name is not in the available set.
    /// Emitted at stream-event level (Level 1 validation) before deserialization.
    InvalidToolCallChunk {
        id: crate::conversation::ToolCallId,
        name: crate::conversation::ToolName,
        arguments_json_delta: String,
    },
    /// A fully assembled tool call issue, emitted after assembly. Availability-policy failures
    /// and parse failures are both normalized into this recoverable issue path.
    ToolCallIssue(RecoverableToolCallIssue),
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

impl PartialEq for TextTurnEvent {
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
                Self::WillRetry {
                    attempt: lhs_attempt,
                    after: lhs_after,
                    kind: lhs_kind,
                    status: lhs_status,
                    request_id: lhs_request_id,
                    accounted_usage: lhs_accounted_usage,
                    cumulative_usage: lhs_cumulative_usage,
                },
                Self::WillRetry {
                    attempt: rhs_attempt,
                    after: rhs_after,
                    kind: rhs_kind,
                    status: rhs_status,
                    request_id: rhs_request_id,
                    accounted_usage: rhs_accounted_usage,
                    cumulative_usage: rhs_cumulative_usage,
                },
            ) => {
                lhs_attempt == rhs_attempt
                    && lhs_after == rhs_after
                    && lhs_kind == rhs_kind
                    && lhs_status == rhs_status
                    && lhs_request_id == rhs_request_id
                    && lhs_accounted_usage == rhs_accounted_usage
                    && lhs_cumulative_usage == rhs_cumulative_usage
            }
            (Self::TextDelta { delta: lhs }, Self::TextDelta { delta: rhs }) => lhs == rhs,
            (Self::ReasoningDelta { delta: lhs }, Self::ReasoningDelta { delta: rhs }) => {
                lhs == rhs
            }
            (Self::RefusalDelta { delta: lhs }, Self::RefusalDelta { delta: rhs }) => lhs == rhs,
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

impl<T> PartialEq for TextTurnEventWithTools<T>
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
            (
                Self::WillRetry {
                    attempt: lhs_attempt,
                    after: lhs_after,
                    kind: lhs_kind,
                    status: lhs_status,
                    request_id: lhs_request_id,
                    accounted_usage: lhs_accounted_usage,
                    cumulative_usage: lhs_cumulative_usage,
                },
                Self::WillRetry {
                    attempt: rhs_attempt,
                    after: rhs_after,
                    kind: rhs_kind,
                    status: rhs_status,
                    request_id: rhs_request_id,
                    accounted_usage: rhs_accounted_usage,
                    cumulative_usage: rhs_cumulative_usage,
                },
            ) => {
                lhs_attempt == rhs_attempt
                    && lhs_after == rhs_after
                    && lhs_kind == rhs_kind
                    && lhs_status == rhs_status
                    && lhs_request_id == rhs_request_id
                    && lhs_accounted_usage == rhs_accounted_usage
                    && lhs_cumulative_usage == rhs_cumulative_usage
            }
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

impl<O> PartialEq for StructuredTurnEvent<O>
where
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
                Self::WillRetry {
                    attempt: lhs_attempt,
                    after: lhs_after,
                    kind: lhs_kind,
                    status: lhs_status,
                    request_id: lhs_request_id,
                    accounted_usage: lhs_accounted_usage,
                    cumulative_usage: lhs_cumulative_usage,
                },
                Self::WillRetry {
                    attempt: rhs_attempt,
                    after: rhs_after,
                    kind: rhs_kind,
                    status: rhs_status,
                    request_id: rhs_request_id,
                    accounted_usage: rhs_accounted_usage,
                    cumulative_usage: rhs_cumulative_usage,
                },
            ) => {
                lhs_attempt == rhs_attempt
                    && lhs_after == rhs_after
                    && lhs_kind == rhs_kind
                    && lhs_status == rhs_status
                    && lhs_request_id == rhs_request_id
                    && lhs_accounted_usage == rhs_accounted_usage
                    && lhs_cumulative_usage == rhs_cumulative_usage
            }
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

impl<T, O> PartialEq for StructuredTurnEventWithTools<T, O>
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
                Self::WillRetry {
                    attempt: lhs_attempt,
                    after: lhs_after,
                    kind: lhs_kind,
                    status: lhs_status,
                    request_id: lhs_request_id,
                    accounted_usage: lhs_accounted_usage,
                    cumulative_usage: lhs_cumulative_usage,
                },
                Self::WillRetry {
                    attempt: rhs_attempt,
                    after: rhs_after,
                    kind: rhs_kind,
                    status: rhs_status,
                    request_id: rhs_request_id,
                    accounted_usage: rhs_accounted_usage,
                    cumulative_usage: rhs_cumulative_usage,
                },
            ) => {
                lhs_attempt == rhs_attempt
                    && lhs_after == rhs_after
                    && lhs_kind == rhs_kind
                    && lhs_status == rhs_status
                    && lhs_request_id == rhs_request_id
                    && lhs_accounted_usage == rhs_accounted_usage
                    && lhs_cumulative_usage == rhs_cumulative_usage
            }
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
    WillRetry {
        attempt: u32,
        after: Duration,
        kind: RequestFailureKind,
        status: Option<u16>,
        request_id: Option<String>,
        accounted_usage: Usage,
        cumulative_usage: Usage,
    },
    TextDelta(String),
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum StructuredCompletionEvent<O: StructuredOutput> {
    Started {
        request_id: Option<String>,
        model: String,
    },
    WillRetry {
        attempt: u32,
        after: Duration,
        kind: RequestFailureKind,
        status: Option<u16>,
        request_id: Option<String>,
        accounted_usage: Usage,
        cumulative_usage: Usage,
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
    Completed {
        request_id: Option<String>,
        finish_reason: FinishReason,
        usage: Usage,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ErasedStructuredCompletionEvent {
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
    StructuredCompletion,
    Completion,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TokenCount {
    pub input_tokens: u64,
}

impl TokenCount {
    pub const fn new(input_tokens: u64) -> Self {
        Self { input_tokens }
    }
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
pub trait TokenCounter: crate::hooks::MaybeSend + crate::hooks::MaybeSync + 'static {
    async fn count_text_turn(
        &self,
        _input: &ModelInput,
        _turn: &AdapterTextTurn,
    ) -> Result<Option<TokenCount>, AgentError> {
        Ok(None)
    }

    async fn count_structured_turn(
        &self,
        _input: &ModelInput,
        _turn: &AdapterStructuredTurn,
    ) -> Result<Option<TokenCount>, AgentError> {
        Ok(None)
    }

    async fn count_completion(
        &self,
        _request: &CompletionRequest,
        _extensions: &RequestExtensions,
    ) -> Result<Option<TokenCount>, AgentError> {
        Ok(None)
    }

    async fn count_structured_completion(
        &self,
        _request: &AdapterStructuredCompletionRequest,
        _extensions: &RequestExtensions,
    ) -> Result<Option<TokenCount>, AgentError> {
        Ok(None)
    }
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
pub trait TurnAdapter: crate::hooks::MaybeSend + crate::hooks::MaybeSync + 'static {
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

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
pub trait CompletionAdapter: crate::hooks::MaybeSend + crate::hooks::MaybeSync + 'static {
    async fn completion(
        &self,
        request: CompletionRequest,
        extensions: &crate::extensions::RequestExtensions,
    ) -> Result<CompletionEventStream, AgentError>;

    async fn structured_completion(
        &self,
        request: AdapterStructuredCompletionRequest,
        extensions: &crate::extensions::RequestExtensions,
    ) -> Result<ErasedStructuredCompletionEventStream, AgentError>;
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
pub trait UsageRecoveryAdapter:
    crate::hooks::MaybeSend + crate::hooks::MaybeSync + 'static
{
    async fn recover_usage(
        &self,
        kind: OperationKind,
        request_id: &str,
    ) -> Result<Option<Usage>, AgentError>;
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
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

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
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

    async fn structured_completion(
        &self,
        request: AdapterStructuredCompletionRequest,
        extensions: &crate::extensions::RequestExtensions,
    ) -> Result<ErasedStructuredCompletionEventStream, AgentError> {
        (**self).structured_completion(request, extensions).await
    }
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
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

#[test]
#[cfg(not(target_family = "wasm"))]
fn test_stream_types_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TextTurnEventStream>();
    assert_send_sync::<TextTurnEventStreamWithTools<NoTools>>();
    assert_send_sync::<StructuredTurnEventStream<()>>();
    assert_send_sync::<StructuredTurnEventStreamWithTools<NoTools, ()>>();
    assert_send_sync::<StructuredCompletionEventStream<()>>();
    assert_send_sync::<CompletionEventStream>();
    assert_send_sync::<ErasedTextTurnEventStream>();
    assert_send_sync::<ErasedStructuredTurnEventStream>();
    assert_send_sync::<ErasedStructuredCompletionEventStream>();
    assert_send_sync::<TextTurnEvent>();
    assert_send_sync::<TextTurnEventWithTools<NoTools>>();
    assert_send_sync::<StructuredTurnEvent<()>>();
    assert_send_sync::<StructuredTurnEventWithTools<NoTools, ()>>();
    assert_send_sync::<StructuredCompletionEvent<()>>();
    assert_send_sync::<ErasedTextTurnEvent>();
    assert_send_sync::<ErasedStructuredTurnEvent>();
    assert_send_sync::<ErasedStructuredCompletionEvent>();
    assert_send_sync::<CompletionEvent>();
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    use super::*;
    use crate::extensions::RequestExtensions;
    use crate::transcript::AssistantTurnView;

    #[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
    struct Summary {
        answer: String,
    }

    #[test]
    fn text_completed_equality_checks_committed_turn_identity() {
        let shared_turn = Arc::new(AssistantTurnView::from_items(&[]));
        let different_turn = Arc::new(AssistantTurnView::from_items(&[]));

        let lhs = TextTurnEvent::Completed {
            request_id: Some("req-1".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
            committed_turn: shared_turn.clone(),
        };
        let same = TextTurnEvent::Completed {
            request_id: Some("req-1".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
            committed_turn: shared_turn,
        };
        let different = TextTurnEvent::Completed {
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

        let lhs = StructuredTurnEvent::<Summary>::Completed {
            request_id: Some("req-2".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
            committed_turn: shared_turn.clone(),
        };
        let same = StructuredTurnEvent::<Summary>::Completed {
            request_id: Some("req-2".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
            committed_turn: shared_turn,
        };
        let different = StructuredTurnEvent::<Summary>::Completed {
            request_id: Some("req-2".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage::zero(),
            committed_turn: different_turn,
        };

        assert_eq!(lhs, same);
        assert_ne!(lhs, different);
    }

    #[test]
    fn generation_params_resolve_each_field_from_extensions() {
        let mut extensions = RequestExtensions::new();
        extensions.insert(GenerationSetting::set(MaxOutputTokens::new(32)));
        extensions.insert(GenerationSetting::set(TopP::new(0.7).unwrap()));

        let resolved = GenerationParams::default().resolved_with_extensions(&extensions);

        assert_eq!(resolved.max_output_tokens, Some(MaxOutputTokens::new(32)));
        assert_eq!(resolved.top_p, Some(TopP::new(0.7).unwrap()));
    }

    #[test]
    fn explicit_generation_params_win_over_extension_defaults() {
        let mut extensions = RequestExtensions::new();
        extensions.insert(GenerationSetting::set(MaxOutputTokens::new(32)));

        let resolved = GenerationParams {
            max_output_tokens: Some(MaxOutputTokens::new(8)),
            ..GenerationParams::default()
        }
        .resolved_with_extensions(&extensions);

        assert_eq!(resolved.max_output_tokens, Some(MaxOutputTokens::new(8)));
    }

    #[test]
    fn generation_setting_clear_blocks_fallback_lookup() {
        let mut root = RequestExtensions::new();
        root.insert(GenerationSetting::set(MaxOutputTokens::new(32)));

        let mut request = RequestExtensions::new();
        request.insert(GenerationSetting::<MaxOutputTokens>::clear());
        request.push_fallback(Arc::new(root));

        let resolved = GenerationParams::default().resolved_with_extensions(&request);

        assert_eq!(resolved.max_output_tokens, None);
    }
}
