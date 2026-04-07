use std::marker::PhantomData;

use async_trait::async_trait;
use lutum::{
    CollectError, GenerationParams, ModelInput, RequestBudget, RequestExtensions, StructuredOutput,
    StructuredTurnOutcome, StructuredTurnReductionError,
};
use thiserror::Error;

use crate::{Eval, TraceSnapshot};

type ExtensionFactory = dyn Fn(&mut RequestExtensions) + Send + Sync;

pub struct JudgeEval<A, R, F> {
    render_input: F,
    generation: GenerationParams,
    budget: RequestBudget,
    extension_factories: Vec<Box<ExtensionFactory>>,
    marker: PhantomData<fn(&A) -> R>,
}

impl<A, R, F> JudgeEval<A, R, F> {
    pub fn new(render_input: F) -> Self {
        Self {
            render_input,
            generation: GenerationParams::default(),
            budget: RequestBudget::unlimited(),
            extension_factories: Vec::new(),
            marker: PhantomData,
        }
    }

    pub fn ext<T>(mut self, extension: T) -> Self
    where
        T: Clone + Send + Sync + 'static,
    {
        self.extension_factories
            .push(Box::new(move |extensions: &mut RequestExtensions| {
                extensions.insert(extension.clone());
            }));
        self
    }

    pub fn extensions_with<G>(mut self, build: G) -> Self
    where
        G: Fn(&mut RequestExtensions) + Send + Sync + 'static,
    {
        self.extension_factories.push(Box::new(build));
        self
    }

    pub fn temperature(mut self, temperature: lutum::Temperature) -> Self {
        self.generation.temperature = Some(temperature);
        self
    }

    pub fn max_output_tokens(mut self, max_output_tokens: u32) -> Self {
        self.generation.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.generation.seed = Some(seed);
        self
    }

    pub fn budget(mut self, budget: RequestBudget) -> Self {
        self.budget = budget;
        self
    }

    pub fn generation_config(mut self, generation: GenerationParams) -> Self {
        self.generation = generation;
        self
    }

    fn build_extensions(&self) -> RequestExtensions {
        let mut extensions = RequestExtensions::new();
        for factory in &self.extension_factories {
            factory(&mut extensions);
        }
        extensions
    }
}

#[derive(Debug, Error)]
pub enum JudgeEvalError {
    #[error("judge execution error: {0}")]
    Execution(#[source] lutum::AgentError),
    #[error("judge reduction error: {0}")]
    Reduction(#[source] StructuredTurnReductionError),
    #[error("judge collection stopped before completion")]
    Stopped,
    #[error("judge stream ended before completion")]
    UnexpectedEof,
    #[error("judge refused to produce a structured report: {reason}")]
    Refusal { reason: String },
}

#[async_trait]
impl<A, R, F> Eval for JudgeEval<A, R, F>
where
    A: Sync,
    R: StructuredOutput + Send + Sync,
    F: Fn(&TraceSnapshot, &A) -> ModelInput + Send + Sync,
{
    type Artifact = A;
    type Report = R;
    type Error = JudgeEvalError;

    async fn evaluate(
        &self,
        ctx: &lutum::Lutum,
        trace: &TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        let input = (self.render_input)(trace, artifact);
        let result = ctx
            .structured_turn::<R>(input)
            .generation_config(self.generation.clone())
            .budget(self.budget)
            .extensions(self.build_extensions())
            .collect()
            .await
            .map_err(map_collect_error)?;

        match result.semantic {
            StructuredTurnOutcome::Structured(report) => Ok(report),
            StructuredTurnOutcome::Refusal(reason) => Err(JudgeEvalError::Refusal { reason }),
        }
    }
}

fn map_collect_error<O>(
    error: CollectError<
        core::convert::Infallible,
        StructuredTurnReductionError,
        lutum::StructuredTurnPartial<O>,
    >,
) -> JudgeEvalError
where
    O: StructuredOutput,
{
    match error {
        CollectError::Execution { source, .. } => JudgeEvalError::Execution(source),
        CollectError::Handler { source, .. } => match source {},
        CollectError::Reduction { source, .. } => JudgeEvalError::Reduction(source),
        CollectError::Stopped { .. } => JudgeEvalError::Stopped,
        CollectError::UnexpectedEof { .. } => JudgeEvalError::UnexpectedEof,
    }
}
