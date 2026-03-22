// OpenRouter adapter - provides UsageRecoveryAdapter via the generations endpoint.
// Use with OpenAiAdapter or ClaudeAdapter pointing base_url at OpenRouter.

use std::sync::Arc;

use agents_protocol::{AgentError, OperationKind, UsageRecoveryAdapter, budget::Usage};

pub const OPENAI_BASE_URL: &str = "https://openrouter.ai/api/v1";
pub const ANTHROPIC_BASE_URL: &str = "https://openrouter.ai/api";

// Raw DTO from GET /api/v1/generation?id={id}
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Generation {
    pub id: String,
    pub model: Option<String>,
    pub total_cost: Option<f64>,
    pub tokens_prompt: Option<u64>,
    pub tokens_completion: Option<u64>,
    pub native_tokens_prompt: Option<u64>,
    pub native_tokens_completion: Option<u64>,
    pub native_tokens_reasoning: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub provider_name: Option<String>,
    pub finish_reason: Option<String>,
}

impl Generation {
    pub fn to_usage(&self) -> Usage {
        let input = self.tokens_prompt.unwrap_or(0);
        let output = self.tokens_completion.unwrap_or(0);
        let cost_micros = self
            .total_cost
            .map(|cost| (cost * 1_000_000.0) as u64)
            .unwrap_or(0);
        Usage {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
            cost_micros_usd: cost_micros,
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct GenerationResponse {
    data: Generation,
}

#[derive(Debug, thiserror::Error)]
pub enum OpenRouterError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("missing OPENROUTER_API_KEY env var")]
    MissingApiKey,
}

pub struct OpenRouterGenerationClient {
    client: Arc<reqwest::Client>,
    api_key: Arc<str>,
    base_url: Arc<str>,
}

impl OpenRouterGenerationClient {
    pub fn from_env() -> Result<Self, OpenRouterError> {
        let key =
            std::env::var("OPENROUTER_API_KEY").map_err(|_| OpenRouterError::MissingApiKey)?;
        Ok(Self::new(key))
    }

    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: Arc::new(reqwest::Client::new()),
            api_key: api_key.into().into(),
            base_url: OPENAI_BASE_URL.into(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().into();
        self
    }

    pub async fn get_generation(&self, id: &str) -> Result<Generation, OpenRouterError> {
        let url = format!("{}/generation?id={id}", self.base_url);
        let resp = self
            .client
            .get(&url)
            .bearer_auth(self.api_key.as_ref())
            .send()
            .await?
            .error_for_status()?
            .json::<GenerationResponse>()
            .await?;
        Ok(resp.data)
    }
}

#[async_trait::async_trait]
impl UsageRecoveryAdapter for OpenRouterGenerationClient {
    async fn recover_usage(
        &self,
        kind: OperationKind,
        request_id: &str,
    ) -> Result<Option<Usage>, AgentError> {
        if kind == OperationKind::Completion {
            return Ok(None);
        }

        match self.get_generation(request_id).await {
            Ok(generation) => Ok(Some(generation.to_usage())),
            Err(err) => {
                tracing::warn!(request_id, error = %err, "OpenRouter generation lookup failed");
                Ok(None)
            }
        }
    }
}
