// OpenRouter adapter - provides UsageRecoveryAdapter via the generations endpoint.
// Use with OpenAiAdapter or ClaudeAdapter pointing base_url at OpenRouter.

use std::{sync::Arc, time::Duration};

pub mod transport;

#[cfg(feature = "reqwest")]
pub use transport::ReqwestHttpClient;
pub use transport::{HttpByteStream, HttpClient, HttpError, HttpRequest, HttpResponse};

use futures::StreamExt;
use http::header::AUTHORIZATION;
use http::{HeaderMap, HeaderValue, Method, StatusCode};
use lutum_protocol::{
    AgentError, OperationKind, UsageRecoveryAdapter, budget::Usage, extensions::RequestExtensions,
};

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
    pub native_tokens_cached: Option<u64>,
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
            cache_read_tokens: self
                .native_tokens_cached
                .or(self.cached_tokens)
                .unwrap_or_default(),
            ..Usage::zero()
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
    Http(#[from] HttpError),
    #[error("request failed with status {status}: {message}")]
    HttpStatus {
        status: StatusCode,
        message: String,
        retry_after: Option<Duration>,
    },
    #[error("missing OPENROUTER_API_KEY env var")]
    MissingApiKey,
    #[error("invalid header value: {0}")]
    InvalidHeader(#[from] http::header::InvalidHeaderValue),
    #[error("failed to encode or decode JSON: {0}")]
    Json(#[from] serde_json::Error),
}

pub trait HeadersCustomizer: Send + Sync {
    fn customize(&self, extensions: &RequestExtensions, headers: &mut HeaderMap);
}

impl<F> HeadersCustomizer for F
where
    F: Fn(&RequestExtensions, &mut HeaderMap) + Send + Sync,
{
    fn customize(&self, extensions: &RequestExtensions, headers: &mut HeaderMap) {
        self(extensions, headers)
    }
}

pub struct OpenRouterGenerationClient {
    client: Arc<dyn HttpClient>,
    api_key: Arc<str>,
    base_url: Arc<str>,
    headers_customizer: Option<Arc<dyn HeadersCustomizer>>,
}

impl OpenRouterGenerationClient {
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Self, OpenRouterError> {
        let key =
            std::env::var("OPENROUTER_API_KEY").map_err(|_| OpenRouterError::MissingApiKey)?;
        Ok(Self::new(key))
    }

    pub fn from_env_with_http_client(
        client: impl HttpClient + 'static,
    ) -> Result<Self, OpenRouterError> {
        let key =
            std::env::var("OPENROUTER_API_KEY").map_err(|_| OpenRouterError::MissingApiKey)?;
        Ok(Self::new_with_http_client(key, client))
    }

    #[cfg(feature = "reqwest")]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::new_with_http_client(api_key, ReqwestHttpClient::new())
    }

    pub fn new_with_http_client(
        api_key: impl Into<String>,
        client: impl HttpClient + 'static,
    ) -> Self {
        Self {
            client: Arc::new(client),
            api_key: api_key.into().into(),
            base_url: OPENAI_BASE_URL.into(),
            headers_customizer: None,
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().into();
        self
    }

    pub fn with_headers_customizer(mut self, customizer: impl HeadersCustomizer + 'static) -> Self {
        self.headers_customizer = Some(Arc::new(customizer));
        self
    }

    fn request_headers(
        &self,
        extensions: &RequestExtensions,
    ) -> Result<HeaderMap, OpenRouterError> {
        let mut headers = HeaderMap::new();
        let bearer = format!("Bearer {}", self.api_key);
        headers.insert(AUTHORIZATION, HeaderValue::from_str(&bearer)?);
        if let Some(customizer) = self.headers_customizer.as_ref() {
            customizer.customize(extensions, &mut headers);
        }
        Ok(headers)
    }

    pub async fn get_generation(&self, id: &str) -> Result<Generation, OpenRouterError> {
        let url = format!("{}/generation?id={id}", self.base_url);
        let headers = self.request_headers(&RequestExtensions::new())?;
        let response = self
            .client
            .send(HttpRequest {
                method: Method::GET,
                url,
                headers,
                body: None,
            })
            .await?;
        let response = error_for_status_with_body(response).await?;
        let body = collect_response_body(response).await?;
        Ok(serde_json::from_slice::<GenerationResponse>(&body)?.data)
    }
}

fn retry_after_from_headers(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(http::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

async fn error_for_status_with_body(
    response: HttpResponse,
) -> Result<HttpResponse, OpenRouterError> {
    let status = response.status;
    if status.is_success() {
        return Ok(response);
    }

    let retry_after = retry_after_from_headers(&response.headers);
    let body = collect_response_body(response).await?;
    let body = String::from_utf8_lossy(&body).into_owned();
    let message = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .and_then(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or(body);

    Err(OpenRouterError::HttpStatus {
        status,
        message,
        retry_after,
    })
}

async fn collect_response_body(mut response: HttpResponse) -> Result<Vec<u8>, OpenRouterError> {
    let mut body = Vec::new();
    while let Some(chunk) = response.body.next().await {
        body.extend_from_slice(&chunk?);
    }
    Ok(body)
}

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use bytes::Bytes;
    use futures::executor::block_on;

    use super::*;

    #[derive(Debug)]
    struct CapturedHttpRequest {
        method: Method,
        url: String,
        headers: HeaderMap,
        body: Option<Vec<u8>>,
    }

    #[derive(Clone)]
    struct FakeHttpClient {
        captured: Arc<Mutex<Vec<CapturedHttpRequest>>>,
        response: Arc<Mutex<Option<Result<HttpResponse, HttpError>>>>,
    }

    impl FakeHttpClient {
        fn new(response: HttpResponse) -> Self {
            Self {
                captured: Arc::new(Mutex::new(Vec::new())),
                response: Arc::new(Mutex::new(Some(Ok(response)))),
            }
        }

        fn captured(&self) -> Vec<CapturedHttpRequest> {
            std::mem::take(&mut *self.captured.lock().unwrap())
        }
    }

    #[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
    impl HttpClient for FakeHttpClient {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
            self.captured.lock().unwrap().push(CapturedHttpRequest {
                method: request.method,
                url: request.url,
                headers: request.headers,
                body: request.body,
            });
            self.response.lock().unwrap().take().expect("fake response")
        }
    }

    fn fake_response(
        status: StatusCode,
        headers: HeaderMap,
        chunks: Vec<Result<Bytes, HttpError>>,
    ) -> HttpResponse {
        HttpResponse {
            status,
            headers,
            body: Box::pin(futures::stream::iter(chunks)),
        }
    }

    fn test_openrouter_client(api_key: &str) -> OpenRouterGenerationClient {
        OpenRouterGenerationClient::new_with_http_client(
            api_key,
            FakeHttpClient::new(fake_response(StatusCode::OK, HeaderMap::new(), Vec::new())),
        )
    }

    fn generation() -> Generation {
        Generation {
            id: "gen-1".to_string(),
            model: Some("openai/gpt-4.1".to_string()),
            total_cost: Some(0.0015),
            tokens_prompt: Some(10),
            tokens_completion: Some(25),
            native_tokens_prompt: Some(10),
            native_tokens_completion: Some(25),
            native_tokens_reasoning: Some(5),
            native_tokens_cached: Some(3),
            cached_tokens: Some(2),
            provider_name: Some("OpenAI".to_string()),
            finish_reason: Some("stop".to_string()),
        }
    }

    #[test]
    fn generation_usage_maps_native_cached_tokens() {
        let usage = generation().to_usage();

        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 25);
        assert_eq!(usage.total_tokens, 35);
        assert_eq!(usage.cost_micros_usd, 1_500);
        assert_eq!(usage.cache_read_tokens, 3);
        assert_eq!(usage.cache_creation_tokens, 0);
    }

    #[test]
    fn generation_usage_keeps_cached_tokens_fallback() {
        let mut generation = generation();
        generation.native_tokens_cached = None;

        let usage = generation.to_usage();

        assert_eq!(usage.cache_read_tokens, 2);
    }

    #[test]
    fn injected_http_client_fetches_generation_json_and_records_request() {
        let fake = FakeHttpClient::new(fake_response(
            StatusCode::OK,
            HeaderMap::new(),
            vec![Ok(Bytes::from_static(
                br#"{"data":{"id":"gen-1","model":"openai/gpt-4.1","total_cost":0.0015,"tokens_prompt":10,"tokens_completion":25,"native_tokens_prompt":10,"native_tokens_completion":25,"native_tokens_reasoning":5,"native_tokens_cached":3,"cached_tokens":2,"provider_name":"OpenAI","finish_reason":"stop"}}"#,
            ))],
        ));
        let client = OpenRouterGenerationClient::new_with_http_client("test-key", fake.clone());

        let generation = block_on(client.get_generation("gen-1")).expect("generation");
        assert_eq!(generation.id, "gen-1");
        assert_eq!(generation.to_usage().total_tokens, 35);

        let captured = fake.captured();
        assert_eq!(captured.len(), 1);
        let request = &captured[0];
        assert_eq!(request.method, Method::GET);
        assert_eq!(
            request.url,
            "https://openrouter.ai/api/v1/generation?id=gen-1"
        );
        assert_eq!(
            request.headers.get(AUTHORIZATION).unwrap(),
            "Bearer test-key"
        );
        assert!(request.body.is_none());
    }

    #[test]
    fn injected_http_client_status_error_preserves_retry_after() {
        let mut headers = HeaderMap::new();
        headers.insert(http::header::RETRY_AFTER, HeaderValue::from_static("7"));
        let fake = FakeHttpClient::new(fake_response(
            StatusCode::TOO_MANY_REQUESTS,
            headers,
            vec![Ok(Bytes::from_static(
                br#"{"error":{"message":"slow down"}}"#,
            ))],
        ));
        let client = OpenRouterGenerationClient::new_with_http_client("test-key", fake);

        let error = block_on(client.get_generation("gen-1")).expect_err("status error");
        match error {
            OpenRouterError::HttpStatus {
                status,
                message,
                retry_after,
            } => {
                assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
                assert_eq!(message, "slow down");
                assert_eq!(retry_after, Some(Duration::from_secs(7)));
            }
            other => panic!("expected status error, got {other:?}"),
        }
    }

    #[test]
    fn headers_customizer_appends_without_touching_defaults() {
        use http::HeaderName;

        let client = test_openrouter_client("test-key").with_headers_customizer(
            |_ext: &RequestExtensions, headers: &mut HeaderMap| {
                headers.insert(
                    HeaderName::from_static("http-referer"),
                    HeaderValue::from_static("https://example.com"),
                );
            },
        );

        let headers = client
            .request_headers(&RequestExtensions::new())
            .expect("headers");

        assert_eq!(headers.get("http-referer").unwrap(), "https://example.com");
        assert_eq!(headers.get(AUTHORIZATION).unwrap(), "Bearer test-key");
    }

    #[test]
    fn headers_customizer_can_override_defaults() {
        let client = test_openrouter_client("test-key").with_headers_customizer(
            |_ext: &RequestExtensions, headers: &mut HeaderMap| {
                headers.insert(AUTHORIZATION, HeaderValue::from_static("Bearer override"));
            },
        );

        let headers = client
            .request_headers(&RequestExtensions::new())
            .expect("headers");

        assert_eq!(headers.get(AUTHORIZATION).unwrap(), "Bearer override");
    }
}
