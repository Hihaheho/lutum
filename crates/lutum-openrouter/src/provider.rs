use lutum_openai::BodyCustomizer;
use lutum_protocol::RequestExtensions;
use serde_json::Value;

/// Typed OpenRouter routing preferences applied to the final request JSON.
///
/// Existing provider fields are preserved; configured preferences overwrite the
/// corresponding fields. An absent preference leaves that field untouched.
/// Installing this replaces any previous body customizer on the adapter. To
/// compose customizations, call this customizer last inside your own closure.
///
/// ```no_run
/// use lutum_openai::OpenAiAdapter;
/// use lutum_openrouter::{OPENAI_BASE_URL, ProviderPreferences};
///
/// # #[cfg(feature = "reqwest")]
/// let adapter = OpenAiAdapter::new("key")
///     .with_base_url(OPENAI_BASE_URL)
///     .with_chat_completions()
///     .with_body_customizer(ProviderPreferences::default().with_zdr(true));
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct ProviderPreferences {
    /// Restrict routing to zero data retention endpoints when set to `true`.
    pub zdr: Option<bool>,
}

impl ProviderPreferences {
    pub fn with_zdr(mut self, enabled: bool) -> Self {
        self.zdr = Some(enabled);
        self
    }
}

impl BodyCustomizer for ProviderPreferences {
    fn customize(&self, _extensions: &RequestExtensions, body: &mut Value) {
        if let Some(zdr) = self.zdr {
            // Adapter requests are objects. Replace malformed provider values so
            // an explicit ZDR preference cannot be silently omitted.
            let provider = &mut body["provider"];
            if !provider.is_object() {
                *provider = serde_json::json!({});
            }
            provider["zdr"] = Value::Bool(zdr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct CaptureClient(std::sync::Arc<std::sync::Mutex<Vec<lutum_openai::HttpRequest>>>);

    #[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
    impl lutum_openai::HttpClient for CaptureClient {
        async fn send(
            &self,
            request: lutum_openai::HttpRequest,
        ) -> Result<lutum_openai::HttpResponse, lutum_openai::HttpError> {
            self.0.lock().unwrap().push(request);
            Ok(lutum_openai::HttpResponse {
                status: http::StatusCode::OK,
                headers: http::HeaderMap::new(),
                body: Box::pin(futures::stream::empty()),
            })
        }
    }

    #[tokio::test]
    async fn zdr_is_present_in_transmitted_chat_and_responses_requests() {
        use lutum_protocol::{
            AdapterTextTurn, AdapterToolChoice, AdapterTurnConfig, GenerationParams,
            InputMessageRole, ModelInput, ModelInputItem, TurnAdapter,
        };

        for chat in [false, true] {
            let client = CaptureClient::default();
            let mut adapter =
                lutum_openai::OpenAiAdapter::new_with_http_client("key", client.clone())
                    .with_base_url(crate::OPENAI_BASE_URL)
                    .with_body_customizer(ProviderPreferences::default().with_zdr(true));
            if chat {
                adapter = adapter.with_chat_completions();
            }
            let input =
                ModelInput::from_items(vec![ModelInputItem::text(InputMessageRole::User, "hello")]);
            let turn = AdapterTextTurn {
                config: AdapterTurnConfig {
                    generation: GenerationParams::default(),
                    tools: Vec::new(),
                    tool_choice: AdapterToolChoice::None,
                },
                extensions: std::sync::Arc::new(RequestExtensions::new()),
            };
            drop(adapter.text_turn(input, turn).await.unwrap());
            let requests = client.0.lock().unwrap();
            assert_eq!(requests.len(), 1);
            assert_eq!(
                requests[0].url,
                format!(
                    "{}{}",
                    crate::OPENAI_BASE_URL,
                    if chat {
                        "/chat/completions"
                    } else {
                        "/responses"
                    }
                )
            );
            let body: Value = serde_json::from_slice(requests[0].body.as_deref().unwrap()).unwrap();
            assert_eq!(body["provider"]["zdr"], true);
            assert_eq!(body["stream"], true);
            assert!(body.get(if chat { "messages" } else { "input" }).is_some());
        }
    }

    #[test]
    fn preferences_preserve_other_fields_and_override_zdr() {
        for provider in [
            serde_json::json!({"only": ["example"], "zdr": false}),
            Value::Null,
            Value::Bool(false),
        ] {
            let mut body = serde_json::json!({"model": "test", "provider": provider});
            ProviderPreferences::default()
                .with_zdr(true)
                .customize(&RequestExtensions::new(), &mut body);
            assert_eq!(body["provider"]["zdr"], true);
            assert_eq!(body["model"], "test");
            if provider.is_object() {
                assert_eq!(body["provider"]["only"], provider["only"]);
            }
        }
    }

    #[test]
    fn absent_preference_is_noop_and_false_is_explicit() {
        let mut body = serde_json::json!({"model": "test"});
        let original = body.clone();
        ProviderPreferences::default().customize(&RequestExtensions::new(), &mut body);
        assert_eq!(body, original);
        ProviderPreferences::default()
            .with_zdr(false)
            .customize(&RequestExtensions::new(), &mut body);
        assert_eq!(body["provider"]["zdr"], false);
    }
}
