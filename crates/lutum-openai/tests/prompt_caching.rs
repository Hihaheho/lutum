use std::sync::Arc;

use lutum::{Lutum, RawTelemetryConfig, Session, SharedPoolBudgetManager, SharedPoolBudgetOptions};
use lutum_openai::{HttpClient, HttpError, HttpRequest, HttpResponse, OpenAiAdapter};
use lutum_trace::RawTraceEntry;
use serde_json::Value;

#[derive(Clone)]
struct FailingHttpClient;

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
impl HttpClient for FailingHttpClient {
    async fn send(&self, _request: HttpRequest) -> Result<HttpResponse, HttpError> {
        Err(HttpError::message("test client stops before network"))
    }
}

fn test_budget() -> SharedPoolBudgetManager {
    SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default())
}

fn request_only_raw_telemetry() -> RawTelemetryConfig {
    RawTelemetryConfig {
        request: true,
        stream: false,
        parse_errors: false,
        collect_errors: false,
    }
}

fn raw_request_body(entries: &[RawTraceEntry]) -> Value {
    let body = entries
        .iter()
        .find_map(|entry| match entry {
            RawTraceEntry::Request {
                provider,
                api,
                operation,
                body,
                ..
            } if provider == "openai" && api == "chat_completions" && operation == "text_turn" => {
                Some(body)
            }
            _ => None,
        })
        .expect("raw trace should include the OpenAI chat completions request body");

    serde_json::from_str(body).expect("request body should be valid JSON")
}

#[tokio::test]
async fn session_ephemeral_marks_stable_openai_chat_message_in_wire_request() {
    let adapter = OpenAiAdapter::new_with_http_client("test-key", FailingHttpClient)
        .with_chat_completions()
        .with_claude_prompt_caching();
    let ctx =
        Lutum::new(Arc::new(adapter), test_budget()).with_extension(request_only_raw_telemetry());
    let mut session = Session::new();

    session.push_user("Stable prompt.");
    session.push_ephemeral_user("Dynamic prompt.");

    let collected = lutum_trace::test::collect_raw(async move {
        let _ = session.text_turn().collect(&ctx).await;
    })
    .await;
    let body = raw_request_body(&collected.raw.entries);

    let messages = body["messages"]
        .as_array()
        .expect("chat request should have messages");
    assert_eq!(
        messages[0]["content"][0]["cache_control"],
        serde_json::json!({ "type": "ephemeral" })
    );
    assert!(
        !messages[1].to_string().contains("cache_control"),
        "ephemeral message itself must not be cache-marked"
    );
}
