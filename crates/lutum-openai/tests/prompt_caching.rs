use std::sync::Arc;

use lutum::{Lutum, RawTelemetryConfig, Session, SharedPoolBudgetManager, SharedPoolBudgetOptions};
use lutum_openai::OpenAiAdapter;
use lutum_trace::RawTraceEntry;
use serde_json::Value;

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
    let adapter = OpenAiAdapter::new("test-key")
        .with_base_url("http://127.0.0.1:9")
        .with_chat_completions()
        .with_claude_prompt_caching();
    let ctx =
        Lutum::new(Arc::new(adapter), test_budget()).with_extension(request_only_raw_telemetry());
    let mut session = Session::new(ctx);

    session.push_user("Stable prompt.");
    session.push_ephemeral_user("Dynamic prompt.");

    let collected = lutum_trace::test::collect_raw(async move {
        let _ = session.text_turn().collect().await;
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
        messages[1].to_string().find("cache_control").is_none(),
        "ephemeral message itself must not be cache-marked"
    );
}
