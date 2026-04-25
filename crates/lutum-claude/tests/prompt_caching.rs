use std::sync::Arc;

use lutum::{Lutum, RawTelemetryConfig, Session, SharedPoolBudgetManager, SharedPoolBudgetOptions};
use lutum_claude::ClaudeAdapter;
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
            } if provider == "claude" && api == "messages" && operation == "text_turn" => {
                Some(body)
            }
            _ => None,
        })
        .expect("raw trace should include the Claude messages request body");

    serde_json::from_str(body).expect("request body should be valid JSON")
}

#[tokio::test]
async fn session_ephemeral_marks_stable_claude_block_in_wire_request() {
    let adapter = ClaudeAdapter::new("test-key").with_base_url("http://127.0.0.1:9");
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

    let content = body["messages"][0]["content"]
        .as_array()
        .expect("Claude user message should have content blocks");
    assert_eq!(
        content[0]["cache_control"],
        serde_json::json!({ "type": "ephemeral" })
    );
    assert!(
        content[1].get("cache_control").is_none(),
        "ephemeral block itself must not be cache-marked"
    );
}
