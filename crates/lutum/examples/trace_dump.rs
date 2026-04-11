use lutum::*;
use lutum_openai::OpenAiAdapter;
use lutum_trace::{EventRecord, FieldValue, SpanNode, TraceSnapshot};
use std::sync::Arc;
use tracing_subscriber::layer::SubscriberExt as _;

// ---- pretty-print helpers --------------------------------------------------

fn format_field(v: &FieldValue) -> String {
    match v {
        FieldValue::Bool(b) => b.to_string(),
        FieldValue::I64(n) => n.to_string(),
        FieldValue::U64(n) => n.to_string(),
        FieldValue::I128(n) => n.to_string(),
        FieldValue::U128(n) => n.to_string(),
        FieldValue::F64(f) => format!("{f:.3}"),
        FieldValue::Str(s) => format!("{s:?}"),
    }
}

fn dump_event(ev: &EventRecord, indent: usize) {
    let pad = "  ".repeat(indent);
    let msg = ev.message().unwrap_or("<no message>");
    let fields: Vec<_> = ev
        .fields
        .iter()
        .filter(|(k, _)| k != "message")
        .map(|(k, v)| format!("{k}={}", format_field(v)))
        .collect();
    if fields.is_empty() {
        println!("{pad}  · {msg}");
    } else {
        println!("{pad}  · {msg}  [{}]", fields.join(", "));
    }
}

fn dump_span(span: &SpanNode, indent: usize) {
    let pad = "  ".repeat(indent);
    let fields: Vec<_> = span
        .fields
        .iter()
        .map(|(k, v)| format!("{k}={}", format_field(v)))
        .collect();
    if fields.is_empty() {
        println!("{pad}▸ {} ({}::{})", span.name, span.target, span.level);
    } else {
        println!(
            "{pad}▸ {} ({}::{})  {{{}}}",
            span.name,
            span.target,
            span.level,
            fields.join(", ")
        );
    }
    for ev in span.events() {
        dump_event(ev, indent);
    }
    for child in span.children() {
        dump_span(child, indent + 1);
    }
}

fn dump_trace(snapshot: &TraceSnapshot) {
    for ev in snapshot.events() {
        dump_event(ev, 0);
    }
    for root in &snapshot.roots {
        dump_span(root, 0);
    }
}

// ---- main ------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Install capture layer as the global subscriber.
    let subscriber = tracing_subscriber::registry().with(lutum_trace::layer());
    tracing::subscriber::set_global_default(subscriber)
        .expect("global subscriber should install once");

    let endpoint =
        std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e2b".into());

    let model = ModelName::new(&model_name)?;
    let adapter = OpenAiAdapter::new(token)
        .with_base_url(endpoint)
        .with_default_model(model);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let llm = Lutum::new(Arc::new(adapter), budget);

    let mut session = Session::new(llm);
    session.push_user("Say hello in one sentence.");

    // Wrap execution so lutum-trace captures all spans emitted during the turn.
    let collected = lutum_trace::capture(async move {
        session.text_turn().collect().await
    })
    .await;

    let result = collected.output?;

    println!("=== Response ===");
    println!("{}", result.assistant_text().trim());
    println!(
        "model: {}  tokens: {}in / {}out",
        result.model, result.usage.input_tokens, result.usage.output_tokens,
    );

    println!("\n=== Trace ===");
    dump_trace(&collected.trace);

    Ok(())
}
