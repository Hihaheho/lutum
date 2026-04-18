/// trace_dump — Ollama turn with lutum-trace dump
///
/// What gets captured:
///   - `llm_turn` spans emitted by the lutum core (one per API call)
///   - `lutum_hook` spans for any `LutumHooks` invocations
///   - User-defined spans that opt-in via `lutum.capture = true`
///
/// User spans below wrap each tool execution so the trace shows the full
/// round-trip: model turn → tool dispatch → next model turn.
///
/// Run:
///   cargo run -p lutum --example trace_dump
///   ENDPOINT=http://localhost:11434/v1 MODEL=gemma4:e2b cargo run -p lutum --example trace_dump
use std::{convert::Infallible, sync::Arc};

use lutum::*;
use lutum_openai::OpenAiAdapter;
use lutum_trace::{EventRecord, FieldValue, SpanNode, TraceSnapshot};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing_subscriber::layer::SubscriberExt as _;

// ---- tools -----------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct AddResult {
    result: i64,
}

#[lutum::tool_input(name = "add", output = AddResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct AddArgs {
    /// First operand.
    a: i64,
    /// Second operand.
    b: i64,
}

#[lutum::tool_fn]
/// Multiply two integers and return the product.
async fn multiply(a: i64, b: i64) -> Result<AddResult, Infallible> {
    Ok(AddResult { result: a * b })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, lutum::Toolset)]
enum MathTools {
    Add(AddArgs),
    Multiply(Multiply),
}

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
        println!("{pad}  · [{level}] {msg}", level = ev.level);
    } else {
        println!(
            "{pad}  · [{level}] {msg}  [{f}]",
            level = ev.level,
            f = fields.join(", ")
        );
    }
}

fn dump_span(span: &SpanNode, indent: usize) {
    let pad = "  ".repeat(indent);
    let fields: Vec<_> = span
        .fields
        .iter()
        .filter(|(k, _)| k != "lutum.capture")
        .map(|(k, v)| format!("{k}={}", format_field(v)))
        .collect();
    if fields.is_empty() {
        println!("{pad}▸ [{}] {}", span.level, span.name);
    } else {
        println!(
            "{pad}▸ [{}] {}  {{{}}}",
            span.level,
            span.name,
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
    if snapshot.roots.is_empty() && snapshot.root_events.is_empty() {
        println!("  (empty)");
        return;
    }
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
    // Install the capture layer.  No other layer is added — the capture layer
    // records all lutum:: spans plus any span that opts in with lutum.capture=true.
    let subscriber = tracing_subscriber::registry().with(lutum_trace::layer());
    tracing::subscriber::set_global_default(subscriber)
        .expect("global subscriber should install once");

    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e2b".into());

    let model = ModelName::new(&model_name)?;
    let adapter = OpenAiAdapter::new(token)
        .with_base_url(endpoint)
        .with_default_model(model);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let llm = Lutum::new(Arc::new(adapter), budget);

    let mut session = Session::new(llm);
    session.push_system(
        "You are a calculator assistant. \
         Use the provided tools to compute results. \
         Do not guess — always call a tool.",
    );
    session.push_user("What is (6 + 7) * 3?");

    // lutum_trace::capture wraps the entire agent loop.
    // All lutum:: spans are automatically recorded.
    // User-defined spans that carry `lutum.capture = true` are also recorded.
    let collected = lutum_trace::capture(async move {
        let mut last_text = String::new();

        for _step in 1..=6 {
            let outcome = session.text_turn().tools::<MathTools>().collect().await?;

            match outcome {
                TextStepOutcomeWithTools::NeedsTools(round) => {
                    let mut tool_results = Vec::with_capacity(round.tool_calls.len());

                    for tool_call in round.tool_calls.iter().cloned() {
                        match tool_call {
                            MathToolsCall::Add(call) => {
                                let args = call.input();
                                // User-defined span — opts in to capture via the field.
                                let _span = tracing::info_span!(
                                    "tool_exec",
                                    lutum.capture = true,
                                    tool = "add",
                                    a = args.a,
                                    b = args.b,
                                )
                                .entered();
                                let result = AddResult { result: args.a + args.b };
                                tracing::info!(target: "lutum", value = result.result, "add_result");
                                tool_results.push(call.complete(result).unwrap());
                            }
                            MathToolsCall::Multiply(call) => {
                                let args = call.input();
                                let _span = tracing::info_span!(
                                    "tool_exec",
                                    lutum.capture = true,
                                    tool = "multiply",
                                    a = args.a,
                                    b = args.b,
                                )
                                .entered();
                                let result = multiply(args.a, args.b).await.unwrap();
                                tracing::info!(target: "lutum", value = result.result, "multiply_result");
                                tool_results.push(call.complete(result).unwrap());
                            }
                        }
                    }

                    round.commit(&mut session, tool_results).unwrap();
                }
                TextStepOutcomeWithTools::Finished(result) => {
                    last_text = result.assistant_text();
                    break;
                }
            }
        }

        anyhow::Ok(last_text)
    })
    .await;

    let answer = collected.output?;
    println!("=== Answer ===");
    println!("{}", answer.trim());

    println!("\n=== Trace ===");
    dump_trace(&collected.trace);

    Ok(())
}
