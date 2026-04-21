/// eval_raw_debug_dump — run a turn under `capture_raw`, then feed the normal
/// trace into `lutum-eval`, while writing raw events to a file only on error.
///
/// Why this is structured manually:
///   - `lutum-eval`'s `run_future()` helpers currently use `lutum_trace::capture()`
///   - if you want the raw timeline as well, wrap the future yourself with
///     `lutum_trace::capture_raw()` and then pass the normal trace/output into
///     `lutum-eval`
///
/// Run:
///   cargo run -p lutum --example eval_raw_debug_dump
///   ENDPOINT=http://localhost:11434/v1 MODEL=gemma4:e2b cargo run -p lutum --example eval_raw_debug_dump
///
/// Useful env vars:
///   - `RAW_DEBUG_PATH`: file path for raw event dumps
///     default: `/tmp/lutum-eval-raw-debug.log`
///   - `REQUIRE_SUBSTRING`: evaluator requirement; set this to a nonsense string
///     to force an eval failure and verify the dump path
use std::{
    fmt::Write as _,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Context as _, bail};
use lutum::*;
use lutum_eval::{Collected, PureEval, PureEvalExt, TraceSnapshot};
use lutum_openai::OpenAiAdapter;
use lutum_trace::RawTraceSnapshot;
use tracing_subscriber::layer::SubscriberExt as _;

#[derive(Debug)]
struct EvalReport {
    saw_llm_turn: bool,
    answer_len: usize,
}

struct RequiresSubstring {
    required: String,
}

impl PureEval for RequiresSubstring {
    type Artifact = String;
    type Report = EvalReport;
    type Error = anyhow::Error;

    fn evaluate(
        &self,
        trace: &TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        if !artifact.contains(&self.required) {
            bail!(
                "answer did not contain required substring {:?}: {:?}",
                self.required,
                artifact
            );
        }

        Ok(EvalReport {
            saw_llm_turn: trace.span("llm_turn").is_some(),
            answer_len: artifact.chars().count(),
        })
    }
}

fn write_raw_debug_dump(path: &Path, raw: &RawTraceSnapshot) -> anyhow::Result<()> {
    let mut out = String::new();

    if raw.entries.is_empty() {
        out.push_str("(no raw entries captured)\n");
    } else {
        for (index, entry) in raw.entries.iter().enumerate() {
            writeln!(&mut out, "## entry[{index}]").unwrap();
            writeln!(&mut out, "{entry:#?}").unwrap();
            out.push('\n');
        }
    }

    std::fs::write(path, out)
        .with_context(|| format!("failed to write raw debug dump to {}", path.display()))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let subscriber = tracing_subscriber::registry().with(lutum_trace::layer());
    tracing::subscriber::set_global_default(subscriber)
        .expect("global subscriber should install once");

    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e2b".into());
    let raw_debug_path = PathBuf::from(
        std::env::var("RAW_DEBUG_PATH")
            .unwrap_or_else(|_| "/tmp/lutum-eval-raw-debug.log".to_string()),
    );
    let required_substring =
        std::env::var("REQUIRE_SUBSTRING").unwrap_or_else(|_| "ownership".to_string());

    let model = ModelName::new(&model_name)?;
    let adapter = OpenAiAdapter::new(token)
        .with_base_url(endpoint)
        .with_default_model(model);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let llm = Lutum::new(Arc::new(adapter), budget).with_extension(RawTelemetryConfig::all());

    let collected = lutum_trace::capture_raw(async {
        let mut session = Session::new(llm.clone());
        session.push_system("You are a concise Rust explainer. Answer in one short paragraph.");
        session.push_user("Explain Rust ownership in plain language.");

        let result = session.text_turn().collect().await?;

        anyhow::Ok::<String>(result.assistant_text())
    })
    .await;

    let lutum_trace::CollectedRaw { output, trace, raw } = collected;

    let answer = match output {
        Ok(answer) => answer,
        Err(err) => {
            write_raw_debug_dump(&raw_debug_path, &raw)?;
            return Err(err.context(format!(
                "generation failed; raw events written to {}",
                raw_debug_path.display()
            )));
        }
    };

    let eval_input = Collected {
        output: answer.clone(),
        trace,
    };

    let report = match (RequiresSubstring {
        required: required_substring.clone(),
    })
    .run_collected(&eval_input)
    {
        Ok(report) => report,
        Err(err) => {
            write_raw_debug_dump(&raw_debug_path, &raw)?;
            return Err(err.context(format!(
                "evaluation failed; raw events written to {}",
                raw_debug_path.display()
            )));
        }
    };

    println!("=== Answer ===");
    println!("{}", answer.trim());

    println!("\n=== Eval Report ===");
    println!("required_substring: {:?}", required_substring);
    println!("saw_llm_turn: {}", report.saw_llm_turn);
    println!("answer_len: {}", report.answer_len);
    println!("\nNo raw debug file written.");

    Ok(())
}
