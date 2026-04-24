//! Completion example using the high-level Lutum API with collect_with.
//!
//! Defaults to a local Ollama instance (requires a base model, not chat/thinking):
//!   ollama pull gemma4:e2b
//!   cargo run --example completion -p lutum
//!
//! Override via environment variables:
//!   ENDPOINT=https://api.openai.com/v1 TOKEN=<key> MODEL=gpt-3.5-turbo-instruct \
//!     cargo run --example completion -p lutum

use std::convert::Infallible;
use std::io::Write;
use std::sync::Arc;

use console::style;
use lutum::*;
use lutum_openai::OpenAiAdapter;

const PROMPTS: &[&str] = &[
    "The capital of France is",
    "In Rust, ownership means",
    "The largest planet in our solar system is",
];

struct PrintDelta;

#[cfg_attr(target_family = "wasm", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait::async_trait)]
impl EventHandler<CompletionEvent, CompletionTurnState> for PrintDelta {
    type Error = Infallible;

    async fn on_event(
        &mut self,
        event: &CompletionEvent,
        _cx: &HandlerContext<CompletionTurnState>,
    ) -> Result<HandlerDirective, Infallible> {
        if let CompletionEvent::TextDelta(delta) = event {
            print!("{delta}");
            let _ = std::io::stdout().flush();
        }
        Ok(HandlerDirective::Continue)
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model = std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e2b".into());

    let adapter = Arc::new(
        OpenAiAdapter::new(token)
            .with_base_url(&endpoint)
            .with_default_model(ModelName::new(&model)?),
    );
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let llm = Lutum::from_parts(adapter.clone(), adapter.clone(), budget);

    println!(
        "\n{}  model={}  endpoint={}",
        style("Lutum completion — collect_with").bold(),
        style(&model).cyan(),
        style(&endpoint).dim(),
    );
    println!("{}", style("─".repeat(60)).dim());

    for prompt in PROMPTS {
        print!(
            "\n{} {}",
            style("PROMPT").bold().cyan(),
            style(prompt).dim()
        );
        print!("\n{} ", style("OUTPUT").bold().green());
        std::io::stdout().flush()?;

        let result = llm
            .completion(*prompt)
            .max_output_tokens(64)
            .collect_with(PrintDelta)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        println!();
        println!(
            "   {} in={} out={}  finish={:?}",
            style("tokens").dim(),
            style(result.usage.input_tokens).dim(),
            style(result.usage.output_tokens).dim(),
            result.finish_reason,
        );
    }

    println!("\n{}", style("─".repeat(60)).dim());
    Ok(())
}
