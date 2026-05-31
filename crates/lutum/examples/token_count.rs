//! Count provider-side input tokens before running a turn.
//!
//! Defaults to a local Ollama OpenAI-compatible endpoint:
//!   cargo run --example token_count -p lutum
//!
//! Override via environment variables:
//!   ENDPOINT=https://api.openai.com/v1 TOKEN=<key> MODEL=gpt-5.4-nano \
//!     cargo run --example token_count -p lutum

use std::sync::Arc;

use console::style;
use lutum::*;
use lutum_openai::OpenAiAdapter;

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model = std::env::var("MODEL").unwrap_or_else(|_| "gemma4:26b".into());

    let adapter = Arc::new(
        OpenAiAdapter::new(token)
            .with_base_url(&endpoint)
            .with_default_model(ModelName::new(&model)?),
    );
    let llm = Lutum::new(
        adapter.clone(),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    )
    .with_token_counter(adapter);

    let mut session = Session::new();
    session.push_system("Answer concisely.");
    session.push_user("Explain what provider-side token counting is useful for.");

    let turn = session.text_turn(&llm).max_output_tokens(64);

    match turn.count_tokens().await? {
        Some(count) => println!(
            "{} input_tokens={}",
            style("count").bold().cyan(),
            count.input_tokens
        ),
        None => println!("{}", style("token counting is unsupported").yellow()),
    }

    let result = turn
        .collect()
        .await
        .map_err(|err| anyhow::anyhow!("{err}"))?;

    println!("\n{}", result.assistant_text());
    println!(
        "\n{} in={} out={} total={}",
        style("usage").dim(),
        result.usage.input_tokens,
        result.usage.output_tokens,
        result.usage.total_tokens
    );

    Ok(())
}
