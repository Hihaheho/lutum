use std::io::{Write, stdout};
use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use lutum::*;
use lutum_openai::OpenAiAdapter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e2b".into());
    let adapter = OpenAiAdapter::new(token)
        .with_base_url(endpoint)
        .with_default_model(ModelName::new(&model_name)?);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let llm = Lutum::new(Arc::new(adapter), budget);
    let input = ModelInput::new()
        .user("Tell a very short story about a robot learning to bake in 2-3 sentences.");

    let mut stream = llm.text_turn(input).stream().await?;
    let start = Instant::now();

    while let Some(event) = stream.next().await {
        match event? {
            TextTurnEvent::TextDelta { delta } => {
                print!("{delta}");
                stdout().flush()?;
            }
            TextTurnEvent::Completed { usage, .. } => {
                println!();
                println!("Elapsed: {:.2?}", start.elapsed());
                println!(
                    "Tokens: in={} out={} total={}",
                    usage.input_tokens, usage.output_tokens, usage.total_tokens
                );
            }
            _ => {}
        }
    }

    Ok(())
}
