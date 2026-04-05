use std::io::{Write, stdout};
use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use lutum::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "qwen3.5:2b".into());
    let adapter = OpenAiAdapter::new(token).with_base_url(endpoint);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Context::new(Arc::new(adapter), budget);
    let input = ModelInput::new()
        .user("Tell a very short story about a robot learning to bake in 2-3 sentences.");

    let pending = ctx
        .text_turn(
            RequestExtensions::new(),
            input,
            TextTurn::<NoTools>::new(ModelName::new(&model_name)?),
            UsageEstimate::zero(),
        )
        .await?;
    let mut stream = pending.into_stream();
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
