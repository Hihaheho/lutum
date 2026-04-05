use std::sync::Arc;

use lutum::*;

const WRITE: &str = "Write a short, catchy tagline for a Rust HTTP client library. One line only.";
const EVALUATE: &str =
    "Is this tagline catchy and memorable? Reply with YES or NO and one sentence why.";
const REFLECT: &str = "What should be done differently to make the tagline catchier? One sentence.";

async fn ask(ctx: &Context, system: &str, prompt: &str) -> anyhow::Result<String> {
    let mut session = Session::new(ctx.clone());
    session.push_system(system);
    session.push_user(prompt);
    let outcome = session
        .prepare_text(
            RequestExtensions::new(),
            session.text_turn::<NoTools>(),
            UsageEstimate::zero(),
        )
        .await?
        .collect_noop()
        .await?;
    match outcome {
        TextStepOutcome::Finished(result) => Ok(result.assistant_text()),
        TextStepOutcome::NeedsToolResults(_) => unreachable!(),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "qwen3.5:2b".into());
    let model = ModelName::new(&model_name)?;
    let adapter = OpenAiAdapter::new(token).with_base_url(endpoint).with_default_model(model);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Context::new(Arc::new(adapter), budget);
    let mut memory = String::new();

    for round in 1..=3 {
        let prompt = if memory.is_empty() {
            "Write the tagline.".to_string()
        } else {
            format!("Previous reflection: {memory}\nWrite the tagline.")
        };
        let tagline = ask(&ctx, WRITE, &prompt).await?;
        println!("Round {round} tagline: {tagline}");

        let evaluation = ask(&ctx, EVALUATE, &tagline).await?;
        println!("Evaluation: {evaluation}");
        if evaluation
            .trim_start()
            .to_ascii_uppercase()
            .starts_with("YES")
        {
            break;
        }

        let reflect_prompt = format!("Tagline: {tagline}\nEvaluation: {evaluation}");
        memory = ask(&ctx, REFLECT, &reflect_prompt).await?;
        println!("Reflection: {memory}\n");
    }

    Ok(())
}
