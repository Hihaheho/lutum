use std::sync::Arc;

use lutum::*;

async fn ask(
    ctx: &Context,
    model: &ModelName,
    system: Option<&str>,
    prompt: &str,
    temperature: Option<Temperature>,
) -> anyhow::Result<String> {
    let mut session = Session::new(ctx.clone()).with_defaults(SessionDefaults {
        model: Some(model.clone()),
        ..Default::default()
    });
    if let Some(system) = system {
        session.push_system(system);
    }
    session.push_user(prompt);
    let mut turn = session.text_turn::<NoTools>().unwrap();
    turn.config.generation.temperature = temperature;
    let outcome = session
        .prepare_text(RequestExtensions::new(), turn, UsageEstimate::zero())
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
    let adapter = OpenAiAdapter::new(token).with_base_url(endpoint);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Context::new(Arc::new(adapter), budget);
    let model = ModelName::new(&model_name)?;
    let mut candidates = Vec::new();

    for _ in 0..3 {
        candidates.push(ask(&ctx, &model, None, "Suggest ONE name for a Rust async runtime library. Output only the name, nothing else.", Some(Temperature::try_from(1.0_f32)?)).await?);
    }

    let options = candidates
        .iter()
        .enumerate()
        .map(|(i, name)| format!("{}. {name}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");
    println!("Candidates:\n{options}\n");

    let judge = ask(&ctx, &model, Some("You are judging library names. Pick the best one based on: memorable, pronounceable, Rust-themed."), &format!("{options}\nWhich is best and why?"), None).await?;
    println!("Judge:\n{judge}");
    Ok(())
}
