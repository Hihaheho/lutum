use std::sync::Arc;

use lutum::*;

fn lint(output: &str) -> Option<String> {
    if output.contains("#[derive(") && output.contains("Debug") {
        None
    } else {
        Some(
            "Constraint violated: the struct must have #[derive(Debug)] in its derive attribute."
                .into(),
        )
    }
}

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
    let ctx = Context::new(
        Arc::new(
            OpenAiAdapter::new(token)
                .with_base_url(endpoint)
                .with_default_model(model),
        ),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let system = "You are a Rust code generator. Output only the struct definition.";
    let mut prompt =
        "Define a Rust struct called `User` with fields: name (String), age (u32).".to_string();

    for attempt in 1..=2 {
        let output = ask(&ctx, system, &prompt).await?;
        match lint(&output) {
            None => {
                println!("Lint passed");
                println!("{output}");
                return Ok(());
            }
            Some(msg) if attempt == 1 => {
                prompt = format!(
                    "Define a Rust struct called User with fields: name (String), age (u32). Constraint: {msg}"
                );
            }
            Some(_) => {
                println!("Lint failed");
                println!("{output}");
                return Ok(());
            }
        }
    }

    Ok(())
}
