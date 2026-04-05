use std::sync::Arc;

use lutum::*;

fn pre_hook(prompt: &str) -> Result<(), String> {
    if prompt.trim().is_empty() {
        return Err("Pre-hook: empty prompt rejected".into());
    }
    println!("[pre-hook] prompt accepted ({} chars)", prompt.len());
    Ok(())
}

fn post_hook(output: &str) -> Result<(), String> {
    if output.contains("rm -rf") {
        return Err("Post-hook: blocked dangerous command in output".into());
    }
    println!("[post-hook] output accepted ({} chars)", output.len());
    Ok(())
}

async fn ask(
    ctx: &Context,
    model: &ModelName,
    system: &str,
    prompt: &str,
) -> anyhow::Result<String> {
    let mut session = Session::new(ctx.clone()).with_defaults(SessionDefaults {
        model: Some(model.clone()),
        ..Default::default()
    });
    session.push_system(system);
    session.push_user(prompt);
    let outcome = session
        .prepare_text(
            RequestExtensions::new(),
            session.text_turn::<NoTools>().unwrap(),
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
    let ctx = Context::new(
        Arc::new(OpenAiAdapter::new(token).with_base_url(endpoint)),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let model = ModelName::new(&model_name)?;
    let prompt = "Write a shell command to list all .rs files recursively.";

    pre_hook(prompt).map_err(|e| anyhow::anyhow!("{e}"))?;
    let output = ask(
        &ctx,
        &model,
        "You are a shell expert. Output only the command, nothing else.",
        prompt,
    )
    .await?;
    post_hook(&output).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{output}");
    Ok(())
}
