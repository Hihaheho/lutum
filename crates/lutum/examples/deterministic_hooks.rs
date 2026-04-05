use std::sync::Arc;

use lutum::*;

#[def_hook(always)]
async fn validate_prompt(
    _ctx: &Context,
    _prompt: &str,
    _last: Option<Result<(), String>>,
) -> Result<(), String> {
    Ok(())
}

#[def_hook(always)]
async fn validate_output(
    _ctx: &Context,
    _output: &str,
    _last: Option<Result<(), String>>,
) -> Result<(), String> {
    Ok(())
}

#[hook(ValidatePrompt)]
async fn reject_empty_prompt(
    _ctx: &Context,
    prompt: &str,
    last: Option<Result<(), String>>,
) -> Result<(), String> {
    if let Some(Err(err)) = last {
        return Err(err);
    }
    if prompt.trim().is_empty() {
        Err("empty prompt rejected".into())
    } else {
        println!("[validate_prompt] accepted ({} chars)", prompt.len());
        Ok(())
    }
}

#[hook(ValidateOutput)]
async fn block_dangerous_output(
    _ctx: &Context,
    output: &str,
    last: Option<Result<(), String>>,
) -> Result<(), String> {
    if let Some(Err(err)) = last {
        return Err(err);
    }
    if output.contains("rm -rf") {
        Err("blocked dangerous command in output".into())
    } else {
        println!("[validate_output] accepted ({} chars)", output.len());
        Ok(())
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
    let prompt = "Write a shell command to list all .rs files recursively.";

    let hooks = HookRegistry::new()
        .register_validate_prompt(RejectEmptyPrompt)
        .register_validate_output(BlockDangerousOutput);
    let ctx = Context::with_hooks(
        Arc::new(OpenAiAdapter::new(token).with_base_url(endpoint).with_default_model(model)),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        hooks,
    );

    ctx.validate_prompt(prompt)
        .await
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    let output = ask(
        &ctx,
        "You are a shell expert. Output only the command, nothing else.",
        prompt,
    )
    .await?;
    ctx.validate_output(&output)
        .await
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    println!("{output}");
    Ok(())
}
