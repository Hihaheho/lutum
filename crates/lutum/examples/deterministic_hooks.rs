use std::sync::Arc;

use lutum::*;
use lutum_openai::OpenAiAdapter;

#[hooks]
trait DeterministicHooks {
    #[hook(fallback)]
    async fn validate_prompt(_prompt: &str) -> Result<(), String> {
        Ok(())
    }

    #[hook(fallback)]
    async fn validate_output(_output: &str) -> Result<(), String> {
        Ok(())
    }
}

#[impl_hook(ValidatePrompt)]
async fn reject_empty_prompt(prompt: &str, last: Option<Result<(), String>>) -> Result<(), String> {
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

#[impl_hook(ValidateOutput)]
async fn block_dangerous_output(
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

async fn ask(llm: &Lutum, system: &str, prompt: &str) -> anyhow::Result<String> {
    let mut session = Session::new(llm.clone());
    session.push_system(system);
    session.push_user(prompt);
    let result = session.text_turn().collect().await?;
    Ok(result.assistant_text())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e2b".into());
    let model = ModelName::new(&model_name)?;
    let prompt = "Write a shell command to list all .rs files recursively.";

    let hooks = DeterministicHooksSet::new()
        .with_validate_prompt(RejectEmptyPrompt)
        .with_validate_output(BlockDangerousOutput);
    let llm = Lutum::with_hooks(
        Arc::new(
            OpenAiAdapter::new(token)
                .with_base_url(endpoint)
                .with_default_model(model),
        ),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        LutumHooksSet::new(),
    );

    hooks
        .validate_prompt(prompt)
        .await
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    let output = ask(
        &llm,
        "You are a shell expert. Output only the command, nothing else.",
        prompt,
    )
    .await?;
    hooks
        .validate_output(&output)
        .await
        .map_err(|err| anyhow::anyhow!("{err}"))?;
    println!("{output}");
    Ok(())
}
