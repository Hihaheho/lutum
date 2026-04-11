use std::sync::Arc;

use lutum::*;
use lutum_openai::OpenAiAdapter;

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

async fn ask(llm: &Lutum, system: &str, prompt: &str) -> anyhow::Result<String> {
    let mut session = Session::new(llm.clone());
    session.push_system(system);
    session.push_user(prompt);
    let result = session.text_turn().collect().await?;
    Ok(result.assistant_text())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e2b".into());
    let model = ModelName::new(&model_name)?;
    let llm = Lutum::new(
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
        let output = ask(&llm, system, &prompt).await?;
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
