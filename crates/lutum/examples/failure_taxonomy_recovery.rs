use lutum::*;
use lutum_openai::OpenAiAdapter;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum Action {
    Accept,
    Retry,
    Compact,
    Escalate,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct RecoveryAction {
    action: Action,
    reason: String,
}

async fn ask(llm: &Lutum, system: &str, user: impl Into<String>) -> anyhow::Result<String> {
    let mut session = Session::new(llm.clone());
    session.push_system(system);
    session.push_user(user);
    let result = session.text_turn().collect().await?;
    Ok(result.assistant_text())
}

async fn classify(
    llm: &Lutum,
    system: &str,
    user: impl Into<String>,
) -> anyhow::Result<RecoveryAction> {
    let mut session = Session::new(llm.clone());
    session.push_system(system);
    session.push_user(user);
    let result = session
        .structured_turn::<RecoveryAction>()
        .collect()
        .await?;
    match result.semantic {
        StructuredTurnOutcome::Structured(action) => Ok(action),
        StructuredTurnOutcome::Refusal(reason) => anyhow::bail!(reason),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "qwen3.5:2b".into());
    let model = ModelName::new(&model_name)?;
    let adapter = OpenAiAdapter::new(token)
        .with_base_url(endpoint)
        .with_default_model(model);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let llm = Lutum::new(Arc::new(adapter), budget);
    let translation = ask(
        &llm,
        "Translate the following to Japanese. Output only the Japanese translation.",
        "Hello, how are you?",
    )
    .await?;
    println!("Translation: {translation}");
    let recovery = classify(&llm, "Classify the translation quality. Choose an action:\n  ACCEPT: looks like valid Japanese.\n  RETRY: output seems wrong but retryable.\n  COMPACT: prompt is confusing; simplify.\n  ESCALATE: cannot recover automatically.", format!("Translation: {translation}")).await?;
    println!("Action: {:?} — {}", recovery.action, recovery.reason);
    match recovery.action {
        Action::Accept => println!("Done."),
        Action::Retry => println!(
            "Retry translation: {}",
            ask(
                &llm,
                "Translate the following to Japanese. Output Japanese characters only.",
                "Hello, how are you?"
            )
            .await?
        ),
        Action::Compact => println!(
            "Compact translation: {}",
            ask(
                &llm,
                "Translate the following to Japanese. Output only the Japanese translation, nothing else.",
                "Hello, how are you?"
            )
            .await?
        ),
        Action::Escalate => println!("Escalating to human review."),
    }
    Ok(())
}
