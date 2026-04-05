use lutum::*;
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

async fn ask(ctx: &Context, system: &str, user: impl Into<String>) -> anyhow::Result<String> {
    let mut session = Session::new(ctx.clone());
    session.push_system(system);
    session.push_user(user);
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

async fn classify(
    ctx: &Context,
    system: &str,
    user: impl Into<String>,
) -> anyhow::Result<RecoveryAction> {
    let mut session = Session::new(ctx.clone());
    session.push_system(system);
    session.push_user(user);
    let outcome = session
        .prepare_structured(
            RequestExtensions::new(),
            session.structured_turn::<NoTools, RecoveryAction>(),
            UsageEstimate::zero(),
        )
        .await?
        .collect_noop()
        .await?;
    match outcome {
        StructuredStepOutcome::Finished(result) => match result.semantic {
            StructuredTurnOutcome::Structured(action) => Ok(action),
            StructuredTurnOutcome::Refusal(reason) => anyhow::bail!(reason),
        },
        StructuredStepOutcome::NeedsToolResults(_) => unreachable!(),
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
    let ctx = Context::new(Arc::new(adapter), budget);
    let translation = ask(
        &ctx,
        "Translate the following to Japanese. Output only the Japanese translation.",
        "Hello, how are you?",
    )
    .await?;
    println!("Translation: {translation}");
    let recovery = classify(&ctx, "Classify the translation quality. Choose an action:\n  ACCEPT: looks like valid Japanese.\n  RETRY: output seems wrong but retryable.\n  COMPACT: prompt is confusing; simplify.\n  ESCALATE: cannot recover automatically.", format!("Translation: {translation}")).await?;
    println!("Action: {:?} — {}", recovery.action, recovery.reason);
    match recovery.action {
        Action::Accept => println!("Done."),
        Action::Retry => println!(
            "Retry translation: {}",
            ask(
                &ctx,
                "Translate the following to Japanese. Output Japanese characters only.",
                "Hello, how are you?"
            )
            .await?
        ),
        Action::Compact => println!(
            "Compact translation: {}",
            ask(
                &ctx,
                "Translate the following to Japanese. Output only the Japanese translation, nothing else.",
                "Hello, how are you?"
            )
            .await?
        ),
        Action::Escalate => println!("Escalating to human review."),
    }
    Ok(())
}
