use std::sync::Arc;

use lutum::*;

const SUBJECT: &str = "fn add(a: i32, b: i32) -> i32 { a + b }";

async fn ask(ctx: &Context, system: &str, prompt: impl Into<String>) -> anyhow::Result<String> {
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
    let adapter = OpenAiAdapter::new(token)
        .with_base_url(endpoint)
        .with_default_model(model);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Context::new(Arc::new(adapter), budget);

    let plan = ask(&ctx, "You are a code review planner. Given a Rust function, list 2-3 concrete improvements and acceptance criteria for each. Be specific. Output ONLY requirements and acceptance criteria. Do not write code.", SUBJECT).await?;
    println!("Planner Output:\n{plan}\n");

    let code = ask(&ctx, "You are a Rust code generator. Given a spec, rewrite the function to meet all acceptance criteria. Output ONLY code. Do not change requirements.", format!("original function:\n{SUBJECT}\n\nspec:\n{plan}")).await?;
    println!("Generator Output:\n{code}\n");

    let evaluation = ask(&ctx, "You are a code evaluator. Score the improved function 1-10 and list any remaining issues vs. the spec. Output ONLY a score from 1-10 and a list of issues. Do not rewrite the code.", format!("spec: {plan}\ncode: {code}")).await?;
    println!("Evaluator Output:\n{evaluation}");
    Ok(())
}
