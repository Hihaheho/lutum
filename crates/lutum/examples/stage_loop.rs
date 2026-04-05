use std::sync::Arc;

use lutum::*;

async fn ask(
    ctx: &Context,
    model: &ModelName,
    prompt: impl Into<String>,
) -> anyhow::Result<String> {
    let mut session = Session::new(ctx.clone()).with_defaults(SessionDefaults {
        model: Some(model.clone()),
        ..Default::default()
    });
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
    let adapter = OpenAiAdapter::new(token).with_base_url(endpoint);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Context::new(Arc::new(adapter), budget);
    let model = ModelName::new(&model_name)?;
    let base_prompt = "Write a haiku about programming. Output only the haiku.";

    // Stage 1: Act
    let mut haiku = ask(&ctx, &model, base_prompt).await?;
    let mut final_haiku = None;

    for attempt in 1..=3 {
        println!("Attempt {attempt}:\n{haiku}\n");

        // Stage 2: Verify
        let verdict = ask(
            &ctx,
            &model,
            format!(
                "Does this haiku follow 5-7-5 syllables? Reply with just PASS or FAIL and why.\n\n{haiku}"
            ),
        )
        .await?;
        println!("Verdict: {verdict}\n");

        if verdict.trim().to_ascii_uppercase().starts_with("PASS") {
            final_haiku = Some(haiku.clone());
            break;
        }

        if attempt == 3 {
            break;
        }

        // Stage 3: Repair
        haiku = ask(
            &ctx,
            &model,
            format!(
                "{base_prompt}\nRepair the draft using this feedback: {verdict}\n\nDraft:\n{haiku}"
            ),
        )
        .await?;
    }

    println!("Final haiku:\n{}", final_haiku.unwrap_or(haiku));
    Ok(())
}
