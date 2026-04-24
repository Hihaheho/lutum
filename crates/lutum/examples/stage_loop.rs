use std::sync::Arc;

use lutum::*;
use lutum_openai::OpenAiAdapter;

async fn ask(llm: &Lutum, prompt: impl Into<String>) -> anyhow::Result<String> {
    let mut session = Session::new(llm.clone());
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
    let adapter = OpenAiAdapter::new(token)
        .with_base_url(endpoint)
        .with_default_model(model);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let llm = Lutum::new(Arc::new(adapter), budget);
    let base_prompt = "Write a haiku about programming. Output only the haiku.";

    // Stage 1: Act
    let mut haiku = ask(&llm, base_prompt).await?;
    let mut final_haiku = None;

    for attempt in 1..=3 {
        println!("Attempt {attempt}:\n{haiku}\n");

        // Stage 2: Verify
        let verdict = ask(
            &llm,
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
            &llm,
            format!(
                "{base_prompt}\nRepair the draft using this feedback: {verdict}\n\nDraft:\n{haiku}"
            ),
        )
        .await?;
    }

    println!("Final haiku:\n{}", final_haiku.unwrap_or(haiku));
    Ok(())
}
