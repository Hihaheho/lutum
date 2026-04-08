use std::sync::Arc;

use lutum::*;
use lutum_openai::OpenAiAdapter;

const WRITE: &str = "Write a short, catchy tagline for a Rust HTTP client library. One line only.";
const EVALUATE: &str =
    "Is this tagline catchy and memorable? Reply with YES or NO and one sentence why.";
const REFLECT: &str = "What should be done differently to make the tagline catchier? One sentence.";

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
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "qwen3.5:2b".into());
    let model = ModelName::new(&model_name)?;
    let adapter = OpenAiAdapter::new(token)
        .with_base_url(endpoint)
        .with_default_model(model);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let llm = Lutum::new(Arc::new(adapter), budget);
    let mut memory = String::new();

    for round in 1..=3 {
        let prompt = if memory.is_empty() {
            "Write the tagline.".to_string()
        } else {
            format!("Previous reflection: {memory}\nWrite the tagline.")
        };
        let tagline = ask(&llm, WRITE, &prompt).await?;
        println!("Round {round} tagline: {tagline}");

        let evaluation = ask(&llm, EVALUATE, &tagline).await?;
        println!("Evaluation: {evaluation}");
        if evaluation
            .trim_start()
            .to_ascii_uppercase()
            .starts_with("YES")
        {
            break;
        }

        let reflect_prompt = format!("Tagline: {tagline}\nEvaluation: {evaluation}");
        memory = ask(&llm, REFLECT, &reflect_prompt).await?;
        println!("Reflection: {memory}\n");
    }

    Ok(())
}
