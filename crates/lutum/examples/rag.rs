use std::sync::Arc;

use lutum::*;

const CORPUS: &[(&str, &str)] = &[
    (
        "Office Hours",
        "Monday to Thursday: 9am-6pm. Friday: 9am-4pm. Weekends: closed.",
    ),
    (
        "Holiday Policy",
        "The office is closed on all public holidays.",
    ),
    (
        "Remote Work Policy",
        "Employees may work remotely up to 3 days per week.",
    ),
    (
        "Contact",
        "For urgent matters, call the main line at +1-800-555-0100.",
    ),
];

fn search<'a>(corpus: &[(&'a str, &'a str)], query: &str) -> Vec<(&'a str, &'a str)> {
    let q = query.to_lowercase();
    corpus
        .iter()
        .filter(|(_, content)| {
            q.split_whitespace()
                .any(|kw| content.to_lowercase().contains(kw))
        })
        .cloned()
        .collect()
}

async fn ask(ctx: &Context, system: &str, user: String) -> anyhow::Result<String> {
    let mut session = Session::new(ctx.clone());
    session.push_system(system);
    session.push_user(user);
    let result = session.text_turn().collect().await?;
    Ok(result.assistant_text())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let question = "What time do we close on Fridays?";
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "qwen3.5:2b".into());
    let model = ModelName::new(&model_name)?;
    let adapter = OpenAiAdapter::new(token)
        .with_base_url(endpoint)
        .with_default_model(model);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Context::new(Arc::new(adapter), budget);

    let keywords = ask(
        &ctx,
        "Extract 2-3 search keywords from the question. Output only the keywords separated by spaces.",
        question.into(),
    )
    .await?;
    let evidence = {
        let hits = search(CORPUS, &keywords);
        if hits.is_empty() {
            search(CORPUS, question)
        } else {
            hits
        }
    };
    let docs = evidence
        .iter()
        .map(|(title, content)| format!("[{title}]: {content}"))
        .collect::<Vec<_>>()
        .join("\n");
    let answer = ask(
        &ctx,
        "Answer using only the provided evidence. Cite the document title.",
        format!("Question: {question}\n\nEvidence:\n{docs}"),
    )
    .await?;

    println!("Keywords: {}", keywords.trim());
    println!(
        "Evidence: {:?}",
        evidence.iter().map(|(title, _)| *title).collect::<Vec<_>>()
    );
    println!("Answer: {answer}");
    Ok(())
}
