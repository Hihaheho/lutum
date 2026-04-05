use lutum::*;
use std::sync::Arc;

const COMPACTION_THRESHOLD_WORDS: usize = 80;
const SYSTEM: &str = "You are a network troubleshooting assistant.";
const COMPACTION_PROMPT: &str = "Summarize the diagnostic session so far using exactly this format:\ntried: <comma-separated list of what was attempted>\ncurrent_hypothesis: <one sentence>\nnext_actions: <comma-separated list>\nunresolved: <anything still unclear>";
const HISTORY: &[&str] = &[
    "My home network keeps dropping every few minutes.",
    "How long has this been happening?",
    "About two weeks. I recently got a new router.",
    "Have you updated the router firmware?",
    "Yes, it's up to date. The drops happen every 10-15 minutes exactly.",
    "Regularity suggests interference or a lease renewal issue. What channel is the router on?",
    "It's on auto. Should I fix it to a specific channel?",
    "Yes, try channel 1, 6, or 11 for 2.4GHz, or a non-overlapping channel for 5GHz.",
    "I switched to channel 6. Still dropping.",
    "The regular 10-15 minute interval strongly suggests DHCP lease renewal. Check your DHCP lease time and try extending it to 24 hours.",
];

fn approx_word_count(text: &str) -> usize {
    text.split_whitespace().count()
}
fn session(ctx: &Context) -> Session {
    Session::new(ctx.clone())
}
fn seed_history(session: &mut Session) {
    session.push_system(SYSTEM);
    for (i, text) in HISTORY.iter().enumerate() {
        if i % 2 == 0 {
            session.push_user(*text);
        } else {
            session
                .input_mut()
                .push(ModelInputItem::assistant_text(*text));
        }
    }
}
async fn ask(session: &Session) -> anyhow::Result<(String, Usage)> {
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
        TextStepOutcome::Finished(result) => Ok((result.assistant_text(), result.usage)),
        TextStepOutcome::NeedsToolResults(_) => unreachable!(),
    }
}
async fn ask_with_prompt(
    ctx: &Context,
    prompt: impl Into<String>,
) -> anyhow::Result<(String, Usage)> {
    let mut session = session(ctx);
    seed_history(&mut session);
    session.push_user(prompt);
    ask(&session).await
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "qwen3.5:2b".into());
    let model = ModelName::new(&model_name)?;
    let ctx = Context::new(
        Arc::new(
            OpenAiAdapter::new(token)
                .with_base_url(endpoint)
                .with_default_model(model),
        ),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let transcript_word_count: usize = HISTORY.iter().map(|text| approx_word_count(text)).sum();
    let (full_answer, _) = ask_with_prompt(&ctx, "What should the user try next?").await?;

    println!("Approx original transcript word count: {transcript_word_count}");

    if transcript_word_count > COMPACTION_THRESHOLD_WORDS {
        let (summary, compaction_usage) = ask_with_prompt(&ctx, COMPACTION_PROMPT).await?;
        println!(
            "Provider-reported compaction input tokens: {}",
            compaction_usage.input_tokens
        );
        println!(
            "Compaction summary ({} words):\n{summary}",
            approx_word_count(&summary)
        );

        let mut compact = session(&ctx);
        compact.push_system(SYSTEM);
        compact.push_user(format!(
            "Summary of the previous troubleshooting session:\n{summary}"
        ));
        compact.push_user("What should the user try next?");
        let (compacted_answer, _) = ask(&compact).await?;

        println!("\nFull-history answer:   {full_answer}");
        println!("Compacted answer:      {compacted_answer}");
    } else {
        println!("Context within budget, no GC needed.");
        println!("\nFull-history answer:   {full_answer}");
    }
    Ok(())
}
