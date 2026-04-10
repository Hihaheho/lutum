use std::{collections::HashSet, sync::Arc};

use lutum::*;
use lutum_openai::OpenAiAdapter;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct Contact {
    name: String,
    email: String,
    phone: String,
}

#[derive(Debug)]
struct GateFailure {
    reasons: Vec<String>,
}

impl GateFailure {
    fn new(reasons: Vec<String>) -> Self {
        Self { reasons }
    }

    fn as_retry_block(&self) -> String {
        self.reasons
            .iter()
            .map(|reason| format!("- {reason}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn normalize_phone(value: &str) -> String {
    value
        .chars()
        .filter(|ch| !ch.is_whitespace() && *ch != '-')
        .collect()
}

#[def_hook(always)]
async fn audit_contact(source: &str, contact: &Contact) -> Result<(), Vec<String>> {
    let mut failures = Vec::new();
    let source_tokens = source
        .split_whitespace()
        .map(|token| token.to_ascii_lowercase())
        .collect::<HashSet<_>>();

    if contact.name.trim().is_empty() {
        failures.push("name must not be empty".into());
    } else {
        for token in contact.name.split_whitespace() {
            if !source_tokens.contains(&token.to_ascii_lowercase()) {
                failures.push(format!("name token '{token}' not found in source"));
            }
        }
    }

    if contact.email.trim().is_empty() {
        failures.push("email must not be empty".into());
    } else if !source.contains(&contact.email) {
        failures.push(format!("email '{}' not found in source", contact.email));
    }

    if contact.phone.trim().is_empty() {
        failures.push("phone must not be empty".into());
    } else {
        let normalized_source = normalize_phone(source);
        let normalized_phone = normalize_phone(&contact.phone);
        if normalized_phone.is_empty() || !normalized_source.contains(&normalized_phone) {
            failures.push(format!("phone '{}' not found in source", contact.phone));
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

#[hooks]
struct VerificationHooks {
    audit_contact: AuditContact,
}

fn build_prompt(source: &str, prior_failure: Option<&GateFailure>) -> String {
    let mut prompt = format!(
        "Source text: \"{source}\"\n\
         Extract one contact and return JSON only with string fields: name, email, phone.\n\
         Rust will parse the JSON and then decide pass/fail with deterministic gates.\n\
         Rust gates:\n\
         - every whitespace-separated token in name must appear in the source text, case-insensitive\n\
         - email must appear verbatim in the source text\n\
         - phone must appear in the source text after removing spaces and dashes for comparison\n\
         Do not claim success. Return the best corrected JSON only."
    );

    if let Some(failure) = prior_failure {
        prompt.push_str(
            "\n\nThe previous attempt failed Rust verification. Fix every issue below before you answer:\n",
        );
        prompt.push_str(&failure.as_retry_block());
    }

    prompt
}

async fn extract(llm: &Lutum, prompt: &str) -> anyhow::Result<Contact> {
    let mut session = Session::new(llm.clone());
    session.push_system(
        "You extract contacts from source text. Rust decides whether the extraction passes verification. Use only values grounded in the source.",
    );
    session.push_user(prompt);

    let result = session.structured_turn::<Contact>().collect().await?;
    match result.semantic {
        StructuredTurnOutcome::Structured(contact) => Ok(contact),
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
    let hooks = VerificationHooks::new();
    let llm = Lutum::with_hooks(Arc::new(adapter), budget, LutumHooks::new());
    let source = "Call John Smith at john@example.com or +1-555-0100";
    let mut prior_failure = None;

    println!("Source: {source}");
    println!("Rust decides whether the extracted contact passes verification.");

    for attempt in 1..=3 {
        let prompt = build_prompt(source, prior_failure.as_ref());
        println!("\nAttempt {attempt}");

        match extract(&llm, &prompt).await {
            Ok(contact) => {
                println!("Extracted contact: {contact:#?}");

                match hooks.audit_contact(source, &contact).await {
                    Ok(()) => {
                        println!("Rust gates: pass");
                        return Ok(());
                    }
                    Err(reasons) => {
                        let failure = GateFailure::new(reasons);
                        println!("Rust gates: fail");
                        for reason in &failure.reasons {
                            println!("- {reason}");
                        }

                        if attempt == 3 {
                            println!("Rejected after 3 attempts");
                            return Ok(());
                        }

                        prior_failure = Some(failure);
                    }
                }
            }
            Err(err) => {
                let failure = GateFailure::new(vec![format!(
                    "output did not produce a valid Contact for Rust to audit: {err}"
                )]);
                println!("Extraction failed before Rust gates: {err}");

                if attempt == 3 {
                    println!("Rejected after 3 attempts");
                    return Ok(());
                }

                prior_failure = Some(failure);
            }
        }
    }

    Ok(())
}
