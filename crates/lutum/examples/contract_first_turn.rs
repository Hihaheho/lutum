use lutum::*;
use std::sync::Arc;

#[rustfmt::skip]
struct Contract { output_format: &'static str, allowed_routes: &'static [&'static str], evidence_terms: &'static [&'static str] }

#[rustfmt::skip]
fn render_contract(c: &Contract) -> String {
    format!("Contract hint for the model:\n- format: {}\n- route_id must be one of: {}\n- reason must be exactly one sentence ending with one period\n- reason must cite at least one evidence term from the ticket: {}", c.output_format, c.allowed_routes.join(", "), c.evidence_terms.join(", "))
}

#[rustfmt::skip]
fn audit(c: &Contract, output: &str) -> Result<(), Vec<String>> {
    let mut failures = Vec::new();
    let (route, reason) = match output.trim().splitn(3, ": ").collect::<Vec<_>>().as_slice() { [route, reason] => (*route, reason.trim()), _ => { failures.push(format!("format must be {}", c.output_format)); ("", "") } };
    if !route.is_empty() && !c.allowed_routes.contains(&route) { failures.push(format!("route_id must be exactly one of: {}", c.allowed_routes.join(", "))); }
    if !reason.is_empty() && (reason.matches('.').count() != 1 || !reason.ends_with('.') || reason.contains('!') || reason.contains('?')) { failures.push("reason must be exactly one sentence ending with one period".into()); }
    if !reason.is_empty() && !c.evidence_terms.iter().any(|term| reason.to_ascii_lowercase().contains(&term.to_ascii_lowercase())) {
        failures.push(format!("reason must include at least one evidence term: {}", c.evidence_terms.join(", ")));
    }
    if failures.is_empty() { Ok(()) } else { Err(failures) }
}

#[rustfmt::skip]
fn add_usage(total: &mut Usage, usage: Usage) {
    total.input_tokens += usage.input_tokens;
    total.output_tokens += usage.output_tokens;
    total.total_tokens += usage.total_tokens;
    total.cost_micros_usd += usage.cost_micros_usd;
}

#[rustfmt::skip]
async fn ask(llm: &Lutum, system: &str, user: impl Into<String>) -> anyhow::Result<(String, Usage)> {
    let mut session = Session::new(llm.clone());
    session.push_system(system);
    session.push_user(user);
    let result = session.text_turn().collect().await?;
    Ok((result.assistant_text(), result.usage))
}

#[tokio::main]
#[rustfmt::skip]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "qwen3.5:2b".into());
    let model = ModelName::new(&model_name)?;
    let llm = Lutum::new(Arc::new(OpenAiAdapter::new(token).with_base_url(endpoint).with_default_model(model)), SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()));
    let ticket = "I was charged twice for my subscription last month.";
    let policy = "You route support tickets.\n- refund_request: duplicate charge, incorrect charge, or refund-needed billing error.\n- billing_inquiry: billing question that does not require a refund.\n- technical_issue: product malfunction, login problem, or bug.";
    let contract = Contract { output_format: "<route_id>: <one-sentence reason>", allowed_routes: &["refund_request", "billing_inquiry", "technical_issue"], evidence_terms: &["charged", "twice", "subscription", "last month", "billing"] };
    let hint = render_contract(&contract);
    let system = format!("{policy}\n\nThe contract is defined in Rust and audited after you answer.\n{hint}");
    let user = format!("Route this ticket. Return only {}.\nTicket: {ticket}", contract.output_format);
    let mut usage = Usage::zero();

    println!("Contract evaluated in Rust:\n{hint}\nTicket: {ticket}\nExpected route: refund_request\n");
    let (first, first_usage) = ask(&llm, &system, &user).await?;
    add_usage(&mut usage, first_usage);
    println!("Attempt 1: {first}");
    match audit(&contract, &first) {
        Ok(()) => println!("Audit: pass"),
        Err(failures) => {
            println!("Audit: fail\n- {}", failures.join("\n- "));
            let retry_user = format!("{user}\n\nAudit failed. Fix every issue:\n- {}", failures.join("\n- "));
            let (retry, retry_usage) = ask(&llm, &system, retry_user).await?;
            add_usage(&mut usage, retry_usage);
            println!("\nRetry: {retry}");
            match audit(&contract, &retry) {
                Ok(()) => println!("Audit after retry: pass"),
                Err(failures) => println!("Audit after retry: fail\n- {}", failures.join("\n- ")),
            }
        }
    }
    println!("\nTokens: in={} out={} total={}", usage.input_tokens, usage.output_tokens, usage.total_tokens);
    Ok(())
}
