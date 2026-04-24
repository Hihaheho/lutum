use std::{collections::HashSet, sync::Arc};

use lutum::*;
use lutum_openai::OpenAiAdapter;

type Validation = Result<(), Vec<String>>;

const ALLOWED_PREFIXES: &[&str] = &["/var/log", "/tmp"];
const FORBIDDEN_TOKENS: &[&str] = &["rm", "mv", "sudo", ">", ">>", "dd"];
const MAX_PIPES: usize = 2;

#[hooks]
trait ShellHooks {
    #[hook(always)]
    async fn validate_command(cmd: &str) -> Validation {
        let failures = policy_failures(cmd);
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
    }
}

#[derive(Default)]
struct RetryMemory {
    rejected_commands: HashSet<String>,
    rejection_count: usize,
}

impl StatefulValidateCommand for RetryMemory {
    async fn call_mut(&mut self, cmd: String, last: Option<Validation>) -> Validation {
        let command = cmd.trim().to_string();
        let mut failures = match last {
            Some(Ok(())) | None => Vec::new(),
            Some(Err(reasons)) => reasons,
        };

        if self.rejected_commands.contains(&command) {
            failures.push(
                "command was already rejected on an earlier attempt; return a materially different command"
                    .into(),
            );
        }

        if failures.is_empty() {
            Ok(())
        } else {
            self.rejection_count += 1;
            self.rejected_commands.insert(command);
            println!(
                "[stateful hook] rejection #{} ({} unique rejected command(s) remembered)",
                self.rejection_count,
                self.rejected_commands.len()
            );
            Err(failures)
        }
    }
}

fn policy_failures(cmd: &str) -> Vec<String> {
    let mut failures = Vec::new();
    let tokens = cmd.split_whitespace().collect::<Vec<_>>();

    for &tok in FORBIDDEN_TOKENS {
        if tokens.contains(&tok) {
            failures.push(format!("forbidden token: `{tok}`"));
        }
    }

    let pipe_count = cmd.chars().filter(|&ch| ch == '|').count();
    if pipe_count > MAX_PIPES {
        failures.push(format!("too many pipes: {pipe_count} > {MAX_PIPES}"));
    }

    for &tok in &tokens {
        if tok.starts_with('/')
            && !ALLOWED_PREFIXES
                .iter()
                .any(|prefix| tok.starts_with(prefix))
        {
            failures.push(format!("path not in allowed roots: `{tok}`"));
        }
    }

    failures
}

fn build_prompt(request: &str, failures: &[String]) -> String {
    if failures.is_empty() {
        request.to_string()
    } else {
        format!(
            "{request}\n\nPrevious attempt was rejected. Fix every issue and return a different command than any rejected attempt:\n{}",
            failures
                .iter()
                .map(|reason| format!("- {reason}"))
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }
}

async fn ask(llm: &Lutum, system: &str, prompt: &str) -> anyhow::Result<String> {
    let mut session = Session::new(llm.clone());
    session.push_system(system);
    session.push_user(prompt);
    let result = session.text_turn().collect().await?;
    Ok(result.assistant_text())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model = ModelName::new(std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e2b".into()))?;
    let hooks = ShellHooksSet::new().with_validate_command(Stateful::new(RetryMemory::default()));
    let llm = Lutum::with_hooks(
        Arc::new(
            OpenAiAdapter::new(token)
                .with_base_url(endpoint)
                .with_default_model(model),
        ),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        LutumHooksSet::new(),
    );
    let system = "You are a shell expert for log triage on a read-only system.\nOutput only the shell command, nothing else.";
    let request = "List the 5 most recent error lines from /var/log/syslog.";
    let mut failures = Vec::new();

    for attempt in 1..=3 {
        let prompt = build_prompt(request, &failures);
        let cmd = ask(&llm, system, &prompt).await?;
        println!("Attempt {attempt}: {cmd}");

        match hooks.validate_command(&cmd).await {
            Ok(()) => {
                println!("Policy: pass");
                return Ok(());
            }
            Err(reasons) => {
                println!("Policy: fail");
                for reason in &reasons {
                    println!("  - {reason}");
                }
                if attempt == 3 {
                    println!("Rejected after 3 attempts");
                    return Ok(());
                }
                failures = reasons;
            }
        }
    }

    Ok(())
}
