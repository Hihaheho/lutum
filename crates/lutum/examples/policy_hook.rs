use lutum::*;
use lutum_openai::OpenAiAdapter;
use std::sync::Arc;

type Validation = Result<(), Vec<String>>;

#[hooks]
trait ShellHooks {
    #[hook(fallback)]
    async fn validate_command(_cmd: &str) -> Validation {
        Ok(())
    }
}

struct CommandPolicy {
    allowed_prefixes: &'static [&'static str],
    forbidden_tokens: &'static [&'static str],
    max_pipes: usize,
}

impl ValidateCommand for CommandPolicy {
    async fn call(&self, cmd: String, last: Option<Validation>) -> Validation {
        if let Some(Err(reasons)) = last {
            return Err(reasons);
        }
        let mut failures = Vec::new();
        let tokens = cmd.split_whitespace().collect::<Vec<_>>();
        for &tok in self.forbidden_tokens {
            if tokens.contains(&tok) {
                failures.push(format!("forbidden token: `{tok}`"));
            }
        }
        let pipe_count = cmd.chars().filter(|&ch| ch == '|').count();
        if pipe_count > self.max_pipes {
            failures.push(format!("too many pipes: {pipe_count} > {}", self.max_pipes));
        }
        for &tok in &tokens {
            if tok.starts_with('/')
                && !self
                    .allowed_prefixes
                    .iter()
                    .any(|prefix| tok.starts_with(prefix))
            {
                failures.push(format!("path not in allowed roots: `{tok}`"));
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
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
    let hooks = ShellHooksSet::new().with_validate_command(CommandPolicy {
        allowed_prefixes: &["/var/log", "/tmp"],
        forbidden_tokens: &["rm", "mv", "sudo", ">", ">>", "dd"],
        max_pipes: 2,
    });
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
        let prompt = if failures.is_empty() {
            request.to_string()
        } else {
            format!(
                "{request}\n\nPrevious attempt was rejected. Fix every issue:\n{}",
                failures
                    .iter()
                    .map(|reason| format!("- {reason}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        };
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
