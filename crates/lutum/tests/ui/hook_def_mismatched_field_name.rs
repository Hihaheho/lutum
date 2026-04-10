use std::sync::Arc;

use lutum::{
    Lutum, LutumHooks, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions,
};

#[lutum::def_hook(fallback)]
async fn validate_command(_ctx: &Lutum, _cmd: &str) -> Result<(), &'static str> {
    Ok(())
}

#[lutum::hook(ValidateCommand)]
async fn reject_secrets(
    _ctx: &Lutum,
    cmd: &str,
    last: Option<Result<(), &'static str>>,
) -> Result<(), &'static str> {
    if let Some(previous) = last {
        previous?;
    }
    if cmd.contains("sk-") {
        Err("looks like a secret")
    } else {
        Ok(())
    }
}

#[lutum::hooks]
struct ShellHooks {
    validate_command: ValidateCommand,
}

fn main() {
    let llm = Lutum::with_hooks(
        Arc::new(MockLlmAdapter::new()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        LutumHooks::new(),
    );
    let mut hooks = ShellHooks::new().with_validate_command(RejectSecrets);
    hooks.register_validate_command(RejectSecrets);

    futures::executor::block_on(async {
        hooks.validate_command(&llm, "echo hello").await.unwrap();
    });
}
