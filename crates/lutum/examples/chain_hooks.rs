use std::sync::Arc;

use futures::executor::block_on;
use lutum::*;

#[def_hook(always, chain = lutum::short_circuit)]
async fn validate_output(_ctx: &Lutum, output: &str) -> Result<(), String> {
    if output.trim().is_empty() {
        Err("output must not be empty".into())
    } else {
        println!("[default validator] output looks non-empty");
        Ok(())
    }
}

#[hook(ValidateOutput)]
async fn block_dangerous_commands(_ctx: &Lutum, output: &str) -> Result<(), String> {
    if output.contains("rm -rf") {
        Err("blocked dangerous command".into())
    } else {
        println!("[command validator] no dangerous command detected");
        Ok(())
    }
}

#[hook(ValidateOutput)]
async fn require_shell_shape(_ctx: &Lutum, output: &str) -> Result<(), String> {
    if output.contains('|') || output.contains("find ") || output.contains("ls ") {
        println!("[shape validator] output looks like a shell command");
        Ok(())
    } else {
        Err("expected something that looks like a shell command".into())
    }
}

#[def_hook(fallback, chain = lutum::first_success)]
async fn choose_rewrite(_ctx: &Lutum, prompt: &str) -> Option<String> {
    Some(format!("Default rewrite: {prompt}"))
}

#[hook(ChooseRewrite)]
async fn choose_shell_rewrite(_ctx: &Lutum, prompt: &str) -> Option<String> {
    if prompt.contains("shell") || prompt.contains("command") {
        Some("Output only a shell command and nothing else.".into())
    } else {
        None
    }
}

#[hook(ChooseRewrite)]
async fn choose_sql_rewrite(_ctx: &Lutum, prompt: &str) -> Option<String> {
    if prompt.contains("sql") || prompt.contains("query") {
        Some("Output only a SQL query and nothing else.".into())
    } else {
        None
    }
}

#[hooks]
struct DemoHooks {
    output_validators: ValidateOutput,
    rewrite_chooser: ChooseRewrite,
}

fn test_context() -> Lutum {
    Lutum::with_hooks(
        Arc::new(MockLlmAdapter::new()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        HookRegistry::new(),
    )
}

fn main() {
    block_on(async {
        let ctx = test_context();

        let hooks = DemoHooks::new()
            .with_validate_output(BlockDangerousCommands)
            .with_validate_output(RequireShellShape)
            .with_choose_rewrite(ChooseShellRewrite)
            .with_choose_rewrite(ChooseSqlRewrite);

        let prompt = "Write a shell command to list all Rust files recursively.";
        let rewrite = hooks
            .choose_rewrite(&ctx, prompt)
            .await
            .expect("fallback chain should always return a rewrite");

        println!("Chosen rewrite: {rewrite}");

        let safe_output = "find . -name '*.rs'";
        println!("\nValidating safe output: {safe_output}");
        match hooks.validate_output(&ctx, safe_output).await {
            Ok(()) => println!("safe output accepted"),
            Err(err) => println!("safe output rejected: {err}"),
        }

        let dangerous_output = "rm -rf .";
        println!("\nValidating dangerous output: {dangerous_output}");
        match hooks.validate_output(&ctx, dangerous_output).await {
            Ok(()) => println!("dangerous output accepted"),
            Err(err) => {
                println!("dangerous output rejected: {err}");
                println!("`short_circuit` stopped the chain at the first error.");
            }
        }

        println!("\nNotes:");
        println!("- `always, chain = lutum::short_circuit` runs the default first.");
        println!("- Each hook sees only the original arguments, not `last`.");
        println!("- `short_circuit` stops dispatch on the first `Err(_)`.");
        println!("- `fallback, chain = lutum::first_success` tries hooks in order.");
        println!("- `first_success` stops dispatch on the first `Some(_)`.");
    });
}
