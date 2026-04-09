use std::sync::Arc;

use lutum::{
    Lutum, LutumHooks, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions,
};

#[lutum::def_hook(always)]
async fn validate_prompt(_ctx: &Lutum, prompt: &str) -> Result<(), String> {
    if prompt.trim().is_empty() {
        Err("empty prompt".into())
    } else {
        Ok(())
    }
}

#[lutum::hook(ValidatePrompt)]
async fn reject_secrets(
    _ctx: &Lutum,
    prompt: &str,
    last: Option<Result<(), String>>,
) -> Result<(), String> {
    if let Some(previous) = last {
        previous?;
    }
    if prompt.contains("sk-") {
        Err("looks like a secret".into())
    } else {
        Ok(())
    }
}

#[lutum::def_hook(fallback)]
async fn choose_label(_ctx: &Lutum, label: &str) -> String {
    format!("default:{label}")
}

#[lutum::hook(ChooseLabel)]
async fn use_registered_label(_ctx: &Lutum, label: &str, last: Option<String>) -> String {
    assert!(last.is_none());
    format!("hook:{label}")
}

#[lutum::def_hook(singleton)]
async fn select_label(_ctx: &Lutum, previous: Option<String>) -> String {
    previous.unwrap_or_else(|| "default".into())
}

#[lutum::hooks]
struct LocalHooks {
    prompt_slot: ValidatePrompt,
    label_slot: ChooseLabel,
    selection_slot: SelectLabel,
}

fn main() {
    let llm = Lutum::with_hooks(
        Arc::new(MockLlmAdapter::new()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        LutumHooks::new(),
    );
    let mut hooks = LocalHooks::new()
        .with_validate_prompt(RejectSecrets)
        .with_choose_label(UseRegisteredLabel);

    futures::executor::block_on(async {
        hooks.validate_prompt(&llm, "hello").await.unwrap();
        let _ = hooks.choose_label(&llm, "base").await;
        let _ = hooks.select_label(&llm, Some(String::from("picked"))).await;
    });

    hooks.register_choose_label(UseRegisteredLabel);
}
