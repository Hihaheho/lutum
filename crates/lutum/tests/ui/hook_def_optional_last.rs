use std::sync::Arc;

use lutum::{
    Lutum, LutumHooks, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions,
};

#[lutum::hooks]
trait LocalHooks {
    #[hook(always)]
    async fn validate_prompt(prompt: &str) -> Result<(), String> {
        if prompt.trim().is_empty() {
            Err("empty prompt".into())
        } else {
            Ok(())
        }
    }

    #[hook(fallback)]
    async fn choose_label(label: &str) -> String {
        format!("default:{label}")
    }

    #[hook(singleton)]
    async fn select_label(previous: Option<String>) -> String {
        previous.unwrap_or_else(|| "default".into())
    }
}

#[lutum::impl_hook(ValidatePrompt)]
async fn reject_secrets(prompt: &str, last: Option<Result<(), String>>) -> Result<(), String> {
    if let Some(previous) = last {
        previous?;
    }
    if prompt.contains("sk-") {
        Err("looks like a secret".into())
    } else {
        Ok(())
    }
}

#[lutum::impl_hook(ChooseLabel)]
async fn use_registered_label(label: &str, last: Option<String>) -> String {
    assert!(last.is_none());
    format!("hook:{label}")
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
        hooks.validate_prompt("hello").await.unwrap();
        let _ = hooks.choose_label("base").await;
        let _ = hooks.select_label(Some(String::from("picked"))).await;
        let _ = llm.resolve_usage_estimate(&lutum::RequestExtensions::new(), lutum::OperationKind::TextTurn).await;
    });

    hooks.register_choose_label(UseRegisteredLabel);
}
