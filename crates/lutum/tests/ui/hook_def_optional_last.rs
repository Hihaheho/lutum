use lutum::{Lutum, HookRegistry};

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

fn main() {
    let _ = HookRegistry::new()
        .register_validate_prompt(RejectSecrets)
        .register_choose_label(UseRegisteredLabel);
}
