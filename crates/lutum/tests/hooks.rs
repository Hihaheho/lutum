use std::sync::Arc;

use futures::executor::block_on;
use lutum::{
    Context, HookRegistry, MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions,
};
use lutum_trace::FieldValue;

#[lutum::def_hook(singleton)]
async fn select_label(_ctx: &Context, default: String) -> String {
    default
}

#[lutum::hook(SelectLabel)]
async fn prefix_label(_ctx: &Context, default: String) -> String {
    format!("hooked:{default}")
}

#[lutum::hook(SelectLabel)]
async fn suffix_label(_ctx: &Context, default: String) -> String {
    format!("{default}:suffix")
}

fn test_context(hooks: HookRegistry) -> Context {
    Context::with_hooks(
        Arc::new(MockLlmAdapter::new()),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
        hooks,
    )
}

#[test]
fn singleton_hook_uses_default_when_unregistered() {
    let ctx = test_context(HookRegistry::new());

    let selected = block_on(ctx.select_label("base".into()));

    assert_eq!(selected, "base");
}

#[test]
fn singleton_hook_uses_registered_override() {
    let ctx = test_context(HookRegistry::new().register_select_label(PrefixLabel));

    let selected = block_on(ctx.select_label("base".into()));

    assert_eq!(selected, "hooked:base");
}

#[test]
fn singleton_hook_warns_and_uses_last_registered_override() {
    let collected = block_on(lutum_trace::test::collect(async {
        let ctx = test_context(
            HookRegistry::new()
                .register_select_label(PrefixLabel)
                .register_select_label(SuffixLabel),
        );

        ctx.select_label("base".into()).await
    }));

    assert_eq!(collected.output, "base:suffix");

    let warning = collected
        .trace
        .events()
        .iter()
        .find(|event| {
            event.level == "WARN"
                && event.message()
                    == Some("singleton hook registration overwritten; last registered hook wins")
        })
        .expect("expected singleton overwrite warning");

    assert_eq!(
        warning.field("slot"),
        Some(&FieldValue::Str("select_label".to_string()))
    );
}
