#[lutum::hooks]
trait AppHooks {
    #[hook(singleton)]
    async fn select_label(label: String) -> String {
        label
    }
}

struct Policy;

#[lutum::impl_hooks(AppHooksSet)]
impl AppHooks for Policy {
    async fn missing_hook(&self) -> String {
        "missing".into()
    }
}

fn main() {}
