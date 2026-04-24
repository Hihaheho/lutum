#[lutum::hooks]
trait AppHooks {
    #[hook(singleton)]
    async fn select_label(label: String) -> String {
        label
    }
}

#[lutum::hooks]
trait OtherHooks {
    #[hook(singleton)]
    async fn other_label(label: String) -> String {
        label
    }
}

struct Policy;

#[lutum::impl_hooks(OtherHooksSet)]
impl AppHooks for Policy {
    async fn select_label(&self, label: String) -> String {
        label
    }
}

fn main() {
    let mut hooks = AppHooksSet::new();
    hooks.register_hooks(Policy);
}
