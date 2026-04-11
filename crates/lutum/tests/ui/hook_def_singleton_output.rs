#[lutum::hooks]
trait HookSet {
    #[hook(singleton, output = usize)]
    async fn select_label(label: &str) -> String {
        label.to_string()
    }
}

fn main() {}
