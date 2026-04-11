#[lutum::hooks]
trait HookSet {
    #[hook(singleton)]
    async fn select_label(label: &str, last: Option<String>) -> String {
        last.unwrap_or_else(|| label.to_string())
    }
}

fn main() {}
