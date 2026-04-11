#[lutum::hooks]
trait HookSet {
    #[hook(singleton)]
    async fn select_label(label: &str) -> String {
        label.to_string()
    }
}

#[lutum::impl_hook(SelectLabel)]
async fn append_suffix(label: &str, last: Option<String>) -> String {
    let _ = last;
    format!("{label}:hook")
}

fn main() {}
