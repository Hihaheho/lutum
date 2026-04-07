#[lutum::def_hook(singleton)]
async fn select_label(_ctx: &lutum::Lutum, label: &str) -> String {
    label.to_string()
}

#[lutum::hook(SelectLabel)]
async fn append_suffix(_ctx: &lutum::Lutum, label: &str, last: Option<String>) -> String {
    let _ = last;
    format!("{label}:hook")
}

fn main() {}
