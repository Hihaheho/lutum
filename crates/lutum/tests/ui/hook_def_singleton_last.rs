#[lutum::def_hook(singleton)]
async fn select_label(_ctx: &lutum::Lutum, default: String, last: Option<String>) -> String {
    let _ = last;
    default
}

fn main() {}
