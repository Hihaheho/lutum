#[lutum::def_hook(singleton)]
async fn select_label(_ctx: &lutum::Context, default: String, last: Option<String>) -> String {
    let _ = last;
    default
}

fn main() {}
