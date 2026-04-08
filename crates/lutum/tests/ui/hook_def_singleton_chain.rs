#[lutum::def_hook(singleton, chain = lutum::ShortCircuit<String, String>)]
async fn select_label(_ctx: &lutum::Lutum, label: &str) -> String {
    label.to_string()
}

fn main() {}
