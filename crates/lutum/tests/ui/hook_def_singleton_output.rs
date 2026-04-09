#[lutum::def_hook(singleton, output = usize)]
async fn invalid_singleton_output(label: &str) -> String {
    label.to_owned()
}

fn main() {}
