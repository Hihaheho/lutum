#[lutum::hooks]
trait HookSet {
    #[hook(singleton, chain = lutum::ShortCircuit<String, String>)]
    async fn select_label(label: &str) -> String {
        label.to_string()
    }
}

fn main() {}
