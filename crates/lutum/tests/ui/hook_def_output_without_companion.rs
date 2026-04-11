#[lutum::hooks]
trait InvalidHooks {
    #[hook(always, output = usize)]
    async fn invalid_output_without_companion(label: &str) -> String {
        label.to_owned()
    }
}

fn main() {}
