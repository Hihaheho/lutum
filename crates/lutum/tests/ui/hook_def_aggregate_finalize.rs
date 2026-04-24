#[derive(Default)]
struct JoinStrings;

impl lutum::Aggregate<String> for JoinStrings {
    async fn call(&self, outputs: Vec<String>) -> String {
        outputs.join(",")
    }
}

#[derive(Default)]
struct WrapResult;

impl lutum::Finalize<String> for WrapResult {
    async fn call(&self, output: String) -> String {
        format!("[{output}]")
    }
}

#[lutum::hooks]
trait InvalidHooks {
    #[hook(always, aggregate = JoinStrings, finalize = WrapResult)]
    async fn invalid_companions(label: &str) -> String {
        label.to_owned()
    }
}

fn main() {}
