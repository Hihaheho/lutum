#[derive(Default)]
struct JoinStrings;

#[async_trait::async_trait]
impl lutum::Aggregate<String> for JoinStrings {
    async fn call(&self, outputs: Vec<String>) -> String {
        outputs.join(",")
    }
}

#[derive(Default)]
struct WrapResult;

#[async_trait::async_trait]
impl lutum::Finalize<String> for WrapResult {
    async fn call(&self, output: String) -> String {
        format!("[{output}]")
    }
}

#[lutum::def_hook(always, aggregate = JoinStrings, finalize = WrapResult)]
async fn invalid_companions(label: &str) -> String {
    label.to_owned()
}

fn main() {}
