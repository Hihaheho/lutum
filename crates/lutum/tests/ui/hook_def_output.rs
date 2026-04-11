#[derive(Debug, Eq, PartialEq)]
struct Count(usize);

#[derive(Default)]
struct CountOutputs;

#[async_trait::async_trait]
impl lutum::AggregateInto<String, Count> for CountOutputs {
    async fn call(&self, outputs: Vec<String>) -> Count {
        Count(outputs.len())
    }
}

#[derive(Default)]
struct CountChars;

#[async_trait::async_trait]
impl lutum::FinalizeInto<String, Count> for CountChars {
    async fn call(&self, output: String) -> Count {
        Count(output.len())
    }
}

#[lutum::hooks]
trait HookSet {
    #[hook(always, aggregate = CountOutputs, output = Count)]
    async fn aggregate_label(label: &str) -> String {
        format!("default:{label}")
    }

    #[hook(always, finalize = CountChars, output = Count)]
    async fn finalize_label(label: &str) -> String {
        format!("default:{label}")
    }
}

#[lutum::impl_hook(AggregateLabel)]
async fn aggregate_more(label: &str) -> String {
    format!("hook:{label}")
}

#[lutum::impl_hook(FinalizeLabel)]
async fn finalize_more(label: &str, last: Option<String>) -> String {
    format!("{}:{label}", last.unwrap())
}

fn expect_count_future<F: std::future::Future<Output = Count>>(_: F) {}

#[allow(unreachable_code, unused_variables)]
fn assert_types() {
    let hooks = HookSet::new()
        .with_aggregate_label(AggregateMore)
        .with_finalize_label(FinalizeMore);
    expect_count_future(hooks.aggregate_label("x"));
    expect_count_future(hooks.finalize_label("x"));
}

fn main() {}
