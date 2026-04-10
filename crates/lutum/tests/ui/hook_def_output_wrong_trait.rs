#[derive(Debug, Eq, PartialEq)]
struct Count(usize);

#[derive(Default)]
struct WrongAggregate;

#[async_trait::async_trait]
impl lutum::Aggregate<String> for WrongAggregate {
    async fn call(&self, outputs: Vec<String>) -> String {
        outputs.join(",")
    }
}

#[lutum::def_hook(always, aggregate = WrongAggregate, output = Count)]
async fn aggregate_label(label: &str) -> String {
    format!("default:{label}")
}

#[lutum::hooks]
struct HookSet {
    aggregate_labels: AggregateLabel,
}

fn expect_count_future<F: std::future::Future<Output = Count>>(_: F) {}

fn main() {
    let hooks = HookSet::new();
    expect_count_future(hooks.aggregate_labels("x"));
}
