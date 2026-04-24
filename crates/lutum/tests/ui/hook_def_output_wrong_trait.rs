#[derive(Debug, Eq, PartialEq)]
struct Count(usize);

#[derive(Default)]
struct WrongAggregate;

impl lutum::Aggregate<String> for WrongAggregate {
    async fn call(&self, outputs: Vec<String>) -> String {
        outputs.join(",")
    }
}

#[lutum::hooks]
trait HookSet {
    #[hook(always, aggregate = WrongAggregate, output = Count)]
    async fn aggregate_label(label: &str) -> String {
        format!("default:{label}")
    }
}

fn expect_count_future<F: std::future::Future<Output = Count>>(_: F) {}

fn main() {
    let hooks = HookSetSet::new();
    expect_count_future(hooks.aggregate_label("x"));
}
