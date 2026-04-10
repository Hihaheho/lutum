#[derive(Debug, Eq, PartialEq)]
pub struct Count(pub usize);

#[derive(Default)]
pub struct CountOutputs;

#[async_trait::async_trait]
impl lutum::AggregateInto<String, Count> for CountOutputs {
    async fn call(&self, outputs: Vec<String>) -> Count {
        Count(outputs.len())
    }
}

mod slots {
    #[lutum::def_hook(always, aggregate = crate::CountOutputs, output = crate::Count)]
    pub async fn summarize_label(_ctx: &lutum::Lutum, label: &str) -> String {
        format!("default:{label}")
    }
}

#[lutum::hook(slots::SummarizeLabel)]
async fn summarize_more(_ctx: &lutum::Lutum, label: &str) -> String {
    format!("hook:{label}")
}

#[lutum::hooks]
struct HookSet {
    summarize_label: slots::SummarizeLabel,
}

fn expect_count_future<F: std::future::Future<Output = Count>>(_: F) {}

#[allow(unreachable_code, unused_variables)]
fn assert_types() {
    let hooks = HookSet::new().with_summarize_label(SummarizeMore);
    let ctx: &lutum::Lutum = unimplemented!();
    expect_count_future(hooks.summarize_label(ctx, "x"));
}

fn main() {}
