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
    #[lutum::hooks]
    pub trait SlotHooks {
        #[hook(always, aggregate = crate::CountOutputs, output = crate::Count)]
        async fn summarize_label(label: &str) -> String {
            format!("default:{label}")
        }
    }
}

#[lutum::impl_hook(slots::SummarizeLabel)]
async fn summarize_more(label: &str) -> String {
    format!("hook:{label}")
}

fn expect_count_future<F: std::future::Future<Output = Count>>(_: F) {}

#[allow(unreachable_code, unused_variables)]
fn assert_types() {
    let hooks = slots::SlotHooks::new().with_summarize_label(SummarizeMore);
    expect_count_future(hooks.summarize_label("x"));
}

fn main() {}
