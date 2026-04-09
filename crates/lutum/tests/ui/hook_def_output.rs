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

#[lutum::def_hook(always, aggregate = CountOutputs, output = Count)]
async fn aggregate_label(_ctx: &lutum::Lutum, label: &str) -> String {
    format!("default:{label}")
}

#[lutum::hook(AggregateLabel)]
async fn aggregate_more(_ctx: &lutum::Lutum, label: &str) -> String {
    format!("hook:{label}")
}

#[lutum::def_hook(always, finalize = CountChars, output = Count)]
async fn finalize_label(_ctx: &lutum::Lutum, label: &str) -> String {
    format!("default:{label}")
}

#[lutum::hook(FinalizeLabel)]
async fn finalize_more(_ctx: &lutum::Lutum, label: &str, last: Option<String>) -> String {
    format!("{}:{label}", last.unwrap())
}

#[lutum::def_global_hook(always, aggregate = CountOutputs, output = Count)]
async fn global_aggregate_label(_ctx: &lutum::Lutum, label: &str) -> String {
    format!("default:{label}")
}

#[lutum::hook(GlobalAggregateLabel)]
async fn global_aggregate_more(_ctx: &lutum::Lutum, label: &str) -> String {
    format!("hook:{label}")
}

#[lutum::hooks]
struct HookSet {
    aggregate_labels: AggregateLabel,
    finalize_labels: FinalizeLabel,
}

fn expect_count_future<F: std::future::Future<Output = Count>>(_: F) {}

#[allow(unreachable_code, unused_variables)]
fn assert_types() {
    let hooks = HookSet::new()
        .with_aggregate_label(AggregateMore)
        .with_finalize_label(FinalizeMore);
    let ctx: &lutum::Lutum = unimplemented!();
    expect_count_future(hooks.aggregate_label(ctx, "x"));
    expect_count_future(hooks.finalize_label(ctx, "x"));

    let registry = lutum::HookRegistry::new().register_global_aggregate_label(GlobalAggregateMore);
    expect_count_future(
        <lutum::HookRegistry as GlobalAggregateLabelRegistryExt>::global_aggregate_label(
            &registry,
            ctx,
            "x",
        ),
    );
}

fn main() {}
