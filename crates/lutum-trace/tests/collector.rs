use tracing::{Instrument, field};
use tracing_subscriber::layer::SubscriberExt as _;

use lutum_trace::FieldValue;

#[tokio::test]
async fn basic_tree() {
    let collected = lutum_trace::test::collect(async {
        let root = tracing::info_span!(target: "lutum", "root", answer = 42_u64);
        let _root = root.enter();

        let child = tracing::info_span!(target: "lutum", "child", ok = true);
        let _child = child.enter();

        tracing::info!(target: "lutum", kind = "note", "child event");
    })
    .await;

    let root = collected.trace.span("root").expect("root span");
    assert_eq!(root.field("answer"), Some(&FieldValue::U64(42)));

    let child = root.child("child").expect("child span");
    assert_eq!(child.field("ok"), Some(&FieldValue::Bool(true)));
    assert_eq!(
        child
            .event("child event")
            .and_then(|event| event.field("kind")),
        Some(&FieldValue::Str("note".to_string()))
    );
}

#[tokio::test]
async fn late_record() {
    let collected = lutum_trace::test::collect(async {
        let span = tracing::info_span!(
            target: "lutum",
            "late_record",
            request_id = field::Empty,
            finish_reason = field::Empty
        );
        let _guard = span.enter();
        span.record("request_id", field::display("req-123"));
    })
    .await;

    let span = collected
        .trace
        .span("late_record")
        .expect("late_record span");
    assert_eq!(
        span.field("request_id"),
        Some(&FieldValue::Str("req-123".to_string()))
    );
    assert_eq!(span.field("finish_reason"), None);
}

#[tokio::test]
async fn contextual_event_under_instrument() {
    let collected = lutum_trace::test::collect(async {
        async {
            tokio::task::yield_now().await;
            tracing::info!(target: "lutum", "inside instrumented future");
        }
        .instrument(tracing::info_span!(target: "lutum", "instrumented"))
        .await;
    })
    .await;

    let span = collected
        .trace
        .span("instrumented")
        .expect("instrumented span");
    assert!(span.event("inside instrumented future").is_some());
}

#[tokio::test]
async fn event_outside_span() {
    let collected = lutum_trace::test::collect(async {
        tracing::info!(target: "lutum", "outside");
    })
    .await;

    assert_eq!(collected.trace.events().len(), 1);
    assert!(collected.trace.has_event_message("outside"));
}

#[tokio::test]
async fn no_scope_no_panic() {
    let dispatch =
        tracing::Dispatch::new(tracing_subscriber::registry().with(lutum_trace::layer()));

    tracing::dispatcher::with_default(&dispatch, || {
        let span = tracing::info_span!(target: "lutum", "outside_scope", request_id = field::Empty);
        let _guard = span.enter();
        tracing::info!(target: "lutum", "still fine");
        span.record("request_id", field::display("req-outside"));
    });
}

#[tokio::test]
async fn span_id_reuse() {
    let collected = lutum_trace::test::collect(async {
        {
            let span = tracing::info_span!(target: "lutum", "reused", iteration = 1_u64);
            let _guard = span.enter();
            tracing::info!(target: "lutum", "first");
        }

        {
            let span = tracing::info_span!(target: "lutum", "reused", iteration = 2_u64);
            let _guard = span.enter();
            tracing::info!(target: "lutum", "second");
        }
    })
    .await;

    let spans = collected.trace.find_all("reused");
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].field("iteration"), Some(&FieldValue::U64(1)));
    assert_eq!(spans[1].field("iteration"), Some(&FieldValue::U64(2)));
    assert!(spans[0].event("first").is_some());
    assert!(spans[1].event("second").is_some());
}

#[tokio::test]
async fn test_collect_no_global() {
    let collected = lutum_trace::test::collect(async {
        tracing::info!(target: "lutum", "helper works");
        7_u64
    })
    .await;

    assert_eq!(collected.output, 7);
    assert!(collected.trace.has_event_message("helper works"));
}

#[tokio::test]
async fn external_parent_becomes_root() {
    use tracing::instrument::WithSubscriber as _;

    let subscriber = tracing_subscriber::registry().with(lutum_trace::layer());
    let dispatch = tracing::Dispatch::new(subscriber);

    let parent = tracing::dispatcher::with_default(
        &dispatch,
        || tracing::info_span!(target: "lutum", "external_parent"),
    );

    let collected = lutum_trace::capture(async {
        let child = tracing::info_span!(
            target: "lutum",
            parent: &parent,
            "captured_child",
            value = 99_u64
        );
        let _guard = child.enter();
        tracing::info!(target: "lutum", "captured event");
    })
    .with_subscriber(dispatch.clone())
    .await;

    assert_eq!(collected.trace.roots.len(), 1);

    let child = collected
        .trace
        .span("captured_child")
        .expect("captured child span");
    assert_eq!(child.field("value"), Some(&FieldValue::U64(99)));
    assert!(child.event("captured event").is_some());
}

#[tokio::test]
async fn lutum_capture_field_opt_in() {
    let collected = lutum_trace::test::collect(async {
        let span = tracing::info_span!("user_span", lutum.capture = true, x = 1_u64);
        let _guard = span.enter();
        tracing::info!(target: "my_app", "hello");
    })
    .await;

    let span = collected.trace.span("user_span").expect("user span");
    assert_eq!(span.field("x"), Some(&FieldValue::U64(1)));
    assert!(span.event("hello").is_some());
}

#[tokio::test]
async fn non_lutum_spans_ignored() {
    let collected = lutum_trace::test::collect(async {
        let span = tracing::info_span!("noise", foo = 1_u64);
        let _guard = span.enter();
        tracing::info!("ignored");
    })
    .await;

    assert!(collected.trace.roots.is_empty());
    assert!(collected.trace.events().is_empty());
}

#[tokio::test]
async fn cross_task_capture() {
    let collected = lutum_trace::test::collect(async {
        let parent = tracing::info_span!(target: "lutum", "parent");
        let handle = {
            let parent = parent.clone();
            tokio::spawn(
                async {
                    tracing::info!(target: "lutum", "from spawned task");
                }
                .instrument(parent),
            )
        };

        handle.await.unwrap();
    })
    .await;

    assert!(
        collected
            .trace
            .roots
            .iter()
            .any(|span| span.has_event_message("from spawned task"))
            || collected
                .trace
                .root_events
                .iter()
                .any(|event| event.message() == Some("from spawned task"))
    );
}
