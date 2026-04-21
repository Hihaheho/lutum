fn ensure_test_subscriber() {
    use std::sync::OnceLock;
    use tracing_subscriber::layer::SubscriberExt as _;

    static TEST_SUBSCRIBER: OnceLock<()> = OnceLock::new();
    TEST_SUBSCRIBER.get_or_init(|| {
        let subscriber = tracing_subscriber::registry().with(crate::layer());
        tracing::subscriber::set_global_default(subscriber)
            .expect("test capture layer global subscriber should install once");
    });
}

pub async fn collect<F, T>(future: F) -> crate::Collected<T>
where
    F: std::future::Future<Output = T>,
{
    ensure_test_subscriber();
    crate::capture(future).await
}

pub async fn collect_raw<F, T>(future: F) -> crate::CollectedRaw<T>
where
    F: std::future::Future<Output = T>,
{
    ensure_test_subscriber();
    crate::capture_raw(future).await
}
