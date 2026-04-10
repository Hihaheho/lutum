pub async fn collect<F, T>(future: F) -> crate::Collected<T>
where
    F: std::future::Future<Output = T>,
{
    use std::sync::OnceLock;
    use tracing_subscriber::layer::SubscriberExt as _;

    static TEST_SUBSCRIBER: OnceLock<()> = OnceLock::new();
    TEST_SUBSCRIBER.get_or_init(|| {
        let subscriber = tracing_subscriber::registry().with(crate::layer());
        tracing::subscriber::set_global_default(subscriber)
            .expect("test capture layer global subscriber should install once");
    });

    crate::capture(future).await
}
