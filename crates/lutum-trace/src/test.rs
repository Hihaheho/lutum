pub async fn collect<F, T>(future: F) -> crate::Collected<T>
where
    F: std::future::Future<Output = T>,
{
    use tracing::instrument::WithSubscriber as _;
    use tracing_subscriber::layer::SubscriberExt as _;

    let subscriber = tracing_subscriber::registry().with(crate::layer());
    crate::capture(future.with_subscriber(subscriber)).await
}
