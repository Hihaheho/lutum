#[lutum::hooks]
trait GenericHooks {
    #[hook(singleton)]
    async fn select_label<T: Send + Sync + 'static>(value: T) -> T {
        value
    }
}

fn main() {}
