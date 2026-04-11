#[lutum::hooks]
trait ShellHooks {
    #[hook(fallback)]
    async fn validate_command(cmd: &str) -> Result<(), &'static str>;
}

fn main() {}
