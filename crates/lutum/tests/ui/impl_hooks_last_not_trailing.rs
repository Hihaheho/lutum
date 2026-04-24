#[lutum::hooks]
trait AppHooks {
    #[hook(always)]
    async fn validate_label(label: &str) -> Result<(), String> {
        if label.is_empty() {
            Err("empty".into())
        } else {
            Ok(())
        }
    }
}

struct Policy;

#[lutum::impl_hooks(AppHooksSet)]
impl AppHooks for Policy {
    async fn validate_label(
        &self,
        last: Option<Result<(), String>>,
        label: &str,
    ) -> Result<(), String> {
        last.unwrap()?;
        if label == "blocked" {
            Err("blocked".into())
        } else {
            Ok(())
        }
    }
}

fn main() {}
