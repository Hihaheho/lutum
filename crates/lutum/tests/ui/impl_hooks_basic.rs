use futures::executor::block_on;

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

    #[hook(fallback)]
    async fn choose_label(label: &str) -> String {
        format!("default:{label}")
    }

    #[hook(singleton)]
    async fn select_label(label: String) -> String {
        label
    }
}

struct Policy<'a> {
    prefix: &'a str,
}

#[lutum::impl_hooks(AppHooksSet)]
impl<'a> AppHooks for Policy<'a> {
    async fn validate_label(
        &self,
        label: &str,
        last: Option<Result<(), String>>,
    ) -> Result<(), String> {
        last.unwrap()?;
        if label == "blocked" {
            Err("blocked".into())
        } else {
            Ok(())
        }
    }

    async fn choose_label(&self, label: &str, last: Option<String>) -> String {
        assert!(last.is_none());
        format!("{}:{label}", self.prefix)
    }
}

fn main() {
    let prefix = String::from("hook");
    let policy = Policy { prefix: &prefix };
    let hooks = AppHooksSet::new().with_hooks(&policy);

    block_on(async {
        hooks.validate_label("ok").await.unwrap();
        assert_eq!(hooks.choose_label("x").await, "hook:x");
        assert_eq!(hooks.select_label("plain".to_owned()).await, "plain");
    });
}
