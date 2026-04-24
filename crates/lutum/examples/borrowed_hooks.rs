use futures::executor::block_on;

type Validation = Result<(), String>;

#[lutum::hooks]
trait PromptHooks {
    #[hook(always)]
    async fn validate_prompt(prompt: &str) -> Validation {
        if prompt.trim().is_empty() {
            Err("prompt is empty".into())
        } else {
            Ok(())
        }
    }

    #[hook(singleton)]
    async fn decorate_prompt(prompt: &str) -> String {
        prompt.to_owned()
    }
}

struct PromptPolicy<'a> {
    tenant: &'a str,
    banned_words: &'a [String],
}

#[lutum::impl_hooks(PromptHooksSet)]
impl<'a> PromptHooks for PromptPolicy<'a> {
    async fn validate_prompt(&self, prompt: &str, last: Option<Validation>) -> Validation {
        if let Some(Err(err)) = last {
            return Err(err);
        }

        if let Some(word) = self
            .banned_words
            .iter()
            .find(|word| prompt.contains(word.as_str()))
        {
            Err(format!("blocked word for {}: {word}", self.tenant))
        } else {
            Ok(())
        }
    }

    async fn decorate_prompt(&self, prompt: &str) -> String {
        format!("[tenant:{}] {}", self.tenant, prompt.trim())
    }
}

fn main() {
    let tenant = String::from("acme");
    let banned_words = vec![String::from("secret"), String::from("password")];
    let policy = PromptPolicy {
        tenant: &tenant,
        banned_words: &banned_words,
    };

    let mut hooks = PromptHooksSet::new();
    hooks.register_hooks(&policy);

    // `hooks` borrows `policy`, and `policy` borrows the local strings above.
    // No hook data needs to be cloned or promoted to `'static`.
    block_on(async {
        hooks
            .validate_prompt("summarize public docs")
            .await
            .unwrap();

        let err = hooks
            .validate_prompt("summarize the secret")
            .await
            .unwrap_err();
        assert_eq!(err, "blocked word for acme: secret");

        let prompt = hooks.decorate_prompt("  build a migration plan  ").await;
        assert_eq!(prompt, "[tenant:acme] build a migration plan");
    });
}
