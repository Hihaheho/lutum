#[lutum::hooks]
trait HookSet {
    #[hook(always, chain = lutum::ShortCircuit<String, String>)]
    async fn validate_output(output: &str) -> Result<String, String> {
        Ok(format!("default:{output}"))
    }

    #[hook(fallback, chain = custom::PreferFirstSome)]
    async fn choose_label(label: &str) -> Option<String> {
        Some(format!("default:{label}"))
    }
}

#[lutum::impl_hook(ValidateOutput)]
async fn append_suffix(output: &str) -> Result<String, String> {
    Ok(format!("{output}:hook"))
}

mod custom {
    pub struct PreferFirstSome;

    impl Default for PreferFirstSome {
        fn default() -> Self {
            Self
        }
    }

    impl lutum::Chain<Option<String>> for PreferFirstSome {
        async fn call<'a>(&'a self, value: &'a Option<String>) -> std::ops::ControlFlow<()> {
            match value {
                Some(_) => std::ops::ControlFlow::Break(()),
                None => std::ops::ControlFlow::Continue(()),
            }
        }
    }
}

#[lutum::impl_hook(ChooseLabel)]
async fn choose_special(label: &str) -> Option<String> {
    Some(format!("special:{label}"))
}

fn main() {
    let _ = HookSetSet::new()
        .with_validate_output(AppendSuffix)
        .with_choose_label(ChooseSpecial);
}
