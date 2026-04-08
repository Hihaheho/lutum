#[lutum::def_hook(always, chain = lutum::ShortCircuit<String, String>)]
async fn validate_output(_ctx: &lutum::Lutum, output: &str) -> Result<String, String> {
    Ok(format!("default:{output}"))
}

#[lutum::hook(ValidateOutput)]
async fn append_suffix(_ctx: &lutum::Lutum, output: &str) -> Result<String, String> {
    Ok(format!("{output}:hook"))
}

mod custom {
    pub struct PreferFirstSome;

    impl Default for PreferFirstSome {
        fn default() -> Self {
            Self
        }
    }

    #[async_trait::async_trait]
    impl lutum::Chain<Option<String>> for PreferFirstSome {
        async fn call(&self, value: &Option<String>) -> std::ops::ControlFlow<()> {
            match value {
                Some(_) => std::ops::ControlFlow::Break(()),
                None => std::ops::ControlFlow::Continue(()),
            }
        }
    }
}

#[lutum::def_hook(fallback, chain = custom::PreferFirstSome)]
async fn choose_label(_ctx: &lutum::Lutum, label: &str) -> Option<String> {
    Some(format!("default:{label}"))
}

#[lutum::hook(ChooseLabel)]
async fn choose_special(_ctx: &lutum::Lutum, label: &str) -> Option<String> {
    Some(format!("special:{label}"))
}

#[lutum::hooks]
struct HookSet {
    validators: ValidateOutput,
    choosers: ChooseLabel,
}

fn main() {
    let _ = HookSet::new()
        .with_validate_output(AppendSuffix)
        .with_choose_label(ChooseSpecial);
}
