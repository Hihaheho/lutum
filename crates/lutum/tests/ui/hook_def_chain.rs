#[lutum::def_hook(always, chain = lutum::short_circuit)]
async fn validate_output(_ctx: &lutum::Lutum, output: &str) -> Result<String, String> {
    Ok(format!("default:{output}"))
}

#[lutum::hook(ValidateOutput)]
async fn append_suffix(_ctx: &lutum::Lutum, output: &str) -> Result<String, String> {
    Ok(format!("{output}:hook"))
}

mod custom {
    pub fn prefer_first_some<T>(value: &Option<T>) -> std::ops::ControlFlow<(), ()> {
        match value {
            Some(_) => std::ops::ControlFlow::Break(()),
            None => std::ops::ControlFlow::Continue(()),
        }
    }
}

#[lutum::def_hook(fallback, chain = custom::prefer_first_some)]
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
