mod slots {
    #[lutum::hooks]
    pub trait SlotHooks {
        #[hook(always, chain = lutum::ShortCircuit<String, String>)]
        async fn validate_output(output: &str) -> Result<String, String> {
            Ok(format!("default:{output}"))
        }
    }
}

#[lutum::impl_hook(slots::ValidateOutput)]
async fn append_suffix(output: &str) -> Result<String, String> {
    Ok(format!("{output}:hook"))
}

fn main() {
    let _ = AppendSuffix;
}
