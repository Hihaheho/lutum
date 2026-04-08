mod slots {
    #[lutum::def_hook(always, chain = lutum::short_circuit)]
    pub async fn validate_output(_ctx: &lutum::Lutum, output: &str) -> Result<String, String> {
        Ok(format!("default:{output}"))
    }
}

#[lutum::hook(slots::ValidateOutput)]
async fn append_suffix(_ctx: &lutum::Lutum, output: &str) -> Result<String, String> {
    Ok(format!("{output}:hook"))
}

fn main() {
    let _ = AppendSuffix;
}
