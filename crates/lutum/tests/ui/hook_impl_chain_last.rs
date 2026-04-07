#[lutum::def_hook(always, chain = lutum::short_circuit)]
async fn validate_output(_ctx: &lutum::Lutum, output: &str) -> Result<String, String> {
    Ok(output.to_string())
}

#[lutum::hook(ValidateOutput)]
async fn append_suffix(
    _ctx: &lutum::Lutum,
    output: &str,
    last: Option<Result<String, String>>,
) -> Result<String, String> {
    let _ = last;
    Ok(format!("{output}:hook"))
}

fn main() {}
