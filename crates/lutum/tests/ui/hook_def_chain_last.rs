#[lutum::def_hook(always, chain = lutum::short_circuit)]
async fn validate_output(
    _ctx: &lutum::Lutum,
    output: &str,
    last: Option<Result<String, String>>,
) -> Result<String, String> {
    let _ = last;
    Ok(output.to_string())
}

fn main() {}
