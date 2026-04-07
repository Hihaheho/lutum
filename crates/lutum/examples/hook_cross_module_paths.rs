use lutum::*;

mod slots {
    use lutum::*;

    #[def_hook(always, chain = lutum::short_circuit)]
    pub async fn validate_output(_ctx: &Lutum, output: &str) -> Result<String, String> {
        Ok(format!("default:{output}"))
    }
}

#[hook(slots::ValidateOutput)]
async fn append_suffix(_ctx: &Lutum, output: &str) -> Result<String, String> {
    Ok(format!("{output}:hook"))
}

// This shorthand is intentionally *not* used here:
//
// use slots::ValidateOutput;
//
// #[hook(ValidateOutput)]
// async fn append_suffix(_ctx: &Lutum, output: &str) -> Result<String, String> {
//     Ok(format!("{output}:hook"))
// }
//
// Importing only the slot type is not enough for cross-module `#[hook(...)]` expansion.
// The supported pattern is to use the full slot path, like `#[hook(slots::ValidateOutput)]`.

fn main() {
    let _ = AppendSuffix;
}
