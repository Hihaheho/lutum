#[lutum::def_hook(always, chain = lutum::short_circuit)]
async fn validate_output(_ctx: &lutum::Lutum, output: &str) -> Result<String, String> {
    Ok(output.to_string())
}

struct StatefulAppender;

#[async_trait::async_trait]
impl StatefulValidateOutput for StatefulAppender {
    async fn call_mut(
        &mut self,
        _ctx: &lutum::Lutum,
        output: String,
        last: Option<Result<String, String>>,
    ) -> Result<String, String> {
        let _ = last;
        Ok(format!("{output}:hook"))
    }
}

fn main() {}
