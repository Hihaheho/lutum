use std::{env, error::Error};

use agents::{
    Context, InputMessageRole, ModelInput, ModelInputItem, NoTools, OpenAiAdapter,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, TextTurnRequest, UsageEstimate,
};

#[derive(Clone, Debug)]
struct GreetingMarker;

impl agents::Marker for GreetingMarker {
    fn span_name(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("greeting")
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let prompt = env::args()
        .nth(1)
        .unwrap_or_else(|| "Say hello in one short sentence.".to_string());
    let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "qwen3.5:latest".to_string());
    let base_url =
        env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
    let api_key = env::var("OPENAI_API_KEY").unwrap_or_default();

    let adapter = OpenAiAdapter::new(api_key).with_base_url(base_url);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx: Context<GreetingMarker, _, _> = Context::new(budget, adapter);

    let input = ModelInput::from_items(vec![
        ModelInputItem::text(
            InputMessageRole::System,
            "You are a concise assistant. Answer in one short paragraph.",
        ),
        ModelInputItem::text(InputMessageRole::User, prompt),
    ]);

    let pending = ctx
        .responses_text(
            GreetingMarker,
            input,
            TextTurnRequest::<NoTools>::new(model),
            UsageEstimate::zero(),
        )
        .await?;
    let result = pending.collect_noop().await?;

    println!("{}", result.assistant_text());
    eprintln!("finish_reason: {:?}", result.finish_reason);
    eprintln!("tokens: {}", result.usage.total_tokens);
    Ok(())
}
