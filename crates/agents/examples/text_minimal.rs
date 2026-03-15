use agents::{
    Context, FinishReason, InputMessageRole, Marker, MockLlmAdapter, MockTextScenario, ModelInput,
    ModelInputItem, NoTools, SharedPoolBudgetManager, SharedPoolBudgetOptions, TextTurn,
    TurnConfig, Usage, UsageEstimate,
};

#[derive(Clone, Debug)]
struct AppMarker;

impl Marker for AppMarker {
    fn span_name(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("text_minimal")
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(agents::RawTextTurnEvent::Started {
            request_id: Some("req-minimal".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(agents::RawTextTurnEvent::TextDelta {
            delta: "hello".into(),
        }),
        Ok(agents::RawTextTurnEvent::Completed {
            request_id: Some("req-minimal".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 4,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx: Context<AppMarker, _, _> = Context::new(budget, adapter);
    let input = ModelInput::from_items(vec![ModelInputItem::text(
        InputMessageRole::User,
        "Say hello.",
    )]);
    let result = ctx
        .responses_text(
            AppMarker,
            input,
            TextTurn::<NoTools> {
                config: TurnConfig::<NoTools>::new(agents::ModelName::new("gpt-4.1-mini")?),
            },
            UsageEstimate::zero(),
        )
        .await?
        .collect_noop()
        .await?;

    println!("{}", result.assistant_text());
    Ok(())
}
