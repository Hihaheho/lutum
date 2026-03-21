use agents::{
    Context, FinishReason, InputMessageRole, MockLlmAdapter, MockTextScenario, ModelInput,
    ModelInputItem, NoTools, RequestExtensions, SharedPoolBudgetManager, SharedPoolBudgetOptions,
    TextTurn, TurnConfig, Usage, UsageEstimate,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(agents::RawTextTurnEvent::Started {
            request_id: Some("req-direct".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(agents::RawTextTurnEvent::TextDelta {
            delta: "direct control".into(),
        }),
        Ok(agents::RawTextTurnEvent::Completed {
            request_id: Some("req-direct".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 6,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = Context::new(budget, adapter);
    let input = ModelInput::from_items(vec![ModelInputItem::text(
        InputMessageRole::User,
        "Run without session helpers.",
    )]);
    let extensions = RequestExtensions::new();
    let result = ctx
        .responses_text(
            extensions,
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
