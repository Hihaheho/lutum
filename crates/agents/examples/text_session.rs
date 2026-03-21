use agents::{
    FinishReason, MockLlmAdapter, MockTextScenario, NoTools, RequestExtensions, Session,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, TextStepOutcome, TextTurn, TurnConfig, Usage,
    UsageEstimate,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = MockLlmAdapter::new().with_text_scenario(MockTextScenario::events(vec![
        Ok(agents::RawTextTurnEvent::Started {
            request_id: Some("req-session".into()),
            model: "gpt-4.1-mini".into(),
        }),
        Ok(agents::RawTextTurnEvent::TextDelta {
            delta: "session reply".into(),
        }),
        Ok(agents::RawTextTurnEvent::Completed {
            request_id: Some("req-session".into()),
            finish_reason: FinishReason::Stop,
            usage: Usage {
                total_tokens: 7,
                ..Usage::zero()
            },
        }),
    ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = agents::Context::new(budget, adapter);
    let mut session = Session::new(ctx);
    session.push_system("You are concise.");
    session.push_user("Reply in two words.");

    let outcome = session
        .prepare_text(
            RequestExtensions::new(),
            TextTurn::<NoTools> {
                config: TurnConfig::<NoTools>::new(agents::ModelName::new("gpt-4.1-mini")?),
            },
            UsageEstimate::zero(),
        )
        .await?
        .collect_noop()
        .await?;

    match outcome {
        TextStepOutcome::Finished(result) => {
            session.commit_text(result)?;
        }
        TextStepOutcome::NeedsToolResults(_) => unreachable!(),
    }

    println!("items={}", session.snapshot().items().len());
    Ok(())
}
