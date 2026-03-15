use agents::{
    FinishReason, Marker, MockLlmAdapter, MockTextScenario, NoTools, Session,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, TextStepOutcome, TextTurn, TurnConfig, Usage,
    UsageEstimate,
};

#[derive(Clone, Debug)]
struct AppMarker;

impl Marker for AppMarker {
    fn span_name(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("text_session")
    }
}

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
    let ctx = agents::Context::<AppMarker, _, _>::new(budget, adapter);
    let mut session = Session::new(ctx, AppMarker);
    session.push_system("You are concise.");
    session.push_user("Reply in two words.");

    let outcome = session
        .prepare_text(
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
