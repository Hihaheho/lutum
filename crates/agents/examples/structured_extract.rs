use agents::{
    FinishReason, MockLlmAdapter, MockStructuredScenario, NoTools, RequestExtensions, Session,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, StructuredStepOutcome, StructuredTurn, Usage,
    UsageEstimate,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct Contact {
    email: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter =
        MockLlmAdapter::new().with_structured_scenario(MockStructuredScenario::events(vec![
            Ok(agents::RawStructuredTurnEvent::Started {
                request_id: Some("req-structured".into()),
                model: "gpt-4.1-mini".into(),
            }),
            Ok(agents::RawStructuredTurnEvent::StructuredOutputChunk {
                json_delta: "{\"email\":\"user@example.com\"}".into(),
            }),
            Ok(agents::RawStructuredTurnEvent::Completed {
                request_id: Some("req-structured".into()),
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    total_tokens: 8,
                    ..Usage::zero()
                },
            }),
        ]));
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx = agents::Context::new(budget, adapter);
    let mut session = Session::new(ctx);
    session.push_user("Extract the email address.");

    let outcome = session
        .prepare_structured(
            RequestExtensions::new(),
            StructuredTurn::<NoTools, Contact>::new(agents::ModelName::new("gpt-4.1-mini")?),
            UsageEstimate::zero(),
        )
        .await?
        .collect_noop()
        .await?;

    match outcome {
        StructuredStepOutcome::Finished(result) => {
            println!("{:?}", result.semantic);
            session.commit_structured(result);
        }
        StructuredStepOutcome::NeedsToolResults(_) => unreachable!(),
    }

    Ok(())
}
