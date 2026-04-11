use lutum::{InputMessageRole, Lutum, ModelInput, ModelInputItem, StructuredTurnOutcome};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlite_agent::QueryResult;

/// Ask an LLM judge whether the assistant's text accurately reflects the SQL result.
/// Returns a score in [0.0, 1.0]: 1.0 = fully consistent, 0.0 = contradictory.
pub async fn score_consistency(
    judge: &Lutum,
    assistant_text: &str,
    query_result: &QueryResult,
) -> anyhow::Result<f32> {
    let result_str = format!("{query_result}");
    let prompt = format!(
        "You are evaluating an AI database assistant. \
         Given the SQL query result and the assistant's response, \
         score how well the response reflects the actual data.\n\n\
         SQL Result:\n{result_str}\n\n\
         Assistant Response:\n{assistant_text}\n\n\
         Respond with a JSON object containing a single field `score` \
         with a float between 0.0 (completely wrong) and 1.0 (fully accurate)."
    );

    let input = ModelInput::from_items(vec![ModelInputItem::text(
        InputMessageRole::User,
        prompt,
    )]);
    let result = judge
        .structured_turn::<ConsistencyScore>(input)
        .collect()
        .await?;

    match result.semantic {
        StructuredTurnOutcome::Structured(score) => Ok(score.score.clamp(0.0, 1.0)),
        StructuredTurnOutcome::Refusal(r) => {
            anyhow::bail!("model refused consistency eval: {r}")
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct ConsistencyScore {
    /// Consistency score between 0.0 (contradictory) and 1.0 (fully accurate).
    score: f32,
}
