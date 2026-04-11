use lutum::{InputMessageRole, ModelInput, ModelInputItem};
use lutum_eval::{JudgeEval, TraceSnapshot};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlite_agent::QueryResult;

/// Input artifact for the consistency judge.
pub struct ConsistencyArtifact {
    pub assistant_text: String,
    pub query_result: QueryResult,
}

/// Structured output produced by the LLM judge.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ConsistencyReport {
    /// Consistency score between 0.0 (contradictory) and 1.0 (fully accurate).
    pub score: f32,
}

fn render_input(_trace: &TraceSnapshot, artifact: &ConsistencyArtifact) -> ModelInput {
    let result_str = format!("{}", artifact.query_result);
    let prompt = format!(
        "You are evaluating an AI database assistant. \
         Given the SQL query result and the assistant's response, \
         score how well the response reflects the actual data.\n\n\
         SQL Result:\n{result_str}\n\n\
         Assistant Response:\n{}\n\n\
         Respond with a JSON object containing a single field `score` \
         with a float between 0.0 (completely wrong) and 1.0 (fully accurate).",
        artifact.assistant_text
    );
    ModelInput::from_items(vec![ModelInputItem::text(InputMessageRole::User, prompt)])
}

/// Build a consistency judge eval.
///
/// Callers run it via [`lutum_eval::EvalExt::run_future`]:
/// ```ignore
/// let report = consistency_judge()
///     .run_future(judge_llm, async { artifact })
///     .await?;
/// ```
pub fn consistency_judge(
) -> JudgeEval<ConsistencyArtifact, ConsistencyReport, fn(&TraceSnapshot, &ConsistencyArtifact) -> ModelInput>
{
    JudgeEval::new(render_input)
}
