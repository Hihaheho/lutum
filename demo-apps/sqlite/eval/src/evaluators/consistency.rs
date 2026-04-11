use lutum::{InputMessageRole, ModelInput, ModelInputItem};
use std::convert::Infallible;

use async_trait::async_trait;
use lutum::Lutum;
use lutum_eval::{Eval, FieldValue, JudgeEval, JudgeEvalError, Objective, Score, TraceSnapshot};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sqlite_agent::TurnOutput;

/// Extract the assistant's final text response from the trace.
///
/// Looks for the last `llm_turn` span that contains a non-empty
/// `assistant_turn` event (emitted by `lutum` when a text turn completes).
pub fn assistant_text_from_trace(trace: &TraceSnapshot) -> String {
    trace
        .find_all("llm_turn")
        .into_iter()
        .rev()
        .find_map(|span| {
            span.event("assistant_turn")
                .and_then(|e| e.field("text"))
                .and_then(|v| match v {
                    FieldValue::Str(s) if !s.is_empty() => Some(s.clone()),
                    _ => None,
                })
        })
        .unwrap_or_default()
}

fn render_input(trace: &TraceSnapshot, artifact: &TurnOutput) -> ModelInput {
    let assistant_text = assistant_text_from_trace(trace);

    let result_str = artifact
        .sql_history
        .iter()
        .rev()
        .find(|e| e.rows_affected.is_none())
        .map(|e| format!("{}", e.result))
        .unwrap_or_else(|| "(no query result)".to_string());

    let prompt = format!(
        "You are evaluating an AI database assistant. \
         Given the SQL query result and the assistant's response, \
         score how well the response reflects the actual data.\n\n\
         SQL Result:\n{result_str}\n\n\
         Assistant Response:\n{assistant_text}\n\n\
         Respond with a JSON object containing a single field `score` \
         with a float between 0.0 (completely wrong) and 1.0 (fully accurate)."
    );
    ModelInput::from_items(vec![ModelInputItem::text(InputMessageRole::User, prompt)])
}

/// Structured output produced by the LLM judge.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ConsistencyReport {
    /// Consistency score between 0.0 (contradictory) and 1.0 (fully accurate).
    pub score: f32,
}

type ConsistencyJudge =
    JudgeEval<TurnOutput, ConsistencyReport, fn(&TraceSnapshot, &TurnOutput) -> ModelInput>;

/// Build a consistency judge eval.
///
/// Pair with [`consistency_objective`] to get a scored result:
///
/// ```ignore
/// let judge = consistency_judge();
/// let scored = judge
///     .scored_by(&consistency_objective())
///     .run_collected(judge_llm, &collected)
///     .await?;
/// // scored.score  — the normalized Score
/// // scored.report — the raw ConsistencyReport
/// ```
pub fn consistency_judge() -> ConsistencyJudge {
    JudgeEval::new(render_input)
}

/// Objective that maximizes the consistency score from a [`ConsistencyReport`].
pub struct ConsistencyObjective;

impl Objective<ConsistencyReport> for ConsistencyObjective {
    type Error = Infallible;

    fn score(&self, report: &ConsistencyReport) -> Result<Score, Infallible> {
        Ok(Score::new_clamped(report.score))
    }
}

pub fn consistency_objective() -> ConsistencyObjective {
    ConsistencyObjective
}

/// End-to-end consistency evaluator: runs the LLM judge and returns a
/// normalized [`Score`] directly.
///
/// `CaseEval` delegates to this so the judge + objective chain stays in one
/// place. `ctx` is the judge LLM.
pub struct ConsistencyEval;

#[async_trait]
impl Eval for ConsistencyEval {
    type Artifact = TurnOutput;
    type Report = Score;
    type Error = JudgeEvalError;

    async fn evaluate(
        &self,
        ctx: &Lutum,
        trace: &TraceSnapshot,
        artifact: &TurnOutput,
    ) -> Result<Score, JudgeEvalError> {
        let report = consistency_judge().evaluate(ctx, trace, artifact).await?;
        Ok(consistency_objective().score(&report).unwrap()) // Error = Infallible
    }
}
