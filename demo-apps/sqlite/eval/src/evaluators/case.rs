use std::sync::Arc;

use async_trait::async_trait;
use lutum::{Lutum, ToolResult};
use lutum_eval::{Eval, PureEval, TraceSnapshot};
use sqlite_agent::{QueryResult, SqliteDb, TurnOutput};

use crate::{
    cases::TestCase,
    evaluators::{
        CaseScore, SqlCheckInput,
        consistency::ConsistencyEval,
        sql_syntax::SqlSyntaxCheck,
        table_scan::TableScanCheck,
    },
};

/// Everything produced by one agent turn execution.
///
/// The future passed to `CaseEval::run_future` should return this struct so
/// that `lutum_trace::capture` wraps the entire execution and the trace
/// carries the `assistant_turn` events emitted by `lutum`.
pub struct CaseArtifact {
    /// Raw outcome of `run_turn`; an `Err` means the agent loop itself
    /// failed rather than a bad SQL answer.
    pub output: anyhow::Result<TurnOutput>,
    /// Tool results accumulated in the session by the time `run_turn` returned.
    pub tool_results: Vec<ToolResult>,
    pub db: Arc<SqliteDb>,
}

/// Top-level evaluator for a single test case.
///
/// Implements [`Eval`] so it can be driven through the standard
/// `scored_by` / `run_future` / `run_collected` combinators:
///
/// ```ignore
/// let eval = CaseEval::new(case, judge_llm);
/// let score = eval
///     .run_future(main_llm, async move { /* build CaseArtifact */ })
///     .await?;
/// ```
pub struct CaseEval {
    pub case: TestCase,
    /// Optional separate LLM for the consistency judge.
    pub judge: Option<Lutum>,
}

impl CaseEval {
    pub fn new(case: TestCase, judge: Option<Lutum>) -> Self {
        Self { case, judge }
    }
}

#[async_trait]
impl Eval for CaseEval {
    type Artifact = CaseArtifact;
    type Report = CaseScore;
    type Error = anyhow::Error;

    async fn evaluate(
        &self,
        _ctx: &Lutum,
        trace: &TraceSnapshot,
        artifact: &CaseArtifact,
    ) -> anyhow::Result<CaseScore> {
        let output = artifact
            .output
            .as_ref()
            .map_err(|e| anyhow::anyhow!("agent turn failed: {e}"))?;

        let last_result = output
            .sql_history
            .iter()
            .rev()
            .find(|e| e.rows_affected.is_none())
            .map(|e| e.result.clone());

        let mut score = CaseScore::default();
        apply_expectations(&mut score, &self.case, &artifact.tool_results, last_result.as_ref());
        apply_select_checks(&mut score, Arc::clone(&artifact.db), &artifact.tool_results, last_result.as_ref());

        if let Some(judge) = &self.judge {
            match ConsistencyEval.evaluate(judge, trace, output).await {
                Ok(s) => score.consistency_score = Some(s),
                Err(error) => tracing::warn!("consistency eval failed: {error}"),
            }
        }

        Ok(score)
    }
}

// ---------------------------------------------------------------------------
// Helpers (moved from runner.rs)
// ---------------------------------------------------------------------------

fn apply_expectations(
    score: &mut CaseScore,
    case: &TestCase,
    tool_results: &[ToolResult],
    last_result: Option<&QueryResult>,
) {
    score.expected_tool_ok = case.expect_tool.as_ref().map(|expected| {
        tool_results
            .iter()
            .any(|tr| tr.name.as_str() == expected)
    });
    score.expected_row_count_ok = case.expect_rows.map(|expected| {
        last_result
            .map(|qr| qr.row_count() == expected)
            .unwrap_or(false)
    });
}

fn apply_select_checks(
    score: &mut CaseScore,
    db: Arc<SqliteDb>,
    tool_results: &[ToolResult],
    last_result: Option<&QueryResult>,
) {
    let Some(last_result) = last_result else {
        return;
    };
    let Some(sql) = find_last_select_sql(tool_results, last_result) else {
        score.sql_syntax_ok = Some(false);
        score.no_large_scan = Some(false);
        return;
    };
    let input = SqlCheckInput { db, sql };
    let empty = TraceSnapshot { roots: vec![], root_events: vec![] };
    score.sql_syntax_ok = Some(SqlSyntaxCheck.evaluate(&empty, &input).unwrap());
    score.no_large_scan = Some(TableScanCheck.evaluate(&empty, &input).unwrap());
}

fn find_last_select_sql(tool_results: &[ToolResult], last_result: &QueryResult) -> Option<String> {
    tool_results.iter().rev().find_map(|tr| {
        if tr.name.as_str() != "select" {
            return None;
        }
        let result = serde_json::from_str::<QueryResult>(tr.result.get()).ok()?;
        if result.columns == last_result.columns && result.rows == last_result.rows {
            serde_json::from_str::<serde_json::Value>(tr.arguments.get())
                .ok()?
                .get("sql")?
                .as_str()
                .map(ToOwned::to_owned)
        } else {
            None
        }
    })
}
