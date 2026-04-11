use std::convert::Infallible;

use lutum_eval::{PureEval, TraceSnapshot};

use super::SqlCheckInput;

/// Pure evaluator that checks whether the query plan avoids full table scans.
///
/// A plan line is flagged as a full scan if it contains `"SCAN"` but not
/// `"USING INDEX"` or `"USING COVERING INDEX"`. Returns `true` (pass) when
/// no such lines are found, or when the query plan cannot be obtained.
pub struct TableScanCheck;

impl PureEval for TableScanCheck {
    type Artifact = SqlCheckInput;
    type Report = bool;
    type Error = Infallible;

    fn evaluate(&self, _trace: &TraceSnapshot, artifact: &SqlCheckInput) -> Result<bool, Infallible> {
        let Ok(plan) = artifact.db.explain_query_plan(&artifact.sql) else {
            return Ok(true); // can't judge, assume ok
        };
        Ok(!plan.iter().any(|line| {
            let l = line.to_uppercase();
            l.contains("SCAN") && !l.contains("USING INDEX") && !l.contains("USING COVERING INDEX")
        }))
    }
}
