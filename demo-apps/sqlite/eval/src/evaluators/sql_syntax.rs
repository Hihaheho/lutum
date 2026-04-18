use std::convert::Infallible;

use lutum_eval::{PureEval, TraceSnapshot};

use super::SqlCheckInput;

/// Pure evaluator that checks SQL syntax by running `EXPLAIN <sql>` against
/// the live database. Returns `true` if the SQL is syntactically valid.
pub struct SqlSyntaxCheck;

impl PureEval for SqlSyntaxCheck {
    type Artifact = SqlCheckInput;
    type Report = bool;
    type Error = Infallible;

    fn evaluate(
        &self,
        _trace: &TraceSnapshot,
        artifact: &SqlCheckInput,
    ) -> Result<bool, Infallible> {
        Ok(artifact.db.check_syntax(&artifact.sql).is_ok())
    }
}
