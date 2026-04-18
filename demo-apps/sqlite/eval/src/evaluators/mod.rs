use std::sync::Arc;

use lutum_eval::Score;
use sqlite_agent::SqliteDb;

pub mod case;
pub mod consistency;
pub mod sql_syntax;
pub mod table_scan;

/// Shared artifact for the SQL-level pure evaluators.
pub struct SqlCheckInput {
    pub db: Arc<SqliteDb>,
    pub sql: String,
}

/// Aggregated scores for one test case run.
#[derive(Debug, Clone, Default)]
pub struct CaseScore {
    pub expected_tool_ok: Option<bool>,
    pub expected_row_count_ok: Option<bool>,
    pub sql_syntax_ok: Option<bool>,
    pub no_large_scan: Option<bool>,
    pub consistency_score: Option<Score>,
}

impl CaseScore {
    /// Simple pass/fail summary: at least one boolean check must be populated,
    /// and every populated boolean check must be true.
    pub fn passed(&self) -> bool {
        let checks = [
            self.expected_tool_ok,
            self.expected_row_count_ok,
            self.sql_syntax_ok,
            self.no_large_scan,
        ];
        checks.iter().any(|check| check.is_some())
            && checks.into_iter().flatten().all(|value| value)
    }

    pub fn display(&self) -> String {
        let mut parts = vec![];
        if let Some(v) = self.expected_tool_ok {
            parts.push(format!("tool:{}", if v { "✓" } else { "✗" }));
        }
        if let Some(v) = self.expected_row_count_ok {
            parts.push(format!("rows:{}", if v { "✓" } else { "✗" }));
        }
        if let Some(v) = self.sql_syntax_ok {
            parts.push(format!("syntax:{}", if v { "✓" } else { "✗" }));
        }
        if let Some(v) = self.no_large_scan {
            parts.push(format!("scan:{}", if v { "✓" } else { "✗" }));
        }
        if let Some(s) = self.consistency_score {
            parts.push(format!("consistency:{:.2}", s.value()));
        }
        if parts.is_empty() {
            "no checks".to_string()
        } else {
            parts.join("  ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CaseScore;

    #[test]
    fn passed_requires_at_least_one_populated_boolean_check() {
        assert!(!CaseScore::default().passed());
    }

    #[test]
    fn passed_requires_all_populated_boolean_checks_to_be_true() {
        let score = CaseScore {
            expected_tool_ok: Some(true),
            expected_row_count_ok: Some(false),
            ..CaseScore::default()
        };

        assert!(!score.passed());
    }

    #[test]
    fn passed_accepts_all_true_populated_checks() {
        let score = CaseScore {
            expected_tool_ok: Some(true),
            sql_syntax_ok: Some(true),
            no_large_scan: Some(true),
            ..CaseScore::default()
        };

        assert!(score.passed());
    }
}
