use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::{
        Arc, LazyLock, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
    },
};

use anyhow::Context;
use async_trait::async_trait;
use lutum::{Lutum, ModelInputItem, Session, ToolResult, TurnItemIter};
use sqlite_agent::{
    AgentConfig, AgentHooks, DbRegistry, QueryResult, SqliteDb, TransactionMode, WriteDecision,
    WritePreview, run_turn,
};

use crate::{
    cases::TestCase,
    evaluators::{self, CaseScore},
};

static CASE_WORKSPACE_COUNTER: AtomicU64 = AtomicU64::new(0);
static CWD_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

struct ScriptedApprover {
    auto_approve: bool,
}

#[async_trait]
impl sqlite_agent::hooks::ApproveWrite for ScriptedApprover {
    async fn call(&self, preview: WritePreview, _last: Option<WriteDecision>) -> WriteDecision {
        if self.auto_approve {
            tracing::debug!(rows = preview.rows_affected, "auto-approving write");
            WriteDecision::Accept
        } else {
            WriteDecision::Reject("eval: writes not auto-approved for this case".to_string())
        }
    }
}

struct FixedMode(TransactionMode);

#[async_trait]
impl sqlite_agent::hooks::GetTransactionMode for FixedMode {
    async fn call(&self, _last: Option<TransactionMode>) -> TransactionMode {
        self.0
    }
}

struct CaseWorkspace {
    root: PathBuf,
    main_db_path: PathBuf,
}

impl CaseWorkspace {
    fn create(seed_db_path: &Path) -> anyhow::Result<Self> {
        let root = unique_temp_dir("sqlite-agent-eval-case")?;
        let main_db_path = root.join("main.db");
        if seed_db_path.exists() {
            copy_sqlite_database(seed_db_path, &main_db_path)?;
        }
        Ok(Self { root, main_db_path })
    }

    fn enter(&self) -> anyhow::Result<CurrentDirGuard> {
        let lock = CWD_LOCK.lock().expect("cwd lock poisoned");
        let previous = env::current_dir().context("failed to capture current directory")?;
        env::set_current_dir(&self.root)
            .with_context(|| format!("failed to switch into {}", self.root.display()))?;
        Ok(CurrentDirGuard {
            previous,
            _lock: lock,
        })
    }
}

impl Drop for CaseWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

struct CurrentDirGuard {
    previous: PathBuf,
    _lock: MutexGuard<'static, ()>,
}

impl Drop for CurrentDirGuard {
    fn drop(&mut self) {
        let _ = env::set_current_dir(&self.previous);
    }
}

fn unique_temp_dir(prefix: &str) -> anyhow::Result<PathBuf> {
    loop {
        let suffix = CASE_WORKSPACE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("{prefix}-{}-{suffix}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create case workspace {}", path.display())
                });
            }
        }
    }
}

fn path_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut raw = path.as_os_str().to_os_string();
    raw.push(suffix);
    PathBuf::from(raw)
}

fn copy_sqlite_database(source: &Path, destination: &Path) -> anyhow::Result<()> {
    fs::copy(source, destination).with_context(|| {
        format!(
            "failed to copy seed database from {} to {}",
            source.display(),
            destination.display()
        )
    })?;
    for suffix in ["-wal", "-shm"] {
        let sidecar_source = path_with_suffix(source, suffix);
        if sidecar_source.exists() {
            let sidecar_destination = path_with_suffix(destination, suffix);
            fs::copy(&sidecar_source, &sidecar_destination).with_context(|| {
                format!(
                    "failed to copy SQLite sidecar from {} to {}",
                    sidecar_source.display(),
                    sidecar_destination.display()
                )
            })?;
        }
    }
    Ok(())
}

/// Extract the last assistant text from committed session turns.
fn last_assistant_text(session: &Session) -> Option<String> {
    let mut result = None;
    for item in session.input().items() {
        if let ModelInputItem::Turn(turn) = item {
            let mut text = String::new();
            for view in TurnItemIter::new(turn.as_ref()) {
                if let Some(t) = view.as_text() {
                    text.push_str(t);
                }
            }
            if !text.is_empty() {
                result = Some(text);
            }
        }
    }
    result
}

fn executed_tool_results(session: &Session) -> Vec<ToolResult> {
    session
        .input()
        .items()
        .iter()
        .filter_map(|item| match item {
            ModelInputItem::ToolResult(tool_result) => Some(tool_result.clone()),
            _ => None,
        })
        .collect()
}

fn apply_case_expectations(
    score: &mut CaseScore,
    case: &TestCase,
    tool_results: &[ToolResult],
    last_result: Option<&QueryResult>,
) {
    score.expected_tool_ok = case.expect_tool.as_ref().map(|expected| {
        tool_results
            .iter()
            .any(|tool_result| tool_result.name.as_str() == expected)
    });
    score.expected_row_count_ok = case.expect_rows.map(|expected| {
        last_result
            .map(|query_result| query_result.row_count() == expected)
            .unwrap_or(false)
    });
}

fn query_result_matches(left: &QueryResult, right: &QueryResult) -> bool {
    left.columns == right.columns && left.rows == right.rows
}

fn tool_result_sql(tool_result: &ToolResult) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(tool_result.arguments.get())
        .ok()?
        .get("sql")?
        .as_str()
        .map(ToOwned::to_owned)
}

fn find_last_select_sql(tool_results: &[ToolResult], last_result: &QueryResult) -> Option<String> {
    tool_results.iter().rev().find_map(|tool_result| {
        if tool_result.name.as_str() != "select" {
            return None;
        }
        let result = serde_json::from_str::<QueryResult>(tool_result.result.get()).ok()?;
        if query_result_matches(&result, last_result) {
            tool_result_sql(tool_result)
        } else {
            None
        }
    })
}

fn apply_select_checks(
    score: &mut CaseScore,
    db: &SqliteDb,
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
    score.sql_syntax_ok = Some(evaluators::sql_syntax::check_syntax(db, &sql));
    score.no_large_scan = Some(evaluators::table_scan::no_large_scan(db, &sql));
}

/// Run a single test case and return its score.
pub async fn run_case(
    case: &TestCase,
    seed_db_path: &Path,
    main_llm: &Lutum,
    judge_llm: Option<&Lutum>,
) -> anyhow::Result<CaseScore> {
    let workspace = CaseWorkspace::create(seed_db_path)?;
    let _cwd = workspace.enter()?;

    let db = Arc::new(SqliteDb::open(&workspace.main_db_path).with_context(|| {
        format!(
            "failed to open case-local database {}",
            workspace.main_db_path.display()
        )
    })?);
    let registry = DbRegistry::new();
    registry.register(
        "main",
        Arc::clone(&db),
        workspace.main_db_path.to_string_lossy().as_ref(),
    );

    let mode = if case.allow_writes {
        TransactionMode::Writable
    } else {
        TransactionMode::ReadOnly
    };

    let hooks = AgentHooks::new()
        .with_approve_write(ScriptedApprover {
            auto_approve: case.auto_approve,
        })
        .with_get_transaction_mode(FixedMode(mode));

    let config = AgentConfig {
        max_rows: case.max_rows.unwrap_or(100),
        max_rounds: 10,
    };

    let mut session = sqlite_agent::init_session(main_llm.clone(), &hooks).await;

    let output = run_turn(&mut session, &registry, &hooks, &config, case.prompt.clone(), None)
        .await
        .with_context(|| format!("case '{}' failed during execution", case.name))?;

    let tool_results = executed_tool_results(&session);
    // The last SELECT result is the last sql_history entry that has no rows_affected
    // (i.e. a SELECT rather than a write).
    let last_result = output
        .sql_history
        .iter()
        .rev()
        .find(|e| e.rows_affected.is_none())
        .map(|e| e.result.clone());
    let mut score = CaseScore::default();
    apply_case_expectations(&mut score, case, &tool_results, last_result.as_ref());
    apply_select_checks(
        &mut score,
        db.as_ref(),
        &tool_results,
        last_result.as_ref(),
    );

    if let (Some(judge), Some(query_result)) = (judge_llm, last_result.as_ref()) {
        let assistant_text = last_assistant_text(&session).unwrap_or_default();
        match crate::evaluators::consistency::score_consistency(
            judge,
            &assistant_text,
            query_result,
        )
        .await
        {
            Ok(value) => score.consistency_score = Some(value),
            Err(error) => tracing::warn!("consistency eval failed: {error}"),
        }
    }

    Ok(score)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lutum::{
        FinishReason, Lutum, MockLlmAdapter, MockTextScenario, RawJson, RawTextTurnEvent,
        SharedPoolBudgetManager, SharedPoolBudgetOptions, ToolCallId, ToolName, Usage,
    };

    use super::*;

    fn test_usage(total_tokens: u64) -> Usage {
        Usage {
            total_tokens,
            ..Usage::zero()
        }
    }

    fn tool_call_scenario(request_id: &str, tool_name: &str, arguments: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some(request_id.to_string()),
                model: "mock-model".to_string(),
            }),
            Ok(RawTextTurnEvent::ToolCallChunk {
                id: ToolCallId::from(format!("{request_id}-call")),
                name: ToolName::from(tool_name),
                arguments_json_delta: arguments.to_string(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some(request_id.to_string()),
                finish_reason: FinishReason::ToolCall,
                usage: test_usage(1),
            }),
        ])
    }

    fn final_text_scenario(request_id: &str, text: &str) -> MockTextScenario {
        MockTextScenario::events(vec![
            Ok(RawTextTurnEvent::Started {
                request_id: Some(request_id.to_string()),
                model: "mock-model".to_string(),
            }),
            Ok(RawTextTurnEvent::TextDelta {
                delta: text.to_string(),
            }),
            Ok(RawTextTurnEvent::Completed {
                request_id: Some(request_id.to_string()),
                finish_reason: FinishReason::Stop,
                usage: test_usage(1),
            }),
        ])
    }

    fn make_tool_result(name: &str, arguments: &str, result: &str) -> ToolResult {
        ToolResult::new(
            "call_1",
            name,
            RawJson::parse(arguments).expect("valid tool arguments"),
            RawJson::parse(result).expect("valid tool result"),
        )
    }

    fn make_query_result(rows: &[&[&str]]) -> QueryResult {
        QueryResult {
            columns: vec!["id".to_string(), "name".to_string()],
            rows: rows
                .iter()
                .map(|row| row.iter().map(|value| (*value).to_string()).collect())
                .collect(),
        }
    }

    fn test_case(expect_tool: Option<&str>, expect_rows: Option<usize>) -> TestCase {
        TestCase {
            name: "test".to_string(),
            prompt: "prompt".to_string(),
            expect_tool: expect_tool.map(ToOwned::to_owned),
            expect_rows,
            auto_approve: false,
            max_rows: None,
            allow_writes: false,
        }
    }

    #[test]
    fn expectations_match_tool_name_and_row_count() {
        let case = test_case(Some("select"), Some(2));
        let query_result = make_query_result(&[&["1", "alice"], &["2", "bob"]]);
        let tool_results = vec![make_tool_result(
            "select",
            r#"{"db_id":"main","sql":"SELECT id, name FROM users"}"#,
            r#"{"columns":["id","name"],"rows":[["1","alice"],["2","bob"]]}"#,
        )];
        let mut score = CaseScore::default();

        apply_case_expectations(&mut score, &case, &tool_results, Some(&query_result));

        assert_eq!(score.expected_tool_ok, Some(true));
        assert_eq!(score.expected_row_count_ok, Some(true));
    }

    #[test]
    fn expectations_fail_when_tool_or_row_count_do_not_match() {
        let case = test_case(Some("delete"), Some(3));
        let query_result = make_query_result(&[&["1", "alice"]]);
        let tool_results = vec![make_tool_result(
            "select",
            r#"{"db_id":"main","sql":"SELECT id, name FROM users"}"#,
            r#"{"columns":["id","name"],"rows":[["1","alice"]]}"#,
        )];
        let mut score = CaseScore::default();

        apply_case_expectations(&mut score, &case, &tool_results, Some(&query_result));

        assert_eq!(score.expected_tool_ok, Some(false));
        assert_eq!(score.expected_row_count_ok, Some(false));
    }

    #[test]
    fn case_workspace_keeps_relative_paths_inside_temp_dir() {
        let original_cwd = env::current_dir().expect("current dir");
        let workspace = CaseWorkspace::create(Path::new("missing-seed.db")).expect("workspace");
        let workspace_root = workspace.root.clone();
        let file_name = format!(
            "relative-extra-{}.db",
            CASE_WORKSPACE_COUNTER.fetch_add(1, Ordering::Relaxed)
        );

        {
            let _guard = workspace.enter().expect("enter workspace");
            fs::write(&file_name, "").expect("create file inside workspace");
            assert!(workspace_root.join(&file_name).exists());
        }

        assert_eq!(
            env::current_dir().expect("current dir restored"),
            original_cwd
        );
        assert!(!original_cwd.join(&file_name).exists());
    }

    #[tokio::test]
    async fn run_case_isolates_database_mutations_between_cases() {
        let seed_root = unique_temp_dir("sqlite-agent-eval-seed").expect("seed root");
        let seed_db_path = seed_root.join("seed.db");
        let seed_db = SqliteDb::open(&seed_db_path).expect("open seed db");
        seed_db
            .execute_ddl("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)")
            .expect("create items");
        seed_db
            .execute_write("INSERT INTO items (id, name) VALUES (1, 'seed')")
            .expect("seed row");

        let adapter = MockLlmAdapter::new()
            .with_text_scenario(tool_call_scenario(
                "write-tool",
                "insert",
                r#"{"db_id":"main","sql":"INSERT INTO items (id, name) VALUES (2, 'added')"}"#,
            ))
            .with_text_scenario(final_text_scenario("write-final", "inserted"))
            .with_text_scenario(tool_call_scenario(
                "read-tool",
                "select",
                r#"{"db_id":"main","sql":"SELECT id, name FROM items WHERE name IN ('seed', 'added') ORDER BY id"}"#,
            ))
            .with_text_scenario(final_text_scenario("read-final", "read"));
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        let main_llm = Lutum::new(Arc::new(adapter), budget);

        let write_case = TestCase {
            name: "write".to_string(),
            prompt: "Insert a new row".to_string(),
            expect_tool: Some("insert".to_string()),
            expect_rows: None,
            auto_approve: true,
            max_rows: Some(10),
            allow_writes: true,
        };
        let read_case = TestCase {
            name: "read".to_string(),
            prompt: "Read the seeded rows".to_string(),
            expect_tool: Some("select".to_string()),
            expect_rows: Some(1),
            auto_approve: false,
            max_rows: None,
            allow_writes: false,
        };

        let write_score = run_case(&write_case, &seed_db_path, &main_llm, None)
            .await
            .expect("write case should succeed");
        assert!(write_score.passed());
        assert_eq!(write_score.expected_tool_ok, Some(true));

        let seed_rows = SqliteDb::open(&seed_db_path)
            .expect("reopen seed db")
            .execute_read("SELECT id, name FROM items ORDER BY id")
            .expect("read seed db");
        assert_eq!(
            seed_rows.rows,
            vec![vec!["1".to_string(), "seed".to_string()]]
        );

        let read_score = run_case(&read_case, &seed_db_path, &main_llm, None)
            .await
            .expect("read case should succeed");
        assert!(read_score.passed());
        assert_eq!(read_score.expected_tool_ok, Some(true));
        assert_eq!(read_score.expected_row_count_ok, Some(true));
        assert_eq!(read_score.sql_syntax_ok, Some(true));
        assert_eq!(read_score.no_large_scan, Some(true));

        let _ = fs::remove_dir_all(seed_root);
    }
}
