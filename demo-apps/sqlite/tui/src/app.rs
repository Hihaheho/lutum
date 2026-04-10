use std::{
    cell::Cell,
    path::Path,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use lutum::{Lutum, Session};
use lutum_claude::persistence::{ClaudeModelInputItem, restore, snapshot};
use tokio::{sync::mpsc, task::JoinHandle};

use sqlite_agent::{
    AgentConfig, AgentError, AgentHooks, CumulativeUsage, DbRegistry, QueryResult, TransactionMode,
    TurnOutput, WriteDecision, WritePreview, run_turn,
};

use crate::{
    hooks::{TuiApprover, TuiModeRequestApprover, TuiModeSource},
    registry_persistence::save_registry_snapshot as persist_registry_snapshot,
};

// ---------------------------------------------------------------------------
// Events from agent task → TUI
// ---------------------------------------------------------------------------

pub enum AgentEvent {
    Finished(TurnOutput, Session),
    Failed(AgentError, Session),
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

pub enum AppState {
    Idle,
    Running,
    Approval(WritePreview),
    ModeRequest(String),
    #[allow(dead_code)]
    Done,
}

pub struct TuiApp {
    // Core
    pub registry: Arc<DbRegistry>,
    pub llm: Lutum,
    pub config: AgentConfig,

    // Session for the next turn (None while agent task runs).
    session: Option<Session>,

    // Session snapshot for rendering. Always present.
    // Pre-task: clone with user message. Post-task: full returned session.
    pub display_session: Session,

    // Streaming text from the current in-progress response.
    pub streaming_text: String,
    streaming_rx: Option<tokio::sync::mpsc::UnboundedReceiver<String>>,

    // Turn errors (e.g. AgentError) surfaced for display.
    pub turn_errors: Vec<String>,

    // Shared mode (toggled by Tab, read by TuiModeSource)
    pub mode: Arc<RwLock<TransactionMode>>,

    // Write-approval channels
    pub preview_rx: mpsc::Receiver<WritePreview>,
    pub decision_tx: mpsc::Sender<WriteDecision>,
    preview_tx: mpsc::Sender<WritePreview>,
    decision_rx_holder: Option<mpsc::Receiver<WriteDecision>>,

    // Mode-request channels
    pub mode_request_rx: mpsc::Receiver<String>,
    pub mode_decision_tx: mpsc::Sender<bool>,
    mode_request_tx: mpsc::Sender<String>,
    mode_decision_rx_holder: Option<mpsc::Receiver<bool>>,

    pub agent_event_rx: mpsc::Receiver<AgentEvent>,
    agent_event_tx: mpsc::Sender<AgentEvent>,

    // Display state
    pub last_result: Option<QueryResult>,
    pub state: AppState,
    pub input_buf: String,
    pub scroll: Cell<usize>,
    pub running_task: Option<JoinHandle<()>>,
    pub token_stats: CumulativeUsage,
    pub scroll_to_bottom: Cell<bool>,

    // Session persistence
    session_path: Option<PathBuf>,
    registry_path: Option<PathBuf>,
}

impl TuiApp {
    pub fn new(
        registry: Arc<DbRegistry>,
        llm: Lutum,
        config: AgentConfig,
        session_path: Option<PathBuf>,
        registry_path: Option<PathBuf>,
        startup_warnings: Vec<String>,
    ) -> Self {
        let (preview_tx, preview_rx) = mpsc::channel(1);
        let (decision_tx, decision_rx) = mpsc::channel(1);
        let (mode_request_tx, mode_request_rx) = mpsc::channel(1);
        let (mode_decision_tx, mode_decision_rx) = mpsc::channel(1);
        let (agent_event_tx, agent_event_rx) = mpsc::channel(16);
        let mode = Arc::new(RwLock::new(TransactionMode::ReadOnly));

        let session = new_session(llm.clone());
        let display_session = session.clone();

        let mut app = Self {
            registry,
            llm,
            config,
            session: Some(session),
            display_session,
            streaming_text: String::new(),
            streaming_rx: None,
            turn_errors: vec![],
            mode,
            preview_rx,
            decision_tx,
            preview_tx,
            decision_rx_holder: Some(decision_rx),
            mode_request_rx,
            mode_decision_tx,
            mode_request_tx,
            mode_decision_rx_holder: Some(mode_decision_rx),
            agent_event_rx,
            agent_event_tx,
            last_result: None,
            state: AppState::Idle,
            input_buf: String::new(),
            scroll: Cell::new(0),
            running_task: None,
            token_stats: CumulativeUsage::default(),
            scroll_to_bottom: Cell::new(false),
            session_path,
            registry_path,
        };

        for warning in startup_warnings {
            app.push_turn_error(warning);
        }

        app
    }

    pub fn toggle_mode(&self) {
        let mut m = self.mode.write().unwrap();
        *m = match *m {
            TransactionMode::ReadOnly => TransactionMode::Writable,
            TransactionMode::Writable => TransactionMode::ReadOnly,
        };
    }

    pub fn current_mode(&self) -> TransactionMode {
        *self.mode.read().unwrap()
    }

    pub fn registry_path(&self) -> Option<&Path> {
        self.registry_path.as_deref()
    }

    pub fn load_persisted_session(&mut self) {
        let (session, warnings) =
            load_or_new_session(self.llm.clone(), self.session_path.as_deref());
        self.display_session = session.clone();
        self.session = Some(session);
        for warning in warnings {
            self.push_turn_error(warning);
        }
    }

    /// Reset to a fresh session and delete the saved session file.
    pub fn reset_session(&mut self) {
        let fresh = new_session(self.llm.clone());
        self.display_session = fresh.clone();
        self.session = Some(fresh);
        self.streaming_text.clear();
        self.turn_errors.clear();
        self.last_result = None;
        self.state = AppState::Idle;
        self.scroll.set(0);
        if let Some(path) = &self.session_path {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Submit the current input buffer as a user message and launch the agent task.
    pub fn submit_input(&mut self) {
        let input = std::mem::take(&mut self.input_buf);
        if input.trim().is_empty() {
            return;
        }

        self.state = AppState::Running;

        // Take session, push user message, snapshot for display, pass to task.
        let mut session = self.session.take().expect("session missing while idle");
        session.push_user(input);
        self.display_session = session.clone();
        self.streaming_text.clear();
        self.scroll_to_bottom.set(true);

        // Streaming channel for text deltas.
        let (text_tx, text_rx) = tokio::sync::mpsc::unbounded_channel();
        self.streaming_rx = Some(text_rx);

        // Build hooks for this run.
        let decision_rx = self
            .decision_rx_holder
            .take()
            .expect("decision_rx already consumed");
        let approver = TuiApprover::new(self.preview_tx.clone(), decision_rx);
        let mode_source = TuiModeSource {
            mode: self.mode.clone(),
        };
        let mode_decision_rx = self
            .mode_decision_rx_holder
            .take()
            .expect("mode_decision_rx already consumed");
        let mode_request_approver = TuiModeRequestApprover::new(
            self.mode_request_tx.clone(),
            mode_decision_rx,
            self.mode.clone(),
        );
        let hooks = AgentHooks::new()
            .with_approve_write(approver)
            .with_get_transaction_mode(mode_source)
            .with_approve_mode_request(mode_request_approver);

        let registry = self.registry.clone();
        let config = self.config.clone();
        let event_tx = self.agent_event_tx.clone();

        self.running_task = Some(tokio::spawn(async move {
            let result = run_turn(&mut session, &registry, &hooks, &config, Some(text_tx)).await;
            let event = match result {
                Ok(output) => AgentEvent::Finished(output, session),
                Err(e) => AgentEvent::Failed(e, session),
            };
            let _ = event_tx.send(event).await;
        }));
    }

    /// Poll all async channels (non-blocking). Returns true if UI state changed.
    pub fn poll(&mut self) -> bool {
        let mut changed = false;

        if let Some(rx) = &mut self.streaming_rx {
            while let Ok(chunk) = rx.try_recv() {
                self.streaming_text.push_str(&chunk);
                self.scroll_to_bottom.set(true);
                changed = true;
            }
        }

        if let Ok(preview) = self.preview_rx.try_recv() {
            self.state = AppState::Approval(preview);
            changed = true;
        }

        if let Ok(reason) = self.mode_request_rx.try_recv() {
            self.state = AppState::ModeRequest(reason);
            changed = true;
        }

        if let Ok(event) = self.agent_event_rx.try_recv() {
            match event {
                AgentEvent::Finished(output, session) => {
                    if let Some(qr) = output.last_result {
                        self.last_result = Some(qr);
                    }
                    self.token_stats.input_tokens += output.usage.input_tokens;
                    self.token_stats.output_tokens += output.usage.output_tokens;
                    self.token_stats.cache_creation_tokens += output.usage.cache_creation_tokens;
                    self.token_stats.cache_read_tokens += output.usage.cache_read_tokens;
                    self.display_session = session.clone();
                    self.session = Some(session);
                    self.streaming_text.clear();
                    self.streaming_rx = None;
                    self.state = AppState::Idle;
                    self.scroll_to_bottom.set(true);
                    self.rebuild_decision_rx();
                    self.rebuild_mode_decision_rx();
                    self.save_session();
                    self.save_registry_snapshot();
                }
                AgentEvent::Failed(e, session) => {
                    self.push_turn_error(e.to_string());
                    self.display_session = session.clone();
                    self.session = Some(session);
                    self.streaming_text.clear();
                    self.streaming_rx = None;
                    self.state = AppState::Idle;
                    self.scroll_to_bottom.set(true);
                    self.rebuild_decision_rx();
                    self.rebuild_mode_decision_rx();
                    self.save_registry_snapshot();
                }
            }
            changed = true;
        }

        changed
    }

    /// Send a write-approval decision to the waiting agent task.
    pub fn send_decision(&self, decision: WriteDecision) {
        send_over(self.decision_tx.clone(), decision);
    }

    /// Send a mode-request decision to the waiting agent task.
    pub fn send_mode_decision(&self, granted: bool) {
        send_over(self.mode_decision_tx.clone(), granted);
    }

    fn save_session(&self) {
        let Some(path) = &self.session_path else {
            return;
        };
        let Some(session) = &self.session else { return };
        let items = match snapshot(session.input()) {
            Ok(items) => items,
            Err(e) => {
                tracing::warn!("session snapshot failed: {e}");
                return;
            }
        };
        match serde_json::to_string(&items) {
            Ok(json) => {
                if let Err(e) = std::fs::write(path, json) {
                    tracing::warn!("session save failed: {e}");
                }
            }
            Err(e) => tracing::warn!("session serialize failed: {e}"),
        }
    }

    fn save_registry_snapshot(&self) {
        if let Err(message) = persist_registry_snapshot(&self.registry, self.registry_path()) {
            tracing::warn!("{message}");
        }
    }

    fn rebuild_decision_rx(&mut self) {
        rebuild_channel(&mut self.decision_tx, &mut self.decision_rx_holder);
    }

    fn rebuild_mode_decision_rx(&mut self) {
        rebuild_channel(
            &mut self.mode_decision_tx,
            &mut self.mode_decision_rx_holder,
        );
    }

    fn push_turn_error(&mut self, message: String) {
        self.turn_errors.push(message);
        if self.turn_errors.len() > 20 {
            self.turn_errors.remove(0);
        }
    }
}

fn new_session(llm: Lutum) -> Session {
    let mut session = Session::new(llm);
    session.push_system(sqlite_agent::SYSTEM_PROMPT);
    session
}

fn load_or_new_session(llm: Lutum, path: Option<&Path>) -> (Session, Vec<String>) {
    if let Some(path) = path
        && let Ok(json) = std::fs::read_to_string(path)
    {
        if let Ok(items) = serde_json::from_str::<Vec<ClaudeModelInputItem>>(&json) {
            let mut session = Session::new(llm);
            *session.input_mut() = restore(items);
            tracing::info!("session loaded from {}", path.display());
            return (session, vec![]);
        }
        let message = format!(
            "session file {} exists but could not be parsed; starting fresh",
            path.display(),
        );
        tracing::warn!("{message}");
        return (new_session(llm), vec![message]);
    }
    (new_session(llm), vec![])
}

fn send_over<T: Send + 'static>(tx: mpsc::Sender<T>, value: T) {
    tokio::spawn(async move {
        let _ = tx.send(value).await;
    });
}

fn rebuild_channel<T>(tx: &mut mpsc::Sender<T>, rx_holder: &mut Option<mpsc::Receiver<T>>) {
    let (new_tx, new_rx) = mpsc::channel(1);
    *tx = new_tx;
    *rx_holder = Some(new_rx);
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    use lutum::{MockLlmAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions};

    use super::*;
    use crate::registry_persistence::{load_registry_snapshot, save_registry_snapshot};

    fn unique_temp_dir(label: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!(
            "sqlite-agent-tui-app-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn test_lutum() -> Lutum {
        let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
        Lutum::new(Arc::new(MockLlmAdapter::new()), budget)
    }

    fn create_db(path: &Path) {
        sqlite_agent::SqliteDb::open(path).unwrap();
    }

    fn write_session_snapshot(path: &Path) {
        let mut session = Session::new(test_lutum());
        session.push_system(sqlite_agent::SYSTEM_PROMPT);
        session.push_user("remember this prompt");
        let items = snapshot(session.input()).unwrap();
        fs::write(path, serde_json::to_string(&items).unwrap()).unwrap();
    }

    #[test]
    fn session_restore_still_works_when_registry_persistence_is_disabled() {
        let temp_dir = unique_temp_dir("session-only");
        let session_path = temp_dir.join("agent-session.json");
        let main_path = temp_dir.join("main.sqlite");

        create_db(&main_path);
        write_session_snapshot(&session_path);

        let registry = Arc::new(DbRegistry::new());
        let main_db = Arc::new(sqlite_agent::SqliteDb::open(&main_path).unwrap());
        assert!(registry.register("main", main_db, main_path.to_string_lossy().to_string()));

        let mut app = TuiApp::new(
            registry,
            test_lutum(),
            AgentConfig::default(),
            Some(session_path),
            None,
            vec![],
        );
        app.load_persisted_session();

        assert!(app.turn_errors.is_empty());
        assert_eq!(app.display_session.input().items().len(), 2);
        assert!(matches!(
            &app.display_session.input().items()[1],
            lutum::ModelInputItem::Message { .. }
        ));

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn bootstrap_loads_registry_before_session_restore() {
        let temp_dir = unique_temp_dir("bootstrap");
        let main_path = temp_dir.join("main.sqlite");
        let analytics_path = temp_dir.join("analytics.sqlite");
        let registry_path = temp_dir.join("sqlite-agent-registry.json");
        let session_path = temp_dir.join("agent-session.json");

        create_db(&main_path);
        create_db(&analytics_path);
        write_session_snapshot(&session_path);

        let source_registry = DbRegistry::new();
        let source_main = Arc::new(sqlite_agent::SqliteDb::open(&main_path).unwrap());
        assert!(source_registry.register(
            "main",
            source_main,
            main_path.to_string_lossy().to_string()
        ));
        source_registry
            .create("analytics", &analytics_path)
            .unwrap();
        save_registry_snapshot(&source_registry, Some(&registry_path)).unwrap();

        let restored_registry = Arc::new(DbRegistry::new());
        let restored_main = Arc::new(sqlite_agent::SqliteDb::open(&main_path).unwrap());
        assert!(restored_registry.register(
            "main",
            restored_main,
            main_path.to_string_lossy().to_string()
        ));
        let warnings = load_registry_snapshot(&restored_registry, Some(&registry_path));

        let mut app = TuiApp::new(
            restored_registry,
            test_lutum(),
            AgentConfig::default(),
            Some(session_path),
            Some(registry_path),
            warnings,
        );
        app.load_persisted_session();

        let databases = app.registry.list().databases;
        assert_eq!(databases.len(), 2);
        assert_eq!(databases[0].db_id, "analytics");
        assert_eq!(databases[1].db_id, "main");
        assert!(app.turn_errors.is_empty());

        let _ = fs::remove_dir_all(temp_dir);
    }
}
