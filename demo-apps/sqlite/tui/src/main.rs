mod app;
mod hooks;
mod registry_persistence;
mod ui;

use std::{path::PathBuf, sync::Arc, time::Duration};

use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use lutum::{Lutum, ModelName, SharedPoolBudgetManager, SharedPoolBudgetOptions};
use lutum_claude::ClaudeAdapter;
use ratatui::{Terminal, backend::CrosstermBackend};
use sqlite_agent::{AgentConfig, DbRegistry, SqliteDb, WriteDecision};

use app::{AppState, TuiApp};
use registry_persistence::{DEFAULT_REGISTRY_PATH, load_registry_snapshot, save_registry_snapshot};

// ---------------------------------------------------------------------------
// CLI args (clap)
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(name = "sqlite-agent-tui", about = "Interactive SQLite agent TUI")]
struct Args {
    /// Path to the SQLite database file (created if it does not exist)
    #[arg(long, default_value = "agent.db")]
    db: PathBuf,

    /// Claude model to use
    #[arg(long, default_value = "claude-haiku-4-5-20251001")]
    model: String,

    /// Maximum rows a single write operation may affect
    #[arg(long, default_value_t = 100)]
    max_rows: u64,

    /// Path to persist the session across runs. Omit to disable persistence.
    #[arg(long, default_value = "agent-session.json")]
    session: Option<PathBuf>,

    /// Path to persist the global database registry across runs.
    #[arg(long, default_value = DEFAULT_REGISTRY_PATH, conflicts_with = "no_registry")]
    registry: PathBuf,

    /// Disable global registry persistence and restore.
    #[arg(long)]
    no_registry: bool,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let args = Args::parse();

    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY env var is not set"))?;

    let db = Arc::new(SqliteDb::open(&args.db)?);
    let registry = Arc::new(DbRegistry::new());
    registry.register("main", db, args.db.to_string_lossy().as_ref());
    let registry_path = if args.no_registry {
        None
    } else {
        Some(args.registry.clone())
    };
    let startup_warnings = load_registry_snapshot(&registry, registry_path.as_deref());

    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let model = ModelName::new(&args.model)?;
    let adapter = ClaudeAdapter::new(api_key)
        .with_default_model(model)
        .with_prompt_caching();
    let llm = Lutum::new(Arc::new(adapter), budget);
    let config = AgentConfig {
        max_rows: args.max_rows,
        ..Default::default()
    };

    let mut app = TuiApp::new(
        registry,
        llm,
        config,
        args.session,
        registry_path,
        startup_warnings,
    );
    app.load_persisted_session();
    if let Err(message) = save_registry_snapshot(&app.registry, app.registry_path()) {
        tracing::warn!("{message}");
    }

    // Enter alternate screen
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_event_loop(&mut terminal, &mut app).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut TuiApp,
) -> anyhow::Result<()> {
    loop {
        terminal.draw(|f| ui::render(f, app))?;

        // Poll agent channels
        app.poll();

        // Poll keyboard with 50ms timeout so the TUI stays responsive
        if event::poll(Duration::from_millis(50))?
            && let Event::Key(key) = event::read()?
        {
            match &app.state {
                AppState::Approval(_) => {
                    handle_approval_key(app, key.code);
                }
                AppState::ModeRequest(_) => {
                    handle_mode_request_key(app, key.code);
                }
                AppState::Running => {
                    // Ignore input while agent is running (Ctrl-C still exits)
                    if key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL)
                    {
                        break;
                    }
                }
                AppState::Idle | AppState::Done => {
                    if handle_idle_key(app, key) {
                        break;
                    }
                }
            }
        }
    }
    Ok(())
}

fn handle_idle_key(app: &mut TuiApp, key: KeyEvent) -> bool {
    let code = key.code;
    let mods = key.modifiers;

    // Quit (only when textarea is empty)
    if code == KeyCode::Char('q') && mods.is_empty() && app.textarea.lines().join("").is_empty() {
        return true;
    }
    if code == KeyCode::Char('c') && mods.contains(KeyModifiers::CONTROL) {
        return true;
    }

    // New chat
    if code == KeyCode::Char('n') && mods.contains(KeyModifiers::CONTROL) {
        app.reset_session();
        return false;
    }

    // Toggle mode
    if code == KeyCode::Tab {
        app.toggle_mode();
        return false;
    }

    // Submit
    if code == KeyCode::Enter && mods.is_empty() {
        app.submit_input();
        return false;
    }

    // Conversation scroll (plain Up/Down/PageUp/PageDown)
    if mods.is_empty() {
        match code {
            KeyCode::Up => {
                app.scroll.set(app.scroll.get().saturating_sub(1));
                return false;
            }
            KeyCode::Down => {
                app.scroll.set(app.scroll.get().saturating_add(1));
                return false;
            }
            KeyCode::PageUp => {
                app.scroll.set(app.scroll.get().saturating_sub(10));
                return false;
            }
            KeyCode::PageDown => {
                app.scroll.set(app.scroll.get().saturating_add(10));
                return false;
            }
            _ => {}
        }
    }

    // SQL history pane scroll (Alt+Up / Alt+Down)
    if mods.contains(KeyModifiers::ALT) {
        match code {
            KeyCode::Up => {
                app.result_scroll
                    .set(app.result_scroll.get().saturating_sub(1));
                return false;
            }
            KeyCode::Down => {
                app.result_scroll
                    .set(app.result_scroll.get().saturating_add(1));
                return false;
            }
            _ => {}
        }
    }

    // Forward all other key presses to the textarea.
    if key.kind == KeyEventKind::Press {
        app.textarea.input(tui_textarea::Input::from(key));
    }
    false
}

fn handle_mode_request_key(app: &mut TuiApp, code: KeyCode) {
    match code {
        KeyCode::Char('y') | KeyCode::Char('Y') => {
            app.send_mode_decision(true);
            app.state = AppState::Running;
        }
        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
            app.send_mode_decision(false);
            app.state = AppState::Running;
        }
        _ => {}
    }
}

fn handle_approval_key(app: &mut TuiApp, code: KeyCode) {
    match code {
        KeyCode::Char('y') | KeyCode::Char('Y') => {
            app.send_decision(WriteDecision::Accept);
            app.state = AppState::Running;
        }
        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
            app.send_decision(WriteDecision::Reject("user declined".to_string()));
            app.state = AppState::Running;
        }
        // 'e' for edit: copy SQL into textarea for user to edit and re-submit
        KeyCode::Char('e') | KeyCode::Char('E') => {
            if let AppState::Approval(ref preview) = app.state {
                app.textarea = tui_textarea::TextArea::from(vec![preview.sql.clone()]);
            }
            app.send_decision(WriteDecision::Reject(
                "user chose to edit — re-submit corrected request".to_string(),
            ));
            app.state = AppState::Running;
        }
        _ => {}
    }
}
