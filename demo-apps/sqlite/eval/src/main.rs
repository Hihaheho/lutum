mod cases;
mod evaluators;
mod runner;

use std::{path::PathBuf, sync::Arc};

use clap::Parser;
use lutum::{Lutum, ModelName, SharedPoolBudgetManager, SharedPoolBudgetOptions};
use lutum_claude::ClaudeAdapter;

// ---------------------------------------------------------------------------
// CLI args (clap)
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "sqlite-agent-eval",
    about = "Evaluation runner for sqlite-agent"
)]
struct Args {
    /// Path to the SQLite database file
    #[arg(long, default_value = "agent.db")]
    db: PathBuf,

    /// Path to the TOML test case file
    #[arg(long, default_value = "cases.toml")]
    cases: PathBuf,

    /// Claude model for the agent under evaluation
    #[arg(long, default_value = "claude-haiku-4-5-20251001")]
    model: String,

    /// Claude model for the LLM judge (defaults to same as --model)
    #[arg(long)]
    judge_model: Option<String>,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let args = Args::parse();
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| anyhow::anyhow!("ANTHROPIC_API_KEY env var is not set"))?;

    // Load test suite
    let cases_toml = std::fs::read_to_string(&args.cases)
        .map_err(|e| anyhow::anyhow!("reading cases file {:?}: {e}", args.cases))?;
    let suite = cases::TestSuite::from_toml(&cases_toml)?;

    // Build main LLM
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let model = ModelName::new(&args.model)?;
    let adapter = ClaudeAdapter::new(api_key.clone()).with_default_model(model);
    let main_llm = Lutum::new(Arc::new(adapter), budget.clone());

    // Optionally build judge LLM
    let judge_llm = if let Some(jm) = args.judge_model {
        let jmodel = ModelName::new(&jm)?;
        let jadapter = ClaudeAdapter::new(api_key.clone()).with_default_model(jmodel);
        Some(Lutum::new(Arc::new(jadapter), budget))
    } else {
        None
    };

    println!(
        "Running {} test cases against {}",
        suite.cases.len(),
        args.db.display()
    );
    println!("{}", "─".repeat(72));

    let mut passed = 0usize;
    let mut failed = 0usize;

    for case in &suite.cases {
        print!("  {:<40}", case.name);
        match runner::run_case(case, &args.db, &main_llm, judge_llm.as_ref()).await {
            Ok(score) => {
                if score.passed() {
                    passed += 1;
                    println!("PASS  {}", score.display());
                } else {
                    failed += 1;
                    println!("FAIL  {}", score.display());
                }
            }
            Err(e) => {
                failed += 1;
                println!("ERROR  {e}");
            }
        }
    }

    println!("{}", "─".repeat(72));
    println!("  {} passed, {} failed", passed, failed);

    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}
