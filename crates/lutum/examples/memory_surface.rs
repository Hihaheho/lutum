use std::{fs, sync::Arc};

use lutum::*;
use lutum_openai::OpenAiAdapter;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

const STATE_PATH: &str = ".tmp/task_state.json";

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
struct TaskState {
    task: String,
    completed: Vec<String>,
    next_steps: Vec<String>,
    blocked_on: Option<String>,
    notes: String,
}

async fn update_task_state(
    llm: &Lutum,
    system: &str,
    prompt: impl Into<String>,
) -> anyhow::Result<TaskState> {
    let mut session = Session::new(llm.clone());
    session.push_system(system);
    session.push_user(prompt);
    let result = session.structured_turn::<TaskState>().collect().await?;
    let state = match &result.semantic {
        StructuredTurnOutcome::Structured(state) => state.clone(),
        StructuredTurnOutcome::Refusal(reason) => anyhow::bail!(reason.clone()),
    };
    Ok(state)
}

fn save_state(state: &TaskState) -> anyhow::Result<()> {
    fs::create_dir_all(".tmp")?;
    fs::write(STATE_PATH, serde_json::to_string_pretty(state)?)?;
    Ok(())
}

fn load_state() -> anyhow::Result<TaskState> {
    Ok(serde_json::from_str(&fs::read_to_string(STATE_PATH)?)?)
}

fn print_state(label: &str, state: &TaskState) -> anyhow::Result<()> {
    println!("{label}:");
    println!("{}", serde_json::to_string_pretty(state)?);
    Ok(())
}

fn print_changes(previous: &TaskState, current: &TaskState) {
    let newly_completed: Vec<&String> = current
        .completed
        .iter()
        .filter(|step| !previous.completed.contains(step))
        .collect();

    println!("Changes since session 1:");
    if newly_completed.is_empty() {
        println!("  Newly completed: none");
    } else {
        println!("  Newly completed:");
        for step in newly_completed {
            println!("    - {step}");
        }
    }

    if current.next_steps.is_empty() {
        println!("  Remaining next steps: none");
    } else {
        println!("  Remaining next steps:");
        for step in &current.next_steps {
            println!("    - {step}");
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e2b".into());
    let model = ModelName::new(&model_name)?;
    let llm = Lutum::new(
        Arc::new(
            OpenAiAdapter::new(token)
                .with_base_url(endpoint)
                .with_default_model(model),
        ),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );
    let system = concat!(
        "You are maintaining an external work-memory record for a coding task. ",
        "Return a valid TaskState object only. ",
        "Keep the task field identical to the original task description. ",
        "Use concise, concrete engineering steps."
    );
    let task = "Implement a REST API endpoint for user registration";

    let session1_state = update_task_state(
        &llm,
        system,
        format!(
            "Task: {task}\n\
             You are starting this task. What have you completed so far and what are the next steps?\n\
             Requirements:\n\
             - completed must be an empty array\n\
             - next_steps must contain 3 to 5 concrete implementation steps\n\
             - blocked_on should be null unless there is a real blocker\n\
             - notes should summarize the initial plan"
        ),
    )
    .await?;
    save_state(&session1_state)?;
    print_state("Session 1 saved state", &session1_state)?;

    let saved_state = load_state()?;
    let session2_state = update_task_state(
        &llm,
        system,
        format!(
            "You are resuming this task. Here is your current state:\n{}\n\
             You have just finished the first next step. Update the state to reflect progress.\n\
             Requirements:\n\
             - move the first item from next_steps into completed\n\
             - keep task unchanged\n\
             - keep remaining next_steps ordered\n\
             - update notes to reflect the latest progress\n\
             - blocked_on should describe a real blocker or be null",
            serde_json::to_string_pretty(&saved_state)?,
        ),
    )
    .await?;
    save_state(&session2_state)?;
    print_state("Session 2 updated state", &session2_state)?;
    print_changes(&saved_state, &session2_state);

    Ok(())
}
