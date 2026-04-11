//! Interactive user approval of tool calls via the lutum hook system.
//!
//! Before each tool call executes, the user is shown the tool name and its JSON input,
//! then prompted to choose:
//!
//! - `[1] Accept` — run the tool as proposed
//! - `[2] Reject with reason` — send a rejection message back to the model
//! - `[3] Edit input` — modify the JSON and run with new arguments
//!
//! The `CliApprover` struct acts as a **hook proxy**: it bridges the generic
//! `approve_tool_call` hook slot to interactive stdin/stdout logic.  Swap it for
//! any other `ApproveToolCall` impl — a policy struct, a test double, a remote
//! approval service — without touching the agent loop.
//!
//! Usage:
//! ```
//! ENDPOINT=http://localhost:11434/v1 TOKEN=local MODEL=gemma4:e2b \
//!   cargo run -p lutum --example user_approval_hook --features openai
//! ```

use lutum::*;
use lutum_openai::OpenAiAdapter;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    convert::Infallible,
    io::Write as _,
    sync::{Arc, LazyLock, Mutex},
};

// ---------------------------------------------------------------------------
// In-memory filesystem (global, mutable)
// ---------------------------------------------------------------------------

static FS: LazyLock<Mutex<HashMap<String, String>>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert(
        "/config.toml".to_owned(),
        "debug = false\nport = 8080\nlog_level = \"info\"".to_owned(),
    );
    m.insert(
        "/readme.md".to_owned(),
        "# Project\n\nA demonstration project for the user-approval hook example.".to_owned(),
    );
    m.insert(
        "/data.csv".to_owned(),
        "id,name,value\n1,alpha,42\n2,beta,7\n3,gamma,99".to_owned(),
    );
    Mutex::new(m)
});

// ---------------------------------------------------------------------------
// Tool functions
// ---------------------------------------------------------------------------

#[lutum::tool_fn]
/// List all files in the filesystem, one path per line.
async fn list_files() -> Result<String, Infallible> {
    let fs = FS.lock().unwrap();
    let mut paths: Vec<&String> = fs.keys().collect();
    paths.sort();
    Ok(paths
        .into_iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join("\n"))
}

#[lutum::tool_fn]
/// Read the contents of a file. Returns an error message if the file does not exist.
async fn read_file(path: String) -> Result<String, Infallible> {
    Ok(match FS.lock().unwrap().get(&path) {
        Some(content) => content.clone(),
        None => format!("error: {path}: no such file"),
    })
}

#[lutum::tool_fn]
/// Write content to a file, creating or overwriting it.
async fn write_file(path: String, content: String) -> Result<String, Infallible> {
    FS.lock().unwrap().insert(path.clone(), content);
    Ok(format!("wrote {path}"))
}

#[lutum::tool_fn]
/// Delete a file. Returns an error message if the file does not exist.
async fn delete_file(path: String) -> Result<String, Infallible> {
    Ok(match FS.lock().unwrap().remove(&path) {
        Some(_) => format!("deleted {path}"),
        None => format!("error: {path}: no such file"),
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum FsTools {
    ListFiles(ListFiles),
    ReadFile(ReadFile),
    WriteFile(WriteFile),
    DeleteFile(DeleteFile),
}

// ---------------------------------------------------------------------------
// Approval hook
// ---------------------------------------------------------------------------

/// The outcome returned by the `approve_tool_call` hook.
#[derive(Debug, Clone)]
enum Approval {
    Accept,
    Reject {
        reason: String,
    },
    /// The user edited the tool input; the new JSON value is enclosed.
    Edit(serde_json::Value),
}

/// Called once per tool call before execution.
///
/// Default: auto-accept (useful for non-interactive contexts and tests).
/// Override with `ApprovalHooks::new().with_approve_tool_call(impl ApproveToolCall)`.
#[hooks]
trait ApprovalHooks {
    #[hook(fallback)]
    async fn approve_tool_call(_name: &str, _args: &serde_json::Value) -> Approval {
        Approval::Accept
    }
}

// ---------------------------------------------------------------------------
// CLI approver — the hook proxy that drives stdin/stdout
// ---------------------------------------------------------------------------

struct CliApprover;

#[async_trait::async_trait]
impl ApproveToolCall for CliApprover {
    async fn call(
        &self,
        name: String,
        args: &serde_json::Value,
        _last: Option<Approval>,
    ) -> Approval {
        let pretty = serde_json::to_string_pretty(args).unwrap_or_default();

        loop {
            eprintln!("\n┌─ Tool call: {name}");
            for line in pretty.lines() {
                eprintln!("│  {line}");
            }
            eprintln!("├──────────────────────────────────");
            eprintln!("│  [1] Accept  [2] Reject  [3] Edit");
            eprint!("└─> ");
            std::io::stderr().flush().ok();

            match read_line().await.as_str() {
                "1" => return Approval::Accept,
                "2" => {
                    eprint!("Reason: ");
                    std::io::stderr().flush().ok();
                    let reason = read_line().await;
                    return Approval::Reject { reason };
                }
                "3" => {
                    eprintln!("Current JSON:");
                    eprintln!("{pretty}");
                    eprintln!("Enter replacement JSON (single line):");
                    eprint!("> ");
                    std::io::stderr().flush().ok();
                    let input = read_line().await;
                    match serde_json::from_str::<serde_json::Value>(&input) {
                        Ok(v) => return Approval::Edit(v),
                        Err(e) => eprintln!("Invalid JSON ({e}), try again"),
                    }
                }
                _ => eprintln!("Enter 1, 2, or 3"),
            }
        }
    }
}

/// Spawns a blocking task to read one line from stdin.
async fn read_line() -> String {
    tokio::task::spawn_blocking(|| {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).ok();
        line.trim().to_owned()
    })
    .await
    .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the tool name and its raw arguments as a `serde_json::Value`.
fn extract_name_and_args(call: &FsToolsCall) -> (String, serde_json::Value) {
    let meta = match call {
        FsToolsCall::ListFiles(c) => &c.metadata,
        FsToolsCall::ReadFile(c) => &c.metadata,
        FsToolsCall::WriteFile(c) => &c.metadata,
        FsToolsCall::DeleteFile(c) => &c.metadata,
    };
    let args = serde_json::from_str(meta.arguments.get()).unwrap_or(serde_json::Value::Null);
    (meta.name.as_str().to_owned(), args)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e2b".into());
    let model = ModelName::new(&model_name)?;

    // The hook proxy: CliApprover bridges the generic slot to interactive I/O.
    // Replacing CliApprover with a different impl changes approval policy
    // without touching the agent loop below.
    let hooks = ApprovalHooks::new().with_approve_tool_call(CliApprover);

    let adapter = OpenAiAdapter::new(token)
        .with_base_url(endpoint)
        .with_default_model(model);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let llm = Lutum::with_hooks(Arc::new(adapter), budget, LutumHooks::new());
    let mut session = Session::new(llm.clone());

    session.push_system(
        "You are a file management assistant. \
         Use tools to explore and modify an in-memory filesystem. \
         The filesystem contains /config.toml, /readme.md, and /data.csv. \
         The user controls what you are allowed to do — each tool call requires their approval. \
         Never call the same tool with the same arguments more than once.",
    );
    session.push_user("Summarise /readme.md and /config.toml, then delete /data.csv.");

    for _step in 1..=10 {
        let outcome = session.text_turn().tools::<FsTools>().collect().await?;

        match outcome {
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let mut tool_results = Vec::with_capacity(round.tool_calls.len());

                for tool_call in round.tool_calls.iter().cloned() {
                    let (name, raw_args) = extract_name_and_args(&tool_call);
                    let approval = hooks.approve_tool_call(&name, &raw_args).await;

                    match tool_call {
                        FsToolsCall::ListFiles(call) => match approval {
                            Approval::Reject { reason } => {
                                tool_results
                                    .push(call.complete(format!("rejected: {reason}")).unwrap());
                            }
                            Approval::Accept | Approval::Edit(_) => {
                                let result = list_files().await.unwrap();
                                tool_results.push(call.complete(result).unwrap());
                            }
                        },

                        FsToolsCall::ReadFile(mut call) => match approval {
                            Approval::Reject { reason } => {
                                tool_results
                                    .push(call.complete(format!("rejected: {reason}")).unwrap());
                            }
                            Approval::Edit(new_args) => {
                                match serde_json::from_value::<ReadFile>(new_args) {
                                    Ok(new_input) => {
                                        call.input = new_input;
                                        let result =
                                            read_file(call.input.path.clone()).await.unwrap();
                                        tool_results.push(call.complete(result).unwrap());
                                    }
                                    Err(e) => {
                                        tool_results.push(
                                            call.complete(format!(
                                                "rejected: edit did not match tool schema: {e}"
                                            ))
                                            .unwrap(),
                                        );
                                    }
                                }
                            }
                            Approval::Accept => {
                                let result = read_file(call.input.path.clone()).await.unwrap();
                                tool_results.push(call.complete(result).unwrap());
                            }
                        },

                        FsToolsCall::WriteFile(mut call) => match approval {
                            Approval::Reject { reason } => {
                                tool_results
                                    .push(call.complete(format!("rejected: {reason}")).unwrap());
                            }
                            Approval::Edit(new_args) => {
                                match serde_json::from_value::<WriteFile>(new_args) {
                                    Ok(new_input) => {
                                        call.input = new_input;
                                        let result = write_file(
                                            call.input.path.clone(),
                                            call.input.content.clone(),
                                        )
                                        .await
                                        .unwrap();
                                        tool_results.push(call.complete(result).unwrap());
                                    }
                                    Err(e) => {
                                        tool_results.push(
                                            call.complete(format!(
                                                "rejected: edit did not match tool schema: {e}"
                                            ))
                                            .unwrap(),
                                        );
                                    }
                                }
                            }
                            Approval::Accept => {
                                let result =
                                    write_file(call.input.path.clone(), call.input.content.clone())
                                        .await
                                        .unwrap();
                                tool_results.push(call.complete(result).unwrap());
                            }
                        },

                        FsToolsCall::DeleteFile(mut call) => match approval {
                            Approval::Reject { reason } => {
                                tool_results
                                    .push(call.complete(format!("rejected: {reason}")).unwrap());
                            }
                            Approval::Edit(new_args) => {
                                match serde_json::from_value::<DeleteFile>(new_args) {
                                    Ok(new_input) => {
                                        call.input = new_input;
                                        let result =
                                            delete_file(call.input.path.clone()).await.unwrap();
                                        tool_results.push(call.complete(result).unwrap());
                                    }
                                    Err(e) => {
                                        tool_results.push(
                                            call.complete(format!(
                                                "rejected: edit did not match tool schema: {e}"
                                            ))
                                            .unwrap(),
                                        );
                                    }
                                }
                            }
                            Approval::Accept => {
                                let result = delete_file(call.input.path.clone()).await.unwrap();
                                tool_results.push(call.complete(result).unwrap());
                            }
                        },
                    }
                }

                round.commit(&mut session, tool_results).unwrap();
            }
            TextStepOutcomeWithTools::Finished(result) => {
                println!("\n{}", result.assistant_text().trim());
                return Ok(());
            }
        }
    }

    anyhow::bail!("hit 10-step limit without a final answer")
}
