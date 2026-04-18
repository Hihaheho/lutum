//! ReAct loop using `Lutum` directly — no `Session`.
//!
//! This example mirrors `react_loop.rs` but manages `ModelInput` by hand instead of
//! delegating to `Session`. The contrast shows what `Session` actually does:
//!
//! - `session.push_system(...)` → `input = ModelInput::new().system(...)`
//! - `session.text_turn().tools::<T>().collect()` → `llm.text_turn(input.clone()).tools::<T>().collect()`
//! - `round.commit(&mut session, results)` → `round.commit_into(&mut input, results)?`
//!
//! Use this pattern when you need to branch transcripts, interleave turns from multiple
//! contexts, or otherwise manage the input sequence explicitly.
//!
//! Requires a running Ollama instance with a tool-capable model:
//!   ollama pull gemma4:e2b
//!   cargo run --example react_loop_direct
//!
//! Or override via environment variables:
//!   ENDPOINT=https://api.openai.com/v1 TOKEN=<key> MODEL=gpt-4o \
//!     cargo run --example react_loop_direct

use std::convert::Infallible;
use std::sync::Arc;

use anyhow::bail;
use lutum::*;
use lutum_openai::OpenAiAdapter;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── In-memory data ────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct User {
    id: u32,
    name: &'static str,
}

#[derive(Clone, Copy)]
struct Order {
    order_id: u32,
    user_id: u32,
    amount_cents: u32,
}

const USERS: [User; 4] = [
    User {
        id: 1,
        name: "Alice",
    },
    User { id: 2, name: "Bob" },
    User {
        id: 3,
        name: "Carol",
    },
    User {
        id: 4,
        name: "Diana",
    },
];

const ORDERS: [Order; 7] = [
    Order {
        order_id: 101,
        user_id: 1,
        amount_cents: 12_000,
    },
    Order {
        order_id: 102,
        user_id: 3,
        amount_cents: 9_000,
    },
    Order {
        order_id: 103,
        user_id: 2,
        amount_cents: 35_000,
    },
    Order {
        order_id: 104,
        user_id: 1,
        amount_cents: 8_000,
    },
    Order {
        order_id: 105,
        user_id: 4,
        amount_cents: 21_000,
    },
    Order {
        order_id: 106,
        user_id: 2,
        amount_cents: 5_000,
    },
    Order {
        order_id: 107,
        user_id: 4,
        amount_cents: 18_000,
    },
];

// ── Tool definitions ──────────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct DbText {
    text: String,
}

#[lutum::tool_fn]
/// List every user in the in-memory database as `id: name` lines.
async fn list_users() -> Result<DbText, Infallible> {
    let text = USERS
        .iter()
        .map(|u| format!("{}: {}", u.id, u.name))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(DbText { text })
}

#[lutum::tool_input(name = "get_orders", output = DbText)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct GetOrdersArgs {
    /// The user ID whose orders should be returned.
    user_id: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, lutum::Toolset)]
enum DbTools {
    ListUsers(ListUsers),
    GetOrders(GetOrdersArgs),
}

fn lookup_orders(user_id: u32) -> DbText {
    let lines: Vec<_> = ORDERS
        .iter()
        .filter(|o| o.user_id == user_id)
        .map(|o| format!("order {}: {}", o.order_id, format_cents(o.amount_cents)))
        .collect();
    DbText {
        text: if lines.is_empty() {
            format!("no orders for user {user_id}")
        } else {
            lines.join("\n")
        },
    }
}

fn format_cents(c: u32) -> String {
    if c % 100 == 0 {
        format!("${}", c / 100)
    } else {
        format!("${}.{:02}", c / 100, c % 100)
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e2b".into());

    let adapter = OpenAiAdapter::new(token)
        .with_base_url(&endpoint)
        .with_default_model(ModelName::new(&model_name)?);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let llm = Lutum::new(Arc::new(adapter), budget);

    // Build the initial ModelInput manually — no Session involved.
    let mut input = ModelInput::new()
        .system(
            "You are investigating an in-memory database through tools. \
             The database contents are not in the conversation, so do not guess. \
             Use list_users() to discover user IDs, then get_orders(user_id) for each user, \
             then answer with the top spender's name and total.",
        )
        .user("Who is the top spender? Give their name and total.");

    for step in 1..=10 {
        // Pass a clone of the current input each iteration; the original is updated below.
        let outcome = llm
            .text_turn(input.clone())
            .tools::<DbTools>()
            .collect()
            .await?;

        match outcome {
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let mut tool_results = Vec::with_capacity(round.tool_calls.len());

                for call in round.tool_calls.iter().cloned() {
                    match call {
                        DbToolsCall::ListUsers(c) => {
                            println!("[step {step}] list_users()");
                            let result = list_users().await.unwrap();
                            println!("         → {}", result.text.replace('\n', "\\n"));
                            tool_results.push(c.complete(result).unwrap());
                        }
                        DbToolsCall::GetOrders(c) => {
                            let uid = c.input().user_id;
                            println!("[step {step}] get_orders(user_id={uid})");
                            let result = lookup_orders(uid);
                            println!("         → {}", result.text.replace('\n', "\\n"));
                            tool_results.push(c.complete(result).unwrap());
                        }
                    }
                }

                // Commit the assistant turn and tool results directly into ModelInput.
                // This is what Session::commit does internally.
                round.commit_into(&mut input, tool_results)?;
            }
            TextStepOutcomeWithTools::Finished(result) => {
                println!("Answer: {}", result.assistant_text().trim());
                return Ok(());
            }
        }
    }

    bail!("react_loop_direct example hit the 10-step limit without a final answer")
}
