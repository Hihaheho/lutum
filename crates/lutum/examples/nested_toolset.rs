//! Nested toolset example — `#[toolset]` variants inside `#[derive(Toolset)]`
//!
//! This example shows how to compose multiple toolsets into a parent enum using
//! the `#[toolset]` variant attribute. The outer `AppTools` enum has:
//!   - `#[toolset] Data(DataTools)` — a nested toolset for database lookups
//!   - `Calc(CalcArgs)`             — a regular direct tool
//!
//! Key API demonstrated:
//!   - `AppToolsHooks::new(DataToolsHooks::new())`
//!   - `hooks.data.register_get_user_hook(...)` (hook registration via nested field)
//!   - `hooks.data.description_overrides()` flows up into `AppToolsHooks::description_overrides()`
//!
//! Requires a running Ollama instance with gemma4:e2b:
//!   ollama pull gemma4:e2b
//!   cargo run --example nested_toolset
use std::sync::Arc;

use lutum::*;
use lutum_openai::OpenAiAdapter;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Inner toolset: DataTools ──────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct UserRecord {
    id: u32,
    name: String,
    role: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct OrderRecord {
    order_id: u32,
    amount_cents: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct OrderList {
    orders: Vec<OrderRecord>,
}

/// Look up a user by ID
#[lutum::tool_input(name = "get_user", output = UserRecord)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct GetUserArgs {
    /// The numeric user ID to look up
    user_id: u32,
}

/// Retrieve orders for a user
#[lutum::tool_input(name = "get_orders", output = OrderList)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct GetOrdersArgs {
    /// The user ID whose orders should be returned
    user_id: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum DataTools {
    GetUser(GetUserArgs),
    GetOrders(GetOrdersArgs),
}

// ── Regular tool: Calc ────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct CalcResult {
    result: f64,
}

/// Evaluate a simple arithmetic expression
#[lutum::tool_input(name = "calc", output = CalcResult)]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct CalcArgs {
    /// e.g. "12.00 + 9.00 + 35.00"
    expression: String,
}

// ── Outer composite toolset ───────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum AppTools {
    /// Nested database toolset (get_user, get_orders)
    #[toolset]
    Data(DataTools),
    /// Direct arithmetic calculator
    Calc(CalcArgs),
}

// ── In-memory data ────────────────────────────────────────────────────────────

struct Db;

impl Db {
    fn get_user(id: u32) -> Option<UserRecord> {
        let users = [
            UserRecord {
                id: 1,
                name: "Alice".into(),
                role: "admin".into(),
            },
            UserRecord {
                id: 2,
                name: "Bob".into(),
                role: "user".into(),
            },
            UserRecord {
                id: 3,
                name: "Carol".into(),
                role: "user".into(),
            },
        ];
        users.into_iter().find(|u| u.id == id)
    }

    fn get_orders(user_id: u32) -> OrderList {
        let all = [
            (1u32, 101u32, 12_000u32),
            (3, 102, 9_000),
            (2, 103, 35_000),
            (1, 104, 8_000),
            (3, 105, 21_000),
            (2, 106, 5_000),
        ];
        OrderList {
            orders: all
                .iter()
                .filter(|(uid, _, _)| *uid == user_id)
                .map(|(_, order_id, amount_cents)| OrderRecord {
                    order_id: *order_id,
                    amount_cents: *amount_cents,
                })
                .collect(),
        }
    }

    fn calc(expr: &str) -> f64 {
        // Minimal: sum of space-separated floats joined by +
        expr.split('+')
            .map(|t| t.trim().parse::<f64>().unwrap_or(0.0))
            .sum()
    }
}

// ── Hook: cache user #1 (Alice) to demonstrate nested hooks registration ──────

#[lutum::impl_hook(GetUserHook)]
async fn cached_alice(
    _meta: &ToolMetadata,
    input: GetUserArgs,
) -> ToolDecision<GetUserArgs, UserRecord> {
    if input.user_id == 1 {
        println!(
            "[hook] get_user({}) served from cache (Alice)",
            input.user_id
        );
        ToolDecision::Complete(UserRecord {
            id: 1,
            name: "Alice (cached)".into(),
            role: "admin".into(),
        })
    } else {
        ToolDecision::RunNormally(input)
    }
}

// ── ReAct loop ────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "gemma4:e2b".into());

    let model = ModelName::new(&model_name)?;
    let adapter = OpenAiAdapter::new(token)
        .with_base_url(&endpoint)
        .with_default_model(model);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let llm = Lutum::new(Arc::new(adapter), budget);
    let mut session = Session::new(llm);

    session.push_system(
        "You are a data analyst with access to a user database and a calculator. \
         Use the tools to answer the question. Do not guess — use the tools.\n\
         - get_user(user_id): returns name and role\n\
         - get_orders(user_id): returns a list of orders with amounts in cents\n\
         - calc(expression): evaluates arithmetic (sum of amounts, etc.)\n\
         User IDs in the database are 1, 2, and 3.",
    );
    session.push_user("What is the total amount (in dollars) of all orders placed by user ID 2?");

    // Build hooks: AppToolsHooks::new(DataToolsHooks::new())
    // Register a cache hook on the nested `data` field.
    let mut app_hooks = AppToolsHooks::new(DataToolsHooks::new());
    app_hooks.data.register_get_user_hook(CachedAlice);

    println!("=== nested_toolset example ===");
    println!("model: {model_name}  endpoint: {endpoint}");
    println!();

    for step in 1..=8 {
        let outcome = session.text_turn().tools::<AppTools>().collect().await?;

        match outcome {
            TextStepOutcomeWithTools::NeedsTools(round) => {
                // Apply hooks in batch first.
                let plan: ToolRoundPlan<AppTools> = round.apply_hooks(&app_hooks).await;

                // Log hook-handled calls (results committed automatically by plan.commit).
                for handled in &plan.handled {
                    match handled {
                        AppToolsHandled::Data(DataToolsHandled::GetUser(h)) => {
                            println!("[hook] get_user({}) — handled by hook", h.input().user_id);
                        }
                        AppToolsHandled::Data(DataToolsHandled::GetOrders(_)) => {
                            println!("[hook] get_orders — handled by hook");
                        }
                        AppToolsHandled::Calc(_) => {
                            println!("[hook] calc — handled by hook");
                        }
                    }
                }

                // Execute pending calls and collect results.
                let mut pending_results = Vec::new();
                for call in plan.pending.iter().cloned() {
                    match call {
                        AppToolsCall::Data(DataToolsCall::GetUser(c)) => {
                            let uid = c.input().user_id;
                            println!("[step {step}] get_user(user_id={uid})");
                            let record = Db::get_user(uid).unwrap_or(UserRecord {
                                id: uid,
                                name: "unknown".into(),
                                role: "unknown".into(),
                            });
                            println!("         → {:?}", record);
                            pending_results.push(c.complete(record)?);
                        }
                        AppToolsCall::Data(DataToolsCall::GetOrders(c)) => {
                            let uid = c.input().user_id;
                            println!("[step {step}] get_orders(user_id={uid})");
                            let order_list = Db::get_orders(uid);
                            println!("         → {} order(s)", order_list.orders.len());
                            pending_results.push(c.complete(order_list)?);
                        }
                        AppToolsCall::Calc(c) => {
                            let expr = c.input().expression.clone();
                            println!("[step {step}] calc({expr})");
                            let val = Db::calc(&expr);
                            println!("         → {val}");
                            pending_results.push(c.complete(CalcResult { result: val })?);
                        }
                    }
                }

                plan.commit(&mut session, pending_results)?;
            }
            TextStepOutcomeWithTools::Finished(result) => {
                println!();
                println!("Answer: {}", result.assistant_text().trim());
                return Ok(());
            }
        }
    }

    println!("(hit step limit without a final answer)");
    Ok(())
}
