use anyhow::bail;
use lutum::*;
use lutum_openai::OpenAiAdapter;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, sync::Arc};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct User {
    id: u32,
    name: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

// Store amounts as cents so the example stays integer-safe.
const ORDERS: [Order; 7] = [
    Order {
        order_id: 101,
        user_id: 1,
        amount_cents: 120_00,
    },
    Order {
        order_id: 102,
        user_id: 3,
        amount_cents: 90_00,
    },
    Order {
        order_id: 103,
        user_id: 2,
        amount_cents: 350_00,
    },
    Order {
        order_id: 104,
        user_id: 1,
        amount_cents: 80_00,
    },
    Order {
        order_id: 105,
        user_id: 4,
        amount_cents: 210_00,
    },
    Order {
        order_id: 106,
        user_id: 2,
        amount_cents: 50_00,
    },
    Order {
        order_id: 107,
        user_id: 4,
        amount_cents: 180_00,
    },
];

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
struct DbText {
    text: String,
}

#[lutum::tool_fn]
/// List every user in the in-memory database as `id: name` lines.
async fn list_users() -> Result<DbText, Infallible> {
    let text = USERS
        .iter()
        .map(|user| format!("{}: {}", user.id, user.name))
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum DbTools {
    ListUsers(ListUsers),
    GetOrders(GetOrdersArgs),
}

fn lookup_orders(user_id: u32) -> DbText {
    let text = ORDERS
        .iter()
        .filter(|order| order.user_id == user_id)
        .map(|order| {
            format!(
                "order {}: {}",
                order.order_id,
                format_cents(order.amount_cents)
            )
        })
        .collect::<Vec<_>>();

    if text.is_empty() {
        return DbText {
            text: format!("no orders for user {user_id}"),
        };
    }

    DbText {
        text: text.join("\n"),
    }
}

fn format_cents(amount_cents: u32) -> String {
    if amount_cents % 100 == 0 {
        format!("${}", amount_cents / 100)
    } else {
        format!("${}.{:02}", amount_cents / 100, amount_cents % 100)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let endpoint = std::env::var("ENDPOINT").unwrap_or_else(|_| "http://localhost:11434/v1".into());
    let token = std::env::var("TOKEN").unwrap_or_else(|_| "local".into());
    let model_name = std::env::var("MODEL").unwrap_or_else(|_| "qwen3.5:2b".into());

    let model = ModelName::new(&model_name)?;
    let adapter = OpenAiAdapter::new(token)
        .with_base_url(endpoint)
        .with_default_model(model);
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let llm = Lutum::new(Arc::new(adapter), budget);
    let mut session = Session::new(llm);

    session.push_system(
        "You are investigating an in-memory database through tools. \
         The database contents are not in the conversation, so do not guess. \
         Use list_users() to discover the user IDs, use get_orders(user_id) to inspect \
         each user's orders, then answer with the top spender's name and total.",
    );
    session.push_user("Who is the top spender? Give their name and total.");

    for _step in 1..=10 {
        let outcome = session
            .text_turn()
            .tools::<DbTools>()
            .allow_all()
            .collect()
            .await?;

        match outcome {
            TextStepOutcomeWithTools::NeedsToolResults(round) => {
                let mut tool_uses = Vec::with_capacity(round.tool_calls.len());

                for tool_call in round.tool_calls.iter().cloned() {
                    match tool_call {
                        DbToolsCall::ListUsers(call) => {
                            println!("[tool call] list_users()");
                            let result = list_users().await.unwrap();
                            println!("[tool result] {}", result.text.replace('\n', "\\n"));
                            tool_uses.push(call.tool_use(result).unwrap());
                        }
                        DbToolsCall::GetOrders(call) => {
                            let user_id = call.input().user_id;
                            println!("[tool call] get_orders(user_id={user_id})");
                            let result = lookup_orders(user_id);
                            println!("[tool result] {}", result.text.replace('\n', "\\n"));
                            tool_uses.push(call.tool_use(result).unwrap());
                        }
                    }
                }

                session.commit_tool_round(round, tool_uses).unwrap();
            }
            TextStepOutcomeWithTools::Finished(result) => {
                println!("Answer: {}", result.assistant_text().trim());
                session.commit_text_with_tools(result);
                return Ok(());
            }
        }
    }

    bail!("react_loop example hit the 10-step limit without a final answer")
}
