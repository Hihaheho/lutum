# agents

`agents` is a small Rust library for building typed, streaming LLM workflows without introducing a
top-level `Agent` abstraction.

The library gives you:

- `Context<M, B, L>` for execution
- typed text and structured turns
- public reducers for deterministic event reduction
- request and response algebras that reflect different invariants
- reservation-based budget management
- an OpenAI-compatible `reqwest` adapter

If you want the design rationale rather than just the surface API, read
[DESIGN.md](DESIGN.md).

## Core ideas

- No monolithic `Agent` trait.
- Request replay is modeled as `ModelInput`.
- Completed model output is modeled as `AssistantTurn`.
- Tools and structured output are typed per turn, not globally.
- Tool execution stays in user code.

## Installation

Add the crate to your project in the usual way, or work from this repository directly.

## Minimal example

```rust
use agents::{
    Context, InputMessageRole, ModelInput, ModelInputItem, NoTools, OpenAiAdapter,
    SharedPoolBudgetManager, SharedPoolBudgetOptions, TextTurnRequest, UsageEstimate,
};

#[derive(Clone, Debug)]
struct AppMarker;

impl agents::Marker for AppMarker {
    fn span_name(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("example")
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = OpenAiAdapter::new(std::env::var("OPENAI_API_KEY").unwrap_or_default())
        .with_base_url(
            std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434/v1".to_string()),
        );
    let budget = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
    let ctx: Context<AppMarker, _, _> = Context::new(budget, adapter);

    let input = ModelInput::from_items(vec![
        ModelInputItem::text(
            InputMessageRole::System,
            "You are a concise assistant.",
        ),
        ModelInputItem::text(InputMessageRole::User, "Say hello."),
    ]);

    let result = ctx
        .responses_text(
            AppMarker,
            input,
            TextTurnRequest::<NoTools>::new("qwen3.5:latest"),
            UsageEstimate::zero(),
        )
        .await?
        .collect_noop()
        .await?;

    println!("{}", result.assistant_text());
    Ok(())
}
```

## Running the included example

This repository includes a minimal example:

- [examples/greeting.rs](examples/greeting.rs)

It uses `OpenAiAdapter` directly and defaults to an Ollama OpenAI-compatible endpoint:

```bash
cargo run --quiet --example greeting -- "Hello!"
```

Useful environment variables:

- `OPENAI_BASE_URL`  
  Default: `http://localhost:11434/v1`
- `OPENAI_MODEL`  
  Default: `qwen3.5:latest`
- `OPENAI_API_KEY`  
  Optional for local Ollama setups; defaults to the empty string

## Main types

### Request side

- `ModelInput`
- `ModelInputItem`
- `ToolUse`

`ModelInput` is the canonical request surface. It is ordered and schema-erased where necessary.

### Response side

- `AssistantTurn`
- `AssistantTurnItem`

`AssistantTurn` is the canonical response surface for a completed model turn.

### Execution

- `Context<M, B, L>`
- `TextTurnRequest<T>`
- `StructuredTurnRequest<T, O>`
- `collect(handler)`

### Typing

- `Toolset`
- `StructuredOutput`
- `TypedToolInvocation<T>`

## Tool flow

The normal tool loop is:

1. collect a turn
2. inspect typed tool calls
3. execute tools in your own code
4. convert results with `ToolUse::from_typed(...)`
5. replay the assistant turn with `AssistantTurn::into_input_items(...)`

This keeps tool execution explicit and avoids hiding important control flow inside the framework.

## Budgeting

Budgeting is reservation-based.

- global control comes from `BudgetManager`
- per-request limits come from `RequestBudget`
- reasoning intensity is separate and uses `ThinkingBudget`

This makes concurrent use less brittle than a single global "max tokens" knob.

## Status

Current scope is intentionally narrow:

- text messages
- text turns
- structured turns
- typed tools
- OpenAI-compatible responses/completions provider

Multimodal request/response content is not modeled yet.

## Development

Useful commands:

```bash
cargo check --all-targets
cargo test
```
