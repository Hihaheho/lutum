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
[docs/DESIGN.md](docs/DESIGN.md).

## Workspace layout

This repository is a Cargo workspace with four crates under `crates/`:

- `crates/agents` — public facade crate and `Context`
- `crates/agents-protocol` — canonical request/response algebras and core traits
- `crates/agents-openai` — OpenAI-compatible adapter
- `crates/agents-macros` — proc-macros for tools

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

- [crates/agents/examples/greeting.rs](crates/agents/examples/greeting.rs)

It uses `OpenAiAdapter` directly and defaults to an Ollama OpenAI-compatible endpoint:

```bash
cargo run -p agents --quiet --example greeting -- "Hello!"
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

- `ToolInput`
- `Toolset`
- `StructuredOutput`
- generated `<ToolsetName>Call` wrapper enums

## Tool flow

The normal tool loop is:

1. collect a turn
2. inspect generated wrapper tool calls
3. execute tools in your own code
4. turn outputs into `ToolUse`
5. replay the assistant turn with `AssistantTurn::into_input_items(...)`

This keeps tool execution explicit and avoids hiding important control flow inside the framework.

For raw tool definitions, use `#[tool_input(...)]` and call `call.tool_use(output)?` on the
generated wrapper:

```rust
#[agents::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
    agents::Toolset,
)]
enum AppTools {
    Weather(WeatherArgs),
}

match tool_call {
    AppToolsCall::Weather(call) => {
        let output = get_weather(&app_ctx, call.input.clone()).await?;
        let tool_use = call.tool_use(output)?;
        // append back into ModelInput
    }
}
```

For convenience, `#[tool_fn(skip(...))]` generates both the `ToolInput` type and a wrapper
`call(...)` method whose arguments are exactly the skipped parameters:

```rust
#[agents::tool_fn(skip(app_ctx, tenant))]
async fn get_weather(
    app_ctx: &AppCtx,
    tenant: TenantId,
    city: String,
) -> Result<WeatherResult, WeatherError> {
    ...
}

match tool_call {
    AppToolsCall::GetWeather(call) => {
        let tool_use = call.call(&app_ctx, tenant).await?;
    }
}
```

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
cargo check --workspace --all-targets
cargo test --workspace
```
