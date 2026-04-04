# lutum

`lutum` is a small Rust library for typed, streaming LLM workflows.

The library is built around three layers:

- `Context`: the exact execution contract
- `TextTurn` / `StructuredTurn`: thin turn facades over shared config
- `Session`: transcript and replay convenience without hidden control flow

If you want the design rationale rather than just the public surface, read
[docs/DESIGN.md](docs/DESIGN.md).

## Workspace layout

- `crates/lutum` - public facade crate, `Context`, `Session`, mocks
- `crates/lutum-protocol` - canonical request/response algebras and core traits
- `crates/lutum-openai` - OpenAI Responses API adapter (also used as Ollama backend)
- `crates/lutum-claude` - Anthropic Claude Messages API adapter
- `crates/lutum-openrouter` - OpenRouter usage-recovery adapter
- `crates/lutum-macros` - proc-macros for tools

## Core ideas

- No monolithic `Agent` abstraction
- Request replay is modeled as `ModelInput`
- Completed model output is modeled as `AssistantTurn`
- Turn configuration is shared through `TurnConfig<T>`
- Text and structured turns stay separate as thin facades
- Tool execution stays in user code
- `Session` is convenience, not hidden control flow

## Minimal example

```rust
use lutum::{
    Context, ModelInput, NoTools, RequestExtensions, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, TextTurn, UsageEstimate,
};

async fn run(ctx: Context) -> Result<(), Box<dyn std::error::Error>> {
    let input = ModelInput::new()
        .system("You are concise.")
        .user("Say hello.");

    let turn = TextTurn::<NoTools>::new(lutum::ModelName::new("gpt-4.1-mini")?);

    let result = ctx
        .text_turn(
            RequestExtensions::default(),
            input,
            turn,
            UsageEstimate::zero(),
        )
        .await?
        .collect_noop()
        .await?;

    println!("{}", result.assistant_text());
    Ok(())
}

# fn main() {
#     let _ = SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default());
# }
```

## Structured Output Without Tools

Use `Context::structured_completion(...)` when you want prompt-based structured output without
tools or transcript integration. `Context::new(...)` wires turn execution only; use
`Context::from_parts(...)` when you want to supply a separate completion adapter:

```rust
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct Contact {
    email: String,
}

let result = ctx
    .structured_completion(
        RequestExtensions::default(),
        lutum::StructuredCompletionRequest::<Contact>::new(
            lutum::ModelName::new("gpt-4.1-mini")?,
            "Extract the email address.",
        ),
        UsageEstimate::zero(),
    )
    .await?
    .collect_noop()
    .await?;

match result.semantic {
    lutum::StructuredTurnOutcome::Structured(contact) => {
        println!("{}", contact.email);
    }
    lutum::StructuredTurnOutcome::Refusal(refusal) => {
        println!("{refusal}");
    }
}
```

If you want transcript/session integration but still no tools, use
`StructuredTurn::<NoTools, O>` instead.

## Shared turn config

Turn-level configuration lives in plain structs:

- `GenerationParams`
- `TurnConfig<T>`
- `ToolPolicy<T>`

The safe primary path is `TextTurn::new(ModelName::new(...)? )` /
`StructuredTurn::new(ModelName::new(...)? )` for top-level turns,
with `..Default::default()` reserved for nested parameter bundles:

```rust
use lutum::{GenerationParams, TextTurn, ToolPolicy};

let mut turn = TextTurn::<AppTools>::new(lutum::ModelName::new("gpt-4.1")?);
turn.config.tools = ToolPolicy::allow_only(vec![AppToolsSelector::Weather]);
turn.config.generation = GenerationParams {
    max_output_tokens: Some(512),
    ..Default::default()
};
```

All of these structs also derive `bon::Builder`, so builder usage stays available as a secondary
construction path.

## Tool selection

`#[derive(Toolset)]` generates two public enums:

- `<Toolset>Call`
- `<Toolset>Selector`

`<Toolset>Selector` is serializable and schema-bearing, so it can be used as part of a structured
output contract when one model selects the tools that another model may use. Selectors can also be
resolved back to their `ToolDef` via `selector.definition()` or `Toolset::definitions_for(...)`.

```rust
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
    lutum::Toolset,
)]
enum AppTools {
    Weather(WeatherArgs),
    Search(SearchArgs),
}

let turn = TextTurn::<AppTools> {
    config: {
        let mut config = lutum::TurnConfig::<AppTools>::new(lutum::ModelName::new("gpt-4.1")?);
        config.tools = ToolPolicy::allow_only(vec![
            AppToolsSelector::Weather,
            AppToolsSelector::Search,
        ]);
        config
    },
};
```

## Session

`Session` keeps transcript state and replay helpers without hiding execution:

- `prepare_text(...)` / `prepare_structured(...)` do not mutate transcript state
- `collect*()` does not auto-commit transcript state
- transcript state changes only on explicit `commit_*`
- you can always access `Context` or raw `ModelInput` directly

This makes branching and tool replay explicit:

```rust
let outcome = session
    .prepare_text(RequestExtensions::default(), turn, UsageEstimate::zero())
    .await?
    .collect_noop()
    .await?;

match outcome {
    lutum::TextStepOutcome::Finished(result) => {
        session.commit_text(result);
    }
    lutum::TextStepOutcome::NeedsToolResults(round) => {
        let tool_uses = execute_tools(round.tool_calls.clone())?;
        session.commit_tool_round(round, tool_uses)?;
    }
}
```

Single-tool tasks can use `round.expect_one()?` or `round.expect_at_most_one()?` instead of
reaching for `first()`.

## Examples

- `crates/lutum-openai/examples/ollama_transcript.rs`
- `crates/lutum-claude/examples/ollama_transcript.rs`

## Development

Useful commands:

```bash
cargo check --workspace --all-targets
cargo test --workspace
```
