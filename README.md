# agents

`agents` is a small Rust library for typed, streaming LLM workflows.

The library is built around three layers:

- `Context<M, B, L>`: the exact execution contract
- `TextTurn` / `StructuredTurn`: thin turn facades over shared config
- `Session<M, B, L>`: transcript and replay convenience without hidden control flow

If you want the design rationale rather than just the public surface, read
[docs/DESIGN.md](docs/DESIGN.md).

## Workspace layout

- `crates/agents` - public facade crate, `Context`, `Session`, mocks
- `crates/agents-protocol` - canonical request/response algebras and core traits
- `crates/agents-openai` - OpenAI-compatible adapter
- `crates/agents-macros` - proc-macros for tools

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
use agents::{
    Context, InputMessageRole, ModelInput, ModelInputItem, NoTools, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, TextTurn, TurnConfig, UsageEstimate,
};

#[derive(Clone, Debug)]
struct AppMarker;

impl agents::Marker for AppMarker {
    fn span_name(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("example")
    }
}

async fn run<L: agents::LlmAdapter>(
    ctx: Context<AppMarker, agents::SharedPoolBudgetManager, L>,
) -> Result<(), Box<dyn std::error::Error>>
where
    L::Error: std::error::Error + 'static,
{
    let input = ModelInput::from_items(vec![
        ModelInputItem::text(InputMessageRole::System, "You are concise."),
        ModelInputItem::text(InputMessageRole::User, "Say hello."),
    ]);

    let result = ctx
        .responses_text(
            AppMarker,
            input,
            TextTurn::<NoTools> {
                config: TurnConfig::<NoTools>::new(agents::ModelName::new("gpt-4.1-mini")?),
            },
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

## Shared turn config

Turn-level configuration lives in plain structs:

- `GenerationParams`
- `ReasoningParams`
- `TurnConfig<T>`
- `ToolPolicy<T>`

The safe primary path is `TextTurn::new(ModelName::new(...)? )` /
`StructuredTurn::new(ModelName::new(...)? )` for top-level turns,
with `..Default::default()` reserved for nested parameter bundles:

```rust
use agents::{GenerationParams, TextTurn, ToolPolicy};

let mut turn = TextTurn::<AppTools>::new(agents::ModelName::new("gpt-4.1")?);
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
output contract when one model selects the tools that another model may use.

```rust
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
    Search(SearchArgs),
}

let turn = TextTurn::<AppTools> {
    config: {
        let mut config = agents::TurnConfig::<AppTools>::new(agents::ModelName::new("gpt-4.1")?);
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
    .prepare_text(turn, UsageEstimate::zero())
    .await?
    .collect_noop()
    .await?;

match outcome {
    agents::TextStepOutcome::Finished(result) => {
        session.commit_text(result)?;
    }
    agents::TextStepOutcome::NeedsToolResults(round) => {
        let tool_uses = execute_tools(round.tool_calls.clone())?;
        session.commit_tool_round(round, tool_uses)?;
    }
}
```

## Examples

- `text_minimal.rs`
- `text_session.rs`
- `structured_extract.rs`
- `single_tool_roundtrip.rs`
- `parallel_tools.rs`
- `context_direct_control.rs`

## Development

Useful commands:

```bash
cargo check --workspace --all-targets
cargo test --workspace
```
