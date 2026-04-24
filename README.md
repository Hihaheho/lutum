<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Hihaheho/lutum/main/assets/lutum-logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Hihaheho/lutum/main/assets/lutum-logo-light.svg">
  <img alt="Lutum" src="https://raw.githubusercontent.com/Hihaheho/lutum/main/assets/lutum-logo-light.svg" height="100px">
</picture>

🕶️ `lutum`, pronounced /ˈlu.tum/, is a composable, streaming LLM toolkit for advanced orchestration.
It keeps control flow in user code, preserves provider-specific power, and makes transcript replay
an explicit part of the API instead of a hidden framework detail.

- Stream typed events or collect them into completed turn results
- Keep tool execution, retries, and branching in user code
- Commit transcript state explicitly with `Session`
- Add typed hooks and in-memory trace capture without adopting a framework-owned agent loop

If you want the design rationale rather than just the public surface, read
[docs/DESIGN.md](docs/DESIGN.md).

## Goals

- Preserve provider-specific power instead of flattening everything into one framework API
- Keep execution, tool calls, retries, and branching in user code
- Make streaming and exact transcript replay first-class
- Stay composable enough to fit advanced orchestration without taking over the runtime

## Workspace Layout

| Crate | Role |
|---|---|
| `crates/lutum` | Public facade crate: `Lutum`, `Session`, turn APIs, mocks |
| `crates/lutum-protocol` | Core traits, request/response algebras, reducers, transcript views |
| `crates/lutum-openai` | OpenAI Responses API adapter, also usable with OpenAI-compatible backends |
| `crates/lutum-claude` | Anthropic Claude Messages API adapter |
| `crates/lutum-openrouter` | OpenRouter wrapper with usage recovery |
| `crates/lutum-macros` | `#[derive(Toolset)]`, `#[tool_fn]`, `#[tool_input]`, hook macros |
| `crates/lutum-trace` | Companion crate for scoped in-memory span and event capture |
| `crates/lutum-eval` | Borrowed trace/artifact metrics, async judge metrics, and live probes |

## Install

This README uses the `openai` feature and `OpenAiAdapter`, which works with OpenAI-compatible
endpoints such as OpenAI or Ollama.

```toml
[dependencies]
lutum = { version = "0.1.0", features = ["openai"] }
# Pick your favorite async runtime
tokio = { version = "1", features = ["macros", "rt"] }

# Only if you use structured output or tool schemas
schemars = { version = "1", features = ["derive"] }
serde = { version = "1", features = ["derive"] }

# Only if you want in-memory trace capture
lutum-trace = "0.1.0"
tracing-subscriber = { version = "0.3", features = ["registry"] }

# Only if you want trace/artifact evaluation helpers or live probes
lutum-eval = "0.1.0"
```

## Mental Model

- `Lutum`: execution boundary. It validates input, reserves budget, emits tracing, and reduces
  provider streams into typed results.
- `TextTurn` / `StructuredTurn<O>`: no-tools-first executable turn builders.
- `TextTurnWithTools<T>` / `StructuredTurnWithTools<T, O>`: tool-enabled turn builders reached
  via `.tools::<T>()`.
- `Session`: transcript helper. Nothing is committed until you call `commit_text`,
  `commit_structured`, `commit_text_with_tools`, `commit_structured_with_tools`, or
  `round.commit(...)`.
- `ModelInput`: low-level request surface if you want to skip `Session` and drive turns directly.
- `RequestExtensions`: opaque per-request metadata for routing, identity, and custom adapter
  logic, attached inline with `.ext(...)` or `.extensions(...)`.

## Quickstart

This is the shortest useful path: build a `Lutum`, start a `Session`, run a text turn, then
explicitly commit the result.

```rust
use std::sync::Arc;

use lutum::{
    Lutum, ModelName, OpenAiAdapter, Session, SharedPoolBudgetManager,
    SharedPoolBudgetOptions,
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = OpenAiAdapter::new(std::env::var("TOKEN")?)
        .with_base_url(std::env::var("ENDPOINT")?)
        .with_default_model(ModelName::new(&std::env::var("MODEL")?)?);

    let llm = Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );

    let mut session = Session::new(llm);
    session.push_system("You are concise.");
    session.push_user("Say hello in one sentence.");

    let result = session
        .text_turn()
        .temperature(lutum::Temperature::new(0.2)?)
        .collect()
        .await?;

    println!("{}", result.assistant_text());
    session.commit_text(result);

    Ok(())
}
```

If you do not want transcript management, call `Lutum::text_turn(input)` directly with a
`ModelInput`, then use `.stream().await?`, `.collect().await?`, or `.collect_with(handler).await?`
on the returned builder.

## Structured Output

Use `Session::structured_turn::<O>()` when structured output should participate in the same
explicit transcript model as text turns. Starting from the quickstart setup, reuse the same
mutable `session` and swap the turn kind:

```rust
use lutum::StructuredTurnOutcome;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct Contact {
    email: String,
}

session.push_user("Extract the email address from `Reach me at user@example.com`.");

let result = session
    .structured_turn::<Contact>()
    .collect()
    .await?;

match result.semantic.clone() {
    StructuredTurnOutcome::Structured(contact) => {
        println!("{}", contact.email);
        session.commit_structured(result);
    }
    StructuredTurnOutcome::Refusal(reason) => {
        eprintln!("{reason}");
    }
}
```

If you want one-off structured extraction without transcript integration, use
`Lutum::structured_completion(...)`. That path requires `Lutum::from_parts(...)`, because
completion adapters are wired separately from turn execution:

```rust
use std::sync::Arc;

use lutum::{
    Lutum, ModelName, OpenAiAdapter, SharedPoolBudgetManager, SharedPoolBudgetOptions,
    StructuredTurnOutcome,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct Contact {
    email: String,
}

let adapter = Arc::new(
    OpenAiAdapter::new(std::env::var("TOKEN")?)
        .with_base_url(std::env::var("ENDPOINT")?)
        .with_default_model(ModelName::new(&std::env::var("MODEL")?)?),
);

let llm = Lutum::from_parts(
    adapter.clone(),
    adapter.clone(),
    adapter,
    SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
);

let result = llm
    .structured_completion::<Contact>("Extract the email address from `Reach me at user@example.com`.")
    .system("Return only the structured data.")
    .collect()
    .await?;

match result.semantic {
    StructuredTurnOutcome::Structured(contact) => println!("{}", contact.email),
    StructuredTurnOutcome::Refusal(reason) => eprintln!("{reason}"),
}
```

## Tools And Explicit Session Loops

Tools are declared with macros, but they are never executed by the library. User code owns the
tool loop, and `Session` only records committed turns and tool results.

```rust
use std::sync::Arc;

use lutum::{
    Lutum, ModelName, OpenAiAdapter, Session, SharedPoolBudgetManager,
    SharedPoolBudgetOptions, TextStepOutcomeWithTools,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[lutum::tool_input(name = "weather", output = WeatherResult)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
struct WeatherResult {
    forecast: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, lutum::Toolset)]
enum Tools {
    Weather(WeatherArgs),
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let adapter = OpenAiAdapter::new(std::env::var("TOKEN")?)
        .with_base_url(std::env::var("ENDPOINT")?)
        .with_default_model(ModelName::new(&std::env::var("MODEL")?)?);

    let llm = Lutum::new(
        Arc::new(adapter),
        SharedPoolBudgetManager::new(SharedPoolBudgetOptions::default()),
    );

    let mut session = Session::new(llm);
    session.push_user("What is the weather in Tokyo?");

    loop {
        let outcome = session
            .text_turn()
            .tools::<Tools>()
            .allow_only([ToolsSelector::Weather])
            .collect()
            .await?;

        match outcome {
            TextStepOutcomeWithTools::NeedsTools(round) => {
                let tool_results = round
                    .tool_calls
                    .iter()
                    .cloned()
                    .map(|tool_call| match tool_call {
                        ToolsCall::Weather(call) => call
                            .complete(WeatherResult {
                                forecast: "sunny, 24C".into(),
                            })
                            .unwrap(),
                    })
                    .collect::<Vec<_>>();

                round.commit(&mut session, tool_results)?;
            }
            TextStepOutcomeWithTools::Finished(result) => {
                println!("{}", result.assistant_text());
                session.commit_text_with_tools(result);
                break;
            }
        }
    }

    Ok(())
}
```

For single-tool rounds, `round.expect_one()?` and `round.expect_at_most_one()?` are often cleaner
than manually inspecting the vector.

For a complete tool-hook example, see
[`crates/lutum/examples/tool_hook.rs`](crates/lutum/examples/tool_hook.rs). It shows a
cache-backed hook that handles some tool calls immediately and falls back to normal Rust tool
execution for the rest.

## Hooks

Hooks are named, typed async slots. Define them inside a local `#[hooks] trait`, implement one
slot with `#[impl_hook(...)]` or several slots with `#[impl_hooks(...)]`, and call the generated
`<TraitName>Set` methods from your own code.

```rust
#[lutum::hooks]
trait AppHooks {
    #[hook(always)]
    async fn validate_prompt(prompt: &str) -> Result<(), String> {
        if prompt.trim().is_empty() {
            Err("prompt must not be empty".into())
        } else {
            Ok(())
        }
    }
}

#[lutum::impl_hook(ValidatePrompt)]
async fn reject_secrets(
    prompt: &str,
    last: Option<Result<(), String>>,
) -> Result<(), String> {
    if let Some(previous) = last {
        previous?;
    }
    if prompt.contains("sk-") {
        Err("looks like an API key".into())
    } else {
        Ok(())
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hooks = AppHooksSet::new().with_validate_prompt(RejectSecrets);

    hooks
        .validate_prompt("Explain borrow checking in one paragraph.")
        .await?;

    Ok(())
}
```

- Slot definitions live inside one `#[hooks] trait` and never declare `last`.
- `#[impl_hook(SlotType)]` is exact-signature: write `last: Option<R>` yourself whenever the slot trait has it.
- `#[impl_hooks(AppHooksSet)] impl AppHooks for Data` registers only the methods present in that impl block.
- `always`: run the default implementation first, then chain registered hooks on top
- `fallback`: use registered hooks if present, otherwise run the default
- `singleton`: pick one override or use the default; the last registration wins and emits a warning

Core hooks often take `&Lutum` as their first argument and are provider-agnostic.
Provider-specific hooks live in owner-local hook sets:

- `LutumHooksSet` configures built-in runtime hooks such as `resolve_usage_estimate`
- `OpenAiHooksSet` configures OpenAI request-shaping hooks and is passed to `OpenAiAdapter::with_hooks(...)`
- `ClaudeHooksSet` configures Claude request-shaping hooks and is passed to `ClaudeAdapter::with_hooks(...)`

Attach request-scoped metadata inline on builders with `.ext(...)` or `.extensions(...)`:

```rust
#[derive(Clone, Debug)]
struct Tenant(&'static str);

let result = session
    .text_turn()
    .ext(Tenant("prod-eu"))
    .collect()
    .await?;
```

## Trace Capture

`lutum-trace` is a separate crate for scoped, in-memory tracing capture. It is useful in tests,
debug tooling, and local observability when you want to inspect span fields and events without
shipping traces anywhere.

`Lutum` emits `llm_turn` spans during execution, so once the layer is installed you can capture
request ids, finish reasons, and nested events. Assuming you already built `llm` as in the
quickstart, wrap the same turn execution with capture:

```rust
use lutum::Session;
use tracing::instrument::WithSubscriber as _;
use tracing_subscriber::layer::SubscriberExt as _;

let subscriber = tracing_subscriber::registry().with(lutum_trace::layer());

let collected = lutum_trace::capture(
    async {
        let mut session = Session::new(llm.clone());
        session.push_user("Say hello.");

        let _ = session
            .text_turn()
            .collect()
            .await?;

        Ok::<(), Box<dyn std::error::Error>>(())
    }
    .with_subscriber(subscriber),
)
.await;

if let Some(turn) = collected.trace.span("llm_turn") {
    println!("request_id = {:?}", turn.field("request_id"));
    println!("finish_reason = {:?}", turn.field("finish_reason"));
}
```

The capture scope is task-local. If you spawn a Tokio task, call `lutum_trace::capture(...)`
inside that task as well. For tests, `lutum_trace::test::collect(...)` provides a small helper on
top of the same machinery.

## Evaluation

`lutum-eval` is a companion crate for scoring a strongly typed artifact together with a
captured `TraceSnapshot`.

- `PureEval`: sync, pure, replay-friendly evaluation over `&TraceSnapshot` and `&Artifact`
- `Eval`: async evaluation that also receives `&Lutum`; every `PureEval` is also an `Eval`
- `Objective<R>`: converts a typed report into a scalar `Score` in `0..=1`, where larger is always better
- `Probe`: async live evaluation over `TraceEvent` plus finalization; every `Eval` is also a `Probe`

The design separates **observation** from **optimization**:

- `Eval` produces a rich typed report
- `Objective` decides how to scalarize that report into a `Score`
- the same report can be rescored under different objectives

Live evaluation uses the future output as the artifact directly:

```rust
use async_trait::async_trait;
use lutum::Lutum;
use lutum_eval::{
    Eval, EvalExt, PureEval, PureEvalExt, Score, maximize,
};

#[lutum::hooks]
trait EvalHooks {
    #[hook(singleton)]
    async fn score_bias(score: usize) -> usize {
        score
    }
}

#[lutum::impl_hook(ScoreBias)]
async fn double_score(score: usize) -> usize {
    score * 2
}

#[derive(Debug)]
struct Draft {
    text: String,
}

struct LengthEval;

struct ScaledLengthEval {
    hooks: EvalHooksSet<'static>,
}

impl PureEval for LengthEval {
    type Artifact = Draft;
    type Report = usize;
    type Error = core::convert::Infallible;

    fn evaluate(
        &self,
        _trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(artifact.text.len())
    }
}

struct ScaledLengthEval;

#[async_trait]
impl Eval for ScaledLengthEval {
    type Artifact = Draft;
    type Report = usize;
    type Error = core::convert::Infallible;

    async fn evaluate(
        &self,
        _llm: &Lutum,
        _trace: &lutum_eval::TraceSnapshot,
        artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(self.hooks.score_bias(artifact.text.len()).await)
    }
}

// Assume `llm: Lutum` was already built as in the quickstart.
let pure_report = LengthEval.run_future(async {
    Draft { text: "hello".into() }
}).await?;

let eval = ScaledLengthEval {
    hooks: EvalHooksSet::new().with_score_bias(DoubleScore),
};

let eval_report = eval.run_future(&llm, async {
    Draft { text: "hello".into() }
}).await?;

let scored = eval
    .scored_by(&maximize(|report: &usize| Score::new_clamped(*report as f32 / 10.0)))
    .run_future(&llm, async { Draft { text: "hello".into() } })
    .await?;

let score = scored.score;
```

`PureEval` is the small synchronous subset. `Eval` is the default async observation abstraction
that can execute through `Lutum`. `Objective` is where scalar scoring lives.

### Probes

`Probe` is the live-only counterpart when you want incremental decisions from trace events or
want hook calls to route back into the same mutable state machine.

- `ProbeRuntime::new(probe)` owns the runtime task and exposes a `ProbeHandle`
- build any local hook sets you need explicitly from `runtime.dispatcher()`
- `ProbeRuntime::run_future(&llm, ...)` forwards live `TraceEvent`s, then calls `finalize(&llm, trace, artifact)`

As with `lutum_trace::capture(...)`, live probe events require the active subscriber stack to
include `lutum_trace::layer()`.

A probe that intercepts a hook usually receives each call through `ProbeHandle::dispatch(...)`
via a local hook set you build alongside the runtime.

```rust
use core::convert::Infallible;
use lutum::Lutum;
use lutum_eval::{
    Probe, ProbeDecision, ProbeHandle, ProbeRuntime, Score, TraceEvent, TraceSnapshot, maximize,
};

#[lutum::hooks]
trait ResponseQualityHooks {
    #[hook(singleton)]
    async fn validate_response(response: &str) -> Result<(), String> {
        let _ = response;
        Ok(())
    }
}

// -- Implement the probe ---------------------------------------------------------

#[derive(Default)]
struct ResponseQuality {
    violations: Vec<String>,
}

impl Probe for ResponseQuality {
    type Artifact = String;
    type Report = Vec<String>;
    type Error = Infallible;

    async fn on_trace_event(
        &mut self,
        _llm: &Lutum,
        _event: &TraceEvent,
    ) -> Result<ProbeDecision<Self::Report>, Self::Error> {
        Ok(ProbeDecision::Continue)
    }

    async fn finalize(
        &mut self,
        _llm: &Lutum,
        _trace: &TraceSnapshot,
        _artifact: &Self::Artifact,
    ) -> Result<Self::Report, Self::Error> {
        Ok(self.violations.clone())
    }
}

fn response_quality_hooks(
    dispatcher: ProbeHandle<ResponseQuality>,
) -> ResponseQualityHooksSet<'static> {
    ResponseQualityHooksSet::new().with_validate_response(move |response: &str| {
        let dispatcher = dispatcher.clone();
        let response = response.to_owned();
        async move {
            dispatcher
                .dispatch(move |probe| {
                    Box::pin(async move {
                        if response.contains("sorry") {
                            probe.violations.push(response.clone());
                            Err("response contains an apology".into())
                        } else {
                            Ok(())
                        }
                    })
                })
                .await
                .expect("probe dispatcher alive")
        }
    })
}

// -- Run -------------------------------------------------------------------------

let runtime = ProbeRuntime::new(ResponseQuality::default());
let hooks = response_quality_hooks(runtime.dispatcher());
let llm = Lutum::new(/* adapter */, /* budget */);

let scored = runtime
    .scored_by(&maximize(|violations: &Vec<String>| {
        Score::new_clamped(1.0 - (violations.len() as f32 / 10.0))
    }))
    .run_future(&llm, async { "Here is your answer.".to_string() })
    .await?;

let score = scored.score;
```

## Examples

All examples live under [crates/lutum/examples](crates/lutum/examples) and use
`OpenAiAdapter` behind the `openai` feature.

Set the endpoint once, then swap `<name>`:

```bash
export TOKEN="$OPENAI_API_KEY"
export MODEL="gpt-4.1-mini"
export ENDPOINT="https://api.openai.com/v1"

cargo run -p lutum --example <name> --features openai
```

- `TOKEN`: API key, or a dummy value like `local` for a local compatible endpoint
- `MODEL`: model name passed to `with_default_model(...)`
- `ENDPOINT`: OpenAI-compatible base URL such as OpenAI or `http://localhost:11434/v1`

| Example | What it shows | Run | Copy this when... |
|---|---|---|---|
| [`streaming_turn_ui`](crates/lutum/examples/streaming_turn_ui.rs) | Stream `TextTurnEvent` deltas directly to a UI or terminal | `cargo run -p lutum --example streaming_turn_ui --features openai` | You want the smallest streaming example without tools or transcript branching |
| [`react_loop`](crates/lutum/examples/react_loop.rs) | Explicit tool loop with `.tools::<T>()`, `NeedsTools`, and `round.commit(...)` | `cargo run -p lutum --example react_loop --features openai` | You want a ReAct-style loop but still keep tool execution in Rust |
| [`tool_hook`](crates/lutum/examples/tool_hook.rs) | Short-circuit selected tool calls with a cache, then fall back to normal tool execution | `cargo run -p lutum --example tool_hook` | You want to intercept tool calls without moving tool execution ownership out of app code |
| [`verification_gates`](crates/lutum/examples/verification_gates.rs) | Structured output checked by deterministic Rust gates and retried | `cargo run -p lutum --example verification_gates --features openai` | You want model output to pass strict post-validation before acceptance |
| [`deterministic_hooks`](crates/lutum/examples/deterministic_hooks.rs) | Prompt and output validation through typed hooks | `cargo run -p lutum --example deterministic_hooks --features openai` | You want reusable policy checks without baking them into every call site |
| [`policy_hook`](crates/lutum/examples/policy_hook.rs) | A configured Rust policy object plugged into a typed hook | `cargo run -p lutum --example policy_hook --features openai` | You want hook behavior to come from app-owned configuration instead of hard-coded closures |
| [`stateful_hook`](crates/lutum/examples/stateful_hook.rs) | A mutable hook that remembers rejected commands across retries | `cargo run -p lutum --example stateful_hook --features openai` | You want hook-local memory without pushing `Mutex` or retry bookkeeping into the rest of your app |
| [`rag`](crates/lutum/examples/rag.rs) | Retrieve supporting text, then answer only from grounded evidence | `cargo run -p lutum --example rag --features openai` | You want a minimal retrieval-augmented generation pattern |
| [`memory_surface`](crates/lutum/examples/memory_surface.rs) | Store and reload structured task state outside the transcript | `cargo run -p lutum --example memory_surface --features openai` | You want long-running state that is explicit and app-owned |
| [`entropy_gc`](crates/lutum/examples/entropy_gc.rs) | Compact long history into a summary, then continue the session | `cargo run -p lutum --example entropy_gc --features openai` | You want to shrink prompt footprint without giving up iterative workflows |

## Development

```bash
cargo check --workspace --all-targets
cargo test --workspace
```
