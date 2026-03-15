# Design

`agents` is a typed, streaming LLM runtime. It still intentionally avoids a top-level `Agent`
abstraction.

The key change in vNext is not hidden automation. It is a clearer split between:

- exact core state and execution
- thin request facades
- explicit convenience for transcript management

## Core principles

- execution stays in user code
- request replay and response capture use different algebras
- tool execution is explicit
- streaming is primary
- reduction is public
- provider details stay at the edge
- convenience must not reduce control

## No hidden `Agent`

The library still does not own the agent loop. The stable exact surface is:

- `Context<M>`
- `ModelInput`
- `AssistantTurn`
- public reducers

User code still decides:

1. what the model sees
2. when a turn starts
3. whether a result is committed to transcript state
4. how tool calls are executed
5. how branches, retries, approvals, and multi-agent handoff work

## Execution boundary

`Context<M>` remains the only official execution boundary.

That is where:

- `ModelInput` validation happens
- budget is reserved and finalized
- tracing/request ids are recorded
- streamed events are reduced into canonical results

Adapters are still public because providers need an SPI boundary, but adapter-direct execution
still bypasses the library's execution contract. `Context::new(...)` accepts concrete budget and
provider implementations and erases them behind `dyn BudgetManager<M>` and `dyn LlmAdapter`, so
backend type noise does not leak into application state.

## Canonical request surface

Request replay is represented by:

- `ModelInput`
- `ModelInputItem`
- `ToolUse`

The request algebra stays exact and provider-neutral:

- messages remain ordered
- assistant replay remains ordered
- tool call/result pairs stay bundled on the request side

The library does not collapse this into a hidden chat history abstraction.

## Canonical response surface

Completed model output is represented by:

- `AssistantTurn`
- `AssistantTurnItem`

This stays richer than the request side because the model can emit tool calls before any tool
results exist.

## Shared turn config

The old duplicated request structs have been replaced by shared config plus thin facades.

Shared config:

- `GenerationParams`
- `ReasoningParams`
- `TurnConfig<T>`
- `ToolPolicy<T>`

Thin facades:

- `TextTurn<T>`
- `StructuredTurn<T, O>`

This keeps:

- one shared place for model, generation, reasoning, tool, and budget policy
- separate public types for text vs structured output
- no dummy output type on text turns

The intended primary path is:

- `TextTurn::new(model)` / `StructuredTurn::new(model)` for top-level turn construction
- the model value itself is validated once via `ModelName::new(...)`
- `..Default::default()` only for nested parameter bundles such as `GenerationParams`
  and `ReasoningParams`

This keeps required fields required while still allowing concise partial overrides. All config and
facade types also derive `bon::Builder`, but builder usage is secondary.

## Tool selection

Tool typing still lives per turn.

`#[derive(Toolset)]` generates:

- `<Toolset>Call`
- `<Toolset>Selector`

`<Toolset>Selector` is:

- typed to its toolset
- serializable/deserializable
- JSON-schema capable

That makes tool subset selection usable in both Rust code and model-facing structured outputs.

The public selection API is now value-based:

- `ToolPolicy::Disabled`
- `ToolPolicy::AllowAll`
- `ToolPolicy::AllowOnly(Vec<T::Selector>)`
- `ToolPolicy::RequireAll`
- `ToolPolicy::RequireOnly(Vec<T::Selector>)`

This preserves type safety without a separate subset macro.

## Session

`Session<M>` is a transcript helper, not a higher-order runtime.

It owns:

- a `ModelInput`
- an execution `Context`
- optional turn defaults
- replay helpers

It deliberately does not own:

- hidden loops
- hidden tool execution
- hidden retries
- hidden transcript mutation

### Explicit commit model

`Session` is designed so convenience never hides control:

- `prepare_text(...)` / `prepare_structured(...)` do not mutate transcript state
- `collect*()` does not mutate transcript state
- transcript state changes only through explicit `commit_*`
- `snapshot()`, `input()`, `input_mut()`, and `into_input()` remain available
- `into_pending()` lets callers drop back to raw `Context`-style collection

This is what makes it safe for:

- branch evaluation
- human approval gates
- speculative execution
- planner/worker tool handoff
- external state machines and blackboards

## Structured output and selector planning

Structured output remains turn-local.

Because selectors are schema-bearing public types, a model can return:

- `Vec<AppToolsSelector>`

and a later turn can feed that directly into:

- `ToolPolicy::AllowOnly(...)`

That supports explicit multi-agent planning without introducing a framework-owned agent model.

## Streaming and reducers

Execution remains streaming-first.

The public surface is:

- start a turn with `Context` or `Session`
- stream events
- reduce through public reducers
- optionally stop via handler directives

Reducers remain public so replay and deterministic tests do not need internal code.

## Provider boundary

Provider-specific wire formats still live behind `LlmAdapter`.

The adapter boundary is intentionally object-safe and erased. Typed tool decoding, structured
output decoding, and canonical reduction stay in the core; adapters only translate between the
provider transport and erased canonical events.

The core algebra is not shrunk for adapter convenience. If a provider cannot represent the
canonical request faithfully, that remains an adapter limitation rather than a reason to weaken the
core.

## Summary

The point of the redesign is not to add abstraction layers that own behavior. The point is:

- shared config instead of duplicated request structs
- typed selector values instead of subset macros
- session convenience without hidden control flow
- examples that cover real application patterns without downstream-specific assumptions
