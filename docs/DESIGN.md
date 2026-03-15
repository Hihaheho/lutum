# Design

`agents` is a typed, streaming LLM runtime. It intentionally does not provide a top-level
`Agent` abstraction.

The crate is built around a few constraints:

- execution stays in plain user code
- request replay and response capture use different algebras
- tool execution is explicit
- streaming is primary
- reduction is public
- provider details stay at the edge

## Why there is no `Agent`

An `Agent` abstraction tends to hide the wrong things:

- what the model actually saw
- what state is durable
- whether a tool call came from the model or the framework
- how budget was spent
- where provider-specific compromises were made

This crate instead exposes:

- `Context<M, B, L>` for execution
- `ModelInput` as the canonical request surface
- `AssistantTurn` as the canonical response surface
- public reducers for deterministic event reduction

The "agent loop" is then just user code:

1. construct `ModelInput`
2. start a turn with `Context`
3. stream/reduce events
4. execute tools if needed
5. turn the resulting `AssistantTurn` back into request input

## Core runtime

The runtime is:

- `Context<M, B, L>`

where:

- `M: Marker`
- `B: BudgetManager<M>`
- `L: LlmAdapter`

`Context` is cloneable because it owns `Arc<B>` and `Arc<L>`. The traits themselves do not need
to be `Clone`.

### Marker

`Marker` is the caller-owned label for tracing and budget partitioning. The library does not
prescribe what a marker means.

### Budgeting

Budgeting is reservation-based.

There are two separate ideas:

- shared/global budget control through `BudgetManager`
- per-request constraints through `RequestBudget`

`ThinkingBudget` is separate again. It shapes model behavior, not spending policy.

This is intentional. A single "max tokens" number is too blunt once calls happen concurrently.

## Canonical request surface: `ModelInput`

Request replay is represented by:

- `ModelInput`
- `ModelInputItem`

`ModelInputItem` has exactly three cases:

- `Message { role: InputMessageRole, content: NonEmpty<MessageContent> }`
- `Assistant(AssistantInputItem)`
- `ToolUse(ToolUse)`

### Why request and response differ

By the time the next request is constructed, a tool call and its tool result should travel
together. So request-side tool state is bundled as:

- `ToolUse { id, name, arguments, result }`

This removes dangling "tool result without call" states from the public request surface.

### Why `ModelInput` is erased

Tool schemas and structured output schemas can change per turn. A full conversation cannot be
described correctly by one global generic parameter.

So the canonical request surface is erased where it needs to be:

- tool arguments/results use `RawJson`
- typed tool and structured semantics stay at the turn boundary

### Validation scope

`ModelInput::validate()` only checks provider-neutral invariants:

- input must be non-empty
- `ToolUse.id` values must be unique

It deliberately does not impose stronger temporal ordering rules. `ModelInput` is a full replay
surface, so the caller is allowed to preserve whatever valid order they intend.

## Canonical response surface: `AssistantTurn`

A completed model response is represented by:

- `AssistantTurn`
- `AssistantTurnItem`

`AssistantTurnItem` can be:

- `Text(String)`
- `Reasoning(String)`
- `Refusal(String)`
- `ToolCall { id, name, arguments }`

This is richer than request-side `AssistantInputItem` because a model can emit a tool call before
any tool result exists.

Tool calls stay on the response side because they are model-authored output. Tool results are
external observations.

## Typed tools and structured output

Typed semantics live on the turn, not in the canonical request algebra.

### Tools

`ToolInput` is the atomic primitive.

- `#[tool_input(output = T)]` defines reusable tool schema and output typing
- `#[derive(Toolset)]` on a normal enum closes over the allowed tool universe for a turn
- the derive generates a metadata-bearing `<ToolsetName>Call` enum and per-tool wrapper structs

Tool calls are surfaced in two forms:

- canonically as erased `AssistantTurnItem::ToolCall`
- ergonomically as generated wrapper values like `AppToolsCall::Weather(WeatherArgsCall)`

Each wrapper carries:

- parsed typed input
- `ToolMetadata { id, name, arguments }`

The raw explicit path is:

1. `match` on the generated wrapper enum
2. execute the tool in user code
3. call `call.tool_use(output)` to build `ToolUse`

`#[tool_fn(skip(...))]` is sugar on top of this model. It generates a `ToolInput` companion type
and a wrapper `.call(...)` method whose arguments are exactly the skipped parameters.

`AssistantTurn::into_input_items(...)` replays an assistant turn into request items while
replacing each `ToolCall` with exactly one matching `ToolUse`. Missing, extra, duplicate, or
mismatched tool uses are rejected.

### Structured output

Structured output is turn-local.

The request decides whether a turn is:

- plain text
- structured output

The result becomes:

- `StructuredTurnOutcome::Structured(O)`
- `StructuredTurnOutcome::Refusal(String)`

Refusal is treated as valid model behavior, not necessarily an execution failure.

## Streaming and public reducers

Execution is streaming-first, but the public surface is not "just expose a raw stream."

The model is:

- create a pending turn from `Context`
- call `collect(handler)`
- let the library reduce streamed events into canonical state and results

Reduction is public:

- `TextTurnReducer<T>`
- `StructuredTurnReducer<T, O>`
- `CompletionReducer`

This keeps replay and deterministic testing possible without re-implementing internal rules.

### Handlers

`collect(handler)` accepts `EventHandler`, with a blanket impl for compatible `FnMut`.

Control is explicit:

- `HandlerDirective::Continue`
- `HandlerDirective::Stop`

The handler sees read-only context:

- current reducer state
- marker
- remaining budget

There is no hidden imperative stop channel.

## Provider boundary

Provider-specific wire formats live behind `LlmAdapter`.

The current provider is:

- `OpenAiAdapter`

It targets OpenAI-compatible:

- `/v1/responses`
- `/v1/completions`

and is implemented directly with `reqwest`.

### Lowering rules

Core ordering must not be rewritten away for adapter convenience.

In particular:

- mid-conversation `System` and `Developer` items remain ordered
- assistant replay input remains ordered
- `ToolUse` lowers as adjacent `function_call` then `function_call_output`

If a provider cannot faithfully represent the canonical request, that is an adapter limitation, not
a reason to shrink the core algebra.

## Current scope

The crate is intentionally narrow today:

- text messages
- reasoning summaries
- refusals
- typed tools
- typed structured output
- OpenAI-compatible responses/completions provider

Not yet modeled:

- multimodal content
- arbitrary provider metadata channels
- built-in multi-turn executors

## Summary

The crate is lower-level than a typical "agent framework" on purpose.

The point is not to hide the loop. The point is to make the loop exact, typed, replayable, and
observable.
