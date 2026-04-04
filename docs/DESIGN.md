# Design

`agents` is a typed, streaming LLM runtime. It intentionally avoids a top-level `Agent`
abstraction.

## Core principles

- execution stays in user code
- tool execution is explicit
- streaming is primary
- provider details stay at the edge
- convenience must not reduce control
- the library must **not** eagerly normalize committed transcript state into a library-owned lossy canonical IR
- transcript storage must remain **exact and replayable**
- public transcript access must be **view-based**
- if users want a portable or application-specific IR, they derive it explicitly in user code

## No hidden `Agent`

The library does not own the agent loop.

User code decides:

1. what the model sees
2. when a turn starts
3. whether a result is committed to transcript state
4. how tool calls are executed
5. how retries, approvals, speculative execution, and multi-agent handoff work

## Execution boundary

`Context` is the only official execution boundary.

That is where:

- `ModelInput` validation happens
- budget is reserved and finalized via `OwnedLease`
- tracing and request ids are recorded
- streamed events are reduced into completed results

Adapters are public because providers need an SPI boundary, but adapter-direct execution
bypasses the library's execution contract. `Context::new(adapter, budget)` erases only
`dyn TurnAdapter` and `dyn UsageRecoveryAdapter`. Completion is wired explicitly through
`Context::from_parts(turns, completion, recovery, budget)`, which allows mixing providers —
for example Claude turns with an OpenAI-compatible completion adapter.

## Canonical request surface

Request input is represented by:

- `ModelInput`
- `ModelInputItem`
- `ToolUse`

`ModelInputItem` is a flat ordered enum:

- `Message { role, content }` — user/system/developer messages
- `Assistant(AssistantInputItem)` — assistant message items for manual construction
- `ToolUse(ToolUse)` — tool results following a tool-call turn
- `Turn(CommittedTurn)` — an exact adapter-owned committed turn at its natural position

The `Turn` variant is what makes context engineering work: committed turns live in the same
ordered list as user messages and tool results, so interleaving like
`[user_1, Turn_0, user_2, Turn_1]` is representable by construction.

`ModelInput` and `ModelInputItem` do **not** implement `Serialize` / `Deserialize`. The
canonical request algebra is an execution concern, not a persistence concern.

## Transcript storage model

Session history is stored as exact committed turns — not as a canonical library-owned IR.

- `CommittedTurn = Arc<dyn TurnView + Send + Sync>`
- exact committed turns are adapter-owned values
- `Session::list_turns()` returns an iterator over `&dyn TurnView`
- committed turns live inside `ModelInput` as `ModelInputItem::Turn` items
- adapters implement `TurnView` and `ItemView` over their exact concrete turn types
- adapters also implement `as_any()` on `TurnView` so the OpenAI adapter can downcast to
  `OpenAiCommittedTurn` for exact same-adapter replay

The library does not collapse session history into `Vec<CanonicalIrTurn>` or any equivalent.

## Why not a universal transcript IR

A universal transcript IR creates these problems:

- hidden information loss at commit
- forced normalization across incompatible transcript structures
- weaker same-adapter replay
- provider-specific details creeping into core through backpressure
- opinionated storage semantics imposed on all users

Most users only need to inspect assistant text, tool calls, and tool results — a view surface
solves that without a library-chosen IR.

If a user wants an application-specific IR, they derive it from `TurnView` / `ItemView`.

## View traits

Views are small, read-only, and fine-grained.

- `TurnView` — role, ordered item iteration
- `ItemView` — typed accessors: text, reasoning, refusal, tool call
- `TurnView::as_any()` — object-safe downcast for same-adapter replay

Views are for user inspection and user-driven reduction into app-specific models.

Views are **not** the adapter's replay mechanism.

## Completed results vs committed transcript turns

These are two distinct concepts.

**Completed result** — used immediately after execution:

- `TextTurnResult<T>`, `StructuredTurnResult<T, O>`
- carries `assistant_turn: AssistantTurn`, `tool_calls`, `finish_reason`, `usage`
- also carries `committed_turn: CommittedTurn` — the exact adapter-owned turn

**Committed transcript turn** — used for history and replay:

- exact, adapter-owned, view-backed
- replayable without canonicalization loss
- stored via `commit_text` / `commit_structured` / `commit_tool_round`

`AssistantTurn` and `AssistantTurnItem` remain useful completed result types for execution and
tool-round validation. They are not the session storage model.

## Exact transcript capture lifecycle

1. a turn starts
2. the adapter creates provider-native capture state
3. each provider-native streaming signal updates that capture state losslessly
4. on `response.completed`, the adapter seals capture state into an exact committed turn and
   includes it in the `ErasedTextTurnEvent::Completed` event
5. the reducer stores that `committed_turn` when processing the `Completed` event
6. explicit `commit_*` moves it into `ModelInput` as a `Turn` item

Core does not attempt `common event projection → exact transcript turn`.

## Adapter responsibilities

A `TurnAdapter` is responsible for:

1. translating canonical request facades into provider transport requests
2. handling `ModelInputItem::Turn` in request compilation — exact downcast first (`OpenAiCommittedTurn` / `ClaudeCommittedTurn`), `ItemView` projection fallback for cross-adapter turns
3. emitting exact committed turn values in `Completed` events
4. implementing `TurnView` / `ItemView` / `as_any()` over those exact values
5. accepting a `ModelSelector` for pluggable model routing
6. accepting a provider-specific `FallbackSerializer` for custom serialization of unrecognized fields

A `UsageRecoveryAdapter` is responsible for:

1. recovering `Usage` for a completed request by `request_id` and `OperationKind`
2. returning `None` when usage is unavailable rather than failing

Adapters are thin in the sense that they do not own hidden runtime behavior, but they do own
exact transport translation and transcript exactness.

## Replay model

Same-adapter replay is exact:

- `ModelInputItem::Turn(committed_turn)` carries the adapter-owned exact turn
- the adapter downcasts via `committed_turn.as_any().downcast_ref::<AdapterTurn>()`
- replay emits exact wire-format items without normalization loss

Cross-adapter replay falls back to `ItemView` projection. This is a userland concern, not the
library's primary job.

## Session

`Session` is a transcript helper, not a higher-order runtime.

It owns:

- a `ModelInput` (which contains `Turn` items for committed history)
- an execution `Context`
- optional turn defaults

It deliberately does not own:

- hidden loops
- hidden tool execution
- hidden retries
- hidden branch graphs
- eager normalization into a library-owned IR

### Explicit commit model

- `prepare_text(...)` / `prepare_structured(...)` do not mutate transcript state
- `collect*()` does not mutate transcript state
- transcript state changes only through explicit `commit_*`
- `input()`, `input_mut()`, and `into_input()` give direct access to the ordered input
- `into_pending()` drops back to raw `Context`-style collection

`commit_text` and `commit_structured` are infallible — they push `ModelInputItem::Turn`.

`commit_tool_round` validates tool use ordering against the committed turn's `AssistantTurn`
and returns `Result<(), AssistantTurnInputError>`.

This keeps it safe for branch evaluation, approval gates, speculative execution, and
planner/worker handoff.

### Branching

`Session::clone()` is the branch mechanism for now. The library does not introduce transcript
graph semantics or first-class branch structure at this stage.

## Shared turn config

Shared config replaces the old duplicated request structs:

- `GenerationParams` — temperature, max_output_tokens, seed
- `TurnConfig<T>`
- `ToolPolicy<T>`

Thin facades:

- `TextTurn<T>`
- `StructuredTurn<T, O>`

`ReasoningParams` no longer exists in core. Reasoning is provider-owned: `OpenAiAdapter`
accepts a `ReasoningEffortResolver`; `ClaudeAdapter` accepts a `BudgetTokensResolver` for
extended thinking. Provider-specific reasoning config is passed via `RequestExtensions`.

## Tool selection

`#[derive(Toolset)]` generates:

- `<Toolset>Call`
- `<Toolset>Selector`

The public selection API is value-based:

- `ToolPolicy::Disabled`
- `ToolPolicy::AllowAll`
- `ToolPolicy::AllowOnly(Vec<T::Selector>)`
- `ToolPolicy::RequireAll`
- `ToolPolicy::RequireOnly(Vec<T::Selector>)`

Because selectors are schema-bearing public types, a model can return `Vec<AppToolsSelector>`
and a later turn can feed that directly into `ToolPolicy::AllowOnly(...)`. That supports
explicit multi-turn planning without a framework-owned agent model.

Selectors also retain a typed path back to `ToolDef`:

- `selector.definition()`
- `Toolset::definitions_for(selectors)`

## Streaming and reducers

Execution is streaming-first.

- start a turn with `Context` or `Session`
- stream typed events via `into_stream()`
- or collect through `collect*()` with an optional event handler

`TextTurnState` and `StructuredTurnState` are the public partial state types. Reducers
(`TextTurnReducer`, `StructuredTurnReducer`) are thin wrappers around state; they are
implementation details, not the primary public concept.

The reducer stores the `committed_turn` from the `Completed` event during `apply()`.
`into_result()` uses that internally — it does not accept a caller-supplied committed turn.

Budget leases are held by `OwnedLease`. When `into_stream()` is called and the stream is
abandoned, `OwnedLease::drop()` releases the reserved capacity by recording zero usage.

## Completion APIs

There are now two completion-style APIs:

- `Context::completion` — raw text completion
- `Context::structured_completion` — prompt-based structured output without tools

`Context::completion` is the lower-level raw text path. It uses:

- `CompletionRequest` — model, prompt, `CompletionOptions` (temperature, max_output_tokens, stop), budget
- `CompletionEvent` — `Started`, `TextDelta(String)`, `Completed { finish_reason, usage }`
- `CompletionAdapter::completion(request, &extensions) -> CompletionEventStream`

`Context::structured_completion` is the tool-less structured path for prompt-based tasks that do
not need transcript integration. It uses:

- `StructuredCompletionRequest<O>` — model, optional system prompt, prompt, `GenerationParams`, budget
- `StructuredCompletionEvent<O>` — `Started`, `StructuredOutputChunk`, `StructuredOutputReady`, `ReasoningDelta`, `RefusalDelta`, `Completed`
- `CompletionAdapter::structured_completion(request, &extensions) -> ErasedStructuredCompletionEventStream`

Unlike turn-based execution, neither completion API produces a `CommittedTurn` or integrates with
the transcript model.

`Context::new(...)` does not configure a completion adapter. Calling either completion API on that
context returns an error; use `Context::from_parts(...)` when completion is needed.

If you need transcript/session/replay but still do not need tools, use `StructuredTurn::<NoTools, O>`.

Budget reservation, tracing, and usage recovery follow the same path as turn execution.

## Provider boundary

Provider-specific wire formats live behind three focused traits:

- `TurnAdapter` — `text_turn` / `structured_turn`: streaming conversation turns
- `CompletionAdapter` — `completion`: non-turn raw text completion
- `UsageRecoveryAdapter` — `recover_usage`: async usage recovery for providers (e.g.
  OpenRouter) that return usage out-of-band

The adapter boundary is intentionally object-safe and erased. Typed tool decoding, structured
output decoding, and canonical reduction stay in core; adapters only translate between the
provider transport and erased canonical events.

The core algebra is not shrunk for adapter convenience. If a provider cannot represent the
canonical request faithfully, that remains an adapter limitation.

### Available adapters

- `agents-openai` — OpenAI Responses API plus OpenAI-compatible raw completion support
- `agents-claude` — Anthropic Claude Messages API with lossless thinking-block replay
- `agents-openrouter` — `OpenRouterGenerationClient` implements `UsageRecoveryAdapter`
  via `GET /api/v1/generation`; can be composed with any `TurnAdapter` via `Context::from_parts`

## RequestExtensions

`RequestExtensions` is a per-request type-map for execution metadata.

The library treats entries as opaque. It passes `Arc<RequestExtensions>` to:

- `BudgetManager` methods so implementors can extract identity, routing keys, or cost centers
- `EventHandler` so handlers can inspect request metadata during streaming
- `CompletionAdapter::completion` for routing context
- `AdapterTextTurn` / `AdapterStructuredTurn` so adapters can inspect extensions at request compilation time
- `ModelSelector::select_model` to drive dynamic model routing

Users insert typed values; adapters read typed values. There is no shared schema.

## ModelSelector and FallbackSerializer

Both `OpenAiAdapter` and `ClaudeAdapter` accept a pluggable `ModelSelector`:

- `ModelSelector::select_model(&RequestExtensions) -> ModelSelection` returns an optional primary model override and optional fallback models
- The selected model is threaded through SSE/completion fallback mapping
- `ModelSelection` contains `primary: Option<Cow<'static, str>>` and `fallbacks: Option<Vec<Cow<'static, str>>>`

Each adapter also accepts a provider-specific `FallbackSerializer` for serializing unrecognized request fields without serde_json::Value mutation.

## Core vs edge

### Core owns

- `Context` — canonical execution boundary; holds `TurnAdapter`, an explicitly supplied `CompletionAdapter`, and `UsageRecoveryAdapter`
- `ModelInput` / `ModelInputItem` — canonical request algebra
- `TurnView` / `ItemView` — transcript view traits
- `BudgetManager` / `OwnedLease` — budget lifecycle
- `Session` — transcript helper and explicit commit model
- `RequestExtensions` — per-request opaque type-map
- `ModelSelector` / `ModelSelection` — pluggable model routing protocol
- public event, state, and result types
- typed tools and turn config
- `TurnAdapter` / `CompletionAdapter` / `UsageRecoveryAdapter` — provider SPI traits

### Edge adapters own

- exact transport request/response mapping
- exact committed transcript turn representation (`OpenAiCommittedTurn`, `ClaudeCommittedTurn`, etc.)
- `TurnView` / `ItemView` / `as_any()` implementations
- replay request compilation from exact committed turns
- provider-specific transport and metadata handling
- provider-specific reasoning config resolution (`ReasoningEffortResolver`, `BudgetTokensResolver`)
- adapter-local `FallbackSerializer` for unrecognized fields

### Core does not own

- provider-specific transcript enums or turn types
- a universal transcript normalization layer
- a mandatory persistence strategy (`Any`, downcast registry, `typetag`)
- eager lossy canonicalization at commit
- transcript branch graph semantics (for now)
- a universal app-facing transcript IR
- reasoning configuration (provider-owned)
