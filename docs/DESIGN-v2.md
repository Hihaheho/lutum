# `agents` transcript and adapter design notes

This document captures the current design direction for transcript storage, replay, adapter boundaries, and public observation APIs.

It is intentionally aligned with these goals:

- execution stays in user code
- request replay and response capture use different algebras
- tool execution is explicit
- streaming is primary
- streamed events are fully observable
- user-facing state transitions should read like ordinary Rust
- provider details stay at the edge
- convenience must not reduce control

It also reflects an important correction:

- the library should **not** eagerly normalize committed transcript state into a library-owned lossy canonical IR
- transcript storage should remain **exact and replayable**
- public transcript access should be **view-based**
- if users want a portable or application-specific IR, they should derive it explicitly in user code

---

## Summary

The library should not store session history as `Vec<CanonicalIrTurn>` or any similar eager normalized transcript representation.

Instead:

- `Session<M>` stores **exact committed turns**
- exact committed turns are **adapter-owned serializable values**
- the adapter boundary is primarily:
  - serde roundtrip
  - request compilation
  - exact streaming event delivery
  - common event projection views
  - public transcript views
- users inspect transcript state through **view traits**
- replay does **not** reconstruct through a canonical transcript IR
- replay uses adapter-native exact transcript data
- turn collection stays available, but manual event folding should be expressed through turn state/result types rather than a separate public `Reducer` concept
- branching is not a first-class transcript graph feature for now; callers can branch with `Session::clone()`

The consequence is deliberate:

- the library owns execution contracts and transcript management convenience
- the library does **not** impose an opinionated universal transcript IR
- the library preserves exact replay fidelity for same-adapter flows
- application-specific IR remains a userland concern

---

## What problem this solves

A lossy canonical transcript IR creates several problems:

1. same-adapter roundtrip fidelity is weakened
2. adapter-specific assistant turn ids and replay metadata are lost
3. impedance mismatches between provider-native transcript structure and canonical transcript structure are forced into core
4. commit performs hidden destructive normalization
5. users pay abstraction cost for a universal transcript representation they often do not want

In typical LLM applications, callers mostly need:

- to execute a turn
- to inspect the resulting turns
- to explicitly decide whether to commit
- to clone or fork a session
- to rebuild future model input from exact committed history

That does **not** require a library-owned portable transcript IR.

A transcript view surface is enough.

---

## Design principles

### 1. Session storage must be exact

Committed session history must remain replayable without hidden lossy normalization.

That means:

- the library should not rewrite committed turns into a canonical transcript DTO
- adapters should be able to persist exact committed turn values
- same-adapter replay should use exact stored turn data

### 2. Views are for users

Views exist so callers can inspect transcript state in a provider-neutral way.

Views are **not** the adapter's persistence format.

The adapter should not need to reconstruct transcript state from view objects. Instead:

- adapters store exact turn values
- adapters roundtrip those values with `Serialize` / `Deserialize`
- adapters also implement public view traits over those exact values

### 3. Shared execution semantics are projections, not storage

Shared public types still make sense where exact execution semantics matter, especially around:

- request construction
- turn configuration
- validation
- common event projections
- turn state progression
- completed result handling

But those shared types should not automatically become the storage model of `Session`, and they should not force streaming to discard provider-specific details.

### 4. Convenience must not hide destructive normalization

`commit_*` should only commit exact turn state.

It should not silently transform provider-native exact transcript state into a weaker library-owned normalized form.

### 5. Provider details stay at the edge

The core crate must not encode provider names or provider-specific enums as part of its transcript storage model.

No core types like:

- `enum RawEnvelope { OpenAi(...), Claude(...) }`
- provider-specific transcript item enums in core
- adapter-specific id fields on canonical core transcript types

Provider-specific exact transcript data belongs in provider crates behind erased adapter boundaries and public-neutral view traits.

### 6. Branch structure is out of scope for now

The library should not introduce transcript graph ownership or first-class branch structure in `Session` at this stage.

For now, branching is explicit and simple:

- callers use `Session::clone()`

That keeps the transcript model linear and avoids premature complexity.

---

## Stable execution surface

The stable exact public execution surface remains:

- `Context<M>`
- `ModelInput`
- `AssistantTurn`
- request-side sidecars for execution hints such as cache hints
- public event view traits and common projection enums
- public turn state/result types

User code still decides:

1. what the model sees
2. when a turn starts
3. whether a result is committed to transcript state
4. how tool calls are executed
5. how retries, approvals, speculative execution, and multi-agent handoff work

No hidden top-level `Agent` abstraction is introduced.

---

## Reducer should not be a first-class user concept

`Reducer` is a reasonable implementation technique, but it is not a good primary noun for the public API.

The user's job is usually one of these:

- execute a turn
- observe events
- inspect partial progress
- finalize a completed turn
- commit the result if desired

Those are naturally expressed in Rust as values and state transitions.

For example, this reads like direct domain code:

- `state.apply(&event)?`
- `let result = state.finish()?`

This reads like library-internal machinery:

- `let mut reducer = TextTurnReducer::new()`
- `reducer.apply(&event)?`
- `let result = reducer.into_result()?`

The distinction matters because the common path already has `collect*()`.

If the library also elevates `Reducer` as a marquee public concept, it teaches the wrong mental model:

- users start thinking in terms of folding machinery instead of turns
- the API exposes implementation vocabulary rather than domain vocabulary
- handlers and tests end up carrying an extra wrapper even though the useful thing is the partial turn state itself

The better direction is:

- keep streamed events fully observable
- keep partial turn state public
- keep completed result types public
- let `collect*()` remain the primary convenience path
- if manual folding is needed, do it through the state type, not a separate reducer type

Concretely, the public shape should be closer to:

- `TextTurnState<T>::apply(&mut self, &dyn TextEventView<T>)`
- `TextTurnState<T>::finish(self) -> Result<TextTurnResult<T>, _>`
- `StructuredTurnState<T, O>::apply(&dyn StructuredEventView<T, O>)`
- `StructuredTurnState<T, O>::finish(...)`
- `CompletionTurnState::apply(&dyn CompletionEventView)`
- `CompletionTurnState::finish(...)`

Optional convenience helpers such as `from_events(...)` are fine, but the important point is that the state value is the user-facing abstraction.

If a compatibility shim is needed, `*TurnReducer` can temporarily exist as a thin wrapper around `*TurnState`, but it should not remain the design center.

---

## Event surface should mirror transcript philosophy

Streaming events are also provider-shaped and heterogeneous.

So the event design should follow the same honesty as transcript design:

- the primary stream should carry exact adapter-owned event values
- user inspection should happen through view traits
- common cross-provider semantics should be exposed as borrowed projection enums
- provider-specific events must not disappear just because they do not fit the common projection

Conceptually, a text event can look like:

- `trait TextEventView<T> { fn project(&self) -> TextEventRef<'_, T>; }`
- `match event.project() { ... }`

That is a good fit.

The important clarification is what `project()` means.

It returns the borrowed common event IR for that exact event.

It is not just a discriminator.

So the design should be:

- the exact event object is the primary streamed value
- `project()` returns the provider-neutral borrowed event IR that generic code can act on
- the exact event object also remains available for provider-specific inspection

This avoids the user-hostile situation where:

- some provider events are silently dropped
- users must read implementation code to learn what was ignored
- the library claims portability by hiding information

Instead, every streamed provider event should be observable through the primary stream.

The common projection enum should be a lens over that stream, not a lossy replacement for it.

### Suggested shape

For example:

- `trait TextEventView<T>: Send + Sync { fn project(&self) -> TextEventRef<'_, T>; }`
- `enum TextEventRef<'a, T> { Started { request_id: Option<&'a str>, model: &'a str }, TextDelta { delta: &'a str }, ReasoningDelta { delta: &'a str }, ToolCallReady(&'a T::ToolCall), Completed { request_id: Option<&'a str>, finish_reason: &'a FinishReason, usage: &'a Usage }, ProviderSpecific(&'a dyn ProviderEventView) }`

The same pattern applies to structured and completion events.

This gives generic code a stable projection while preserving full observability.

### Provider-specific event view

`ProviderEventView` is the escape hatch for exact event details that are not promoted into the shared projection.

Its job is not to erase provider identity completely.

Its job is to make provider-specific handling explicit and inspectable.

Conceptually it should provide:

- provider identity
- provider-specific event name/category
- provider-owned metadata accessors as needed

For example conceptually:

- `trait ProviderEventView { fn provider(&self) -> ProviderId; fn name(&self) -> &str; }`

Provider adapters can then expose richer typed helpers on their own exact event types without forcing those details into the core cross-provider surface.

### State progression must depend only on the shared projection

This boundary should also be explicit.

`TextTurnState::apply(...)`, `StructuredTurnState::apply(...)`, and `CompletionTurnState::apply(...)` should advance state using only the shared projected event IR.

In other words:

- generic turn-state progression must not depend on provider-specific event payloads
- provider-specific events may be observed by handlers and application code
- but they should not be required for the core state machine to remain coherent

That is what makes the state machine portable while still keeping full streamed observability.

---

## Session model

`Session<M>` is a transcript helper, not a higher-order runtime and not a canonical transcript IR container.

It owns:

- a `Context`
- committed exact transcript turns
- replay helpers
- optional turn defaults

It deliberately does not own:

- hidden loops
- hidden tool execution
- hidden retries
- hidden branch graphs
- eager normalization into a library-owned portable IR

### Explicit commit model

The explicit commit model remains:

- `prepare_text(...)` / `prepare_structured(...)` do not mutate transcript state
- `collect*()` does not mutate transcript state
- transcript state changes only through explicit commit
- `snapshot()`, `input()`, `input_mut()`, and `into_input()` remain available where applicable
- `into_pending()` remains available for dropping back to raw collection style

This section is describing explicit commit semantics, not fixing a persistence API shape.

In particular:

- this document is not using `snapshot()` to imply a core-owned serialized transcript boundary
- the exact persistence form of committed turns remains adapter-owned
- the presence of `snapshot()` here should not be read as a commitment to a universal library DTO or envelope

The meaning of commit becomes stricter:

- commit stores exact replayable transcript state
- commit does **not** perform hidden canonical transcript rewriting

---

## Transcript storage model

The storage model should be conceptualized as:

- a linear ordered list of committed exact turns
- each exact turn is adapter-owned
- each exact turn is serializable/deserializable
- each exact turn exposes provider-neutral read-only views

The core library should not require knowledge of the concrete exact turn type.

Instead, the adapter SPI should erase it behind object-safe traits.

Conceptually:

- adapter stores exact turn payloads
- adapter can roundtrip them with serde
- adapter can expose `TurnView` / `ItemView`
- adapter can compile replay input from exact stored turns

### Concrete type ownership and runtime erasure

The concrete Rust type for an exact committed turn lives in the adapter crate.

That means:

- the adapter owns the concrete exact-turn type definition
- serde roundtrip is centered on that adapter-defined concrete type
- the core library does not define a universal serialized exact-turn DTO

The object-safe erasure described above is a runtime abstraction boundary.

In other words:

- core may hold adapter-owned exact turns behind object-safe transcript traits at runtime
- that runtime erasure is not itself the persistence contract
- this document is not saying that core serializes or deserializes `Box<dyn ...>` directly

The important distinction is:

- adapter-defined concrete types are the persistence types
- erased trait objects are the runtime abstraction used so core does not need to know those concrete types

---

### Example

For example, an OpenAI adapter could define an `OpenAiCommittedTurn` concrete type that derives
`Serialize` and `Deserialize` and implements the relevant transcript view traits.

Core would then interact with that value through erased traits for runtime transcript handling and
replay, while serde roundtrip would remain centered on `OpenAiCommittedTurn` itself.

This example is only meant to show the ownership boundary. It does not freeze the exact public API
spelling.

---

## Public transcript access is view-based

The primary public transcript API should be observation-oriented.

Example direction:

- `Session::list_turns() -> impl Iterator<Item = &dyn TurnView>`

The important part is not the exact return type spelling, but the contract:

- users read transcript state through neutral views
- users do not need to know adapter-native exact types
- users are not forced into a library-defined transcript IR

### Why views are sufficient

In most applications, callers want to:

- inspect assistant text
- inspect tool calls
- inspect tool results
- inspect user turns
- derive summaries, memory, or app-specific state

That is a read-oriented problem.

A transcript view surface solves it without requiring the library to choose a universal IR for everyone.

---

## View traits

Views should be small, read-only, and fine-grained.

They should not be giant DTO structs.

They should expose exactly the kinds of inspection typical applications need.

Suggested shape:

### Session-level

A session-level transcript view may provide:

- ordered turn iteration
- length queries
- emptiness queries

For example conceptually:

- `list_turns()`
- `len()`
- `is_empty()`

### Turn-level

A turn view may provide:

- role or turn category
- ordered item iteration
- optional metadata access where metadata is truly neutral

The design should avoid provider-specific assumptions.

### Item-level

An item view may provide typed accessors such as:

- text
- tool call
- tool result
- structured payload
- attachments or media if supported by the adapter

Item access should be ordered and inspectable without forcing users to allocate new transcript structures.

### Important constraint

Views are for user inspection and user-driven reduction into app-specific models.

Views are **not** the adapter's replay mechanism.

---

## Adapter responsibilities

Adapters should be thin in the sense that they do not own hidden runtime behavior, but they do own exact transport translation and transcript exactness.

An adapter is responsible for:

1. translating canonical request facades into provider transport requests
2. emitting exact streamed event values for the turn
3. implementing common event projection views over those exact event values
4. capturing exact committed transcript turns
5. roundtripping exact committed transcript turns with serde
6. implementing public-neutral view traits over exact committed transcript turns
7. compiling replay input from exact committed transcript turns

That means the adapter is effectively:

- a serde-derive-backed wrapper around exact transport-adjacent turn data
- plus exact event wrappers/views for streaming
- plus implementations of transcript view traits
- plus request compilation logic
- plus common semantic projection logic

This is intentionally narrower and cleaner than a design where adapters reconstruct from a lossy canonical transcript IR.

This section is also intentionally not prescribing one Rust implementation technique for type
erasure or persistence wiring.

For example, it does not require:

- `Any`
- downcasting
- registries
- `typetag`

If an adapter needs one of those internally, that is an adapter implementation choice rather than
part of this document's public contract.

---

## Exact transcript capture lifecycle

This is an important point and the design should be explicit:

- exact transcript turns must **not** be reconstructed in core from projected common event semantics
- exact transcript capture happens in the adapter layer, from provider-native transport data
- capture should happen incrementally while streaming, not by trying to rebuild later from reduced/common projections

The intended flow is:

1. a turn starts
2. the adapter creates provider-native capture state for that turn
3. each provider-native streaming signal updates that capture state losslessly
4. the adapter emits an exact streamed event object for that signal
5. that event object exposes a common semantic projection through its view trait
6. when the provider turn completes, the adapter seals the capture state into an exact committed-turn candidate
7. explicit `commit_*` stores that exact turn into `Session`

So streamed event delivery and exact transcript capture happen in parallel, but they serve different purposes:

- exact streamed events are for user code, handlers, diagnostics, and provider-specific inspection
- common event projections are for generic turn-state progression
- exact capture is for persistence, replay, and transcript fidelity

This also means:

- core should not attempt `common event projection -> exact transcript turn`
- `commit_*` should commit the adapter-produced exact turn candidate associated with the completed step
- if a caller drops the step result instead of committing, the exact turn candidate is discarded rather than silently written into transcript state

Early stop/error semantics should be conservative by default:

- if the turn did not reach a commit-worthy completed state, there is no exact committed turn to store

Adapters may internally keep partial capture state for recovery or diagnostics, but that is not the same as a committed transcript turn.

---

## Common event projection scope

The common event projection should be:

- platform-agnostic
- execution-oriented
- explicitly partial

Its job is to answer:

- what generic user code can act on across providers
- what generic turn state can advance from

Its job is **not** to define the universe of observable streamed events.

So the common projection enum should contain semantics such as:

- turn started
- text delta
- reasoning delta
- refusal delta
- tool call progress/readiness
- turn completed
- provider-specific event passthrough

The `ProviderSpecific(...)` branch is important.

It makes the partial nature of the common projection explicit instead of pretending the common enum is exhaustive over reality.

If a provider-specific feature later proves to have stable cross-provider semantics, it can be promoted from `ProviderSpecific(...)` into a first-class common variant deliberately.

---

## Replay model

Replay is exact and adapter-native.

It is **not** defined as:

- committed turns -> library-owned canonical transcript IR -> adapter request

Instead, replay is defined as:

- committed exact adapter-owned turns -> adapter request compilation

This avoids fidelity loss and avoids forcing the core library to model every adapter's transcript structure as a universal algebra.

### Same-adapter replay

Same-adapter replay should be high-fidelity by default because:

- committed exact turns remain exact
- no hidden normalization occurred at commit
- adapter-specific ids and structural boundaries remain available to the adapter

### Cross-adapter replay

Cross-adapter portability is not the transcript model's primary job.

If callers want cross-adapter replay or an application-specific portable transcript abstraction, they can derive it explicitly from transcript views.

That is a userland concern.

The library should not impose one canonical cross-adapter transcript IR as the default storage model.

---

## Canonical request surface

The canonical request surface still matters and remains exact.

Request replay is represented by:

- `ModelInput`
- `ModelInputItem`
- `ToolUse`

The request algebra remains:

- ordered
- explicit
- provider-neutral
- exact enough for validation and execution

This is not in conflict with exact transcript storage.

The key distinction is:

- request-side canonical types are part of the execution boundary
- transcript-side committed storage is not automatically rewritten into those same canonical shapes

---

## Request-side cache hints

Prompt cache planning should be modeled as request-side execution metadata, not transcript content.

That means:

- cache hints are not stored in `ModelInput`
- cache hints are not committed into exact transcript turns
- transcript persistence and `snapshot()` do not imply cache hint persistence
- cache hints may influence transport compilation for a single request

The intended shape is a typed sidecar attached to the user-facing request/session surface.

Conceptually:

- `RequestSidecar<H>`
- `Session<M, H>`
- `ErasedRequestSidecar`
- `ErasedCacheHint`

This gives users a typed API while allowing the adapter boundary to remain object-safe.

### Single cache hint slot

A request should have at most one cache hint slot.

If a caller needs more sophistication, the caller should supply a richer hint type rather than stacking multiple unrelated hint values.

This keeps the model simple:

- no priority lattice
- no heterogeneous hint merge rules
- no multiple-hint conflict semantics at request execution time

### Adapter-local hint handlers

Cache hint lowering is adapter-local.

Conceptually:

- `OpenAiCacheHintHandler<H>`
- `ClaudeCacheHintHandler<H>`
- `VertexCacheHintHandler<H>`

An adapter instance may own a registry keyed by hint type so the same application code and same backend can behave differently depending on which handler is installed.

What remains provider-local is:

- boundary lowering
- TTL mapping
- fallback/strictness handling
- provider-specific warnings and telemetry

### Transport-draft decoration boundary

Cache hint handlers should operate on a provider request draft after canonical request compilation but before serialization and network I/O.

That means:

- user-owned logical input remains unchanged
- handlers may attach provider cache metadata to the transport draft
- the decorated draft is transient and only used for the outbound request

This is the right place for:

- Claude `cache_control`
- OpenAI `prompt_cache_key` / `prompt_cache_retention`
- Vertex `cachedContent` references or equivalent explicit-cache lowering

This preserves the principle that user code decides the logical request while still allowing adapters to inject provider-specific cache metadata at the edge.

### Built-in and user-defined hints

The library may ship a small set of built-in hint types, such as:

- a simple single-boundary hint
- a richer segmented/best-effort hint
- a no-cache-hint marker type

Users should also be able to define their own hint type and install matching adapter-local handlers.

Because provider capabilities differ, hint semantics are intent-based rather than guaranteed feature parity.

This means a handler may:

- apply the hint exactly
- approximate it
- ignore it
- return an error

That decision belongs to the adapter-local handler, not to transcript storage or canonical request types.

---

## Canonical response surface

Completed model output may still be represented by:

- `AssistantTurn`
- `AssistantTurnItem`

This remains useful as a completed result type for execution and finalization.

But its role should be precise:

- `AssistantTurn` is a completed result surface
- it is not necessarily the persistent storage representation inside `Session`
- committing a completed result should preserve exact adapter transcript state, not force transcript storage into `Vec<AssistantTurn>`

This distinction is important.

A result type can be canonical without becoming the storage model.

---

## Relationship between completed results and committed transcript turns

There are two legitimate concepts here:

### Completed result

Used immediately after execution:

- exact enough for handlers and state finalization
- useful for collection APIs
- useful for tests and deterministic replay of common event projections

### Committed transcript turn

Used for transcript history and replay:

- exact
- adapter-owned
- serializable/deserializable
- view-backed
- replayable without canonicalization loss

These do not have to be the same concrete type.

That separation is acceptable and often desirable.

---

## Structured output and typed tools

The existing typed turn-local tool design remains good.

`#[derive(Toolset)]` can continue generating:

- `<Toolset>Call`
- `<Toolset>Selector`

`ToolPolicy<T>` can remain value-based:

- `Disabled`
- `AllowAll`
- `AllowOnly(Vec<T::Selector>)`
- `RequireAll`
- `RequireOnly(Vec<T::Selector>)`

Structured output also remains turn-local.

None of this requires a library-owned canonical transcript IR.

These are execution concerns, not transcript storage concerns.

---

## Streaming and turn progress

Execution remains streaming-first.

The public execution pattern remains:

- start a turn with `Context` or `Session`
- stream events
- inspect provider-specific details if desired
- inspect partial turn state if needed
- finalize into a completed result if needed
- optionally stop via handler directives

The important part is that streamed events are public and fully observable, while common projections remain stable and deterministic for generic code.

The crucial refinement is this:

- manual folding should use public turn state types rather than a standalone reducer abstraction
- committed transcript state is not required to be stored as the shared completed result type
- collection convenience can internally use whatever machinery it wants, but that machinery should not define the user's mental model
- provider-specific streamed events must remain observable even when generic collection only reacts to common projections

Again, runtime algebra and transcript persistence model are different concerns.

---

## Serialization and persistence

Exact committed transcript turns should be serializable and deserializable by the adapter.

This gives the library and downstream applications:

- persistence
- process restarts
- test fixtures
- session cloning
- exact same-adapter replay

without requiring the core crate to know any provider-specific transcript representation.

The persistence contract should be centered on the adapter's exact committed turn type, not a universal transcript DTO.

This section is about responsibility, not about introducing a new core-level persistence API.

In particular, this design does not define:

- a universal serialized snapshot DTO in core
- a core-owned opaque payload envelope for exact turns
- a requirement that core serialize or deserialize `Box<dyn ...>` directly

What it does define is narrower:

- adapters own the concrete exact committed turn types
- adapters own serde roundtrip for those concrete types
- core stays provider-neutral at the runtime abstraction boundary

---

## Why not a universal canonical transcript IR

A universal transcript IR sounds attractive but is a poor default for this library.

It tends to create these problems:

- hidden information loss at commit
- forced normalization across incompatible transcript structures
- provider-specific details creeping into core through backpressure
- weaker same-adapter replay
- opinionated storage semantics imposed on all users
- maintenance burden in core for every transcript shape mismatch

Most users do not benefit enough from that cost.

If a user wants an application-specific IR, that user can derive it from `TurnView` / `ItemView`.

That is more honest and more flexible.

---

## What the library should provide instead of a transcript IR

The library should provide:

1. exact execution boundary
2. exact committed transcript storage
3. explicit commit control
4. view-based transcript inspection
5. exact same-adapter replay
6. session cloning
7. public event/state/result semantics
8. typed turn-local tools
9. thin request facades
10. no hidden agent loop

This is a much cleaner and more composable foundation.

---

## Branching

For now, branching should stay simple and explicit.

The library should not yet make branch graphs part of `Session`'s semantic model.

Instead:

- `Session::clone()` is the branch mechanism

This is enough for:

- speculative execution
- approval gates
- alternative tool outcomes
- branch evaluation

It avoids premature complexity in transcript identity and lineage semantics.

If explicit branch metadata is ever introduced later, it can be added deliberately rather than smuggled into the first transcript design.

---

## Core vs edge split

### Core should own

- `Context<M>`
- canonical request validation
- request-side sidecar carriers for execution hints
- budget reservation/finalization
- tracing and request ids at execution time
- streaming collection/finalization semantics
- `Session<M>` as transcript helper
- explicit commit model
- transcript view traits
- public turn events and turn state/result types
- typed tools
- turn config and generation/reasoning config

### Edge adapters should own

- exact transport request/response mapping
- cache hint handler registries and request-draft cache lowering
- exact committed transcript turn representation
- serde roundtrip for committed turns
- replay request compilation from exact committed turns
- implementations of transcript view traits
- provider-specific transport and metadata handling

### Core should not own

- provider-specific transcript enums
- provider-specific exact-turn serialization formats
- universal transcript normalization logic
- a core-level persistence API for adapter exact turns
- eager lossy canonicalization at commit
- a mandated transcript-persistence strategy based on `Any`, downcast, registry, or `typetag`
- transcript branch graph semantics for now
- a universal app-facing transcript IR

---

## Consequences for API design

### Good direction

- `Session::list_turns() -> impl Iterator<Item = &dyn TurnView>`
- exact committed turns stored internally
- users derive their own summaries or IRs from views
- adapters replay from exact stored turns
- session clone is the branch mechanism

### Bad direction

- `Vec<CanonicalIrTurn>` as session storage
- commit that rewrites exact turns into library-owned transcript DTOs
- replay that goes through a lossy transcript normalization step
- provider-specific transcript enums in core
- hidden branch graph semantics in early versions

---

## Practical user experience

The intended user experience is:

1. construct a turn request
2. execute with `Context` or `Session`
3. stream and collect as needed
4. explicitly commit if desired
5. inspect transcript turns through views
6. clone session to branch
7. build any application-specific IR explicitly from transcript views if needed

This keeps convenience without taking control away.

For rare cases where a caller wants to replay event semantics manually, the code should still look like normal Rust operating on a turn state value, not like framework-specific folding machinery.

---

## Final position

The library should not treat a canonical transcript IR as the central truth of session state.

Instead:

- session state should remain exact
- adapters should own exact committed turn representations
- adapters should serde-roundtrip those representations
- adapters should implement public-neutral transcript views
- users should inspect transcript state through views
- replay should compile from exact committed turns
- branching should be `Session::clone()` for now
- application-specific IR should remain userland

This preserves exactness, improves replay fidelity, avoids hidden normalization, and keeps provider details at the edge without forcing the core crate to ossify around an opinionated universal transcript model.
