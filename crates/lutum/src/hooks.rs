//! # Hooks
//!
//! Hooks are named, typed, pluggable async function slots. Each hook has a default
//! implementation and can be overridden at runtime via [`HookRegistry`].
//!
//! ## Defining a hook slot
//!
//! Use `#[def_hook(always)]` when the base implementation must always run, with registered
//! hooks layered on top. This is a good fit for validation, safety, and logging slots:
//!
//! ```rust,ignore
//! use lutum::*;
//!
//! #[def_hook(always)]
//! async fn validate_output(_ctx: &Lutum, output: &str) -> Result<(), String> {
//!     if output.trim().is_empty() { Err("output must not be empty".into()) } else { Ok(()) }
//! }
//! ```
//!
//! Use `#[def_hook(fallback)]` when the base implementation should run only if no hooks are
//! registered. This is a good fit for override and routing slots that intentionally support
//! chaining.
//!
//! Use `#[def_hook(singleton)]` when the slot is conceptually "pick one override or use the
//! default". This keeps the hook signature simpler because there is no `last` argument. If the
//! same slot is registered multiple times, the last registration wins and a warning is emitted.
//!
//! This generates:
//! - `ValidateOutput` - slot marker type
//! - `ValidateOutputArgs` - named args struct for manual/stateful hook implementations
//! - `ValidateOutputHook` - hook trait to implement
//! - `ValidateOutputRegistryExt` - `register_validate_output` and `validate_output` on `HookRegistry`
//! - `ValidateOutputLutumExt` - `validate_output` on `Lutum` (only for `&Lutum` first-arg hooks)
//!
//! ## Defining a named implementation
//!
//! Use `#[hook(SlotType)]` on an async fn to create a concrete implementor:
//!
//! ```rust,ignore
//! #[hook(ValidateOutput)]
//! async fn block_dangerous_output(_ctx: &Lutum, output: &str, last: Option<Result<(), String>>) -> Result<(), String> {
//!     if let Some(Err(err)) = last { return Err(err); }
//!     if output.contains("rm -rf") { Err("blocked dangerous command".into()) } else { Ok(()) }
//! }
//! ```
//!
//! This generates a `BlockDangerousOutput` struct implementing `ValidateOutputHook`.
//! For mutable state, implement the generated `StatefulValidateOutputHook` trait on your
//! type and register it with [`Stateful`].
//!
//! ## Chaining
//!
//! Multiple hooks registered for the same slot run in order for chaining slots. The slot
//! definition may omit `last`, but each registered hook receives the previous hook's result
//! as `last`:
//!
//! - `#[def_hook(always)]`: the default runs first, then the first registered hook gets `last = Some(default_result)`
//! - `#[def_hook(fallback)]`: the first registered hook gets `last = None`
//! - Subsequent hooks always get `last = Some(previous_result)`
//! - With `#[def_hook(fallback)]`, the default implementation runs only when no hooks are registered
//! - With `#[def_hook(singleton)]`, zero registered hooks uses the default and one registered
//!   hook replaces it; later registrations overwrite earlier ones and emit a warning
//!
//! ## Registration and usage
//!
//! ```rust,ignore
//! let hooks = HookRegistry::new()
//!     .register_validate_output(BlockDangerousOutput);
//!     // or with a closure:
//!     // .register_validate_output(|_ctx, output, last| async move { Ok(()) });
//!     // or for mutable state:
//!     // .register_validate_output(Stateful::new(MyMutableHook::default()));
//!
//! let llm = Lutum::with_hooks(adapter, budget, hooks);
//!
//! // Explicit call via Lutum extension method:
//! llm.validate_output(&output).await?;
//! ```
//!
//! ## Hook kinds
//!
//! - **Core hooks** (`llm: &Lutum` first arg): called by `Lutum` — provider-agnostic decisions
//! - **Adapter-local hooks** (`extensions: &RequestExtensions` first arg): called by adapters
//!   using the `&HookRegistry` passed from `Lutum` — provider-specific request shaping
//!
//! ## Builtin adapter hooks
//!
//! Each adapter (`ClaudeAdapter`, `OpenAiAdapter`) defines its own model-selection and
//! resolver hooks. Register them to override the adapter's default behaviour.
//!
//! `lutum` itself also defines `resolve_usage_estimate(&Lutum, &RequestExtensions,
//! OperationKind) -> UsageEstimate`. The default implementation reads a typed estimate from
//! `RequestExtensions` and falls back to zero.

use crate::{Lutum, OperationKind, RequestExtensions, budget::UsageEstimate};

pub use lutum_protocol::hooks::{HookReentrancyError, HookRegistry, Stateful};

#[lutum_macros::def_hook(singleton)]
pub async fn resolve_usage_estimate(
    _ctx: &Lutum,
    extensions: &RequestExtensions,
    _kind: OperationKind,
) -> UsageEstimate {
    extensions
        .get::<UsageEstimate>()
        .copied()
        .unwrap_or_else(UsageEstimate::zero)
}
