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
//! async fn validate_output(_ctx: &Context, output: &str, last: Option<Result<(), String>>) -> Result<(), String> {
//!     Ok(())
//! }
//! ```
//!
//! Use `#[def_hook(fallback)]` when the base implementation should run only if no hooks are
//! registered. This is a good fit for override and routing slots.
//!
//! This generates:
//! - `ValidateOutput` - slot marker type
//! - `ValidateOutputHook` - hook trait to implement
//! - `ValidateOutputRegistryExt` - `register_validate_output` and `validate_output` on `HookRegistry`
//! - `ValidateOutputContextExt` - `validate_output` on `Context` (only for `&Context` first-arg hooks)
//!
//! ## Defining a named implementation
//!
//! Use `#[hook(SlotType)]` on an async fn to create a concrete implementor:
//!
//! ```rust,ignore
//! #[hook(ValidateOutput)]
//! async fn block_dangerous_output(_ctx: &Context, output: &str, last: Option<Result<(), String>>) -> Result<(), String> {
//!     if let Some(Err(err)) = last { return Err(err); }
//!     if output.contains("rm -rf") { Err("blocked dangerous command".into()) } else { Ok(()) }
//! }
//! ```
//!
//! This generates a `BlockDangerousOutput` struct implementing `ValidateOutputHook`.
//!
//! ## Chaining
//!
//! Multiple hooks registered for the same slot run in order. Each hook receives the
//! previous hook's result as `last`:
//!
//! - `#[def_hook(always)]`: the default runs first, then the first registered hook gets `last = Some(default_result)`
//! - `#[def_hook(fallback)]`: the first registered hook gets `last = None`
//! - Subsequent hooks always get `last = Some(previous_result)`
//! - With `#[def_hook(fallback)]`, the default implementation runs only when no hooks are registered
//!
//! ## Registration and usage
//!
//! ```rust,ignore
//! let hooks = HookRegistry::new()
//!     .register_validate_output(BlockDangerousOutput);
//!     // or with a closure:
//!     // .register_validate_output(|_ctx, output, last| async move { Ok(()) });
//!
//! let ctx = Context::with_hooks(adapter, budget, hooks);
//!
//! // Explicit call via Context extension method:
//! ctx.validate_output(&output).await?;
//! ```
//!
//! ## Hook kinds
//!
//! - **Core hooks** (`ctx: &Context` first arg): called by `Context` — provider-agnostic decisions
//! - **Adapter-local hooks** (`extensions: &RequestExtensions` first arg): called by adapters
//!   using the `&HookRegistry` passed from `Context` — provider-specific request shaping
//!
//! ## Builtin adapter hooks
//!
//! Each adapter (`ClaudeAdapter`, `OpenAiAdapter`) defines its own model-selection and
//! resolver hooks. Register them to override the adapter's default behaviour.

pub use lutum_protocol::hooks::HookRegistry;
