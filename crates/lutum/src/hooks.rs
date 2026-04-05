//! # Hooks
//!
//! Hooks are named, typed, pluggable async function slots. Each hook has a default
//! implementation and can be overridden at runtime via [`HookRegistry`].
//!
//! ## Defining a hook slot
//!
//! Use `#[hook_always]` when the base implementation must always run, with registered
//! hooks layered on top. This is a good fit for validation, safety, and logging slots:
//!
//! ```rust,ignore
//! use lutum::*;
//!
//! #[hook_always]
//! async fn validate_output(_ctx: &Context, output: &str, last: Option<Result<(), String>>) -> Result<(), String> {
//!     Ok(())
//! }
//! ```
//!
//! Use `#[hook_fallback]` when the base implementation should run only if no hooks are
//! registered. This is a good fit for override and routing slots.
//!
//! This generates:
//! - `ValidateOutput` - slot marker type
//! - `ValidateOutputHook` - hook trait to implement
//! - `ValidateOutputRegistryExt` - `register_validate_output` and `validate_output` on `HookRegistry`
//! - `ValidateOutputContextExt` - `validate_output` on `Context`
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
//! - `#[hook_always]`: the default runs first, then the first registered hook gets `last = Some(default_result)`
//! - `#[hook_fallback]`: the first registered hook gets `last = None`
//! - Subsequent hooks always get `last = Some(previous_result)`
//! - With `#[hook_fallback]`, the default implementation runs only when no hooks are registered
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
//! ## Builtin hooks
//!
//! [`select_model`] is a builtin hook called automatically by [`Context`] before every turn.
//! Register a handler to override model selection per request.
use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

use lutum_protocol::{ModelName, extensions::RequestExtensions};

use crate::Context;

/// Storage for hook chains.
///
/// Build a registry with [`HookRegistry::new`], register handlers with the
/// generated `register_*` methods, then pass it to [`Context::with_hooks`].
pub struct HookRegistry {
    slots: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self {
            slots: HashMap::new(),
        }
    }

    #[doc(hidden)]
    pub fn slots(&self) -> &HashMap<TypeId, Box<dyn Any + Send + Sync>> {
        &self.slots
    }

    #[doc(hidden)]
    pub fn slots_mut(&mut self) -> &mut HashMap<TypeId, Box<dyn Any + Send + Sync>> {
        &mut self.slots
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[lutum_macros::hook_fallback]
pub async fn select_model(
    ctx: &Context,
    _extensions: &RequestExtensions,
    current: ModelName,
    last: Option<ModelName>,
) -> ModelName {
    let _ = ctx;
    last.unwrap_or(current)
}
