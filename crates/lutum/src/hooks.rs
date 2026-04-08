//! # Hooks
//!
//! Hooks are named, typed, pluggable async function slots. Each hook has a default
//! implementation and can be overridden at runtime via [`HookRegistry`].
//!
//! ## Defining a hook slot
//!
//! Hook slots come in two dispatch families plus a singleton mode:
//!
//! - **Fold dispatch**: registered hooks receive `last: Option<Output>`
//! - **Chain dispatch**: registered hooks do **not** receive `last`; a chain function decides
//!   whether dispatch should stop after each result
//! - **Singleton**: zero or one active override; no chaining
//!
//! ### Fold dispatch
//!
//! Use `#[def_hook(always)]` when the base implementation must always run, with registered
//! hooks layered on top. This is a good fit for validation, safety, and logging slots:
//!
//! ```rust,ignore
//! use lutum::*;
//!
//! #[def_hook(always)]
//! async fn validate_output(output: &str) -> Result<(), String> {
//!     if output.trim().is_empty() { Err("output must not be empty".into()) } else { Ok(()) }
//! }
//!
//! #[hook(ValidateOutput)]
//! async fn block_dangerous_output(
//!     output: &str,
//!     last: Option<Result<(), String>>,
//! ) -> Result<(), String> {
//!     if let Some(Err(err)) = last {
//!         return Err(err);
//!     }
//!     if output.contains("rm -rf") {
//!         Err("blocked dangerous command".into())
//!     } else {
//!         Ok(())
//!     }
//! }
//! ```
//!
//! Use `#[def_hook(fallback)]` when the base implementation should run only if no hooks are
//! registered. This is a good fit for override and routing slots that intentionally support
//! fold-style chaining.
//!
//! Fold dispatch semantics:
//!
//! - `#[def_hook(always)]`: the default runs first, then the first registered hook gets
//!   `last = Some(default_result)`
//! - `#[def_hook(fallback)]`: the first registered hook gets `last = None`
//! - Subsequent hooks always get `last = Some(previous_result)`
//!
//! The default slot definition itself may include `last` or omit it.
//!
//! ### Chain dispatch
//!
//! Use `#[def_hook(always, chain = Type)]` or `#[def_hook(fallback, chain = Type)]` when
//! dispatch should be controlled by a [`Chain<Output>`] implementation instead of passing
//! `last` forward.
//!
//! `chain = Type` specifies the **default** `Chain<Output>` implementation. The hooks struct
//! companion field (`{hook_name}_chain`) can be set at runtime to override it.
//!
//! Two built-in implementations are provided:
//!
//! - [`ShortCircuit<T, E>`] for `Result<T, E>` hooks: stop on `Err(_)`
//! - [`FirstSuccess<T>`] for `Option<T>` hooks: stop on `Some(_)`
//!
//! ```rust,ignore
//! use lutum::*;
//!
//! #[def_hook(always, chain = ShortCircuit<(), String>)]
//! async fn validate_output(output: &str) -> Result<(), String> {
//!     if output.trim().is_empty() { Err("output must not be empty".into()) } else { Ok(()) }
//! }
//!
//! #[hook(ValidateOutput)]
//! async fn block_dangerous_output(output: &str) -> Result<(), String> {
//!     if output.contains("rm -rf") {
//!         Err("blocked dangerous command".into())
//!     } else {
//!         Ok(())
//!     }
//! }
//! ```
//!
//! Chain dispatch semantics:
//!
//! - `#[def_hook(always, chain = ...)]`: run the default first; if the chain returns
//!   `Break`, return that result immediately; otherwise run registered hooks in order
//! - `#[def_hook(fallback, chain = ...)]`: if no hooks are registered, run the default; if hooks
//!   are registered, run them first and only fall back to the default if all results continue
//! - In chain mode, neither the default hook function nor `#[hook(...)]` implementations accept
//!   a `last` parameter
//!
//! ### Singleton dispatch
//!
//! Use `#[def_hook(singleton)]` when the slot is conceptually "pick one override or use the
//! default". This keeps the hook signature simpler because there is no `last` argument. If the
//! same slot is registered multiple times, the last registration wins and a warning is emitted.
//!
//! `singleton` does not support `chain = ...`.
//!
//! This generates:
//! - `ValidateOutputArgs` - named args struct for manual/stateful hook implementations
//! - `ValidateOutput` - hook trait to implement
//! - `ValidateOutputRegistryExt` - `register_validate_output` and `validate_output` on `HookRegistry`
//! - `ValidateOutputLutumExt` - `validate_output` on `Lutum`
//!
//! ## Defining a named implementation
//!
//! Use `#[hook(SlotType)]` on an async fn to create a concrete implementor.
//!
//! In fold mode, the generated trait includes `last`.
//! In chain mode and singleton mode, it does not.
//!
//! This generates a `BlockDangerousOutput` struct implementing `ValidateOutput`.
//! For mutable state, implement the generated `StatefulValidateOutput` trait on your
//! type and register it with [`Stateful`].
//!
//! ## Registration and usage
//!
//! ```rust,ignore
//! let hooks = HookRegistry::new()
//!     .register_validate_output(BlockDangerousOutput);
//!     // or with a closure in fold mode:
//!     // .register_validate_output(|args: ValidateOutputArgs<'_>, last| async move { Ok(()) });
//!     // or with a closure in chain/singleton mode:
//!     // .register_validate_output(|args: ValidateOutputArgs<'_>| async move { Ok(()) });
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
//! - **Core hooks**: called by `Lutum` — provider-agnostic decisions
//! - **Adapter-local hooks**: called by adapters using the `&HookRegistry` passed from `Lutum`
//!   — provider-specific request shaping
//!
//! If a hook needs `&Lutum` or any other context, either store it in the implementing struct or
//! add it as a normal explicit argument.
//!
//! ## Builtin adapter hooks
//!
//! Each adapter (`ClaudeAdapter`, `OpenAiAdapter`) defines its own model-selection and
//! resolver hooks. Register them to override the adapter's default behaviour.
//!
//! `lutum` itself also defines `resolve_usage_estimate(&RequestExtensions,
//! OperationKind) -> UsageEstimate`. The default implementation reads a typed estimate from
//! `RequestExtensions` and falls back to zero.

use crate::{OperationKind, RequestExtensions, budget::UsageEstimate};

pub use lutum_protocol::hooks::{HookReentrancyError, HookRegistry, Stateful};

/// Companion chain trait — decides whether dispatch should stop after each result.
///
/// Use as `chain = ShortCircuit<T, E>` or `chain = FirstSuccess<T>` in
/// `#[def_hook(always, chain = ...)]` or `#[def_hook(fallback, chain = ...)]`.
/// The `chain = ...` value specifies the default implementation used when no custom
/// `Chain<Output>` is registered on the hooks struct.
pub trait Chain<Output: Send + Sync + 'static>: Send + Sync {
    fn call(&self, output: &Output) -> std::ops::ControlFlow<()>;
}

/// Default chain implementation for `Result<T, E>` hooks — stops dispatch on the first `Err`.
///
/// Use as `chain = ShortCircuit<T, E>`.
pub struct ShortCircuit<T, E>(std::marker::PhantomData<fn(T, E)>);

impl<T, E> Default for ShortCircuit<T, E> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T: Send + Sync + 'static, E: Send + Sync + 'static> Chain<Result<T, E>>
    for ShortCircuit<T, E>
{
    fn call(&self, output: &Result<T, E>) -> std::ops::ControlFlow<()> {
        match output {
            Ok(_) => std::ops::ControlFlow::Continue(()),
            Err(_) => std::ops::ControlFlow::Break(()),
        }
    }
}

/// Default chain implementation for `Option<T>` hooks — stops dispatch on the first `Some`.
///
/// Use as `chain = FirstSuccess<T>`.
pub struct FirstSuccess<T>(std::marker::PhantomData<fn(T)>);

impl<T> Default for FirstSuccess<T> {
    fn default() -> Self {
        Self(std::marker::PhantomData)
    }
}

impl<T: Send + Sync + 'static> Chain<Option<T>> for FirstSuccess<T> {
    fn call(&self, output: &Option<T>) -> std::ops::ControlFlow<()> {
        match output {
            Some(_) => std::ops::ControlFlow::Break(()),
            None => std::ops::ControlFlow::Continue(()),
        }
    }
}

#[lutum_macros::def_global_hook(singleton)]
pub async fn resolve_usage_estimate(
    extensions: &RequestExtensions,
    _kind: OperationKind,
) -> UsageEstimate {
    extensions
        .get::<UsageEstimate>()
        .copied()
        .unwrap_or_else(UsageEstimate::zero)
}
