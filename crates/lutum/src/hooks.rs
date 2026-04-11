//! # Hooks
//!
//! Hooks are named, typed, pluggable async function slots. Each hook set is declared with
//! `#[hooks] trait`, and each slot is declared by annotating a default async method with
//! `#[hook(always | fallback | singleton, ...)]`.
//!
//! ## Defining a hook set
//!
//! ```rust,ignore
//! use lutum::*;
//!
//! #[hooks]
//! trait AppHooks {
//!     #[hook(always)]
//!     async fn validate_output(output: &str) -> Result<(), String> {
//!         if output.trim().is_empty() {
//!             Err("output must not be empty".into())
//!         } else {
//!             Ok(())
//!         }
//!     }
//! }
//! ```
//!
//! This expands to:
//! - a hook container struct `AppHooks`
//! - a slot trait `ValidateOutput`
//! - a stateful slot trait `StatefulValidateOutput`
//! - `with_validate_output`, `register_validate_output`, and `validate_output` methods
//!
//! ## Defining named implementations
//!
//! ```rust,ignore
//! #[impl_hook(ValidateOutput)]
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
//! Fold hooks always require `last: Option<Return>` on named implementations.
//! Chain, aggregate, and singleton hooks never receive `last`.
//!
//! ## Companion traits
//!
//! `chain = Type`, `aggregate = Type`, and `finalize = Type` on hook definitions refer to the
//! hand-written companion traits in this module:
//! - [`Chain`]
//! - [`Aggregate`] / [`AggregateInto`]
//! - [`Finalize`] / [`FinalizeInto`]
//!
//! Built-in helper implementations such as [`ShortCircuit`] and [`FirstSuccess`] are provided
//! because they are common enough to deserve first-class support.

use crate::{OperationKind, RequestExtensions, budget::UsageEstimate};

pub use lutum_protocol::hooks::{HookReentrancyError, Stateful};

#[async_trait::async_trait]
pub trait Chain<Output>: Send + Sync {
    async fn call(&self, output: &Output) -> ::std::ops::ControlFlow<()>;
}

#[async_trait::async_trait]
pub trait Aggregate<Output>: Send + Sync {
    async fn call(&self, outputs: Vec<Output>) -> Output;
}

#[async_trait::async_trait]
pub trait AggregateInto<Input, Output>: Send + Sync {
    async fn call(&self, outputs: Vec<Input>) -> Output;
}

#[async_trait::async_trait]
pub trait Finalize<Output>: Send + Sync {
    async fn call(&self, output: Output) -> Output;
}

#[async_trait::async_trait]
pub trait FinalizeInto<Input, Output>: Send + Sync {
    async fn call(&self, output: Input) -> Output;
}

/// Default chain implementation for `Result<T, E>` hooks — stops dispatch on the first `Err`.
#[lutum_macros::impl_hook(Chain<Result<T, E>>)]
pub async fn short_circuit<T: Send + Sync + 'static, E: Send + Sync + 'static>(
    output: &Result<T, E>,
) -> ::std::ops::ControlFlow<()> {
    match output {
        Ok(_) => ::std::ops::ControlFlow::Continue(()),
        Err(_) => ::std::ops::ControlFlow::Break(()),
    }
}

/// Default chain implementation for `Option<T>` hooks — stops dispatch on the first `Some`.
#[lutum_macros::impl_hook(Chain<Option<T>>)]
pub async fn first_success<T: Send + Sync + 'static>(
    output: &Option<T>,
) -> ::std::ops::ControlFlow<()> {
    match output {
        Some(_) => ::std::ops::ControlFlow::Break(()),
        None => ::std::ops::ControlFlow::Continue(()),
    }
}

#[lutum_macros::hooks]
pub trait LutumHooks {
    #[hook(singleton)]
    async fn resolve_usage_estimate(
        extensions: &RequestExtensions,
        _kind: OperationKind,
    ) -> UsageEstimate {
        extensions
            .get::<UsageEstimate>()
            .copied()
            .unwrap_or_else(UsageEstimate::zero)
    }
}
