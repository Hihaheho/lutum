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
//! - an implementation trait `AppHooks`
//! - a hook container struct `AppHooksSet<'h>`
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

pub use lutum_protocol::hooks::{
    HookFuture, HookObject, HookReentrancyError, MaybeSend, MaybeSync, Stateful, boxed_hook_future,
};

pub trait Chain<Output>: HookObject {
    fn call<'a>(
        &'a self,
        output: &'a Output,
    ) -> impl ::std::future::Future<Output = ::std::ops::ControlFlow<()>> + MaybeSend + 'a;
}

#[doc(hidden)]
pub trait DynChain<Output>: HookObject {
    fn call_dyn<'a>(&'a self, output: &'a Output) -> HookFuture<'a, ::std::ops::ControlFlow<()>>;
}

impl<T, Output> DynChain<Output> for T
where
    T: Chain<Output>,
    Output: MaybeSync,
{
    fn call_dyn<'a>(&'a self, output: &'a Output) -> HookFuture<'a, ::std::ops::ControlFlow<()>> {
        boxed_hook_future(async move { <T as Chain<Output>>::call(self, output).await })
    }
}

impl<T, Output> Chain<Output> for &T
where
    T: Chain<Output> + ?Sized,
{
    fn call<'a>(
        &'a self,
        output: &'a Output,
    ) -> impl ::std::future::Future<Output = ::std::ops::ControlFlow<()>> + MaybeSend + 'a {
        (**self).call(output)
    }
}

pub trait Aggregate<Output>: HookObject {
    fn call(
        &self,
        outputs: Vec<Output>,
    ) -> impl ::std::future::Future<Output = Output> + MaybeSend + '_;
}

#[doc(hidden)]
pub trait DynAggregate<Output>: HookObject {
    fn call_dyn<'a>(&'a self, outputs: Vec<Output>) -> HookFuture<'a, Output>
    where
        Output: 'a;
}

impl<T, Output> DynAggregate<Output> for T
where
    T: Aggregate<Output>,
    Output: MaybeSend,
{
    fn call_dyn<'a>(&'a self, outputs: Vec<Output>) -> HookFuture<'a, Output>
    where
        Output: 'a,
    {
        boxed_hook_future(async move { <T as Aggregate<Output>>::call(self, outputs).await })
    }
}

impl<T, Output> Aggregate<Output> for &T
where
    T: Aggregate<Output> + ?Sized,
{
    fn call(
        &self,
        outputs: Vec<Output>,
    ) -> impl ::std::future::Future<Output = Output> + MaybeSend + '_ {
        (**self).call(outputs)
    }
}

pub trait AggregateInto<Input, Output>: HookObject {
    fn call(
        &self,
        outputs: Vec<Input>,
    ) -> impl ::std::future::Future<Output = Output> + MaybeSend + '_;
}

#[doc(hidden)]
pub trait DynAggregateInto<Input, Output>: HookObject {
    fn call_dyn<'a>(&'a self, outputs: Vec<Input>) -> HookFuture<'a, Output>
    where
        Input: 'a;
}

impl<T, Input, Output> DynAggregateInto<Input, Output> for T
where
    T: AggregateInto<Input, Output>,
    Input: MaybeSend,
{
    fn call_dyn<'a>(&'a self, outputs: Vec<Input>) -> HookFuture<'a, Output>
    where
        Input: 'a,
    {
        boxed_hook_future(
            async move { <T as AggregateInto<Input, Output>>::call(self, outputs).await },
        )
    }
}

impl<T, Input, Output> AggregateInto<Input, Output> for &T
where
    T: AggregateInto<Input, Output> + ?Sized,
{
    fn call(
        &self,
        outputs: Vec<Input>,
    ) -> impl ::std::future::Future<Output = Output> + MaybeSend + '_ {
        (**self).call(outputs)
    }
}

pub trait Finalize<Output>: HookObject {
    fn call(&self, output: Output) -> impl ::std::future::Future<Output = Output> + MaybeSend + '_;
}

#[doc(hidden)]
pub trait DynFinalize<Output>: HookObject {
    fn call_dyn<'a>(&'a self, output: Output) -> HookFuture<'a, Output>
    where
        Output: 'a;
}

impl<T, Output> DynFinalize<Output> for T
where
    T: Finalize<Output>,
    Output: MaybeSend,
{
    fn call_dyn<'a>(&'a self, output: Output) -> HookFuture<'a, Output>
    where
        Output: 'a,
    {
        boxed_hook_future(async move { <T as Finalize<Output>>::call(self, output).await })
    }
}

impl<T, Output> Finalize<Output> for &T
where
    T: Finalize<Output> + ?Sized,
{
    fn call(&self, output: Output) -> impl ::std::future::Future<Output = Output> + MaybeSend + '_ {
        (**self).call(output)
    }
}

pub trait FinalizeInto<Input, Output>: HookObject {
    fn call(&self, output: Input) -> impl ::std::future::Future<Output = Output> + MaybeSend + '_;
}

#[doc(hidden)]
pub trait DynFinalizeInto<Input, Output>: HookObject {
    fn call_dyn<'a>(&'a self, output: Input) -> HookFuture<'a, Output>
    where
        Input: 'a;
}

impl<T, Input, Output> DynFinalizeInto<Input, Output> for T
where
    T: FinalizeInto<Input, Output>,
    Input: MaybeSend,
{
    fn call_dyn<'a>(&'a self, output: Input) -> HookFuture<'a, Output>
    where
        Input: 'a,
    {
        boxed_hook_future(
            async move { <T as FinalizeInto<Input, Output>>::call(self, output).await },
        )
    }
}

impl<T, Input, Output> FinalizeInto<Input, Output> for &T
where
    T: FinalizeInto<Input, Output> + ?Sized,
{
    fn call(&self, output: Input) -> impl ::std::future::Future<Output = Output> + MaybeSend + '_ {
        (**self).call(output)
    }
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
