use std::{future::Future, pin::Pin, sync::Arc};

use futures::lock::{Mutex, MutexGuard};
use thiserror::Error;

#[cfg(not(target_family = "wasm"))]
pub type HookFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[cfg(target_family = "wasm")]
pub type HookFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[cfg(not(target_family = "wasm"))]
pub trait MaybeSend: Send {}

#[cfg(not(target_family = "wasm"))]
impl<T> MaybeSend for T where T: Send {}

#[cfg(target_family = "wasm")]
pub trait MaybeSend {}

#[cfg(target_family = "wasm")]
impl<T> MaybeSend for T {}

#[cfg(not(target_family = "wasm"))]
pub trait MaybeSync: Sync {}

#[cfg(not(target_family = "wasm"))]
impl<T> MaybeSync for T where T: Sync {}

#[cfg(target_family = "wasm")]
pub trait MaybeSync {}

#[cfg(target_family = "wasm")]
impl<T> MaybeSync for T {}

pub trait HookObject: MaybeSend + MaybeSync {}

impl<T> HookObject for T where T: MaybeSend + MaybeSync + ?Sized {}

pub fn boxed_hook_future<'a, T, F>(future: F) -> HookFuture<'a, T>
where
    F: Future<Output = T> + MaybeSend + 'a,
{
    Box::pin(future)
}

#[derive(Clone)]
pub struct Stateful<H> {
    inner: Arc<Mutex<H>>,
}

impl<H> Stateful<H> {
    pub fn new(inner: H) -> Self {
        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    pub fn try_lock(&self) -> Option<MutexGuard<'_, H>> {
        self.inner.try_lock()
    }
}

impl<H> Default for Stateful<H>
where
    H: Default,
{
    fn default() -> Self {
        Self::new(H::default())
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
#[error("stateful hook reentered: slot={slot}, hook={hook_type}")]
pub struct HookReentrancyError {
    pub slot: &'static str,
    pub hook_type: &'static str,
}
