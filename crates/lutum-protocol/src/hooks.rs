use std::sync::Arc;

use futures::lock::{Mutex, MutexGuard};
use thiserror::Error;

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
