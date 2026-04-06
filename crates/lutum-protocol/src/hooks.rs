use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::Arc,
};

use futures::lock::{Mutex, MutexGuard};
use thiserror::Error;

/// Storage for hook chains.
///
/// Build a registry with [`HookRegistry::new`], register handlers with the
/// generated `register_*` methods, then pass it to [`lutum::Lutum::with_hooks`].
pub struct HookRegistry {
    slots: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
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
