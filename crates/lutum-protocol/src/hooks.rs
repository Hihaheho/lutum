use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

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
