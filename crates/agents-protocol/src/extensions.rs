use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

/// A neutral type-map for per-request execution metadata.
///
/// The library treats this as an opaque slot. It passes extensions to
/// `BudgetManager` methods so implementors can extract identity, routing keys,
/// or any other metadata they need. It also passes extensions to `EventHandler`
/// so handlers can inspect them during streaming.
///
/// The library treats entries in `RequestExtensions` as opaque. All entries are
/// user-defined.
#[derive(Default)]
pub struct RequestExtensions {
    map: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl RequestExtensions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<T: Any + Send + Sync + 'static>(&mut self, val: T) -> Option<T> {
        self.map
            .insert(TypeId::of::<T>(), Box::new(val))
            .and_then(|old| old.downcast::<T>().ok().map(|b| *b))
    }

    pub fn get<T: Any + Send + Sync + 'static>(&self) -> Option<&T> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<T>())
    }

    pub fn get_mut<T: Any + Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.map
            .get_mut(&TypeId::of::<T>())
            .and_then(|b| b.downcast_mut::<T>())
    }

    pub fn remove<T: Any + Send + Sync + 'static>(&mut self) -> Option<T> {
        self.map
            .remove(&TypeId::of::<T>())
            .and_then(|b| b.downcast::<T>().ok().map(|b| *b))
    }

    pub fn contains<T: Any + Send + Sync + 'static>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<T>())
    }
}
