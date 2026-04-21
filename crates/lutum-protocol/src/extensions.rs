use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::Arc,
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
///
/// Reads walk the newest layer first, then any older fallback layers. Mutations
/// apply only to the newest local layer.
#[derive(Default)]
pub struct RequestExtensions {
    fallbacks: Vec<Arc<RequestExtensions>>,
    extensions: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl RequestExtensions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<T: Any + Send + Sync + 'static>(&mut self, val: T) -> Option<T> {
        self.extensions
            .insert(TypeId::of::<T>(), Box::new(val))
            .and_then(|old| old.downcast::<T>().ok().map(|b| *b))
    }

    pub fn get<T: Any + Send + Sync + 'static>(&self) -> Option<&T> {
        self.extensions
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<T>())
            .or_else(|| {
                self.fallbacks
                    .iter()
                    .find_map(|fallback| fallback.get::<T>())
            })
    }

    pub fn get_mut<T: Any + Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.extensions
            .get_mut(&TypeId::of::<T>())
            .and_then(|b| b.downcast_mut::<T>())
    }

    pub fn remove<T: Any + Send + Sync + 'static>(&mut self) -> Option<T> {
        self.extensions
            .remove(&TypeId::of::<T>())
            .and_then(|b| b.downcast::<T>().ok().map(|b| *b))
    }

    pub fn contains<T: Any + Send + Sync + 'static>(&self) -> bool {
        self.get::<T>().is_some()
    }

    pub fn extend(&mut self, other: Self) {
        if other.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = other;
            return;
        }

        let previous = Arc::new(std::mem::take(self));
        self.fallbacks = other.fallbacks;
        self.extensions = other.extensions;
        self.fallbacks.push(previous);
    }

    pub fn is_empty(&self) -> bool {
        self.extensions.is_empty() && self.fallbacks.is_empty()
    }

    pub fn push_fallback(&mut self, fallback: Arc<RequestExtensions>) {
        if fallback.is_empty() {
            return;
        }
        self.fallbacks.push(fallback);
    }
}

#[cfg(test)]
mod tests {
    use super::RequestExtensions;
    use std::sync::Arc;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Marker(u32);

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Extra(u32);

    #[test]
    fn get_reads_fallback_layers() {
        let mut root = RequestExtensions::new();
        root.insert(Marker(1));

        let mut request = RequestExtensions::new();
        request.insert(Extra(2));
        request.push_fallback(Arc::new(root));

        assert_eq!(request.get::<Marker>(), Some(&Marker(1)));
        assert_eq!(request.get::<Extra>(), Some(&Extra(2)));
    }

    #[test]
    fn extend_preserves_newest_wins_order() {
        let mut existing = RequestExtensions::new();
        existing.insert(Marker(1));

        let mut appended = RequestExtensions::new();
        appended.insert(Marker(2));
        appended.insert(Extra(3));

        existing.extend(appended);

        assert_eq!(existing.get::<Marker>(), Some(&Marker(2)));
        assert_eq!(existing.get::<Extra>(), Some(&Extra(3)));
    }

    #[test]
    fn extend_with_empty_is_noop() {
        let mut extensions = RequestExtensions::new();
        extensions.insert(Marker(1));

        extensions.extend(RequestExtensions::new());

        assert_eq!(extensions.get::<Marker>(), Some(&Marker(1)));
        assert_eq!(
            extensions.get_mut::<Marker>().map(|marker| marker.0),
            Some(1)
        );
    }
}
