use std::borrow::Cow;

pub trait Marker: Clone + Send + Sync + 'static {
    fn span_name(&self) -> Cow<'static, str>;
}

impl Marker for &'static str {
    fn span_name(&self) -> Cow<'static, str> {
        Cow::Borrowed(*self)
    }
}

impl Marker for String {
    fn span_name(&self) -> Cow<'static, str> {
        Cow::Owned(self.clone())
    }
}
