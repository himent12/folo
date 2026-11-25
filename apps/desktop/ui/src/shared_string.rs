// Adapted from GPUI (Apache-2.0): crates/gpui/src/shared_string.rs
//! `SharedString`: an immutable string that clones cheaply (a `&'static str` or an `Arc<str>`).

use std::borrow::{Borrow, Cow};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

/// An immutable string that can be cheaply cloned: a `&'static str` or an `Arc<str>`.
#[derive(Clone)]
pub struct SharedString(Repr);

#[derive(Clone)]
enum Repr {
    Static(&'static str),
    Owned(Arc<str>),
}

impl SharedString {
    /// Creates a static `SharedString` from a `&'static str`.
    pub const fn new_static(str: &'static str) -> Self {
        Self(Repr::Static(str))
    }

    /// Creates a `SharedString` from anything that can become an `Arc<str>`.
    pub fn new(str: impl Into<Arc<str>>) -> Self {
        Self(Repr::Owned(str.into()))
    }

    /// The string.
    pub fn as_str(&self) -> &str {
        match &self.0 {
            Repr::Static(s) => s,
            Repr::Owned(s) => s,
        }
    }
}

impl Deref for SharedString {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl Default for SharedString {
    fn default() -> Self {
        Self::new_static("")
    }
}

impl AsRef<str> for SharedString {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for SharedString {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl PartialEq for SharedString {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for SharedString {}

impl PartialOrd for SharedString {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SharedString {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl Hash for SharedString {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}

impl std::fmt::Debug for SharedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}

impl std::fmt::Display for SharedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<String> for SharedString {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<SharedString> for String {
    fn eq(&self, other: &SharedString) -> bool {
        self == other.as_str()
    }
}

impl PartialEq<str> for SharedString {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl<'a> PartialEq<&'a str> for SharedString {
    fn eq(&self, other: &&'a str) -> bool {
        self.as_str() == *other
    }
}

impl From<&SharedString> for SharedString {
    fn from(value: &SharedString) -> Self {
        value.clone()
    }
}

impl From<&'static str> for SharedString {
    fn from(value: &'static str) -> Self {
        Self(Repr::Static(value))
    }
}

impl From<String> for SharedString {
    fn from(value: String) -> Self {
        Self(Repr::Owned(value.into()))
    }
}

impl From<&String> for SharedString {
    fn from(value: &String) -> Self {
        Self(Repr::Owned(value.as_str().into()))
    }
}

impl From<Arc<str>> for SharedString {
    fn from(value: Arc<str>) -> Self {
        Self(Repr::Owned(value))
    }
}

impl From<&Arc<str>> for SharedString {
    fn from(value: &Arc<str>) -> Self {
        Self(Repr::Owned(value.clone()))
    }
}

impl From<Cow<'static, str>> for SharedString {
    fn from(value: Cow<'static, str>) -> Self {
        match value {
            Cow::Borrowed(s) => Self(Repr::Static(s)),
            Cow::Owned(s) => s.into(),
        }
    }
}

impl From<SharedString> for Arc<str> {
    fn from(val: SharedString) -> Self {
        match val.0 {
            Repr::Static(s) => Arc::from(s),
            Repr::Owned(s) => s,
        }
    }
}

impl From<SharedString> for String {
    fn from(val: SharedString) -> Self {
        val.as_str().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_whatever_the_storage() {
        let a = SharedString::new_static("abc");
        let b = SharedString::from(String::from("abc"));
        assert_eq!(a, b);
        assert_eq!(a, "abc");
        assert_eq!(&*b, "abc");
        assert_eq!(String::from(b.clone()), "abc");
        assert!(SharedString::default().is_empty());
    }
}
