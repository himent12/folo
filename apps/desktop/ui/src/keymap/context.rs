//! Key contexts (`div().key_context("TextInput")`) and binding context predicates. Owner tag: `"runtime"`.
#![allow(dead_code)] // skeleton

use smallvec::SmallVec;

use crate::SharedString;

/// The identifiers an element in the dispatch tree adds to the context stack.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct KeyContext(pub(crate) SmallVec<[SharedString; 1]>);

impl KeyContext {
    /// Parses a context: space-separated identifiers (the app passes one).
    pub fn parse(_source: &str) -> anyhow::Result<Self> {
        todo!("runtime")
    }

    /// Adds an identifier.
    pub fn add<I: Into<SharedString>>(&mut self, identifier: I) {
        self.0.push(identifier.into());
    }

    /// Whether the context has `identifier`.
    pub fn contains(&self, identifier: &str) -> bool {
        self.0.iter().any(|i| i.as_str() == identifier)
    }

    /// Whether it has no identifiers.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl TryFrom<&str> for KeyContext {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> anyhow::Result<Self> {
        Self::parse(value)
    }
}

/// When a binding applies: the grammar the app's bindings use is a single identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KeyBindingContextPredicate {
    /// Active when some context on the stack contains the identifier.
    Identifier(SharedString),
}

impl KeyBindingContextPredicate {
    /// Parses a predicate (an identifier; anything else is an error).
    pub fn parse(_source: &str) -> anyhow::Result<Self> {
        todo!("runtime")
    }

    /// The depth (index into `contexts`, root first) of the deepest context the predicate matches, if any.
    pub fn depth_of(&self, _contexts: &[KeyContext]) -> Option<usize> {
        todo!("runtime")
    }
}
