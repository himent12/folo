//! Key contexts (`div().key_context("TextInput")`) and binding context predicates. Owner tag: `"runtime"`.

use smallvec::SmallVec;

use crate::SharedString;

/// The identifiers an element in the dispatch tree adds to the context stack.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct KeyContext(pub(crate) SmallVec<[SharedString; 1]>);

impl KeyContext {
    /// Parses a context: space-separated identifiers (the app passes one).
    pub fn parse(source: &str) -> anyhow::Result<Self> {
        let mut context = KeyContext::default();
        for identifier in source.split_whitespace() {
            anyhow::ensure!(is_identifier(identifier), "invalid key context identifier {identifier:?}");
            context.add(SharedString::from(identifier.to_string()));
        }
        Ok(context)
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
    pub fn parse(source: &str) -> anyhow::Result<Self> {
        let source = source.trim();
        anyhow::ensure!(
            is_identifier(source),
            "unsupported key context predicate {source:?} (only a single identifier is supported)"
        );
        Ok(KeyBindingContextPredicate::Identifier(SharedString::from(source.to_string())))
    }

    /// How deep in `contexts` (root first) the predicate matches: GPUI's `depth_of`, which is the length
    /// of the shortest prefix of the stack whose last context satisfies the predicate, searched from the
    /// full stack down (so the deepest matching context, counted from 1). `None` when nothing matches.
    pub fn depth_of(&self, contexts: &[KeyContext]) -> Option<usize> {
        match self {
            KeyBindingContextPredicate::Identifier(name) => {
                contexts.iter().rposition(|context| context.contains(name)).map(|ix| ix + 1)
            }
        }
    }
}

/// GPUI's identifier grammar: letters, digits, `_` and `-`, not starting with a digit.
fn is_identifier(source: &str) -> bool {
    let mut chars = source.chars();
    chars.next().is_some_and(|c| c.is_alphabetic() || c == '_')
        && chars.all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}
