//! Key bindings and the keymap: `KeyBinding::new(keystrokes, action, context)`, `Keymap` and matching a
//! keystroke against the bindings under a context stack. Owner tag: `"runtime"`.
//!
//! The app binds single keystrokes only, with either no context or one bare identifier (`"TextInput"`,
//! `"FoloApp"`), so a binding is one `KeybindingKeystroke` and `KeyBindingContextPredicate` is an identifier.
//! Precedence follows GPUI: the deepest matching context wins, then the binding added last. A binding
//! with no context counts as matching at the deepest level (GPUI's `binding_enabled`), so among equally
//! deep matches the later binding wins whether or not it names a context.

mod context;

pub use context::*;

use std::rc::Rc;

use smallvec::SmallVec;

use crate::{Action, KeybindingKeystroke, Keystroke};

/// One key binding: keystrokes, the action they dispatch, and the context it is active in.
pub struct KeyBinding {
    /// The keystroke sequence (one keystroke for every binding the app makes).
    pub(crate) keystrokes: SmallVec<[KeybindingKeystroke; 2]>,
    /// The action to dispatch.
    pub(crate) action: Box<dyn Action>,
    /// Where the binding applies; `None` is everywhere.
    pub(crate) context_predicate: Option<Rc<KeyBindingContextPredicate>>,
}

impl KeyBinding {
    /// A binding of `keystrokes` (`"ctrl-shift-l"`; space-separated for a sequence) to `action` in `context`.
    /// Panics on a keystroke or context that does not parse, as GPUI does.
    pub fn new<A: Action>(keystrokes: &str, action: A, context: Option<&str>) -> Self {
        Self::load(keystrokes, Box::new(action), context).expect("invalid key binding")
    }

    /// `new`, returning the parse error instead of panicking.
    pub fn load(keystrokes: &str, action: Box<dyn Action>, context: Option<&str>) -> anyhow::Result<Self> {
        let context_predicate = context.map(KeyBindingContextPredicate::parse).transpose()?.map(Rc::new);
        let keystrokes = keystrokes
            .split_whitespace()
            .map(|source| Keystroke::parse(source).map(KeybindingKeystroke::from_keystroke))
            .collect::<Result<SmallVec<_>, _>>()?;
        anyhow::ensure!(!keystrokes.is_empty(), "a key binding needs a keystroke");
        Ok(KeyBinding { keystrokes, action, context_predicate })
    }

    /// The keystrokes.
    pub fn keystrokes(&self) -> &[KeybindingKeystroke] {
        &self.keystrokes
    }

    /// The action.
    pub fn action(&self) -> &dyn Action {
        self.action.as_ref()
    }

    /// Whether `typed` is this binding's keystroke sequence.
    pub(crate) fn match_keystrokes(&self, typed: &[Keystroke]) -> bool {
        self.keystrokes.len() == typed.len()
            && self.keystrokes.iter().zip(typed).all(|(target, typed)| typed.should_match(target))
    }

    /// The deepest context depth this binding is enabled at under `contexts` (GPUI's `binding_enabled`).
    fn enabled_depth(&self, contexts: &[KeyContext]) -> Option<usize> {
        match &self.context_predicate {
            Some(predicate) => predicate.depth_of(contexts),
            None => Some(contexts.len()),
        }
    }
}

/// Every binding, in the order added.
#[derive(Default)]
pub struct Keymap {
    /// The bindings.
    pub(crate) bindings: Vec<KeyBinding>,
}

impl Keymap {
    /// Adds bindings (later ones take precedence at equal context depth).
    pub fn add_bindings<T: IntoIterator<Item = KeyBinding>>(&mut self, bindings: T) {
        self.bindings.extend(bindings);
    }

    /// The actions bound to `input` under `context_stack` (root first), best match first.
    pub fn bindings_for_input(
        &self,
        input: &[Keystroke],
        context_stack: &[KeyContext],
    ) -> SmallVec<[&KeyBinding; 1]> {
        let mut matched: SmallVec<[(usize, usize, &KeyBinding); 1]> = self
            .bindings
            .iter()
            .enumerate()
            .filter_map(|(ix, binding)| {
                let depth = binding.enabled_depth(context_stack)?;
                binding.match_keystrokes(input).then_some((depth, ix, binding))
            })
            .collect();
        // Deeper first, then later first.
        matched.sort_by(|(depth_a, ix_a, _), (depth_b, ix_b, _)| depth_b.cmp(depth_a).then(ix_b.cmp(ix_a)));
        matched.into_iter().map(|(_, _, binding)| binding).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    crate::actions!(keymap_test, [Global, InInput, InApp, Later]);

    fn ctx(source: &str) -> KeyContext {
        KeyContext::parse(source).unwrap()
    }

    fn names(bindings: &[&KeyBinding]) -> Vec<&'static str> {
        bindings.iter().map(|b| b.action().name()).collect()
    }

    #[test]
    fn deeper_contexts_then_later_bindings_win() {
        let mut keymap = Keymap::default();
        keymap.add_bindings([
            KeyBinding::new("ctrl-a", InApp, Some("FoloApp")),
            KeyBinding::new("ctrl-a", InInput, Some("TextInput")),
            KeyBinding::new("ctrl-a", Global, None),
        ]);
        let typed = [Keystroke::parse("ctrl-a").unwrap()];
        let stack = [ctx("FoloApp"), ctx("TextInput")];
        // No context counts as the deepest level; among equals the later binding wins (GPUI).
        assert_eq!(
            names(&keymap.bindings_for_input(&typed, &stack)),
            ["keymap_test::Global", "keymap_test::InInput", "keymap_test::InApp"]
        );
        let outside = [ctx("FoloApp")];
        assert_eq!(
            names(&keymap.bindings_for_input(&typed, &outside)),
            ["keymap_test::Global", "keymap_test::InApp"]
        );
        keymap.add_bindings([KeyBinding::new("ctrl-a", Later, Some("TextInput"))]);
        assert_eq!(
            names(&keymap.bindings_for_input(&typed, &stack))[..2],
            ["keymap_test::Later", "keymap_test::Global"]
        );
    }

    #[test]
    fn keystrokes_must_match_exactly() {
        let mut keymap = Keymap::default();
        keymap.add_bindings([KeyBinding::new("ctrl-shift-l", Global, None)]);
        let hit = [Keystroke::parse("ctrl-shift-l").unwrap()];
        let miss = [Keystroke::parse("ctrl-l").unwrap()];
        assert_eq!(keymap.bindings_for_input(&hit, &[]).len(), 1);
        assert!(keymap.bindings_for_input(&miss, &[]).is_empty());
        assert!(KeyBinding::load("  ", Box::new(Global), None).is_err());
        assert!(KeyBinding::load("a", Box::new(Global), Some("a && b")).is_err());
    }

    #[test]
    fn predicate_depth_is_gpuis() {
        let p = KeyBindingContextPredicate::parse("TextInput").unwrap();
        assert_eq!(p.depth_of(&[ctx("FoloApp"), ctx("TextInput")]), Some(2));
        assert_eq!(p.depth_of(&[ctx("TextInput"), ctx("FoloApp")]), Some(1));
        assert_eq!(p.depth_of(&[ctx("FoloApp")]), None);
        assert!(ctx("a b").contains("b"));
        assert!(KeyContext::parse("9x").is_err());
    }
}
