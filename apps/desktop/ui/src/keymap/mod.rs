//! Key bindings and the keymap: `KeyBinding::new(keystrokes, action, context)`, `Keymap` and matching a
//! keystroke against the bindings under a context stack. Owner tag: `"runtime"`.
//!
//! The app binds single keystrokes only, with either no context or one bare identifier (`"TextInput"`,
//! `"FoloApp"`), so a binding is one `KeybindingKeystroke` and `KeyBindingContextPredicate` is an identifier.
//! Precedence follows GPUI: the deepest matching context wins, then the binding added last.
#![allow(dead_code)] // skeleton

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
    pub fn load(_keystrokes: &str, _action: Box<dyn Action>, _context: Option<&str>) -> anyhow::Result<Self> {
        todo!("runtime")
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
    pub(crate) fn match_keystrokes(&self, _typed: &[Keystroke]) -> bool {
        todo!("runtime")
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
    pub fn bindings_for_input(&self, _input: &[Keystroke], _context_stack: &[KeyContext]) -> SmallVec<[&KeyBinding; 1]> {
        todo!("runtime")
    }
}
