//! The key dispatch tree: one node per element that has a key context, key / action listeners, a focus id
//! or a tab stop, built during paint. Key events and actions walk it along the focus path (capture root to
//! leaf, then bubble leaf to root), as GPUI's `key_dispatch.rs`. Owner tag: `"window"`.
//!
//! Performance: nodes and listeners live in `Vec`s that keep capacity across frames (`clear`); the focus id
//! to node map is rebuilt in place.
#![allow(dead_code)] // skeleton

use std::{any::TypeId, collections::HashMap, rc::Rc};

use smallvec::SmallVec;

use crate::{Action, App, DispatchPhase, FocusId, KeyContext, Keymap, Keystroke, Window};

/// A node's index in the tree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct DispatchNodeId(pub(crate) usize);

/// A key listener (`on_key_down`, `capture_key_down`, ...): gets the event as `Any`.
pub(crate) type KeyListener = Rc<dyn Fn(&dyn std::any::Any, DispatchPhase, &mut Window, &mut App)>;

/// An action listener (`on_action`): gets the action as `Any`.
pub(crate) type ActionListener = Rc<dyn Fn(&dyn std::any::Any, DispatchPhase, &mut Window, &mut App)>;

/// An action listener and the action type it handles.
#[derive(Clone)]
pub(crate) struct DispatchActionListener {
    /// The action's type.
    pub(crate) action_type: TypeId,
    /// The listener.
    pub(crate) listener: ActionListener,
}

/// One node.
#[derive(Default)]
pub(crate) struct DispatchNode {
    /// Key listeners in registration order.
    pub(crate) key_listeners: Vec<KeyListener>,
    /// Action listeners in registration order.
    pub(crate) action_listeners: Vec<DispatchActionListener>,
    /// The key context it adds, if any.
    pub(crate) context: Option<KeyContext>,
    /// The focus id it tracks, if any.
    pub(crate) focus_id: Option<FocusId>,
    /// Its parent.
    pub(crate) parent: Option<DispatchNodeId>,
}

/// A tab stop in tree order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TabStop {
    /// The focus id.
    pub(crate) focus_id: FocusId,
    /// `tab_index` (lower first; ties in tree order).
    pub(crate) tab_index: isize,
}

/// What matching a keystroke found: the bindings to try, in order, with the context path they matched on.
pub(crate) struct KeyMatch {
    /// The actions bound to the keystroke under the focus path's contexts, best first.
    pub(crate) bindings: SmallVec<[Box<dyn Action>; 1]>,
}

/// The tree.
pub struct DispatchTree {
    /// Nodes in creation order (index = id).
    pub(crate) nodes: Vec<DispatchNode>,
    /// The path of nodes currently being painted.
    pub(crate) node_stack: Vec<DispatchNodeId>,
    /// The key contexts on the current path.
    pub(crate) context_stack: Vec<KeyContext>,
    /// Node of each focus id.
    pub(crate) focusable_node_ids: HashMap<FocusId, DispatchNodeId>,
    /// Tab stops in tree order.
    pub(crate) tab_stops: Vec<TabStop>,
    /// The keymap (shared with the app).
    pub(crate) keymap: Option<Rc<std::cell::RefCell<Keymap>>>,
}

impl DispatchTree {
    /// An empty tree.
    pub(crate) fn new() -> Self {
        DispatchTree {
            nodes: Vec::new(),
            node_stack: Vec::new(),
            context_stack: Vec::new(),
            focusable_node_ids: HashMap::new(),
            tab_stops: Vec::new(),
            keymap: None,
        }
    }

    /// Empties the tree, keeping capacity.
    pub(crate) fn clear(&mut self) {
        todo!("window")
    }

    /// Starts a node under the current one (an element's paint begins).
    pub(crate) fn push_node(&mut self) -> DispatchNodeId {
        todo!("window")
    }

    /// Ends the current node.
    pub(crate) fn pop_node(&mut self) {
        todo!("window")
    }

    /// Sets the current node's key context.
    pub(crate) fn set_key_context(&mut self, _context: KeyContext) {
        todo!("window")
    }

    /// Sets the current node's focus id (and records a tab stop if the handle is one).
    pub(crate) fn set_focus_id(&mut self, _focus_id: FocusId, _tab_stop: Option<isize>) {
        todo!("window")
    }

    /// Adds a key listener to the current node.
    pub(crate) fn on_key_event(&mut self, _listener: KeyListener) {
        todo!("window")
    }

    /// Adds an action listener to the current node.
    pub(crate) fn on_action(&mut self, _action_type: TypeId, _listener: ActionListener) {
        todo!("window")
    }

    /// The node of `focus_id`, if it was painted.
    pub(crate) fn focusable_node_id(&self, focus_id: FocusId) -> Option<DispatchNodeId> {
        self.focusable_node_ids.get(&focus_id).copied()
    }

    /// The path from the root to `node`.
    pub(crate) fn dispatch_path(&self, _node: DispatchNodeId) -> SmallVec<[DispatchNodeId; 32]> {
        todo!("window")
    }

    /// The focus ids on the path from the root to `node`.
    pub(crate) fn focus_path(&self, _node: DispatchNodeId) -> SmallVec<[FocusId; 8]> {
        todo!("window")
    }

    /// The key contexts on the path from the root to `node`.
    pub(crate) fn context_path(&self, _node: DispatchNodeId) -> SmallVec<[KeyContext; 4]> {
        todo!("window")
    }

    /// A node.
    pub(crate) fn node(&self, id: DispatchNodeId) -> &DispatchNode {
        &self.nodes[id.0]
    }

    /// The root node (the window's root element), if any was painted.
    pub(crate) fn root_node_id(&self) -> Option<DispatchNodeId> {
        (!self.nodes.is_empty()).then_some(DispatchNodeId(0))
    }

    /// The bindings for `keystroke` under the contexts of `dispatch_path`.
    pub(crate) fn bindings_for_input(
        &self,
        _keystroke: &Keystroke,
        _dispatch_path: &[DispatchNodeId],
    ) -> KeyMatch {
        todo!("window")
    }

    /// The tab stop after (or before, with `reverse`) `focused`, wrapping.
    pub(crate) fn next_tab_stop(&self, _focused: Option<FocusId>, _reverse: bool) -> Option<FocusId> {
        todo!("window")
    }
}
