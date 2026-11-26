//! The key dispatch tree: one node per interactive element, built during prepaint (focus ids) and paint
//! (key contexts, key / action listeners), as GPUI's `key_dispatch.rs`. Key events and actions walk it
//! along the focus path (capture root to leaf, then bubble leaf to root). Tab stops are kept here too
//! (GPUI's `TabStopMap`: every tracked focus handle in paint order, ordered by `tab_index` then paint
//! order; only `tab_stop` handles are stopped at). Owner tag: `"window"`.
//!
//! Performance: nodes, listeners and tab stops live in `Vec`s that keep capacity across frames (`clear`);
//! the focus id to node map is cleared in place.

use std::{any::TypeId, cell::RefCell, collections::HashMap, rc::Rc};

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

/// A tracked focus handle in paint order (GPUI's `TabStopNode`, without tab groups: the app uses none).
#[derive(Clone, Copy, Debug)]
pub(crate) struct TabStop {
    /// The focus id.
    pub(crate) focus_id: FocusId,
    /// `tab_index` (lower first; ties in paint order).
    pub(crate) tab_index: isize,
    /// Whether Tab stops here (`FocusHandle::tab_stop`); other handles only anchor the search.
    pub(crate) tab_stop: bool,
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
    /// Tracked focus handles in paint order.
    pub(crate) tab_stops: Vec<TabStop>,
    /// The keymap (shared with the app).
    pub(crate) keymap: Option<Rc<RefCell<Keymap>>>,
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
        self.nodes.clear();
        self.node_stack.clear();
        self.context_stack.clear();
        self.focusable_node_ids.clear();
        self.tab_stops.clear();
    }

    /// How many nodes the tree has.
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Starts a node under the current one (an element's prepaint begins).
    pub(crate) fn push_node(&mut self) -> DispatchNodeId {
        let parent = self.node_stack.last().copied();
        let node_id = DispatchNodeId(self.nodes.len());
        self.nodes.push(DispatchNode { parent, ..Default::default() });
        self.node_stack.push(node_id);
        node_id
    }

    /// Ends the current node.
    pub(crate) fn pop_node(&mut self) {
        let Some(node_id) = self.node_stack.pop() else { return };
        if self.nodes[node_id.0].context.is_some() {
            self.context_stack.pop();
        }
    }

    /// Makes `node_id` the current node for paint (GPUI's `set_active_node`): pops back to its parent, or
    /// rebuilds the stack from the root when the parent is not on it (deferred draws).
    pub(crate) fn set_active_node(&mut self, node_id: DispatchNodeId) {
        let parent = self.nodes[node_id.0].parent;
        while self.node_stack.last().copied() != parent && !self.node_stack.is_empty() {
            self.pop_node();
        }
        if self.node_stack.last().copied() == parent {
            self.node_stack.push(node_id);
            if let Some(context) = self.nodes[node_id.0].context.clone() {
                self.context_stack.push(context);
            }
        } else {
            let mut current = Some(node_id);
            while let Some(id) = current {
                let node = &self.nodes[id.0];
                if let Some(context) = node.context.clone() {
                    self.context_stack.push(context);
                }
                self.node_stack.push(id);
                current = node.parent;
            }
            self.context_stack.reverse();
            self.node_stack.reverse();
        }
    }

    /// The current node, if any.
    pub(crate) fn active_node_id(&self) -> Option<DispatchNodeId> {
        self.node_stack.last().copied()
    }

    fn active_node(&mut self) -> Option<&mut DispatchNode> {
        let id = self.active_node_id()?;
        Some(&mut self.nodes[id.0])
    }

    /// Sets the current node's key context.
    pub(crate) fn set_key_context(&mut self, context: KeyContext) {
        if let Some(node) = self.active_node() {
            node.context = Some(context.clone());
            self.context_stack.push(context);
        }
    }

    /// Sets the current node's focus id.
    pub(crate) fn set_focus_id(&mut self, focus_id: FocusId) {
        if let Some(node_id) = self.active_node_id() {
            self.nodes[node_id.0].focus_id = Some(focus_id);
            self.focusable_node_ids.insert(focus_id, node_id);
        }
    }

    /// Records a tracked focus handle for Tab order (GPUI's `TabStopMap::insert`, called at paint).
    pub(crate) fn insert_tab_stop(&mut self, focus_id: FocusId, tab_index: isize, tab_stop: bool) {
        self.tab_stops.push(TabStop { focus_id, tab_index, tab_stop });
    }

    /// Adds a key listener to the current node.
    pub(crate) fn on_key_event(&mut self, listener: KeyListener) {
        if let Some(node) = self.active_node() {
            node.key_listeners.push(listener);
        }
    }

    /// Adds an action listener to the current node.
    pub(crate) fn on_action(&mut self, action_type: TypeId, listener: ActionListener) {
        if let Some(node) = self.active_node() {
            node.action_listeners.push(DispatchActionListener { action_type, listener });
        }
    }

    /// The node of `focus_id`, if it was painted.
    pub(crate) fn focusable_node_id(&self, focus_id: FocusId) -> Option<DispatchNodeId> {
        self.focusable_node_ids.get(&focus_id).copied()
    }

    /// Whether `child` is `parent` or inside it in this tree.
    pub(crate) fn focus_contains(&self, parent: FocusId, child: FocusId) -> bool {
        if parent == child {
            return true;
        }
        let Some(parent_node) = self.focusable_node_ids.get(&parent).copied() else { return false };
        let mut current = self.focusable_node_ids.get(&child).copied();
        while let Some(node_id) = current {
            if node_id == parent_node {
                return true;
            }
            current = self.nodes[node_id.0].parent;
        }
        false
    }

    /// The path from the root to `node`.
    pub(crate) fn dispatch_path(&self, node: DispatchNodeId) -> SmallVec<[DispatchNodeId; 32]> {
        let mut path = SmallVec::new();
        let mut current = Some(node);
        while let Some(id) = current {
            path.push(id);
            current = self.nodes.get(id.0).and_then(|node| node.parent);
        }
        path.reverse();
        path
    }

    /// The key contexts on `dispatch_path` (root first).
    pub(crate) fn context_path(&self, dispatch_path: &[DispatchNodeId]) -> SmallVec<[KeyContext; 4]> {
        dispatch_path.iter().filter_map(|id| self.nodes.get(id.0)?.context.clone()).collect()
    }

    /// A node.
    pub(crate) fn node(&self, id: DispatchNodeId) -> &DispatchNode {
        &self.nodes[id.0]
    }

    /// The root node (the window's root element), if any was painted.
    pub(crate) fn root_node_id(&self) -> Option<DispatchNodeId> {
        (!self.nodes.is_empty()).then_some(DispatchNodeId(0))
    }

    /// The bindings for `keystroke` under the contexts of `dispatch_path`, best first.
    pub(crate) fn bindings_for_input(
        &self,
        keystroke: &Keystroke,
        dispatch_path: &[DispatchNodeId],
    ) -> KeyMatch {
        let mut bindings = SmallVec::new();
        if let Some(keymap) = &self.keymap {
            let contexts = self.context_path(dispatch_path);
            let keymap = keymap.borrow();
            for binding in keymap.bindings_for_input(std::slice::from_ref(keystroke), &contexts) {
                bindings.push(binding.action().boxed_clone());
            }
        }
        KeyMatch { bindings }
    }

    /// The tab stop after (or before, with `reverse`) `focused`, wrapping (GPUI's `TabStopMap::next` /
    /// `prev`). The order is `(tab_index, paint order)`; a focused handle that is not a tab stop still
    /// anchors the search; one that was not painted starts from the beginning (or the end).
    pub(crate) fn next_tab_stop(&self, focused: Option<FocusId>, reverse: bool) -> Option<FocusId> {
        let key = |ix: usize| (self.tab_stops[ix].tab_index, ix);
        // GPUI looks the focused handle up by id; a handle inserted twice resolves to its last insertion.
        let anchor = focused.and_then(|id| self.tab_stops.iter().rposition(|stop| stop.focus_id == id));
        let candidates = || {
            (0..self.tab_stops.len()).filter(|&ix| {
                let stop = &self.tab_stops[ix];
                stop.tab_stop && self.tab_stops[ix + 1..].iter().all(|later| later.focus_id != stop.focus_id)
            })
        };
        let found = match (anchor, reverse) {
            (Some(anchor), false) => {
                candidates().filter(|ix| key(*ix) > key(anchor)).min_by_key(|ix| key(*ix))
            }
            (Some(anchor), true) => {
                candidates().filter(|ix| key(*ix) < key(anchor)).max_by_key(|ix| key(*ix))
            }
            (None, _) => None,
        };
        let found = found.or_else(|| {
            if reverse {
                candidates().max_by_key(|ix| key(*ix))
            } else {
                candidates().min_by_key(|ix| key(*ix))
            }
        });
        found.map(|ix| self.tab_stops[ix].focus_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree_with(stops: &[(u64, isize, bool)]) -> DispatchTree {
        let mut tree = DispatchTree::new();
        for &(id, tab_index, tab_stop) in stops {
            tree.insert_tab_stop(FocusId(id), tab_index, tab_stop);
        }
        tree
    }

    #[test]
    fn tab_order_follows_tab_index_then_paint_order_and_wraps() {
        // GPUI's `test_tab_handles` order: index 0 handles in paint order, then index 1, then 2.
        let tree = tree_with(&[
            (1, 0, true),
            (2, 1, true),
            (3, 1, true),
            (4, 0, false),
            (5, 2, false),
            (6, 0, true),
            (7, 2, true),
        ]);
        let mut order = Vec::new();
        let mut focused = None;
        for _ in 0..6 {
            focused = tree.next_tab_stop(focused, false);
            order.push(focused.unwrap().0);
        }
        assert_eq!(order, [1, 6, 2, 3, 7, 1]);
        let mut back = Vec::new();
        let mut focused = Some(FocusId(1));
        for _ in 0..5 {
            focused = tree.next_tab_stop(focused, true);
            back.push(focused.unwrap().0);
        }
        assert_eq!(back, [7, 3, 2, 6, 1]);
        // A non-tab-stop handle anchors the search; an unknown one starts over.
        assert_eq!(tree.next_tab_stop(Some(FocusId(4)), false), Some(FocusId(6)));
        assert_eq!(tree.next_tab_stop(Some(FocusId(99)), false), Some(FocusId(1)));
        assert_eq!(tree.next_tab_stop(Some(FocusId(99)), true), Some(FocusId(7)));
        assert_eq!(tree_with(&[(1, 0, false)]).next_tab_stop(None, false), None);

        // Duplicate insertions of the same FocusHandle resolve to the last occurrence and never
        // trap focus forward or backward.
        let dup_split = tree_with(&[(1, 0, true), (2, 0, true), (1, 0, true)]);
        assert_eq!(dup_split.next_tab_stop(None, false), Some(FocusId(2)));
        assert_eq!(dup_split.next_tab_stop(Some(FocusId(2)), false), Some(FocusId(1)));
        assert_eq!(dup_split.next_tab_stop(Some(FocusId(1)), false), Some(FocusId(2)));
        assert_eq!(dup_split.next_tab_stop(Some(FocusId(1)), true), Some(FocusId(2)));

        let dup_adj = tree_with(&[(1, 0, true), (1, 0, true), (2, 0, true)]);
        assert_eq!(dup_adj.next_tab_stop(Some(FocusId(1)), false), Some(FocusId(2)));
        assert_eq!(dup_adj.next_tab_stop(Some(FocusId(1)), true), Some(FocusId(2)));
    }

    #[test]
    fn paths_contexts_and_active_node() {
        let mut tree = DispatchTree::new();
        let root = tree.push_node();
        tree.set_focus_id(FocusId(10));
        let a = tree.push_node();
        let b = tree.push_node();
        tree.set_focus_id(FocusId(12));
        tree.pop_node();
        tree.pop_node();
        let c = tree.push_node();
        tree.set_focus_id(FocusId(13));
        tree.pop_node();
        tree.pop_node();
        assert!(tree.node_stack.is_empty());
        assert_eq!(tree.dispatch_path(b).as_slice(), [root, a, b]);
        assert!(tree.focus_contains(FocusId(10), FocusId(13)));
        assert!(!tree.focus_contains(FocusId(12), FocusId(13)));

        // Paint: contexts follow the active node, siblings pop back to the shared parent.
        tree.set_active_node(root);
        tree.set_key_context(KeyContext(smallvec::smallvec!["Root".into()]));
        tree.set_active_node(a);
        tree.set_key_context(KeyContext(smallvec::smallvec!["A".into()]));
        tree.set_active_node(b);
        assert_eq!(tree.context_stack.len(), 2);
        tree.set_active_node(c);
        assert_eq!(tree.node_stack.as_slice(), [root, c]);
        assert_eq!(tree.context_stack.len(), 1);
        assert_eq!(tree.context_path(&tree.dispatch_path(b)).len(), 2);
        // From an empty stack (a deferred draw), the whole path is rebuilt.
        tree.node_stack.clear();
        tree.context_stack.clear();
        tree.set_active_node(b);
        assert_eq!(tree.node_stack.as_slice(), [root, a, b]);
        assert_eq!(tree.context_stack.len(), 2);
    }
}
