//! Focus: `FocusId`, `FocusHandle`, `WeakFocusHandle`, the ref-counted `FocusMap` the app owns, and the
//! `Focusable` trait. Owner tag: `"window"`.
//!
//! A handle's id is in the map while any strong handle lives. Which handle is focused is per window
//! (`Window::focus`); whether an element "contains" focus is answered from the last rendered frame's
//! dispatch tree (the focus path), as GPUI does.
use std::{
    collections::HashMap,
    fmt,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use parking_lot::RwLock;

use crate::{App, Window};

/// A focus handle's id (process-unique).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FocusId(pub(crate) u64);

impl FocusId {
    /// Whether this id is the window's focused element.
    pub fn is_focused(&self, window: &Window) -> bool {
        window.focus == Some(*self)
    }

    /// Whether the focused element is this one or a descendant of it in the last frame.
    pub fn contains_focused(&self, window: &Window, cx: &App) -> bool {
        window.focused_id(cx).is_some_and(|focused| self.contains(focused, window))
    }

    /// Whether this element is the focused one or inside it in the last frame.
    pub fn within_focused(&self, window: &Window, cx: &App) -> bool {
        window.focused_id(cx).is_some_and(|focused| focused.contains(*self, window))
    }

    /// Whether `other` is this element or inside it in the last rendered frame.
    pub(crate) fn contains(&self, other: FocusId, window: &Window) -> bool {
        window.rendered_frame.dispatch_tree.focus_contains(*self, other)
    }
}

/// One focus handle's entry.
pub(crate) struct FocusRef {
    /// Strong handles alive.
    pub(crate) ref_count: AtomicUsize,
    /// `FocusHandle::tab_stop`.
    pub(crate) tab_stop: bool,
    /// `FocusHandle::tab_index`.
    pub(crate) tab_index: isize,
}

/// Every live focus handle (owned by the app, shared with each handle).
#[derive(Default)]
pub struct FocusMap {
    /// The entries by id.
    pub(crate) handles: RwLock<HashMap<FocusId, FocusRef>>,
    /// The next id.
    pub(crate) next_id: AtomicU64,
}

/// A focusable thing: track it with `.track_focus(&handle)`, focus it with `window.focus(&handle)`.
pub struct FocusHandle {
    /// The id.
    pub id: FocusId,
    /// The map this handle counts in.
    pub(crate) handles: Arc<FocusMap>,
    /// `tab_stop` / `tab_index` (copied into the map's entry).
    pub tab_stop: bool,
    /// The order among tab stops (lower first; equal ones in tree order).
    pub tab_index: isize,
}

impl FocusHandle {
    /// A new handle in `handles`.
    pub(crate) fn new(handles: &Arc<FocusMap>) -> Self {
        let id = FocusId(handles.next_id.fetch_add(1, Ordering::Relaxed) + 1);
        handles
            .handles
            .write()
            .insert(id, FocusRef { ref_count: AtomicUsize::new(1), tab_stop: false, tab_index: 0 });
        FocusHandle { id, handles: handles.clone(), tab_stop: false, tab_index: 0 }
    }

    /// A strong handle to an existing id, if it is still alive.
    pub(crate) fn for_id(id: FocusId, handles: &Arc<FocusMap>) -> Option<Self> {
        let map = handles.handles.read();
        let entry = map.get(&id)?;
        // A count of zero means the last handle is being dropped right now.
        let mut count = entry.ref_count.load(Ordering::SeqCst);
        loop {
            if count == 0 {
                return None;
            }
            match entry.ref_count.compare_exchange(count, count + 1, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => break,
                Err(actual) => count = actual,
            }
        }
        Some(FocusHandle {
            id,
            handles: handles.clone(),
            tab_stop: entry.tab_stop,
            tab_index: entry.tab_index,
        })
    }

    /// Makes the handle a Tab stop (or not).
    pub fn tab_stop(mut self, tab_stop: bool) -> Self {
        self.tab_stop = tab_stop;
        if let Some(entry) = self.handles.handles.write().get_mut(&self.id) {
            entry.tab_stop = tab_stop;
        }
        self
    }

    /// Sets its position in Tab order.
    pub fn tab_index(mut self, tab_index: isize) -> Self {
        self.tab_index = tab_index;
        if let Some(entry) = self.handles.handles.write().get_mut(&self.id) {
            entry.tab_index = tab_index;
        }
        self
    }

    /// A weak handle.
    pub fn downgrade(&self) -> WeakFocusHandle {
        WeakFocusHandle { id: self.id, handles: Arc::downgrade(&self.handles) }
    }

    /// Focuses this handle in `window`.
    pub fn focus(&self, window: &mut Window) {
        window.focus(self)
    }

    /// Whether it is the window's focused handle.
    pub fn is_focused(&self, window: &Window) -> bool {
        self.id.is_focused(window)
    }

    /// Whether the focused element is this one or inside it.
    pub fn contains_focused(&self, window: &Window, cx: &App) -> bool {
        self.id.contains_focused(window, cx)
    }

    /// Whether this element is inside the focused one.
    pub fn within_focused(&self, window: &Window, cx: &App) -> bool {
        self.id.within_focused(window, cx)
    }
}

impl Clone for FocusHandle {
    fn clone(&self) -> Self {
        // `self` keeps the entry alive, so it is there.
        Self::for_id(self.id, &self.handles).expect("a live focus handle has an entry")
    }
}

impl Drop for FocusHandle {
    fn drop(&mut self) {
        let mut map = self.handles.handles.write();
        if let Some(entry) = map.get(&self.id)
            && entry.ref_count.fetch_sub(1, Ordering::SeqCst) == 1
        {
            map.remove(&self.id);
        }
    }
}

impl PartialEq for FocusHandle {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for FocusHandle {}

impl fmt::Debug for FocusHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FocusHandle").field("id", &self.id).finish()
    }
}

/// A focus handle that does not keep its id alive.
#[derive(Clone)]
pub struct WeakFocusHandle {
    /// The id.
    pub(crate) id: FocusId,
    /// The map.
    pub(crate) handles: Weak<FocusMap>,
}

impl WeakFocusHandle {
    /// A strong handle, if the id is still alive.
    pub fn upgrade(&self) -> Option<FocusHandle> {
        FocusHandle::for_id(self.id, &self.handles.upgrade()?)
    }
}

/// An entity that has a focus handle (`TextInput`, `FoloApp`).
pub trait Focusable: 'static {
    /// Its focus handle.
    fn focus_handle(&self, cx: &App) -> FocusHandle;
}
