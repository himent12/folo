//! `ScrollHandle`: a scroll container's shared state (offset, bounds, max offset, child bounds), which the
//! app reads and writes between frames, as GPUI 0.2.2's. Owner tag: `"window"`.
//!
//! The offset is negative when scrolled down / right. The container clamps it at prepaint (to
//! `-max_offset..=0`), and `scroll_to_item` / `scroll_to_bottom` take effect at the next prepaint.

use std::{cell::RefCell, rc::Rc};

use crate::{Bounds, Overflow, Pixels, Point, Size};

/// How `scroll_to_item` brings a child into view (GPUI's private `ScrollStrategy` of `div.rs`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ScrollToStrategy {
    /// Scroll the least that makes the child visible.
    #[default]
    FirstVisible,
    /// Put the child's top at the top.
    Top,
}

/// The state behind a `ScrollHandle`.
#[derive(Default, Debug)]
pub(crate) struct ScrollHandleState {
    /// The scroll offset (negative when scrolled down / right, as GPUI).
    pub(crate) offset: Rc<RefCell<Point<Pixels>>>,
    /// The container's bounds last frame.
    pub(crate) bounds: Bounds<Pixels>,
    /// How far it can scroll.
    pub(crate) max_offset: Size<Pixels>,
    /// Each child's bounds last frame.
    pub(crate) child_bounds: Vec<Bounds<Pixels>>,
    /// Scroll to the end at the next prepaint.
    pub(crate) scroll_to_bottom: bool,
    /// The container's overflow last frame (which axes scroll).
    pub(crate) overflow: Point<Overflow>,
    /// Scroll a child into view at the next prepaint.
    pub(crate) active_item: Option<(usize, ScrollToStrategy)>,
}

/// A handle to a scroll container (`.track_scroll(&handle)`).
#[derive(Clone, Debug, Default)]
pub struct ScrollHandle(pub(crate) Rc<RefCell<ScrollHandleState>>);

impl ScrollHandle {
    /// A new handle.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `self` and `other` point to the same underlying scroll state.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    /// The current offset.
    pub fn offset(&self) -> Point<Pixels> {
        *self.0.borrow().offset.borrow()
    }

    /// Sets the offset (clamped at the next prepaint) and clears any queued `scroll_to_item` or
    /// `scroll_to_bottom` request so an earlier deferred scroll cannot overwrite it on prepaint.
    pub fn set_offset(&self, position: Point<Pixels>) {
        let mut state = self.0.borrow_mut();
        state.active_item = None;
        state.scroll_to_bottom = false;
        *state.offset.borrow_mut() = position;
    }

    /// The largest offset (content size plus padding minus container size, two decimals).
    pub fn max_offset(&self) -> Size<Pixels> {
        self.0.borrow().max_offset
    }

    /// The container's bounds last frame.
    pub fn bounds(&self) -> Bounds<Pixels> {
        self.0.borrow().bounds
    }

    /// The bounds of child `ix` last frame.
    pub fn bounds_for_item(&self, ix: usize) -> Option<Bounds<Pixels>> {
        self.0.borrow().child_bounds.get(ix).copied()
    }

    /// Scrolls the least that makes child `ix` visible, at the next frame.
    pub fn scroll_to_item(&self, ix: usize) {
        self.0.borrow_mut().active_item = Some((ix, ScrollToStrategy::FirstVisible));
    }

    /// Scrolls child `ix` to the top, at the next frame.
    pub fn scroll_to_top_of_item(&self, ix: usize) {
        self.0.borrow_mut().active_item = Some((ix, ScrollToStrategy::Top));
    }

    /// Scrolls to the end at the next frame.
    pub fn scroll_to_bottom(&self) {
        self.0.borrow_mut().scroll_to_bottom = true;
    }

    /// How many children the container had last frame.
    pub fn children_count(&self) -> usize {
        self.0.borrow().child_bounds.len()
    }
}

impl ScrollHandleState {
    /// Applies a pending `scroll_to_item` against the current container bounds, overflow, and child
    /// bounds (GPUI's `scroll_to_active_item`; a child that does not exist yet keeps the request
    /// for later).
    pub(crate) fn scroll_to_active_item(&mut self) {
        let Some((index, strategy)) = self.active_item else { return };
        let Some(bounds) = self.child_bounds.get(index).copied() else { return };
        let container = self.bounds;
        let overflow = self.overflow;
        {
            let mut offset = self.offset.borrow_mut();
            match strategy {
                ScrollToStrategy::FirstVisible => {
                    if overflow.y == Overflow::Scroll {
                        if bounds.top() + offset.y < container.top() {
                            offset.y = container.top() - bounds.top();
                        } else if bounds.bottom() + offset.y > container.bottom() {
                            offset.y = container.bottom() - bounds.bottom();
                        }
                    }
                }
                ScrollToStrategy::Top => {
                    if overflow.y == Overflow::Scroll {
                        offset.y = container.top() - bounds.top();
                    }
                }
            }
            if overflow.x == Overflow::Scroll {
                if bounds.left() + offset.x < container.left() {
                    offset.x = container.left() - bounds.left();
                } else if bounds.right() + offset.x > container.right() {
                    offset.x = container.right() - bounds.right();
                }
            }
        }
        self.active_item = None;
    }
}
