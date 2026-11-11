//! `ScrollHandle`: a scroll container's shared state (offset, bounds, max offset, child bounds), which the
//! app reads and writes between frames. Owner tag: `"window"`.
#![allow(dead_code)] // skeleton

use std::{cell::RefCell, rc::Rc};

use crate::{Bounds, Pixels, Point, ScrollStrategy, Size};

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
    /// Scroll a child into view at the next prepaint.
    pub(crate) active_item: Option<(usize, ScrollStrategy)>,
}

/// A handle to a scroll container (`.track_scroll(&handle)`).
#[derive(Clone, Debug, Default)]
pub struct ScrollHandle(pub(crate) Rc<RefCell<ScrollHandleState>>);

impl ScrollHandle {
    /// A new handle.
    pub fn new() -> Self {
        Self::default()
    }

    /// The current offset.
    pub fn offset(&self) -> Point<Pixels> {
        *self.0.borrow().offset.borrow()
    }

    /// Sets the offset (clamped at the next prepaint).
    pub fn set_offset(&self, position: Point<Pixels>) {
        *self.0.borrow().offset.borrow_mut() = position;
    }

    /// The largest offset (content size minus container size).
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

    /// Scrolls child `ix` into view at the next frame.
    pub fn scroll_to_item(&self, ix: usize) {
        self.0.borrow_mut().active_item = Some((ix, ScrollStrategy::default()));
    }

    /// Scrolls to the end at the next frame.
    pub fn scroll_to_bottom(&self) {
        self.0.borrow_mut().scroll_to_bottom = true;
    }
}
