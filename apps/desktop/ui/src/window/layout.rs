//! Layout through taffy (`=0.9.0`, GPUI 0.2.2's version): the per-window `TaffyLayoutEngine` and the
//! `Window` layout API (`request_layout`, `request_measured_layout`, `compute_layout`, `layout_bounds`).
//! Owner tag: `"window"`. Styles become taffy styles through the style package's `Style::to_taffy`, and
//! bounds are rounded exactly as GPUI rounds them.
//!
//! Performance: the taffy tree is cleared, not rebuilt, each frame (its node storage keeps capacity);
//! measure functions are boxed once per measured node per frame.
#![allow(dead_code)] // skeleton

use std::collections::HashMap;

use crate::{App, AvailableSpace, Bounds, LayoutId, Pixels, Point, Size, Style, Window};

/// A measure function for a leaf whose size depends on its content (text, `canvas` with a size).
pub(crate) type NodeMeasureFn =
    Box<dyn FnMut(Size<Option<Pixels>>, Size<AvailableSpace>, &mut Window, &mut App) -> Size<Pixels>>;

/// The context taffy keeps per measured node.
pub(crate) struct NodeContext {
    /// The measure function.
    pub(crate) measure: NodeMeasureFn,
}

/// The window's taffy tree and the bounds it computed this frame.
pub(crate) struct TaffyLayoutEngine {
    /// The tree.
    pub(crate) taffy: taffy::TaffyTree<NodeContext>,
    /// Absolute bounds already computed this frame, by node.
    pub(crate) absolute_layout_bounds: HashMap<LayoutId, Bounds<Pixels>>,
    /// Roots computed this frame (so `compute_layout` runs once per root and available space).
    pub(crate) computed_layouts: std::collections::HashSet<LayoutId>,
}

impl TaffyLayoutEngine {
    /// An empty engine.
    pub(crate) fn new() -> Self {
        TaffyLayoutEngine {
            taffy: taffy::TaffyTree::new(),
            absolute_layout_bounds: HashMap::new(),
            computed_layouts: std::collections::HashSet::new(),
        }
    }

    /// Empties the tree for the next frame, keeping capacity.
    pub(crate) fn clear(&mut self) {
        todo!("window")
    }

    /// Adds a node with `style` and `children`.
    pub(crate) fn request_layout(&mut self, _style: Style, _rem_size: Pixels, _children: &[LayoutId]) -> LayoutId {
        todo!("window")
    }

    /// Adds a leaf measured by `measure`.
    pub(crate) fn request_measured_layout(&mut self, _style: Style, _rem_size: Pixels, _measure: NodeMeasureFn) -> LayoutId {
        todo!("window")
    }

    /// Computes the layout under `id` in `available_space` (measured leaves call back into the window).
    pub(crate) fn compute_layout(
        &mut self,
        _id: LayoutId,
        _available_space: Size<AvailableSpace>,
        _window: &mut Window,
        _cx: &mut App,
    ) {
        todo!("window")
    }

    /// A node's bounds relative to the root it was computed under, rounded as GPUI.
    pub(crate) fn layout_bounds(&mut self, _id: LayoutId) -> Bounds<Pixels> {
        todo!("window")
    }
}

impl Window {
    /// Adds a layout node for an element with `style` and child nodes `children`.
    pub fn request_layout(
        &mut self,
        _style: Style,
        _children: impl IntoIterator<Item = LayoutId>,
        _cx: &mut App,
    ) -> LayoutId {
        todo!("window")
    }

    /// Adds a leaf whose size `measure` computes from the known dimensions and available space.
    pub fn request_measured_layout<F>(&mut self, _style: Style, _measure: F) -> LayoutId
    where
        F: FnMut(Size<Option<Pixels>>, Size<AvailableSpace>, &mut Window, &mut App) -> Size<Pixels> + 'static,
    {
        todo!("window")
    }

    /// Computes the layout under `layout_id` in `available_space`.
    pub fn compute_layout(&mut self, _layout_id: LayoutId, _available_space: Size<AvailableSpace>, _cx: &mut App) {
        todo!("window")
    }

    /// A node's bounds in window coordinates (adds the current element offset).
    pub fn layout_bounds(&mut self, _layout_id: LayoutId) -> Bounds<Pixels> {
        todo!("window")
    }

    /// The current element offset (scroll containers and `prepaint_at` push it).
    pub fn element_offset(&self) -> Point<Pixels> {
        self.element_offset_stack.last().copied().unwrap_or_default()
    }
}
