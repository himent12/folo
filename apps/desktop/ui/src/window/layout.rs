//! Layout through taffy (`=0.9.0`, GPUI 0.2.2's version): the per-window `TaffyLayoutEngine` and the
//! `Window` layout API (`request_layout`, `request_measured_layout`, `compute_layout`, `layout_bounds`).
//! Owner tag: `"window"`.
//!
//! GPUI's arithmetic, exactly: styles become taffy styles in device pixels (`Style::to_taffy(rem, scale)`
//! multiplies lengths by the scale factor and does not round), taffy's own rounding is on, so every
//! location and size taffy reports is a whole device pixel relative to its parent, and the bounds handed
//! back are those divided by the scale factor, with parent origins added (cached per node). Available
//! space and measure results cross the seam scaled the same way.
//!
//! Performance: the tree (`layout_tree.rs`) runs taffy's algorithms over its own nodes and keeps last
//! frame's results, so a subtree that did not change is answered from them instead of laid out again,
//! with exactly the results a fresh layout gives; measure functions are boxed once per measured node.

use std::collections::HashSet;

use smallvec::SmallVec;

use crate::{App, AvailableSpace, Bounds, LayoutId, Pixels, Point, Size, Style, Window, point, size};

use super::layout_tree::LayoutTree;
pub(crate) use super::layout_tree::{MeasureKey, Measured};

/// The window's layout tree and the bounds it computed this frame.
pub(crate) struct TaffyLayoutEngine {
    /// The tree (it keeps last frame's results for the subtrees that did not change).
    pub(crate) tree: LayoutTree,
    /// Absolute bounds already computed this frame, indexed densely by `usize::from(id.0)`.
    pub(crate) absolute_layout_bounds: Vec<Option<Bounds<Pixels>>>,
    /// Roots computed this frame (so `compute_layout` runs once per root and available space).
    pub(crate) computed_layouts: HashSet<LayoutId>,
    /// Reused buffer for the children of the node being added.
    pub(crate) child_buffer: Vec<taffy::NodeId>,
}

impl TaffyLayoutEngine {
    /// An empty engine (taffy's rounding on, as GPUI).
    pub(crate) fn new() -> Self {
        TaffyLayoutEngine {
            tree: LayoutTree::default(),
            absolute_layout_bounds: Vec::new(),
            computed_layouts: HashSet::new(),
            child_buffer: Vec::new(),
        }
    }

    /// Starts the next frame: this frame's nodes become the ones the next frame can reuse.
    pub(crate) fn clear(&mut self) {
        self.tree.clear();
        self.absolute_layout_bounds.clear();
        self.computed_layouts.clear();
    }

    /// Adds a node with `style` and `children`.
    pub(crate) fn request_layout(
        &mut self,
        style: &Style,
        rem_size: Pixels,
        scale_factor: f32,
        font_generation: u64,
        children: impl IntoIterator<Item = LayoutId>,
    ) -> LayoutId {
        let taffy_style = style.to_taffy(rem_size, scale_factor);
        self.child_buffer.clear();
        self.child_buffer.extend(children.into_iter().map(|id| id.0));
        LayoutId(self.tree.add(taffy_style, &self.child_buffer, None, scale_factor, font_generation))
    }

    /// Adds a leaf measured by `measured`.
    pub(crate) fn request_measured_layout(
        &mut self,
        style: &Style,
        rem_size: Pixels,
        scale_factor: f32,
        font_generation: u64,
        measured: Measured,
    ) -> LayoutId {
        let taffy_style = style.to_taffy(rem_size, scale_factor);
        LayoutId(self.tree.add(taffy_style, &[], Some(measured), scale_factor, font_generation))
    }

    /// Computes the layout under `id` in `available_space` (measured leaves call back into the window).
    /// A second computation of the same root forgets the bounds cached under it (GPUI).
    pub(crate) fn compute_layout(
        &mut self,
        id: LayoutId,
        available_space: Size<AvailableSpace>,
        window: &mut Window,
        cx: &mut App,
    ) {
        if !self.computed_layouts.insert(id) {
            let mut stack = SmallVec::<[LayoutId; 64]>::new();
            stack.push(id);
            while let Some(id) = stack.pop() {
                if let Some(slot) = self.absolute_layout_bounds.get_mut(usize::from(id.0)) {
                    *slot = None;
                }
                stack.extend(self.tree.children(id.0).iter().copied().map(LayoutId));
            }
        }

        let scale_factor = window.scale_factor();
        let scale = |space: AvailableSpace| match space {
            AvailableSpace::Definite(pixels) => AvailableSpace::Definite(Pixels(pixels.0 * scale_factor)),
            other => other,
        };
        let available_space = size(scale(available_space.width), scale(available_space.height));
        let taffy_space =
            taffy::Size { width: available_space.width.into(), height: available_space.height.into() };
        self.tree.compute(id.0, taffy_space, window, cx);
    }

    /// A node's bounds relative to the root it was computed under: taffy's rounded device pixels divided by
    /// the scale factor, parent origins added (GPUI's `layout_bounds`).
    pub(crate) fn layout_bounds(&mut self, id: LayoutId, scale_factor: f32) -> Bounds<Pixels> {
        let index = usize::from(id.0);
        if let Some(bounds) = self.absolute_layout_bounds.get(index).copied().flatten() {
            return bounds;
        }
        let layout = self.tree.final_layout(id.0);
        let mut bounds = Bounds {
            origin: point(Pixels(layout.location.x / scale_factor), Pixels(layout.location.y / scale_factor)),
            size: size(Pixels(layout.size.width / scale_factor), Pixels(layout.size.height / scale_factor)),
        };
        if let Some(parent) = self.tree.parent(id.0) {
            bounds.origin += self.layout_bounds(LayoutId(parent), scale_factor).origin;
        }
        if index >= self.absolute_layout_bounds.len() {
            self.absolute_layout_bounds.resize(index + 1, None);
        }
        self.absolute_layout_bounds[index] = Some(bounds);
        bounds
    }
}

impl Window {
    /// Adds a layout node for an element with `style` and child nodes `children`.
    pub fn request_layout(
        &mut self,
        style: Style,
        children: impl IntoIterator<Item = LayoutId>,
        _cx: &mut App,
    ) -> LayoutId {
        self.request_layout_ref(&style, children)
    }

    /// Adds a layout node from a borrowed `style` (avoids cloning `Style` for internal elements).
    pub(crate) fn request_layout_ref(
        &mut self,
        style: &Style,
        children: impl IntoIterator<Item = LayoutId>,
    ) -> LayoutId {
        let rem_size = self.rem_size();
        let scale_factor = self.scale_factor();
        let font_generation = self.text_system.font_generation();
        self.layout_engine_mut().request_layout(style, rem_size, scale_factor, font_generation, children)
    }

    /// Adds a leaf whose size `measure` computes from the known dimensions and available space.
    pub fn request_measured_layout<F>(&mut self, style: Style, measure: F) -> LayoutId
    where
        F: FnMut(Size<Option<Pixels>>, Size<AvailableSpace>, &mut Window, &mut App) -> Size<Pixels> + 'static,
    {
        let measured = Measured { measure: Box::new(measure), key: None, reset: None };
        let rem_size = self.rem_size();
        let scale_factor = self.scale_factor();
        let font_generation = self.text_system.font_generation();
        self.layout_engine_mut().request_measured_layout(
            &style,
            rem_size,
            scale_factor,
            font_generation,
            measured,
        )
    }

    /// `request_measured_layout` for a measure function whose results depend only on `key` and the requests
    /// (so the node can be reused next frame); `reset` forgets the state measuring leaves behind.
    pub(crate) fn request_keyed_measured_layout(
        &mut self,
        style: Style,
        key: MeasureKey,
        reset: impl Fn() + 'static,
        measure: impl FnMut(Size<Option<Pixels>>, Size<AvailableSpace>, &mut Window, &mut App) -> Size<Pixels>
        + 'static,
    ) -> LayoutId {
        let measured = Measured { measure: Box::new(measure), key: Some(key), reset: Some(Box::new(reset)) };
        let rem_size = self.rem_size();
        let scale_factor = self.scale_factor();
        let font_generation = self.text_system.font_generation();
        self.layout_engine_mut().request_measured_layout(
            &style,
            rem_size,
            scale_factor,
            font_generation,
            measured,
        )
    }

    /// Computes the layout under `layout_id` in `available_space`.
    pub fn compute_layout(
        &mut self,
        layout_id: LayoutId,
        available_space: Size<AvailableSpace>,
        cx: &mut App,
    ) {
        // The engine is taken out for the call: measure functions get the window.
        let mut engine = self.layout_engine.take().expect("compute_layout is not re-entrant");
        engine.compute_layout(layout_id, available_space, self, cx);
        self.layout_engine = Some(engine);
    }

    /// A node's bounds in window coordinates (adds the current element offset).
    pub fn layout_bounds(&mut self, layout_id: LayoutId) -> Bounds<Pixels> {
        let scale_factor = self.scale_factor();
        let mut bounds = self.layout_engine_mut().layout_bounds(layout_id, scale_factor);
        bounds.origin += self.element_offset();
        bounds
    }

    /// The current element offset (scroll containers and `prepaint_at` push it).
    pub fn element_offset(&self) -> Point<Pixels> {
        self.element_offset_stack.last().copied().unwrap_or_default()
    }

    fn layout_engine_mut(&mut self) -> &mut TaffyLayoutEngine {
        self.layout_engine.as_mut().expect("the layout engine is out only during compute_layout")
    }
}
