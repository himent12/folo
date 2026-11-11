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
//! Performance: the taffy tree is cleared, not rebuilt, each frame (its node storage keeps capacity), and
//! so are the bounds cache and the child id buffer; measure functions are boxed once per measured node.

use std::collections::{HashMap, HashSet};

use smallvec::SmallVec;

use crate::{App, AvailableSpace, Bounds, LayoutId, Pixels, Point, Size, Style, Window, point, size};

/// A measure function for a leaf whose size depends on its content (text, `canvas` with a size).
pub(crate) type NodeMeasureFn =
    Box<dyn FnMut(Size<Option<Pixels>>, Size<AvailableSpace>, &mut Window, &mut App) -> Size<Pixels>>;

/// The context taffy keeps per measured node.
pub(crate) struct NodeContext {
    /// The measure function.
    pub(crate) measure: NodeMeasureFn,
}

const EXPECT_MESSAGE: &str = "taffy layout errors are avoided by construction";

/// The window's taffy tree and the bounds it computed this frame.
pub(crate) struct TaffyLayoutEngine {
    /// The tree.
    pub(crate) taffy: taffy::TaffyTree<NodeContext>,
    /// Absolute bounds already computed this frame, by node.
    pub(crate) absolute_layout_bounds: HashMap<LayoutId, Bounds<Pixels>>,
    /// Roots computed this frame (so `compute_layout` runs once per root and available space).
    pub(crate) computed_layouts: HashSet<LayoutId>,
    /// Reused buffer for the children of the node being added.
    pub(crate) child_buffer: Vec<taffy::NodeId>,
}

impl TaffyLayoutEngine {
    /// An empty engine (taffy's rounding on, as GPUI).
    pub(crate) fn new() -> Self {
        let mut taffy = taffy::TaffyTree::new();
        taffy.enable_rounding();
        TaffyLayoutEngine {
            taffy,
            absolute_layout_bounds: HashMap::new(),
            computed_layouts: HashSet::new(),
            child_buffer: Vec::new(),
        }
    }

    /// Empties the tree for the next frame, keeping capacity.
    pub(crate) fn clear(&mut self) {
        self.taffy.clear();
        self.absolute_layout_bounds.clear();
        self.computed_layouts.clear();
    }

    /// Adds a node with `style` and `children`.
    pub(crate) fn request_layout(
        &mut self,
        style: &Style,
        rem_size: Pixels,
        scale_factor: f32,
        children: impl IntoIterator<Item = LayoutId>,
    ) -> LayoutId {
        let taffy_style = style.to_taffy(rem_size, scale_factor);
        self.child_buffer.clear();
        self.child_buffer.extend(children.into_iter().map(|id| id.0));
        let node = if self.child_buffer.is_empty() {
            self.taffy.new_leaf(taffy_style)
        } else {
            self.taffy.new_with_children(taffy_style, &self.child_buffer)
        };
        LayoutId(node.expect(EXPECT_MESSAGE))
    }

    /// Adds a leaf measured by `measure`.
    pub(crate) fn request_measured_layout(
        &mut self,
        style: &Style,
        rem_size: Pixels,
        scale_factor: f32,
        measure: NodeMeasureFn,
    ) -> LayoutId {
        let taffy_style = style.to_taffy(rem_size, scale_factor);
        LayoutId(
            self.taffy.new_leaf_with_context(taffy_style, NodeContext { measure }).expect(EXPECT_MESSAGE),
        )
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
                self.absolute_layout_bounds.remove(&id);
                stack.extend(self.taffy.children(id.0).expect(EXPECT_MESSAGE).into_iter().map(LayoutId));
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

        self.taffy
            .compute_layout_with_measure(
                id.0,
                taffy_space,
                |known_dimensions, available_space, _id, node_context, _style| {
                    let Some(node_context) = node_context else {
                        return taffy::Size::ZERO;
                    };
                    let known_dimensions = Size {
                        width: known_dimensions.width.map(|e| Pixels(e / scale_factor)),
                        height: known_dimensions.height.map(|e| Pixels(e / scale_factor)),
                    };
                    let unscale = |space: taffy::AvailableSpace| match AvailableSpace::from(space) {
                        AvailableSpace::Definite(pixels) => {
                            AvailableSpace::Definite(Pixels(pixels.0 / scale_factor))
                        }
                        other => other,
                    };
                    let available_space =
                        size(unscale(available_space.width), unscale(available_space.height));
                    let measured = (node_context.measure)(known_dimensions, available_space, window, cx);
                    taffy::Size {
                        width: measured.width.0 * scale_factor,
                        height: measured.height.0 * scale_factor,
                    }
                },
            )
            .expect(EXPECT_MESSAGE);
    }

    /// A node's bounds relative to the root it was computed under: taffy's rounded device pixels divided by
    /// the scale factor, parent origins added (GPUI's `layout_bounds`).
    pub(crate) fn layout_bounds(&mut self, id: LayoutId, scale_factor: f32) -> Bounds<Pixels> {
        if let Some(bounds) = self.absolute_layout_bounds.get(&id) {
            return *bounds;
        }
        let layout = self.taffy.layout(id.0).expect(EXPECT_MESSAGE);
        let mut bounds = Bounds {
            origin: point(Pixels(layout.location.x / scale_factor), Pixels(layout.location.y / scale_factor)),
            size: size(Pixels(layout.size.width / scale_factor), Pixels(layout.size.height / scale_factor)),
        };
        if let Some(parent) = self.taffy.parent(id.0) {
            bounds.origin += self.layout_bounds(LayoutId(parent), scale_factor).origin;
        }
        self.absolute_layout_bounds.insert(id, bounds);
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
        let rem_size = self.rem_size();
        let scale_factor = self.scale_factor();
        self.layout_engine_mut().request_layout(&style, rem_size, scale_factor, children)
    }

    /// Adds a leaf whose size `measure` computes from the known dimensions and available space.
    pub fn request_measured_layout<F>(&mut self, style: Style, measure: F) -> LayoutId
    where
        F: FnMut(Size<Option<Pixels>>, Size<AvailableSpace>, &mut Window, &mut App) -> Size<Pixels> + 'static,
    {
        let rem_size = self.rem_size();
        let scale_factor = self.scale_factor();
        self.layout_engine_mut().request_measured_layout(&style, rem_size, scale_factor, Box::new(measure))
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
