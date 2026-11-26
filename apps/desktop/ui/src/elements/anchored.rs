// Adapted from GPUI (Apache-2.0): crates/gpui/src/elements/anchored.rs (the fit and snap rules)
//! `anchored()`: positions its children at a point (or its own layout position), anchored by a corner, and
//! keeps them inside the window (`snap_to_window_with_margin`). Owner tag: `"elements"`.

use std::panic;

use smallvec::SmallVec;

use crate::{
    AnyElement, App, Axis, Bounds, Corner, Display, Edges, Element, ElementId, GlobalElementId,
    InspectorElementId, IntoElement, LayoutId, ParentElement, Pixels, Point, Position, Size, Style, Window,
    point,
};

/// A new anchored element.
pub fn anchored() -> Anchored {
    Anchored {
        children: SmallVec::new(),
        anchor_corner: Corner::TopLeft,
        fit_mode: AnchoredFitMode::SwitchAnchor,
        anchor_position: None,
        offset: None,
    }
}

/// How an anchored element stays on screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AnchoredFitMode {
    /// Move to the window's edge (with a margin).
    SnapToWindowWithMargin(Edges<Pixels>),
    /// Flip the anchor corner when it would overflow.
    SwitchAnchor,
}

/// The anchored element.
pub struct Anchored {
    /// The children.
    pub(crate) children: SmallVec<[AnyElement; 2]>,
    /// The corner placed at the position.
    pub(crate) anchor_corner: Corner,
    /// How it stays on screen.
    pub(crate) fit_mode: AnchoredFitMode,
    /// Where (window px); `None` is its own layout position.
    pub(crate) anchor_position: Option<Point<Pixels>>,
    /// Added to the position.
    pub(crate) offset: Option<Point<Pixels>>,
}

impl Anchored {
    /// Which corner sits at the position.
    pub fn anchor(mut self, anchor: Corner) -> Self {
        self.anchor_corner = anchor;
        self
    }

    /// Where to put it (window px).
    pub fn position(mut self, anchor: Point<Pixels>) -> Self {
        self.anchor_position = Some(anchor);
        self
    }

    /// An offset added to the position.
    pub fn offset(mut self, offset: Point<Pixels>) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Keeps it inside the window, `edges` from its sides.
    pub fn snap_to_window_with_margin(mut self, edges: impl Into<Edges<Pixels>>) -> Self {
        self.fit_mode = AnchoredFitMode::SnapToWindowWithMargin(edges.into());
        self
    }
}

impl ParentElement for Anchored {
    fn extend(&mut self, elements: impl IntoIterator<Item = AnyElement>) {
        self.children.extend(elements)
    }
}

/// What `Anchored::request_layout` keeps: the children's layout ids.
pub struct AnchoredState {
    /// The children's layout ids.
    pub(crate) child_layout_ids: SmallVec<[crate::LayoutId; 4]>,
}

impl Element for Anchored {
    type RequestLayoutState = AnchoredState;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, AnchoredState) {
        let child_layout_ids: SmallVec<[LayoutId; 4]> =
            self.children.iter_mut().map(|child| child.request_layout(window, cx)).collect();
        let style = Style { position: Position::Absolute, display: Display::Flex, ..Style::default() };
        let layout_id = window.request_layout(style, child_layout_ids.iter().copied(), cx);
        (layout_id, AnchoredState { child_layout_ids })
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut AnchoredState,
        window: &mut Window,
        cx: &mut App,
    ) {
        if request_layout.child_layout_ids.is_empty() {
            return;
        }
        let mut child_min = point(Pixels::MAX, Pixels::MAX);
        let mut child_max = Point::default();
        for child_layout_id in &request_layout.child_layout_ids {
            let child_bounds = window.layout_bounds(*child_layout_id);
            child_min = child_min.min(&child_bounds.origin);
            child_max = child_max.max(&child_bounds.bottom_right());
        }
        let size: Size<Pixels> = (child_max - child_min).into();
        let limits = Bounds { origin: Point::default(), size: window.viewport_size() };
        let desired = fit(
            self.anchor_position.unwrap_or(bounds.origin),
            self.offset.unwrap_or_default(),
            self.anchor_corner,
            self.fit_mode,
            size,
            limits,
        );
        let offset = desired.origin - bounds.origin;
        let offset = point(offset.x.round(), offset.y.round());
        window.with_element_offset(offset, |window| {
            for child in &mut self.children {
                child.prepaint(window, cx);
            }
        })
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut AnchoredState,
        _prepaint: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        for child in &mut self.children {
            child.paint(window, cx);
        }
    }
}

impl IntoElement for Anchored {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// Where an anchored box of `size` lands (window px): GPUI's `AnchoredPositionMode::Window` placement,
/// then its corner switch (`SwitchAnchor`) or its snapping to the window's edges with a margin.
pub(crate) fn fit(
    anchor_position: Point<Pixels>,
    offset: Point<Pixels>,
    anchor_corner: Corner,
    fit_mode: AnchoredFitMode,
    size: Size<Pixels>,
    limits: Bounds<Pixels>,
) -> Bounds<Pixels> {
    let origin = anchor_position;
    let mut desired = Bounds::from_corner_and_size(anchor_corner, anchor_position + offset, size);

    if fit_mode == AnchoredFitMode::SwitchAnchor {
        let mut anchor_corner = anchor_corner;
        let mut cur_offset = offset;
        if desired.left() < limits.left() || desired.right() > limits.right() {
            let flipped_corner = anchor_corner.other_side_corner_along(Axis::Horizontal);
            let flipped_offset = point(-cur_offset.x, cur_offset.y);
            let switched = Bounds::from_corner_and_size(flipped_corner, origin + flipped_offset, size);
            if !(switched.left() < limits.left() || switched.right() > limits.right()) {
                anchor_corner = flipped_corner;
                cur_offset = flipped_offset;
                desired = switched;
            }
        }
        if desired.top() < limits.top() || desired.bottom() > limits.bottom() {
            let flipped_corner = anchor_corner.other_side_corner_along(Axis::Vertical);
            let flipped_offset = point(cur_offset.x, -cur_offset.y);
            let switched = Bounds::from_corner_and_size(flipped_corner, origin + flipped_offset, size);
            if !(switched.top() < limits.top() || switched.bottom() > limits.bottom()) {
                desired = switched;
            }
        }
    }

    // GPUI adds the client inset (Linux client-side decorations) here; our windows have none.
    let edges = match fit_mode {
        AnchoredFitMode::SnapToWindowWithMargin(edges) => edges,
        AnchoredFitMode::SwitchAnchor => Edges::default(),
    };
    if desired.right() > limits.right() - edges.right {
        desired.origin.x -= desired.right() - (limits.right() - edges.right);
    }
    if desired.left() < limits.left() + edges.left {
        desired.origin.x = limits.origin.x + edges.left;
    }
    if desired.bottom() > limits.bottom() - edges.bottom {
        desired.origin.y -= desired.bottom() - (limits.bottom() - edges.bottom);
    }
    if desired.top() < limits.top() + edges.top {
        desired.origin.y = limits.origin.y + edges.top;
    }
    desired
}
