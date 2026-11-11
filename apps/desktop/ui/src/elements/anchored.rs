//! `anchored()`: positions its children at a point (or its own layout position), anchored by a corner, and
//! keeps them inside the window (`snap_to_window_with_margin`). Owner tag: `"elements"`.
#![allow(dead_code)] // skeleton

use smallvec::SmallVec;

use crate::{AnyElement, Corner, Edges, ParentElement, Pixels, Point, elements::skeleton_element};

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

skeleton_element!([] Anchored, AnchoredState, (), "elements");
