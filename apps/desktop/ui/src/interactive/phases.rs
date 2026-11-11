//! `Interactivity`'s three phases, which `Div` (and `Svg`) call around their children: computing the style
//! (base + focus + hover + group + active refinements), inserting the hitbox, registering every listener for
//! this frame, click detection, scroll containers and tooltips, as GPUI 0.2.2's `div.rs`. Owner tag:
//! `"window"`.
#![allow(dead_code)] // skeleton

use crate::{
    App, Bounds, GlobalElementId, Hitbox, InspectorElementId, Interactivity, LayoutId, Pixels, Point, Style,
    Window,
};

impl Interactivity {
    /// Computes the style for this frame and requests the layout (the closure adds children).
    pub fn request_layout(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _window: &mut Window,
        _cx: &mut App,
        _f: impl FnOnce(Style, &mut Window, &mut App) -> LayoutId,
    ) -> LayoutId {
        todo!("window")
    }

    /// Inserts the hitbox, applies the scroll offset, and prepaints children through `f` (which gets the
    /// style, the scroll offset and the hitbox).
    #[allow(clippy::too_many_arguments)]
    pub fn prepaint<R>(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _content_size: crate::Size<Pixels>,
        _window: &mut Window,
        _cx: &mut App,
        _f: impl FnOnce(&Style, Point<Pixels>, Option<Hitbox>, &mut Window, &mut App) -> R,
    ) -> R {
        todo!("window")
    }

    /// Paints background, borders and shadows, registers listeners, key context, focus and cursor, and
    /// paints children through `f`.
    #[allow(clippy::too_many_arguments)]
    pub fn paint(
        &mut self,
        _global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _hitbox: Option<&Hitbox>,
        _window: &mut Window,
        _cx: &mut App,
        _f: impl FnOnce(&Style, &mut Window, &mut App),
    ) {
        todo!("window")
    }

    /// The style for this frame: the base refined by focus, hover, group hover, active and group active.
    pub fn compute_style(
        &self,
        _global_id: Option<&GlobalElementId>,
        _hitbox: Option<&Hitbox>,
        _window: &mut Window,
        _cx: &mut App,
    ) -> Style {
        todo!("window")
    }
}

/// The hitboxes of named groups this frame (for `group_hover` / `group_active`), kept in a global.
#[derive(Default)]
pub(crate) struct GroupHitboxes(
    pub(crate) std::collections::HashMap<crate::SharedString, smallvec::SmallVec<[crate::HitboxId; 1]>>,
);

impl crate::Global for GroupHitboxes {}
