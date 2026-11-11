//! `svg()`: a single-colour SVG icon from the asset source, tinted with the text colour. Owner tag:
//! `"elements"`. The mask comes from the window's icon `MaskCache` (keyed by path and device size), so an
//! icon is rasterized once, not per frame.

use std::panic;

use crate::{
    App, Bounds, Element, ElementId, GlobalElementId, Hitbox, InspectorElementId, InteractiveElement,
    Interactivity, IntoElement, LayoutId, Pixels, SharedString, StyleRefinement, Styled, Window,
};

/// A new SVG element (give it `.path(..)` and a size).
pub fn svg() -> Svg {
    Svg { interactivity: Interactivity::default(), path: None }
}

/// An SVG icon.
pub struct Svg {
    /// Styles and listeners.
    pub(crate) interactivity: Interactivity,
    /// The asset path.
    pub(crate) path: Option<SharedString>,
}

impl Svg {
    /// Sets the asset path (`"icons/logo-mark.svg"`).
    pub fn path(mut self, path: impl Into<SharedString>) -> Self {
        self.path = Some(path.into());
        self
    }
}

impl Styled for Svg {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.interactivity.base_style
    }
}

impl InteractiveElement for Svg {
    fn interactivity(&mut self) -> &mut Interactivity {
        &mut self.interactivity
    }
}

impl Element for Svg {
    type RequestLayoutState = ();
    type PrepaintState = Option<Hitbox>;

    fn id(&self) -> Option<ElementId> {
        self.interactivity.element_id.clone()
    }

    fn source_location(&self) -> Option<&'static panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        let layout_id =
            self.interactivity.request_layout(global_id, inspector_id, window, cx, |style, window, cx| {
                window.request_layout(style, None, cx)
            });
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Hitbox> {
        self.interactivity.prepaint(
            global_id,
            inspector_id,
            bounds,
            bounds.size,
            window,
            cx,
            |_, _, hitbox, _, _| hitbox,
        )
    }

    /// The icon is a mask tinted with the text colour; with no text colour set, nothing is drawn (GPUI).
    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _request_layout: &mut (),
        hitbox: &mut Option<Hitbox>,
        window: &mut Window,
        cx: &mut App,
    ) {
        let path = self.path.clone();
        self.interactivity.paint(
            global_id,
            inspector_id,
            bounds,
            hitbox.as_ref(),
            window,
            cx,
            |style, window, cx| {
                if let Some((path, color)) = path.zip(style.text.color)
                    && let Err(error) = window.paint_svg(bounds, path.clone(), color, cx)
                {
                    eprintln!("folo-ui: svg {path}: {error:#}");
                }
            },
        )
    }
}

impl IntoElement for Svg {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}
