//! `svg()`: a single-colour SVG icon from the asset source, tinted with the text colour. Owner tag:
//! `"elements"`. The mask comes from the window's icon `MaskCache` (keyed by path and device size), so an
//! icon is rasterized once, not per frame.
#![allow(dead_code)] // skeleton

use crate::{
    InteractiveElement, Interactivity, SharedString, StyleRefinement, Styled, elements::skeleton_element,
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
    /// Sets the asset path (`"icons/logo-frame.svg"`).
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

skeleton_element!([] Svg, (), Option<crate::Hitbox>, "elements");
