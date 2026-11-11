//! `canvas(prepaint, paint)`: an element the app paints by hand (charts, scroll thumbs, measuring).
//! Owner tag: `"elements"`.

use std::panic;

use crate::{
    App, Bounds, Element, ElementId, GlobalElementId, InspectorElementId, IntoElement, LayoutId, Pixels, Refineable,
    Style, StyleRefinement, Styled, Window, elements::paint_style,
};

/// A canvas: `prepaint` runs with the laid-out bounds and returns a value `paint` gets.
pub fn canvas<T>(
    prepaint: impl 'static + FnOnce(Bounds<Pixels>, &mut Window, &mut App) -> T,
    paint: impl 'static + FnOnce(Bounds<Pixels>, T, &mut Window, &mut App),
) -> Canvas<T> {
    Canvas {
        prepaint: Some(Box::new(prepaint)),
        paint: Some(Box::new(paint)),
        style: StyleRefinement::default(),
    }
}

/// A canvas's prepaint closure.
pub(crate) type PrepaintFn<T> = Box<dyn FnOnce(Bounds<Pixels>, &mut Window, &mut App) -> T>;
/// A canvas's paint closure.
pub(crate) type PaintFn<T> = Box<dyn FnOnce(Bounds<Pixels>, T, &mut Window, &mut App)>;

/// The canvas element.
pub struct Canvas<T> {
    /// Called in prepaint.
    pub(crate) prepaint: Option<PrepaintFn<T>>,
    /// Called in paint.
    pub(crate) paint: Option<PaintFn<T>>,
    /// Its style (size and position).
    pub(crate) style: StyleRefinement,
}

impl<T> Styled for Canvas<T> {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.style
    }
}

impl<T: 'static> Element for Canvas<T> {
    type RequestLayoutState = Style;
    type PrepaintState = Option<T>;

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
    ) -> (LayoutId, Style) {
        let mut style = Style::default();
        style.refine(&self.style);
        let layout_id = window.request_layout(style.clone(), None, cx);
        (layout_id, style)
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _style: &mut Style,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<T> {
        Some((self.prepaint.take().expect("canvas prepaints once"))(bounds, window, cx))
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        style: &mut Style,
        prepaint: &mut Option<T>,
        window: &mut Window,
        cx: &mut App,
    ) {
        let prepaint = prepaint.take().expect("canvas paints once");
        let paint = self.paint.take().expect("canvas paints once");
        paint_style(style, bounds, window, cx, |window, cx| paint(bounds, prepaint, window, cx));
    }
}

impl<T: 'static> IntoElement for Canvas<T> {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}
