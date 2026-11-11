//! `canvas(prepaint, paint)`: an element the app paints by hand (charts, scroll thumbs, measuring).
//! Owner tag: `"elements"`.
#![allow(dead_code)] // skeleton

use crate::{App, Bounds, Pixels, StyleRefinement, Styled, Window, elements::skeleton_element};

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

skeleton_element!([T: 'static] Canvas<T>, crate::Style, Option<T>, "elements");
