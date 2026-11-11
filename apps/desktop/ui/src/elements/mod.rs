//! The built-in elements. Owner tag: `"elements"`, except `div.rs` (`"window"`, with `interactive/`).
//!
//! Files: `div.rs` (`Div`, `Stateful`), `text.rs` (strings as elements, `StyledText`), `svg.rs` (`Svg`),
//! `canvas.rs`, `uniform_list.rs` (+ `UniformListScrollHandle`, `ScrollStrategy`), `anchored.rs`,
//! `deferred.rs`, `animation.rs` (`Animation`, `AnimationExt`), `input_handler.rs` (`InputHandler`,
//! `EntityInputHandler`, `ElementInputHandler`).

mod anchored;
mod animation;
mod canvas;
mod deferred;
mod div;
mod input_handler;
mod svg;
#[cfg(test)]
mod tests;
mod text;
mod uniform_list;

pub use anchored::*;
pub use animation::*;
pub use canvas::*;
pub use deferred::*;
pub use div::*;
pub use input_handler::*;
pub use svg::*;
pub use text::*;
pub use uniform_list::*;

use crate::{App, Bounds, Pixels, Style, Window};

/// GPUI's `Style::paint`: the shadows and background, then `children`, then the border on top. The colours
/// take the current element opacity and the current content mask, as GPUI's `Window::paint_quad` does.
pub(crate) fn paint_style(
    style: &Style,
    bounds: Bounds<Pixels>,
    window: &mut Window,
    cx: &mut App,
    children: impl FnOnce(&mut Window, &mut App),
) {
    let paint = window.style_paint();
    style.paint_before_children(bounds, &paint, &mut window.next_frame.scene);
    children(window, cx);
    style.paint_after_children(bounds, &paint, &mut window.next_frame.scene);
}
