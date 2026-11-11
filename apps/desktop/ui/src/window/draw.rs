//! The frame pipeline: `draw` (root render -> layout -> prepaint -> paint, then the tooltip and deferred
//! draws, then the frame swap) and `present` (render the scene into the framebuffer and present the damage).
//! Owner tag: `"window"`.
//!
//! Dirtiness: a dirty window calls `platform_window.request_frame()` once; the platform calls back
//! `on_request_frame` at the next refresh, and the callback draws only if the window is still dirty. No
//! frame, no timer and no wake-up while nothing is dirty. `request_animation_frame` makes the window dirty
//! again right after the frame it was called in.
#![allow(dead_code)] // skeleton

use crate::{App, Bounds, Pixels, RequestFrameOptions, Size, Window};

impl Window {
    /// The `on_request_frame` callback: draws and presents if dirty (or if `options.force_render`).
    pub(crate) fn on_request_frame(&mut self, _options: RequestFrameOptions, _cx: &mut App) {
        todo!("window")
    }

    /// Builds the next frame: runs `on_next_frame` callbacks, renders the root view, lays it out in the
    /// viewport, prepaints and paints it, then the active tooltip and the deferred draws by priority, then
    /// swaps `next_frame` and `rendered_frame`, recomputes hover from the new frame, hands the focused
    /// element's input handler to the platform and applies the cursor style.
    pub fn draw(&mut self, _cx: &mut App) {
        todo!("window")
    }

    /// Renders `rendered_frame.scene` into the framebuffer (`raster::Renderer::render`, with the text
    /// backend's `GlyphBlend`) and presents the damage, unless it is empty.
    pub(crate) fn present(&mut self) {
        todo!("window")
    }

    /// The platform resized the window: new viewport and scale, framebuffer resized, renderer invalidated,
    /// and a synchronous draw + present (no black edges while dragging).
    pub(crate) fn on_resize(&mut self, _content_size: Size<Pixels>, _scale_factor: f32, _cx: &mut App) {
        todo!("window")
    }

    /// Paints the tooltip of the hovered element, if one is showing, above the tree.
    pub(crate) fn prepaint_tooltip(&mut self, _cx: &mut App) -> Option<Bounds<Pixels>> {
        todo!("window")
    }

    /// Lays out and paints the deferred draws, lowest priority first.
    pub(crate) fn draw_deferred(&mut self, _cx: &mut App) {
        todo!("window")
    }
}
