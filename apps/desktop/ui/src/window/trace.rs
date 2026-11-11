//! Per-frame phase timings: what `Window::draw` and `present` spent, and the `FOLO_UI_TRACE=frames` switch
//! that prints one line per frame to stderr. Owner tag: `"window"`.

use std::{sync::OnceLock, time::Instant};

/// Microseconds spent in each phase of one frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameTimes {
    /// The frame's number in this window (1 for the first).
    pub frame: u64,
    /// Rendering the views and building the element and layout trees (`render` + `request_layout`).
    pub build_us: u32,
    /// The views' `render` calls alone (part of `build_us`).
    pub render_us: u32,
    /// Computing the layout (taffy, measure functions).
    pub layout_us: u32,
    /// Prepaint: bounds, hitboxes, dispatch nodes, scroll offsets, deferred draws and the tooltip.
    pub prepaint_us: u32,
    /// Paint: the scene, listeners, cursor styles, key contexts.
    pub paint_us: u32,
    /// The frame swap and end-of-frame bookkeeping (caches, input handler, cursor).
    pub finish_us: u32,
    /// Rasterizing the scene into the framebuffer (`Renderer::render`).
    pub raster_us: u32,
    /// Handing the damage to the platform (`PlatformWindow::present`).
    pub present_us: u32,
    /// Primitives in the scene.
    pub primitives: u32,
    /// Hitboxes in the frame.
    pub hitboxes: u32,
    /// Elements laid out this frame (every `AnyElement` that ran `request_layout`).
    pub elements: u32,
}

impl FrameTimes {
    /// The whole frame, in microseconds.
    pub fn total_us(&self) -> u32 {
        self.build_us
            + self.layout_us
            + self.prepaint_us
            + self.paint_us
            + self.finish_us
            + self.raster_us
            + self.present_us
    }
}

/// Whether `FOLO_UI_TRACE` asks for frame lines (`frames`, or a comma list containing it). Read once.
pub(crate) fn trace_frames() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("FOLO_UI_TRACE").is_ok_and(|value| value.split(',').any(|part| part.trim() == "frames"))
    })
}

/// Microseconds since `start`, saturating.
pub(crate) fn micros_since(start: Instant) -> u32 {
    start.elapsed().as_micros().min(u32::MAX as u128) as u32
}

/// Prints `times` as one line to stderr.
pub(crate) fn print_frame(times: &FrameTimes) {
    eprintln!(
        "folo-ui frame {}: build {} (render {}) layout {} prepaint {} paint {} finish {} raster {} present {} = {} us ({} prims, {} elements, {} hitboxes)",
        times.frame,
        times.build_us,
        times.render_us,
        times.layout_us,
        times.prepaint_us,
        times.paint_us,
        times.finish_us,
        times.raster_us,
        times.present_us,
        times.total_us(),
        times.primitives,
        times.elements,
        times.hitboxes,
    );
}
