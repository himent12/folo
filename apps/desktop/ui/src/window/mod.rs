//! The window: `Window` (state, frames, the platform window, the framebuffer and renderer). Owner tag:
//! `"window"`.
//!
//! Files: `handle.rs` (ids and handles), `focus.rs` (focus handles), `frame.rs` (what a paint produces),
//! `dispatch.rs` (the key dispatch tree), `layout.rs` (taffy), `draw.rs` (the frame pipeline and
//! presenting), `paint.rs` (the element-author paint API), `element_api.rs` (stacks, ids, element state,
//! hitboxes, listeners), `events.rs` (mouse and key dispatch), `ops.rs` (window operations and
//! queries), `input_handler.rs` (the IME seam).
//!
//! `App.windows` holds each `Window` in an `Option<Box<Window>>` slot; a window update leases it out, so
//! every callback gets `(&mut Window, &mut App)`.
#![allow(dead_code)] // skeleton

mod dispatch;
mod draw;
mod element_api;
mod events;
mod focus;
mod frame;
mod handle;
pub(crate) mod input_handler;
mod layout;
mod ops;
mod paint;

pub(crate) use dispatch::*;
pub use focus::*;
pub use frame::*;
pub use handle::*;
pub(crate) use layout::*;
pub use paint::*;

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::Arc,
};

use smallvec::SmallVec;

use crate::{
    AnyDrag, AnyView, App, ContentMask, DisplayId, Framebuffer, Modifiers, Pixels, PlatformWindow, Point,
    Size, TextStyleRefinement, WindowOptions, WindowTextSystem, element::ElementIdStack,
};
// skeleton: `crate::raster` once the raster package lands.
use crate::pending::raster_pending as raster;

/// The `on_window_should_close` handler.
pub(crate) type ShouldCloseHandler = Box<dyn FnMut(&mut Window, &mut App) -> bool>;

/// A callback for `on_next_frame`.
pub(crate) type FrameCallback = Box<dyn FnOnce(&mut Window, &mut App)>;

/// A window: its platform window, the root view, the two frames, the input state, the framebuffer.
pub struct Window {
    /// Its handle.
    pub(crate) handle: AnyWindowHandle,
    /// Set by `remove_window`; the app closes it after the current update.
    pub(crate) removed: bool,
    /// The OS (or headless) window.
    pub(crate) platform_window: Box<dyn PlatformWindow>,
    /// The display it opened on.
    pub(crate) display_id: Option<DisplayId>,
    /// Per-window text: the line layout cache that keeps what the previous frame used.
    pub(crate) text_system: Arc<WindowTextSystem>,
    /// The root view.
    pub(crate) root: Option<AnyView>,
    /// `rem_size` (16 px, as GPUI's default).
    pub(crate) rem_size: Pixels,
    /// The client area's size (logical px).
    pub(crate) viewport_size: Size<Pixels>,
    /// Device pixels per logical pixel.
    pub(crate) scale_factor: f32,
    /// The taffy tree for this frame.
    pub(crate) layout_engine: Option<TaffyLayoutEngine>,
    /// The element id path while walking the tree.
    pub(crate) element_id_stack: ElementIdStack,
    /// Text style refinements pushed by `with_text_style`.
    pub(crate) text_style_stack: Vec<TextStyleRefinement>,
    /// Element offsets pushed by scroll containers and `prepaint_at`.
    pub(crate) element_offset_stack: Vec<Point<Pixels>>,
    /// Content masks pushed by `with_content_mask` (each already intersected with its parent).
    pub(crate) content_mask_stack: Vec<ContentMask<Pixels>>,
    /// The opacity applied to colours painted now (`with_element_opacity`), if any.
    pub(crate) element_opacity: Option<f32>,
    /// The frame shown on screen (hit testing and key dispatch use it).
    pub(crate) rendered_frame: Frame,
    /// The frame being painted (swapped with `rendered_frame` after `draw`).
    pub(crate) next_frame: Frame,
    /// `on_next_frame` callbacks, run at the start of the next `draw`.
    pub(crate) next_frame_callbacks: Rc<RefCell<Vec<FrameCallback>>>,
    /// Something changed: draw at the next refresh. Shared with the platform callbacks.
    pub(crate) dirty: Rc<Cell<bool>>,
    /// `request_animation_frame` was called during this frame: dirty again after it.
    pub(crate) animation_frame_requested: bool,
    /// The mouse position (window logical px).
    pub(crate) mouse_position: Point<Pixels>,
    /// The hovered hitboxes for `mouse_position` against `rendered_frame`.
    pub(crate) mouse_hit_test: HitTest,
    /// Modifier keys held.
    pub(crate) modifiers: Modifiers,
    /// Whether the window is active (focused by the OS).
    pub(crate) active: Rc<Cell<bool>>,
    /// Whether the mouse is over the window.
    pub(crate) hovered: Rc<Cell<bool>>,
    /// The focused element.
    pub(crate) focus: Option<FocusId>,
    /// The drag in progress (a file drag carrying `ExternalPaths`).
    pub(crate) active_drag: Option<AnyDrag>,
    /// The tooltip shown now, if any.
    pub(crate) active_tooltip: Option<ActiveTooltip>,
    /// `on_window_should_close` handler.
    pub(crate) should_close_handler: Option<ShouldCloseHandler>,
    /// The pixels on screen (device px).
    pub(crate) framebuffer: Framebuffer,
    /// The tile renderer (keeps last frame's tile hashes).
    pub(crate) renderer: raster::Renderer,
    /// Glyph masks, reused across frames.
    pub(crate) glyph_masks: raster::MaskCache,
    /// SVG icon masks, reused across frames.
    pub(crate) icon_masks: raster::MaskCache,
    /// Damage rectangles of the last render (reused buffer).
    pub(crate) damage: Vec<crate::Bounds<crate::DevicePixels>>,
    /// Scratch for hit tests (reused).
    pub(crate) scratch_hitboxes: SmallVec<[HitboxId; 8]>,
}

/// A tooltip being shown: its element, where, and the hitbox that asked for it.
pub(crate) struct ActiveTooltip {
    /// Which request it came from.
    pub(crate) tooltip_id: TooltipId,
    /// The rendered tooltip view.
    pub(crate) view: AnyView,
    /// Where the mouse was when it appeared.
    pub(crate) mouse_position: Point<Pixels>,
}

impl Window {
    /// Opens the platform window, wires its callbacks to the app, and builds the window (no frame is drawn
    /// until the first `on_request_frame`).
    pub(crate) fn new(
        _handle: AnyWindowHandle,
        _options: WindowOptions,
        _cx: &mut App,
    ) -> anyhow::Result<Self> {
        todo!("window")
    }

    /// The window's handle.
    pub fn window_handle(&self) -> AnyWindowHandle {
        self.handle
    }

    /// Marks the window dirty and asks the platform for a frame (idempotent).
    pub fn refresh(&mut self) {
        if !self.dirty.replace(true) {
            self.platform_window.request_frame();
        }
    }

    /// Keeps drawing: the window is dirty again right after this frame (animations).
    pub fn request_animation_frame(&mut self) {
        self.animation_frame_requested = true;
    }

    /// Runs `callback` at the start of the next frame (and requests one).
    pub fn on_next_frame(&mut self, callback: impl FnOnce(&mut Window, &mut App) + 'static) {
        self.next_frame_callbacks.borrow_mut().push(Box::new(callback));
        self.refresh();
    }

    /// Registers the handler asked when the user closes the window; `false` keeps it open.
    pub fn on_window_should_close(
        &mut self,
        _cx: &App,
        f: impl FnMut(&mut Window, &mut App) -> bool + 'static,
    ) {
        self.should_close_handler = Some(Box::new(f));
    }
}
