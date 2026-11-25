//! The window: `Window` (state, frames, the platform window, the framebuffer and renderer). Owner tag:
//! `"window"`.
//!
//! Files: `handle.rs` (ids and handles), `focus.rs` (focus handles), `frame.rs` (what a paint produces),
//! `dispatch.rs` (the key dispatch tree and tab stops), `layout.rs` (taffy), `draw.rs` (the frame pipeline
//! and presenting), `trace.rs` (`FOLO_UI_TRACE=frames`), `paint.rs` (`PaintQuad` helpers), `primitives.rs`
//! (the element-author paint API), `paint_animation.rs` (opacity groups and paint animations), `text_paint.rs` (`GlyphPainter`, line painting), `element_api.rs`
//! (stacks, ids, element state, hitboxes, listeners), `events.rs` (mouse and key dispatch), `ops.rs`
//! (window operations and queries), `input_handler.rs` (the IME seam).
//!
//! `App.windows` holds each `Window` in an `Option<Box<Window>>` slot; a window update leases it out, so
//! every callback gets `(&mut Window, &mut App)`. Platform callbacks hold only a weak reference to the app
//! and the window's handle, and lease the window for each call.

mod dispatch;
mod draw;
mod element_api;
mod events;
mod focus;
mod frame;
mod handle;
pub(crate) mod input_handler;
mod layout;
mod layout_tree;
mod ops;
mod paint;
mod paint_animation;
mod primitives;
mod text_paint;
mod trace;

#[cfg(test)]
mod tests;

pub(crate) use dispatch::*;
pub use focus::*;
pub use frame::*;
pub use handle::*;
pub(crate) use layout::*;
pub use paint::*;
pub(crate) use paint_animation::PaintAnimation;
pub use trace::FrameTimes;

use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
    sync::Arc,
};

use smallvec::SmallVec;

use crate::raster;
use crate::{
    AnyDrag, AnyView, App, AppCell, Bounds, DisplayId, Framebuffer, GlyphBlend, GroupHitboxes, Modifiers,
    Pixels, PlatformWindow, Point, RenderGlyphParams, RenderSvgParams, Size, SvgRenderer, TextStyle,
    WindowBounds, WindowOptions, WindowParams, WindowTextSystem, element::ElementIdStack, px,
};

/// The `on_window_should_close` handler.
pub(crate) type ShouldCloseHandler = Box<dyn FnMut(&mut Window, &mut App) -> bool>;
pub(crate) type AppearanceHandler = Box<dyn FnMut(&mut Window, &mut App)>;

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
    /// The text backend's glyph correction, passed to the renderer (asked once).
    pub(crate) glyph_blend: GlyphBlend,
    /// SVG icons to masks (shares the app's asset source).
    pub(crate) svg_renderer: SvgRenderer,
    /// The root view.
    pub(crate) root: Option<AnyView>,
    /// `rem_size` (16 px, as GPUI's default).
    pub(crate) rem_size: Pixels,
    /// The client area's size (logical px).
    pub(crate) viewport_size: Size<Pixels>,
    /// Device pixels per logical pixel.
    pub(crate) scale_factor: f32,
    /// The taffy tree for this frame (out of the window only during `compute_layout`).
    pub(crate) layout_engine: Option<TaffyLayoutEngine>,
    /// The element id path while walking the tree.
    pub(crate) element_id_stack: ElementIdStack,
    /// Resolved text styles pushed by `with_text_style` (each already refined on its parent).
    pub(crate) text_style_stack: SmallVec<[TextStyle; 8]>,
    /// Element offsets pushed by scroll containers and `prepaint_at`.
    pub(crate) element_offset_stack: Vec<Point<Pixels>>,
    /// Content masks pushed by `with_content_mask` (each already intersected with its parent).
    pub(crate) content_mask_stack: Vec<crate::ContentMask<Pixels>>,
    /// The opacity colours are painted at now (product of `with_element_opacity`; 1 at the root, and 1
    /// inside an opacity group, which fades its primitives after paint).
    pub(crate) element_opacity: f32,
    /// The opacity groups being painted, innermost last, each with where its current span starts.
    pub(crate) opacity_group_stack: Vec<(u32, usize)>,
    /// The frame shown on screen (hit testing and key dispatch use it).
    pub(crate) rendered_frame: Frame,
    /// The frame being painted (swapped with `rendered_frame` after `draw`).
    pub(crate) next_frame: Frame,
    /// `on_next_frame` callbacks, run at the start of the next frame request.
    pub(crate) next_frame_callbacks: Rc<RefCell<Vec<FrameCallback>>>,
    /// Something changed: draw at the next refresh.
    pub(crate) dirty: Rc<Cell<bool>>,
    /// A frame is being drawn (`refresh` is ignored meanwhile, as GPUI's).
    pub(crate) drawing: bool,
    /// The last drawn frame has not been presented yet.
    pub(crate) needs_present: bool,
    /// `request_animation_frame` was called during this frame: dirty again after it.
    pub(crate) animation_frame_requested: bool,
    /// The next hitbox id (grows across frames, as GPUI's).
    pub(crate) next_hitbox_id: usize,
    /// The next tooltip id.
    pub(crate) next_tooltip_id: usize,
    /// The tooltip shown this frame and where (for `TooltipId::is_hovered`).
    pub(crate) tooltip_bounds: Option<(TooltipId, Bounds<Pixels>)>,
    /// Hitboxes of the named groups being painted (`group_hover` / `group_active`).
    pub(crate) group_hitboxes: GroupHitboxes,
    /// The mouse position (window logical px).
    pub(crate) mouse_position: Point<Pixels>,
    /// The hovered hitboxes for `mouse_position` (against `rendered_frame`, or `next_frame` during paint).
    pub(crate) mouse_hit_test: HitTest,
    /// Modifier keys held.
    pub(crate) modifiers: Modifiers,
    /// Whether the window is active (focused by the OS).
    pub(crate) active: Rc<Cell<bool>>,
    /// How many times the window became active: counted, so a deactivate and reactivate between two
    /// frames is not lost to a reader that compares at its next frame.
    pub(crate) activations: u64,
    /// Whether the mouse is over the window.
    pub(crate) hovered: Rc<Cell<bool>>,
    /// The focused element.
    pub(crate) focus: Option<FocusId>,
    /// `prevent_default` was called for the event being dispatched.
    pub(crate) default_prevented: bool,
    /// The drag in progress (a file drag carrying `ExternalPaths`).
    pub(crate) active_drag: Option<AnyDrag>,
    /// `on_window_should_close` handler.
    pub(crate) should_close_handler: Option<ShouldCloseHandler>,
    /// `on_appearance_changed` handler.
    pub(crate) appearance_handler: Option<AppearanceHandler>,
    /// The focused element's text input handler, shared with the platform's `WindowInputHandler`.
    pub(crate) input_handler: Option<input_handler::SharedInputHandler>,
    /// The pixels on screen (device px).
    pub(crate) framebuffer: Framebuffer,
    /// The tile renderer (keeps last frame's tile hashes).
    pub(crate) renderer: raster::Renderer,
    /// Glyph and emoji masks (with their raster bounds), reused across frames.
    pub(crate) glyph_masks: raster::MaskCache<RenderGlyphParams, raster::GlyphMask>,
    /// SVG icon masks, reused across frames.
    pub(crate) icon_masks: raster::MaskCache<RenderSvgParams>,
    /// Damage rectangles of the last render.
    pub(crate) damage: Vec<crate::Bounds<crate::DevicePixels>>,
    /// Phase timings of the last frame (`FOLO_UI_TRACE=frames` prints them).
    pub(crate) frame_times: FrameTimes,
    /// Scratch for the deferred draw order (reused).
    pub(crate) scratch_order: SmallVec<[usize; 8]>,
}

impl Window {
    /// Opens the platform window, wires its callbacks to the app, and builds the window. No frame is drawn
    /// here: the window is dirty and has asked the platform for its first frame.
    pub(crate) fn new(handle: AnyWindowHandle, options: WindowOptions, cx: &mut App) -> anyhow::Result<Self> {
        let WindowOptions {
            window_bounds,
            titlebar,
            focus,
            show,
            kind,
            is_movable,
            is_resizable,
            is_minimizable,
            display_id,
            app_id,
            window_min_size,
            ..
        } = options;
        let bounds = window_bounds.map(|bounds| bounds.get_bounds()).unwrap_or_else(|| {
            display_id
                .and_then(|id| cx.displays().into_iter().find(|display| display.id() == id))
                .or_else(|| cx.primary_display())
                .map(|display| display.default_bounds())
                .unwrap_or_else(|| Bounds::new(Point::default(), crate::DEFAULT_WINDOW_SIZE))
        });
        let platform_window = cx.platform.open_window(WindowParams {
            bounds,
            titlebar,
            kind,
            is_movable,
            is_resizable,
            is_minimizable,
            focus,
            show,
            display_id,
            window_min_size,
            app_id,
        })?;
        match window_bounds {
            Some(WindowBounds::Fullscreen(_)) => platform_window.toggle_fullscreen(),
            Some(WindowBounds::Maximized(_)) => platform_window.zoom(),
            _ => {}
        }
        wire_callbacks(platform_window.as_ref(), cx.this.clone(), handle);

        let content_size = platform_window.content_size();
        let scale_factor = platform_window.scale_factor();
        let mut frames = [Frame::new(), Frame::new()];
        for frame in &mut frames {
            frame.dispatch_tree.keymap = Some(cx.keymap.clone());
        }
        let [rendered_frame, next_frame] = frames;
        let mut window = Window {
            handle,
            removed: false,
            display_id: platform_window.display().map(|display| display.id()),
            text_system: Arc::new(WindowTextSystem::new(cx.text_system().clone())),
            glyph_blend: cx.text_system().glyph_blend(),
            svg_renderer: SvgRenderer::new(cx.asset_source().clone()),
            root: None,
            rem_size: px(16.),
            viewport_size: content_size,
            scale_factor,
            layout_engine: Some(TaffyLayoutEngine::new()),
            element_id_stack: SmallVec::new(),
            text_style_stack: SmallVec::new(),
            element_offset_stack: Vec::new(),
            content_mask_stack: Vec::new(),
            element_opacity: 1.0,
            opacity_group_stack: Vec::new(),
            rendered_frame,
            next_frame,
            next_frame_callbacks: Rc::default(),
            dirty: Rc::new(Cell::new(false)),
            drawing: false,
            needs_present: false,
            animation_frame_requested: false,
            next_hitbox_id: 0,
            next_tooltip_id: 0,
            tooltip_bounds: None,
            group_hitboxes: GroupHitboxes::default(),
            mouse_position: platform_window.mouse_position(),
            mouse_hit_test: HitTest::default(),
            modifiers: platform_window.modifiers(),
            active: Rc::new(Cell::new(platform_window.is_active())),
            activations: 0,
            hovered: Rc::new(Cell::new(platform_window.is_hovered())),
            focus: None,
            default_prevented: false,
            active_drag: None,
            should_close_handler: None,
            appearance_handler: None,
            input_handler: None,
            framebuffer: Framebuffer::new(0, 0),
            renderer: raster::Renderer::new(),
            glyph_masks: raster::MaskCache::new(),
            icon_masks: raster::MaskCache::new(),
            damage: Vec::new(),
            frame_times: FrameTimes::default(),
            scratch_order: SmallVec::new(),
            platform_window,
        };
        window.resize_framebuffer();
        window.refresh();
        Ok(window)
    }

    /// The window's handle.
    pub fn window_handle(&self) -> AnyWindowHandle {
        self.handle
    }

    /// Marks the window dirty and asks the platform for one frame (idempotent; ignored while drawing,
    /// as GPUI's).
    pub fn refresh(&mut self) {
        if self.drawing {
            return;
        }
        if !self.dirty.replace(true) {
            self.platform_window.request_frame();
        }
    }

    /// Keeps drawing: the window is dirty again right after this frame (animations).
    pub fn request_animation_frame(&mut self) {
        self.animation_frame_requested = true;
        if !self.drawing {
            self.refresh();
        }
    }

    /// Runs `callback` at the start of the next frame (and requests one).
    pub fn on_next_frame(&mut self, callback: impl FnOnce(&mut Window, &mut App) + 'static) {
        self.next_frame_callbacks.borrow_mut().push(Box::new(callback));
        self.platform_window.request_frame();
    }

    /// Registers the handler asked when the user closes the window; `false` keeps it open.
    pub fn on_window_should_close(
        &mut self,
        _cx: &App,
        f: impl FnMut(&mut Window, &mut App) -> bool + 'static,
    ) {
        self.should_close_handler = Some(Box::new(f));
    }

    /// The system's light / dark appearance, as this window sees it (the headless platform: dark
    /// until a test changes it with `HeadlessApp::set_system_appearance`).
    pub fn appearance(&self) -> crate::WindowAppearance {
        self.platform_window.appearance()
    }

    /// Registers the handler called after the system's appearance changes (Windows:
    /// `WM_SETTINGCHANGE` "ImmersiveColorSet"; winit: its theme-changed event; headless:
    /// `HeadlessApp::set_system_appearance`). The window is redrawn either way.
    pub fn on_appearance_changed(&mut self, _cx: &App, f: impl FnMut(&mut Window, &mut App) + 'static) {
        self.appearance_handler = Some(Box::new(f));
    }

    /// Draws the window's system frame (Windows: DWM's caption and border) dark or light whatever the
    /// system's appearance, or following it again (`None`).
    pub fn set_frame_appearance(&self, appearance: Option<crate::WindowAppearance>) {
        self.platform_window.set_frame_appearance(appearance);
    }

    /// Phase timings of the last drawn frame.
    pub fn last_frame_times(&self) -> FrameTimes {
        self.frame_times
    }

    /// The display list of the frame on screen.
    pub fn rendered_scene(&self) -> &crate::Scene {
        &self.rendered_frame.scene
    }

    /// The framebuffer the last frame was rendered into.
    pub fn framebuffer(&self) -> &Framebuffer {
        &self.framebuffer
    }

    /// Sizes the framebuffer to the viewport in device pixels.
    pub(crate) fn resize_framebuffer(&mut self) {
        let width = (self.viewport_size.width.0 * self.scale_factor).round().max(0.) as u32;
        let height = (self.viewport_size.height.0 * self.scale_factor).round().max(0.) as u32;
        if self.framebuffer.width != width || self.framebuffer.height != height {
            self.framebuffer.resize(width, height);
            self.renderer.invalidate();
        }
    }
}

/// Leases window `handle` out of the app for `f`, inside an app update (effects flush after it). `None`
/// when the app or the window is gone.
pub(crate) fn update_window<R>(
    app: &Weak<AppCell>,
    handle: AnyWindowHandle,
    f: impl FnOnce(&mut Window, &mut App) -> R,
) -> Option<R> {
    let app = app.upgrade()?;
    let mut cx = app.borrow_mut();
    cx.update(|cx| cx.update_window_id(handle, |_, window, cx| f(window, cx)).ok())
}

/// Registers the window's platform callbacks: each leases the window for its call.
fn wire_callbacks(platform_window: &dyn PlatformWindow, app: Weak<AppCell>, handle: AnyWindowHandle) {
    platform_window.on_request_frame(Box::new({
        let app = app.clone();
        move |options| {
            update_window(&app, handle, |window, cx| window.on_request_frame(options, cx));
        }
    }));
    platform_window.on_input(Box::new({
        let app = app.clone();
        move |event| {
            update_window(&app, handle, |window, cx| window.dispatch_event(event, cx)).unwrap_or_default()
        }
    }));
    platform_window.on_resize(Box::new({
        let app = app.clone();
        move |size, scale_factor| {
            update_window(&app, handle, |window, cx| window.on_resize(size, scale_factor, cx));
        }
    }));
    platform_window.on_moved(Box::new({
        let app = app.clone();
        move || {
            update_window(&app, handle, |window, cx| window.on_moved(cx));
        }
    }));
    platform_window.on_active_status_change(Box::new({
        let app = app.clone();
        move |active| {
            update_window(&app, handle, |window, _| {
                if active && !window.active.replace(active) {
                    window.activations += 1;
                }
                window.active.set(active);
                window.modifiers = window.platform_window.modifiers();
                window.refresh();
            });
        }
    }));
    platform_window.on_hover_status_change(Box::new({
        let app = app.clone();
        move |hovered| {
            update_window(&app, handle, |window, cx| window.on_hover_status_change(hovered, cx));
        }
    }));
    platform_window.on_should_close(Box::new({
        let app = app.clone();
        move || {
            update_window(&app, handle, |window, cx| match window.should_close_handler.take() {
                Some(mut handler) => {
                    let close = handler(window, cx);
                    window.should_close_handler.get_or_insert(handler);
                    close
                }
                None => true,
            })
            .unwrap_or(true)
        }
    }));
    platform_window.on_hit_test_window_control(Box::new({
        let app = app.clone();
        move || update_window(&app, handle, |window, _| window.hit_test_window_control()).flatten()
    }));
    platform_window.on_appearance_changed(Box::new({
        let app = app.clone();
        move || {
            update_window(&app, handle, |window, cx| {
                window.refresh();
                if let Some(mut handler) = window.appearance_handler.take() {
                    handler(window, cx);
                    window.appearance_handler.get_or_insert(handler);
                }
            });
        }
    }));
    platform_window.on_close(Box::new(move || {
        update_window(&app, handle, |window, _| window.remove_window());
    }));
}
