//! The frame pipeline, in GPUI 0.2.2's order (`Window::draw` / `draw_roots`): render and lay out the root
//! at the viewport size, prepaint it, prepaint the deferred draws by priority, prepaint the tooltip, hit
//! test the mouse against the new frame, paint the root, the deferred draws and the tooltip, hand the
//! focused input handler to the platform, swap the frames; then `present` rasterizes the scene and
//! presents the damage. Owner tag: `"window"`.
//!
//! Dirtiness: a dirty window calls `platform_window.request_frame()` once; the platform calls back
//! `on_request_frame` at the next refresh, and the callback draws only if the window is still dirty. No
//! frame, no timer and no wake-up while nothing is dirty. `request_animation_frame` makes the window dirty
//! again right after the frame it was called in. A paint animation (`paint_animation.rs`) asks for frames
//! without making the window dirty: those frames only re-fade its opacity group and present the damage.
//!
//! Cost: each phase is timed (`FrameTimes`, printed with `FOLO_UI_TRACE=frames`). The pipeline reuses its
//! buffers: both frames' lists, the taffy tree, the dispatch tree and the deferred-draw order keep their
//! capacity, so a steady frame allocates only what elements themselves build.

use std::time::Instant;

use crate::{
    AnyElement, App, AvailableSpace, Bounds, IntoElement, Pixels, Point, RequestFrameOptions, Size, Window,
    point, px,
    window::{
        FrameTimes,
        input_handler::{SharedInputHandler, WindowInputHandler},
        trace::{micros_since, print_frame, trace_frames},
    },
};

impl Window {
    /// The `on_request_frame` callback: runs `on_next_frame` callbacks, then draws and presents if dirty
    /// (`force_render` also repaints every tile), or only presents a frame drawn earlier and not yet shown.
    /// With `require_presentation` the platform gets a `present` even when nothing changed (it lost what it
    /// showed: expose, resize, DPI change, capture), and copies the whole frame.
    pub(crate) fn on_request_frame(&mut self, options: RequestFrameOptions, cx: &mut App) {
        let callbacks = std::mem::take(&mut *self.next_frame_callbacks.borrow_mut());
        for callback in callbacks {
            callback(self, cx);
        }
        if options.force_render {
            self.renderer.invalidate();
        }
        let mut presented = false;
        if self.dirty.get() || options.force_render {
            self.draw(cx);
            presented = self.present();
            if trace_frames() {
                print_frame(&self.frame_times);
            }
        } else if self.tick_paint_animations(cx) {
            // Only paint animations moved: the scene's group table changed; no render, layout or paint.
            let frame = self.frame_times.frame + 1;
            self.frame_times = FrameTimes { frame, ..FrameTimes::default() };
            presented = self.present();
            if trace_frames() {
                print_frame(&self.frame_times);
            }
        } else if self.needs_present {
            presented = self.present();
        }
        if options.require_presentation && !presented {
            self.present_all();
        }
        self.platform_window.completed_frame();
    }

    /// Builds the next frame and makes it the rendered one (see the module docs for the order).
    pub fn draw(&mut self, cx: &mut App) {
        let start = Instant::now();
        self.dirty.set(false);
        self.drawing = true;
        self.frame_styles.clear();
        self.tooltip_bounds = None;
        self.frame_times.frame += 1;
        self.frame_times.render_us = 0;
        self.frame_times.elements = 0;

        // The platform's handler is the previous frame's: dropped; this frame's replaces it below.
        self.platform_window.take_input_handler();
        self.input_handler = None;

        let Some(root) = self.root.clone() else {
            self.drawing = false;
            return;
        };
        // A notify while this frame is built does not dirty the windows again (`App::drawing_window`).
        let outer_drawing = cx.drawing_window.replace(self.handle.id);
        let mut root_element = root.into_any_element();
        root_element.request_layout(self, cx);
        let built = Instant::now();
        self.frame_times.build_us = micros_since(start);

        let viewport: Size<AvailableSpace> = self.viewport_size.into();
        root_element.layout_as_root(viewport, self, cx);
        let laid_out = Instant::now();
        self.frame_times.layout_us = micros_since(built);

        root_element.prepaint_at(Point::default(), self, cx);
        self.sort_deferred_draws();
        self.prepaint_deferred_draws(cx);
        let mut tooltip_element = if self.active_drag.is_none() { self.prepaint_tooltip(cx) } else { None };
        self.mouse_hit_test = self.hit_test_mouse(&self.next_frame);
        let prepainted = Instant::now();
        self.frame_times.prepaint_us = micros_since(laid_out);

        root_element.paint(self, cx);
        self.paint_deferred_draws(cx);
        if let Some(tooltip) = tooltip_element.as_mut() {
            tooltip.paint(self, cx);
        }
        let painted = Instant::now();
        self.frame_times.paint_us = micros_since(prepainted);

        self.finish_frame(cx);
        cx.drawing_window = outer_drawing;
        drop(root_element);
        drop(tooltip_element);
        self.frame_times.finish_us = micros_since(painted);
    }

    /// End of a draw: input handler to the platform, caches told the frame ended, the frames swapped,
    /// the cursor applied, animation frames requested.
    fn finish_frame(&mut self, cx: &mut App) {
        if let Some(Some(slot)) = self.next_frame.input_handlers.pop()
            && Some(slot.focus_id) == self.focus
        {
            let handler = SharedInputHandler::new(slot.handler);
            self.input_handler = Some(handler.clone());
            self.platform_window.set_input_handler(Box::new(WindowInputHandler {
                app: cx.this.clone(),
                window: self.handle,
                handler,
            }));
        }
        if let Some(engine) = self.layout_engine.as_mut() {
            engine.clear();
        }
        self.text_system.finish_frame();
        self.glyph_masks.end_frame();
        self.icon_masks.end_frame();
        self.next_frame.scene.seal_opacity_groups();

        std::mem::swap(&mut self.rendered_frame, &mut self.next_frame);
        self.next_frame.clear();
        self.frame_styles.clear();
        self.frame_times.primitives = self.rendered_frame.scene.len() as u32;
        self.frame_times.hitboxes = self.rendered_frame.hitboxes.len() as u32;

        self.reset_cursor_style(cx);
        self.needs_present = true;
        self.drawing = false;
        if std::mem::take(&mut self.animation_frame_requested) {
            self.refresh();
        } else if self.has_running_paint_animations() {
            self.platform_window.request_frame();
        }
    }

    /// The hit test of the mouse against `frame`: nothing while the mouse is outside the window.
    pub(crate) fn hit_test_mouse(&self, frame: &crate::window::Frame) -> crate::window::HitTest {
        if self.hovered.get() { frame.hit_test(self.mouse_position) } else { Default::default() }
    }

    /// Sets the platform cursor from the rendered frame, when the mouse is over the window (GPUI).
    pub(crate) fn reset_cursor_style(&self, cx: &App) {
        if self.hovered.get() {
            let style = self.rendered_frame.cursor_style(self).unwrap_or_default();
            cx.platform.set_cursor_style(style);
        }
    }

    /// Renders `rendered_frame.scene` into the framebuffer (`Renderer::render`, with the text backend's
    /// `GlyphBlend`) and presents the damage, unless it is empty. Returns whether it presented.
    pub(crate) fn present(&mut self) -> bool {
        let start = Instant::now();
        self.damage =
            self.renderer.render(&self.rendered_frame.scene, self.glyph_blend, &mut self.framebuffer);
        let rastered = Instant::now();
        self.frame_times.raster_us = micros_since(start);
        if !self.damage.is_empty() {
            self.platform_window.present(&self.framebuffer, &self.damage);
            crate::app::trace::frame_presented();
        }
        self.frame_times.present_us = micros_since(rastered);
        self.needs_present = false;
        !self.damage.is_empty()
    }

    /// Presents the whole framebuffer again (the platform lost what it showed).
    fn present_all(&mut self) {
        let full = Bounds::new(
            Point::default(),
            crate::size(
                crate::DevicePixels(self.framebuffer.width as i32),
                crate::DevicePixels(self.framebuffer.height as i32),
            ),
        );
        self.platform_window.present(&self.framebuffer, &[full]);
    }

    /// The platform resized the window or changed its scale: new viewport and scale, framebuffer resized,
    /// renderer invalidated, window dirty. Both backends follow this callback at once (inside `WM_SIZE` /
    /// `Resized`) with a `require_presentation` frame request, which draws and presents synchronously, so a
    /// live resize never shows an unpainted edge; drawing here as well would draw every size twice.
    pub(crate) fn on_resize(&mut self, content_size: Size<Pixels>, scale_factor: f32, _cx: &mut App) {
        self.viewport_size = content_size;
        self.scale_factor = scale_factor;
        self.display_id = self.platform_window.display().map(|display| display.id());
        self.resize_framebuffer();
        self.renderer.invalidate();
        self.refresh();
    }

    /// The window moved: the display (and its scale) may have changed.
    pub(crate) fn on_moved(&mut self, cx: &mut App) {
        self.display_id = self.platform_window.display().map(|display| display.id());
        let scale_factor = self.platform_window.scale_factor();
        let size = self.platform_window.content_size();
        if scale_factor != self.scale_factor || size != self.viewport_size {
            self.on_resize(size, scale_factor, cx);
        }
    }

    /// Lays out and prepaints the tooltip of the hovered element, placed next to the mouse and kept inside
    /// the window (GPUI's `prepaint_tooltip`). The topmost request whose element still wants it wins.
    pub(crate) fn prepaint_tooltip(&mut self, cx: &mut App) -> Option<AnyElement> {
        for ix in (0..self.next_frame.tooltip_requests.len()).rev() {
            let Some(Some(request)) = self.next_frame.tooltip_requests.get(ix).cloned() else { continue };
            let mut element = request.tooltip.view.clone().into_any_element();
            let mouse_position = request.tooltip.mouse_position;
            let tooltip_size = element.layout_as_root(AvailableSpace::min_size(), self, cx);
            let mut bounds = Bounds::new(mouse_position + point(px(1.), px(1.)), tooltip_size);
            let window_bounds = Bounds { origin: Point::default(), size: self.viewport_size };
            if bounds.right() > window_bounds.right() {
                let new_x = mouse_position.x - bounds.size.width - px(1.);
                bounds.origin.x = if new_x >= Pixels::ZERO {
                    new_x
                } else {
                    Pixels::ZERO.max(window_bounds.right() - bounds.size.width)
                };
            }
            if bounds.bottom() > window_bounds.bottom() {
                let new_y = mouse_position.y - bounds.size.height - px(1.);
                bounds.origin.y = if new_y >= Pixels::ZERO {
                    new_y
                } else {
                    Pixels::ZERO.max(window_bounds.bottom() - bounds.size.height)
                };
            }
            if !(request.tooltip.check_visible_and_update)(bounds, self, cx) {
                continue;
            }
            self.with_absolute_element_offset(bounds.origin, |window| element.prepaint(window, cx));
            self.tooltip_bounds = Some((request.id, bounds));
            return Some(element);
        }
        None
    }

    /// The deferred draws' order: by priority, ties in insertion order (a stable sort, as GPUI's).
    fn sort_deferred_draws(&mut self) {
        let mut order = std::mem::take(&mut self.scratch_order);
        order.clear();
        order.extend(0..self.next_frame.deferred_draws.len());
        order.sort_by_key(|ix| self.next_frame.deferred_draws[*ix].priority);
        self.scratch_order = order;
    }

    /// Prepaints the deferred draws, lowest priority first, each with the id path, text style and
    /// dispatch node it was deferred under.
    fn prepaint_deferred_draws(&mut self, cx: &mut App) {
        let mut draws = std::mem::take(&mut self.next_frame.deferred_draws);
        let order = std::mem::take(&mut self.scratch_order);
        for &ix in &order {
            let draw = &mut draws[ix];
            self.element_id_stack.clone_from(&draw.element_id_stack);
            self.text_style_stack.clear();
            if let Some(style) = draw.text_style.clone() {
                self.text_style_stack.push(style);
            }
            if draw.parent_node.0 < self.next_frame.dispatch_tree.len() {
                self.next_frame.dispatch_tree.set_active_node(draw.parent_node);
            }
            if let Some(element) = draw.element.as_mut() {
                self.with_absolute_element_offset(draw.absolute_offset, |window| {
                    element.prepaint(window, cx)
                });
            }
        }
        debug_assert!(self.next_frame.deferred_draws.is_empty(), "defer_draw during deferred drawing");
        self.next_frame.deferred_draws = draws;
        self.scratch_order = order;
        self.element_id_stack.clear();
        self.text_style_stack.clear();
    }

    /// Paints the deferred draws in the same order.
    fn paint_deferred_draws(&mut self, cx: &mut App) {
        let mut draws = std::mem::take(&mut self.next_frame.deferred_draws);
        let order = std::mem::take(&mut self.scratch_order);
        for &ix in &order {
            let draw = &mut draws[ix];
            self.element_id_stack.clone_from(&draw.element_id_stack);
            if draw.parent_node.0 < self.next_frame.dispatch_tree.len() {
                self.next_frame.dispatch_tree.set_active_node(draw.parent_node);
            }
            if let Some(element) = draw.element.as_mut() {
                element.paint(self, cx);
            }
        }
        self.next_frame.deferred_draws = draws;
        self.scratch_order = order;
        self.element_id_stack.clear();
    }
}
