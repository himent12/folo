//! Opacity groups while painting, and paint animations: opacity animations that run without rebuilding the
//! UI (`AnimationExt::with_opacity_animation`). Owner tag: `"window"`.
//!
//! A full frame paints each animated element inside an opacity group (`with_opacity_group`) and registers
//! a `PaintAnimation` for it (group, start time, animation, closure). When a frame is due and the window
//! is not dirty, `tick_paint_animations` recomputes each running animation's factor; only if one changed by
//! at least 1/255 (or an animation just finished, which lands exactly on its last value) does it update
//! the rendered scene's group table, with every factor at its exact value for this moment, and present
//! the damage: no render, layout or paint. Every frame shown is therefore byte for byte the picture of one
//! moment of the animations (a skipped tick leaves the previous moment on screen). The window keeps
//! asking for frames while any paint animation runs (a repeating one runs forever, a finished one stops).
//! A full redraw for any other reason paints the groups again, and the elements re-register with the start
//! times they keep in element state, so the animations continue where they were.

use std::time::Instant;

use crate::{Animation, App, Window, elements::progress};

/// The smallest factor change a tick shows (one 8-bit level of alpha).
const MIN_STEP: f32 = 1.0 / 255.0;

/// An opacity animation registered by a paint.
pub(crate) struct PaintAnimation {
    /// The scene's opacity group it drives.
    pub(crate) group: u32,
    /// When it started (the dispatcher's clock).
    pub(crate) start: Instant,
    /// Duration, repeat, easing.
    pub(crate) animation: Animation,
    /// Eased progress to the group's factor.
    pub(crate) animator: Box<dyn Fn(f32) -> f32>,
    /// It finished (a one-shot animation past its end).
    pub(crate) done: bool,
    /// Its factor at the last tick (or at paint).
    pub(crate) factor: f32,
}

impl Window {
    /// Paints `f` inside a new opacity group with `factor` (see `opacity.rs`): its primitives are painted at
    /// base colours and faded by the group after paint. Returns `f`'s result and the group.
    pub(crate) fn with_opacity_group<R>(&mut self, factor: f32, f: impl FnOnce(&mut Self) -> R) -> (R, u32) {
        let scene = &mut self.next_frame.scene;
        let parent = self.opacity_group_stack.last().copied();
        let outer = if parent.is_some() { 1.0 } else { self.element_opacity };
        if let Some((parent, start)) = parent {
            scene.push_group_span(parent, start);
        }
        let group = scene.push_opacity_group(parent.map(|(group, _)| group), outer, factor);
        self.opacity_group_stack.push((group, scene.len()));
        let element_opacity = std::mem::replace(&mut self.element_opacity, 1.0);
        let result = f(self);
        self.element_opacity = element_opacity;
        let (_, start) = self.opacity_group_stack.pop().expect("pushed above");
        let scene = &mut self.next_frame.scene;
        scene.push_group_span(group, start);
        if let Some((_, start)) = self.opacity_group_stack.last_mut() {
            *start = scene.len();
        }
        (result, group)
    }

    /// Registers the paint animation of `group` for this frame.
    pub(crate) fn register_paint_animation(&mut self, animation: PaintAnimation) {
        self.next_frame.paint_animations.push(animation);
    }

    /// Whether a paint animation of the frame on screen is still running.
    pub(crate) fn has_running_paint_animations(&self) -> bool {
        self.rendered_frame.paint_animations.iter().any(|animation| !animation.done)
    }

    /// Advances the paint animations of the frame on screen to now; returns whether the scene changed. Asks
    /// for the next frame while any of them runs.
    pub(crate) fn tick_paint_animations(&mut self, cx: &mut App) -> bool {
        if self.rendered_frame.paint_animations.is_empty() {
            return false;
        }
        let now = cx.background_executor().now();
        let frame = &mut self.rendered_frame;
        let (mut moved, mut running) = (false, false);
        for animation in &mut frame.paint_animations {
            if animation.done {
                continue;
            }
            let (delta, done) =
                progress(&animation.animation, now.saturating_duration_since(animation.start));
            let factor = (animation.animator)(delta);
            let current = frame.scene.opacity_groups[animation.group as usize].factor;
            moved |= factor != current && (done || (factor - current).abs() >= MIN_STEP);
            animation.done = done;
            animation.factor = factor;
            running |= !done;
        }
        // When one moved enough, every one takes its exact value now, so the frame shown is the picture
        // at this moment exactly; otherwise the frame on screen (an exact earlier moment) stays.
        if moved {
            for animation in &frame.paint_animations {
                frame.scene.set_group_factor(animation.group, animation.factor);
            }
            frame.scene.apply_opacity_groups();
        }
        if running {
            self.platform_window.request_frame();
        }
        moved
    }
}
