// Adapted from GPUI (Apache-2.0): crates/gpui/src/elements/animation.rs (progress rule)
//! `Animation` and `AnimationExt::with_animation`: re-renders an element with a progress `0..=1` each frame
//! while the animation runs (it calls `request_animation_frame`; a finished, non-repeating animation stops
//! asking, so the window goes idle). Start time is element state keyed by the animation id. Owner tag:
//! `"elements"`. The headless clock drives it in tests, so a frozen-clock screenshot is deterministic.
//!
//! `AnimationExt::with_opacity_animation` animates only an opacity, and does it without rebuilding the UI:
//! the element is painted once inside an opacity group and the window re-fades the group at each refresh
//! (`window/paint_animation.rs`). The picture at any time is the one `with_animation` setting
//! `.opacity(animator(progress))` on the same element gives.

use std::{panic, rc::Rc, time::Duration};

use crate::{
    AnyElement, App, Bounds, Element, ElementId, GlobalElementId, InspectorElementId, IntoElement, LayoutId,
    Pixels, Window, window::PaintAnimation,
};

/// An animation: duration, repeat, easing.
#[derive(Clone)]
pub struct Animation {
    /// One cycle.
    pub duration: Duration,
    /// Runs once (`false`: repeats forever).
    pub oneshot: bool,
    /// Maps linear progress to eased progress.
    pub easing: Rc<dyn Fn(f32) -> f32>,
}

impl Animation {
    /// A one-shot linear animation of `duration`.
    pub fn new(duration: Duration) -> Self {
        Animation { duration, oneshot: true, easing: Rc::new(|t| t) }
    }

    /// Repeats forever.
    pub fn repeat(mut self) -> Self {
        self.oneshot = false;
        self
    }

    /// Sets the easing function.
    pub fn with_easing(mut self, easing: impl Fn(f32) -> f32 + 'static) -> Self {
        self.easing = Rc::new(easing);
        self
    }
}

/// Adds `with_animation` to every element.
pub trait AnimationExt {
    /// Renders `self` through `animator(self, progress)` each frame of `animation`.
    fn with_animation(
        self,
        id: impl Into<ElementId>,
        animation: Animation,
        animator: impl Fn(Self, f32) -> Self + 'static,
    ) -> AnimationElement<Self>
    where
        Self: Sized,
    {
        AnimationElement { id: id.into(), element: Some(self), animator: Box::new(animator), animation }
    }

    /// Paints `self` at the opacity `animator(progress)` while `animation` runs, without re-rendering,
    /// laying out or repainting anything (the same picture as `with_animation` setting
    /// `.opacity(animator(progress))` on the element).
    fn with_opacity_animation(
        self,
        id: impl Into<ElementId>,
        animation: Animation,
        animator: impl Fn(f32) -> f32 + 'static,
    ) -> OpacityAnimationElement<Self>
    where
        Self: Sized,
    {
        OpacityAnimationElement {
            id: id.into(),
            element: Some(self),
            animation,
            animator: Some(Box::new(animator)),
            start: None,
            factor: 1.0,
            done: true,
        }
    }
}

impl<E: IntoElement + 'static> AnimationExt for E {}

/// An element being animated.
pub struct AnimationElement<E> {
    /// The animation's id (keys its start time).
    pub(crate) id: ElementId,
    /// The element before the animator ran.
    pub(crate) element: Option<E>,
    /// The animation.
    pub(crate) animation: Animation,
    /// Applies progress.
    pub(crate) animator: Box<dyn Fn(E, f32) -> E + 'static>,
}

/// The start time of an animation, kept as element state.
pub(crate) struct AnimationState {
    /// When it started (the dispatcher's clock).
    pub(crate) start: std::time::Instant,
}

impl<E: IntoElement + 'static> Element for AnimationElement<E> {
    type RequestLayoutState = AnyElement;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        Some(self.id.clone())
    }

    fn source_location(&self) -> Option<&'static panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, AnyElement) {
        let global_id = global_id.expect("an animation element has an id");
        // The dispatcher's clock: real time on a real platform, the fake clock headless.
        let now = cx.background_executor().now();
        window.with_element_state(global_id, |state: Option<AnimationState>, window| {
            let state = state.unwrap_or(AnimationState { start: now });
            let (delta, done) = progress(&self.animation, now.saturating_duration_since(state.start));
            let element = self.element.take().expect("an animation element lays out once");
            let mut element = (self.animator)(element, delta).into_any_element();
            if !done {
                window.request_animation_frame();
            }
            ((element.request_layout(window, cx), element), state)
        })
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        element: &mut AnyElement,
        window: &mut Window,
        cx: &mut App,
    ) {
        element.prepaint(window, cx);
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        element: &mut AnyElement,
        _prepaint: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        element.paint(window, cx);
    }
}

impl<E: IntoElement + 'static> IntoElement for AnimationElement<E> {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// An element whose opacity is animated by the window (`with_opacity_animation`).
pub struct OpacityAnimationElement<E> {
    /// The animation's id (keys its start time).
    id: ElementId,
    /// The element, until it is laid out.
    element: Option<E>,
    /// The animation.
    animation: Animation,
    /// Eased progress to opacity (handed to the window at paint).
    animator: Option<Box<dyn Fn(f32) -> f32>>,
    /// When the animation started (known after `request_layout`).
    start: Option<std::time::Instant>,
    /// The opacity this frame.
    factor: f32,
    /// Whether it has finished.
    done: bool,
}

impl<E: IntoElement + 'static> Element for OpacityAnimationElement<E> {
    type RequestLayoutState = AnyElement;
    type PrepaintState = ();

    fn id(&self) -> Option<ElementId> {
        Some(self.id.clone())
    }

    fn source_location(&self) -> Option<&'static panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        global_id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, AnyElement) {
        let global_id = global_id.expect("an animation element has an id");
        let now = cx.background_executor().now();
        let start = window.with_element_state(global_id, |state: Option<AnimationState>, _| {
            let state = state.unwrap_or(AnimationState { start: now });
            (state.start, state)
        });
        let (delta, done) = progress(&self.animation, now.saturating_duration_since(start));
        let animator = self.animator.as_ref().expect("the animator is kept until paint");
        self.factor = animator(delta);
        self.start = Some(start);
        self.done = done;
        let mut element = self.element.take().expect("an animation element lays out once").into_any_element();
        (element.request_layout(window, cx), element)
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        element: &mut AnyElement,
        window: &mut Window,
        cx: &mut App,
    ) {
        element.prepaint(window, cx);
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        element: &mut AnyElement,
        _prepaint: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        let ((), group) = window.with_opacity_group(self.factor, |window| element.paint(window, cx));
        if let (Some(animator), Some(start)) = (self.animator.take(), self.start) {
            window.register_paint_animation(PaintAnimation {
                group,
                start,
                animation: self.animation.clone(),
                animator,
                done: self.done,
                factor: self.factor,
            });
        }
    }
}

impl<E: IntoElement + 'static> IntoElement for OpacityAnimationElement<E> {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

/// The eased progress of `animation` after `elapsed`, and whether it has finished (GPUI's rule: at or past
/// the end a one-shot animation stays at 1 and is done; a repeating one wraps).
pub(crate) fn progress(animation: &Animation, elapsed: Duration) -> (f32, bool) {
    let duration = animation.duration.as_secs_f32();
    if !duration.is_finite() || duration <= 0.0 {
        let eased = (animation.easing)(1.0);
        return (if eased.is_finite() { eased } else { 1.0 }, animation.oneshot);
    }
    let mut delta = elapsed.as_secs_f32() / duration;
    let mut done = false;
    if !delta.is_finite() || delta >= 1.0 {
        if animation.oneshot {
            done = true;
            delta = 1.0;
        } else if delta.is_finite() {
            delta %= 1.0;
        } else {
            delta = 0.0;
        }
    }
    let eased = (animation.easing)(delta.clamp(0.0, 1.0));
    (if eased.is_finite() { eased } else { 1.0 }, done)
}
