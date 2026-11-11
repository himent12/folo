//! `Animation` and `AnimationExt::with_animation`: re-renders an element with a progress `0..=1` each frame
//! while the animation runs (it calls `request_animation_frame`; a finished, non-repeating animation stops
//! asking, so the window goes idle). Start time is element state keyed by the animation id. Owner tag:
//! `"elements"`. The headless clock drives it in tests, so a frozen-clock screenshot is deterministic.
#![allow(dead_code)] // skeleton

use std::{rc::Rc, time::Duration};

use crate::{ElementId, IntoElement, elements::skeleton_element};

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

skeleton_element!([E: IntoElement + 'static] AnimationElement<E>, crate::AnyElement, (), "elements");
