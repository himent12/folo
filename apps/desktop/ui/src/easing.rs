// Adapted from GPUI (Apache-2.0): crates/gpui/src/elements/animation.rs (easing)
//! Easing functions for animations: each maps progress `delta` in `0..=1` to eased progress.

use std::f32::consts::PI;

/// No easing: `delta` itself.
pub fn linear(delta: f32) -> f32 {
    delta
}

/// `delta * delta`.
pub fn quadratic(delta: f32) -> f32 {
    delta * delta
}

/// Quadratic ease-in-out: slow at both ends, fast in the middle.
pub fn ease_in_out(delta: f32) -> f32 {
    if delta < 0.5 {
        2.0 * delta * delta
    } else {
        let x = -2.0 * delta + 2.0;
        1.0 - x * x / 2.0
    }
}

/// Quintic ease-out: fast at first, decelerating to a stop.
pub fn ease_out_quint() -> impl Fn(f32) -> f32 {
    move |delta| 1.0 - (1.0 - delta).powi(5)
}

/// `easing` forwards over the first half, then backwards over the second.
pub fn bounce(easing: impl Fn(f32) -> f32) -> impl Fn(f32) -> f32 {
    move |delta| if delta < 0.5 { easing(delta * 2.0) } else { easing((1.0 - delta) * 2.0) }
}

/// A breathing pulse between `min` and `max` (sine and cubic), for pulsating alpha.
pub fn pulsating_between(min: f32, max: f32) -> impl Fn(f32) -> f32 {
    let range = max - min;
    move |delta| {
        let t = (delta * 2.0 * PI).sin();
        let breath = (t * t * t + t) / 2.0;
        let normalized_alpha = (breath + 1.0) / 2.0;
        min + (normalized_alpha * range)
    }
}
