// Adapted from GPUI (Apache-2.0): crates/gpui/src/platform/windows/shaders.hlsl (gradient_color,
// prepare_gradient_color, srgb_to_oklab, oklab_to_srgb)
//! Backgrounds: solid, two-stop linear gradients (sRGB and Oklab) and the slash pattern, evaluated as
//! GPUI's shader does, plus an integer form of the gradient for spans.

use super::blend::{Src, rgba, sat};
use crate::{Background, BackgroundTag, ColorSpace};
use std::sync::Arc;

/// The HLSL `M_PI_F`.
const M_PI_F: f32 = 3.141_592_7;

// GPUI's shader names these the other way round; the maths is copied as it is.
fn srgb_to_linear(c: f32) -> f32 {
    c.max(0.).powf(1.0 / 2.2)
}

fn linear_to_srgb(c: f32) -> f32 {
    c.max(0.).powf(2.2)
}

#[allow(clippy::excessive_precision)]
fn srgb_to_oklab(c: [f32; 4]) -> [f32; 4] {
    let (r, g, b) = (srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]));
    let l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    let m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    let s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
    let (l, m, s) = (l.max(0.).powf(1. / 3.), m.max(0.).powf(1. / 3.), s.max(0.).powf(1. / 3.));
    [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
        c[3],
    ]
}

#[allow(clippy::excessive_precision)]
fn oklab_to_srgb(c: [f32; 4]) -> [f32; 4] {
    let l = c[0] + 0.3963377774 * c[1] + 0.2158037573 * c[2];
    let m = c[0] - 0.1055613458 * c[1] - 0.0638541728 * c[2];
    let s = c[0] - 0.0894841775 * c[1] - 1.2914855480 * c[2];
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    [
        linear_to_srgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
        linear_to_srgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
        linear_to_srgb(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s),
        c[3],
    ]
}

fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

/// A background ready to evaluate at the pixel centres of one primitive's `bounds`.
#[derive(Clone, Debug)]
pub(crate) struct Paint {
    pub tag: BackgroundTag,
    oklab: bool,
    /// The solid colour (solid and pattern).
    pub solid: [f32; 4],
    c0: [f32; 4],
    c1: [f32; 4],
    p0: f32,
    p1: f32,
    /// `t = ax * x + ay * y + a0` before the stop adjustment and clamp (gradients).
    ax: f32,
    ay: f32,
    a0: f32,
    pattern_height: f32,
    origin: [f32; 2],
}

impl Paint {
    /// Prepares `bg` over `bounds` (`origin`, `size`), as `prepare_gradient_color` does.
    pub fn new(bg: &Background, origin: [f32; 2], size: [f32; 2]) -> Self {
        let mut p = Paint {
            tag: bg.tag,
            oklab: bg.color_space == ColorSpace::Oklab,
            solid: rgba(bg.solid),
            c0: [0.; 4],
            c1: [0.; 4],
            p0: bg.colors[0].percentage,
            p1: bg.colors[1].percentage,
            ax: 0.,
            ay: 0.,
            a0: 0.,
            pattern_height: bg.gradient_angle_or_pattern_height,
            origin,
        };
        if bg.tag == BackgroundTag::LinearGradient {
            p.c0 = rgba(bg.colors[0].color);
            p.c1 = rgba(bg.colors[1].color);
            if p.oklab {
                p.c0 = srgb_to_oklab(p.c0);
                p.c1 = srgb_to_oklab(p.c1);
            }
            // -90 degrees to match the CSS gradient angle.
            let radians = (bg.gradient_angle_or_pattern_height % 360.0 - 90.0) * (M_PI_F / 180.0);
            let mut dir = [radians.cos(), radians.sin()];
            if size[0] > size[1] {
                dir[1] *= size[1] / size[0];
            } else {
                dir[0] *= size[0] / size[1];
            }
            let len = (dir[0] * dir[0] + dir[1] * dir[1]).sqrt();
            let half = [size[0] * 0.5, size[1] * 0.5];
            let center = [origin[0] + half[0], origin[1] + half[1]];
            // t = dot(pos - center, dir) / len, then (t + half) / size along the dominant axis.
            let (h, s) = if dir[0].abs() > dir[1].abs() { (half[0], size[0]) } else { (half[1], size[1]) };
            p.ax = dir[0] / len / s;
            p.ay = dir[1] / len / s;
            p.a0 = (-(center[0] * dir[0] + center[1] * dir[1]) / len + h) / s;
        }
        p
    }

    /// The gradient parameter at a pixel centre, after the stops and the clamp.
    #[inline]
    pub fn t_at(&self, x: f32, y: f32) -> f32 {
        let t = self.ax * x + self.ay * y + self.a0;
        sat((t - self.p0) / (self.p1 - self.p0))
    }

    /// The colour of the gradient at parameter `t` (straight).
    pub fn gradient(&self, t: f32) -> [f32; 4] {
        let c = lerp4(self.c0, self.c1, t);
        if self.oklab { oklab_to_srgb(c) } else { c }
    }

    /// The background colour at a pixel centre (straight rgb and alpha).
    pub fn color_at(&self, x: f32, y: f32) -> [f32; 4] {
        match self.tag {
            BackgroundTag::Solid => self.solid,
            BackgroundTag::LinearGradient => self.gradient(self.t_at(x, y)),
            BackgroundTag::PatternSlash => {
                let h = self.pattern_height;
                let width = (h / 65535.0) / 255.0;
                let interval = (h % 65535.0) / 255.0;
                let height = width + interval;
                let angle = M_PI_F / 4.0;
                let period = height * angle.sin();
                let (s, c) = angle.sin_cos();
                let rel = [x - self.origin[0], y - self.origin[1]];
                // mul(row vector, float2x2(c, -s, s, c)).x
                let rotated_x = rel[0] * c + rel[1] * s;
                let pattern = rotated_x % period;
                let distance = pattern.min(period - pattern) - period * (width / height) / 2.0;
                let mut color = self.solid;
                color[3] *= sat(0.5 - distance);
                color
            }
        }
    }

    pub fn is_solid(&self) -> bool {
        self.tag == BackgroundTag::Solid
    }

    pub fn is_gradient(&self) -> bool {
        self.tag == BackgroundTag::LinearGradient
    }
}

/// Gradient colours are tabulated at `LUT_STEPS + 1` evenly spaced values of `t`.
pub(crate) const LUT_STEPS: usize = 1024;

/// `t * LUT_STEPS` carries 16 fractional bits.
const FRAC: i64 = 1 << 16;

/// A gradient in integer form: `t` is affine in the pixel's integer coordinates, so every tile
/// computes the same value for the same pixel and no pixel needs float maths.
pub(crate) struct GradientSpan {
    /// Colours at `t = i / LUT_STEPS`.
    lut: Arc<[Src]>,
    kx: i64,
    ky: i64,
    k0: i64,
}

impl GradientSpan {
    /// The colour table of a gradient (it depends on the stops' colours and the colour space only).
    pub fn table(p: &Paint) -> Vec<Src> {
        (0..=LUT_STEPS).map(|i| Src::new(p.gradient(i as f32 / LUT_STEPS as f32))).collect()
    }

    /// The span form of a gradient `Paint` with its colour `table`.
    pub fn new(p: &Paint, lut: Arc<[Src]>) -> Self {
        debug_assert_eq!(lut.len(), LUT_STEPS + 1);
        let d = (p.p1 - p.p0) as f64;
        let scale = LUT_STEPS as f64 * FRAC as f64 / d;
        let finite = |v: f64| if v.is_finite() { v.round() as i64 } else { 0 };
        let kx = finite(p.ax as f64 * scale);
        let ky = finite(p.ay as f64 * scale);
        let k0 = finite(((p.ax + p.ay) as f64 * 0.5 + p.a0 as f64 - p.p0 as f64) * scale);
        GradientSpan { lut, kx, ky, k0 }
    }

    #[inline]
    fn index(u: i64) -> usize {
        ((u.clamp(0, LUT_STEPS as i64 * FRAC) + FRAC / 2) >> 16) as usize
    }

    /// The colour of pixel (`x`, `y`).
    #[inline]
    pub fn at(&self, x: i32, y: i32) -> &Src {
        &self.lut[Self::index(self.k0 + self.kx * x as i64 + self.ky * y as i64)]
    }

    /// Draws the gradient over `span`, the pixels from `x0` of row `y`.
    pub fn draw(&self, span: &mut [u32], x0: i32, y: i32) {
        let mut u = self.k0 + self.kx * x0 as i64 + self.ky * y as i64;
        for p in span {
            *p = self.lut[Self::index(u)].over(*p);
            u += self.kx;
        }
    }
}
