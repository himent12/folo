//! The span decomposition of quads (`quad.rs`) against the shader at every pixel (`quad_eval.rs`).

use super::blend::Src;
use super::geom::{IRect, pixel_rect};
use super::gradient::{GradientSpan, Paint};
use super::quad::{QuadPrep, opaque_core};
use super::quad_eval::QuadShader;
use super::target::FrameMut;
use crate::{
    Background, BorderStyle, Bounds, Corners, Edges, Hsla, Quad, ScaledPixels, hsla, linear_color_stop, linear_gradient,
    point, size,
};
use std::sync::Arc;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn f(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next() >> 40) as f32 / (1u64 << 24) as f32 * (hi - lo)
    }

    fn pick<T: Copy>(&mut self, v: &[T]) -> T {
        v[(self.next() % v.len() as u64) as usize]
    }
}

fn px(v: f32) -> ScaledPixels {
    ScaledPixels(v)
}

fn random_quad(rng: &mut Rng, gradient: bool) -> Quad {
    let x = rng.pick(&[3.0, 3.5, 4.25, 2.75]);
    let y = rng.pick(&[2.0, 2.5, 3.1, 4.6]);
    let w = rng.f(4., 90.);
    let h = rng.f(4., 70.);
    let r = |rng: &mut Rng| if rng.next() % 3 == 0 { 0. } else { rng.pick(&[1.5, 3.0, 4.5, 6.0, 9.0, 12.0, 30.0]) };
    let bw = |rng: &mut Rng| if rng.next() % 2 == 0 { 0. } else { rng.pick(&[1.0, 1.5, 2.0, 3.0, 0.75]) };
    let color = |rng: &mut Rng| -> Hsla { hsla(rng.f(0., 1.), rng.f(0., 1.), rng.f(0., 1.), rng.pick(&[1.0, 0.5, 0.9, 0.2])) };
    let background: Background = if gradient {
        linear_gradient(rng.f(0., 360.), linear_color_stop(color(rng), 0.), linear_color_stop(color(rng), 1.))
    } else {
        color(rng).into()
    };
    Quad {
        bounds: Bounds::new(point(px(x), px(y)), size(px(w), px(h))),
        clip: Bounds::new(point(px(0.), px(0.)), size(px(200.), px(200.))),
        background,
        border_color: color(rng),
        corner_radii: Corners { top_left: px(r(rng)), top_right: px(r(rng)), bottom_right: px(r(rng)), bottom_left: px(r(rng)) },
        border_widths: Edges { top: px(bw(rng)), right: px(bw(rng)), bottom: px(bw(rng)), left: px(bw(rng)) },
        border_style: BorderStyle::Solid,
    }
}

fn prep(q: &Quad) -> QuadPrep {
    let sh = QuadShader::new(q);
    let grad = q.background.tag == crate::BackgroundTag::LinearGradient;
    let g = grad.then(|| {
        let p: &Paint = &sh.paint;
        let t: Arc<[Src]> = GradientSpan::table(p).into();
        GradientSpan::new(p, t)
    });
    QuadPrep::new(sh, g)
}

/// Draws `q` with spans in tiles of `tile` px into a 128x96 buffer of `bg`.
fn spans(q: &Quad, tile: i32, bg: u32) -> Vec<u32> {
    let (w, h) = (128, 96);
    let mut buf = vec![bg; (w * h) as usize];
    let p = prep(q);
    let rect = pixel_rect(&q.bounds, &q.clip).intersect(&IRect::new(0, 0, w, h));
    let fm = FrameMut::new(&mut buf, w as u32);
    for ty in (0..h).step_by(tile as usize) {
        for tx in (0..w).step_by(tile as usize) {
            let t = IRect::new(tx, ty, (tx + tile).min(w), (ty + tile).min(h));
            let r = rect.intersect(&t);
            if !r.is_empty() {
                // SAFETY: one target at a time, inside the buffer.
                let mut target = unsafe { fm.target(t) };
                p.draw(&mut target, r);
            }
        }
    }
    buf
}

fn reference(q: &Quad, bg: u32) -> Vec<u32> {
    let (w, h) = (128, 96);
    let mut buf = vec![bg; (w * h) as usize];
    let sh = QuadShader::new(q);
    let rect = pixel_rect(&q.bounds, &q.clip).intersect(&IRect::new(0, 0, w, h));
    for y in rect.y0..rect.y1 {
        for x in rect.x0..rect.x1 {
            let p = &mut buf[(y * w + x) as usize];
            *p = Src::new(sh.eval(x, y)).over(*p);
        }
    }
    buf
}

fn max_diff(a: &[u32], b: &[u32]) -> u32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (0..3).map(|c| ((x >> (c * 8)) & 0xff).abs_diff((y >> (c * 8)) & 0xff)).max().unwrap())
        .max()
        .unwrap_or(0)
}

#[test]
fn solid_quad_spans_equal_the_shader_at_every_pixel() {
    let mut rng = Rng(7);
    for case in 0..400 {
        let q = random_quad(&mut rng, false);
        let want = reference(&q, 0x203040);
        for tile in [64, 7] {
            let got = spans(&q, tile, 0x203040);
            assert_eq!(got, want, "case {case} tile {tile}: {q:?}");
        }
    }
}

#[test]
fn gradient_quad_spans_are_within_one_level_of_the_shader() {
    let mut rng = Rng(11);
    for case in 0..100 {
        let q = random_quad(&mut rng, true);
        let want = reference(&q, 0x102030);
        let a = spans(&q, 64, 0x102030);
        let b = spans(&q, 5, 0x102030);
        assert_eq!(a, b, "case {case}: tiles must not change pixels");
        assert!(max_diff(&a, &want) <= 1, "case {case}: {}", max_diff(&a, &want));
    }
}

#[test]
fn opaque_core_only_covers_background_pixels() {
    let mut rng = Rng(3);
    for case in 0..300 {
        let mut q = random_quad(&mut rng, false);
        q.background = hsla(0.3, 0.5, 0.5, 1.).into();
        let rect = pixel_rect(&q.bounds, &q.clip);
        let core = opaque_core(&q, rect);
        let sh = QuadShader::new(&q);
        let bg = Src::new(sh.paint.solid).packed;
        for y in core.y0..core.y1 {
            for x in core.x0..core.x1 {
                assert_eq!(Src::new(sh.eval(x, y)).over(0x123456), bg, "case {case} ({x}, {y})");
            }
        }
    }
}
