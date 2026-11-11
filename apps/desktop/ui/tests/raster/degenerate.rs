//! Degenerate inputs keep the invariant: empty frames, non-finite colours and geometry, glyph blends
//! whose parameters differ only in `apply`.

use crate::scenes::{dsize, rect, sp};
use folo_ui::raster::Renderer;
use folo_ui::*;
use std::sync::Arc;

const W: u32 = 128;
const H: u32 = 96;

fn quad(b: Bounds<ScaledPixels>, bg: impl Into<Background>) -> Quad {
    Quad {
        bounds: b,
        clip: rect(0., 0., W as f32, H as f32),
        background: bg.into(),
        border_color: transparent_black(),
        corner_radii: Corners::default(),
        border_widths: Edges::default(),
        border_style: BorderStyle::Solid,
    }
}

fn scene(prims: Vec<Primitive>) -> Scene {
    let mut s = Scene::new();
    for p in prims {
        s.push(p);
    }
    s
}

fn fresh(s: &Scene, blend: GlyphBlend) -> Framebuffer {
    let mut fb = Framebuffer::new(W, H);
    fb.fill(0x00de_adbe);
    Renderer::with_threads(1).render(s, blend, &mut fb);
    fb
}

/// Renders `a` then `b` into one buffer; `Err` names the first pixel that differs from `b` fresh.
fn incremental(a: &Scene, b: &Scene, blend_a: GlyphBlend, blend_b: GlyphBlend) -> Result<(), String> {
    let mut r = Renderer::with_threads(1);
    let mut fb = Framebuffer::new(W, H);
    r.render(a, blend_a, &mut fb);
    r.render(b, blend_b, &mut fb);
    let want = fresh(b, blend_b);
    match fb.pixels.iter().zip(&want.pixels).position(|(g, w)| g != w) {
        None => Ok(()),
        Some(i) => Err(format!(
            "({}, {}): incremental {:#08x}, fresh {:#08x}",
            i as u32 % W,
            i as u32 / W,
            fb.pixels[i],
            want.pixels[i]
        )),
    }
}

fn assert_incremental(a: &Scene, b: &Scene, blend_a: GlyphBlend, blend_b: GlyphBlend, what: &str) {
    if let Err(e) = incremental(a, b, blend_a, blend_b) {
        panic!("{what}: {e}");
    }
}

fn white() -> Hsla {
    hsla(0., 0., 1., 1.)
}

#[test]
fn zero_sized_frames_draw_nothing_and_the_next_frame_repaints() {
    let s = scene(vec![Primitive::Quad(quad(rect(4., 4., 100., 70.), rgb(0x336699)))]);
    let mut r = Renderer::with_threads(2);
    let mut fb = Framebuffer::new(W, H);
    r.render(&s, GlyphBlend::IDENTITY, &mut fb);
    // A minimized window: either side zero, then both.
    for (w, h) in [(0, 0), (W, 0), (0, H), (0, 0)] {
        let mut empty = Framebuffer::new(w, h);
        let damage = r.render(&s, GlyphBlend::IDENTITY, &mut empty);
        assert!(damage.is_empty(), "{w}x{h}: {damage:?}");
        assert_eq!(r.stats().tiles, 0);
    }
    // Restored: a scribbled buffer of the old size is repainted whole.
    fb.fill(0x00ff_00ff);
    let damage = r.render(&s, GlyphBlend::IDENTITY, &mut fb);
    let area: i32 = damage.iter().map(|b| b.size.width.0 * b.size.height.0).sum();
    assert_eq!(area, (W * H) as i32);
    assert_eq!(fb.pixels, fresh(&s, GlyphBlend::IDENTITY).pixels);
}

#[test]
fn a_nan_alpha_quad_does_not_hide_what_is_under_it() {
    let opaque = quad(rect(0., 0., 64., 64.), white());
    let mut nan = opaque.clone();
    nan.background.solid.a = f32::NAN;
    let a = scene(vec![Primitive::Quad(opaque)]);
    let b = scene(vec![Primitive::Quad(nan)]);
    assert_incremental(&a, &b, GlyphBlend::IDENTITY, GlyphBlend::IDENTITY, "NaN alpha");
    // Over another quad: the NaN one is transparent, so the one under it shows.
    let under = quad(rect(0., 0., 64., 64.), rgb(0x336699));
    let mut nan = quad(rect(0., 0., 64., 64.), white());
    nan.background.solid.a = f32::NAN;
    let fb = fresh(&scene(vec![Primitive::Quad(under), Primitive::Quad(nan)]), GlyphBlend::IDENTITY);
    assert_eq!(fb.pixel(10, 10), 0x336699);
}

#[test]
fn non_finite_quad_fields_keep_the_invariant() {
    let base = || {
        let mut q = quad(rect(3., 5., 90., 70.), white());
        q.corner_radii = Corners::all(sp(6.));
        q.border_widths = Edges::all(sp(2.));
        q.border_color = hsla(0.6, 0.5, 0.5, 1.);
        q
    };
    type Poke = fn(&mut Quad, f32);
    let pokes: [(&str, Poke); 9] = [
        ("background alpha", |q, v| q.background.solid.a = v),
        ("top-left radius", |q, v| q.corner_radii.top_left = sp(v)),
        ("bottom-right radius", |q, v| q.corner_radii.bottom_right = sp(v)),
        ("left border", |q, v| q.border_widths.left = sp(v)),
        ("top border", |q, v| q.border_widths.top = sp(v)),
        ("border alpha", |q, v| q.border_color.a = v),
        ("width", |q, v| q.bounds.size.width = sp(v)),
        ("x", |q, v| q.bounds.origin.x = sp(v)),
        ("plain radius", |q, v| {
            q.border_widths = Edges::default();
            q.corner_radii = Corners::all(sp(v));
        }),
    ];
    let mut failures = Vec::new();
    for (what, poke) in pokes {
        for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut q = base();
            poke(&mut q, v);
            // A frame of the base quad alone, then the poked quad on a full-frame white one, and back.
            let full = quad(rect(0., 0., W as f32, H as f32), white());
            let a = scene(vec![Primitive::Quad(base())]);
            let b = scene(vec![Primitive::Quad(full), Primitive::Quad(q)]);
            for (from, to, dir) in [(&a, &b, "on"), (&b, &a, "back")] {
                if let Err(e) = incremental(from, to, GlyphBlend::IDENTITY, GlyphBlend::IDENTITY) {
                    failures.push(format!("{what} = {v} ({dir}): {e}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn glyph_tables_tell_blends_apart_by_apply() {
    // Two blends whose gamma bits differ by exactly the bit `apply` would flip if they were combined.
    let off = GlyphBlend { gamma_ratios: [0., 0., 0., 1.], grayscale_enhanced_contrast: 1., apply: false };
    let on = GlyphBlend {
        gamma_ratios: [f32::from_bits(1), 0., 0., 1.],
        grayscale_enhanced_contrast: 1.,
        apply: true,
    };
    let mask = Arc::new(AlphaMask::new(dsize(8, 8), vec![128u8; 64]));
    let s = scene(vec![Primitive::MonoSprite(MonoSprite {
        bounds: rect(10., 10., 8., 8.),
        clip: rect(0., 0., W as f32, H as f32),
        color: white(),
        mask,
        kind: SpriteKind::Glyph,
    })]);
    assert_ne!(fresh(&s, off).pixel(12, 12), fresh(&s, on).pixel(12, 12), "the blends draw differently");
    assert_incremental(&s, &s, off, on, "uncorrected then corrected");
    assert_incremental(&s, &s, on, off, "corrected then uncorrected");
}
