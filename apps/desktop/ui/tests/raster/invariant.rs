//! DESIGN.md's invariant: incremental rendering equals a fresh render of the last scene byte for
//! byte, for every thread count, and the damage covers every changed pixel.

use crate::scenes::{Assets, Rng, mutate, scene};
use folo_ui::raster::Renderer;
use folo_ui::*;

const W: u32 = 333;
const H: u32 = 211;

fn blend() -> GlyphBlend {
    GlyphBlend::new([0.149_355_4, -0.226_006_7, 0.026_559_83, -0.034_591_47], 1.0)
}

fn fresh(s: &Scene, threads: usize) -> Framebuffer {
    let mut fb = Framebuffer::new(W, H);
    // Garbage: a fresh render must write every pixel.
    fb.fill(0x00de_adbe);
    let mut r = Renderer::with_threads(threads);
    r.render(s, blend(), &mut fb);
    fb
}

fn covered(damage: &[Bounds<DevicePixels>], x: u32, y: u32) -> bool {
    damage.iter().any(|b| {
        let (x, y) = (x as i32, y as i32);
        x >= b.origin.x.0 && y >= b.origin.y.0 && x < b.origin.x.0 + b.size.width.0 && y < b.origin.y.0 + b.size.height.0
    })
}

#[test]
fn incremental_equals_fresh_for_every_thread_count() {
    for seed in 0..12u64 {
        let mut rng = Rng(seed * 7919 + 1);
        let assets = Assets::new(&mut rng);
        let n = 60 + rng.i(0, 60) as usize;
        let mut s = scene(&mut rng, &assets, n, W as f32, H as f32);
        let mut incr = [Renderer::with_threads(1), Renderer::with_threads(4)];
        let mut fbs = [Framebuffer::new(W, H), Framebuffer::new(W, H)];
        let first = s.clone();
        for step in 0..6 {
            if step == 4 {
                // Back to the first scene: the tiles it covers repaint to what they were.
                s = first.clone();
            } else if step > 0 {
                s = mutate(&mut rng, &assets, &s, W as f32, H as f32);
            }
            let want = fresh(&s, 1);
            assert_eq!(fresh(&s, 4).pixels, want.pixels, "seed {seed} step {step}: fresh renders differ by thread count");
            for (r, fb) in incr.iter_mut().zip(fbs.iter_mut()) {
                let before = fb.pixels.clone();
                let damage = r.render(&s, blend(), fb);
                assert_eq!(fb.pixels, want.pixels, "seed {seed} step {step} threads {}", r.threads());
                for y in 0..H {
                    for x in 0..W {
                        let i = (y * W + x) as usize;
                        if before[i] != fb.pixels[i] {
                            assert!(covered(&damage, x, y), "seed {seed} step {step}: ({x}, {y}) changed outside the damage");
                        }
                    }
                }
                // Damage is tile-aligned and inside the frame.
                for b in &damage {
                    assert!(b.origin.x.0 % 64 == 0 && b.origin.y.0 % 64 == 0);
                    assert!(b.origin.x.0 + b.size.width.0 <= W as i32 && b.origin.y.0 + b.size.height.0 <= H as i32);
                }
                // The same scene again: nothing to do.
                let again = r.render(&s.clone(), blend(), fb);
                assert!(again.is_empty(), "seed {seed} step {step}: unchanged scene returned damage {again:?}");
                assert_eq!(fb.pixels, want.pixels);
            }
        }
    }
}

#[test]
fn resize_and_invalidate_repaint_everything() {
    let mut rng = Rng(99);
    let assets = Assets::new(&mut rng);
    let s = scene(&mut rng, &assets, 80, W as f32, H as f32);
    let mut r = Renderer::with_threads(2);
    let mut fb = Framebuffer::new(W, H);
    r.render(&s, blend(), &mut fb);
    // A new buffer of another size.
    let mut fb2 = Framebuffer::new(W - 50, H + 13);
    fb2.fill(0x0012_3456);
    let damage = r.render(&s, blend(), &mut fb2);
    let area: i32 = damage.iter().map(|b| b.size.width.0 * b.size.height.0).sum();
    assert_eq!(area, ((W - 50) * (H + 13)) as i32);
    let mut want = Framebuffer::new(W - 50, H + 13);
    Renderer::with_threads(1).render(&s, blend(), &mut want);
    assert_eq!(fb2.pixels, want.pixels);
    // `invalidate`: the same size, scribbled buffer, all repainted.
    fb2.fill(0x00ff_00ff);
    r.invalidate();
    r.render(&s, blend(), &mut fb2);
    assert_eq!(fb2.pixels, want.pixels);
}

#[test]
fn a_new_glyph_blend_repaints() {
    let mut rng = Rng(5);
    let assets = Assets::new(&mut rng);
    let s = scene(&mut rng, &assets, 40, W as f32, H as f32);
    let mut r = Renderer::with_threads(1);
    let mut fb = Framebuffer::new(W, H);
    r.render(&s, blend(), &mut fb);
    assert!(!r.render(&s, GlyphBlend::IDENTITY, &mut fb).is_empty());
    let mut want = Framebuffer::new(W, H);
    Renderer::with_threads(1).render(&s, GlyphBlend::IDENTITY, &mut want);
    assert_eq!(fb.pixels, want.pixels);
}
