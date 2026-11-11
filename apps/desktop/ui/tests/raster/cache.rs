//! Cache bounds: byte budgets hold under churn, and hits come back without re-rendering.

use crate::scenes::{Rng, dsize, image, rect, sp};
use folo_ui::raster::{Mask, MaskCache, Renderer};
use folo_ui::*;
use std::sync::Arc;

#[test]
fn mask_cache_holds_its_budget_under_churn() {
    let budget = 64 << 10;
    let mut cache: MaskCache<u32> = MaskCache::with_budget(budget, 1000);
    let mut made = 0;
    for frame in 0..200u32 {
        cache.begin_frame();
        let mut used = 0;
        // A few masks every frame stay hot; the rest are new each frame.
        for k in (0..4).chain(frame * 10 + 1000..frame * 10 + 1010) {
            let m = cache
                .get_or_insert_with(&k, || {
                    made += 1;
                    Ok(Some(Mask::Alpha(Arc::new(AlphaMask::new(dsize(32, 32), vec![7u8; 1024])))))
                })
                .unwrap()
                .unwrap();
            used += m.byte_size() + 128;
        }
        cache.end_frame();
        assert!(cache.bytes() <= budget + used, "frame {frame}: {} bytes", cache.bytes());
    }
    // The hot keys were made once.
    assert_eq!(made, 4 + 200 * 10);
    // `None` is not cached.
    let mut calls = 0;
    for _ in 0..2 {
        assert!(
            cache
                .get_or_insert_with(&7, || {
                    calls += 1;
                    Ok(None)
                })
                .unwrap()
                .is_none()
        );
    }
    assert_eq!(calls, 2);
}

#[test]
fn mask_cache_drops_idle_entries() {
    let mut cache: MaskCache<u32> = MaskCache::with_budget(1 << 30, 3);
    cache
        .get_or_insert_with(&1, || {
            Ok(Some(Mask::Alpha(Arc::new(AlphaMask::new(dsize(4, 4), vec![0u8; 16])))))
        })
        .unwrap();
    for _ in 0..5 {
        cache.end_frame();
    }
    assert!(cache.is_empty());
}

#[test]
fn renderer_caches_hold_their_budget_under_image_churn() {
    let mut rng = Rng(77);
    let budget = 2 << 20;
    let mut r = Renderer::with_threads(1);
    r.set_cache_budget(budget);
    let mut fb = Framebuffer::new(400, 300);
    let mut peak = 0;
    for frame in 0..40 {
        let mut s = Scene::new();
        // Every frame: new images drawn scaled (each copy is 200x150x4 = 120 KB), new shadows.
        for i in 0..4 {
            s.push_image(Image {
                bounds: rect((i * 100) as f32, 0., 200., 150.),
                clip: rect(0., 0., 400., 300.),
                corner_radii: Corners::default(),
                image: image(&mut rng, 30, 20, false),
                frame: 0,
                grayscale: false,
                opacity: 1.,
            });
            s.push_shadow(Shadow {
                bounds: rect(20. + frame as f32 * 3. + i as f32, 160., 300., 100.),
                clip: rect(0., 0., 400., 300.),
                corner_radii: Corners::all(sp(6.)),
                color: hsla(0., 0., 0., 0.3),
                blur_radius: sp(8.),
            });
        }
        r.render(&s, GlyphBlend::IDENTITY, &mut fb);
        peak = peak.max(r.cache_bytes());
    }
    // One frame's worth (4 copies + 4 shadow masks, ~1 MB) may sit above the budget.
    assert!(peak <= budget + (1 << 20), "peak {peak}");
}

/// Removes a scratch file when dropped, also when the test fails.
struct Scratch(std::path::PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn png_round_trip() {
    // Scratch files go under FOLO_TEST_TMP when it is set (docs/CI.md, Test temp dirs).
    let dir =
        std::env::var_os("FOLO_TEST_TMP").map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&dir).unwrap();
    let file = Scratch(dir.join(format!("folo-raster-png-{}.png", std::process::id())));
    let mut fb = Framebuffer::new(5, 3);
    for (i, p) in fb.pixels.iter_mut().enumerate() {
        *p = (i as u32 * 0x0012_3457) & 0x00ff_ffff;
    }
    folo_ui::raster::debug::write_png(&fb, &file.0).unwrap();
    let back = folo_ui::raster::debug::read_png(&file.0).unwrap();
    assert_eq!(back, fb);
}
