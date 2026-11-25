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
fn glyph_mask_cache_stores_bounds_and_empty_glyphs_and_evicts_idle() {
    use folo_ui::raster::GlyphMask;

    let mut cache: MaskCache<u32, GlyphMask> = MaskCache::with_budget(1 << 20, 2);
    let bounds = Bounds::new(point(DevicePixels(3), DevicePixels(-7)), dsize(4, 5));
    let mut present_calls = 0;
    let mut empty_calls = 0;
    for _ in 0..3 {
        let entry = cache
            .get_or_insert_with(&1, || {
                present_calls += 1;
                Ok(Some(GlyphMask::Present {
                    raster_bounds: bounds,
                    mask: Mask::Alpha(Arc::new(AlphaMask::new(dsize(4, 5), vec![255u8; 20]))),
                }))
            })
            .unwrap()
            .unwrap();
        let GlyphMask::Present { raster_bounds, mask } = entry else { panic!("expected Present") };
        assert_eq!(raster_bounds, bounds);
        assert_eq!(mask.into_alpha().unwrap().size, dsize(4, 5));

        let empty = cache
            .get_or_insert_with(&2, || {
                empty_calls += 1;
                Ok(Some(GlyphMask::Empty))
            })
            .unwrap()
            .unwrap();
        assert!(matches!(empty, GlyphMask::Empty));
    }
    assert_eq!(present_calls, 1, "present glyph should hit cache after first insert");
    assert_eq!(empty_calls, 1, "empty glyph should hit cache after first insert");
    assert_eq!(cache.len(), 2);

    for _ in 0..4 {
        cache.end_frame();
    }
    assert!(cache.is_empty(), "idle present and empty glyph entries should both evict");
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

#[test]
fn png_round_trip() {
    // Under FOLO_TEST_TMP, removed on drop (docs/CI.md, Test temp dirs).
    let file = folo_test_support::TestFile::new("raster-png", "frame.png");
    let mut fb = Framebuffer::new(5, 3);
    for (i, p) in fb.pixels.iter_mut().enumerate() {
        *p = (i as u32 * 0x0012_3457) & 0x00ff_ffff;
    }
    folo_ui::raster::debug::write_png(&fb, file.path()).unwrap();
    let back = folo_ui::raster::debug::read_png(file.path()).unwrap();
    assert_eq!(back, fb);
}

#[test]
fn mask_cache_throttles_long_idle_sweeps_and_reclaims_under_budget() {
    // 1. Long idle limit (>= 64 frames): entry survives through max_idle and is evicted by the
    //    next periodic 64-frame sweep.
    let mut cache: MaskCache<u32> = MaskCache::with_budget(1 << 30, 64);
    cache
        .get_or_insert_with(&1, || {
            Ok(Some(Mask::Alpha(Arc::new(AlphaMask::new(dsize(4, 4), vec![1u8; 16])))))
        })
        .unwrap();
    for _ in 0..=64 {
        cache.end_frame();
    }
    assert_eq!(cache.len(), 1, "entry should still be live at frame == max_idle");
    for _ in 65..=128 {
        cache.end_frame();
    }
    assert!(cache.is_empty(), "idle entry must be evicted once the periodic sweep runs");

    // 2. Under budget pressure on a non-sweep frame, expired idle entries are reclaimed immediately.
    let mut tight: MaskCache<u32> = MaskCache::with_budget(200, 64);
    tight
        .get_or_insert_with(&1, || {
            Ok(Some(Mask::Alpha(Arc::new(AlphaMask::new(dsize(8, 8), vec![1u8; 64])))))
        })
        .unwrap();
    // Advance to frame 66 (past max_idle=64, not a multiple of 64).
    for _ in 0..66 {
        tight.end_frame();
    }
    assert_eq!(tight.len(), 1);
    tight
        .get_or_insert_with(&2, || {
            Ok(Some(Mask::Alpha(Arc::new(AlphaMask::new(dsize(10, 10), vec![2u8; 100])))))
        })
        .unwrap();
    // Total bytes now exceed the 200-byte budget; end_frame on frame 66 drops idle key 1 immediately.
    tight.end_frame();
    assert_eq!(tight.len(), 1);
    let mut rebuilt = false;
    tight
        .get_or_insert_with(&2, || {
            rebuilt = true;
            Ok(None)
        })
        .unwrap();
    assert!(!rebuilt, "fresh key 2 should remain cached after idle key 1 is reclaimed");
}

#[test]
fn mask_cache_get_or_insert_with_by_avoids_key_construction_on_hit() {
    let mut cache: MaskCache<String> = MaskCache::with_budget(1 << 20, 64);
    let mut key_builds = 0;
    let mut val_builds = 0;
    for _ in 0..4 {
        let mask = cache
            .get_or_insert_with_by(
                "icons/check.svg",
                || {
                    key_builds += 1;
                    String::from("icons/check.svg")
                },
                || {
                    val_builds += 1;
                    Ok(Some(Mask::Alpha(Arc::new(AlphaMask::new(dsize(4, 4), vec![9u8; 16])))))
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(mask.into_alpha().unwrap().size, dsize(4, 4));
    }
    assert_eq!(key_builds, 1);
    assert_eq!(val_builds, 1);

    // `Ok(None)` never constructs an owned key.
    let res = cache
        .get_or_insert_with_by(
            "icons/missing.svg",
            || {
                key_builds += 1;
                String::from("icons/missing.svg")
            },
            || Ok(None),
        )
        .unwrap();
    assert!(res.is_none());
    assert_eq!(key_builds, 1);
}
