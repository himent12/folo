//! Regression tests for Sol Playtest Round 8 fixes:
//! 1. Dataset Statistics `Broken labels` tile keeps its severity breakdown (`1 warning`) and copy
//!    action + icon unclipped across the 1120 px sidebar expansion breakpoint (`1120×620` & `1119×620`).
//! 2. Benchmark Leaderboard and Plan tables compact their columns at narrow widths (`780×580`) so
//!    all five axes, row action icons, and Plan `Share` percentages remain visible and unclipped.
//! 3. Benchmark Entry Images footer wraps at narrow widths (`780×580`) so the `Show more` button
//!    stays inside the viewport instead of clipping off the right edge.

use folo_ui::{Primitive, point, px};

use super::super::tests::drive;
use super::super::{DsTab, Route};
use super::{open_at, settle};
use crate::api::{BenchEntryStatus, IssueCounts};
use crate::bench::BenchTab;
use crate::store::Load;

#[test]
fn dataset_statistics_broken_labels_tile_fits_at_1120x620_and_1119x620() {
    for width in [1119.0_f32, 1120.0, 960.0] {
        let (mut app, window) = open_at(width, 620.0);
        let ds_id = drive(&mut app, window, |this, w, cx| {
            let ds = this.store.datasets.ready().unwrap()[0].clone();
            this.navigate(Route::Dataset(ds.id.clone()), w, cx);
            ds.id
        });
        settle(&mut app, window);

        drive(&mut app, window, |this, _, cx| {
            if let Some(ds) = this.store.dataset.get_mut(&ds_id).and_then(|l| l.ready_mut())
                && let Some(stats) = ds.stats.as_mut()
            {
                stats.broken_labels = 3;
            }
            this.store.issues.insert(
                ds_id.clone(),
                Load::Ready(IssueCounts { error: 2, warning: 1, ..Default::default() }),
            );
            this.ds_tab = DsTab::Statistics;
            cx.notify();
        });
        settle(&mut app, window);

        drive(&mut app, window, |this, w, cx| {
            let fh = this.fh("ds-broken-tile", cx);
            assert!(w.is_focus_mounted(&fh), "ds-broken-tile mounted at {width}×620");
        });

        let scene = app.scene(window.into());
        // Check that sprites inside the Statistics strip (upper band, right-most Broken labels tile)
        // are not clipped by their tile's `overflow_hidden` content mask.
        let right_tile_min_x = px(width - 260.0).scale(1.0);
        let strip_max_y = px(260.0).scale(1.0);
        let tile_sprites: Vec<_> = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                Primitive::MonoSprite(s)
                    if s.bounds.origin.x >= right_tile_min_x && s.bounds.origin.y <= strip_max_y =>
                {
                    Some(s)
                }
                _ => None,
            })
            .collect();
        assert!(!tile_sprites.is_empty(), "broken labels tile painted glyphs/icons at {width}×620");
        for s in &tile_sprites {
            assert!(
                s.bounds.right() <= s.clip.right() + px(1.0).scale(1.0),
                "broken labels tile sprite clipped horizontally at {width}×620: bounds={:?}, mask={:?}",
                s.bounds,
                s.clip
            );
        }
    }
}

#[test]
fn bench_leaderboard_and_plan_tables_fit_all_columns_and_actions_at_780x580() {
    let (mut app, window) = open_at(780.0, 580.0);
    let bid = "bn_3c90";
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Bench(bid.into()), w, cx);
    });
    settle(&mut app, window);

    // 1. Leaderboard at 780×580: Localization header and row action icons must be mounted and
    // stay inside the panel clip mask (not cut off past the right edge).
    let done_id = drive(&mut app, window, |this, w, cx| {
        assert!(
            w.is_focus_mounted(&this.fh("bench-sort-localization", cx)),
            "Localization sort header mounted at 780×580"
        );
        let id = this
            .bench
            .entries
            .get(bid)
            .and_then(|l| l.ready())
            .unwrap()
            .iter()
            .find(|e| e.status == BenchEntryStatus::Done)
            .unwrap()
            .id
            .clone();
        assert!(
            w.is_focus_mounted(&this.fh(format!("bench-entry-{id}-pin"), cx)),
            "row pin action mounted at 780×580"
        );
        assert!(
            w.is_focus_mounted(&this.fh(format!("bench-entry-{id}-del"), cx)),
            "row delete action mounted at 780×580"
        );
        id
    });
    let _ = done_id;
    let scene = app.scene(window.into());
    // Rightmost row action icon sprites live near x = 700..760 inside the 780 px viewport.
    let right_edge_sprites: Vec<_> = scene
        .primitives
        .iter()
        .filter_map(|p| match p {
            Primitive::MonoSprite(s) if s.bounds.origin.y >= px(200.0).scale(1.0) => Some(s),
            _ => None,
        })
        .collect();
    assert!(
        right_edge_sprites.iter().any(|s| s.bounds.origin.x >= px(650.0).scale(1.0)),
        "leaderboard row actions must paint inside the right side of the 780 px viewport"
    );
    for s in &right_edge_sprites {
        assert!(
            s.bounds.right() <= px(780.5).scale(1.0),
            "leaderboard sprite escaped 780 px viewport: {:?}",
            s.bounds
        );
        assert!(
            s.bounds.right() <= s.clip.right() + px(1.0).scale(1.0),
            "leaderboard sprite clipped by panel mask at 780×580: bounds={:?}, mask={:?}",
            s.bounds,
            s.clip
        );
    }

    // 2. Plan tab at 780×580: Share column header and percentages must stay inside the 780 px
    // viewport and unclipped by the panel mask.
    drive(&mut app, window, |this, _, cx| {
        this.bench.tab = BenchTab::Plan;
        cx.notify();
    });
    settle(&mut app, window);
    let scene = app.scene(window.into());
    let plan_sprites: Vec<_> = scene
        .primitives
        .iter()
        .filter_map(|p| match p {
            Primitive::MonoSprite(s) if s.bounds.origin.y >= px(200.0).scale(1.0) => Some(s),
            _ => None,
        })
        .collect();
    assert!(
        plan_sprites.iter().any(|s| s.bounds.origin.x >= px(680.0).scale(1.0)),
        "plan Share percentages must paint visibly on the right of the 780 px viewport"
    );
    for s in &plan_sprites {
        assert!(
            s.bounds.right() <= px(780.5).scale(1.0),
            "plan sprite escaped 780 px viewport: {:?}",
            s.bounds
        );
        assert!(
            s.bounds.right() <= s.clip.right() + px(1.0).scale(1.0),
            "plan sprite clipped by panel mask at 780×580: bounds={:?}, mask={:?}",
            s.bounds,
            s.clip
        );
    }
}

#[test]
fn bench_entry_images_show_more_button_fits_at_780x580() {
    let (mut app, window) = open_at(780.0, 580.0);
    let bid = "bn_3c90";
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Bench(bid.into()), w, cx);
    });
    settle(&mut app, window);

    let done_id = drive(&mut app, window, |this, w, cx| {
        let id = this
            .bench
            .entries
            .get(bid)
            .and_then(|l| l.ready())
            .unwrap()
            .iter()
            .find(|e| e.status == BenchEntryStatus::Done)
            .unwrap()
            .id
            .clone();
        this.navigate(Route::BenchEntry(bid.into(), id.clone()), w, cx);
        id
    });
    settle(&mut app, window);

    // Ensure `page.total > page.images.len()` so the footer renders "Show more", then scroll to
    // the bottom of the page so the footer is visible in the 780×580 viewport.
    drive(&mut app, window, |this, _, cx| {
        if let Some(entry_imgs) = this.bench.entry_images.get_mut(&(bid.into(), done_id.clone()))
            && let Some(page) = entry_imgs.page.ready_mut()
        {
            page.total = page.images.len() as u64 + 12;
        }
        this.page_scroll.set_offset(point(px(0.), px(-2000.)));
        cx.notify();
    });
    settle(&mut app, window);

    drive(&mut app, window, |this, w, cx| {
        assert!(w.is_focus_mounted(&this.fh("entry-img-more", cx)), "Show more button mounted at 780×580");
    });

    let scene = app.scene(window.into());
    for p in &scene.primitives {
        if let Primitive::MonoSprite(s) = p
            && s.bounds.origin.y >= px(420.0).scale(1.0)
        {
            assert!(
                s.bounds.right() <= px(780.5).scale(1.0),
                "entry images footer sprite escaped 780 px viewport: {:?}",
                s.bounds
            );
            assert!(
                s.bounds.right() <= s.clip.right() + px(1.0).scale(1.0),
                "entry images footer sprite clipped horizontally: bounds={:?}, mask={:?}",
                s.bounds,
                s.clip
            );
        }
    }
}
