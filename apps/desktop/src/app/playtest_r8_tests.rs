//! Regression tests for Sol Playtest Round 8 fixes:
//! 1. Dataset Statistics `Broken labels` tile keeps its severity breakdown (`1 warning`) and copy
//!    action + icon unclipped across the 1120 px sidebar expansion breakpoint (`1120×620` & `1119×620`)
//!    and at narrow widths (`960×620`, `780×580`).
//! 2. Benchmark Leaderboard and Plan tables compact their columns at narrow widths (`780×580`,
//!    `1120×620`) so all five axes, row action icons, inline `Keep`/`Delete` confirmation buttons,
//!    and Plan `Share` percentages remain visible and non-overlapping.
//! 3. Benchmark Entry Images footer wraps at narrow widths (`780×580`) so the `Show more` button
//!    paints inside the viewport and its hit target is clickable.

use folo_ui::{MouseButton, MouseDownEvent, MouseUpEvent, PlatformInput, Primitive, point, px};

use super::super::tests::drive;
use super::super::{DsTab, Route};
use super::{open_at, settle};
use crate::api::{BenchEntryStatus, IssueCounts};
use crate::bench::BenchTab;
use crate::store::Load;

#[test]
fn dataset_statistics_broken_labels_tile_fits_at_1120x620_and_1119x620() {
    for width in [1119.0_f32, 1120.0, 960.0, 780.0] {
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
        // Anchor the strip bounds dynamically using the 4 KPI strip `vline` dividers:
        // [Images | Boxes | Classes | Empty images | Broken labels].
        let mut dividers: Vec<_> = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                Primitive::Quad(q)
                    if q.bounds.size.width == px(1.).scale(1.)
                        && q.bounds.size.height >= px(60.).scale(1.)
                        && q.bounds.origin.y >= px(100.).scale(1.)
                        && q.bounds.origin.y <= px(350.).scale(1.) =>
                {
                    Some(q)
                }
                _ => None,
            })
            .collect();
        dividers.sort_by(|a, b| a.bounds.origin.x.partial_cmp(&b.bounds.origin.x).unwrap());
        assert_eq!(dividers.len(), 4, "expected 4 KPI strip dividers at {width}×620");
        let tile_left = dividers[3].bounds.origin.x;
        let strip_top = dividers[3].bounds.origin.y;
        let strip_bottom = dividers[3].bounds.bottom();

        let tile_sprites: Vec<_> = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                Primitive::MonoSprite(s)
                    if s.bounds.origin.x >= tile_left
                        && s.bounds.origin.y >= strip_top
                        && s.bounds.bottom() <= strip_bottom =>
                {
                    Some(s)
                }
                _ => None,
            })
            .collect();
        // Verify the lower detail rows (`severity_detail` and `copy_cmd`, well below the tile
        // heading) actually painted inside the strip and that no sprite was clipped horizontally.
        let lower_detail_sprites: Vec<_> =
            tile_sprites.iter().filter(|s| s.bounds.origin.y >= strip_top + px(55.).scale(1.)).collect();
        assert!(
            lower_detail_sprites.len() >= 15,
            "expected severity + copy-action sprites inside Broken labels tile at {width}×620, got {}",
            lower_detail_sprites.len()
        );
        for s in &tile_sprites {
            assert!(
                s.bounds.right() <= s.clip.right() + px(1.0).scale(1.0),
                "broken labels tile sprite clipped horizontally at {width}×620: bounds={:?}, clip={:?}",
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
    let scene = app.scene(window.into());
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
            "leaderboard sprite clipped by panel mask at 780×580: bounds={:?}, clip={:?}",
            s.bounds,
            s.clip
        );
    }

    // 1b. Inline delete confirmation (`Keep` + `Delete`) in compact leaderboard must fit inside
    // the actions column without overlapping the `Loc.` score cell to its left.
    drive(&mut app, window, |this, _, cx| {
        this.bench.confirm_delete = Some((bid.into(), Some(done_id.clone())));
        cx.notify();
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        assert!(
            w.is_focus_mounted(&this.fh(format!("bench-entry-{done_id}-keep"), cx)),
            "Keep button mounted in compact leaderboard"
        );
        assert!(
            w.is_focus_mounted(&this.fh(format!("bench-entry-{done_id}-del-ok"), cx)),
            "Delete button mounted in compact leaderboard"
        );
    });
    let scene = app.scene(window.into());
    // Locate the `Keep` and `Delete` button quads (height 28 px) in the leaderboard body (`y >= 300`).
    let confirm_quads: Vec<_> = scene
        .primitives
        .iter()
        .filter_map(|p| match p {
            Primitive::Quad(q)
                if q.bounds.size.height == px(28.0).scale(1.0)
                    && q.bounds.origin.y >= px(300.0).scale(1.0)
                    && q.bounds.origin.y <= px(450.0).scale(1.0)
                    && q.bounds.origin.x >= px(560.0).scale(1.0) =>
            {
                Some(q)
            }
            _ => None,
        })
        .collect();
    assert!(
        confirm_quads.len() >= 2,
        "expected Keep and Delete button quads in compact row, got {}",
        confirm_quads.len()
    );
    let row_top =
        confirm_quads.iter().map(|q| q.bounds.origin.y).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let row_bottom =
        confirm_quads.iter().map(|q| q.bounds.bottom()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let leftmost_confirm =
        confirm_quads.iter().map(|q| q.bounds.origin.x).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    // On the exact same row, the `Loc.` score glyphs sit to the left of the actions column (`x < 620`).
    let loc_right = scene
        .primitives
        .iter()
        .filter_map(|p| match p {
            Primitive::MonoSprite(s)
                if s.bounds.origin.y >= row_top
                    && s.bounds.bottom() <= row_bottom
                    && s.bounds.origin.x < px(620.0).scale(1.0) =>
            {
                Some(s.bounds.right())
            }
            _ => None,
        })
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .expect("scored row Loc. sprites painted");
    assert!(
        leftmost_confirm > loc_right,
        "Keep/Delete confirmation ({leftmost_confirm:?}) must not overlap Loc. score ({loc_right:?})"
    );
    for q in &confirm_quads {
        assert!(
            q.bounds.right() <= px(780.5).scale(1.0),
            "confirmation button escaped 780 px viewport: {:?}",
            q.bounds
        );
    }

    // 2. Plan tab at 780×580: Share column header and percentages must stay inside the 780 px
    // viewport and unclipped by the panel mask.
    drive(&mut app, window, |this, _, cx| {
        this.bench.confirm_delete = None;
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
            "plan sprite clipped by panel mask at 780×580: bounds={:?}, clip={:?}",
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

    // Ensure `page.total > page.images.len()` so the footer renders "Show more", settle layout so
    // `page_scroll.max_offset()` is known, then scroll directly to the bottom.
    drive(&mut app, window, |this, _, cx| {
        if let Some(entry_imgs) = this.bench.entry_images.get_mut(&(bid.into(), done_id.clone()))
            && let Some(page) = entry_imgs.page.ready_mut()
        {
            page.total = page.images.len() as u64 + 12;
        }
        cx.notify();
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        let max_y = this.page_scroll.max_offset().height;
        assert!(max_y > px(0.), "entry images page should be scrollable at 780×580");
        this.page_scroll.set_offset(point(px(0.), -max_y));
        cx.notify();
    });
    settle(&mut app, window);

    drive(&mut app, window, |this, w, cx| {
        assert!(w.is_focus_mounted(&this.fh("entry-img-more", cx)), "Show more button mounted at 780×580");
    });

    let scene = app.scene(window.into());
    // Locate the painted `Show more` button quad (height 28 px, right side of the footer) and
    // verify both its quad and its sprites stay strictly inside the 780×580 viewport.
    let btn_quad = scene
        .primitives
        .iter()
        .filter_map(|p| match p {
            Primitive::Quad(q)
                if q.bounds.size.height == px(28.0).scale(1.0)
                    && q.bounds.size.width >= px(80.0).scale(1.0)
                    && q.bounds.origin.y >= px(480.0).scale(1.0)
                    && q.bounds.origin.x >= px(580.0).scale(1.0) =>
            {
                Some(q)
            }
            _ => None,
        })
        .next()
        .expect("Show more button quad must paint visibly inside the 780×580 footer");
    assert!(
        btn_quad.bounds.right() <= px(780.0).scale(1.0),
        "Show more button quad escaped 780 px viewport: {:?}",
        btn_quad.bounds
    );
    assert!(
        btn_quad.bounds.bottom() <= px(580.0).scale(1.0),
        "Show more button quad escaped 580 px viewport height: {:?}",
        btn_quad.bounds
    );

    let btn_sprites: Vec<_> = scene
        .primitives
        .iter()
        .filter_map(|p| match p {
            Primitive::MonoSprite(s)
                if s.bounds.origin.x >= btn_quad.bounds.origin.x
                    && s.bounds.origin.y >= btn_quad.bounds.origin.y
                    && s.bounds.bottom() <= btn_quad.bounds.bottom() =>
            {
                Some(s)
            }
            _ => None,
        })
        .collect();
    assert!(!btn_sprites.is_empty(), "Show more button label/icon sprites must paint");
    for s in &btn_sprites {
        assert!(
            s.bounds.right() <= s.clip.right() + px(1.0).scale(1.0),
            "Show more button sprite clipped horizontally: bounds={:?}, clip={:?}",
            s.bounds,
            s.clip
        );
    }

    // Clicking the painted center of `Show more` must hit the button and request the next page.
    let unit = px(1.).scale(1.);
    let center = btn_quad.bounds.center();
    let click_pt = point(px(center.x / unit), px(center.y / unit));
    app.simulate_input(
        window.into(),
        PlatformInput::MouseDown(MouseDownEvent {
            button: MouseButton::Left,
            position: click_pt,
            modifiers: Default::default(),
            click_count: 1,
            first_mouse: false,
        }),
    );
    app.simulate_input(
        window.into(),
        PlatformInput::MouseUp(MouseUpEvent {
            button: MouseButton::Left,
            position: click_pt,
            modifiers: Default::default(),
            click_count: 1,
        }),
    );
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let entry_imgs = this.bench.entry_images.get(&(bid.into(), done_id.clone())).unwrap();
        assert!(
            entry_imgs.more || entry_imgs.page.ready().is_some_and(|p| p.images.len() > 24),
            "clicking Show more at its painted center must trigger loading more images"
        );
    });
}
