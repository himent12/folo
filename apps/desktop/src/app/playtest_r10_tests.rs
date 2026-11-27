//! Regression tests for Sol Playtest Round 10 fixes:
//! 1. Playground stage toolbar sheds secondary readouts on narrow stages (`780×700`) so `Fit` and
//!    `Zoom in` stay unclipped and clickable inside the stage panel.
//! 2. Benchmark Entry KPI strip switches to two balanced 3-tile rows below 960 px content width
//!    (`780×700`, `1120×760`) so axis labels never collide or overflow the right panel border,
//!    while keeping a single 6-tile row on wide viewports (`1440×900`).
//! 3. Overview live-run hardware footer wraps cleanly at `780×700` so the Memory readout is never
//!    clipped past the card's right edge.
//! 4. Benchmark Compare side-by-side entry headers stay on a single ellipsized line (no 1-char
//!    orphan wraps) with aligned swatches across all columns.

use folo_ui::{MouseButton, MouseDownEvent, MouseUpEvent, PlatformInput, Primitive, point, px};

use super::super::Route;
use super::super::tests::drive;
use super::{open_at, settle, sp};

#[test]
fn playground_toolbar_keeps_zoom_controls_unclipped_at_780x700() {
    for (width, height) in [(780.0_f32, 700.0_f32), (1120.0, 760.0), (1440.0, 900.0)] {
        let (mut app, window) = open_at(width, height);
        drive(&mut app, window, |this, w, cx| {
            this.navigate(Route::Playground, w, cx);
            this.play_model = Some("m_44c0".into());
            this.play_dataset_image("ds_3b91", "img_03417", cx);
        });
        settle(&mut app, window);

        let stage_bounds = drive(&mut app, window, |this, w, cx| {
            assert!(
                w.is_focus_mounted(&this.fh("play-zoom-out", cx)),
                "play-zoom-out mounted at {width}×{height}"
            );
            assert!(
                w.is_focus_mounted(&this.fh("play-zoom-in", cx)),
                "play-zoom-in mounted at {width}×{height}"
            );
            assert!(this.play_pred.ready().is_some(), "prediction ready at {width}×{height}");
            this.play_view.get().expect("stage canvas bounds painted")
        });

        let scene = app.scene(window.into());
        // Toolbar sits directly above the stage canvas (40 px tall) within the stage's horizontal span.
        let stage_left = f32::from(stage_bounds.origin.x) - 16.0;
        let stage_right = f32::from(stage_bounds.right()) + 16.0;
        let bar_bottom = f32::from(stage_bounds.origin.y) - 16.0;
        let bar_top = bar_bottom - 42.0;

        let toolbar_sprites: Vec<_> = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                Primitive::MonoSprite(s)
                    if sp(s.bounds.origin.y) >= bar_top
                        && sp(s.bounds.bottom()) <= bar_bottom + 2.0
                        && sp(s.bounds.origin.x) >= stage_left - 2.0
                        && sp(s.bounds.origin.x) <= stage_right + 2.0 =>
                {
                    Some(s)
                }
                _ => None,
            })
            .collect();
        assert!(!toolbar_sprites.is_empty(), "expected toolbar sprites inside stage at {width}×{height}");
        for s in &toolbar_sprites {
            assert!(
                sp(s.bounds.right()) <= sp(s.clip.right()) + 1.0,
                "playground toolbar sprite clipped at {width}×{height}: bounds={:?}, clip={:?}",
                s.bounds,
                s.clip
            );
            assert!(
                sp(s.bounds.right()) <= stage_right + 2.0,
                "playground toolbar sprite escaped stage right edge ({stage_right}) at {width}×{height}: {:?}",
                s.bounds
            );
        }

        // Click `play-zoom-in` (rightmost 32 px control inside the toolbar's 12 px right padding).
        let zoom_in_pt = point(px(stage_right - 12.0 - 16.0), px(bar_top + 20.0));
        let click = |app: &mut folo_ui::HeadlessApp, pt| {
            app.simulate_input(
                window.into(),
                PlatformInput::MouseDown(MouseDownEvent {
                    button: MouseButton::Left,
                    position: pt,
                    modifiers: Default::default(),
                    click_count: 1,
                    first_mouse: false,
                }),
            );
            app.simulate_input(
                window.into(),
                PlatformInput::MouseUp(MouseUpEvent {
                    button: MouseButton::Left,
                    position: pt,
                    modifiers: Default::default(),
                    click_count: 1,
                }),
            );
            app.frame(window.into());
        };
        click(&mut app, zoom_in_pt);
        drive(&mut app, window, |this, w, cx| {
            assert!(
                this.play_zoom > 1.0,
                "clicking play-zoom-in inside stage toolbar at {width}×{height} must zoom in (zoom={})",
                this.play_zoom
            );
            assert!(
                w.is_focus_mounted(&this.fh("play-fit", cx)),
                "play-fit button mounted once zoomed at {width}×{height}"
            );
        });

        // Click `play-fit` (48 px wide, 2 px left of the 32 px zoom-in button) to reset zoom.
        let fit_pt = point(px(stage_right - 12.0 - 32.0 - 2.0 - 24.0), px(bar_top + 20.0));
        click(&mut app, fit_pt);
        drive(&mut app, window, |this, _, _| {
            assert!(
                (this.play_zoom - 1.0).abs() < 1e-3,
                "clicking play-fit inside stage toolbar at {width}×{height} must reset zoom (zoom={})",
                this.play_zoom
            );
        });
    }
}

#[test]
fn bench_entry_kpi_strip_wraps_cleanly_without_overflow_at_780x700() {
    for (width, height, expect_two_rows) in
        [(780.0_f32, 700.0_f32, true), (1120.0, 760.0, true), (1440.0, 900.0, false)]
    {
        let (mut app, window) = open_at(width, height);
        drive(&mut app, window, |this, w, cx| {
            this.navigate(Route::BenchEntry("bn_3c90".into(), "be_3c05".into()), w, cx);
        });
        settle(&mut app, window);

        let scene = app.scene(window.into());
        // No text sprite in the entry body may clip horizontally or escape the viewport.
        let body_sprites: Vec<_> = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                Primitive::MonoSprite(s)
                    if sp(s.bounds.origin.y) >= 140.0 && sp(s.bounds.origin.y) <= 420.0 =>
                {
                    Some(s)
                }
                _ => None,
            })
            .collect();
        assert!(!body_sprites.is_empty(), "expected KPI strip sprites at {width}×{height}");
        for s in &body_sprites {
            assert!(
                sp(s.bounds.right()) <= width + 0.5,
                "bench entry sprite escaped viewport at {width}×{height}: {:?}",
                s.bounds
            );
            assert!(
                sp(s.bounds.right()) <= sp(s.clip.right()) + 1.0,
                "bench entry sprite clipped horizontally at {width}×{height}: bounds={:?}, clip={:?}",
                s.bounds,
                s.clip
            );
        }

        // Check the rendered height of the KPI panel (first panel below the page header):
        // 2-row layout is >= 180 px tall; 1-row layout is <= 140 px tall.
        let kpi_panel = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                Primitive::Quad(q)
                    if sp(q.bounds.size.width) >= 500.0
                        && sp(q.bounds.origin.y) >= 120.0
                        && sp(q.bounds.origin.y) <= 220.0
                        && sp(q.bounds.size.height) >= 80.0 =>
                {
                    Some(q)
                }
                _ => None,
            })
            .min_by(|a, b| sp(a.bounds.origin.y).partial_cmp(&sp(b.bounds.origin.y)).unwrap())
            .expect("KPI strip panel quad");
        let kpi_h = sp(kpi_panel.bounds.size.height);
        if expect_two_rows {
            assert!(
                kpi_h >= 180.0,
                "expected 2-row KPI strip height >= 180 px at {width}×{height}, got {kpi_h:.1}"
            );
        } else {
            assert!(
                kpi_h <= 140.0,
                "expected single-row KPI strip height <= 140 px at {width}×{height}, got {kpi_h:.1}"
            );
        }
    }
}

#[test]
fn overview_system_footer_wraps_without_clipping_at_780x700() {
    for (width, height) in [(780.0_f32, 700.0_f32), (1440.0, 900.0)] {
        let (mut app, window) = open_at(width, height);
        settle(&mut app, window);

        let scene = app.scene(window.into());
        // The system footer contains 3 hardware meters (GPU, CPU, Memory), each with a 48×3 progress track.
        let meter_tracks: Vec<_> = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                Primitive::Quad(q)
                    if (sp(q.bounds.size.width) - 48.0).abs() <= 1.0
                        && (sp(q.bounds.size.height) - 3.0).abs() <= 1.0 =>
                {
                    Some(q)
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            meter_tracks.len(),
            3,
            "expected 3 hardware meter tracks (GPU, CPU, Memory) in system footer at {width}×{height}"
        );
        let track_min_y = meter_tracks.iter().map(|q| sp(q.bounds.origin.y)).fold(f32::INFINITY, f32::min);
        let track_max_y =
            meter_tracks.iter().map(|q| sp(q.bounds.origin.y)).fold(f32::NEG_INFINITY, f32::max);
        if width <= 800.0 {
            assert!(
                track_max_y - track_min_y >= 14.0,
                "expected system footer meters to wrap onto 2 lines at {width}×{height} (track y {:.1}..{:.1})",
                track_min_y,
                track_max_y
            );
        } else {
            assert!(
                track_max_y - track_min_y <= 1.5,
                "expected system footer meters on 1 line at {width}×{height} (track y {:.1}..{:.1})",
                track_min_y,
                track_max_y
            );
        }

        let footer_sprites: Vec<_> = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                Primitive::MonoSprite(s)
                    if sp(s.bounds.origin.y) >= track_min_y - 14.0
                        && sp(s.bounds.origin.y) <= track_max_y + 14.0 =>
                {
                    Some(s)
                }
                _ => None,
            })
            .collect();
        assert!(!footer_sprites.is_empty(), "expected system footer text sprites at {width}×{height}");
        for s in &footer_sprites {
            assert!(
                sp(s.bounds.right()) <= width + 0.5,
                "system footer sprite escaped viewport at {width}×{height}: {:?}",
                s.bounds
            );
            assert!(
                sp(s.bounds.right()) <= sp(s.clip.right()) + 1.0,
                "system footer sprite clipped horizontally at {width}×{height}: bounds={:?}, clip={:?}",
                s.bounds,
                s.clip
            );
        }
    }
}

#[test]
fn bench_compare_headers_stay_single_line_and_unclipped_at_780x700() {
    for (width, height) in [(780.0_f32, 700.0_f32), (1440.0, 900.0)] {
        let (mut app, window) = open_at(width, height);
        drive(&mut app, window, |this, w, cx| {
            this.navigate(Route::BenchCompare("bn_3c90".into()), w, cx);
        });
        settle(&mut app, window);

        let scene = app.scene(window.into());
        // The 4 Side-by-side column header swatches are 12×4 rounded quads in the table header below
        // the selected-entry chips (y >= 340 at 780×700, y >= 260 at 1440×900).
        let mut swatches: Vec<_> = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                Primitive::Quad(q)
                    if (sp(q.bounds.size.width) - 12.0).abs() <= 1.0
                        && (sp(q.bounds.size.height) - 4.0).abs() <= 1.0
                        && sp(q.bounds.origin.y) >= 260.0
                        && sp(q.bounds.origin.y) <= 480.0 =>
                {
                    Some(q)
                }
                _ => None,
            })
            .collect();
        swatches.sort_by(|a, b| sp(a.bounds.origin.y).partial_cmp(&sp(b.bounds.origin.y)).unwrap());
        // The Side-by-side table header is the first group of 4 swatches below the chips (or exact 4).
        let head_swatches: Vec<_> = swatches.into_iter().take(4).collect();
        assert_eq!(head_swatches.len(), 4, "expected 4 compare header swatches at {width}×{height}");
        let y0 = sp(head_swatches[0].bounds.origin.y);
        for sw in &head_swatches {
            assert!(
                (sp(sw.bounds.origin.y) - y0).abs() <= 1.5,
                "compare header swatches must share one baseline at {width}×{height}: y0={y0}, sw={:?}",
                sw.bounds
            );
        }

        // Entry-name glyphs on the header line sit right beside the swatches (within ±12 px of y0)
        // and must stay on a single line (no 2-line orphan wrap like `t` or `x`) without clipping.
        let name_sprites: Vec<_> = scene
            .primitives
            .iter()
            .filter_map(|p| match p {
                Primitive::MonoSprite(s)
                    if sp(s.bounds.origin.y) >= y0 - 12.0 && sp(s.bounds.origin.y) <= y0 + 18.0 =>
                {
                    Some(s)
                }
                _ => None,
            })
            .collect();
        assert!(!name_sprites.is_empty(), "expected compare header name sprites at {width}×{height}");
        let min_y = name_sprites.iter().map(|s| sp(s.bounds.origin.y)).fold(f32::INFINITY, f32::min);
        let max_bottom = name_sprites.iter().map(|s| sp(s.bounds.bottom())).fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_bottom - min_y <= 18.0,
            "compare header entry names must stay single-line (no orphan wrap) at {width}×{height}: span {:.1}..{:.1}",
            min_y,
            max_bottom
        );
        for s in &name_sprites {
            assert!(
                sp(s.bounds.right()) <= sp(s.clip.right()) + 1.0,
                "compare header sprite clipped at {width}×{height}: bounds={:?}, clip={:?}",
                s.bounds,
                s.clip
            );
        }
    }
}
