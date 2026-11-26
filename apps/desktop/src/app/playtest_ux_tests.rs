//! Regression tests for Desktop playtest UX fixes:
//! - Detail page headings stay visible (never collapse to 0 px) on dataset, run, bench,
//!   bench-entry, and model pages at both 1280×800 and 960×620, and long titles truncate cleanly
//!   without pushing header actions off-screen.
//! - Sliders and scrollbars inside `.occlude()` dialogs drag smoothly across and outside their
//!   tracks.

use std::sync::Arc;

use folo_ui::{
    AppContext, Bounds, HeadlessApp, HeadlessOptions, KeyDownEvent, Keystroke, Modifiers, MouseButton,
    MouseDownEvent, MouseMoveEvent, MouseUpEvent, PlatformInput, Primitive, WindowBounds, WindowHandle,
    WindowOptions, point, px, size,
};

use super::tests::{drive, open};
use super::{FoloApp, Route};
use crate::api::mock::MockBackend;
use crate::theme::{Mode, Theme, theme};

fn open_at(width: f32, height: f32) -> (HeadlessApp, WindowHandle<FoloApp>) {
    let api = Arc::new(MockBackend::new(4200, true));
    let window_size = size(px(width), px(height));
    let mut app = HeadlessApp::new(HeadlessOptions { size: window_size, ..Default::default() });
    let window = app.update(|cx| {
        cx.set_global(Theme::new(Mode::Dark));
        let bounds = Bounds::new(point(px(0.), px(0.)), window_size);
        let options =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        cx.open_window(options, |window, cx| cx.new(|cx| FoloApp::new(api, true, window, cx)))
            .expect("open window")
    });
    drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
    (app, window)
}

fn sp(v: folo_ui::ScaledPixels) -> f32 {
    f64::from(v) as f32
}

/// Counts primary-ink glyphs drawn in the detail header title band (`y` in `48..105`, `x >= min_x`),
/// along with their horizontal span `(min_x, max_x)`.
fn title_band_glyphs(
    app: &mut HeadlessApp,
    window: WindowHandle<FoloApp>,
    min_x: f32,
    max_x: f32,
) -> (usize, f32, f32) {
    let primary = drive(app, window, |_, _, cx| theme(cx).text_primary);
    let mut count = 0usize;
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for p in &app.scene(window.into()).primitives {
        if let Primitive::MonoSprite(g) = p
            && g.kind == folo_ui::SpriteKind::Glyph
            && g.color == primary
            && sp(g.bounds.origin.y) >= 48.
            && sp(g.bounds.origin.y) <= 105.
            && sp(g.bounds.origin.x) >= min_x
            && sp(g.bounds.origin.x) <= max_x
        {
            count += 1;
            lo = lo.min(sp(g.bounds.origin.x));
            hi = hi.max(sp(g.bounds.origin.x) + sp(g.bounds.size.width));
        }
    }
    (count, lo, hi)
}

fn settle(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) {
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }
}

#[test]
fn detail_page_headings_paint_across_all_detail_routes() {
    for (w, h, min_x, max_x) in [(1280., 800., 240., 760.), (960., 620., 64., 540.)] {
        let (mut app, window) = open_at(w, h);
        let (ds_id, run_id, model_id) = drive(&mut app, window, |this, _, _| {
            let ds = this.store.datasets.ready().unwrap()[0].id.clone();
            let run = this.store.runs.ready().unwrap()[0].id.clone();
            let model = this.store.models.ready().unwrap()[0].id.clone();
            (ds, run, model)
        });
        let bench_id = "bn_3c90".to_string();
        let entry_id = "be_3c01".to_string();

        let routes = [
            ("dataset", Route::Dataset(ds_id)),
            ("run", Route::Run(run_id)),
            ("model", Route::Model(model_id)),
            ("bench", Route::Bench(bench_id.clone())),
            ("bench-entry", Route::BenchEntry(bench_id, entry_id)),
        ];

        for (label, route) in routes {
            drive(&mut app, window, |this, win, cx| this.navigate(route.clone(), win, cx));
            settle(&mut app, window);
            let (count, lo, hi) = title_band_glyphs(&mut app, window, min_x, max_x);
            assert!(
                count >= 4 && hi - lo >= 30.,
                "{label} at {w}x{h} detail heading must render visible title glyphs (got {count} glyphs, span {lo:.1}..{hi:.1})"
            );
        }
    }

    // Long model/run titles truncate cleanly without pushing right-hand action buttons off-screen.
    let (mut app, window) = open_at(960., 620.);
    let model_id = drive(&mut app, window, |this, _, _| {
        let m = &mut this.store.models.ready_mut().unwrap()[0];
        m.name = "very-long-model-name-that-must-truncate-cleanly-on-a-960px-window-without-clipping-actions"
            .into();
        m.id.clone()
    });
    drive(&mut app, window, |this, win, cx| this.navigate(Route::Model(model_id), win, cx));
    settle(&mut app, window);
    // Find both header action buttons (`Export` and `Try in Playground`) in the header band.
    let (raised, accent) = drive(&mut app, window, |_, _, cx| {
        let t = theme(cx);
        (t.bg_raised, t.accent_fill())
    });
    let mut action_left = f32::INFINITY;
    let mut action_right = f32::NEG_INFINITY;
    for p in &app.scene(window.into()).primitives {
        if let Primitive::Quad(q) = p
            && (q.background.solid == raised || q.background.solid == accent)
            && sp(q.bounds.origin.y) >= 48.
            && sp(q.bounds.origin.y) <= 105.
            && sp(q.bounds.origin.x) >= 400.
            && sp(q.bounds.size.height) >= 24.
        {
            action_left = action_left.min(sp(q.bounds.origin.x));
            action_right = action_right.max(sp(q.bounds.origin.x) + sp(q.bounds.size.width));
        }
    }
    assert!(
        action_left.is_finite() && action_right <= 960.,
        "both header actions fit inside the 960px window ({action_left:.1}..{action_right:.1})"
    );
    // Check all primary-ink glyphs left of the action buttons: the 90-char model name must paint
    // truncated with an ellipsis rather than overflowing into the actions.
    let (count, lo, hi) = title_band_glyphs(&mut app, window, 64., action_left);
    assert!(
        (8..75).contains(&count) && hi - lo >= 80.,
        "long model title renders truncated visible glyphs ({count}, {lo:.1}..{hi:.1})"
    );
    assert!(hi < action_left, "long model title stops left of header actions ({hi:.1} < {action_left:.1})");
}

#[test]
fn new_run_dialog_slider_drags_across_and_outside_its_track() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Runs, w, cx);
        this.open_new_run(None, w, cx);
    });
    settle(&mut app, window);
    // Scroll the New Run dialog down so the Schedule section (nr-epochs slider) is in view.
    drive(&mut app, window, |this, _, cx| {
        let max_y = this.dialog_scroll.max_offset().height;
        this.dialog_scroll.set_offset(point(px(0.), -max_y));
        cx.notify();
    });
    app.frame(window.into());

    let (bounds, before) = drive(&mut app, window, |this, _, _| {
        let b = *this.slider_bounds.borrow().get("nr-epochs").expect("nr-epochs slider bounds");
        let epochs = this.new_run.as_ref().expect("new_run open").epochs;
        (b, epochs)
    });
    let cy = f32::from(bounds.top()) + f32::from(bounds.size.height) * 0.5;
    let start_x = f32::from(bounds.left()) + f32::from(bounds.size.width) * 0.15;
    let mid_x = f32::from(bounds.left()) + f32::from(bounds.size.width) * 0.75;
    let far_x = f32::from(bounds.right()) + 80.;

    // Press on the slider track.
    app.simulate_input(
        window.into(),
        PlatformInput::MouseDown(MouseDownEvent {
            button: MouseButton::Left,
            position: point(px(start_x), px(cy)),
            modifiers: Modifiers::default(),
            click_count: 1,
            first_mouse: false,
        }),
    );
    app.run_until_idle();
    app.frame(window.into());
    let after_down = drive(&mut app, window, |this, _, _| this.new_run.as_ref().unwrap().epochs);
    assert_ne!(after_down, before, "clicking nr-epochs sets initial drag value");

    // Drag rightward along the track while holding the button.
    app.simulate_input(
        window.into(),
        PlatformInput::MouseMove(MouseMoveEvent {
            position: point(px(mid_x), px(cy)),
            pressed_button: Some(MouseButton::Left),
            modifiers: Modifiers::default(),
        }),
    );
    app.run_until_idle();
    app.frame(window.into());
    let after_mid = drive(&mut app, window, |this, _, _| this.new_run.as_ref().unwrap().epochs);
    assert!(
        after_mid > after_down + 100.,
        "dragging nr-epochs right must advance value (down={after_down}, mid={after_mid})"
    );

    // Drag past the right edge of the track / dialog: clamps to max (300).
    app.simulate_input(
        window.into(),
        PlatformInput::MouseMove(MouseMoveEvent {
            position: point(px(far_x), px(cy + 40.)),
            pressed_button: Some(MouseButton::Left),
            modifiers: Modifiers::default(),
        }),
    );
    app.run_until_idle();
    app.frame(window.into());
    let after_far = drive(&mut app, window, |this, _, _| this.new_run.as_ref().unwrap().epochs);
    assert_eq!(after_far, 300., "dragging past right edge clamps to slider max");

    // Release outside the dialog: drag ends cleanly and further mouse moves do not change epochs.
    app.simulate_input(
        window.into(),
        PlatformInput::MouseUp(MouseUpEvent {
            button: MouseButton::Left,
            position: point(px(far_x), px(cy + 40.)),
            modifiers: Modifiers::default(),
            click_count: 1,
        }),
    );
    app.run_until_idle();
    app.simulate_input(
        window.into(),
        PlatformInput::MouseMove(MouseMoveEvent {
            position: point(px(start_x), px(cy)),
            pressed_button: None,
            modifiers: Modifiers::default(),
        }),
    );
    app.run_until_idle();
    let (still_300, dragging) =
        drive(&mut app, window, |this, _, _| (this.new_run.as_ref().unwrap().epochs, this.drag.is_some()));
    assert!(!dragging, "releasing mouse ends slider drag");
    assert_eq!(still_300, 300., "moving pointer after release does not drag slider");
}

#[test]
fn new_run_dialog_scrollbar_thumb_drags_inside_occluding_dialog() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Runs, w, cx);
        this.open_new_run(None, w, cx);
    });
    settle(&mut app, window);

    let geo = drive(&mut app, window, |this, _, _| {
        crate::ui::ThumbGeo::of(&this.dialog_scroll).expect("New Run dialog body scrolls")
    });
    let track_x = f32::from(this_dialog_right(&mut app, window)) - 6.;
    let start_y = f32::from(geo.y) + f32::from(geo.thumb) * 0.5;
    let end_y = f32::from(geo.top) + f32::from(geo.h) - 10.;

    app.simulate_input(
        window.into(),
        PlatformInput::MouseDown(MouseDownEvent {
            button: MouseButton::Left,
            position: point(px(track_x), px(start_y)),
            modifiers: Modifiers::default(),
            click_count: 1,
            first_mouse: false,
        }),
    );
    app.run_until_idle();
    app.simulate_input(
        window.into(),
        PlatformInput::MouseMove(MouseMoveEvent {
            position: point(px(track_x + 30.), px(end_y)),
            pressed_button: Some(MouseButton::Left),
            modifiers: Modifiers::default(),
        }),
    );
    app.run_until_idle();
    app.frame(window.into());

    let scrolled_y = drive(&mut app, window, |this, _, _| f32::from(-this.dialog_scroll.offset().y));
    assert!(scrolled_y > 40., "dragging dialog scrollbar thumb must scroll body (got {scrolled_y})");

    app.simulate_input(
        window.into(),
        PlatformInput::MouseUp(MouseUpEvent {
            button: MouseButton::Left,
            position: point(px(track_x + 30.), px(end_y)),
            modifiers: Modifiers::default(),
            click_count: 1,
        }),
    );
    app.run_until_idle();
    assert!(drive(&mut app, window, |this, _, _| this.scroll_drag.is_none()), "release ends scroll drag");
}

fn this_dialog_right(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> folo_ui::Pixels {
    drive(app, window, |this, _, _| this.dialog_scroll.bounds().right())
}

fn press_key(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, key: &str, key_char: Option<&str>) {
    app.simulate_input(
        window.into(),
        PlatformInput::KeyDown(KeyDownEvent {
            keystroke: Keystroke {
                modifiers: Modifiers::default(),
                key: key.to_string(),
                key_char: key_char.map(str::to_string),
            },
            is_held: false,
        }),
    );
    app.run_until_idle();
    app.frame(window.into());
}

#[test]
fn palette_query_edit_resets_scroll_to_top() {
    let (mut app, window) = open_at(960., 620.);
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| this.open_palette(w, cx));
    settle(&mut app, window);

    // Jump to the bottom of the palette list with End so the list scrolls down.
    press_key(&mut app, window, "end", None);
    let (sel_end, scroll_end) = drive(&mut app, window, |this, _, _| {
        let p = this.palette.as_ref().expect("palette open");
        (p.selected, f32::from(-p.scroll.offset().y))
    });
    assert!(sel_end > 0, "End selects the last palette item");
    assert!(scroll_end > 20., "End scrolls the palette list down (got {scroll_end})");

    // Typing a query resets selection to 0 AND scrolls back to the top so row 0 is visible.
    press_key(&mut app, window, "a", Some("a"));
    let (sel_after, scroll_after, query) = drive(&mut app, window, |this, _, _| {
        let p = this.palette.as_ref().expect("palette open");
        (p.selected, f32::from(-p.scroll.offset().y), p.query.clone())
    });
    assert_eq!(query, "a");
    assert_eq!(sel_after, 0, "query edit resets selection to the first match");
    assert_eq!(scroll_after, 0., "query edit scrolls the palette back to the top");

    // Same-frame race: a paging key queues `scroll_to_item` and a query edit fires before the next
    // paint. Pick an off-screen index (`25`) that remains in-bounds and below the fold in the
    // rebuilt empty-query list so the test genuinely fails if `set_offset` leaves `active_item` queued.
    let target = app.update(|cx| {
        window
            .update(cx, |this, _, cx| {
                let p = this.palette.as_mut().expect("palette open");
                let ix = 25.min(p.scroll.children_count().saturating_sub(1));
                assert!(ix >= 20, "palette list has enough off-screen rows (got {ix})");
                p.scroll.scroll_to_item(ix);
                let input = p.input.clone();
                input.update(cx, |i, cx| i.set_text("", cx));
                ix
            })
            .unwrap()
    });
    app.frame(window.into());
    let (sel_same_frame, scroll_same_frame, q_same_frame, new_count, offscreen) =
        drive(&mut app, window, |this, _, _| {
            let p = this.palette.as_ref().expect("palette open");
            let vp_bottom = p.scroll.bounds().bottom();
            let item_bottom = p.scroll.bounds_for_item(target).map(|b| b.bottom());
            (
                p.selected,
                f32::from(-p.scroll.offset().y),
                p.query.clone(),
                p.scroll.children_count(),
                item_bottom.is_some_and(|b| b > vp_bottom),
            )
        });
    assert_eq!(q_same_frame, "");
    assert!(new_count > target, "target index {target} must stay in bounds (len {new_count})");
    assert!(
        offscreen,
        "target index {target} must lie below the palette viewport so stale scroll_to_item would scroll"
    );
    assert_eq!(sel_same_frame, 0);
    assert_eq!(
        scroll_same_frame, 0.,
        "pending scroll_to_item must not override same-frame query scroll reset"
    );
}

#[test]
fn cloud_connect_locks_provider_while_submitting_and_scopes_callback() {
    let (mut app, window) = open_at(960., 620.);
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Settings, w, cx));
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| this.open_connect(w, cx));
    settle(&mut app, window);

    let provider_count =
        drive(&mut app, window, |this, _, _| this.cloud.providers.ready().map_or(0, Vec::len));
    assert!(provider_count >= 2, "mock cloud has at least two providers");

    // 1. While `submitting` is true, the provider segmented control disables other providers so
    // keyboard navigation (`Right`) and direct `set_connect_provider` calls cannot switch away.
    drive(&mut app, window, |this, w, cx| {
        let form = this.cloud.connect.as_mut().unwrap();
        form.provider = 0;
        form.submitting = true;
        let fh = this.fh("cc-provider", cx);
        w.focus(&fh);
        cx.notify();
    });
    app.frame(window.into());
    press_key(&mut app, window, "right", None);
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(
            this.cloud.connect.as_ref().unwrap().provider,
            0,
            "Right arrow is disabled while submitting"
        );
        this.set_connect_provider(1, cx);
        assert_eq!(
            this.cloud.connect.as_ref().unwrap().provider,
            0,
            "set_connect_provider is ignored while submitting"
        );
        this.cloud.connect.as_mut().unwrap().submitting = false;
    });

    // 2. Real `submit_connect` with a rejected RunPod key locks `set_connect_provider` immediately
    // and scopes its async callback to the submitted provider (even if provider changed before
    // delivery, RunPod's error never lands on Remote's form).
    app.update(|cx| {
        window
            .update(cx, |this, w, cx| {
                let first = this.cloud.connect.as_ref().unwrap().fields[0].clone();
                first.update(cx, |i, cx| i.set_text("rpa_bad_1234", cx));
                this.submit_connect(w, cx);
                assert!(this.cloud.connect.as_ref().unwrap().submitting);
                this.set_connect_provider(1, cx);
                assert_eq!(
                    this.cloud.connect.as_ref().unwrap().provider,
                    0,
                    "set_connect_provider stays blocked right after submit_connect"
                );
                // Force a provider mismatch to verify the callback's defense-in-depth guard.
                this.cloud.connect.as_mut().unwrap().provider = 1;
            })
            .unwrap();
    });
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let form = this.cloud.connect.as_ref().expect("connect dialog remains open");
        assert!(
            form.error.is_none(),
            "stale provider 0 error must not land on provider 1 (got {:?})",
            form.error
        );
    });
}

#[test]
fn lost_banner_does_not_cover_or_intercept_overview_actions_at_960x620() {
    let (mut app, window) = open_at(960., 620.);
    settle(&mut app, window);
    // Simulate a lost server connection so the reconnect banner renders on Overview at 960×620.
    drive(&mut app, window, |this, _, cx| {
        this.store
            .note_reachability(Some(&crate::api::ApiError::new("server_unreachable", "connection refused")));
        assert!(this.lost_banner_shown(), "reconnect banner must be shown");
        cx.notify();
    });
    settle(&mut app, window);

    let (warn_bg, accent_bg) = drive(&mut app, window, |_, _, cx| {
        let t = theme(cx);
        (t.warning_muted, t.accent_solid)
    });
    let mut banner_bottom = f32::NEG_INFINITY;
    let mut new_run_rect: Option<(f32, f32, f32, f32)> = None;
    for p in &app.scene(window.into()).primitives {
        if let Primitive::Quad(q) = p {
            let x = sp(q.bounds.origin.x);
            let y = sp(q.bounds.origin.y);
            let w = sp(q.bounds.size.width);
            let h = sp(q.bounds.size.height);
            if q.background == warn_bg.into() && w > 400. && (24.0..=36.0).contains(&h) {
                banner_bottom = banner_bottom.max(y + h);
            }
            if q.background == accent_bg.into() && w >= 60. && (24.0..=36.0).contains(&h) && y < 140. {
                new_run_rect = Some((x, y, w, h));
            }
        }
    }
    let (bx, by, bw, bh) = new_run_rect.expect("Overview New run button painted");
    assert!(banner_bottom.is_finite(), "lost-connection warning strip painted");
    assert!(
        banner_bottom <= by + 0.5,
        "lost-connection strip (bottom {banner_bottom:.1}) must not overlap New run button (top {by:.1})"
    );

    // Clicking near the very top edge of the New run button opens the dialog (not intercepted).
    let click_pt = point(px(bx + bw * 0.5), px(by + 4.));
    assert!(bh >= 24.);
    app.simulate_input(
        window.into(),
        PlatformInput::MouseDown(MouseDownEvent {
            button: MouseButton::Left,
            position: click_pt,
            modifiers: Modifiers::default(),
            click_count: 1,
            first_mouse: false,
        }),
    );
    app.simulate_input(
        window.into(),
        PlatformInput::MouseUp(MouseUpEvent {
            button: MouseButton::Left,
            position: click_pt,
            modifiers: Modifiers::default(),
            click_count: 1,
        }),
    );
    app.run_until_idle();
    app.frame(window.into());
    assert!(
        drive(&mut app, window, |this, _, _| this.new_run.is_some()),
        "top of New run button remains clickable while reconnect banner is shown"
    );
}

#[test]
fn f6_closes_open_dropdown_menu_when_moving_focus_away() {
    let (mut app, window) = open_at(1280., 800.);
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
    });
    settle(&mut app, window);

    // Open the Model dropdown via keyboard.
    drive(&mut app, window, |this, w, cx| {
        let fh = this.fh("play-model", cx);
        w.focus(&fh);
    });
    app.frame(window.into());
    press_key(&mut app, window, "enter", None);
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.menu_open),
        Some("play-model"),
        "Enter opens Model dropdown"
    );

    // F6 (`cycle_region`) moves focus away to the sidebar and closes the open dropdown.
    drive(&mut app, window, |this, w, cx| this.cycle_region(true, w, cx));
    app.frame(window.into());
    let (menu, nav_focused) =
        drive(&mut app, window, |this, w, cx| (this.menu_open, this.fh("nav-Playground", cx).is_focused(w)));
    assert!(menu.is_none(), "F6 closes any open select dropdown");
    assert!(nav_focused, "F6 moves focus to the Playground sidebar item");
}

#[test]
fn page_down_and_page_up_scroll_new_run_dialog_and_playground_side_rail() {
    let (mut app, window) = open_at(960., 620.);
    settle(&mut app, window);

    // 1. New Run modal at 960×620: focusing Dataset and pressing Page Down scrolls dialog_scroll.
    drive(&mut app, window, |this, w, cx| {
        this.open_new_run(None, w, cx);
    });
    settle(&mut app, window);
    let max_dialog = drive(&mut app, window, |this, w, cx| {
        let fh = this.fh("nr-dataset", cx);
        w.focus(&fh);
        f32::from(this.dialog_scroll.max_offset().height)
    });
    assert!(max_dialog > 40., "New Run dialog overflows at 960×620 (max {max_dialog})");
    app.frame(window.into());
    press_key(&mut app, window, "pagedown", None);
    let y_down = drive(&mut app, window, |this, _, _| f32::from(-this.dialog_scroll.offset().y));
    assert!(y_down > 40., "Page Down scrolls New Run dialog body (got {y_down})");
    press_key(&mut app, window, "pageup", None);
    let y_up = drive(&mut app, window, |this, _, _| f32::from(-this.dialog_scroll.offset().y));
    assert_eq!(y_up, 0., "Page Up returns New Run dialog body to top");

    // 2. Playground at 960×620: focusing Open image and pressing Page Down scrolls play_side_scroll.
    drive(&mut app, window, |this, w, cx| {
        this.close_new_run(w, cx);
        this.play_model = Some("m_44c0".into());
        this.play_dataset_image("ds_3b91", "img_03417", cx);
        this.navigate(Route::Playground, w, cx);
    });
    settle(&mut app, window);
    let max_side = drive(&mut app, window, |this, w, cx| {
        let fh = this.fh("play-open", cx);
        w.focus(&fh);
        f32::from(this.play_side_scroll.max_offset().height)
    });
    assert!(max_side > 40., "Playground side rail overflows at 960×620 (max {max_side})");
    app.frame(window.into());
    press_key(&mut app, window, "pagedown", None);
    let side_down = drive(&mut app, window, |this, _, _| f32::from(-this.play_side_scroll.offset().y));
    assert!(side_down > 40., "Page Down scrolls Playground side rail (got {side_down})");
    press_key(&mut app, window, "pageup", None);
    let side_up = drive(&mut app, window, |this, _, _| f32::from(-this.play_side_scroll.offset().y));
    assert_eq!(side_up, 0., "Page Up returns Playground side rail to top");

    // 3. When a dropdown menu is open, Page Down does not scroll behind it.
    drive(&mut app, window, |this, w, cx| {
        let fh = this.fh("play-model", cx);
        w.focus(&fh);
    });
    app.frame(window.into());
    press_key(&mut app, window, "enter", None);
    assert_eq!(drive(&mut app, window, |this, _, _| this.menu_open), Some("play-model"));
    press_key(&mut app, window, "pagedown", None);
    assert_eq!(
        drive(&mut app, window, |this, _, _| f32::from(-this.play_side_scroll.offset().y)),
        0.,
        "Page Down does not scroll Playground rail behind an open dropdown"
    );

    // 4. Submitting New Run while its Dataset dropdown is open closes `menu_open` cleanly so the
    // new run dashboard stays pageable with Page Down.
    drive(&mut app, window, |this, w, cx| {
        this.menu_open = None;
        this.open_new_run(None, w, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, w, cx| {
        let fh = this.fh("nr-dataset", cx);
        w.focus(&fh);
    });
    app.frame(window.into());
    press_key(&mut app, window, "enter", None);
    assert_eq!(drive(&mut app, window, |this, _, _| this.menu_open), Some("nr-dataset"));
    drive(&mut app, window, |this, w, cx| this.submit_new_run(w, cx));
    settle(&mut app, window);
    let (route, menu, max_page) = drive(&mut app, window, |this, _, _| {
        (this.route.clone(), this.menu_open, f32::from(this.page_scroll.max_offset().height))
    });
    assert!(matches!(route, Route::Run(_)), "submitting New Run navigates to the run page");
    assert!(menu.is_none(), "submitting New Run clears the closed dialog's dropdown menu");
    if max_page > 40. {
        press_key(&mut app, window, "pagedown", None);
        let run_y = drive(&mut app, window, |this, _, _| f32::from(-this.page_scroll.offset().y));
        assert!(run_y > 40., "Page Down still scrolls the run page after submitting New Run (got {run_y})");
    }
}
