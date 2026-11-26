//! Regression tests for Desktop playtest UX fixes:
//! - Detail page headings stay visible (never collapse to 0 px) on dataset, run, bench,
//!   bench-entry, and model pages at both 1280×800 and 960×620, and long titles truncate cleanly
//!   without pushing header actions off-screen.
//! - Sliders and scrollbars inside `.occlude()` dialogs drag smoothly across and outside their
//!   tracks.

use std::sync::Arc;

use folo_ui::{
    AppContext, Bounds, HeadlessApp, HeadlessOptions, Modifiers, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, PlatformInput, Primitive, WindowBounds, WindowHandle, WindowOptions, point, px, size,
};

use super::tests::{drive, open};
use super::{FoloApp, Route};
use crate::api::mock::MockBackend;
use crate::theme::{Mode, Theme, theme};

fn open_at(width: f32, height: f32) -> (HeadlessApp, WindowHandle<FoloApp>) {
    let api = Arc::new(MockBackend::new(4200, true));
    let mut app = HeadlessApp::new(HeadlessOptions::default());
    let window = app.update(|cx| {
        cx.set_global(Theme::new(Mode::Dark));
        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(width), px(height)));
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
