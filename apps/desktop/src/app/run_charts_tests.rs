//! The run page's charts and Per class, driven as a person does: the keys' cursor reaches every
//! chart, and Per class beside a short samples panel paints every class whole.

use std::sync::Arc;

use folo_ui::{
    HeadlessApp, KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput, Primitive, SpriteKind, WindowHandle,
};

use super::tests::drive;
use super::{FoloApp, Route};
use crate::api::mock::MockBackend;
use crate::store::Load;

/// The demo's finished run with per-class metrics.
const RUN: &str = "run_52e8";

/// `RUN`'s page in a window wide enough for Per class beside the samples and tall enough to show
/// them, with the app's key bindings as `main` binds them.
fn run_page() -> (HeadlessApp, WindowHandle<FoloApp>) {
    use folo_ui::{AppContext, Bounds, HeadlessOptions, WindowBounds, WindowOptions, point, px, size};
    let api = Arc::new(MockBackend::new(4200, true));
    let mut app =
        HeadlessApp::new(HeadlessOptions { size: size(px(1280.), px(2400.)), ..Default::default() })
            .with_assets(crate::Assets);
    let window = app.update(|cx| {
        cx.set_global(crate::theme::Theme::new(crate::theme::Mode::Dark));
        crate::app::bind_keys(cx);
        crate::input::bind_keys(cx);
        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(1280.), px(2400.)));
        let options =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        cx.open_window(options, |window, cx| cx.new(|cx| FoloApp::new(api, true, window, cx)))
            .expect("open window")
    });
    drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Run(RUN.into()), w, cx));
    settle(&mut app, window);
    (app, window)
}

/// A few frames, each with what it asked for answered.
fn settle(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) {
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }
}

/// One press of `key`: down, then up, a frame drawn after each.
fn press(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, key: &str) {
    let keystroke = Keystroke::parse(key).unwrap();
    app.simulate_input(
        window.into(),
        PlatformInput::KeyDown(KeyDownEvent { keystroke: keystroke.clone(), is_held: false }),
    );
    app.run_until_idle();
    app.frame(window.into());
    app.simulate_input(window.into(), PlatformInput::KeyUp(KeyUpEvent { keystroke }));
    app.run_until_idle();
    app.frame(window.into());
}

#[test]
fn the_keys_cursor_reaches_every_chart_with_up_and_down() {
    let (mut app, window) = run_page();
    drive(&mut app, window, |this, w, cx| w.focus(&this.fh("run-charts", cx)));
    app.frame(window.into());
    let at = |app: &mut HeadlessApp| drive(app, window, |this, _, _| (this.chart_hover, this.chart_hover_on));
    // ← shows the cursor at the latest point, on the first chart.
    press(&mut app, window, "left");
    let (cursor, on) = at(&mut app);
    assert!(cursor.is_some(), "← shows the cursor");
    assert_eq!(on, None, "on the first chart");
    // ↓ takes it to Training loss, Precision / Recall, Learning rate, Throughput, and holds there.
    for want in ["run-loss", "run-pr", "run-lr", "run-ips", "run-ips"] {
        press(&mut app, window, "down");
        assert_eq!(at(&mut app), (cursor, Some(want)), "the same epoch, on {want}");
    }
    // ← / → step the epochs there; ↑ takes it back a chart.
    press(&mut app, window, "left");
    let (stepped, on) = at(&mut app);
    assert!(stepped < cursor, "← steps back an epoch on Throughput: {stepped:?} {cursor:?}");
    assert_eq!(on, Some("run-ips"));
    press(&mut app, window, "up");
    assert_eq!(at(&mut app), (stepped, Some("run-lr")));
    // Esc clears it.
    press(&mut app, window, "escape");
    assert_eq!(at(&mut app).0, None);
}

#[test]
fn loss_chart_toggles_switch_smoothing_and_log_scale() {
    let (mut app, window) = run_page();
    drive(&mut app, window, |this, _, _| {
        assert!(this.run_chart_opts.loss_smooth, "smoothed by default");
        assert!(this.run_chart_opts.loss_log_y, "log scale by default");
    });
    // Switch Training loss from Smooth to Raw via the segmented control.
    drive(&mut app, window, |this, w, cx| this.focus_key("run-loss-smooth", w, cx));
    app.frame(window.into());
    press(&mut app, window, "right");
    drive(&mut app, window, |this, _, _| {
        assert!(!this.run_chart_opts.loss_smooth, "right arrow switches to Raw");
    });
    // Switch Training loss from Log to Lin via the segmented control.
    drive(&mut app, window, |this, w, cx| this.focus_key("run-loss-scale", w, cx));
    app.frame(window.into());
    press(&mut app, window, "right");
    drive(&mut app, window, |this, _, _| {
        assert!(!this.run_chart_opts.loss_log_y, "right arrow switches to Linear scale");
    });
    // Back to Log scale.
    press(&mut app, window, "left");
    drive(&mut app, window, |this, _, _| {
        assert!(this.run_chart_opts.loss_log_y, "left arrow switches back to Log scale");
    });
}

/// At 1120 × 620 (the exact breakpoint where the sidebar expands and short mode is active), the
/// Training loss card's header controls fit on a single line without wrapping or clipping the title.
#[test]
fn training_loss_header_fits_without_wrapping_at_1120x620() {
    use folo_ui::{AppContext, Bounds, HeadlessOptions, WindowBounds, WindowOptions, point, px, size};
    let api = Arc::new(MockBackend::new(4200, true));
    let mut app =
        HeadlessApp::new(HeadlessOptions { size: size(px(1120.), px(620.)), ..Default::default() })
            .with_assets(crate::Assets);
    let window = app.update(|cx| {
        cx.set_global(crate::theme::Theme::new(crate::theme::Mode::Dark));
        crate::app::bind_keys(cx);
        crate::input::bind_keys(cx);
        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(1120.), px(620.)));
        let options =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        cx.open_window(options, |window, cx| cx.new(|cx| FoloApp::new(api, true, window, cx)))
            .expect("open window")
    });
    drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Run(RUN.into()), w, cx));
    settle(&mut app, window);
    // Both Training loss toggles are mounted and clickable without overflowing the 1120 px window,
    // and no glyph in the page clips vertically against its clip rect.
    drive(&mut app, window, |this, w, cx| this.focus_key("run-loss-smooth", w, cx));
    app.frame(window.into());
    let clipped: Vec<_> = app
        .scene(window.into())
        .primitives
        .iter()
        .filter_map(|p| match p {
            Primitive::MonoSprite(s) if s.kind == SpriteKind::Glyph => {
                let (b, clip) = (&s.bounds, &s.clip);
                let py = |v| f64::from(v) as f32;
                // Only inspect visible glyphs in the top band (where the primary chart headers sit).
                (py(b.origin.y) >= 40.
                    && py(b.bottom()) <= 300.
                    && (py(b.origin.y) < py(clip.origin.y) - 0.5 || py(b.bottom()) > py(clip.bottom()) + 0.5))
                .then_some((*b, *clip))
            }
            _ => None,
        })
        .collect();
    assert!(clipped.is_empty(), "primary chart headers must not clip vertically at 1120x620: {clipped:?}");
    press(&mut app, window, "right");
    drive(&mut app, window, |this, _, _| {
        assert!(!this.run_chart_opts.loss_smooth, "Smooth/Raw toggle works at 1120x620");
    });
}

#[test]
fn per_class_beside_a_short_samples_panel_paints_every_class_whole() {
    let (mut app, window) = run_page();
    // The server keeps no samples for the run (a dataset without validation images): the samples
    // panel is its header and controls only.
    drive(&mut app, window, |this, _, cx| {
        let Some(Load::Ready(s)) = this.store.samples.get_mut(RUN) else { panic!("the samples loaded") };
        s.samples.clear();
        cx.notify();
    });
    settle(&mut app, window);
    let (list, rows) = drive(&mut app, window, |this, _, _| {
        let Some(Load::Ready(d)) = this.store.run_data.get(RUN) else { panic!("the history loaded") };
        let rows = d.metrics.epochs.last().map_or(0, |e| e.val.per_class.len());
        (this.per_class_scroll.bounds(), rows)
    });
    assert!(rows >= 2, "the demo run has classes: {rows}");
    let (left, right, top) = (f32::from(list.origin.x), f32::from(list.right()), f32::from(list.origin.y));
    let row_h = 32.;
    let glyphs: Vec<_> = app
        .scene(window.into())
        .primitives
        .iter()
        .filter_map(|p| match p {
            Primitive::MonoSprite(s) if s.kind == SpriteKind::Glyph => Some(s.clone()),
            _ => None,
        })
        .collect();
    let px = |v| f64::from(v) as f32;
    for i in 0..rows.min(8) {
        let band = top + i as f32 * row_h..top + (i + 1) as f32 * row_h;
        let row: Vec<_> = glyphs
            .iter()
            .filter(|g| {
                let c = g.bounds.center();
                (left..right).contains(&px(c.x)) && band.contains(&px(c.y))
            })
            .collect();
        assert!(!row.is_empty(), "row {i} paints its class: list {list:?}");
        for g in row {
            let (b, clip) = (&g.bounds, &g.clip);
            assert!(
                px(clip.origin.y) <= px(b.origin.y) && px(b.bottom()) <= px(clip.bottom()),
                "row {i} is whole, not clipped: glyph {b:?} clip {clip:?}"
            );
        }
    }
}
