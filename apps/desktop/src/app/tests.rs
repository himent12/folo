//! The app driven headless over the frozen demo data: what a person does in a dialog or on a page,
//! and what the state says afterwards. Nothing is drawn; each test is a few updates.

use std::sync::Arc;

use folo_ui::{AppContext, Bounds, HeadlessApp, HeadlessOptions, WindowBounds, WindowHandle, WindowOptions};
use folo_ui::{Context, Window, point, px, size};

use super::FoloApp;
use crate::api::mock::MockBackend;
use crate::theme::{Mode, Theme};

/// The app on the demo data (frozen: no live clock), its lists loaded.
pub(crate) fn open() -> (HeadlessApp, WindowHandle<FoloApp>) {
    let mut app = HeadlessApp::new(HeadlessOptions::default());
    let window = app.update(|cx| {
        cx.set_global(Theme::new(Mode::Dark));
        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(1280.), px(800.)));
        let options =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        cx.open_window(options, |window, cx| {
            cx.new(|cx| FoloApp::new(Arc::new(MockBackend::new(4200, true)), true, window, cx))
        })
        .expect("open window")
    });
    drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
    (app, window)
}

/// Runs `f` on the app, then lets every fetch it started answer.
pub(crate) fn drive<R>(
    app: &mut HeadlessApp,
    window: WindowHandle<FoloApp>,
    f: impl FnOnce(&mut FoloApp, &mut Window, &mut Context<FoloApp>) -> R,
) -> R {
    let r = app.update(|cx| window.update(cx, f).expect("the window is open"));
    app.run_until_idle();
    r
}

/// The New run dialog's name field.
pub(crate) fn run_name(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> String {
    drive(app, window, |this, _, cx| this.new_run.as_ref().map(|f| f.name.read(cx).text().to_string()))
        .unwrap_or_default()
}

#[test]
fn the_suggested_run_name_follows_the_choices_until_typed() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| this.open_new_run(None, w, cx));
    let first = run_name(&mut app, window);
    assert!(first.contains("-s-"), "{first}");
    // Architecture n: the untouched name follows.
    drive(&mut app, window, |this, _, cx| {
        this.new_run.as_mut().unwrap().arch = 0;
        this.resuggest_run_name(cx);
    });
    assert_eq!(run_name(&mut app, window), first.replace("-s-", "-n-"));
    // Another dataset: so does its name.
    drive(&mut app, window, |this, _, cx| {
        this.new_run.as_mut().unwrap().dataset = 1;
        this.resuggest_run_name(cx);
    });
    let other = run_name(&mut app, window);
    assert!(other != first.replace("-s-", "-n-") && other.contains("-n-"), "{other}");
    // Typed: it stays as typed whatever is picked next.
    drive(&mut app, window, |this, _, cx| {
        let input = this.new_run.as_ref().unwrap().name.clone();
        input.update(cx, |i, cx| i.set_text("my run", cx));
    });
    drive(&mut app, window, |this, _, cx| {
        this.new_run.as_mut().unwrap().arch = 3;
        this.resuggest_run_name(cx);
    });
    assert_eq!(run_name(&mut app, window), "my run");
}
