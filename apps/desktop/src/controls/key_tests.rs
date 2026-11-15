//! Enter and Space on a focused control, pressed and released as a keyboard does: one press, one
//! action. (folo-ui, as GPUI, also turns the release into a click on the focused element.)

use folo_ui::{HeadlessApp, KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput, WindowHandle};

use crate::app::tests::{drive, open};
use crate::app::{FoloApp, Route};

/// One press of `key`: down, then up, each answered, a frame drawn after each as on screen.
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

/// The Playground with the demo's model on one of its images, `key`'s control focused.
fn playground_focused(key: &'static str) -> (HeadlessApp, WindowHandle<FoloApp>) {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
        this.pick_play_model("m_44c0".into(), cx);
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| w.focus(&this.fh(key, cx)));
    app.frame(window.into());
    (app, window)
}

/// [`open`] with the app's key bindings (Esc, Ctrl+Enter, the text fields'), as `main` binds them.
fn open_keyed() -> (HeadlessApp, WindowHandle<FoloApp>) {
    let (mut app, window) = open();
    app.update(|cx| {
        crate::app::bind_keys(cx);
        crate::input::bind_keys(cx);
    });
    (app, window)
}

/// The Benchmarks list with the New benchmark form open, its name field focused.
fn new_bench_open() -> (HeadlessApp, WindowHandle<FoloApp>) {
    let (mut app, window) = open_keyed();
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Benches, w, cx));
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| this.open_new_bench(w, cx));
    app.frame(window.into());
    (app, window)
}

#[test]
fn esc_closes_the_new_benchmark_form_and_stays_on_the_list() {
    let (mut app, window) = new_bench_open();
    press(&mut app, window, "escape");
    drive(&mut app, window, |this, _, _| {
        assert!(this.bench.new_bench.is_none(), "Esc closes the form");
        assert_eq!(this.route, Route::Benches, "and stays on the list");
    });
}

#[test]
fn enter_or_ctrl_enter_in_the_new_benchmark_form_creates_it() {
    for key in ["enter", "ctrl-enter"] {
        let (mut app, window) = new_bench_open();
        let name = format!("made with {key}");
        drive(&mut app, window, |this, _, cx| {
            let f = this.bench.new_bench.as_ref().unwrap();
            f.name.update(cx, |i, cx| i.set_text(name.clone(), cx));
        });
        app.frame(window.into());
        press(&mut app, window, key);
        let api = drive(&mut app, window, |this, _, _| {
            assert!(this.bench.new_bench.is_none(), "{key}: the form closed on success");
            this.store.api.clone()
        });
        let made = api.benches().unwrap().into_iter().filter(|b| b.name == name).count();
        assert_eq!(made, 1, "{key} creates it, once");
    }
}

#[test]
fn esc_on_a_benchmark_stands_a_delete_and_the_new_loop_form_down_before_going_back() {
    use crate::bench::BenchTab;
    let bench = Route::Bench("bn_5e12".into());
    let (mut app, window) = open_keyed();
    drive(&mut app, window, |this, w, cx| this.navigate(bench.clone(), w, cx));
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| {
        this.bench.tab = BenchTab::Loops;
        this.open_new_loop(cx);
        this.bench.confirm_delete = Some(("bn_5e12".into(), None));
    });
    app.frame(window.into());
    press(&mut app, window, "escape");
    drive(&mut app, window, |this, _, _| {
        assert!(this.bench.confirm_delete.is_none(), "the delete waits no more");
        assert!(this.bench.new_loop.is_some(), "one Esc, one thing");
        assert_eq!(this.route, bench);
    });
    press(&mut app, window, "escape");
    drive(&mut app, window, |this, _, _| {
        assert!(this.bench.new_loop.is_none(), "then the form closes");
        assert_eq!(this.route, bench, "on the page");
    });
    press(&mut app, window, "escape");
    assert_eq!(drive(&mut app, window, |this, _, _| this.route.clone()), Route::Benches, "then back");
}

#[test]
fn enter_opens_a_select_and_enter_again_picks_once() {
    let (mut app, window) = playground_focused("play-model");
    press(&mut app, window, "enter");
    assert_eq!(drive(&mut app, window, |this, _, _| this.menu_open), Some("play-model"), "Enter opens it");
    let before = drive(&mut app, window, |this, _, _| this.play_model.clone());
    press(&mut app, window, "down");
    press(&mut app, window, "enter");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.menu_open, None, "Enter picks and closes it");
        assert_ne!(this.play_model, before, "the highlighted model is picked");
    });
    press(&mut app, window, "space");
    assert_eq!(drive(&mut app, window, |this, _, _| this.menu_open), Some("play-model"), "so does Space");
}

#[test]
fn enter_on_a_button_acts_once() {
    let (mut app, window) = playground_focused("play-picker-toggle");
    let open_before = drive(&mut app, window, |this, _, _| this.play_picker_open);
    press(&mut app, window, "enter");
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.play_picker_open),
        !open_before,
        "one press, one toggle"
    );
}
