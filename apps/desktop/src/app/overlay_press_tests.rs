//! Presses on an overlay that a click opened (#350): the click that opens a dialog is often the
//! first half of a double-click, and the second press lands on the dialog, not on the button. An
//! overlay that a key opened takes every press, whatever click it continues.

use folo_ui::{HeadlessApp, WindowHandle};

use super::select_driving::{click_n_at, press};
use super::tests::{drive, open};
use super::{FoloApp, Route, ServerMode};

/// A spot on the page clear of every centred dialog and the palette: the backdrop once one is up,
/// blank page (below the Runs table, beside the Datasets cards) before.
const BACKDROP: (f32, f32) = (1200., 700.);
/// Generate synthetic, in the Datasets header (1280 × 800): under the title bar, so the backdrop
/// once the form is up.
const SYNTH_BUTTON: (f32, f32) = (1050., 73.);

fn click(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, (x, y): (f32, f32), count: usize) {
    click_n_at(app, window, x, y, count);
}

/// The Datasets page of a local server (Generate synthetic is offered).
fn datasets() -> (HeadlessApp, WindowHandle<FoloApp>) {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.server_mode = ServerMode::Local;
        this.navigate(Route::Datasets, w, cx);
    });
    app.frame(window.into());
    (app, window)
}

fn synth_open(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> bool {
    drive(app, window, |this, _, _| this.synth.is_some())
}

#[test]
fn the_second_press_of_the_double_click_that_opened_a_dialog_leaves_it_open() {
    let (mut app, window) = datasets();
    click(&mut app, window, SYNTH_BUTTON, 1);
    assert!(synth_open(&mut app, window), "the click on Generate synthetic opens the form");
    // The second press of the double-click lands on the backdrop, and the triple click's third.
    click(&mut app, window, SYNTH_BUTTON, 2);
    assert!(synth_open(&mut app, window), "the double-click keeps the form");
    click(&mut app, window, SYNTH_BUTTON, 3);
    assert!(synth_open(&mut app, window), "and a triple click");
    // A deliberate press on the backdrop still dismisses it.
    click(&mut app, window, BACKDROP, 1);
    assert!(!synth_open(&mut app, window), "a later press dismisses");
    // Opened again by a double-click: the new one keeps out the rest of its own click too.
    click(&mut app, window, SYNTH_BUTTON, 1);
    click(&mut app, window, SYNTH_BUTTON, 2);
    assert!(synth_open(&mut app, window), "a reopened dialog stays too");
}

/// #382: a double-click on the backdrop (or on Cancel) closes the dialog with its first press; the
/// second went on to the page under it, here Generate synthetic, which opened the form again.
#[test]
fn the_second_press_of_the_double_click_that_closed_a_dialog_does_not_reach_the_page() {
    let (mut app, window) = datasets();
    drive(&mut app, window, |this, w, cx| this.open_synth(w, cx));
    app.frame(window.into());
    assert!(synth_open(&mut app, window), "opened by a command: it takes every press");
    click(&mut app, window, SYNTH_BUTTON, 1);
    assert!(!synth_open(&mut app, window), "the backdrop over Generate synthetic dismisses it");
    click(&mut app, window, SYNTH_BUTTON, 2);
    assert!(!synth_open(&mut app, window), "the rest of that double-click is not the button's");
    // A click of its own there is.
    click(&mut app, window, SYNTH_BUTTON, 1);
    assert!(synth_open(&mut app, window), "a later click on Generate synthetic opens the form");
}

#[test]
fn a_dialog_opened_by_a_key_takes_the_next_press_of_an_earlier_click() {
    for key in ["enter", "space"] {
        let (mut app, window) = datasets();
        // A click on the page, then the focused button activated by its key within the
        // double-click time: the next press on the backdrop is the platform's second of that click.
        click(&mut app, window, BACKDROP, 1);
        drive(&mut app, window, |this, w, cx| w.focus(&this.fh("ds-synth", cx)));
        app.frame(window.into());
        press(&mut app, window, key);
        assert!(synth_open(&mut app, window), "{key} on Generate synthetic opens the form");
        click(&mut app, window, BACKDROP, 2);
        assert!(!synth_open(&mut app, window), "a press on the backdrop dismisses what {key} opened");
    }
}

#[test]
fn the_palette_opened_by_its_shortcut_takes_the_next_press_of_an_earlier_click() {
    let (mut app, window) = open();
    app.update(|cx| {
        super::bind_keys(cx);
        crate::input::bind_keys(cx);
    });
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Runs, w, cx));
    app.frame(window.into());
    // A click on the blank page under the table, Ctrl+K, then the same spot clicked again within
    // the double-click time (the platform reports it as the click's second press).
    click(&mut app, window, BACKDROP, 1);
    press(&mut app, window, "ctrl-k");
    assert!(drive(&mut app, window, |this, _, _| this.palette.is_some()), "Ctrl+K opens the palette");
    click(&mut app, window, BACKDROP, 2);
    assert!(drive(&mut app, window, |this, _, _| this.palette.is_none()), "the press dismisses it");
}

#[test]
fn a_double_click_that_opens_a_confirmation_cannot_confirm_it() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| this.navigate(Route::Run("run_52e8".into()), w, cx));
    app.frame(window.into());
    // Delete leads the run header's action buttons (leftmost secondary button in the header band).
    let raised = drive(&mut app, window, |_, _, cx| crate::theme::theme(cx).bg_raised);
    let sp = |v: folo_ui::ScaledPixels| f64::from(v) as f32;
    let delete_btn = app
        .scene(window.into())
        .primitives
        .iter()
        .filter_map(|p| match p {
            folo_ui::Primitive::Quad(q)
                if q.background.solid == raised
                    && sp(q.bounds.origin.y) >= 40.
                    && sp(q.bounds.origin.y) <= 140.
                    && sp(q.bounds.origin.x) >= 240.
                    && sp(q.bounds.size.height) >= 24.
                    && sp(q.bounds.size.height) <= 36. =>
            {
                Some(q.bounds)
            }
            _ => None,
        })
        .min_by(|a, b| {
            (sp(a.origin.y).round() as i32)
                .cmp(&(sp(b.origin.y).round() as i32))
                .then_with(|| sp(a.origin.x).total_cmp(&sp(b.origin.x)))
        })
        .map(|b| (sp(b.origin.x) + sp(b.size.width) / 2., sp(b.origin.y) + sp(b.size.height) / 2.))
        .expect("Delete button painted in run header");
    click(&mut app, window, delete_btn, 1);
    assert!(drive(&mut app, window, |this, _, _| this.confirm_delete.is_some()), "Delete asks");
    // The confirm button: rightmost in the footer under the body (20 px in, 14 px down).
    let body = drive(&mut app, window, |this, _, _| this.dialog_scroll.bounds());
    let confirm = (f32::from(body.right()) - 20. - 8., f32::from(body.bottom()) + 14. + 8.);
    // The second press of the double-click that opened it lands on Delete run: nothing happens.
    click(&mut app, window, confirm, 2);
    drive(&mut app, window, |this, _, _| {
        assert!(this.confirm_delete.is_some(), "still asking");
        assert!(this.store.run("run_52e8").is_some(), "nothing deleted");
    });
    // A click of its own there confirms (the spot is the button).
    click(&mut app, window, confirm, 1);
    drive(&mut app, window, |this, _, _| {
        assert!(this.confirm_delete.is_none());
        assert!(this.store.run("run_52e8").is_none(), "deleted");
    });
}

fn input(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, (x, y): (f32, f32), down: bool, count: usize) {
    use folo_ui::{Modifiers, MouseButton, MouseDownEvent, MouseUpEvent, PlatformInput, point, px};
    let (position, modifiers, button) = (point(px(x), px(y)), Modifiers::default(), MouseButton::Left);
    let ev = if down {
        PlatformInput::MouseDown(MouseDownEvent {
            button,
            position,
            modifiers,
            click_count: count,
            first_mouse: false,
        })
    } else {
        PlatformInput::MouseUp(MouseUpEvent { button, position, modifiers, click_count: count })
    };
    app.simulate_input(window.into(), ev);
    app.run_until_idle();
}

/// Review round 1: a press on the dialog, Escape closes it while the button is held, and the
/// release comes before the next frame. The release was taken for the one that closed it, so the
/// next press of that click was kept from the page. It closed nothing: the page takes that press.
#[test]
fn a_dialog_closed_by_a_key_while_the_button_is_held_leaves_the_next_press_to_the_page() {
    let (mut app, window) = datasets();
    drive(&mut app, window, |this, w, cx| this.open_synth(w, cx));
    app.frame(window.into());
    // The dialog's middle: on its panel, not the backdrop.
    input(&mut app, window, (640., 400.), true, 1);
    app.frame(window.into());
    assert!(synth_open(&mut app, window), "a press on the panel keeps it");
    // Escape (its action: the tests' app binds no keys), with no frame drawn after it.
    drive(&mut app, window, |this, w, cx| this.dismiss(w, cx));
    assert!(!synth_open(&mut app, window), "Escape closes it");
    input(&mut app, window, (640., 400.), false, 1);
    app.frame(window.into());
    click(&mut app, window, SYNTH_BUTTON, 2);
    assert!(synth_open(&mut app, window), "the press reaches Generate synthetic");
}

/// Review round 1: both presses of a double-click on the backdrop before a frame is drawn. The
/// backdrop still in the last frame took the second as well and closed the dialog again, moving
/// focus from the control it went back to (Generate synthetic) to the page. Stopped at once.
#[test]
fn the_second_press_before_a_frame_does_not_close_the_dialog_twice() {
    let (mut app, window) = datasets();
    drive(&mut app, window, |this, w, cx| w.focus(&this.fh("ds-synth", cx)));
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| this.open_synth(w, cx));
    app.frame(window.into());
    input(&mut app, window, BACKDROP, true, 1);
    input(&mut app, window, BACKDROP, false, 1);
    assert!(!synth_open(&mut app, window), "the backdrop dismissed it");
    let back =
        |app: &mut HeadlessApp| drive(app, window, |this, w, cx| this.fh("ds-synth", cx).is_focused(w));
    assert!(back(&mut app), "focus back on Generate synthetic");
    input(&mut app, window, BACKDROP, true, 2);
    input(&mut app, window, BACKDROP, false, 2);
    assert!(back(&mut app), "the second press closed nothing more");
}

/// The title bar's control area at `x` (its middle row): the drag band, a caption button, or none.
fn caption_at(
    app: &mut HeadlessApp,
    window: WindowHandle<FoloApp>,
    x: f32,
) -> Option<folo_ui::WindowControlArea> {
    use folo_ui::{Modifiers, MouseMoveEvent, PlatformInput, point, px};
    let position = point(px(x), px(super::titlebar::TITLEBAR / 2.));
    let ev = MouseMoveEvent { position, pressed_button: None, modifiers: Modifiers::default() };
    app.simulate_input(window.into(), PlatformInput::MouseMove(ev));
    drive(app, window, |_, w, _| w.hit_test_window_control())
}

/// Review round 2: a breadcrumb that opens a page shared the caption's hit area, so the press
/// guard let the rest of the double-click that closed a dialog through to it. It is not the
/// caption: its press is the app's, and that press is kept from it.
#[test]
fn the_double_click_that_closed_a_dialog_does_not_reach_a_breadcrumb() {
    use folo_ui::WindowControlArea;
    let (mut app, window) = open();
    let page = Route::Dataset("ds_3b91".into());
    let to_page = |app: &mut HeadlessApp| {
        let page = page.clone();
        drive(app, window, move |this, w, cx| {
            this.server_mode = ServerMode::Local;
            this.navigate(page, w, cx);
        });
        app.frame(window.into());
        app.run_until_idle();
        app.frame(window.into());
    };
    to_page(&mut app);
    // Right of the sidebar, the first place on the band that is not the band: the "Datasets" crumb.
    let from = crate::theme::size::SIDEBAR as i32 + 2;
    let band = (from..1280)
        .step_by(2)
        .find(|&x| caption_at(&mut app, window, x as f32) == Some(WindowControlArea::Drag));
    let band = band.expect("the title bar's drag band");
    let crumb = (band..1280).step_by(2).find(|&x| caption_at(&mut app, window, x as f32).is_none());
    let crumb = (crumb.expect("a crumb on the band") + 4) as f32;
    let route = |app: &mut HeadlessApp| drive(app, window, |this, _, _| this.route.clone());
    // Round 3: in keyboard mode, a press on the crumb leaves it, as one on the band does.
    drive(&mut app, window, |this, _, _| this.kb = true);
    click(&mut app, window, (crumb, super::titlebar::TITLEBAR / 2.), 1);
    assert_eq!(route(&mut app), Route::Datasets, "the crumb opens Datasets");
    assert!(!drive(&mut app, window, |this, _, _| this.kb), "a press on the crumb leaves keyboard mode");
    to_page(&mut app);
    drive(&mut app, window, |this, w, cx| this.open_synth(w, cx));
    app.frame(window.into());
    assert!(synth_open(&mut app, window));
    // The backdrop just under the crumb dismisses the form; the double-click's second press lands
    // on the crumb.
    click(&mut app, window, (crumb, super::titlebar::TITLEBAR + 12.), 1);
    assert!(!synth_open(&mut app, window), "the backdrop dismisses it");
    click(&mut app, window, (crumb, super::titlebar::TITLEBAR / 2.), 2);
    assert_eq!(route(&mut app), page, "the rest of that double-click is not the crumb's");
}
