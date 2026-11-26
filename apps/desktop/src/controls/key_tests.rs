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
    drive(&mut app, window, |this, w, cx| {
        assert!(this.bench.new_bench.is_none(), "Esc closes the form");
        assert_eq!(this.route, Route::Benches, "and stays on the list");
        // Review round 1: the focus fell to the app root, so Enter did nothing and Tab restarted.
        assert!(this.fh("bench-new", cx).is_focused(w), "the focus is back on New benchmark");
    });
}

#[test]
fn ctrl_enter_under_the_palette_creates_no_benchmark() {
    // Review round 1: the palette isn't a modal, and the bindings run before its own Enter.
    let (mut app, window) = new_bench_open();
    let name = "typed under the palette";
    drive(&mut app, window, |this, w, cx| {
        let f = this.bench.new_bench.as_ref().unwrap();
        f.name.update(cx, |i, cx| i.set_text(name.to_string(), cx));
        this.open_palette(w, cx);
    });
    app.frame(window.into());
    assert!(drive(&mut app, window, |this, _, _| this.palette.is_some()), "the palette is open");
    press(&mut app, window, "ctrl-enter");
    let api = drive(&mut app, window, |this, _, _| this.store.api.clone());
    let made = api.benches().unwrap().into_iter().filter(|b| b.name == name).count();
    assert_eq!(made, 0, "the form under the palette sent nothing");
}

#[test]
fn ctrl_enter_on_a_design_proposal_neither_uses_it_nor_asks_again() {
    // A proposal replaces the plan: it is used with its own button, after a look at it.
    let bench = Route::Bench("bn_7a04".into());
    let (mut app, window) = open_keyed();
    drive(&mut app, window, |this, w, cx| this.navigate(bench.clone(), w, cx));
    app.frame(window.into());
    let plan = |api: &std::sync::Arc<dyn crate::api::Backend>| {
        let b = api.benches().unwrap().into_iter().find(|b| b.id == "bn_7a04").unwrap();
        format!("{:?}", b.plan)
    };
    let before = drive(&mut app, window, |this, w, cx| {
        this.bench.tab = crate::bench::BenchTab::Plan;
        this.open_design("bn_7a04", w, cx);
        let req = crate::api::DesignBench {
            agent: crate::api::LabelAgentId::Claude,
            model: None,
            effort: None,
            note: String::new(),
        };
        let mut proposal = this.store.api.design_bench("bn_7a04", &req).unwrap();
        // Marked: the mock answers every ask alike, so only a mark shows that no new answer
        // replaced it (review round 2).
        proposal.plan.rationale = Some("the proposal on screen".into());
        this.bench.design.as_mut().unwrap().proposal = Some(proposal.clone());
        (this.store.api.clone(), proposal)
    });
    let (before, shown) = (plan(&before.0), before.1);
    app.frame(window.into());
    // With no agent, asking does nothing anyway: the test would prove nothing.
    assert!(drive(&mut app, window, |this, _, _| !this.bench_agents().is_empty()), "an agent can be asked");
    press(&mut app, window, "ctrl-enter");
    let api = drive(&mut app, window, |this, _, _| {
        let f = this.bench.design.as_ref().expect("the proposal stays on screen");
        assert!(!f.asking && !f.applying && f.error.is_none(), "nothing was sent");
        assert_eq!(f.proposal.as_ref(), Some(&shown), "no new ask replaced the proposal");
        this.store.api.clone()
    });
    assert_eq!(plan(&api), before, "the plan is unchanged");
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

#[test]
fn modified_chords_do_not_nudge_sliders_or_activate_buttons() {
    let (mut app, window) = open();
    app.update(|cx| {
        crate::app::bind_keys(cx);
        crate::input::bind_keys(cx);
    });
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
    });
    app.frame(window.into());

    // Alt+Enter on a focused button must not activate it.
    drive(&mut app, window, |this, w, cx| w.focus(&this.fh("play-picker-toggle", cx)));
    app.frame(window.into());
    let picker_before = drive(&mut app, window, |this, _, _| this.play_picker_open);
    press(&mut app, window, "alt-enter");
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.play_picker_open),
        picker_before,
        "Alt+Enter must not toggle focused button"
    );

    // Unbound modified arrow chords (Ctrl+Right, Alt+Up) reach the slider's own key handler and
    // must NOT nudge it; Alt+Left also navigates back cleanly.
    drive(&mut app, window, |this, w, cx| w.focus(&this.fh("play-conf", cx)));
    app.frame(window.into());
    let conf_before = drive(&mut app, window, |this, _, _| this.play_conf);
    press(&mut app, window, "ctrl-right");
    press(&mut app, window, "alt-up");
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.play_conf),
        conf_before,
        "Ctrl+Right / Alt+Up must not nudge play-conf slider"
    );
    press(&mut app, window, "alt-left");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.play_conf, conf_before, "Alt+Left must not nudge play-conf slider");
        assert_eq!(this.route, Route::Overview, "Alt+Left navigates back instead of being swallowed");
    });
}

#[test]
fn modal_waiting_scrim_keeps_tab_focus_trapped() {
    let (mut app, window) = open();
    app.update(|cx| {
        crate::app::bind_keys(cx);
        crate::input::bind_keys(cx);
    });
    // Put inspector into waiting state (images still Loading for a dataset with no loaded images),
    // so only the loading scrim is mounted and no `insp-*` focus stops exist yet.
    drive(&mut app, window, |this, w, cx| {
        this.store.images.insert("loading-ds".into(), crate::store::Load::Loading);
        this.inspector = Some(crate::app::Inspector {
            dataset: "loading-ds".into(),
            index: 0,
            sample: None,
            zoom: 1.0,
            pan: (0., 0.),
            boxes: true,
            labels: true,
            hover_box: None,
            hover_from_stage: false,
            focused_box: None,
            drag_from: None,
            want: None,
            shown: None,
        });
        w.focus(&this.inspector_focus);
        cx.notify();
    });
    app.frame(window.into());
    press(&mut app, window, "tab");
    drive(&mut app, window, |this, w, _| {
        assert!(
            this.inspector_focus.is_focused(w),
            "Tab inside an empty loading modal stays on inspector_focus instead of leaking to page controls"
        );
    });
}

#[test]
fn model_not_found_offers_all_models_recovery_and_select_clears_stale_trigger() {
    let (mut app, window) = open();
    app.update(|cx| {
        crate::app::bind_keys(cx);
        crate::input::bind_keys(cx);
    });
    // Opening select A measures its trigger; dismissing via the bound Escape (`Dismiss`) action or
    // Tab clears stale `menu_trigger` so the next select never inherits old bounds.
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| w.focus(&this.fh("play-model", cx)));
    app.frame(window.into());
    press(&mut app, window, "enter");
    assert!(drive(&mut app, window, |this, _, _| this.menu_trigger.get().is_some()));
    press(&mut app, window, "escape");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.menu_open, None);
        assert!(
            this.menu_trigger.get().is_none(),
            "Dismiss clears menu_trigger so the next select never inherits stale bounds"
        );
    });
    press(&mut app, window, "enter");
    assert!(drive(&mut app, window, |this, _, _| this.menu_trigger.get().is_some()));
    press(&mut app, window, "tab");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.menu_open, None);
        assert!(this.menu_trigger.get().is_none(), "Tab clears menu_trigger too");
    });

    // Navigating to a deleted/missing model shows an "All models" button that returns to Models.
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Model("does-not-exist".into()), w, cx);
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| this.focus_key("model-all", w, cx));
    app.frame(window.into());
    press(&mut app, window, "enter");
    assert_eq!(drive(&mut app, window, |this, _, _| this.route.clone()), Route::Models);
}

#[test]
fn tiny_charts_do_not_panic_or_emit_nan_hover() {
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Arc;

    use folo_ui::{
        AppContext, Bounds, HeadlessApp, HeadlessOptions, IntoElement, Modifiers, MouseMoveEvent,
        ParentElement, PlatformInput, Styled, WindowBounds, WindowOptions, div, point, px, size,
    };

    let saw_nan = Rc::new(Cell::new(false));
    let saw_nan_line = saw_nan.clone();
    let saw_nan_hist = saw_nan.clone();

    let mut app = HeadlessApp::new(HeadlessOptions { size: size(px(12.), px(12.)), ..Default::default() });
    let win = app.update(|cx| {
        let t = crate::theme::Theme::new(crate::theme::Mode::Dark);
        cx.set_global(t);
        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(12.), px(12.)));
        let options =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        cx.open_window(options, |_, cx| {
            cx.new(|_| {
                struct TinyCharts {
                    saw_nan_line: Rc<Cell<bool>>,
                    saw_nan_hist: Rc<Cell<bool>>,
                }
                impl folo_ui::Render for TinyCharts {
                    fn render(
                        &mut self,
                        _window: &mut folo_ui::Window,
                        cx: &mut folo_ui::Context<Self>,
                    ) -> impl IntoElement {
                        let t = crate::theme::theme(cx);
                        let l_flag = self.saw_nan_line.clone();
                        let h_flag = self.saw_nan_hist.clone();
                        let mut spec = crate::charts::LineChart::new(
                            "tiny-line",
                            vec![crate::charts::Series::new(
                                "loss",
                                t.accent_solid,
                                Arc::new(vec![(0., 1.), (1., 0.5)]),
                            )],
                        );
                        spec.height = 10.;
                        spec.hover = Some(0.5);
                        spec.on_hover = Some(Rc::new(move |f, _, _| {
                            if f.is_some_and(f32::is_nan) {
                                l_flag.set(true);
                            }
                        }));
                        div().size_full().child(crate::charts::line_chart(&t, spec)).child(
                            crate::charts::histogram(
                                &t,
                                "tiny-hist",
                                Arc::new(vec![1, 5, 2]),
                                Arc::new(vec!["a".into(), "b".into(), "c".into()]),
                                t.accent_solid,
                                10.,
                                Some(0.5),
                                Some(Rc::new(move |f, _, _| {
                                    if f.is_some_and(f32::is_nan) {
                                        h_flag.set(true);
                                    }
                                })),
                            ),
                        )
                    }
                }
                TinyCharts { saw_nan_line, saw_nan_hist }
            })
        })
        .expect("open tiny chart window")
    });
    app.frame(win.into());
    app.simulate_input(
        win.into(),
        PlatformInput::MouseMove(MouseMoveEvent {
            position: point(px(5.), px(5.)),
            pressed_button: None,
            modifiers: Modifiers::none(),
        }),
    );
    app.frame(win.into());
    assert!(!saw_nan.get(), "tiny charts must never emit NaN hover fractions");
}
