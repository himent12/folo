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

/// Types `text` into one of the New run form's inputs.
fn type_into(
    app: &mut HeadlessApp,
    window: WindowHandle<FoloApp>,
    which: fn(&super::NewRunForm) -> folo_ui::Entity<crate::input::TextInput>,
    text: &str,
) {
    let text = text.to_string();
    drive(app, window, move |this, _, cx| {
        let input = which(this.new_run.as_ref().unwrap());
        input.update(cx, |i, cx| i.set_text(text, cx));
    });
}

#[test]
fn close_mosaic_follows_the_epochs_until_set_by_hand() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| this.open_new_run(None, w, cx));
    let state = |app: &mut HeadlessApp| {
        drive(app, window, |this, _, cx| {
            let f = this.new_run.as_ref().unwrap();
            (f.close_mosaic as u32, f.close_invalid, f.close_input.read(cx).text().to_string())
        })
    };
    assert_eq!(state(&mut app), (10, false, "10".into()));
    // 3 epochs: the untouched default follows (half the run) instead of turning invalid.
    type_into(&mut app, window, |f| f.epochs_input.clone(), "3");
    assert_eq!(state(&mut app), (1, false, "1".into()));
    type_into(&mut app, window, |f| f.epochs_input.clone(), "50");
    assert_eq!(state(&mut app), (10, false, "10".into()));
    // Set by hand, it stays, and is checked against the epochs.
    type_into(&mut app, window, |f| f.close_input.clone(), "20");
    type_into(&mut app, window, |f| f.epochs_input.clone(), "15");
    assert_eq!(state(&mut app), (20, true, "20".into()));
}

#[test]
fn a_runs_filter_does_not_wait_for_the_next_visit() {
    use super::{Route, RunFilter};
    let (mut app, window) = open();
    let filter = |app: &mut HeadlessApp| drive(app, window, |this, _, _| this.run_filter);
    // The queue toast's View queue: Runs, filtered to the active runs.
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Runs, w, cx);
        this.run_filter = RunFilter::Active;
    });
    // Into a run and back up: the list as it was left.
    drive(&mut app, window, |this, w, cx| {
        let id = this.store.runs.ready().unwrap()[0].id.clone();
        this.navigate(Route::Run(id), w, cx);
        this.navigate(Route::Runs, w, cx);
    });
    assert_eq!(filter(&mut app), RunFilter::Active);
    // Elsewhere and back later (sidebar, Ctrl+3): all runs again.
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Overview, w, cx);
        this.navigate(Route::Runs, w, cx);
    });
    assert_eq!(filter(&mut app), RunFilter::All);
}

#[test]
fn a_dialog_forgets_could_not_reach_once_the_server_answers() {
    use crate::api::ApiError;
    use crate::store::Load;
    let (mut app, window) = open();
    let down = ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7926");
    drive(&mut app, window, |this, w, cx| {
        this.open_new_run(None, w, cx);
        this.new_run.as_mut().unwrap().error = Some(down.clone());
        this.nr_estimate = Some(("key".into(), Load::Failed(down.clone())));
        this.store.health = Load::Failed(down.clone());
        // The 2 s refresh, now answered.
        this.store.system = Load::Idle;
        this.ensure_core(cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(!this.store.unreachable());
        assert!(this.new_run.as_ref().unwrap().error.is_none(), "the stale error is gone");
        assert!(this.nr_estimate.is_none(), "the estimate is asked for again");
    });
}

#[test]
fn deleting_a_run_or_a_dataset_leaves_its_page_for_the_list() {
    use super::{DeleteTarget, Route};
    let (mut app, window) = open();
    // A finished run, from its page: it and its models go, and the page gives way to Runs.
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Run("run_52e8".into()), w, cx);
        this.ask_delete(DeleteTarget::Run("run_52e8".into()), w, cx);
        this.delete_confirmed(cx);
    });
    drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
    drive(&mut app, window, |this, _, _| {
        assert!(this.confirm_delete.is_none());
        assert!(this.store.run("run_52e8").is_none());
        assert_eq!(this.route_pending, Some(Route::Runs));
        let models = this.store.models.ready().unwrap();
        assert!(models.iter().all(|m| m.run_id.as_deref() != Some("run_52e8")), "its models went with it");
        assert!(this.toasts.iter().any(|t| t.title.starts_with("Deleted")));
    });
    // An active run is refused (409 run_active) and stays.
    drive(&mut app, window, |this, w, cx| {
        this.ask_delete(DeleteTarget::Run("run_a19f".into()), w, cx);
        this.delete_confirmed(cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.run("run_a19f").is_some());
        assert!(this.toasts.iter().any(|t| t.title.starts_with("Couldn't delete")));
    });
    // A dataset is unregistered.
    drive(&mut app, window, |this, w, cx| {
        this.route_pending = None;
        this.navigate(Route::Dataset("ds_3b91".into()), w, cx);
        this.ask_delete(DeleteTarget::Dataset("ds_3b91".into()), w, cx);
        this.delete_confirmed(cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.dataset_summary("ds_3b91").is_none());
        assert_eq!(this.route_pending, Some(Route::Datasets));
    });
}

#[test]
fn another_device_keeps_the_boxes_up_until_it_answers() {
    use crate::store::Load;
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        this.play_model = Some("m_44c0".into());
        this.play_dataset_image("ds_3b91", "img_03417", cx);
    });
    assert!(drive(&mut app, window, |this, _, _| this.play_pred.ready().is_some()), "the first result is in");
    // Another device: the same boxes stay up (dimmed) while it runs, no placeholder rows.
    let during = app.update(|cx| {
        window
            .update(cx, |this, _, cx| {
                this.set_play_target(Some(crate::local::Target::Cpu), cx);
                (matches!(this.play_pred, Load::Ready(_)), this.play_pred_dim, this.play_pred_pending)
            })
            .unwrap()
    });
    assert_eq!(during, (true, true, true));
    // Its answer (here the demo's refusal) replaces them, undimmed.
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| assert!(!this.play_pred_dim && !this.play_pred_pending));
}

#[test]
fn new_run_opens_on_the_last_started_choices() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.open_new_run(None, w, cx);
        let f = this.new_run.as_mut().unwrap();
        (f.arch, f.imgsz) = (0, 0);
        let input = f.epochs_input.clone();
        input.update(cx, |i, cx| i.set_text("30", cx));
    });
    drive(&mut app, window, |this, w, cx| this.submit_new_run(w, cx));
    drive(&mut app, window, |this, w, cx| {
        assert!(this.new_run.is_none(), "started");
        this.open_new_run(None, w, cx);
    });
    let name = run_name(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        let f = this.new_run.as_ref().unwrap();
        assert_eq!((f.arch, f.imgsz, f.epochs as u32), (0, 0, 30));
        assert_eq!(f.epochs_input.read(cx).text().to_string(), "30");
        // Close mosaic follows the remembered epochs (half of 30 caps nothing: 10).
        assert_eq!(f.close_mosaic as u32, 10);
    });
    assert!(name.contains("-n-"), "{name}");
}
