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
    open_with(Arc::new(MockBackend::new(4200, true)))
}

/// [`open`] on a demo backend the test keeps a hand on.
pub(crate) fn open_with(api: Arc<MockBackend>) -> (HeadlessApp, WindowHandle<FoloApp>) {
    let mut app = HeadlessApp::new(HeadlessOptions::default());
    let window = app.update(|cx| {
        cx.set_global(Theme::new(Mode::Dark));
        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(1280.), px(800.)));
        let options =
            WindowOptions { window_bounds: Some(WindowBounds::Windowed(bounds)), ..Default::default() };
        cx.open_window(options, |window, cx| cx.new(|cx| FoloApp::new(api, true, window, cx)))
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
        this.new_run.as_mut().unwrap().dataset = this.store.datasets.ready().unwrap()[1].id.clone();
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
fn new_run_offers_gpu_0_only_on_a_server_that_has_a_gpu() {
    use crate::store::Load;
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| this.open_new_run(None, w, cx));
    app.frame(window.into());
    // The demo server has a GPU: offered, and the right arrow from Auto picks it.
    drive(&mut app, window, |this, _, _| assert_eq!(this.nr_no_gpu(), None));
    assert_eq!(device_after_right(&mut app, window, 0), 1);
    // Never answered: still offered (the server decides at Start).
    drive(&mut app, window, |this, _, _| {
        let kept = std::mem::replace(&mut this.store.system, Load::Loading);
        let last = this.store.last_gpu_count.take();
        assert_eq!(this.nr_no_gpu(), None);
        this.store.system = kept;
        this.store.last_gpu_count = last;
    });
    // A server with no GPU (`/api/system` gpus = [], d2-runs-2): GPU 0 is disabled, with the reason,
    // and the right arrow from Auto steps over it to CPU.
    drive(&mut app, window, |this, _, _| {
        if let Load::Ready(s) = &mut this.store.system {
            s.gpus.clear();
        }
        assert!(this.nr_no_gpu().is_some_and(|why| why.contains("no GPU")));
    });
    assert_eq!(device_after_right(&mut app, window, 0), 2);
    // While `/system` is asked again (every 2 s it is Loading) the last answer, no GPU, stands.
    drive(&mut app, window, |this, _, _| {
        this.store.last_gpu_count = Some(0);
        this.store.system = Load::Loading;
        assert!(this.nr_no_gpu().is_some());
    });
    assert_eq!(device_after_right(&mut app, window, 0), 2);
    // GPU 0 still picked (prefilled from a run that had it, or picked before the server said):
    // Start says why and nothing is sent until Auto or CPU is picked.
    drive(&mut app, window, |this, w, cx| {
        this.new_run.as_mut().unwrap().device = 1;
        assert!(this.nr_device_gone().is_some_and(|why| why.contains("pick Auto or CPU")));
        this.submit_new_run(w, cx);
        assert!(!this.new_run.as_ref().unwrap().submitting, "nothing is sent");
        this.new_run.as_mut().unwrap().device = 2;
        assert_eq!(this.nr_device_gone(), None);
    });
    // The dialog draws with it.
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| assert!(this.new_run.is_some()));
}

/// New run's Device control focused at option `from`, then the right arrow pressed as a keyboard
/// does: the option it picks.
fn device_after_right(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, from: usize) -> usize {
    use folo_ui::{KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput};
    drive(app, window, |this, _, _| this.new_run.as_mut().unwrap().device = from);
    app.frame(window.into());
    drive(app, window, |this, w, cx| w.focus(&this.fh("nr-device", cx)));
    app.frame(window.into());
    let keystroke = Keystroke::parse("right").unwrap();
    app.simulate_input(
        window.into(),
        PlatformInput::KeyDown(KeyDownEvent { keystroke: keystroke.clone(), is_held: false }),
    );
    app.run_until_idle();
    app.simulate_input(window.into(), PlatformInput::KeyUp(KeyUpEvent { keystroke }));
    app.run_until_idle();
    drive(app, window, |this, _, _| this.new_run.as_ref().unwrap().device)
}

#[test]
fn new_run_on_a_server_without_datasets_says_so_once() {
    use crate::store::Load;
    let (mut app, window) = open();
    // Still loading: nothing yet (the list lands in a moment).
    drive(&mut app, window, |this, w, cx| {
        this.store.datasets = Load::Loading;
        this.open_new_run(None, w, cx);
        assert!(this.new_run.is_none() && this.toasts.is_empty());
    });
    // An empty server (d2-runs-4): Ctrl+N twice, one toast that leads to Add dataset.
    drive(&mut app, window, |this, w, cx| {
        this.store.datasets = Load::Ready(Vec::new());
        this.open_new_run(None, w, cx);
        this.open_new_run(None, w, cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(this.new_run.is_none());
        let shown: Vec<_> = this.toasts.iter().filter(|t| &*t.title == "Add a dataset first").collect();
        assert_eq!(shown.len(), 1);
        assert!(shown[0].action.as_ref().is_some_and(|(label, _)| &**label == "Add dataset"));
    });
    // `--open new-run` on it says so too, at its first frame, and doesn't open the dialog later
    // when a dataset appears.
    drive(&mut app, window, |this, _, _| {
        this.toasts.clear();
        this.open_new_run_pending = true;
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert!(!this.open_new_run_pending, "the request is answered");
        assert!(this.toasts.iter().any(|t| &*t.title == "Add a dataset first"));
        assert!(this.new_run.is_none());
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
    // Drawn: the page shown is the list.
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert!(this.confirm_delete.is_none());
        assert!(this.store.run("run_52e8").is_none());
        assert_eq!(this.route, Route::Runs);
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
        this.navigate(Route::Dataset("ds_3b91".into()), w, cx);
        this.ask_delete(DeleteTarget::Dataset("ds_3b91".into()), w, cx);
        this.delete_confirmed(cx);
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.dataset_summary("ds_3b91").is_none());
        assert_eq!(this.route, Route::Datasets);
    });
}

#[test]
fn an_earlier_toasts_open_goes_to_the_list_once_its_dataset_is_deleted() {
    use super::{DeleteTarget, Route};
    let (mut app, window) = open();
    // A dataset's toast with Open (the demo lists it already: "… is already added").
    let (id, name) = drive(&mut app, window, |this, _, cx| {
        let ds = this.store.datasets.ready().unwrap()[0].clone();
        let id_name = (ds.id.clone(), ds.name.clone());
        this.dataset_added(ds, "Added", cx);
        id_name
    });
    // Its page opened another way (the toast stays), and the dataset deleted from there.
    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Dataset(id.clone()), w, cx);
        this.ask_delete(DeleteTarget::Dataset(id.clone()), w, cx);
        this.delete_confirmed(cx);
    });
    app.frame(window.into());
    // The earlier toast's Open: the list, saying why, not the deleted dataset's page.
    drive(&mut app, window, |this, w, cx| {
        assert_eq!(this.route, Route::Datasets);
        let open = this.toasts.iter().find_map(|t| t.action.clone()).expect("the toast with Open");
        (open.1)(this, w, cx);
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| {
        assert_eq!(this.route, Route::Datasets);
        let said = format!("{name} was removed");
        assert!(this.toasts.iter().any(|t| t.title == said.as_str()), "{said}");
        // Nor does Back lead there.
        this.back(w, cx);
        assert_ne!(this.route, Route::Dataset(id.clone()));
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

#[test]
fn remove_from_queue_deletes_a_run_that_never_started() {
    use crate::api::RunExt;
    let (mut app, window) = open();
    let id = drive(&mut app, window, |this, _, _| {
        this.store.runs.ready().unwrap().iter().find(|r| r.never_started()).map(|r| r.id.clone())
    })
    .expect("the demo data has a run waiting in the queue");
    drive(&mut app, window, |this, _, cx| this.remove_queued(id.clone(), cx));
    // Gone from the list and from the server, never left behind as a stopped run.
    let held = drive(&mut app, window, |this, _, _| this.store.run(&id).is_some());
    assert!(!held);
    let api = drive(&mut app, window, |this, _, _| this.store.api.clone());
    assert!(api.runs().unwrap().iter().all(|r| r.id != id));
}

/// What the New run and New benchmark forms show as their dataset: (New run's, New benchmark's).
fn form_datasets(this: &FoloApp) -> (Option<String>, Option<String>) {
    let sources = this.bench_sources();
    let picked = this.bench.new_bench.as_ref().and_then(|f| f.source.as_ref());
    let bench = this.source_choice(&sources, picked).index().map(|i| sources[i].name().to_string());
    (this.nr_dataset().map(|d| d.name.clone()), bench)
}

/// Confirms deleting dataset `id`, runs `then` before the server's answer is applied, then lets it
/// be applied.
fn delete_dataset_while(
    app: &mut HeadlessApp,
    window: WindowHandle<FoloApp>,
    id: &str,
    then: impl FnOnce(&mut FoloApp, &mut Window, &mut Context<FoloApp>),
) {
    let id = id.to_string();
    app.update(|cx| {
        window
            .update(cx, |this, w, cx| {
                this.ask_delete(super::DeleteTarget::Dataset(id), w, cx);
                this.delete_confirmed(cx);
                then(this, w, cx);
            })
            .unwrap()
    });
    app.run_until_idle();
}

#[test]
fn an_open_form_keeps_its_dataset_while_another_is_deleted() {
    use crate::bench::{SourceChoice, SourceId};
    let (mut app, window) = open();
    let ds: Vec<(String, String)> = drive(&mut app, window, |this, _, _| {
        this.store.datasets.ready().unwrap().iter().map(|d| (d.id.clone(), d.name.clone())).collect()
    });
    assert!(ds.len() >= 3, "the demo has three datasets");
    let (a, b, c) = (&ds[0], &ds[1], &ds[2]);
    // A's deletion is out when New run and New benchmark open and B is picked in both.
    drive(&mut app, window, |this, w, cx| {
        this.open_new_run(None, w, cx);
        this.open_new_bench(w, cx);
    });
    delete_dataset_while(&mut app, window, &a.0, |this, _, cx| {
        // New run's dataset menu open on the row after A's.
        (this.menu_open, this.menu_hl) = (Some("nr-dataset"), 1);
        this.new_run.as_mut().unwrap().dataset = b.0.clone();
        this.resuggest_run_name(cx);
        this.pick_bench_source(Some(SourceId::Dataset(b.0.clone())), None, cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert!(this.store.dataset_summary(&a.0).is_none(), "A is gone");
        assert!(this.menu_open.is_none(), "the menu whose rows moved closed");
        assert_eq!(form_datasets(this), (Some(b.1.clone()), Some(b.1.clone())), "both still say B");
        assert_eq!(this.form_config(cx).map(|c| c.dataset_id), Some(b.0.clone()));
        let name = this.new_run.as_ref().unwrap().name.read(cx).text().to_string();
        assert!(name.starts_with(&b.1), "the suggested name names B: {name}");
    });
    // Now B itself is deleted: both forms say so, and neither sends anything until another is picked.
    delete_dataset_while(&mut app, window, &b.0, |_, _, _| {});
    drive(&mut app, window, |this, w, cx| {
        assert!(this.nr_dataset_gone());
        assert_eq!(form_datasets(this), (None, None));
        assert!(this.form_config(cx).is_none());
        let sources = this.bench_sources();
        let picked = this.bench.new_bench.as_ref().and_then(|f| f.source.as_ref());
        assert_eq!(this.source_choice(&sources, picked), SourceChoice::Deleted);
        this.submit_new_run(w, cx);
        this.submit_new_bench(cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(this.new_run.as_ref().is_some_and(|f| !f.submitting), "nothing sent");
        let error = this.bench.new_bench.as_ref().and_then(|f| f.error.as_ref()).map(|e| e.message.clone());
        assert_eq!(error.as_deref(), Some("That source was deleted: pick another"));
    });
    // C, picked in both: that is what each sends.
    let run_name = drive(&mut app, window, |this, w, cx| {
        this.new_run.as_mut().unwrap().dataset = c.0.clone();
        this.resuggest_run_name(cx);
        this.pick_bench_source(Some(SourceId::Dataset(c.0.clone())), None, cx);
        assert_eq!(form_datasets(this), (Some(c.1.clone()), Some(c.1.clone())));
        let name = this.new_run.as_ref().unwrap().name.read(cx).text().to_string();
        this.submit_new_run(w, cx);
        this.submit_new_bench(cx);
        name
    });
    let api = drive(&mut app, window, |this, _, _| {
        assert!(this.new_run.is_none() && this.bench.new_bench.is_none(), "both sent");
        this.store.api.clone()
    });
    let run = api.runs().unwrap().into_iter().find(|r| r.name == run_name).expect("created");
    assert_eq!(run.config.dataset_id, c.0);
    let bench = api.benches().unwrap().into_iter().find(|x| x.name.starts_with(&c.1)).expect("created");
    assert!(
        matches!(&bench.source, crate::api::BenchSource::Dataset { dataset_id, .. } if *dataset_id == c.0)
    );
}

#[test]
fn the_palette_highlight_stays_on_its_row_while_a_run_before_it_is_deleted() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| this.open_palette(w, cx));
    app.frame(window.into());
    // The row after the run about to be deleted is highlighted.
    let row = drive(&mut app, window, |this, _, _| {
        let name = this.store.run("run_52e8").unwrap().name.clone();
        let p = this.palette.as_mut().unwrap();
        let at = p.shown.iter().position(|r| *r == ("Runs", name.clone())).expect("listed");
        p.selected = at + 1;
        p.shown[at + 1].clone()
    });
    assert_eq!(row.0, "Runs", "a run listed after it");
    app.frame(window.into());
    // Its deletion (asked for before the palette opened) is answered.
    drive(&mut app, window, |this, _, cx| {
        this.confirm_delete = Some(super::DeleteTarget::Run("run_52e8".into()));
        this.delete_confirmed(cx);
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let p = this.palette.as_ref().unwrap();
        assert!(!p.shown.iter().any(|r| r.1 == "run_52e8"));
        assert_eq!(p.shown.get(p.selected), Some(&row), "still on the same run");
    });
}

#[test]
fn a_toast_leaves_on_the_clock_but_not_while_hovered() {
    use std::time::Duration;
    let (mut app, window) = open();
    // The frozen demo keeps toasts up for screenshots; these two leave as a live app's do. The
    // second is under the pointer.
    let held = drive(&mut app, window, |this, _, cx| {
        this.frozen = false;
        this.toast(crate::ui::Tone::Info, "plain", None, cx);
        this.toast(crate::ui::Tone::Info, "hovered", None, cx);
        this.frozen = true;
        let held = this.toasts.last().unwrap().id;
        this.toast_hover = Some(held);
        held
    });
    let titles = |app: &mut HeadlessApp| {
        drive(app, window, |this, _, _| this.toasts.iter().map(|t| t.title.to_string()).collect::<Vec<_>>())
    };
    let ttl = super::overlay_state::toast_ttl(false);
    app.advance_clock(ttl - Duration::from_millis(1));
    assert_eq!(titles(&mut app), ["plain", "hovered"]);
    app.advance_clock(Duration::from_millis(1));
    assert_eq!(titles(&mut app), ["hovered"], "the plain one left at its time");
    app.advance_clock(Duration::from_secs(10));
    assert_eq!(titles(&mut app), ["hovered"], "hovered, it stays");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.toast_hover, Some(held));
        this.toast_hover = None;
    });
    app.advance_clock(Duration::from_millis(800));
    assert!(titles(&mut app).is_empty(), "left once the pointer did");
}

#[test]
fn lists_changed_elsewhere_show_at_the_next_refresh() {
    use super::{DeleteTarget, Route};
    use crate::api::Backend;
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let (id, run) = drive(&mut app, window, |this, _, _| {
        let runs = this.store.runs.ready().unwrap();
        let run = runs.iter().find(|r| this.store.dataset_summary(&r.config.dataset_id).is_some()).unwrap();
        (run.config.dataset_id.clone(), run.clone())
    });
    // Removed elsewhere (Studio, the CLI): gone at the next refresh, and its runs say so.
    api.delete_dataset(&id, false).unwrap();
    drive(&mut app, window, |this, _, cx| this.refresh_lists(cx));
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.dataset_summary(&id).is_none());
        assert!(
            this.store.run_dataset_name(&run).contains("removed"),
            "{}",
            this.store.run_dataset_name(&run)
        );
    });
    // Its folder registered again: listed, and "(removed)" goes.
    api.restore(&id);
    drive(&mut app, window, |this, _, cx| this.refresh_lists(cx));
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.dataset_summary(&id).is_some());
        assert!(!this.store.run_dataset_name(&run).contains("removed"));
    });
    // Deleted here, then registered again elsewhere: no longer treated as deleted (its page opens).
    drive(&mut app, window, |this, w, cx| {
        this.ask_delete(DeleteTarget::Dataset(id.clone()), w, cx);
        this.delete_confirmed(cx);
    });
    api.restore(&id);
    drive(&mut app, window, |this, _, cx| this.refresh_lists(cx));
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| {
        assert!(this.store.dataset_summary(&id).is_some());
        this.navigate(Route::Dataset(id.clone()), w, cx);
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| assert_eq!(this.route, Route::Dataset(id.clone())));
}

#[test]
fn a_generated_dataset_can_be_removed_with_its_files_as_in_studio() {
    use super::DeleteTarget;
    use crate::api::RunStatus;
    let (mut app, window) = open();
    let id = drive(&mut app, window, |this, _, _| {
        let runs = this.store.runs.ready().unwrap();
        let ds = this.store.datasets.ready().unwrap();
        let idle =
            ds.iter().find(|d| !runs.iter().any(|r| r.config.dataset_id == d.id && r.status.is_active()));
        let id = idle.expect("a dataset no run is using").id.clone();
        assert!(runs.iter().any(|r| r.status.is_active()), "the demo trains on another dataset");
        // The demo has no data dir: its datasets are folders of the person's own, never offered.
        assert!(!this.dataset_files_deletable(&id));
        // A server's own: under <data dir>/datasets.
        this.store.data_dir = Some("D:/srv/folo".into());
        let d = this.store.datasets.ready_mut().unwrap().iter_mut().find(|d| d.id == id).unwrap();
        d.path = r"D:\srv\folo\datasets\shapes".into();
        // The server deletes files only while no run is active, on any dataset: not offered yet.
        assert!(!this.dataset_files_deletable(&id));
        for r in this.store.runs.ready_mut().unwrap().iter_mut().filter(|r| r.status.is_active()) {
            r.status = RunStatus::Stopped;
        }
        assert!(this.dataset_files_deletable(&id));
        id
    });
    // Asked for: the call carries ?files=true (the demo refuses it, which says it got there).
    drive(&mut app, window, |this, w, cx| {
        this.ask_delete(DeleteTarget::Dataset(id.clone()), w, cx);
        assert!(!this.delete_files, "off each time it opens");
        this.delete_files = true;
        this.delete_confirmed(cx);
    });
    drive(&mut app, window, |this, _, _| {
        let t = this.toasts.last().unwrap();
        assert!(t.title.starts_with("Couldn't remove"), "{}", t.title);
        assert!(this.store.dataset_summary(&id).is_some());
    });
    // Not asked: unregistered, in Studio's words.
    drive(&mut app, window, |this, w, cx| {
        this.ask_delete(DeleteTarget::Dataset(id.clone()), w, cx);
        this.delete_confirmed(cx);
    });
    drive(&mut app, window, |this, _, _| {
        let t = this.toasts.last().unwrap();
        assert_eq!(t.title.as_ref(), "Dataset removed");
        assert!(
            t.body
                .as_ref()
                .is_some_and(|b| b.ends_with("is no longer registered. Files on disk were not touched."))
        );
        assert!(this.store.dataset_summary(&id).is_none());
    });
}

#[test]
fn a_run_that_ends_shows_its_model_and_end_at_once() {
    use crate::api::RunStatus;
    let (mut app, window) = open();
    // As the live stream leaves it: the status says completed, the list doesn't know the rest yet.
    let id = drive(&mut app, window, |this, _, _| {
        let runs = this.store.runs.ready_mut().unwrap();
        let r = runs.iter_mut().find(|r| r.status == RunStatus::Completed && r.model_id.is_some()).unwrap();
        r.model_id = None;
        r.finished = None;
        r.id.clone()
    });
    drive(&mut app, window, |this, _, cx| this.run_ended(&id, RunStatus::Completed, cx));
    drive(&mut app, window, |this, _, _| {
        let r = this.store.run(&id).unwrap();
        assert!(r.model_id.is_some(), "Open model and Try in Playground show now");
        assert!(r.finished.is_some(), "its duration stops counting now");
        assert!(this.toasts.iter().any(|t| t.title.ends_with("finished")));
    });
}

#[test]
fn closing_the_inspector_lands_on_the_image_it_showed_last() {
    use super::Route;
    let (mut app, window) = open();
    let ds = drive(&mut app, window, |this, w, cx| {
        let ds = this.store.datasets.ready().unwrap()[0].id.clone();
        this.navigate(Route::Dataset(ds.clone()), w, cx);
        ds
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| this.ensure_dataset(&ds, cx));
    // Opened on the first tile (focused by its click), then walked on to the 24th image.
    drive(&mut app, window, |this, w, cx| {
        let first = this.filtered_image(&ds, 0).unwrap().id.clone();
        let fh = this.fh(format!("tile-{first}"), cx);
        w.focus(&fh);
        this.open_inspector(ds.clone(), 0, w, cx);
        for _ in 0..23 {
            this.step_inspector(1, cx);
        }
    });
    drive(&mut app, window, |this, w, cx| {
        let shown = this.filtered_image(&ds, 23).unwrap().id.clone();
        this.close_inspector(w, cx);
        assert!(this.fh(format!("tile-{shown}"), cx).is_focused(w), "focus is on the image shown last");
    });
}

#[test]
fn new_run_refuses_a_name_longer_than_the_server_takes_and_fits_its_own() {
    let (mut app, window) = open();
    // New run from a run whose name is at the limit: the suggestion still fits.
    let long = "r".repeat(super::forms::MAX_RUN_NAME);
    drive(&mut app, window, |this, w, cx| {
        let mut config = this.store.runs.ready().unwrap()[0].config.clone();
        config.name = long.clone();
        this.open_new_run_from(&config, None, w, cx);
    });
    let name = run_name(&mut app, window);
    assert!(name.chars().count() <= 64 && name.ends_with("-2"), "{name}");
    // Typed one character over: refused in Studio's words, and nothing is sent.
    type_into(&mut app, window, |f| f.name.clone(), &format!("{long}x"));
    let api = drive(&mut app, window, |this, w, cx| {
        this.submit_new_run(w, cx);
        this.store.api.clone()
    });
    let runs = api.runs().unwrap().len();
    drive(&mut app, window, |this, _, _| {
        let f = this.new_run.as_ref().expect("still open");
        assert!(!f.submitting);
        let said = f.error.as_ref().map(|e| e.message.clone());
        assert_eq!(said.as_deref(), Some("At most 64 characters (this one has 65)."));
    });
    assert_eq!(api.runs().unwrap().len(), runs);
}

#[test]
fn closing_while_its_own_server_trains_asks_first_alone_and_keep_open_keeps_it() {
    let (mut app, window) = open();
    // Nothing would stop: the window just closes.
    assert!(drive(&mut app, window, |this, w, cx| this.should_close_with(Some(Vec::new()), w, cx)));
    // New run and a Stop confirmation are open when the window is asked to close while its server
    // trains: the question opens alone (New run kept as a draft), so nothing under it takes keys.
    drive(&mut app, window, |this, w, cx| this.open_new_run(None, w, cx));
    drive(&mut app, window, |this, w, cx| this.ask_stop("run_a19f".into(), w, cx));
    assert!(!drive(&mut app, window, |this, w, cx| this.should_close_with(
        Some(vec!["shapes-hard-m".into()]),
        w,
        cx
    )));
    drive(&mut app, window, |this, w, cx| {
        assert_eq!(this.confirm_close.as_deref(), Some(&["shapes-hard-m".to_string()][..]));
        assert!(this.confirm_stop.is_none() && this.new_run.is_none() && this.new_run_draft.is_some());
        assert!(this.modal_open());
        // Asked again while asking (Close pressed twice): still asking, still open.
        assert!(!this.should_close(w, cx));
        // Esc is Keep open.
        this.dismiss(w, cx);
        assert!(this.confirm_close.is_none());
    });
    // `--open new-run` waiting for the datasets doesn't open under the question.
    drive(&mut app, window, |this, w, cx| {
        this.open_new_run_pending = true;
        assert!(!this.should_close_with(Some(vec!["shapes-hard-m".into()]), w, cx));
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| {
        assert!(this.new_run.is_none() && this.open_new_run_pending, "it waits");
        this.dismiss(w, cx);
    });
    // The server didn't say: asked too (no names: "couldn't check").
    assert!(!drive(&mut app, window, |this, w, cx| this.should_close_with(None, w, cx)));
    drive(&mut app, window, |this, _, _| assert_eq!(this.confirm_close.as_deref(), Some(&[][..])));
}

#[test]
fn a_dataset_page_for_an_id_the_server_does_not_have_leads_to_the_list_not_to_settings() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| this.navigate(super::Route::Dataset("ds_nope".into()), w, cx));
    // A frame asks for the dataset, the answer comes, the next frame draws it.
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "ds-missing-all"), "{keys:?}");
        assert!(
            !keys.iter().any(|k| k.starts_with("ds-detail-retry")),
            "no Connection settings or Retry: {keys:?}"
        );
    });
}

#[test]
fn a_dataset_not_found_once_loads_again_once_it_is_listed() {
    use crate::api::ApiError;
    use crate::store::Load;
    let (mut app, window) = open();
    // Its page was opened while it wasn't registered; now the list has it again.
    let id = drive(&mut app, window, |this, _, _| {
        let id = this.store.datasets.ready().unwrap()[0].id.clone();
        this.store.dataset.insert(id.clone(), Load::Failed(ApiError::new("dataset_not_found", "gone")));
        id
    });
    drive(&mut app, window, |this, _, cx| this.ensure_dataset(&id, cx));
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.dataset.get(&id).is_some_and(|l| l.ready().is_some()))
    });
    // One the list doesn't have stays "not found" (no request loop).
    drive(&mut app, window, |this, _, cx| {
        this.store.dataset.insert("ds_nope".into(), Load::Failed(ApiError::new("dataset_not_found", "gone")));
        this.ensure_dataset("ds_nope", cx);
        assert!(this.store.dataset.get("ds_nope").is_some_and(|l| l.error().is_some()));
    });
}

/// x2-newcomer-1 (fixed by d2-runs-4's change, kept): New run (Ctrl N, the palette, a New run button) with no
/// dataset at all stays on the page and says why once, and the toast's Add dataset leads to Datasets.
#[test]
fn new_run_without_any_dataset_offers_add_dataset() {
    let (mut app, window) = open_with(Arc::new(MockBackend::new(4200, true).empty()));
    drive(&mut app, window, |this, w, cx| {
        assert!(
            this.store.datasets.ready().is_some_and(|d| d.is_empty()),
            "an empty server, its list loaded"
        );
        this.navigate(super::Route::Playground, w, cx);
    });
    // Ctrl N twice: one toast, the user not moved.
    drive(&mut app, window, |this, w, cx| this.open_new_run(None, w, cx));
    drive(&mut app, window, |this, w, cx| this.open_new_run(None, w, cx));
    let add = drive(&mut app, window, |this, _, _| {
        assert!(this.new_run.is_none(), "no form without a dataset");
        assert_eq!(this.route, super::Route::Playground);
        let shown: Vec<_> = this.toasts.iter().filter(|t| t.title.as_ref() == "Add a dataset first").collect();
        assert_eq!(shown.len(), 1, "a second Ctrl N adds no second toast");
        let (label, add) = shown[0].action.clone().expect("the toast offers the way on");
        assert_eq!(label.as_ref(), "Add dataset");
        add
    });
    // Its action: Datasets and the folder step (on the demo that says it needs folo serve).
    drive(&mut app, window, |this, w, cx| add(this, w, cx));
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.route, super::Route::Datasets);
        assert!(this.toasts.iter().any(|t| t.title.as_ref() == "Registering folders needs folo serve"));
    });
}

/// x2-newcomer-3: Enter in the Generate synthetic dialog's Name field generates (as Ctrl Enter does).
#[test]
fn enter_in_the_synth_dialog_generates() {
    use folo_ui::{KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput};
    let (mut app, window) = open_with(Arc::new(MockBackend::new(4200, true).empty()));
    // The dialog is for folo serve (on the demo, Generate synthetic only explains itself).
    drive(&mut app, window, |this, w, cx| {
        this.server_mode = super::ServerMode::Local;
        this.open_synth(w, cx);
        assert!(this.synth.as_ref().is_some_and(|f| !f.submitting), "the dialog is open, not sent");
    });
    app.frame(window.into());
    let keystroke = Keystroke::parse("enter").unwrap();
    app.simulate_input(
        window.into(),
        PlatformInput::KeyDown(KeyDownEvent { keystroke: keystroke.clone(), is_held: false }),
    );
    app.run_until_idle();
    app.frame(window.into());
    app.simulate_input(window.into(), PlatformInput::KeyUp(KeyUpEvent { keystroke }));
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        // The demo backend refuses the request (the dialog stays with its error); what counts is that it went.
        let f = this.synth.as_ref().expect("the dialog stays open on the demo's refusal");
        assert_eq!(f.sent_name, "shapes", "Enter sent the dialog");
    });
}
