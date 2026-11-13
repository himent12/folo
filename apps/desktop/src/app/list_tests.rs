//! The lists against a server that answers late and out of order (the demo, with calls held back and
//! released by the test): an answer asked for before a change here, or overtaken by a later one,
//! never undoes what is newer.

use std::sync::Arc;

use folo_ui::{Context, HeadlessApp, Window, WindowHandle};

use super::tests::{drive, open_with};
use super::{DeleteTarget, FoloApp, Route};
use crate::api::Backend;
use crate::api::mock::MockBackend;

/// Runs `f` on the app, then lets every fetch answer but the `held` ones the test holds back.
fn start<R>(
    app: &mut HeadlessApp,
    window: WindowHandle<FoloApp>,
    held: usize,
    f: impl FnOnce(&mut FoloApp, &mut Window, &mut Context<FoloApp>) -> R,
) -> R {
    let r = app.update(|cx| window.update(cx, f).expect("the window is open"));
    app.run_until_held(held);
    r
}

fn listed(this: &FoloApp, id: &str) -> bool {
    this.store.dataset_summary(id).is_some()
}

#[test]
fn a_list_asked_for_before_a_delete_here_does_not_bring_it_back() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let id = drive(&mut app, window, |this, _, _| this.store.datasets.ready().unwrap()[0].id.clone());
    // The 5 s refresh asks; its answer (still listing the dataset) is made, and held.
    let held = api.hold("datasets");
    start(&mut app, window, 1, |this, _, cx| this.refresh_lists(cx));
    held.wait_held();
    // Meanwhile the dataset is removed here; the fetch that follows answers at once.
    start(&mut app, window, 1, |this, w, cx| {
        this.ask_delete(DeleteTarget::Dataset(id.clone()), w, cx);
        this.delete_confirmed(cx);
    });
    assert!(held.is_held(), "the old answer was still out when the delete landed");
    assert!(!drive_held(&mut app, window, &id));
    // The old answer lands last: dropped. The dataset stays gone, and a link to it says so.
    held.release();
    app.run_until_idle();
    drive(&mut app, window, |this, w, cx| {
        assert!(!listed(this, &id), "not brought back");
        assert!(this.deleted.get(&id).is_some(), "still known as deleted");
        this.navigate(Route::Dataset(id.clone()), w, cx);
        assert_eq!(this.route, Route::Datasets);
    });
}

/// Whether the dataset is listed, without letting the held call go.
fn drive_held(app: &mut HeadlessApp, window: WindowHandle<FoloApp>, id: &str) -> bool {
    start(app, window, 1, |this, _, _| listed(this, id))
}

#[test]
fn models_asked_for_before_their_run_is_deleted_here_stay_gone() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let (run, made) = drive(&mut app, window, |this, _, _| {
        let m = this.store.models.ready().unwrap().iter().find(|m| m.run_id.as_deref() == Some("run_52e8"));
        ("run_52e8".to_string(), m.expect("the run made a model").id.clone())
    });
    let held = api.hold("models");
    start(&mut app, window, 1, |this, _, cx| this.refresh_lists(cx));
    held.wait_held();
    start(&mut app, window, 1, |this, w, cx| {
        this.ask_delete(DeleteTarget::Run(run.clone()), w, cx);
        this.delete_confirmed(cx);
    });
    assert!(held.is_held());
    held.release();
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        let models = this.store.models.ready().unwrap();
        assert!(models.iter().all(|m| m.id != made), "the deleted run's model is not reinstalled");
        assert!(this.store.run(&run).is_none());
    });
}

#[test]
fn answers_released_out_of_order_keep_the_newer_list() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let id = drive(&mut app, window, |this, _, _| this.store.datasets.ready().unwrap()[1].id.clone());
    // The first refresh's answer lists the dataset and is held; then it is removed elsewhere.
    let held = api.hold("datasets");
    start(&mut app, window, 1, |this, _, cx| this.refresh_lists(cx));
    held.wait_held();
    api.delete_dataset(&id, false).unwrap();
    // The next refresh answers first: without it.
    start(&mut app, window, 1, |this, _, cx| this.refresh_lists(cx));
    assert!(held.is_held());
    assert!(!drive_held(&mut app, window, &id));
    // The first answer, older, lands now: dropped.
    held.release();
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| assert!(!listed(this, &id), "the older answer is dropped"));
}

#[test]
fn deleting_the_playground_models_run_drops_its_late_answer_and_predicts_with_the_next() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    // Playground on the best model, an image on stage, its boxes in.
    let (first, run) = drive(&mut app, window, |this, _, cx| {
        this.play_dataset_image("ds_3b91", "img_03417", cx);
        let first = this.play_model.clone().expect("the best model is picked");
        let run = this.store.models.ready().unwrap().iter().find(|m| m.id == first).unwrap().run_id.clone();
        (first, run.expect("made by a run"))
    });
    assert!(drive(&mut app, window, |this, _, _| this.play_pred.ready().is_some()));
    // Asked again (the result kept up meanwhile), on a slow device: its answer is made and held.
    let held = api.hold("predict");
    start(&mut app, window, 1, |this, _, cx| {
        this.play_pred_dim = true;
        this.run_predict(cx);
    });
    held.wait_held();
    // The model's run is deleted while it is out; the model list is in again before it lands.
    start(&mut app, window, 1, |this, w, cx| {
        this.ask_delete(DeleteTarget::Run(run.clone()), w, cx);
        this.delete_confirmed(cx);
    });
    assert!(held.is_held(), "the old answer was still out when the delete landed");
    held.release();
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        let next = this.play_model.clone().expect("another model is picked");
        assert_ne!(next, first);
        // The image on stage is predicted with it, and nothing of the deleted model's is shown.
        let shown = this.play_pred.ready().map(|p| p.model_id.clone());
        assert_eq!(shown.as_deref(), Some(next.as_str()));
        assert_eq!(this.play_pred_for.as_ref().map(|(_, m)| m), Some(&next));
        assert!(!this.play_pred_pending && !this.play_pred_dim);
    });
}

/// Playground on the best model with an image on stage and its boxes in: (that model, its run, the
/// model that comes next once it is gone).
fn playground_on_best(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> (String, String, String) {
    let (first, run) = drive(app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
        this.play_dataset_image("ds_3b91", "img_03417", cx);
        let first = this.play_model.clone().expect("the best model is picked");
        let run = this.store.model(&first).unwrap().run_id.clone().expect("made by a run");
        (first, run)
    });
    let next = drive(app, window, |this, _, _| {
        assert_eq!(this.play_pred.ready().map(|p| p.model_id.clone()), Some(first.clone()));
        let mut rest = this.store.models.ready().unwrap().clone();
        rest.retain(|m| m.run_id.as_deref() != Some(&*run));
        let mut store = crate::store::Store::new(this.store.api.clone());
        store.runs = this.store.runs.clone();
        store.models = crate::store::Load::Ready(rest);
        store.best_model(None).map(|m| m.id.clone()).expect("another model is left")
    });
    (first, run, next)
}

/// What Playground shows: the model it runs, the side panel's model, the boxes' model, and whether
/// anything is still out or dimmed; then how often `title` was toasted.
fn playground_state(
    this: &FoloApp,
    title: &str,
) -> (Option<String>, Option<String>, Option<String>, bool, usize) {
    let shown = this.play_shown_model().map(|m| m.id.clone());
    let boxes = this.play_pred.ready().map(|p| p.model_id.clone());
    let busy = this.play_pred_pending || this.play_pred_dim;
    let told = this.toasts.iter().filter(|t| t.title == title).count();
    (this.play_model.clone(), shown, boxes, busy, told)
}

#[test]
fn a_playground_model_deleted_elsewhere_is_replaced_at_the_next_refresh() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let (first, run, next) = playground_on_best(&mut app, window);
    let name = drive(&mut app, window, |this, _, _| this.store.model(&first).unwrap().name.clone());
    // An IoU change asks again; its answer (the old model's boxes) is made, and held.
    let held = api.hold("predict");
    start(&mut app, window, 1, |this, _, cx| this.run_predict(cx));
    held.wait_held();
    // The model's run is deleted through the API (Studio, the CLI); the 5 s refresh finds it gone.
    api.delete_run(&run).unwrap();
    start(&mut app, window, 1, |this, _, cx| this.refresh_lists(cx));
    assert!(held.is_held(), "the old answer was still out when the refresh landed");
    let title = format!("{name} was deleted");
    let now = start(&mut app, window, 1, |this, _, _| playground_state(this, &title));
    assert_eq!(now, (Some(next.clone()), Some(next.clone()), Some(next.clone()), false, 1));
    // The old model's answer lands last: dropped. Another refresh says nothing more.
    held.release();
    app.run_until_idle();
    drive(&mut app, window, |this, _, cx| this.refresh_lists(cx));
    let after = drive(&mut app, window, |this, _, _| playground_state(this, &title));
    assert_eq!(after, (Some(next.clone()), Some(next.clone()), Some(next), false, 1));
    let body = drive(&mut app, window, |this, _, _| {
        this.toasts.iter().find(|t| t.title == title.as_str()).and_then(|t| t.body.clone())
    });
    assert!(body.is_some_and(|b| b.starts_with("Playground now uses ")), "it names the model used now");
}

#[test]
fn a_playground_model_deleted_here_with_a_model_reload_out_is_replaced_at_once() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let (first, run, next) = playground_on_best(&mut app, window);
    let name = drive(&mut app, window, |this, _, _| this.store.model(&first).unwrap().name.clone());
    // A prediction of the old model is out, and so is a model reload (another run just finished).
    let predict = api.hold("predict");
    start(&mut app, window, 1, |this, _, cx| this.run_predict(cx));
    predict.wait_held();
    let reload = api.hold("models");
    let other = start(&mut app, window, 1, |this, _, _| {
        let runs = this.store.runs.ready().unwrap();
        runs.iter().find(|r| r.id != run && r.status == crate::api::RunStatus::Completed).unwrap().id.clone()
    });
    start(&mut app, window, 2, |this, _, cx| this.run_ended(&other, crate::api::RunStatus::Completed, cx));
    reload.wait_held();
    // The model's run is deleted here while both are out: the next model at once.
    start(&mut app, window, 2, |this, w, cx| {
        this.ask_delete(DeleteTarget::Run(run.clone()), w, cx);
        this.delete_confirmed(cx);
    });
    assert!(predict.is_held() && reload.is_held(), "both answers were still out when the delete landed");
    let title = format!("{name} was deleted");
    let now = start(&mut app, window, 2, |this, _, _| playground_state(this, &title));
    assert_eq!(now, (Some(next.clone()), Some(next.clone()), Some(next.clone()), false, 1));
    // The reload from before the delete lands (it lists the old model): dropped. Then the old answer.
    reload.release();
    app.run_until_held(1);
    predict.release();
    app.run_until_idle();
    let after = drive(&mut app, window, |this, _, _| {
        assert!(this.store.model(&first).is_none(), "the old model is not listed again");
        playground_state(this, &title)
    });
    assert_eq!(after, (Some(next.clone()), Some(next.clone()), Some(next), false, 1));
}

#[test]
fn a_cached_dataset_page_removed_elsewhere_says_so_and_never_trains_on_another() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let (id, name) = drive(&mut app, window, |this, w, cx| {
        let ds = this.store.datasets.ready().unwrap()[0].clone();
        this.navigate(Route::Dataset(ds.id.clone()), w, cx);
        (ds.id, ds.name)
    });
    // Its page is read (detail and images cached), then it is removed through the API.
    app.frame(window.into());
    app.run_until_idle();
    assert!(drive(&mut app, window, |this, _, _| this
        .store
        .dataset
        .get(&id)
        .is_some_and(|d| d.ready().is_some())));
    api.delete_dataset(&id, false).unwrap();
    drive(&mut app, window, |this, _, cx| this.refresh_lists(cx));
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.route, Route::Datasets, "its page gave way to the list");
        let said = format!("{name} was removed");
        assert!(this.toasts.iter().any(|t| t.title == said.as_str()), "{said}");
        assert!(!this.store.dataset.contains_key(&id), "what was read of it is dropped");
    });
    // The page's Train on this dataset, from before (a stale action): New run opens on it, says it was
    // removed, and starts nothing until another is picked; it never picks one itself.
    drive(&mut app, window, |this, w, cx| this.open_new_run(Some(&id), w, cx));
    let runs_before = api.runs().unwrap().len();
    drive(&mut app, window, |this, w, cx| {
        assert!(this.nr_dataset_gone());
        assert_eq!(this.nr_gone_name(), name);
        assert!(this.nr_dataset().is_none() && this.form_config(cx).is_none());
        this.submit_new_run(w, cx);
    });
    assert_eq!(api.runs().unwrap().len(), runs_before, "nothing was started");
}

#[test]
fn a_run_or_model_page_removed_elsewhere_gives_way_to_its_list() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let (model, model_name) = drive(&mut app, window, |this, w, cx| {
        let m = this
            .store
            .models
            .ready()
            .unwrap()
            .iter()
            .find(|m| m.run_id.as_deref() == Some("run_52e8"))
            .unwrap();
        let (id, name) = (m.id.clone(), m.name.clone());
        this.navigate(Route::Model(id.clone()), w, cx);
        (id, name)
    });
    app.frame(window.into());
    app.run_until_idle();
    // Its run is deleted through the API: the model page goes at the next refresh, and says so.
    api.delete_run("run_52e8").unwrap();
    drive(&mut app, window, |this, _, cx| this.refresh_lists(cx));
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| {
        assert_eq!(this.route, Route::Models);
        let said = format!("{model_name} was deleted");
        assert!(this.toasts.iter().any(|t| t.title == said.as_str()), "{said}");
        assert!(!this.store.evals.contains_key(&model));
        // The run's page, from an older link: its list, saying so.
        this.navigate(Route::Run("run_52e8".into()), w, cx);
        assert_eq!(this.route, Route::Runs);
    });
    // A run page open while its run is deleted elsewhere.
    let run = drive(&mut app, window, |this, w, cx| {
        let r = this.store.runs.ready().unwrap().iter().find(|r| !r.status.is_active()).unwrap().clone();
        this.navigate(Route::Run(r.id.clone()), w, cx);
        r
    });
    app.frame(window.into());
    api.delete_run(&run.id).unwrap();
    drive(&mut app, window, |this, _, cx| this.refresh_lists(cx));
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.route, Route::Runs);
        assert!(!this.store.run_data.contains_key(&run.id));
        let said = format!("{} was deleted", run.name);
        assert!(this.toasts.iter().any(|t| t.title == said.as_str()), "{said}");
    });
}

#[test]
fn a_runs_history_answering_after_it_was_deleted_elsewhere_stays_gone() {
    // What made `a_run_or_model_page_removed_elsewhere_gives_way_to_its_list` flaky on loaded CI
    // runners: the run page's history request answers after the refreshed list dropped the run.
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let held = api.hold("metrics run_52e8");
    start(&mut app, window, 1, |this, _, cx| this.ensure_run_data("run_52e8", cx));
    held.wait_held();
    api.delete_run("run_52e8").unwrap();
    start(&mut app, window, 1, |this, _, cx| this.refresh_lists(cx));
    // (Read in place: `drive` would wait for the held answer.)
    app.update(|cx| window.update(cx, |this, _, _| assert!(this.store.run("run_52e8").is_none())).unwrap());
    drop(held);
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        assert!(!this.store.run_data.contains_key("run_52e8"), "the late answer does not bring it back");
    });
}
