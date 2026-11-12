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
