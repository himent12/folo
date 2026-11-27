//! The Playground against the demo data: which samples it offers, what a local run sends, and how it
//! comes back after losing the server, and how an image opened from another page gets there.

use std::sync::Arc;

use folo_ui::{FileDropEvent, PlatformInput, point, px};

use super::PlayImage;
use super::tests::{drive, open};
use crate::api::{ApiError, PredictInput};
use crate::store::Load;

#[test]
fn validation_samples_come_from_a_dataset_the_model_knows() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, _| {
        // A run's model: its run's dataset.
        let model = this.store.models.ready().unwrap()[0].clone();
        let of_run = this.store.run(model.run_id.as_deref().unwrap()).unwrap().config.dataset_id.clone();
        this.play_model = Some(model.id.clone());
        assert_eq!(this.play_sample_dataset(), Some(of_run.clone()));
        // A file model with the same classes and no run: the first dataset with exactly those classes.
        let first = this
            .store
            .datasets
            .ready()
            .unwrap()
            .iter()
            .find(|d| d.classes == model.classes)
            .unwrap()
            .id
            .clone();
        let mut file = model.clone();
        file.id = "m_file".into();
        file.run_id = None;
        // Another set of classes (a COCO model): no dataset knows them, so no samples at all.
        let mut coco = file.clone();
        coco.id = "m_coco".into();
        coco.classes = vec!["person".into(), "bicycle".into()];
        this.store.models.ready_mut().unwrap().extend([file, coco]);
        this.play_model = Some("m_file".into());
        assert_eq!(this.play_sample_dataset(), Some(first));
        this.play_model = Some("m_coco".into());
        assert_eq!(this.play_sample_dataset(), None, "not the first dataset whatever its classes");
    });
}

#[test]
fn a_local_run_predicts_on_the_dataset_image_already_here() {
    let input = PredictInput::Dataset { dataset_id: "ds_1".into(), image_id: "205".into() };
    let pixels = image::RgbaImage::new(1, 1);
    let image = Arc::new(folo_ui::RenderImage::new(vec![image::Frame::new(pixels)]));
    let bytes = Arc::new(vec![0xFF, 0xD8, 1, 2]);
    let shown = PlayImage {
        name: "bccd / 205".into(),
        input: input.clone(),
        encoded: Some(bytes),
        image,
        width: 1,
        height: 1,
        batch_ref: None,
    };
    // The server reads it from its disk by id; a local run needs no server for it.
    assert_eq!(shown.predict_input(false), input);
    assert!(
        matches!(shown.predict_input(true), PredictInput::File { bytes, .. } if *bytes == [0xFF, 0xD8, 1, 2])
    );
    // Without the file here (the demo's pixels), a local run asks for it as before.
    let demo = PlayImage { encoded: None, ..shown };
    assert_eq!(demo.predict_input(true), input);
}

#[test]
fn a_prediction_that_could_not_reach_the_server_runs_again_once_it_answers() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        this.play_model = Some("m_44c0".into());
        this.play_dataset_image("ds_3b91", "img_03417", cx);
    });
    let down = ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7926");
    drive(&mut app, window, |this, _, cx| {
        assert!(this.play_pred.ready().is_some(), "the first result is in");
        this.play_pred = Load::Failed(down.clone());
        this.store.health = Load::Failed(down.clone());
        // The 2 s refresh, now answered.
        this.store.system = Load::Idle;
        this.ensure_core(cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(!this.store.unreachable());
        assert!(this.play_pred.ready().is_some(), "the boxes are back without a Retry");
    });
}

/// Review round 4 of x1-parity-7: a dataset's files change (its image ids are positions: an id may
/// name another picture now) while the Playground shows one of its images, and again while it is
/// still fetching one. The server predicts on the file the id names now, so the old pixels never
/// stay up under those boxes: they go, with every prediction for them, the image is read again, and
/// an answer from before the change never lands.
#[test]
fn a_playground_image_of_a_dataset_whose_files_changed_is_read_again() {
    use crate::api::mock::MockBackend;
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = super::tests::open_with(api.clone());
    let id = "ds_3b91";
    let on_stage = |app: &mut folo_ui::HeadlessApp| {
        app.update(|cx| {
            window
                .update(cx, |this, _, _| {
                    this.play_image.as_ref().map(|p| match &p.input {
                        PredictInput::Dataset { image_id, .. } => image_id.clone(),
                        _ => "a file".to_string(),
                    })
                })
                .unwrap()
        })
    };
    let forget = |app: &mut folo_ui::HeadlessApp| {
        app.update(|cx| {
            window
                .update(cx, |this, _, cx| {
                    this.forget_dataset_contents(id, cx);
                    assert!(this.play_image.is_none(), "the old pixels stayed up");
                    assert!(this.play_pred.ready().is_none() && !this.play_pred_pending, "their boxes too");
                })
                .unwrap()
        })
    };
    // Shown: on stage with its boxes, then its files change.
    drive(&mut app, window, |this, _, cx| {
        this.play_model = Some("m_44c0".into());
        this.play_dataset_image(id, "img_03417", cx);
    });
    drive(&mut app, window, |this, _, _| assert!(this.play_pred.ready().is_some(), "predicted"));
    let again = api.hold("pixels ds_3b91 img_03417");
    forget(&mut app);
    app.run_until_held(1);
    again.wait_held();
    drop(again);
    app.run_until_idle();
    assert_eq!(on_stage(&mut app).as_deref(), Some("img_03417"), "read again under its id");
    drive(&mut app, window, |this, _, _| assert!(this.play_pred.ready().is_some(), "and predicted again"));
    // Still on its way (A, held) when the files change: asked for again (B); A answers first.
    let (a, b) = (api.hold("pixels ds_3b91 img_03418"), api.hold("pixels ds_3b91 img_03418"));
    app.update(|cx| window.update(cx, |this, _, cx| this.play_dataset_image(id, "img_03418", cx)).unwrap());
    app.run_until_held(1);
    a.wait_held();
    forget(&mut app);
    app.run_until_held(2);
    b.wait_held();
    drop(a);
    app.run_until_held(1);
    assert_eq!(on_stage(&mut app), None, "the picture from before the change landed");
    drop(b);
    app.run_until_idle();
    assert_eq!(on_stage(&mut app).as_deref(), Some("img_03418"), "the picture the id names now");
}

/// Review round 4: a dataset read and an issue check still out on server A when the app switches to
/// server B never answer (`fetch` drops them), so their tickets go with the switch instead of
/// staying for good; the numbers keep counting up.
#[test]
fn reads_left_on_the_old_server_leave_no_tickets_behind() {
    use crate::api::mock::MockBackend;
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = super::tests::open_with(api.clone());
    let id = "ds_3b91";
    let (read, check) = (api.hold("dataset"), api.hold("issues"));
    app.update(|cx| {
        window
            .update(cx, |this, _, cx| {
                this.store.dataset.remove(id);
                this.store.issues.remove(id);
                this.ensure_dataset(id, cx);
                this.ensure_issues(id, cx);
            })
            .unwrap()
    });
    app.run_until_held(2);
    read.wait_held();
    check.wait_held();
    let (ds_seq, issue_seq) = app.update(|cx| {
        window
            .update(cx, |this, _, cx| {
                let seqs = (this.ds_read_seq, this.issue_read_seq);
                assert!(this.ds_reads.contains_key(id) && this.issue_reads.contains_key(id));
                this.set_backend(Arc::new(MockBackend::new(4200, true)), super::ServerMode::Local, cx);
                seqs
            })
            .unwrap()
    });
    drop((read, check));
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        assert!(this.ds_reads.values().all(|t| *t > ds_seq), "server A's read left its ticket");
        assert!(this.issue_reads.values().all(|t| *t > issue_seq), "server A's check left its ticket");
        assert!(this.ds_read_seq >= ds_seq && this.issue_read_seq >= issue_seq, "numbers never go back");
    });
}

#[test]
fn an_outage_only_the_prediction_saw_is_recovered_too() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        this.play_model = Some("m_44c0".into());
        this.play_dataset_image("ds_3b91", "img_03417", cx);
    });
    // The server was down for less than the 2 s refresh: only the prediction's answer saw it.
    let down = ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7926");
    drive(&mut app, window, |this, _, cx| {
        assert!(!this.store.unreachable());
        let key = this.play_pred_for.clone().unwrap();
        this.play_pred_seq += 1;
        this.play_pred_answered(this.play_pred_seq, key, Err(down.clone()));
        assert!(this.play_pred.error().is_some() && this.store.unreachable(), "the outage is noted");
        // The next refresh is answered.
        this.store.system = Load::Idle;
        this.ensure_core(cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(!this.store.unreachable());
        assert!(this.play_pred.ready().is_some(), "the boxes are back without a Retry");
    });
}

/// A small PNG on disk, for the file prompt to answer with.
fn png_file(tag: &str) -> folo_test_support::TestFile {
    png_file_named(tag, "cells.png")
}

fn png_file_named(tag: &str, name: &str) -> folo_test_support::TestFile {
    let file = folo_test_support::TestFile::new(tag, name);
    image::RgbImage::new(8, 8).save(file.path()).unwrap();
    file
}

/// s4 #336: Open image on a model's page runs that model, as its Try in Playground does, and Back returns
/// to that page; on a finished run's page it runs the run's model; elsewhere the Playground keeps its own.
#[test]
fn open_image_runs_the_model_of_the_page_it_was_asked_on_and_back_returns_there() {
    let (mut app, window) = open();
    let file = png_file("desktop-open-image");
    app.platform().answer_prompts(Some(vec![file.path().to_path_buf()]));
    let (current, other) = drive(&mut app, window, |this, _, _| {
        let current = this.play_model.clone().expect("the best model is picked");
        let other = this.store.models.ready().unwrap().iter().find(|m| m.id != current).unwrap().id.clone();
        (current, other)
    });
    let page = super::Route::Model(other.clone());
    drive(&mut app, window, |this, w, cx| this.navigate(page.clone(), w, cx));
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.route, super::Route::Playground);
        assert_eq!(this.play_model.as_ref(), Some(&other), "the page's model, not {current}");
        assert_eq!(this.play_image.as_ref().map(|i| i.name.to_string()).as_deref(), Some("cells.png"));
        let ran = this.play_pred_for.as_ref().map(|(_, m)| m);
        assert_eq!(ran, Some(&other), "the prediction is the page's model's");
    });
    drive(&mut app, window, |this, w, cx| this.back(w, cx));
    drive(&mut app, window, |this, _, _| assert_eq!(this.route, page, "Back returns to the model"));

    // A finished run's page: its model (the run page's Try in Playground).
    let (run, of_run) = drive(&mut app, window, |this, _, _| {
        let r = this.store.runs.ready().unwrap().iter().find(|r| {
            r.status == crate::api::RunStatus::Completed && r.model_id.as_ref().is_some_and(|m| *m != other)
        });
        let r = r.expect("a finished run with another model");
        (r.id.clone(), r.model_id.clone().unwrap())
    });
    drive(&mut app, window, |this, w, cx| this.navigate(super::Route::Run(run.clone()), w, cx));
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.route, super::Route::Playground);
        assert_eq!(this.play_model.as_ref(), Some(&of_run));
    });
    drive(&mut app, window, |this, w, cx| this.back(w, cx));
    drive(&mut app, window, |this, _, _| assert_eq!(this.route, super::Route::Run(run.clone())));

    // Any other page (and Playground itself) keeps the Playground's choice.
    drive(&mut app, window, |this, w, cx| this.navigate(super::Route::Overview, w, cx));
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.route, super::Route::Playground);
        assert_eq!(this.play_model.as_ref(), Some(&of_run), "the Overview has no model of its own");
    });
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    drive(&mut app, window, |this, _, _| assert_eq!(this.play_model.as_ref(), Some(&of_run)));
}

/// s4 #336: a cancelled picker changes nothing: no page change, no model switch, no history.
#[test]
fn a_cancelled_open_image_changes_nothing() {
    let (mut app, window) = open();
    let (current, other) = drive(&mut app, window, |this, _, _| {
        let current = this.play_model.clone().expect("the best model is picked");
        let other = this.store.models.ready().unwrap().iter().find(|m| m.id != current).unwrap().id.clone();
        (current, other)
    });
    let page = super::Route::Model(other);
    drive(&mut app, window, |this, w, cx| this.navigate(page.clone(), w, cx));
    let history = drive(&mut app, window, |this, _, _| this.history.len());
    app.platform().answer_prompts(None);
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.route, page);
        assert_eq!(this.play_model.as_ref(), Some(&current));
        assert_eq!(this.history.len(), history);
        assert!(this.play_image.is_none());
    });
    // With an image on stage: it, its model and its boxes stay, and nothing is asked for.
    let (_, other) = stage_an_image(&mut app, window);
    let before = drive(&mut app, window, |this, w, cx| {
        this.navigate(super::Route::Model(other), w, cx);
        playground(this)
    });
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    drive(&mut app, window, |this, _, _| assert_eq!(playground(this), before));
}

/// What the Playground shows and has asked for: its model, its image (by name and generation) and the
/// number of predictions asked for so far.
fn playground(this: &super::FoloApp) -> (Option<String>, Option<String>, u64, u64) {
    let image = this.play_image.as_ref().map(|i| i.name.to_string());
    (this.play_model.clone(), image, this.play_image_gen, this.play_pred_seq)
}

/// A dataset image on the Playground's stage under its model, predicted; returns that model and
/// another one.
fn stage_an_image(
    app: &mut folo_ui::HeadlessApp,
    window: folo_ui::WindowHandle<super::FoloApp>,
) -> (String, String) {
    let ids = drive(app, window, |this, w, cx| {
        this.navigate(super::Route::Playground, w, cx);
        this.play_dataset_image("ds_3b91", "img_03417", cx);
        let current = this.play_model.clone().expect("the best model is picked");
        let other = this.store.models.ready().unwrap().iter().find(|m| m.id != current).unwrap().id.clone();
        (current, other)
    });
    drive(app, window, |this, _, _| assert!(this.play_pred.ready().is_some(), "predicted"));
    ids
}

/// The last toast's title and body.
fn last_toast(this: &super::FoloApp) -> (String, String) {
    let t = this.toasts.last().expect("a toast");
    (t.title.to_string(), t.body.as_ref().map(|b| b.to_string()).unwrap_or_default())
}

/// s4 #336 review: the page's model left the catalog while the picker was open (deleted elsewhere, or
/// a finished run's checkpoint gone): the image is not opened under another model in its place.
/// Nothing in the Playground changes, nothing is asked for, and a toast says why, naming the model.
#[test]
fn an_image_for_a_page_model_no_longer_available_is_not_opened() {
    let (mut app, window) = open();
    let file = png_file("desktop-open-gone");
    app.platform().answer_prompts(Some(vec![file.path().to_path_buf()]));
    let (current, other) = stage_an_image(&mut app, window);
    let page = super::Route::Model(other.clone());
    let (before, name) = drive(&mut app, window, |this, w, cx| {
        this.navigate(page.clone(), w, cx);
        (playground(this), this.store.model(&other).unwrap().name.clone())
    });
    // The picker opens; the model goes before it answers.
    drive(&mut app, window, |this, w, cx| {
        this.pick_image(w, cx);
        this.store.models.ready_mut().unwrap().retain(|m| m.id != other);
    });
    drive(&mut app, window, |this, _, _| {
        assert_eq!(playground(this), before, "the Playground is left as it was");
        assert_eq!(this.route, page, "and not shown");
        let body = format!("Model {name} is no longer available.");
        assert_eq!(last_toast(this), ("The image was not opened".into(), body));
    });

    // A finished run whose model id is still there, but not its checkpoint in the catalog.
    let (run, of_run) = drive(&mut app, window, |this, _, _| {
        let r = this.store.runs.ready().unwrap().iter().find(|r| {
            r.status == crate::api::RunStatus::Completed && r.model_id.as_ref().is_some_and(|m| *m != current)
        });
        let r = r.expect("a finished run with another model");
        (r.id.clone(), r.model_id.clone().unwrap())
    });
    drive(&mut app, window, |this, w, cx| {
        this.store.models.ready_mut().unwrap().retain(|m| m.id != of_run);
        this.navigate(super::Route::Run(run.clone()), w, cx);
    });
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    drive(&mut app, window, |this, _, _| {
        assert_eq!(playground(this), before);
        assert_eq!(this.route, super::Route::Run(run.clone()));
        assert_eq!(last_toast(this).1, format!("Model {of_run} is no longer available."));
    });
}

/// s4 #336 review: two pickers open at once (the second from the Overview) answered in reverse order:
/// the older answer, coming last, replaces neither the newer image nor its model.
#[test]
fn an_older_open_image_answered_last_is_dropped() {
    let (mut app, window) = open();
    let (newer, older) = (png_file("desktop-open-newer"), png_file_named("desktop-open-older", "older.png"));
    let (current, other) = stage_an_image(&mut app, window);
    // The first picker opens on another model's page, and stays open.
    let first = drive(&mut app, window, |this, w, cx| {
        this.navigate(super::Route::Model(other.clone()), w, cx);
        this.open_request()
    });
    let seq = drive(&mut app, window, |this, w, cx| {
        this.navigate(super::Route::Overview, w, cx);
        this.play_pred_seq
    });
    // The second, from the Overview, is answered first.
    app.platform().answer_prompts(Some(vec![newer.path().to_path_buf()]));
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    let shown = drive(&mut app, window, |this, _, _| {
        assert_eq!(this.play_image.as_ref().map(|i| i.name.to_string()).as_deref(), Some("cells.png"));
        playground(this)
    });
    drive(&mut app, window, |this, w, cx| this.open_play_file(older.path().to_path_buf(), first, w, cx));
    drive(&mut app, window, |this, _, _| {
        assert_eq!(playground(this), shown, "the older answer changed the Playground");
        assert_eq!(this.play_model.as_ref(), Some(&current), "the Overview has no model of its own");
        assert_eq!(this.play_pred_seq, seq + 1, "one prediction, for the newer image");
    });
}

/// s4 #336 review: opening an image on another model's page with an image already in the Playground
/// asks for one prediction, the new image's under the page's model, not one for the old image too.
#[test]
fn opening_an_image_on_another_model_page_asks_for_one_prediction() {
    let (mut app, window) = open();
    let file = png_file("desktop-open-one");
    app.platform().answer_prompts(Some(vec![file.path().to_path_buf()]));
    let (_, other) = stage_an_image(&mut app, window);
    let seq = drive(&mut app, window, |this, w, cx| {
        this.navigate(super::Route::Model(other.clone()), w, cx);
        this.play_pred_seq
    });
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.play_pred_seq, seq + 1, "one prediction");
        assert_eq!(this.play_pred_for, Some((this.play_image_gen, other.clone())), "the new image's");
        assert_eq!(this.play_image.as_ref().map(|i| i.name.to_string()).as_deref(), Some("cells.png"));
    });
}

/// s4 #336: a file dropped on a model's page opens the same way as Open image.
#[test]
fn an_image_dropped_on_a_model_page_runs_that_model() {
    let (mut app, window) = open();
    let file = png_file("desktop-drop-image");
    let other = drive(&mut app, window, |this, _, _| {
        let current = this.play_model.clone().expect("the best model is picked");
        this.store.models.ready().unwrap().iter().find(|m| m.id != current).unwrap().id.clone()
    });
    let page = super::Route::Model(other.clone());
    drive(&mut app, window, |this, w, cx| this.navigate(page.clone(), w, cx));
    app.frame(window.into());
    let at = point(px(640.), px(400.));
    let paths = folo_ui::ExternalPaths::new(vec![file.path().to_path_buf()]);
    app.simulate_input(
        window.into(),
        PlatformInput::FileDrop(FileDropEvent::Entered { position: at, paths }),
    );
    app.simulate_input(window.into(), PlatformInput::FileDrop(FileDropEvent::Submit { position: at }));
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.route, super::Route::Playground);
        assert_eq!(this.play_model.as_ref(), Some(&other));
        assert_eq!(this.play_pred_for.as_ref().map(|(_, m)| m), Some(&other));
    });
    drive(&mut app, window, |this, w, cx| this.back(w, cx));
    drive(&mut app, window, |this, _, _| assert_eq!(this.route, page));
}

/// A 1×1 image named `name`, for an open request's decode to answer with.
fn decoded(name: &str) -> crate::api::ApiResult<PlayImage> {
    let image = Arc::new(folo_ui::RenderImage::new(vec![image::Frame::new(image::RgbaImage::new(1, 1))]));
    let input = PredictInput::File { name: name.into(), bytes: Arc::new(Vec::new()) };
    Ok(PlayImage {
        name: name.to_string().into(),
        input,
        encoded: None,
        image,
        width: 1,
        height: 1,
        batch_ref: None,
    })
}

/// The name of the image on the Playground's stage.
fn on_stage(this: &super::FoloApp) -> Option<String> {
    this.play_image.as_ref().map(|i| i.name.to_string())
}

/// s4 #336 review round 2: an Open image asked for on another model's page is still decoding (held)
/// when a newer one, from the Playground, is applied; the older decode, answering last, replaces
/// neither the newer image nor its model, and leaves nothing loading.
#[test]
fn an_older_open_image_still_decoding_when_a_newer_lands_is_dropped() {
    let (mut app, window) = open();
    let newer = png_file("desktop-open-decoding");
    let (current, other) = stage_an_image(&mut app, window);
    let (release, held) = folo_ui::oneshot::channel::<()>();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(super::Route::Model(other.clone()), w, cx);
        let req = this.open_request();
        let older = async move {
            held.await.ok();
            decoded("older.png")
        };
        this.open_decoding(req, older, w, cx);
        this.navigate(super::Route::Playground, w, cx);
    });
    app.platform().answer_prompts(Some(vec![newer.path().to_path_buf()]));
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    let shown = drive(&mut app, window, |this, _, _| {
        assert_eq!(on_stage(this).as_deref(), Some("cells.png"), "the newer image is applied");
        playground(this)
    });
    release.send(()).unwrap();
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        assert_eq!(playground(this), shown, "the older decode replaced the newer image");
        assert_eq!(this.play_model.as_ref(), Some(&current));
        assert!(this.play_loading.is_none());
    });
}

/// s4 #336 review round 2: a picker opened on another model's page answers after the person chose a
/// validation image in the Playground: the newer choice stays, and so does the model.
#[test]
fn an_open_image_answered_after_a_validation_image_was_chosen_is_dropped() {
    let (mut app, window) = open();
    let file = png_file("desktop-open-then-sample");
    let (current, other) = stage_an_image(&mut app, window);
    let picker = drive(&mut app, window, |this, w, cx| {
        this.navigate(super::Route::Model(other.clone()), w, cx);
        let picker = this.open_request();
        this.navigate(super::Route::Playground, w, cx);
        this.play_dataset_image("ds_3b91", "img_03418", cx);
        picker
    });
    let shown = drive(&mut app, window, |this, _, _| {
        let input = this.play_image.as_ref().map(|i| &i.input);
        assert!(matches!(input, Some(PredictInput::Dataset { image_id, .. }) if image_id == "img_03418"));
        playground(this)
    });
    drive(&mut app, window, |this, w, cx| this.open_play_file(file.path().to_path_buf(), picker, w, cx));
    drive(&mut app, window, |this, _, _| {
        assert_eq!(playground(this), shown, "the older picker replaced the validation image");
        assert_eq!(this.play_model.as_ref(), Some(&current));
    });
}

/// s4 #336 review round 2: an image dropped on model A's page is still decoding when the person
/// clicks model B's Try in Playground: that later, explicit choice stands, and the drop's answer is
/// dropped whole (its image too), leaving nothing loading.
#[test]
fn try_in_playground_after_a_drop_still_decoding_keeps_its_model() {
    let (mut app, window) = open();
    let (b, a) = stage_an_image(&mut app, window);
    let (release, held) = folo_ui::oneshot::channel::<()>();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(super::Route::Model(a.clone()), w, cx);
        let drop = this.open_request();
        let dropped = async move {
            held.await.ok();
            decoded("dropped.png")
        };
        this.open_decoding(drop, dropped, w, cx);
        this.navigate(super::Route::Model(b.clone()), w, cx);
    });
    // Model B's Try in Playground, as its button does it.
    drive(&mut app, window, |this, w, cx| {
        this.play_pred = Load::Idle;
        this.navigate(super::Route::Playground, w, cx);
        this.pick_play_model(b.clone(), cx);
    });
    let shown = drive(&mut app, window, |this, _, _| playground(this));
    release.send(()).unwrap();
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        assert_eq!(playground(this), shown, "the drop replaced Try in Playground's choice");
        assert_eq!(this.play_model.as_ref(), Some(&b));
        assert_eq!(this.route, super::Route::Playground);
        assert!(this.play_loading.is_none());
    });
}

#[test]
fn playground_boxes_labels_class_filter_and_copy_json() {
    use crate::api::ScoredBox;
    let (mut app, window) = open();
    let (_current, other) = stage_an_image(&mut app, window);
    let pred = drive(&mut app, window, |this, _, cx| {
        let mut pred = this.play_pred.ready().expect("staged prediction").clone();
        pred.boxes = vec![
            ScoredBox { class: 0, name: "rbc".into(), score: 0.92, x1: 10., y1: 10., x2: 50., y2: 50. },
            ScoredBox { class: 1, name: "wbc".into(), score: 0.78, x1: 60., y1: 60., x2: 120., y2: 120. },
            ScoredBox { class: 0, name: "rbc".into(), score: 0.12, x1: 2., y1: 2., x2: 8., y2: 8. },
        ];
        this.play_conf = 0.25;
        this.play_pred = Load::Ready(pred.clone());
        cx.notify();
        pred
    });
    app.frame(window.into());
    // Default: boxes + labels on, All classes, 2 detections above 0.25.
    drive(&mut app, window, |this, _, cx| {
        assert!(this.play_boxes && this.play_labels && this.play_class.is_none());
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        for expected in
            ["play-boxes", "play-labels", "play-copy-json", "play-class-all", "play-class-0", "play-class-1"]
        {
            assert!(keys.iter().any(|k| k == expected), "missing {expected} in {keys:?}");
        }
        assert_eq!(this.visible_boxes(&pred, None).len(), 2);
        // Filter to class 1 ("wbc") and copy JSON.
        this.play_class = Some(1);
        assert_eq!(this.visible_boxes(&pred, None).len(), 1);
        this.copy_play_json(cx);
        let (title, body) = last_toast(this);
        assert_eq!(title, "Copied detection JSON");
        assert_eq!(body, "1 detection");
    });
    let clip = app.update(|cx| cx.read_from_clipboard().and_then(|c| c.text()).unwrap_or_default());
    let parsed: serde_json::Value = serde_json::from_str(&clip).expect("valid JSON");
    assert_eq!(parsed["class_filter"], 1);
    assert_eq!(parsed["detections"].as_array().unwrap().len(), 1);
    assert_eq!(parsed["detections"][0]["name"], "wbc");
    // End-to-end (O2o) models omit `"iou"` from copied JSON; NMS (O2m) models include it.
    drive(&mut app, window, |this, _, cx| {
        let id = this.play_model.clone().unwrap();
        let m = this.store.models.ready_mut().unwrap().iter_mut().find(|m| m.id == id).unwrap();
        m.head = crate::api::Head::O2o;
        this.copy_play_json(cx);
    });
    let clip_o2o = app.update(|cx| cx.read_from_clipboard().and_then(|c| c.text()).unwrap_or_default());
    let parsed_o2o: serde_json::Value = serde_json::from_str(&clip_o2o).expect("valid JSON");
    assert!(parsed_o2o.get("iou").is_none(), "O2o models should omit iou: {parsed_o2o}");
    drive(&mut app, window, |this, _, cx| {
        let id = this.play_model.clone().unwrap();
        let m = this.store.models.ready_mut().unwrap().iter_mut().find(|m| m.id == id).unwrap();
        m.head = crate::api::Head::O2m;
        this.copy_play_json(cx);
    });
    let clip_o2m = app.update(|cx| cx.read_from_clipboard().and_then(|c| c.text()).unwrap_or_default());
    let parsed_o2m: serde_json::Value = serde_json::from_str(&clip_o2m).expect("valid JSON");
    assert!(parsed_o2m.get("iou").is_some(), "O2m models should include iou: {parsed_o2m}");
    // Out-of-bounds play_hover never dims remaining boxes, and set_play_conf clears play_hover.
    drive(&mut app, window, |this, w, cx| {
        this.play_class = None;
        this.play_hover = Some(99);
        assert_eq!(this.play_active(w, cx), None, "out-of-bounds hover is ignored");
        let drawn = this.visible_boxes(&pred, this.play_active(w, cx));
        assert!(drawn.iter().all(|b| !b.dim), "remaining boxes must not all be dimmed");
        this.play_hover = Some(1);
        this.set_play_conf(0.85, cx);
        assert_eq!(this.play_hover, None, "confidence change clears hover");
    });
    // Toggling boxes off keeps detection list intact while hiding overlay; switching models resets class filter.
    drive(&mut app, window, |this, _, cx| {
        this.play_boxes = false;
        this.play_labels = false;
        this.play_class = Some(1);
        this.play_hover = Some(0);
        this.pick_play_model(other.clone(), cx);
        assert_eq!(this.play_class, None, "switching model clears stale class filter");
        assert_eq!(this.play_hover, None, "switching model clears hover");
        // Switching backend resets boxes/labels/class/hover to defaults.
        this.play_hover = Some(0);
        this.set_backend(
            Arc::new(crate::api::mock::MockBackend::new(4200, true)),
            super::ServerMode::Local,
            cx,
        );
        assert!(
            this.play_boxes && this.play_labels && this.play_class.is_none() && this.play_hover.is_none()
        );
    });
}

#[test]
fn play_sample_dataset_skips_train_only_datasets() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, _| {
        let model = this.store.models.ready().unwrap()[0].clone();
        this.play_model = Some(model.id.clone());
        assert!(this.play_sample_dataset().is_some());
        for ds in this.store.datasets.ready_mut().unwrap().iter_mut() {
            ds.splits.val = 0;
        }
        assert_eq!(this.play_sample_dataset(), None, "datasets with 0 val images offer no samples");
    });
}

#[test]
fn failed_dataset_load_shows_retry_on_runs_and_toast_on_new_run() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.navigate(super::Route::Runs, w, cx);
        this.store.datasets = Load::Failed(ApiError::new("io_error", "dataset catalog unreadable"));
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "runs-retry-ds"), "Runs should offer Retry datasets: {keys:?}");
        // Ctrl+N / open_new_run explains the failure with an inline Retry action rather than saying "Loading…".
        this.open_new_run(None, w, cx);
        assert!(this.new_run.is_none());
        let t = this.toasts.last().expect("error toast");
        assert_eq!(t.title.as_ref(), "Datasets didn't load");
        assert_eq!(t.body.as_deref(), Some("dataset catalog unreadable"));
        let (label, retry) = t.action.as_ref().expect("Retry action");
        assert_eq!(label.as_ref(), "Retry");
        let retry = retry.clone();
        retry(this, w, cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.datasets.ready().is_some(), "Retry action reloads datasets");
    });
    // When both runs and datasets failed (e.g. outage), Retry reloads all core lists, not datasets alone.
    drive(&mut app, window, |this, w, cx| {
        this.store.datasets = Load::Failed(ApiError::new("server_unreachable", "server unreachable"));
        this.store.runs = Load::Failed(ApiError::new("server_unreachable", "server unreachable"));
        this.open_new_run(None, w, cx);
        let (_, retry) = this.toasts.last().unwrap().action.as_ref().unwrap().clone();
        retry(this, w, cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.datasets.ready().is_some() && this.store.runs.ready().is_some());
    });
}

/// Playtest R3 finding #3: moving the pointer directly downward from row `i` to row `i + 1` in
/// Playground's detections list (and in the Image Inspector's box list) must preserve the new row's
/// hover highlight instead of letting the earlier row's exit callback overwrite it with `None`.
#[test]
fn hovering_downward_across_playground_and_inspector_rows_preserves_highlight() {
    use folo_ui::{Modifiers, MouseMoveEvent, PlatformInput, point, px};

    let move_to =
        |app: &mut folo_ui::HeadlessApp, w: folo_ui::WindowHandle<super::FoloApp>, x: f32, y: f32| {
            app.simulate_input(
                w.into(),
                PlatformInput::MouseMove(MouseMoveEvent {
                    position: point(px(x), px(y)),
                    pressed_button: None,
                    modifiers: Modifiers::default(),
                }),
            );
            app.frame(w.into());
        };

    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        this.play_model = Some("m_44c0".into());
        this.play_dataset_image("ds_3b91", "img_03417", cx);
        this.play_conf = 0.05;
        this.navigate(super::Route::Playground, w, cx);
    });
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }

    // Scan vertically in Playground's right-hand rail (x = 1120) to locate row 0, then verify:
    // 1. Hovering a detection box on the stage and moving directly onto row 0 preserves `Some(0)`
    //    (stage exit must not overwrite row hover ownership).
    // 2. Moving directly downward from row 0 to row 1 preserves `Some(1)`.
    // 3. Hovering a box on the stage and leaving the stage to empty chrome clears `play_hover`.
    let mut row0_y = None;
    for y in (200..780).step_by(14) {
        move_to(&mut app, window, 1120., y as f32);
        if drive(&mut app, window, |this, _, _| this.play_hover) == Some(0) {
            row0_y = Some(y as f32);
            break;
        }
    }
    let y0 = row0_y.expect("located Playground detection row 0");

    let mut stage_hit = None;
    'scan: for y in (160..700).step_by(24) {
        for x in (280..860).step_by(24) {
            move_to(&mut app, window, x as f32, y as f32);
            if drive(&mut app, window, |this, _, _| this.play_hover.is_some() && this.play_hover_from_stage) {
                stage_hit = Some((x as f32, y as f32));
                break 'scan;
            }
        }
    }
    let (sx, sy) = stage_hit.expect("located a detection box on Playground stage");
    move_to(&mut app, window, 1120., y0);
    assert_eq!(
        drive(&mut app, window, |this, _, _| (this.play_hover, this.play_hover_from_stage)),
        (Some(0), false),
        "moving directly from Playground stage to row 0 preserves row 0 hover"
    );

    move_to(&mut app, window, 1120., y0 + 29.);
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.play_hover),
        Some(1),
        "moving directly down from det-0 to det-1 preserves play_hover == Some(1)"
    );

    move_to(&mut app, window, sx, sy);
    assert!(
        drive(&mut app, window, |this, _, _| this.play_hover.is_some() && this.play_hover_from_stage),
        "re-entering stage box assigns stage hover ownership"
    );
    move_to(&mut app, window, 260., 70.);
    assert_eq!(
        drive(&mut app, window, |this, _, _| (this.play_hover, this.play_hover_from_stage)),
        (None, false),
        "leaving stage to empty area clears stage-owned hover"
    );

    // Same invariant in the Image Inspector's box list: find an image with >= 2 boxes, locate
    // label row 0 in the right-hand inspector sidebar, and move directly down into row 1.
    drive(&mut app, window, |this, w, cx| {
        this.navigate(super::Route::Dataset("ds_8f2c".into()), w, cx);
    });
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }
    drive(&mut app, window, |this, w, cx| {
        let idx = this
            .filtered_indices("ds_8f2c")
            .iter()
            .enumerate()
            .find_map(|(pos, &raw)| {
                let im = this.store.images.get("ds_8f2c")?.ready()?.get(raw)?;
                (im.boxes.len() >= 2).then_some(pos)
            })
            .expect("demo dataset has an image with at least 2 boxes");
        this.open_inspector("ds_8f2c".into(), idx, w, cx);
    });
    for _ in 0..2 {
        app.frame(window.into());
        app.run_until_idle();
    }
    let mut insp_first = None;
    for y in (80..720).step_by(12) {
        move_to(&mut app, window, 1120., y as f32);
        if let Some(first_id) =
            drive(&mut app, window, |this, _, _| this.inspector.as_ref().and_then(|i| i.hover_box))
        {
            insp_first = Some((first_id, y as f32));
            break;
        }
    }
    let (first_id, iy0) = insp_first.expect("located first visual Inspector label row");
    move_to(&mut app, window, 1120., iy0 + 32.);
    let second = drive(&mut app, window, |this, _, _| this.inspector.as_ref().and_then(|i| i.hover_box));
    assert!(
        second.is_some_and(|id| id != first_id),
        "moving directly down from first inspector label row ({first_id}) to next row preserves hover (got {second:?})"
    );
}

#[test]
fn playground_shortcuts_iou_noop_detection_keys_and_stage_margin_hover() {
    use folo_ui::{KeyDownEvent, Keystroke, MouseMoveEvent, PlatformInput, point, px};

    let (mut app, window) = open();
    let _ = stage_an_image(&mut app, window);
    app.frame(window.into());

    // 1. Playground zoom shortcuts must be ignored while a Select dropdown is open.
    drive(&mut app, window, |this, _, cx| {
        this.menu_open = Some("play-model");
        cx.notify();
    });
    app.frame(window.into());
    app.simulate_input(
        window.into(),
        PlatformInput::KeyDown(KeyDownEvent { keystroke: Keystroke::parse("=").unwrap(), is_held: false }),
    );
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.play_zoom, 1.0, "zoom shortcut ignored while select menu is open");
        this.menu_open = None;
        cx.notify();
    });
    app.frame(window.into());

    // 2. Pressing Left on `play-iou` when already at its minimum (0.30) does not trigger a redundant predict.
    let seq_before = drive(&mut app, window, |this, w, cx| {
        this.play_iou = 0.30;
        this.focus_key("play-iou", w, cx);
        this.play_pred_seq
    });
    app.frame(window.into());
    app.simulate_input(
        window.into(),
        PlatformInput::KeyDown(KeyDownEvent { keystroke: Keystroke::parse("left").unwrap(), is_held: false }),
    );
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.play_pred_seq, seq_before, "no-op IoU arrow at min must not re-run predict");
    });

    // 3. Detection list rows (`det-*`) support Down / End / Home / Up, scroll `play_side_scroll`,
    //    and boundary Up on `det-0` does not bubble to zoomed stage panning.
    app.resize_window(window.into(), folo_ui::size(px(1280.), px(420.)));
    app.frame(window.into());
    let n_vis = drive(&mut app, window, |this, w, cx| {
        let n = this
            .play_pred
            .ready()
            .map(|p| {
                crate::screens::playground::filter_detections(&p.boxes, this.play_conf, this.play_class).len()
            })
            .unwrap_or(0);
        if n >= 2 {
            this.play_zoom = 2.0;
            this.play_pan = (0., 0.);
            this.focus_key("det-0", w, cx);
        }
        n
    });
    assert!(n_vis >= 2, "demo playground prediction has >= 2 visible detections");
    app.frame(window.into());
    app.simulate_input(
        window.into(),
        PlatformInput::KeyDown(KeyDownEvent { keystroke: Keystroke::parse("up").unwrap(), is_held: false }),
    );
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.play_pan, (0., 0.), "boundary Up on det-0 must not pan zoomed stage");
    });
    for (key, expected) in
        [("down", "det-1".to_string()), ("end", format!("det-{}", n_vis - 1)), ("home", "det-0".to_string())]
    {
        app.simulate_input(
            window.into(),
            PlatformInput::KeyDown(KeyDownEvent {
                keystroke: Keystroke::parse(key).unwrap(),
                is_held: false,
            }),
        );
        app.frame(window.into());
        if key == "end" {
            assert!(app.wants_frame(window.into()), "det-last reveal schedules a follow-up frame");
            app.frame(window.into());
            drive(&mut app, window, |this, _, _| {
                assert!(-this.play_side_scroll.offset().y > px(0.), "End scrolled play_side_scroll");
            });
        }
        drive(&mut app, window, |this, w, cx| {
            assert!(this.fh(expected.clone(), cx).is_focused(w), "{key} focused {expected}");
        });
    }
    app.resize_window(window.into(), folo_ui::size(px(1280.), px(800.)));
    app.frame(window.into());

    // 4. Moving into the 16 px dark margin outside the painted stage `inner` bounds when zoomed
    //    clears `play_hover` instead of hitting a clipped-off box.
    let margin_pt = drive(&mut app, window, |this, _, cx| {
        this.play_zoom = 3.0;
        this.play_hover = Some(0);
        this.play_hover_from_stage = true;
        cx.notify();
        let inner = this.play_view.get().expect("play_view painted");
        point(inner.left() - px(8.), inner.top() + inner.size.height * 0.5)
    });
    app.frame(window.into());
    app.simulate_input(
        window.into(),
        PlatformInput::MouseMove(MouseMoveEvent {
            position: margin_pt,
            pressed_button: None,
            ..Default::default()
        }),
    );
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.play_hover, None, "hovering outside clipped stage inner clears play_hover");
    });
}
