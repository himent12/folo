//! The Playground against the demo data: which samples it offers, what a local run sends, and how it
//! comes back after losing the server.

use std::sync::Arc;

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
