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
