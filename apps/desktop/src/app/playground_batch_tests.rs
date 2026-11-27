//! Headless UI tests for Sprint 13 Playground multi-image / folder batch filmstrip, sequential
//! Run All with cancellation/staleness guards, keyboard navigation, and Batch JSON / CSV export.

use std::sync::Arc;

use folo_ui::{KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput};

use super::Route;
use super::tests::{drive, open};
use crate::api::{ImageSize, PredictResponse, ScoredBox, Timing};
use crate::screens::playground::{BatchItemResult, BatchItemStatus};

fn press(app: &mut folo_ui::HeadlessApp, w: folo_ui::WindowHandle<super::FoloApp>, key: &str) {
    let ks = Keystroke::parse(key).unwrap();
    app.simulate_input(
        w.into(),
        PlatformInput::KeyDown(KeyDownEvent { keystroke: ks.clone(), is_held: false }),
    );
    app.run_until_idle();
    app.frame(w.into());
    app.simulate_input(w.into(), PlatformInput::KeyUp(KeyUpEvent { keystroke: ks }));
    app.run_until_idle();
    app.frame(w.into());
}

fn settle(app: &mut folo_ui::HeadlessApp, w: folo_ui::WindowHandle<super::FoloApp>) {
    for _ in 0..4 {
        app.frame(w.into());
        app.run_until_idle();
    }
}

fn write_test_png(path: &std::path::Path, r: u8, g: u8, b: u8) {
    let img = image::RgbaImage::from_pixel(64, 48, image::Rgba([r, g, b, 255]));
    img.save(path).expect("write test png");
}

fn fake_batch_result(model_id: &str, iou: f32, floor: f32, n: usize) -> BatchItemResult {
    let boxes = (0..n)
        .map(|i| ScoredBox {
            class: (i % 3) as u32,
            name: format!("cls_{}", i % 3),
            score: 0.9 - (i as f32) * 0.05,
            x1: 10.0 + i as f32 * 5.0,
            y1: 12.0,
            x2: 40.0 + i as f32 * 5.0,
            y2: 42.0,
        })
        .collect();
    BatchItemResult {
        pred: PredictResponse {
            model_id: model_id.into(),
            image: ImageSize { width: 64, height: 48 },
            boxes,
            timing: Timing { preprocess_ms: 1.0, inference_ms: 4.0, postprocess_ms: 0.5, total_ms: 5.5 },
            device: "cpu".into(),
            cold: false,
        },
        width: 64,
        height: 48,
        model_id: model_id.into(),
        iou_bits: iou.to_bits(),
        target: None,
        floor,
        per_class_conf: Default::default(),
        content_hash: None,
    }
}

#[test]
fn playground_demo_batch_filmstrip_navigation_run_all_and_exports() {
    let (mut app, window) = open();
    app.update(|cx| {
        super::bind_keys(cx);
        crate::input::bind_keys(cx);
    });
    let dir = folo_test_support::TestDir::new("desktop-play-batch-ui");

    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
    });
    settle(&mut app, window);

    // Load demo batch (6 validation samples from shapes dataset).
    drive(&mut app, window, |this, w, cx| {
        this.load_play_demo_batch(w, cx);
    });
    settle(&mut app, window);

    let total = drive(&mut app, window, |this, _, _| {
        assert!(this.play_batch.is_active(), "batch filmstrip activates for 2+ images");
        assert_eq!(this.play_batch.active, 0);
        assert!(this.play_image.is_some());
        assert!(matches!(this.play_batch.items[0].status, BatchItemStatus::Done(_)));
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        for expected in [
            "play-batch-prev",
            "play-batch-next",
            "play-batch-run-all",
            "play-batch-clear",
            "play-export-batch-json",
            "play-export-batch-csv",
        ] {
            assert!(keys.iter().any(|k| k == expected), "missing batch control {expected} in {keys:?}");
        }
        this.play_batch.items.len()
    });
    assert!(total >= 2, "expected demo batch to load >= 2 images, got {total}");

    // Keyboard navigation: Right / Left when unzoomed, and '[' / ']' even when zoomed.
    press(&mut app, window, "right");
    assert_eq!(drive(&mut app, window, |this, _, _| this.play_batch.active), 1);
    press(&mut app, window, "left");
    assert_eq!(drive(&mut app, window, |this, _, _| this.play_batch.active), 0);
    press(&mut app, window, "]");
    assert_eq!(drive(&mut app, window, |this, _, _| this.play_batch.active), 1);
    press(&mut app, window, "[");
    assert_eq!(drive(&mut app, window, |this, _, _| this.play_batch.active), 0);

    // When zoomed, Right pans instead of stepping batch, while ']' still steps batch!
    drive(&mut app, window, |this, _, cx| {
        this.play_zoom = 2.0;
        cx.notify();
    });
    settle(&mut app, window);
    press(&mut app, window, "right");
    assert_eq!(drive(&mut app, window, |this, _, _| this.play_batch.active), 0, "Right pans when zoomed");
    press(&mut app, window, "]");
    assert_eq!(
        drive(&mut app, window, |this, _, _| this.play_batch.active),
        1,
        "']' steps batch even when zoomed"
    );

    // When palette (a TextInput) is open, '[' / ']' / arrows do not step the batch!
    drive(&mut app, window, |this, w, cx| this.open_palette(w, cx));
    settle(&mut app, window);
    press(&mut app, window, "]");
    assert_eq!(drive(&mut app, window, |this, _, _| this.play_batch.active), 1);
    drive(&mut app, window, |this, w, cx| this.close_palette(w, cx));
    settle(&mut app, window);

    // Run All runs every remaining item sequentially to completion.
    drive(&mut app, window, |this, _, cx| {
        this.start_play_batch(false, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(!this.play_batch.running, "Run All finished cleanly");
        assert!(
            this.play_batch.items.iter().all(|it| matches!(it.status, BatchItemStatus::Done(_))),
            "every batch item completed"
        );
    });

    // Export Batch JSON and Batch CSV via file save prompt to TestDir.
    let json_out = dir.path().join("batch-out.json");
    app.platform().answer_prompts(Some(vec![json_out.clone()]));
    drive(&mut app, window, |this, _, cx| this.export_play_batch_json(cx));
    settle(&mut app, window);
    let parsed: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json_out).expect("batch json exists"))
            .expect("valid json");
    assert_eq!(parsed["mode"], "batch");
    assert_eq!(parsed["total_images"].as_u64(), Some(total as u64));
    assert_eq!(parsed["exported_images"].as_u64(), Some(total as u64));

    let csv_out = dir.path().join("batch-out.csv");
    app.platform().answer_prompts(Some(vec![csv_out.clone()]));
    drive(&mut app, window, |this, _, cx| this.export_play_batch_csv(cx));
    settle(&mut app, window);
    let csv_text = std::fs::read_to_string(&csv_out).expect("batch csv exists");
    let mut lines = csv_text.lines();
    assert_eq!(lines.next(), Some("filename,class,confidence,x1,y1,x2,y2,width,height"));
    assert!(lines.next().is_some(), "demo batch CSV contains detection rows");
}

#[test]
fn playground_folder_and_multifile_pick_and_cancellation_staleness_guards() {
    let (mut app, window) = open();
    let dir = folo_test_support::TestDir::new("desktop-play-batch-files");
    let img_a = dir.path().join("01_alpha.png");
    let img_b = dir.path().join("02_beta.png");
    let img_c = dir.path().join("03_gamma.png");
    write_test_png(&img_a, 200, 40, 40);
    write_test_png(&img_b, 40, 200, 40);
    write_test_png(&img_c, 40, 40, 200);
    std::fs::write(dir.path().join("readme.txt"), b"ignored").unwrap();

    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
    });
    settle(&mut app, window);

    // 1. Folder pick loads all 3 PNGs (skipping readme.txt) into the batch filmstrip.
    app.platform().answer_prompts(Some(vec![dir.path().to_path_buf()]));
    drive(&mut app, window, |this, w, cx| this.pick_play_folder(w, cx));
    settle(&mut app, window);

    let (model_a, model_b) = drive(&mut app, window, |this, _, _| {
        assert_eq!(this.play_batch.items.len(), 3);
        assert_eq!(this.play_batch.items[0].name.as_ref(), "01_alpha.png");
        assert_eq!(this.play_batch.items[1].name.as_ref(), "02_beta.png");
        assert_eq!(this.play_batch.items[2].name.as_ref(), "03_gamma.png");
        let ma = this.play_model.clone().unwrap();
        let mb = this.store.models.ready().unwrap().iter().find(|m| m.id != ma).unwrap().id.clone();
        (ma, mb)
    });

    // 2. Cancellation & staleness guards: simulate an in-flight Run All step and verify EVERY
    //    trigger cancels it cleanly and stale callbacks perform zero writes!
    drive(&mut app, window, |this, w, cx| {
        // Trigger A: explicit cancel_play_batch drops stale callback.
        this.play_batch.running = true;
        this.play_batch.run_gen += 1;
        let stale_gen = this.play_batch.run_gen;
        this.play_batch.items[1].status = BatchItemStatus::Running;
        this.cancel_play_batch();
        assert!(!this.play_batch.running);
        assert!(matches!(this.play_batch.items[1].status, BatchItemStatus::Pending));
        let bogus = fake_batch_result(&model_a, this.play_iou, super::PLAY_CONF_FLOOR, 5);
        this.play_batch_step_answered(stale_gen, 1, &model_a, Ok(bogus.clone()), cx);
        assert!(
            matches!(this.play_batch.items[1].status, BatchItemStatus::Pending),
            "stale cancelled callback must not write into batch item"
        );

        // Trigger B: switching model cancels in-flight batch and invalidates prior results.
        this.play_batch.running = true;
        this.play_batch.run_gen += 1;
        let gen_before_model = this.play_batch.run_gen;
        this.play_batch.items[0].status = BatchItemStatus::Done(bogus.clone());
        this.pick_play_model(model_b.clone(), cx);
        assert!(!this.play_batch.running, "switching model cancels in-flight batch");
        assert!(this.play_batch.run_gen > gen_before_model);
        assert!(matches!(this.play_batch.items[0].status, BatchItemStatus::Pending));
        this.play_batch_step_answered(gen_before_model, 0, &model_a, Ok(bogus.clone()), cx);
        assert!(matches!(this.play_batch.items[0].status, BatchItemStatus::Pending));

        // Trigger C: changing confidence threshold cancels in-flight batch.
        this.play_batch.running = true;
        this.play_batch.run_gen += 1;
        let gen_before_conf = this.play_batch.run_gen;
        this.set_play_conf(0.42, cx);
        assert!(!this.play_batch.running, "changing conf cancels in-flight batch");
        assert!(this.play_batch.run_gen > gen_before_conf);

        // Trigger D: toggling Compare A/B mode cancels in-flight batch.
        this.play_batch.running = true;
        this.play_batch.run_gen += 1;
        let gen_before_ab = this.play_batch.run_gen;
        this.set_play_compare(true, cx);
        assert!(!this.play_batch.running, "toggling A/B mode cancels in-flight batch");
        assert!(this.play_batch.run_gen > gen_before_ab);

        // Trigger E: navigating away from Playground cancels in-flight batch.
        this.play_batch.running = true;
        this.play_batch.run_gen += 1;
        let gen_before_nav = this.play_batch.run_gen;
        this.navigate(Route::Overview, w, cx);
        assert!(!this.play_batch.running, "leaving Playground cancels in-flight batch");
        assert!(this.play_batch.run_gen > gen_before_nav);
    });

    // 3. Multi-file pick with 2 files replaces batch cleanly; picking 1 file exits batch mode.
    app.platform().answer_prompts(Some(vec![img_a.clone(), img_c.clone()]));
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    settle(&mut app, window);
    assert_eq!(drive(&mut app, window, |this, _, _| this.play_batch.items.len()), 2);

    app.platform().answer_prompts(Some(vec![img_b]));
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(!this.play_batch.is_active(), "picking a single file clears batch filmstrip");
        assert_eq!(this.play_image.as_ref().map(|i| i.name.as_ref()), Some("02_beta.png"));
    });

    // 4. Switching backend clears batch cleanly.
    drive(&mut app, window, |this, w, cx| {
        this.load_play_demo_batch(w, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        assert!(this.play_batch.is_active());
        this.set_backend(
            Arc::new(crate::api::mock::MockBackend::new(4200, true)),
            super::ServerMode::Local,
            cx,
        );
        assert!(!this.play_batch.is_active());
        assert!(!this.play_batch.running);
    });
}

#[test]
fn playground_batch_same_basename_corrupt_first_iou_and_dataset_refresh_guards() {
    let (mut app, window) = open();
    app.update(|cx| {
        crate::app::bind_keys(cx);
        crate::input::bind_keys(cx);
    });
    settle(&mut app, window);

    // 1. Corrupt first image on empty Playground still exposes batch strip & keyboard stepping.
    let dir = folo_test_support::TestDir::new("desktop-play-batch-corrupt-first");
    let bad = dir.path().join("01_corrupt.png");
    let good = dir.path().join("02_valid.png");
    std::fs::write(&bad, b"\x89PNG\r\n\x1a\nnot-a-real-png").unwrap();
    write_test_png(&good, 30, 180, 90);

    drive(&mut app, window, |this, w, cx| this.navigate(Route::Playground, w, cx));
    settle(&mut app, window);
    assert!(drive(&mut app, window, |this, _, _| this.play_image.is_none()));

    app.platform().answer_prompts(Some(vec![dir.path().to_path_buf()]));
    drive(&mut app, window, |this, w, cx| this.pick_play_folder(w, cx));
    settle(&mut app, window);

    drive(&mut app, window, |this, _, _| {
        assert!(this.play_batch.is_active(), "batch stays active even when item 0 fails to decode");
        assert!(this.play_image.is_none(), "corrupt item 0 leaves stage without an image");
        assert!(matches!(this.play_batch.items[0].status, BatchItemStatus::Failed(_)));
    });

    // Keyboard ']' steps past corrupt item 0 onto valid item 1 even while play_image is None.
    press(&mut app, window, "]");
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.play_batch.active, 1);
        assert!(this.play_image.is_some(), "stepping from empty failed item 0 loads valid item 1");
        assert_eq!(this.play_image.as_ref().map(|i| i.name.as_ref()), Some("02_valid.png"));
    });

    // Dropping/picking a single corrupt file must NOT wipe the active batch.
    app.platform().answer_prompts(Some(vec![bad.clone()]));
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    settle(&mut app, window);
    assert!(
        drive(&mut app, window, |this, _, _| this.play_batch.is_active()),
        "failed single-file open preserves existing batch"
    );

    // 2. Same-basename files from different directories never cross-contaminate cache or stage.
    let sub_a = dir.path().join("dir_a");
    let sub_b = dir.path().join("dir_b");
    std::fs::create_dir_all(&sub_a).unwrap();
    std::fs::create_dir_all(&sub_b).unwrap();
    let file_a = sub_a.join("frame.png");
    let file_b = sub_b.join("frame.png");
    write_test_png(&file_a, 220, 40, 40);
    write_test_png(&file_b, 40, 40, 220);

    app.platform().answer_prompts(Some(vec![file_a, file_b]));
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    settle(&mut app, window);

    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.play_batch.items.len(), 2);
        assert_eq!(this.play_batch.items[0].name.as_ref(), "dir_a/frame.png");
        assert_eq!(this.play_batch.items[1].name.as_ref(), "dir_b/frame.png");
        let bid = this.play_batch.batch_id;
        assert_eq!(this.play_image.as_ref().and_then(|i| i.batch_ref), Some((bid, 0)));

        // Simulate active advancing to item 1 while item 0 is still staged on screen.
        let model_id = this.play_model.clone().unwrap();
        this.play_batch.active = 1;
        this.play_batch.items[1].status = BatchItemStatus::Pending;
        let meta = crate::screens::playground::PlayPredMeta {
            image_gen: this.play_image_gen,
            model_id: model_id.clone(),
            iou_bits: this.play_iou.to_bits(),
            target: this.play_target,
            floor: super::PLAY_CONF_FLOOR,
            per_class_conf: Default::default(),
        };
        let pred_a = fake_batch_result(&model_id, this.play_iou, super::PLAY_CONF_FLOOR, 3).pred;
        this.sync_stage_pred_to_batch(Some(&meta), &Ok(pred_a));
        assert!(
            matches!(this.play_batch.items[1].status, BatchItemStatus::Pending),
            "item 0 prediction must NEVER write into item 1 while item 0 is still staged"
        );
        assert!(
            matches!(this.play_batch.items[0].status, BatchItemStatus::Done(_)),
            "item 0 prediction records into its own batch_ref slot 0"
        );

        // Reverse check: Run All answering for active item 1 while item 0 is still on stage
        // must NOT overwrite stage play_pred with item 1's detections.
        let stage_boxes_before = this.play_pred.ready().map(|p| p.boxes.len());
        this.play_batch.running = true;
        this.play_batch.run_gen += 1;
        let rgen = this.play_batch.run_gen;
        let res_b = fake_batch_result(&model_id, this.play_iou, super::PLAY_CONF_FLOOR, 7);
        this.play_batch_step_answered(rgen, 1, &model_id, Ok(res_b), cx);
        assert_eq!(
            this.play_pred.ready().map(|p| p.boxes.len()),
            stage_boxes_before,
            "item 1 Run All answer must not paint onto staged item 0"
        );
        assert!(matches!(this.play_batch.items[1].status, BatchItemStatus::Done(_)));

        // 3. IoU slider drag cancels immediately; no-op release preserves completed results.
        this.play_batch.running = true;
        this.play_batch.run_gen += 1;
        this.on_play_iou_changed(cx);
        assert!(
            matches!(this.play_batch.items[1].status, BatchItemStatus::Done(_)),
            "no-op IoU release preserves completed batch items"
        );
        this.set_slider("play-iou", 0.1, cx);
        assert!(!this.play_batch.running, "dragging IoU slider cancels Run All immediately");
        this.on_play_iou_changed(cx);
        assert!(
            matches!(this.play_batch.items[1].status, BatchItemStatus::Pending),
            "committing a different IoU invalidates mismatched batch items"
        );

        // 4. Changing inference target cancels Run All and invalidates batch results.
        this.play_batch.running = true;
        this.play_batch.items[0].status =
            BatchItemStatus::Done(fake_batch_result(&model_id, this.play_iou, super::PLAY_CONF_FLOOR, 2));
        this.set_play_target(Some(crate::local::Target::Cpu), cx);
        assert!(!this.play_batch.running);
        assert!(matches!(this.play_batch.items[0].status, BatchItemStatus::Pending));
        this.set_play_target(None, cx);
    });

    // 5. Dataset refresh clears affected dataset batch so shifted positional IDs never export stale names.
    drive(&mut app, window, |this, w, cx| this.load_play_demo_batch(w, cx));
    settle(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        assert!(this.play_batch.is_active());
        let ds_id = match &this.play_batch.items[0].source {
            crate::screens::playground::PlayBatchSource::Dataset { dataset_id, .. } => dataset_id.clone(),
            _ => panic!("expected dataset demo batch"),
        };
        let _ = this.forget_play_dataset(&ds_id, cx);
        assert!(!this.play_batch.is_active(), "dataset edit clears affected dataset batch");
    });

    // 6. Replacing a batch PNG in place invalidates cached result reuse (content_hash mismatch),
    //    and switching models while a batch item is loading still settles play_loading cleanly.
    let repl_a = dir.path().join("repl_a.png");
    let repl_b = dir.path().join("repl_b.png");
    write_test_png(&repl_a, 200, 20, 20);
    write_test_png(&repl_b, 20, 20, 200);
    app.platform().answer_prompts(Some(vec![repl_a.clone(), repl_b.clone()]));
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    settle(&mut app, window);

    let old_hash = drive(&mut app, window, |this, _, _| {
        assert_eq!(this.play_batch.items.len(), 2);
        match &this.play_batch.items[0].status {
            BatchItemStatus::Done(r) => r.content_hash.expect("file result carries content_hash"),
            other => panic!("expected Done for item 0, got {other:?}"),
        }
    });
    // Step to item 1, replace repl_a.png in place with different pixels, then step back to item 0.
    drive(&mut app, window, |this, _, cx| this.select_play_batch_item(1, cx));
    settle(&mut app, window);
    write_test_png(&repl_a, 10, 240, 10);
    drive(&mut app, window, |this, _, cx| this.select_play_batch_item(0, cx));
    settle(&mut app, window);
    let new_hash = drive(&mut app, window, |this, _, _| match &this.play_batch.items[0].status {
        BatchItemStatus::Done(r) => r.content_hash.expect("updated content_hash"),
        other => panic!("expected fresh Done after replacement, got {other:?}"),
    });
    assert_ne!(old_hash, new_hash, "in-place file replacement must recompute instead of reusing stale cache");

    // Step to item 1 and immediately switch models while item 1 is loading -> settles cleanly.
    drive(&mut app, window, |this, _, cx| {
        let cur = this.play_model.clone().unwrap();
        let other = this.store.models.ready().unwrap().iter().find(|m| m.id != cur).unwrap().id.clone();
        this.select_play_batch_item(1, cx);
        this.pick_play_model(other, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(
            this.play_loading.is_none(),
            "switching models during batch load must not leave play_loading stuck"
        );
        let bid = this.play_batch.batch_id;
        assert_eq!(
            this.play_image.as_ref().and_then(|i| i.batch_ref),
            Some((bid, 1)),
            "selected item 1 still stages cleanly after mid-load model switch"
        );
    });
}

#[test]
fn playtest_r13_compare_failed_banner_model_a_labels_inline_completion_and_mid_load_close() {
    use crate::screens::playground::batch_action_labels;

    // 1. Compare A/B mode explicitly labels Run All & Batch JSON/CSV as Model A.
    assert_eq!(batch_action_labels(false, false), ("Run all", "Batch JSON", "Batch CSV"));
    assert_eq!(batch_action_labels(false, true), ("Re-run all", "Batch JSON", "Batch CSV"));
    assert_eq!(batch_action_labels(true, false), ("Run all (A)", "Batch JSON (A)", "Batch CSV (A)"));
    assert_eq!(batch_action_labels(true, true), ("Re-run all (A)", "Batch JSON (A)", "Batch CSV (A)"));

    let (mut app, window) = open();
    app.update(|cx| {
        super::bind_keys(cx);
        crate::input::bind_keys(cx);
    });
    settle(&mut app, window);

    // 2. Compare A/B renders persistent failure banner when stepping to a corrupt batch image.
    let dir = folo_test_support::TestDir::new("desktop-playtest-r13-batch");
    let good_a = dir.path().join("01-good.png");
    let broken = dir.path().join("02-broken.png");
    let good_b = dir.path().join("03-good.png");
    write_test_png(&good_a, 200, 40, 40);
    std::fs::write(&broken, b"\x89PNG\r\n\x1a\ncorrupt-payload").unwrap();
    write_test_png(&good_b, 40, 200, 40);

    drive(&mut app, window, |this, w, cx| {
        this.navigate(Route::Playground, w, cx);
        this.set_play_compare(true, cx);
    });
    settle(&mut app, window);

    app.platform().answer_prompts(Some(vec![dir.path().to_path_buf()]));
    drive(&mut app, window, |this, w, cx| this.pick_play_folder(w, cx));
    settle(&mut app, window);

    // Step from valid item 0 to corrupt item 1 in Compare A/B mode -> stage keeps rendering cleanly
    // and exposes the persistent retry/skip controls in Compare mode.
    drive(&mut app, window, |this, _, cx| {
        assert!(this.play_compare.enabled);
        assert_eq!(this.play_batch.active, 0);
        assert!(this.play_image.is_some());
        this.select_play_batch_item(1, cx);
    });
    settle(&mut app, window);

    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.play_batch.active, 1);
        assert!(matches!(this.play_batch.items[1].status, BatchItemStatus::Failed(_)));
        assert!(
            this.focus.borrow().contains_key("play-batch-retry-active"),
            "Compare A/B stage must render the persistent failure banner with Retry action"
        );
        assert!(
            this.focus.borrow().contains_key("play-batch-skip-next"),
            "Compare A/B stage must render Next image action on failed batch item"
        );
        // Switching Model A while the corrupt item is selected must preserve its load-failure
        // status so the persistent banner does not disappear while the previous image stays shown.
        let cur = this.play_model.clone().unwrap();
        let other = this.store.models.ready().unwrap().iter().find(|m| m.id != cur).unwrap().id.clone();
        this.pick_play_model(other, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, _| {
        assert!(
            matches!(this.play_batch.items[1].status, BatchItemStatus::Failed(_)),
            "switching Model A must preserve active unstaged load failure"
        );
        assert!(
            this.focus.borrow().contains_key("play-batch-retry-active"),
            "persistent failure banner remains rendered after switching Model A"
        );
    });

    // 3. Clean Run All completion updates inline filmstrip summary with ZERO toasts (never
    //    occluding the bottom-right export row); partial failure DOES emit a Warning toast.
    drive(&mut app, window, |this, w, cx| {
        this.set_play_compare(false, cx);
        this.load_play_demo_batch(w, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        this.toasts.clear();
        this.start_play_batch(true, cx);
    });
    settle(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        assert!(!this.play_batch.running);
        let model_id = this.play_model.clone().unwrap();
        let e2e = this.play_shown_model().is_some_and(|m| this.model_e2e(m));
        let summary = this.play_batch.summary(
            Some(&model_id),
            this.play_iou,
            this.play_target,
            this.play_conf,
            &this.play_thresholds.per_class,
            this.play_class,
            e2e,
        );
        assert_eq!(summary.done, summary.total, "inline filmstrip summary reports all items completed");
        assert_eq!(summary.failed, 0);
        assert!(
            this.toasts.is_empty(),
            "clean batch completion stays inline in the filmstrip instead of blocking export buttons: {:?}",
            this.toasts.iter().map(|t| t.title.as_ref()).collect::<Vec<_>>()
        );
        // Simulate a partial-failure completion -> warning toast is still shown.
        let last = this.play_batch.items.len() - 1;
        this.play_batch.running = true;
        this.play_batch.run_gen += 1;
        let rgen = this.play_batch.run_gen;
        this.play_batch_step_answered(
            rgen,
            last,
            &model_id,
            Err(crate::api::ApiError::new("decode_failed", "corrupt")),
            cx,
        );
        assert!(
            this.toasts.iter().any(|t| t.tone == crate::ui::Tone::Warning),
            "partial batch failure still surfaces a Warning toast"
        );
    });

    // 4. Closing the batch strip while a batch item load is in flight clears play_loading cleanly
    //    for both Dataset and File batch sources.
    drive(&mut app, window, |this, _, cx| {
        this.select_play_batch_item(2, cx);
        assert!(this.play_loading.is_some());
        this.clear_play_batch();
    });
    settle(&mut app, window);
    assert!(
        drive(&mut app, window, |this, _, _| this.play_loading.is_none()),
        "closing dataset batch strip mid-load must clear play_loading once settled"
    );

    app.platform().answer_prompts(Some(vec![good_a.clone(), good_b.clone()]));
    drive(&mut app, window, |this, w, cx| this.pick_image(w, cx));
    settle(&mut app, window);
    drive(&mut app, window, |this, _, cx| {
        this.select_play_batch_item(1, cx);
        assert!(this.play_loading.is_some());
        this.clear_play_batch();
    });
    settle(&mut app, window);
    assert!(
        drive(&mut app, window, |this, _, _| this.play_loading.is_none()),
        "closing file batch strip mid-load must clear play_loading once settled"
    );

    // 5. Blocked intake checks open_image_blocked() BEFORE emitting skipped-file info toasts
    //    for both multi-file batches and mixed 1-supported + unsupported selections.
    let txt = dir.path().join("notes.txt");
    std::fs::write(&txt, b"hello").unwrap();
    drive(&mut app, window, |this, w, cx| {
        this.toasts.clear();
        this.store.models = crate::store::Load::Ready(Vec::new());
        let req = super::playground::OpenRequest { ticket: this.play_open_seq, model: None };
        this.open_play_paths(vec![good_a.clone(), good_b, txt.clone()], req, w, cx);
        assert_eq!(this.toasts.len(), 1, "blocked batch open must emit only the blocked toast");
        assert_eq!(this.toasts[0].title.as_ref(), "Can't open images yet");

        this.toasts.clear();
        let req2 = super::playground::OpenRequest { ticket: this.play_open_seq, model: None };
        this.open_play_paths(vec![good_a, txt], req2, w, cx);
        assert_eq!(
            this.toasts.len(),
            1,
            "blocked 1-supported + unsupported open must emit only the blocked toast"
        );
        assert_eq!(this.toasts[0].title.as_ref(), "Can't open an image yet");
    });
}
