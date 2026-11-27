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
