//! Sprint 13 end-to-end CLI tests: `folo train --batch auto` (CPU fallback + `--auto-batch-target` validation +
//! structured `--json` reporting) and `folo quantize` selective-`f32` calibration policies (`--f32-policy`,
//! `--max-f32-macs`, `--sqnr-threshold`, `--no-p5-head-out`, `--no-pair-bottlenecks`, `--f32-conv`).

use folo_test_support::TestDir;

use super::{folo, s, tiny_run};

/// `folo train --batch auto --auto-batch-target 0.75 --device cpu --json` runs cleanly with CPU fallback, emits
/// an `auto_batch` event and `done.auto_batch` payload, and rejects invalid `--auto-batch-target` usage upfront.
#[test]
fn train_batch_auto_cpu_fallback_and_validation() {
    let tiny = tiny_run();
    let root = TestDir::new("cli-train-auto-batch");
    let project = root.join("runs");

    // `--auto-batch-target` without `--batch auto` is rejected before touching disk.
    let (code, lines, _) = folo(&[
        "train",
        "--json",
        "--data",
        s(&tiny.data),
        "--imgsz",
        "64",
        "--batch",
        "4",
        "--auto-batch-target",
        "0.75",
        "--epochs",
        "1",
        "--project",
        s(&project),
        "--name",
        "bad-target",
    ]);
    assert_eq!(code, 2, "{lines:?}");
    assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_config");
    assert!(!project.join("bad-target").exists());

    // Out-of-range `--auto-batch-target` is rejected.
    let (code, lines, _) = folo(&[
        "train",
        "--json",
        "--data",
        s(&tiny.data),
        "--imgsz",
        "64",
        "--batch",
        "auto",
        "--auto-batch-target",
        "0.99",
        "--epochs",
        "1",
        "--project",
        s(&project),
        "--name",
        "bad-oob",
    ]);
    assert_eq!(code, 2, "{lines:?}");
    assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_config");

    // Valid `--batch auto --auto-batch-target 0.75` on CPU falls back cleanly and reports structured JSON.
    let (code, lines, stderr) = folo(&[
        "train",
        "--json",
        "--data",
        s(&tiny.data),
        "--imgsz",
        "64",
        "--batch",
        "auto",
        "--auto-batch-target",
        "0.75",
        "--epochs",
        "1",
        "--workers",
        "2",
        "--close-mosaic",
        "1",
        "--project",
        s(&project),
        "--name",
        "auto-cpu",
    ]);
    assert_eq!(code, 0, "{stderr}");
    let ab_event = lines
        .iter()
        .find(|l| l["event"] == "auto_batch")
        .unwrap_or_else(|| panic!("expected auto_batch event in {lines:?}"));
    assert_eq!(ab_event["probed"], false);
    assert_eq!(ab_event["fallback"], "cpu");
    assert_eq!(ab_event["chosen_batch"], 16);
    assert!(ab_event["grad_accum"].as_u64().unwrap() >= 1);
    assert_eq!(
        ab_event["effective_batch"].as_u64().unwrap(),
        ab_event["chosen_batch"].as_u64().unwrap() * ab_event["grad_accum"].as_u64().unwrap()
    );
    assert!((ab_event["target_utilization"].as_f64().unwrap() - 0.75).abs() < 1e-9);

    let done = lines.last().unwrap();
    assert_eq!(done["event"], "done");
    assert_eq!(done["auto_batch"]["probed"], false);
    assert_eq!(done["auto_batch"]["chosen_batch"], 16);
    assert_eq!(done["auto_batch"]["grad_accum"], ab_event["grad_accum"]);

    // CPU fallback log line was also emitted.
    assert!(
        lines.iter().any(|l| l["event"] == "log"
            && l["message"].as_str().is_some_and(|m| m.contains("auto_batch") || m.contains("auto-batch"))),
        "expected auto-batch log record: {lines:?}"
    );
}

/// Resuming an auto-batch checkpoint reports `"probed": false, "resumed": true` (never `"fallback": "cpu"`)
/// in both the `auto_batch` event and `done.auto_batch`, and rejects `--auto-batch-target NaN`.
#[test]
fn train_batch_auto_resume_reports_resumed_and_rejects_nan_target() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let tiny = tiny_run();
    let root = TestDir::new("cli-train-auto-batch-resume");
    let run_dir = root.join("auto-resume");
    std::fs::create_dir_all(&run_dir).unwrap();

    // Produce a real 1-of-2 epoch auto-batch checkpoint deterministically (stop right after epoch 1).
    let ds = folo_data::open_splits(&tiny.data, &folo_data::OpenOptions::default(), &folo_data::Split::ALL)
        .unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_after_ep1 = stop.clone();
    let cfg = folo_vision::TrainConfig {
        name: "auto-resume".into(),
        dataset: folo_api::json::path(&tiny.data),
        arch: "folo-n".into(),
        epochs: 3,
        batch: 2,
        auto_batch: true,
        auto_batch_target: 0.75,
        imgsz: 64,
        device: "cpu".into(),
        workers: 0,
        ..folo_vision::TrainConfig::default()
    };
    let data = folo_vision::TrainData {
        train: Arc::new(ds.split(folo_data::Split::Train).clone()),
        val: Arc::new(ds.split(folo_data::Split::Val).clone()),
        classes: ds.classes.clone(),
    };
    let opts = folo_vision::TrainOptions {
        run_dir: Some(run_dir.clone()),
        stop,
        events: Some(Box::new(move |ev| {
            if let folo_api::RunEvent::Epoch(ep) = ev
                && ep.epoch >= 1
            {
                stop_after_ep1.store(true, Ordering::Relaxed);
            }
        })),
        ..folo_vision::TrainOptions::default()
    };
    let summary = folo_vision::Trainer::new(cfg, data, opts).unwrap().run().unwrap();
    assert!(
        (1..3).contains(&summary.epochs_completed),
        "expected partial run, got {}",
        summary.epochs_completed
    );
    let ckpt = run_dir.join("weights/last.folo");
    assert!(ckpt.is_file());

    // `--auto-batch-target NaN` on `--resume` is rejected upfront.
    let (code, lines, _) = folo(&[
        "train",
        "--json",
        "--resume",
        s(&ckpt),
        "--auto-batch-target",
        "NaN",
        "--device",
        "cpu",
        "--workers",
        "0",
    ]);
    assert_eq!(code, 2, "{lines:?}");
    assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_config");

    // Resuming emits `auto_batch` with `resumed: true` and NO `fallback: "cpu"`.
    let (code, lines, stderr) =
        folo(&["train", "--json", "--resume", s(&ckpt), "--device", "cpu", "--workers", "0"]);
    assert_eq!(code, 0, "{stderr}");
    let ab = lines
        .iter()
        .find(|l| l["event"] == "auto_batch")
        .unwrap_or_else(|| panic!("expected auto_batch event in {lines:?}"));
    assert_eq!(ab["probed"], false);
    assert_eq!(ab["resumed"], true);
    assert!(ab.get("fallback").is_none(), "resumed run must not emit fallback=cpu: {ab}");
    assert_eq!(ab["chosen_batch"], 2);

    let done = lines.last().unwrap();
    assert_eq!(done["event"], "done");
    assert_eq!(done["epochs_completed"], 3);
    assert_eq!(done["auto_batch"]["resumed"], true);
    assert!(done["auto_batch"].get("fallback").is_none());
}

/// `folo quantize` wires selective-`f32` calibration policies (`auto`, `p5-tail`, `none`, `--f32-conv`),
/// surfaces `f32_convs` / `f32_mac_fraction` / `f32_layers` in `--json`, rejects invalid inputs, and produces a
/// valid int8 model that evaluates with `folo val --precision int8`.
#[test]
fn quantize_selective_f32_policies_and_validation() {
    let tiny = tiny_run();
    let best = tiny.run.join("weights/best.folo");
    let root = TestDir::new("cli-quantize-selective-f32");

    // 1. Default `--f32-policy auto` keeps at least `neck.out5.cv2` and reports `f32_layers` + `f32_mac_fraction`.
    let out_auto = root.join("auto.folo");
    let (code, lines, stderr) = folo(&[
        "quantize",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&tiny.data),
        "--images",
        "4",
        "--batch",
        "4",
        "--imgsz",
        "64",
        "--workers",
        "1",
        "--output",
        s(&out_auto),
    ]);
    assert_eq!(code, 0, "{stderr}");
    let rec = &lines[0];
    assert_eq!(rec["f32_policy"], "auto");
    assert!(rec["sensitivity_policy"].is_object(), "{rec}");
    let auto_convs: Vec<&str> =
        rec["f32_convs"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert!(auto_convs.contains(&"neck.out5.cv2"), "expected neck.out5.cv2 in {auto_convs:?}");
    assert!(rec["f32_mac_fraction"].as_f64().unwrap() > 0.0, "{rec}");
    assert_eq!(rec["f32_layers"].as_array().unwrap().len(), auto_convs.len());

    // 2. `--f32-policy none` quantizes every eligible conv (`f32_convs: []`, `f32_mac_fraction: 0.0`).
    let out_none = root.join("none.folo");
    let (code, lines, stderr) = folo(&[
        "quantize",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&tiny.data),
        "--images",
        "4",
        "--batch",
        "4",
        "--imgsz",
        "64",
        "--workers",
        "1",
        "--f32-policy",
        "none",
        "--output",
        s(&out_none),
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines[0]["f32_policy"], "none");
    assert!(lines[0]["sensitivity_policy"].is_null());
    assert_eq!(lines[0]["f32_convs"].as_array().unwrap().len(), 0);
    assert_eq!(lines[0]["f32_mac_fraction"].as_f64(), Some(0.0));

    // 3. `--f32-policy none --f32-conv neck.out5.cv2` keeps exactly that explicit layer and marks `explicit: true`.
    let out_explicit = root.join("explicit.folo");
    let (code, lines, stderr) = folo(&[
        "quantize",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&tiny.data),
        "--images",
        "4",
        "--batch",
        "4",
        "--imgsz",
        "64",
        "--workers",
        "1",
        "--f32-policy",
        "none",
        "--f32-conv",
        "neck.out5.cv2",
        "--output",
        s(&out_explicit),
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines[0]["f32_convs"], serde_json::json!(["neck.out5.cv2"]));
    assert_eq!(lines[0]["f32_layers"][0]["explicit"], true);

    // 4. `--f32-policy p5-tail` keeps the whole `neck.out5.*` tail in f32.
    let out_p5 = root.join("p5.folo");
    let (code, lines, stderr) = folo(&[
        "quantize",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&tiny.data),
        "--images",
        "4",
        "--batch",
        "4",
        "--imgsz",
        "64",
        "--workers",
        "1",
        "--f32-policy",
        "p5-tail",
        "--output",
        s(&out_p5),
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines[0]["f32_policy"], "p5-tail");
    let p5_convs: Vec<&str> =
        lines[0]["f32_convs"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert!(p5_convs.len() >= 3 && p5_convs.iter().all(|c| c.starts_with("neck.out5.")), "{p5_convs:?}");

    // 5. Unknown `--f32-conv` fails with exit 2 (`usage`) BEFORE opening the dataset or running calibration,
    // and lists eligible conv examples in hint.
    let out_bad = root.join("bad.folo");
    let (code, lines, _) = folo(&[
        "quantize",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&root.join("nonexistent-dataset")),
        "--images",
        "2",
        "--batch",
        "2",
        "--imgsz",
        "64",
        "--workers",
        "1",
        "--f32-conv",
        "backbone.does_not_exist",
        "--output",
        s(&out_bad),
    ]);
    assert_eq!(code, 2, "{lines:?}");
    let err = &lines.last().unwrap()["error"];
    assert_eq!(err["code"], "usage");
    assert!(err["message"].as_str().unwrap().contains("backbone.does_not_exist"), "{err}");
    assert!(err["hint"].as_str().unwrap().contains("eligible conv"), "{err}");
    assert!(!out_bad.exists());

    // 6. The selectively quantized model runs `folo val --precision int8` cleanly.
    let (code, lines, stderr) = folo(&[
        "val",
        "--json",
        "--model",
        s(&out_auto),
        "--data",
        s(&tiny.data),
        "--imgsz",
        "64",
        "--batch",
        "4",
        "--workers",
        "1",
        "--precision",
        "int8",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines.last().unwrap()["dtype"], "int8");
}
