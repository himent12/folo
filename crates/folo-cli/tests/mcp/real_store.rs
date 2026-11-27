//! The real store end to end, and published weights as models.

use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::{Value, json};

use super::harness::{ALL_TOOLS, Client, check_tools, checked, init_old, spawn, spawn_with_weights, tmp_dir};

/// Polls `run_status` until `done` holds (at most `secs` seconds) and returns the last status.
fn wait_run(c: &mut Client, run_id: &str, secs: u64, done: impl Fn(&Value) -> bool) -> Value {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        let s = c.ok("run_status", json!({ "run_id": run_id, "log_lines": 5 }));
        if done(&s) {
            return s;
        }
        assert!(Instant::now() < deadline, "run {run_id} did not get there: {s}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// The whole agent flow on the **real** store (CPU, 64 px): synthetic dataset → train (subscribed) → models →
/// val → predict → sample grid → export, then stop / resume a second run, and a graceful shutdown on disconnect
/// that the next session sees.
#[test]
fn real_store_end_to_end() {
    let dir = tmp_dir("real");
    let data_dir = dir.to_str().unwrap();
    let mut c = spawn(&["--data-dir", data_dir, "--device", "cpu"], None);
    init_old(&mut c);
    let schemas = check_tools(&mut c, &ALL_TOOLS);
    let info = checked(&mut c, &schemas, "folo_info", json!({}));
    assert_eq!(info["mode"], "embedded");
    assert_eq!(c.ok("list_runs", json!({}))["total"], 0);
    c.tool_error("val", json!({ "model_id": "m_1", "dataset_id": "ds_1" }), "model_not_found");

    let ds = checked(
        &mut c,
        &schemas,
        "create_synthetic_dataset",
        json!({ "name": "tiny", "images": 12, "imgsz": 64, "classes": ["circle", "square"], "seed": 5 }),
    );
    let ds_id = ds["id"].as_str().unwrap().to_string();
    assert_eq!((ds["splits"]["train"].as_u64(), ds["splits"]["val"].as_u64()), (Some(10), Some(2)));
    let tiny = |epochs: u32, name: &str| {
        json!({ "dataset_id": ds_id, "name": name, "epochs": epochs, "batch": 4, "imgsz": 64, "device": "cpu",
                "augment": { "close_mosaic": 1 } })
    };

    // A GPU this machine does not have: the same code as the CLI, the hint names the devices it has.
    let mut gpu7 = tiny(1, "nogpu");
    gpu7["device"] = json!("gpu:7");
    let e = c.tool_error("train", gpu7, "device_not_found");
    assert!(
        e["hint"].as_str().unwrap().contains("cpu") && !e["message"].as_str().unwrap().contains("--device")
    );
    // Mixed precision this server cannot train is refused at once (not started, then failed), with a hint for an MCP
    // client: the switch, or how to restart `folo mcp` with tensor cores.
    let mut amp = tiny(1, "amp");
    amp["amp"] = json!(true);
    let e = c.tool_error("train", amp, "not_supported");
    let hint = e["hint"].as_str().unwrap();
    assert!(
        hint.contains("`amp: false`") && hint.contains("`folo mcp`") && hint.contains("--tensor-cores"),
        "{e}"
    );
    assert_eq!(c.ok("list_runs", json!({}))["total"], 0, "nothing was started");

    let t = checked(&mut c, &schemas, "train", tiny(1, "tiny-n"));
    let run_id = t["run_id"].as_str().unwrap().to_string();
    let run_uri = format!("folo://runs/{run_id}");
    let done =
        wait_run(&mut c, &run_id, 120, |s| !matches!(s["status"].as_str(), Some("queued" | "running")));
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(done["last_epoch"]["epoch"], 1);
    assert!(!done["log"].as_array().unwrap().is_empty());
    let model_id = done["model_id"].as_str().unwrap().to_string();

    let models = checked(&mut c, &schemas, "list_models", json!({}));
    let ids: Vec<&str> =
        models["items"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert!(
        ids.contains(&model_id.as_str()) && ids.contains(&format!("{model_id}-last").as_str()),
        "{ids:?}"
    );
    // A fine-tune takes the weights' architecture: an explicit other `arch` is refused, as the CLI refuses
    // `--model` with `--weights`.
    let mut other = tiny(1, "other-arch");
    other["weights"] = json!(model_id);
    other["arch"] = json!("folo-s");
    let e = c.tool_error("train", other, "invalid_config");
    assert!(e["message"].as_str().unwrap().contains("folo-n"), "{e}");

    let val = checked(&mut c, &schemas, "val", json!({ "model_id": model_id, "dataset_id": ds_id }));
    assert_eq!((val["split"].as_str(), val["images"].as_u64()), (Some("val"), Some(2)));
    assert_eq!(val["dataset_name"], "tiny");
    assert_eq!(val["imgsz"], 64);
    assert!(val["timing"]["per_image_ms"].as_f64().unwrap() > 0.0);
    let on_train = checked(
        &mut c,
        &schemas,
        "val",
        json!({ "model_id": model_id, "dataset_id": ds_id, "split": "train", "conf": 0.25, "iou": 0.6 }),
    );
    assert_eq!(on_train["images"], 10);
    c.tool_error(
        "val",
        json!({ "model_id": model_id, "dataset_id": ds_id, "split": "test" }),
        "invalid_dataset",
    );

    let imgs = checked(
        &mut c,
        &schemas,
        "list_dataset_images",
        json!({ "dataset_id": ds_id, "split": "val", "limit": 1 }),
    );
    let image_id = imgs["items"][0]["image_id"].as_str().unwrap().to_string();
    let pred = checked(
        &mut c,
        &schemas,
        "predict",
        json!({ "model_id": model_id, "dataset_id": ds_id, "image_id": image_id, "conf": 0.001 }),
    );
    assert!(!pred["boxes"].as_array().unwrap().is_empty(), "{pred}");
    assert_eq!((pred["image"]["width"].as_u64(), pred["device"].as_str()), (Some(64), Some("cpu")));
    let path = dir.join("datasets/tiny").join(imgs["items"][0]["file"].as_str().unwrap());
    let by_path = checked(
        &mut c,
        &schemas,
        "predict",
        json!({ "model_id": model_id, "image_path": path.to_str().unwrap(), "conf": 0.001 }),
    );
    assert_eq!(by_path["boxes"], pred["boxes"], "the same image gives the same boxes");

    // The run's validation samples as an annotated JPEG grid.
    let grid = c.result("resources/read", json!({ "uri": format!("{run_uri}/samples/1") }));
    assert_eq!(grid["contents"][0]["mimeType"], "image/jpeg");
    let jpeg = base64::engine::general_purpose::STANDARD
        .decode(grid["contents"][0]["blob"].as_str().unwrap())
        .unwrap();
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
    let missing = c.request("resources/read", json!({ "uri": format!("{run_uri}/samples/9") }));
    assert!(missing.get("error").is_some(), "{missing}");

    let x = checked(&mut c, &schemas, "export", json!({ "model_id": model_id, "format": "folo-f16" }));
    assert_eq!(std::fs::metadata(x["path"].as_str().unwrap()).unwrap().len(), x["bytes"].as_u64().unwrap());
    let x = checked(&mut c, &schemas, "export", json!({ "model_id": model_id, "format": "onnx" }));
    assert!(x["path"].as_str().unwrap().ends_with(".onnx"), "{x}");
    assert_eq!(std::fs::metadata(x["path"].as_str().unwrap()).unwrap().len(), x["bytes"].as_u64().unwrap());
    // The ONNX switches: f16 weights make the file about half the size, and name it -f16 (-decoded-f16 decoded).
    let h = checked(
        &mut c,
        &schemas,
        "export",
        json!({ "model_id": model_id, "format": "onnx", "half": true, "decode": true, "dynamic": true }),
    );
    assert!(h["path"].as_str().unwrap().ends_with("-decoded-f16.onnx"), "{h}");
    assert!(h["bytes"].as_u64().unwrap() < x["bytes"].as_u64().unwrap() * 3 / 4, "{h} vs {x}");
    c.tool_error(
        "export",
        json!({ "model_id": model_id, "format": "folo-f16", "half": true }),
        "invalid_arguments",
    );

    // Stop a longer run after its first epoch, resume it (same id, from its checkpoint), and disconnect while it
    // trains: the shutdown stops it gracefully and the next session lists it as stopped.
    let long = checked(&mut c, &schemas, "train", tiny(200, "tiny-long"));
    let long_id = long["run_id"].as_str().unwrap().to_string();
    // Followed through its resource subscription (fed by the run's live events; the first notice comes within the
    // watch interval, while it trains). A subscription sees only later changes, so it is made on a run that keeps
    // training: a one-epoch run can end before the subscription's first look on a busy machine, and then nothing
    // changes again.
    let long_uri = format!("folo://runs/{long_id}");
    c.result("resources/subscribe", json!({ "uri": long_uri }));
    let update =
        c.notification(Duration::from_secs(30), |n| n["method"] == "notifications/resources/updated");
    assert_eq!(update["params"]["uri"], long_uri);
    c.result("resources/unsubscribe", json!({ "uri": long_uri }));
    let done_epochs = |s: &Value| s["last_epoch"]["epoch"].as_u64().unwrap_or(0);
    wait_run(&mut c, &long_id, 60, |s| done_epochs(s) >= 1);
    checked(&mut c, &schemas, "stop_run", json!({ "run_id": long_id }));
    let stopped = wait_run(&mut c, &long_id, 60, |s| s["status"] == "stopped");
    let epoch = done_epochs(&stopped);
    assert!(epoch >= 1, "{stopped}");
    c.tool_error("train", json!({ "resume_run_id": long_id, "epochs": 3 }), "invalid_config");
    c.tool_error("train", json!({ "resume_run_id": run_id }), "run_not_resumable");
    let resumed = checked(&mut c, &schemas, "train", json!({ "resume_run_id": long_id }));
    assert_eq!(resumed, json!({ "run_id": long_id, "status": "queued" }));
    wait_run(&mut c, &long_id, 60, |s| s["status"] == "running" && done_epochs(s) > epoch);
    let cmp = checked(&mut c, &schemas, "compare_runs", json!({ "run_ids": [run_id, long_id] }));
    assert_eq!(cmp["config_diff"]["epochs"], json!([1, 200]));
    let running = checked(&mut c, &schemas, "list_runs", json!({ "status": "running" }));
    assert_eq!(running["total"], 1);
    c.finish();

    let mut c = spawn(&["--data-dir", data_dir, "--device", "cpu"], None);
    init_old(&mut c);
    let after = c.ok("run_status", json!({ "run_id": long_id, "log_lines": 200 }));
    assert_eq!(after["status"], "stopped", "{after}");
    // Stopped by the shutdown, not found active by the next start.
    let log = after["log"].as_array().unwrap();
    assert!(log.iter().all(|l| !l["message"].as_str().unwrap().contains("the server stopped")), "{after}");
    assert!(done_epochs(&after) > epoch, "{after}");
    assert_eq!(c.ok("list_runs", json!({}))["total"], 2);
    assert!(c.ok("list_models", json!({}))["total"].as_u64().unwrap() >= 2);
    c.finish();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Published weights in the `folo weights` cache (`FOLO_WEIGHTS_DIR`) are models in embedded mode: listed with
/// `source: published` and the registry's `about`, and `train { weights: "folo-n-coco" }` fine-tunes from them.
/// (That such a run trains to the end is folo-server's `published_weights_list_as_models_and_fine_tune_by_name`.)
#[test]
fn published_weights_are_models() {
    let dir = tmp_dir("published");
    let weights = dir.join("weights");
    std::fs::create_dir_all(&weights).unwrap();
    let home = dir.join("home");
    let mut c =
        spawn_with_weights(&["--data-dir", home.to_str().unwrap(), "--device", "cpu"], None, &weights);
    init_old(&mut c);
    let ds = c.ok(
        "create_synthetic_dataset",
        json!({ "name": "pub", "images": 8, "imgsz": 64, "classes": ["circle", "square"], "seed": 5 }),
    );
    let ds_id = ds["id"].as_str().unwrap().to_string();
    // Not downloaded yet: the hint is the pull command.
    let cfg = json!({ "dataset_id": ds_id, "weights": "folo-n-coco", "epochs": 1, "batch": 4, "imgsz": 64,
                      "device": "cpu", "augment": { "close_mosaic": 1 } });
    let e = c.tool_error("train", cfg.clone(), "model_not_found");
    assert!(e["hint"].as_str().unwrap().contains("folo weights pull folo-n-coco"), "{e}");
    // "Downloaded": a random folo-n stands in for the release asset.
    let mut m = folo_vision::FoloModel::with_rng(
        folo_vision::ModelConfig::new("n", 2).unwrap(),
        &mut folo_core::rng::Rng::seed(9),
    )
    .unwrap();
    m.set_classes(vec!["circle".into(), "square".into()]).unwrap();
    m.save(weights.join("folo-n-coco.folo")).unwrap();
    let models = c.ok("list_models", json!({}));
    let p = models["items"].as_array().unwrap().iter().find(|m| m["id"] == "folo-n-coco").cloned();
    let p = p.unwrap_or_else(|| panic!("not listed: {models}"));
    assert_eq!((p["source"].as_str(), p["arch"].as_str()), (Some("published"), Some("folo-n")), "{p}");
    assert!(p["about"].as_str().unwrap().contains("COCO"), "{p}");
    let t = c.ok("train", cfg);
    let status = c.ok("run_status", json!({ "run_id": t["run_id"] }));
    assert_eq!(status["config"]["weights"], "folo-n-coco", "{status}");
    c.finish();
}

/// MCP `predict` and `predict_batch` support SAHI tiling (`tile`) and TTA (`tta`), reject invalid combinations, and
/// `register_dataset` + `list_dataset_issues` seamlessly handle Pascal VOC datasets and split-leakage warnings.
#[test]
fn predict_sahi_tta_and_voc_leakage() {
    let dir = tmp_dir("sahi-tta-voc");
    let weights = dir.join("weights");
    std::fs::create_dir_all(&weights).unwrap();
    let home = dir.join("home");
    std::fs::create_dir_all(&home).unwrap();

    // Create a tiny YOLO dataset, convert to Pascal VOC, and inject a cross-split duplicate image.
    let yolo_dir = dir.join("yolo");
    let syn = folo_data::synthetic::SyntheticConfig {
        images: 6,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        seed: 7,
        vary_aspect: false,
        val_fraction: 0.34,
        ..Default::default()
    };
    folo_data::synthetic::write_yolo(&syn, &yolo_dir, folo_data::ImageFormat::Png).unwrap();
    let loaded = folo_data::open(&yolo_dir).unwrap();
    let voc_dir = dir.join("voc");
    folo_data::convert::convert(&loaded, &voc_dir, folo_data::Format::Voc).unwrap();
    // Copy the first train image/annotation into JPEGImages & Annotations and list it in val.txt too (split leakage).
    let train_txt = std::fs::read_to_string(voc_dir.join("ImageSets/Main/train.txt")).unwrap();
    let first_stem = train_txt.lines().next().unwrap().trim().to_string();
    let mut val_txt = std::fs::read_to_string(voc_dir.join("ImageSets/Main/val.txt")).unwrap();
    val_txt.push_str(&format!("{first_stem}\n"));
    std::fs::write(voc_dir.join("ImageSets/Main/val.txt"), val_txt).unwrap();

    // Save a tiny 2-class model as a published weight so we don't need to train.
    let mut m = folo_vision::FoloModel::with_rng(
        folo_vision::ModelConfig::new("n", 2).unwrap(),
        &mut folo_core::rng::Rng::seed(11),
    )
    .unwrap();
    m.set_classes(vec!["circle".into(), "square".into()]).unwrap();
    m.save(weights.join("folo-n-coco.folo")).unwrap();

    let mut c =
        spawn_with_weights(&["--data-dir", home.to_str().unwrap(), "--device", "cpu"], None, &weights);
    init_old(&mut c);
    let schemas = check_tools(&mut c, &ALL_TOOLS);

    // Register Pascal VOC dataset with `format: "auto"` and confirm split-leakage warning surfaces cleanly.
    let ds = checked(
        &mut c,
        &schemas,
        "register_dataset",
        json!({ "name": "voc-leaked", "path": voc_dir.to_str().unwrap(), "format": "auto" }),
    );
    let ds_id = ds["id"].as_str().unwrap().to_string();
    assert!(ds["splits"]["train"].as_u64().unwrap() > 0 && ds["splits"]["val"].as_u64().unwrap() > 0);
    let issues = checked(
        &mut c,
        &schemas,
        "list_dataset_issues",
        json!({ "dataset_id": ds_id, "kind": "duplicate_image" }),
    );
    assert!(
        issues["items"]
            .as_array()
            .unwrap()
            .iter()
            .any(|i| { i["message"].as_str().is_some_and(|m| m.starts_with("split leakage (")) }),
        "expected split leakage warning in VOC dataset issues: {issues}"
    );

    // `predict` with SAHI tiling (`tile`) and TTA (`tta`) on dataset image and on file path.
    let imgs = checked(&mut c, &schemas, "list_dataset_images", json!({ "dataset_id": ds_id, "limit": 2 }));
    let image_id = imgs["items"][0]["image_id"].as_str().unwrap().to_string();
    let img_path = voc_dir.join(imgs["items"][0]["file"].as_str().unwrap());

    let tiled = checked(
        &mut c,
        &schemas,
        "predict",
        json!({
            "model_id": "folo-n-coco",
            "dataset_id": ds_id,
            "image_id": image_id,
            "conf": 0.001,
            "max_det": 1,
            "tile": 32,
            "tile_overlap": 0.25,
            "tile_full_image": true
        }),
    );
    assert_eq!(tiled["image"]["width"], 64);
    assert!(tiled["boxes"].as_array().unwrap().len() <= 1, "top-level max_det must cap tiled boxes: {tiled}");

    let tta = checked(
        &mut c,
        &schemas,
        "predict",
        json!({
            "model_id": "folo-n-coco",
            "image_path": img_path.to_str().unwrap(),
            "conf": 0.001,
            "tta": true,
            "tta_scales": [1.0],
            "tta_hflip": true
        }),
    );
    assert_eq!(tta["image"]["width"], 64);

    // `predict_batch` supports `tile` and `tta` too.
    let batch_tile = checked(
        &mut c,
        &schemas,
        "predict_batch",
        json!({
            "model_id": "folo-n-coco",
            "paths": [img_path.to_str().unwrap()],
            "conf": 0.001,
            "tile": 32,
            "tile_full_image": false
        }),
    );
    assert_eq!((batch_tile["processed"].as_u64(), batch_tile["failed_total"].as_u64()), (Some(1), Some(0)));

    let batch_tta = checked(
        &mut c,
        &schemas,
        "predict_batch",
        json!({
            "model_id": "folo-n-coco",
            "paths": [img_path.to_str().unwrap()],
            "conf": 0.001,
            "tta": true,
            "tta_scales": [1.0],
            "tta_hflip": false
        }),
    );
    assert_eq!((batch_tta["processed"].as_u64(), batch_tta["failed_total"].as_u64()), (Some(1), Some(0)));

    // Combining `tile` and `tta` fails with `invalid_arguments`.
    c.tool_error(
        "predict",
        json!({
            "model_id": "folo-n-coco",
            "image_path": img_path.to_str().unwrap(),
            "tile": 32,
            "tta": true
        }),
        "invalid_arguments",
    );
    c.finish();
}
