//! Sprint 12 end-to-end CLI tests: `folo predict --track` (ByteTrack), `folo val --diagnostics` / `--worst`,
//! and `folo export --dynamic-batch` / `--dynamic-hw` + FOLO v2-Iso (`folo2-*`) ONNX export.

use std::collections::BTreeMap;
use std::path::Path;

use folo_nn::Module as _;
use folo_nn::io::safetensors::write;
use folo_test_support::TestDir;
use folo_vision::model::v2::{FoloV2Config, FoloV2Model};

use super::{folo, s, tiny_run};

/// `folo predict --track` persists `track_id`, `age`, and `hits` across multi-frame directory inputs, annotates
/// saved images with `#<id>`, validates tracker thresholds upfront, and exposes all `--track*` flags in
/// `folo describe predict`.
#[test]
fn predict_track_persists_ids_and_validates_thresholds() {
    let tiny = tiny_run();
    let best = tiny.run.join("weights/best.folo");
    let src_img = tiny.data.join("images/val").read_dir().unwrap().next().unwrap().unwrap().path();

    let root = TestDir::new("cli-predict-track");
    let frames = root.join("frames");
    let out_dir = root.join("saved");
    std::fs::create_dir_all(&frames).unwrap();
    for i in 0..3 {
        std::fs::copy(&src_img, frames.join(format!("frame_{i:02}.png"))).unwrap();
    }

    // Use a very low confidence threshold so the tiny 1-epoch model emits detections across identical frames.
    let (code, lines, stderr) = folo(&[
        "predict",
        "--json",
        "--model",
        s(&best),
        s(&frames),
        "--conf",
        "0.001",
        "--track",
        "--track-high-thresh",
        "0.001",
        "--track-low-thresh",
        "0.0005",
        "--track-new-thresh",
        "0.001",
        "--track-min-hits",
        "1",
        "--save",
        s(&out_dir),
    ]);
    assert_eq!(code, 0, "{stderr}");
    let preds: Vec<_> = lines.iter().filter(|l| l["event"] == "prediction").collect();
    assert_eq!(preds.len(), 3, "{lines:?}");
    for (idx, p) in preds.iter().enumerate() {
        assert_eq!(p["frame_index"].as_u64(), Some(idx as u64), "{p}");
        assert!(p["saved"].is_string(), "{p}");
        let saved = Path::new(p["saved"].as_str().unwrap());
        assert!(saved.is_file(), "saved image missing: {}", saved.display());
    }
    let first_boxes = preds[0]["boxes"].as_array().unwrap();
    assert!(!first_boxes.is_empty(), "expected tracked boxes on frame 0: {}", preds[0]);
    for b in first_boxes {
        assert!(b["track_id"].as_u64().unwrap() >= 1, "{b}");
        assert!(b["age"].as_u64().unwrap() >= 1, "{b}");
        assert!(b["hits"].as_u64().unwrap() >= 1, "{b}");
    }
    // Identical frames keep track IDs persistent and increment `hits`.
    let first_id = first_boxes[0]["track_id"].as_u64().unwrap();
    let last_boxes = preds[2]["boxes"].as_array().unwrap();
    let matched = last_boxes.iter().find(|b| b["track_id"].as_u64() == Some(first_id));
    assert!(
        matched.is_some_and(|b| b["hits"].as_u64().unwrap() >= 2),
        "expected track_id {first_id} to persist across identical frames: {last_boxes:?}"
    );
    let done = lines.last().unwrap();
    assert_eq!(done["event"], "done");
    assert_eq!(done["tracked"], true);
    assert!(done["unique_tracks"].as_u64().unwrap() >= 1, "{done}");

    // Upfront validation: invalid threshold combinations fail with exit 2 (`invalid_config`) before loading weights.
    let (code, err_lines, _) = folo(&[
        "predict",
        "--json",
        "--model",
        "nonexistent.folo",
        s(&frames),
        "--track",
        "--track-low-thresh",
        "0.8",
        "--track-high-thresh",
        "0.5",
    ]);
    assert_eq!(code, 2);
    assert_eq!(err_lines.last().unwrap()["error"]["code"], "invalid_config");

    // `folo describe predict` exposes `--track` and its peer flags with `requires: ["track"]`.
    let (code, desc, stderr) = folo(&["describe", "predict", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let args = desc[0]["args"].as_array().unwrap();
    assert!(args.iter().any(|a| a["long"] == "--track"), "missing --track in describe");
    let high = args.iter().find(|a| a["long"] == "--track-high-thresh").unwrap();
    assert_eq!(high["requires"], serde_json::json!(["--track"]));
}

/// `folo val --diagnostics` / `--worst` emits conservation-law-exact error diagnostics and worst-failing-image
/// rankings, rejects invalid diagnostic flags upfront, and exposes them in `folo describe val`.
#[test]
fn val_diagnostics_emits_error_breakdown_and_worst_images() {
    let tiny = tiny_run();
    let best = tiny.run.join("weights/best.folo");

    // `--worst` implies `--diagnostics`, so `--worst 2 --diag-conf 0.05` works without `--diagnostics`.
    let (code, lines, stderr) = folo(&[
        "val",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&tiny.data),
        "--split",
        "val",
        "--worst",
        "2",
        "--diag-conf",
        "0.05",
    ]);
    assert_eq!(code, 0, "{stderr}");
    let rec = lines.last().unwrap();
    let diag = &rec["diagnostics"];
    assert!(diag.is_object(), "missing diagnostics field: {rec}");
    assert_eq!(diag["images_diagnosed"].as_u64(), Some(4));
    let sum = &diag["summary"];
    let tp = sum["tp"].as_u64().unwrap();
    let fp = sum["fp"].as_u64().unwrap();
    let fn_count = sum["fn_count"].as_u64().unwrap();
    let num_truths = sum["num_truths"].as_u64().unwrap();
    let num_preds = sum["num_preds"].as_u64().unwrap();
    let loc = sum["localization_errors"].as_u64().unwrap();
    let cls = sum["class_confusions"].as_u64().unwrap();
    let dup = sum["duplicate_errors"].as_u64().unwrap();
    let bg = sum["background_fp"].as_u64().unwrap();
    // Exact conservation laws from `folo_vision::eval::diagnose_image`.
    assert_eq!(tp + fn_count, num_truths, "tp + fn == truths: {sum}");
    assert_eq!(tp + fp, num_preds, "tp + fp == preds: {sum}");
    assert_eq!(loc + cls + dup + bg, fp, "FP taxonomy sums to fp: {sum}");
    let worst = diag["worst_images"].as_array().unwrap();
    assert!(worst.len() <= 2, "worst cap respected: {worst:?}");

    // Upfront validation: `--diag-bg-iou > --diag-tp-iou` fails with exit 2 (`invalid_config`).
    let (code, err_lines, _) = folo(&[
        "val",
        "--json",
        "--model",
        "nonexistent.folo",
        "--data",
        "nonexistent",
        "--diagnostics",
        "--diag-tp-iou",
        "0.3",
        "--diag-bg-iou",
        "0.6",
    ]);
    assert_eq!(code, 2);
    assert_eq!(err_lines.last().unwrap()["error"]["code"], "invalid_config");

    let (code, desc, stderr) = folo(&["describe", "val", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let args = desc[0]["args"].as_array().unwrap();
    assert!(args.iter().any(|a| a["long"] == "--diagnostics"));
    assert!(args.iter().any(|a| a["long"] == "--worst"));
}

/// `folo export` wires independent `--dynamic-batch` / `--dynamic-hw` axes for v1 models AND exports `folo2-*`
/// (FOLO v2-Iso) checkpoints to ONNX via `export_v2_with_axes`.
#[test]
fn export_independent_dynamic_axes_and_folo2_v2_onnx() {
    let tiny = tiny_run();
    let best = tiny.run.join("weights/best.folo");
    let root = TestDir::new("cli-export-s12");

    // 1. Dynamic batch only (static H/W).
    let batch_out = root.join("v1-dyn-batch.onnx");
    let (code, lines, stderr) = folo(&[
        "export",
        "--json",
        "--model",
        s(&best),
        "--format",
        "onnx",
        "--imgsz",
        "64",
        "--dynamic-batch",
        "--output",
        s(&batch_out),
    ]);
    assert_eq!(code, 0, "{stderr}");
    let rec = lines.last().unwrap();
    assert_eq!(rec["dynamic"], true);
    assert_eq!(rec["dynamic_batch"], true);
    assert_eq!(rec["dynamic_hw"], false);
    assert_eq!(rec["inputs"][0]["shape"], serde_json::json!(["batch", 3, 64, 64]));
    assert!(batch_out.is_file());

    // 2. Dynamic H/W only with fixed batch 2.
    let hw_out = root.join("v1-dyn-hw.onnx");
    let (code, lines, stderr) = folo(&[
        "export",
        "--json",
        "--model",
        s(&best),
        "--format",
        "onnx",
        "--batch",
        "2",
        "--dynamic-hw",
        "--output",
        s(&hw_out),
    ]);
    assert_eq!(code, 0, "{stderr}");
    let rec = lines.last().unwrap();
    assert_eq!(rec["dynamic"], true);
    assert_eq!(rec["dynamic_batch"], false);
    assert_eq!(rec["dynamic_hw"], true);
    assert_eq!(rec["batch"], 2);
    assert_eq!(rec["inputs"][0]["shape"], serde_json::json!([2, 3, "height", "width"]));
    assert!(hw_out.is_file());

    // 3. Conflicting `--batch 2 --dynamic-batch` fails upfront with exit 2 (`usage`).
    let (code, err_lines, _) = folo(&[
        "export",
        "--json",
        "--model",
        s(&best),
        "--format",
        "onnx",
        "--batch",
        "2",
        "--dynamic-batch",
    ]);
    assert_eq!(code, 2);
    assert_eq!(err_lines.last().unwrap()["error"]["code"], "usage");

    // 4. FOLO v2-Iso (`folo2-n`) `.folo` file exported to ONNX with `--dynamic-batch`.
    let v2_cfg = FoloV2Config::new("n", 2).unwrap();
    let v2_model = FoloV2Model::new(v2_cfg.clone()).unwrap();
    let v2_path = root.join("folo2-n.folo");
    let mut meta = BTreeMap::new();
    meta.insert("folo.format".into(), "1".into());
    meta.insert("folo.arch".into(), "folo2-n".into());
    meta.insert("folo.config".into(), serde_json::to_string(&v2_cfg).unwrap());
    meta.insert("folo.classes".into(), serde_json::to_string(&["circle", "square"]).unwrap());
    meta.insert("folo.imgsz".into(), "64".into());
    meta.insert("folo.kind".into(), "checkpoint".into());
    meta.insert("folo.dtype".into(), "F32".into());
    meta.insert("folo.fused".into(), "false".into());
    let named: Vec<(String, folo_core::Tensor)> =
        v2_model.named_tensors().into_iter().map(|nt| (nt.name, nt.tensor)).collect();
    let mut file = std::fs::File::create(&v2_path).unwrap();
    write(&mut file, Some(&meta), &named).unwrap();

    let v2_onnx = root.join("folo2-n.onnx");
    let (code, lines, stderr) = folo(&[
        "export",
        "--json",
        "--model",
        s(&v2_path),
        "--format",
        "onnx",
        "--imgsz",
        "64",
        "--dynamic-batch",
        "--output",
        s(&v2_onnx),
    ]);
    assert_eq!(code, 0, "{stderr}");
    let rec = lines.last().unwrap();
    assert_eq!(rec["arch"], "folo2-n");
    assert_eq!(rec["dynamic_batch"], true);
    assert_eq!(rec["dynamic_hw"], false);
    assert_eq!(rec["inputs"][0]["shape"], serde_json::json!(["batch", 3, 64, 64]));
    assert!(v2_onnx.is_file() && std::fs::metadata(&v2_onnx).unwrap().len() > 0);

    // Non-ONNX export of `folo2-n` returns `not_supported` (exit 4).
    let (code, err_lines, _) = folo(&["export", "--json", "--model", s(&v2_path), "--format", "folo-f16"]);
    assert_eq!(code, 4);
    assert_eq!(err_lines.last().unwrap()["error"]["code"], "not_supported");
}
