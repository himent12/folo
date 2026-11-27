//! End-to-end `folo2` (`folo2-n`..`folo2-x`) parity across `folo describe`, `train`, `val`, `predict` (plain,
//! `--tile`, `--tta`, `--track`), `export` (`folo-f16`, `folo`, `weights`, `onnx`), and `quantize` / int8 refusal.

use std::collections::BTreeMap;
use std::path::Path;

use folo_data::ImageFormat;
use folo_data::synthetic::{SyntheticConfig, write_yolo};
use folo_test_support::TestDir;
use serde_json::Value;

use super::{UNSTOPPED_EPOCHS, assert_no_f32_noise, folo, folo_cmd, s, train_until_stopped};

fn header_meta(path: &Path) -> BTreeMap<String, String> {
    let bytes = std::fs::read(path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    header["__metadata__"]
        .as_object()
        .unwrap()
        .iter()
        .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
        .collect()
}

/// `folo describe` surfaces every `folo-*` and `folo2-*` architecture in both `--json` and pretty text modes.
#[test]
fn describe_surfaces_folo2_architectures() {
    let out_j = folo_cmd().args(["describe", "--json", "--no-mcp-tools"]).output().unwrap();
    assert_eq!(out_j.status.code(), Some(0));
    let text_j = String::from_utf8(out_j.stdout).unwrap();
    assert_eq!(text_j.lines().count(), 1);
    let doc_j: Value = serde_json::from_str(&text_j).unwrap();

    let out_h = folo_cmd().args(["describe", "--no-mcp-tools"]).output().unwrap();
    assert_eq!(out_h.status.code(), Some(0));
    let text_h = String::from_utf8(out_h.stdout).unwrap();
    assert!(text_h.lines().count() > 20);
    let doc_h: Value = serde_json::from_str(&text_h).unwrap();
    assert_eq!(doc_h, doc_j);

    let archs = doc_j["architectures"].as_array().expect("architectures array");
    let names: Vec<&str> = archs.iter().map(|a| a["arch"].as_str().unwrap()).collect();
    for expected in [
        "folo-n", "folo-s", "folo-m", "folo-l", "folo-x", "folo2-n", "folo2-s", "folo2-m", "folo2-l",
        "folo2-x",
    ] {
        assert!(names.contains(&expected), "missing {expected} in {names:?}");
    }
    for entry in archs {
        assert!(entry["family"].is_string() && entry["scale"].is_string(), "{entry}");
        assert!(entry["depth"].as_f64().unwrap() > 0.0, "{entry}");
        assert!(entry["width"].as_f64().unwrap() > 0.0, "{entry}");
        assert!(entry["max_channels"].as_u64().unwrap() > 0, "{entry}");
    }

    let (code, train_lines, stderr) = folo(&["describe", "train", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let model_arg = train_lines[0]["args"].as_array().unwrap().iter().find(|a| a["name"] == "model").unwrap();
    assert!(
        model_arg["help"].as_str().unwrap().contains("folo2-n"),
        "train --model help should list folo2-*: {model_arg}"
    );
}

/// Full `folo2-n` lifecycle on CPU (64 px): auto-batch train -> val (+ diagnostics) -> predict (plain, `--tile`,
/// `--tta`, `--track`) -> export (`folo-f16`, `folo`, `weights`, `onnx`) -> fine-tune from exported weights ->
/// stop & resume -> clean `not_supported` on int8 quantize/val.
#[test]
fn folo2_train_val_predict_export_and_resume_end_to_end() {
    let root = TestDir::new("cli-folo2");
    let data = root.join("shapes");
    let cfg = SyntheticConfig {
        images: 8,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        seed: 14,
        vary_aspect: false,
        val_fraction: 0.25,
        ..SyntheticConfig::default()
    };
    write_yolo(&cfg, &data, ImageFormat::Png).unwrap();
    let project = root.join("runs");

    // 1. Train `folo2-n` with `--batch auto` (CPU fallback) for 1 epoch.
    let (code, lines, stderr) = folo(&[
        "train",
        "--json",
        "--model",
        "folo2-n",
        "--data",
        s(&data),
        "--imgsz",
        "64",
        "--batch",
        "auto",
        "--epochs",
        "1",
        "--workers",
        "1",
        "--close-mosaic",
        "1",
        "--project",
        s(&project),
        "--name",
        "v2-auto",
    ]);
    assert_eq!(code, 0, "{stderr}");
    for l in &lines {
        assert_no_f32_noise(l);
    }
    let ab = lines.iter().find(|l| l["event"] == "auto_batch").expect("auto_batch event");
    assert_eq!(ab["fallback"], "cpu", "{ab}");
    let done = lines.last().unwrap();
    assert_eq!(done["event"], "done");
    assert_eq!(done["status"], "completed");
    assert_eq!(done["epochs_completed"], 1);
    assert_eq!(done["auto_batch"]["fallback"], "cpu", "{done}");

    let run = project.join("v2-auto");
    let best = run.join("weights/best.folo");
    assert!(best.is_file(), "best.folo missing");
    let best_meta = header_meta(&best);
    assert_eq!(best_meta.get("folo.arch").map(String::as_str), Some("folo2-n"));

    // 2. `folo val` with `--diagnostics` on the trained `folo2-n` checkpoint.
    let (code, lines, stderr) = folo(&[
        "val",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&data),
        "--workers",
        "1",
        "--diagnostics",
        "--worst",
        "2",
    ]);
    assert_eq!(code, 0, "{stderr}");
    let v = &lines[0];
    assert_eq!(
        (v["imgsz"].as_u64(), v["images"].as_u64(), v["device"].as_str()),
        (Some(64), Some(2), Some("cpu"))
    );
    assert!(v["map50"].as_f64().unwrap() >= 0.0, "{v}");
    assert_eq!(v["diagnostics"]["images_diagnosed"].as_u64(), Some(2), "{v}");

    // 3. `folo predict`: plain, SAHI `--tile`, `--tta`, and `--track`.
    let val_dir = data.join("images/val");
    let first_img = val_dir.read_dir().unwrap().next().unwrap().unwrap().path();

    let (code, lines, stderr) =
        folo(&["predict", "--json", "--model", s(&best), "--conf", "0.01", s(&first_img)]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines[0]["event"], "prediction");
    assert_eq!(lines.last().unwrap()["event"], "done");

    let (code, lines, stderr) = folo(&[
        "predict",
        "--json",
        "--model",
        s(&best),
        "--conf",
        "0.01",
        "--tile",
        "32",
        "--tile-overlap",
        "0.2",
        s(&first_img),
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines[0]["event"], "prediction");

    let (code, lines, stderr) = folo(&[
        "predict",
        "--json",
        "--model",
        s(&best),
        "--conf",
        "0.01",
        "--tta",
        "--tta-scales",
        "1.0",
        s(&first_img),
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines[0]["event"], "prediction");

    let (code, lines, stderr) =
        folo(&["predict", "--json", "--model", s(&best), "--conf", "0.01", "--track", s(&val_dir)]);
    assert_eq!(code, 0, "{stderr}");
    let preds: Vec<&Value> = lines.iter().filter(|l| l["event"] == "prediction").collect();
    assert_eq!(preds.len(), 2);
    assert_eq!(preds[0]["frame_index"], 0);
    assert_eq!(preds[1]["frame_index"], 1);
    assert_eq!(lines.last().unwrap()["tracked"], true);

    // 4. `folo export`: all 4 formats (`folo-f16`, `folo`, `weights`, `onnx`).
    let f16_out = root.join("v2-f16.folo");
    let (code, lines, stderr) =
        folo(&["export", "--json", "--model", s(&best), "--format", "folo-f16", "--output", s(&f16_out)]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(
        (lines[0]["arch"].as_str(), lines[0]["dtype"].as_str(), lines[0]["fused"].as_bool()),
        (Some("folo2-n"), Some("f16"), Some(true)),
        "{}",
        lines[0]
    );
    let f16_meta = header_meta(&f16_out);
    assert_eq!(f16_meta.get("folo.arch").map(String::as_str), Some("folo2-n"));
    assert_eq!(f16_meta.get("folo.kind").map(String::as_str), Some("weights"));
    assert!(!f16_meta.contains_key("folo.train"), "checkpoint metadata stripped");
    let (code, _, stderr) = folo(&["predict", "--json", "--model", s(&f16_out), s(&first_img)]);
    assert_eq!(code, 0, "predict on exported folo2 f16 failed: {stderr}");

    let f32_out = root.join("v2-fused.folo");
    let (code, lines, stderr) =
        folo(&["export", "--json", "--model", s(&best), "--format", "folo", "--output", s(&f32_out)]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(
        (lines[0]["arch"].as_str(), lines[0]["dtype"].as_str(), lines[0]["fused"].as_bool()),
        (Some("folo2-n"), Some("f32"), Some(true)),
        "{}",
        lines[0]
    );
    let (code, _, stderr) = folo(&["predict", "--json", "--model", s(&f32_out), s(&first_img)]);
    assert_eq!(code, 0, "predict on exported folo2 fused f32 failed: {stderr}");

    let weights_out = root.join("v2-weights.folo");
    let (code, lines, stderr) =
        folo(&["export", "--json", "--model", s(&best), "--format", "weights", "--output", s(&weights_out)]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(
        (lines[0]["arch"].as_str(), lines[0]["dtype"].as_str(), lines[0]["fused"].as_bool()),
        (Some("folo2-n"), Some("f32"), Some(false)),
        "{}",
        lines[0]
    );
    let w_meta = header_meta(&weights_out);
    assert_eq!(w_meta.get("folo.arch").map(String::as_str), Some("folo2-n"));
    assert!(w_meta.contains_key("folo.metrics"), "folo.metrics preserved: {w_meta:?}");
    assert!(!w_meta.contains_key("folo.train"), "checkpoint keys stripped: {w_meta:?}");
    // Fine-tuning from exported `folo2-n` weights succeeds and stays `folo2-n`.
    let (code, lines, stderr) = folo(&[
        "train",
        "--json",
        "--weights",
        s(&weights_out),
        "--data",
        s(&data),
        "--imgsz",
        "64",
        "--batch",
        "4",
        "--epochs",
        "1",
        "--workers",
        "1",
        "--close-mosaic",
        "1",
        "--project",
        s(&project),
        "--name",
        "v2-ft",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines.last().unwrap()["status"], "completed");
    assert_eq!(
        header_meta(&project.join("v2-ft/weights/best.folo")).get("folo.arch").map(String::as_str),
        Some("folo2-n")
    );

    let onnx_out = root.join("v2-decoded.onnx");
    let (code, lines, stderr) = folo(&[
        "export",
        "--json",
        "--model",
        s(&best),
        "--format",
        "onnx",
        "--decode",
        "--dynamic-batch",
        "--output",
        s(&onnx_out),
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines[0]["arch"], "folo2-n");
    assert_eq!(lines[0]["decode"], true);
    assert_eq!(lines[0]["dynamic_batch"], true);
    let onnx_bytes = std::fs::read(&onnx_out).unwrap();
    assert!(!onnx_bytes.is_empty());
    let onnx_lossy = String::from_utf8_lossy(&onnx_bytes);
    assert!(
        onnx_lossy.contains("\"0\":\"circle\"") && onnx_lossy.contains("\"1\":\"square\""),
        "exported folo2 ONNX must preserve real class names in `names` metadata"
    );

    // 5. `folo quantize` + `--precision int8` on `folo2-n` calibrates and runs CPU int8 val and predict end to end.
    let v2_int8 = root.join("v2-int8.folo");
    let (code, lines, stderr) = folo(&[
        "quantize",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&data),
        "--images",
        "4",
        "--output",
        s(&v2_int8),
    ]);
    assert_eq!(code, 0, "folo2 quantize failed: {stderr}");
    assert!(lines[0]["boundaries"].as_u64().unwrap() > 10, "{}", lines[0]);
    assert_eq!(lines[0]["f32_policy"], "auto");
    let (code, lines, stderr) =
        folo(&["val", "--json", "--model", s(&v2_int8), "--data", s(&data), "--precision", "int8"]);
    assert_eq!(code, 0, "folo2 int8 val failed: {stderr}");
    assert_eq!((lines[0]["dtype"].as_str(), lines[0]["device"].as_str()), (Some("int8"), Some("cpu")));
    let (code, lines, stderr) =
        folo(&["predict", "--json", "--model", s(&v2_int8), "--precision", "int8", s(&first_img)]);
    assert_eq!(code, 0, "folo2 int8 predict failed: {stderr}");
    assert_eq!(lines[0]["dtype"], "int8");

    // Explicit `--f32-conv` works on `folo2-n` (valid conv accepted, unknown conv rejected upfront with `usage`).
    let v2_int8_explicit = root.join("v2-int8-explicit.folo");
    let (code, lines, stderr) = folo(&[
        "quantize",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&data),
        "--images",
        "2",
        "--f32-policy",
        "none",
        "--f32-conv",
        "neck.out5.cv2",
        "--output",
        s(&v2_int8_explicit),
    ]);
    assert_eq!(code, 0, "folo2 explicit --f32-conv failed: {stderr}");
    assert_eq!(lines[0]["f32_convs"], serde_json::json!(["neck.out5.cv2"]), "{}", lines[0]);

    let v2_int8_bad = root.join("v2-int8-bad.folo");
    let (code, lines, _) = folo(&[
        "quantize",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&data),
        "--images",
        "2",
        "--f32-policy",
        "none",
        "--f32-conv",
        "bogus.layer",
        "--output",
        s(&v2_int8_bad),
    ]);
    assert_eq!((code, lines[0]["error"]["code"].as_str()), (2, Some("usage")), "{}", lines[0]);
    assert!(!v2_int8_bad.exists());

    // 6. Stop + resume on `folo2-n` (refuses mismatched `--model`, resumes cleanly to epoch 2+).
    let long_run = project.join("v2-long");
    let stopped = train_until_stopped(
        &[
            "--device",
            "cpu",
            "--model",
            "folo2-n",
            "--data",
            s(&data),
            "--imgsz",
            "64",
            "--batch",
            "4",
            "--epochs",
            UNSTOPPED_EPOCHS,
            "--workers",
            "1",
            "--close-mosaic",
            "1",
            "--project",
            s(&project),
            "--name",
            "v2-long",
        ],
        &long_run,
    );
    let finished = stopped.last().unwrap()["epochs_completed"].as_u64().unwrap();
    assert!(finished >= 1);
    let last_ckpt = long_run.join("weights/last.folo");
    let (code, lines, _) = folo(&["train", "--json", "--resume", s(&last_ckpt), "--model", "folo-n"]);
    assert_eq!((code, lines.last().unwrap()["error"]["code"].as_str()), (2, Some("invalid_config")));
    let resumed = train_until_stopped(
        &["--resume", s(&last_ckpt), "--model", "folo2-n", "--json-events", "epoch"],
        &long_run,
    );
    assert!(resumed.last().unwrap()["epochs_completed"].as_u64().unwrap() > finished);
}
