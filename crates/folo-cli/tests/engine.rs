//! `folo train / val / predict / export` end to end with `--json` on a tiny synthetic dataset.

use std::path::{Path, PathBuf};
use std::process::Command;

use folo_data::ImageFormat;
use folo_data::synthetic::{SyntheticConfig, write_yolo};
use serde_json::Value;

/// Runs `folo` with `args`; engine commands get `--device cpu` (their `auto` would pick a GPU when there is one).
fn folo(args: &[&str]) -> (i32, Vec<Value>, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_folo"));
    cmd.args(args);
    if matches!(args.first(), Some(&"train" | &"val" | &"predict")) && !args.contains(&"--device") {
        cmd.args(["--device", "cpu"]);
    }
    let out = cmd.output().unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let lines = stdout
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("not JSON: {l}: {e}")))
        .collect();
    (out.status.code().unwrap(), lines, String::from_utf8(out.stderr).unwrap())
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn train_val_predict_export_with_json() {
    let root = std::env::temp_dir().join(format!("folo-cli-engine-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let data = root.join("shapes");
    let cfg = SyntheticConfig {
        images: 12,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        seed: 5,
        vary_aspect: false,
        val_fraction: 4.0 / 12.0,
        ..SyntheticConfig::default()
    };
    write_yolo(&cfg, &data, ImageFormat::Png).unwrap();
    let project = root.join("runs");

    // train: JSON lines, one record per event, a final `done` record.
    let (code, lines, stderr) = folo(&[
        "train",
        "--json",
        "--data",
        s(&data),
        "--imgsz",
        "64",
        "--batch",
        "4",
        "--epochs",
        "2",
        "--workers",
        "2",
        "--close-mosaic",
        "1",
        "--project",
        s(&project),
        "--name",
        "tiny",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stderr.contains("epoch 2/2"), "epoch summaries go to stderr: {stderr}");
    let events: Vec<&str> = lines.iter().map(|l| l["event"].as_str().unwrap()).collect();
    assert_eq!(events.first(), Some(&"status"));
    assert_eq!(lines[0]["status"], "running");
    assert_eq!(events.iter().filter(|e| **e == "epoch").count(), 2);
    assert!(events.contains(&"step") && events.contains(&"log"));
    let done = lines.last().unwrap();
    assert_eq!(done["event"], "done");
    assert_eq!(done["status"], "completed");
    assert_eq!(done["epochs_completed"], 2);
    let run = project.join("tiny");
    for f in
        ["config.json", "metrics.jsonl", "weights/last.folo", "weights/best.folo", "samples/epoch_002.json"]
    {
        assert!(run.join(f).is_file(), "{f} missing");
    }
    let best = run.join("weights/best.folo");

    // resume a finished run: nothing left to do, but the checkpoint and the dataset are found again.
    let (code, lines, stderr) = folo(&["train", "--json", "--resume", s(&run.join("weights/last.folo"))]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines.last().unwrap()["epochs_completed"], 2);

    // val: one object with the metrics.
    let (code, lines, stderr) =
        folo(&["val", "--json", "--model", s(&best), "--data", s(&data), "--workers", "1"]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines.len(), 1);
    let v = &lines[0];
    assert_eq!(
        (v["imgsz"].as_u64(), v["images"].as_u64(), v["split"].as_str()),
        (Some(64), Some(4), Some("val"))
    );
    assert!(v["map50"].as_f64().unwrap() >= 0.0 && v["per_class"].is_array());
    assert_eq!(v["device"], "cpu");

    // predict: one record per image plus `done`; --save writes annotated copies.
    let saved = root.join("pred");
    let images = data.join("images/val");
    let (code, lines, stderr) =
        folo(&["predict", "--json", "--model", s(&best), s(&images), "--conf", "0.01", "--save", s(&saved)]);
    assert_eq!(code, 0, "{stderr}");
    let preds: Vec<&Value> = lines.iter().filter(|l| l["event"] == "prediction").collect();
    assert_eq!(preds.len(), 4);
    for p in &preds {
        assert_eq!((p["image"]["width"].as_u64(), p["image"]["height"].as_u64()), (Some(64), Some(64)));
        assert!(p["timing"]["total_ms"].as_f64().unwrap() > 0.0);
        for b in p["boxes"].as_array().unwrap() {
            assert!(["circle", "square"].contains(&b["name"].as_str().unwrap()));
            assert!(b["score"].as_f64().unwrap() >= 0.01);
        }
        assert!(PathBuf::from(p["saved"].as_str().unwrap()).is_file());
    }
    assert_eq!(lines.last().unwrap()["event"], "done");
    // --save never overwrites a source, and same-named sources get distinct outputs.
    let (code, lines, _) =
        folo(&["predict", "--json", "--model", s(&best), s(&images), "--save", s(&images)]);
    assert_eq!(code, 2);
    assert_eq!(lines.last().unwrap()["error"]["code"], "usage");
    let other = root.join("other");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::copy(images.join("000008.png"), other.join("000008.png")).unwrap();
    let both = root.join("pred2");
    let (code, _, stderr) = folo(&[
        "predict",
        "--json",
        "--model",
        s(&best),
        s(&images.join("000008.png")),
        s(&other),
        "--save",
        s(&both),
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert!(both.join("000008.png").is_file() && both.join("000008_2.png").is_file());
    let (code, lines, _) = folo(&["val", "--json", "--model", s(&best), "--data", s(&data), "--conf", "2"]);
    assert_eq!(code, 2);
    assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_config");

    // export: a fused f16 weights file that predicts again.
    let out = root.join("tiny-f16.folo");
    let (code, lines, stderr) = folo(&["export", "--json", "--model", s(&best), "--output", s(&out)]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!((lines[0]["dtype"].as_str(), lines[0]["source"].as_str()), (Some("F16"), Some("ema")));
    assert!(lines[0]["bytes"].as_u64().unwrap() < std::fs::metadata(&best).unwrap().len() / 4);
    let (code, _, stderr) = folo(&["predict", "--json", "--model", s(&out), s(&images.join("000008.png"))]);
    assert_eq!(code, 0, "{stderr}");
    let (code, lines, _) = folo(&["export", "--json", "--model", s(&best), "--format", "onnx"]);
    assert_eq!(code, 4);
    assert_eq!(lines[0]["error"]["code"], "not_implemented");

    // export never touches its input, even when the input is named like the writer's temporary file.
    let decoy = root.join("decoy.folo.tmp");
    std::fs::copy(&best, &decoy).unwrap();
    let before = std::fs::read(&decoy).unwrap();
    let (code, _, stderr) =
        folo(&["export", "--json", "--model", s(&decoy), "--output", s(&root.join("decoy.folo"))]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(std::fs::read(&decoy).unwrap(), before, "the input stays byte-identical");
    // export never replaces its input, even through another spelling of the same path.
    let rel = run.join("weights/../weights/best.folo");
    let (code, lines, _) = folo(&["export", "--json", "--model", s(&best), "--output", s(&rel)]);
    assert_eq!(code, 2);
    assert_eq!(lines.last().unwrap()["error"]["code"], "usage");
    // A --save destination hard-linked to a source is replaced, never written through.
    let linked = root.join("linked");
    std::fs::create_dir_all(&linked).unwrap();
    let src = root.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::copy(images.join("000009.png"), src.join("h.png")).unwrap();
    std::fs::hard_link(src.join("h.png"), linked.join("h.png")).unwrap();
    let before = std::fs::read(src.join("h.png")).unwrap();
    let (code, _, stderr) = folo(&["predict", "--json", "--model", s(&best), s(&src), "--save", s(&linked)]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(std::fs::read(src.join("h.png")).unwrap(), before, "the source stays byte-identical");
    // Same-named sources that differ only in case get distinct outputs too.
    let upper = root.join("upper");
    std::fs::create_dir_all(&upper).unwrap();
    std::fs::copy(images.join("000009.png"), upper.join("A.png")).unwrap();
    let lower = root.join("lower");
    std::fs::create_dir_all(&lower).unwrap();
    std::fs::copy(images.join("000009.png"), lower.join("a.png")).unwrap();
    let cased = root.join("pred3");
    let (code, lines, stderr) =
        folo(&["predict", "--json", "--model", s(&best), s(&upper), s(&lower), "--save", s(&cased)]);
    assert_eq!(code, 0, "{stderr}");
    let saved: Vec<String> =
        lines.iter().filter_map(|l| l["saved"].as_str().map(str::to_lowercase)).collect();
    assert_eq!(saved.len(), 2);
    assert_ne!(saved[0], saved[1]);

    // a bad config is a usage error and leaves no run folder behind.
    let (code, lines, _) = folo(&[
        "train",
        "--json",
        "--data",
        s(&data),
        "--imgsz",
        "100",
        "--project",
        s(&project),
        "--name",
        "bad",
    ]);
    assert_eq!(code, 2);
    assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_config");
    assert!(!project.join("bad").exists());
    let _ = std::fs::remove_dir_all(&root);
}
