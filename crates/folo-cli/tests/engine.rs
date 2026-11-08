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

    // JSON paths use `/` and floats carry no f32 -> f64 noise.
    assert!(!done["run_dir"].as_str().unwrap().contains('\\'), "{done}");
    assert!(!done["weights"]["best"].as_str().unwrap().contains('\\'), "{done}");
    for l in &lines {
        assert_no_f32_noise(l);
    }

    // resume a finished run: refused like MCP's, with the command that trains further.
    let (code, lines, _) = folo(&["train", "--json", "--resume", s(&run.join("weights/last.folo"))]);
    assert_eq!(code, 1);
    let e = &lines.last().unwrap()["error"];
    assert_eq!(e["code"], "run_not_resumable");
    assert!(
        e["hint"].as_str().unwrap().contains("--model") && e["hint"].as_str().unwrap().contains("best.folo")
    );

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
    // Without --output the default name is never overwritten (as MCP `export`); an existing --output is refused.
    let (code, first, stderr) = folo(&["export", "--json", "--model", s(&best)]);
    assert_eq!(code, 0, "{stderr}");
    let (code, second, stderr) = folo(&["export", "--json", "--model", s(&best)]);
    assert_eq!(code, 0, "{stderr}");
    assert!(first[0]["path"].as_str().unwrap().ends_with("weights/best-f16.folo"), "{}", first[0]);
    assert!(second[0]["path"].as_str().unwrap().ends_with("weights/best-f16-2.folo"), "{}", second[0]);
    let (code, lines, _) = folo(&["export", "--json", "--model", s(&best), "--output", s(&out)]);
    assert_eq!((code, lines[0]["error"]["code"].as_str()), (1, Some("output_exists")));
    // Concurrent exports claim distinct files.
    let racers: Vec<_> = (0..4)
        .map(|_| {
            let best = best.clone();
            std::thread::spawn(move || folo(&["export", "--json", "--model", s(&best)]))
        })
        .collect();
    let mut paths: Vec<String> = racers
        .into_iter()
        .map(|h| {
            let (code, lines, stderr) = h.join().unwrap();
            assert_eq!(code, 0, "{stderr}");
            lines[0]["path"].as_str().unwrap().to_string()
        })
        .collect();
    paths.sort();
    paths.dedup();
    assert_eq!(paths.len(), 4, "{paths:?}");
    // Something that is not a weights file says so.
    let (code, lines, _) = folo(&["export", "--json", "--model", s(&data.join("data.yaml"))]);
    assert_eq!(code, 1);
    let e = &lines[0]["error"];
    assert_eq!(e["code"], "invalid_weights");
    assert!(e["message"].as_str().unwrap().contains("is not a .folo weights file"), "{e}");
    // ONNX: `<name>.onnx` next to the input at the model's imgsz, raw head outputs by default.
    let (code, lines, stderr) = folo(&["export", "--json", "--model", s(&best), "--format", "onnx"]);
    assert_eq!(code, 0, "{stderr}");
    let r = &lines[0];
    assert!(r["path"].as_str().unwrap().ends_with("best.onnx"), "{r}");
    assert_eq!(
        (r["format"].as_str(), r["opset"].as_i64(), r["imgsz"].as_i64()),
        (Some("onnx"), Some(17), Some(64))
    );
    assert_eq!(
        r["inputs"][0],
        serde_json::json!({ "name": "images", "dtype": "float", "shape": [1, 3, 64, 64] })
    );
    assert_eq!(r["outputs"][0]["shape"], serde_json::json!([1, 64, 84]));
    assert_eq!(r["outputs"][1]["name"], "cls_raw");
    assert_eq!(std::fs::metadata(r["path"].as_str().unwrap()).unwrap().len(), r["bytes"].as_u64().unwrap());
    let (code, lines, stderr) = folo(&[
        "export",
        "--json",
        "--model",
        s(&best),
        "--format",
        "onnx",
        "--half",
        "--decode",
        "--dynamic",
    ]);
    assert_eq!(code, 0, "{stderr}");
    let r = &lines[0];
    assert!(r["path"].as_str().unwrap().ends_with("best-f16.onnx"), "{r}");
    assert_eq!(
        r["outputs"][0],
        serde_json::json!({
            "name": "detections", "dtype": "float16", "shape": ["batch", "anchors", 6]
        })
    );
    // ONNX-only flags with a .folo format, --batch with --dynamic and a bad --imgsz are usage errors.
    for bad in [
        &["--half"][..],
        &["--format", "onnx", "--dynamic", "--batch", "2"],
        &["--format", "onnx", "--imgsz", "100"],
    ] {
        let mut args = vec!["export", "--json", "--model", s(&best)];
        args.extend(bad);
        let (code, lines, _) = folo(&args);
        assert_eq!(code, 2, "{bad:?}");
        assert_eq!(lines.last().unwrap()["error"]["code"], "usage", "{bad:?}");
    }

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

    // a negative number reaches the range check (not a clap parse error).
    let (code, lines, _) =
        folo(&["train", "--json", "--data", s(&data), "--lr0", "-1", "--project", s(&project)]);
    assert_eq!((code, lines.last().unwrap()["error"]["code"].as_str()), (2, Some("invalid_config")));
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
    // a GPU this machine does not have: device_not_found naming what it has, and no run folder either.
    let (code, lines, _) = folo(&[
        "train",
        "--json",
        "--data",
        s(&data),
        "--imgsz",
        "64",
        "--device",
        "gpu:7",
        "--project",
        s(&project),
        "--name",
        "nogpu",
    ]);
    assert_eq!(code, 3);
    let e = &lines.last().unwrap()["error"];
    assert_eq!(e["code"], "device_not_found");
    assert!(e["hint"].as_str().unwrap().contains("cpu"), "{e}");
    assert!(!project.join("nogpu").exists());
    let _ = std::fs::remove_dir_all(&root);
}

/// Every float that is exactly an f32 prints as the f32 does (no `0.009999999776482582`).
fn assert_no_f32_noise(v: &Value) {
    match v {
        Value::Number(n) if n.is_f64() => {
            let x = n.as_f64().unwrap();
            if (x as f32) as f64 == x {
                assert_eq!(x.to_string(), (x as f32).to_string(), "f32 noise in {v}");
            }
        }
        Value::Array(a) => a.iter().for_each(assert_no_f32_noise),
        Value::Object(o) => o.values().for_each(assert_no_f32_noise),
        _ => {}
    }
}

/// `folo stop` ends a running `folo train` after its current batch (status `stopped`); `--resume` then refuses a
/// changed setting, and `--json-events epoch` leaves the step records out.
#[test]
fn stop_resume_and_epoch_only_records() {
    use std::io::BufRead;
    let root = std::env::temp_dir().join(format!("folo-cli-stop-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let data = root.join("shapes");
    let cfg = SyntheticConfig {
        images: 16,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        vary_aspect: false,
        val_fraction: 0.25,
        ..SyntheticConfig::default()
    };
    write_yolo(&cfg, &data, ImageFormat::Png).unwrap();
    let project = root.join("runs");
    let run = project.join("long");
    let mut child = Command::new(env!("CARGO_BIN_EXE_folo"))
        .args(["train", "--json", "--device", "cpu", "--data", s(&data), "--imgsz", "64", "--batch", "4"])
        .args(["--epochs", "6", "--workers", "1", "--project", s(&project), "--name", "long"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let stdout = std::io::BufReader::new(child.stdout.take().unwrap());
    let mut records = Vec::new();
    let mut asked = false;
    for line in stdout.lines() {
        let v: Value = serde_json::from_str(&line.unwrap()).unwrap();
        if v["event"] == "epoch" && !asked {
            let (code, lines, stderr) = folo(&["stop", "--json", s(&run)]);
            assert_eq!(code, 0, "{stderr}");
            assert_eq!(lines[0]["event"], "stop_requested");
            asked = true;
        }
        records.push(v);
    }
    assert!(child.wait().unwrap().success());
    let done = records.last().unwrap();
    assert_eq!(done["event"], "done");
    assert_eq!(done["status"], "stopped", "{done}");
    let finished = done["epochs_completed"].as_u64().unwrap();
    assert!((1..6).contains(&finished), "{done}");
    assert!(!run.join("stop").exists(), "the stop file is consumed");
    // folo stop on something that is not a run.
    let (code, lines, _) = folo(&["stop", "--json", s(&data)]);
    assert_eq!((code, lines[0]["error"]["code"].as_str()), (3, Some("run_not_found")));

    // A changed setting is refused, naming the flag; a repeated one is fine.
    let last = run.join("weights/last.folo");
    let (code, lines, _) = folo(&["train", "--json", "--resume", s(&last), "--epochs", "9"]);
    assert_eq!(code, 2);
    let e = &lines.last().unwrap()["error"];
    assert_eq!(e["code"], "invalid_config");
    assert!(e["message"].as_str().unwrap().contains("--epochs 9 differs from the run's 6"), "{e}");
    let (code, lines, stderr) = folo(&[
        "train",
        "--json",
        "--resume",
        s(&last),
        "--epochs",
        "6",
        "--batch",
        "4",
        "--json-events",
        "epoch",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert!(lines.iter().all(|l| l["event"] != "step"), "no step records");
    assert_eq!(lines.iter().filter(|l| l["event"] == "epoch").count() as u64, 6 - finished);
    assert_eq!(lines.last().unwrap()["status"], "completed");
    let _ = std::fs::remove_dir_all(&root);
}
