//! `folo train / val / predict / export` end to end with `--json` on a tiny synthetic dataset, and the pretrained
//! weights plumbing (`export --format weights`, `weights list`, names in `--weights` / `--model`, `val --coco`).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use folo_data::ImageFormat;
use folo_data::synthetic::{SyntheticConfig, write_yolo};
use folo_test_support::{ChildGuard, TestDir};
use serde_json::Value;

/// Runs `folo` with `args`; engine commands get `--device cpu` (their `auto` would pick a GPU when there is one).
fn folo(args: &[&str]) -> (i32, Vec<Value>, String) {
    folo_with(args, None)
}

/// [`folo`] with `FOLO_WEIGHTS_DIR` (the `folo weights` cache) set to `cache`.
fn folo_with(args: &[&str], cache: Option<&Path>) -> (i32, Vec<Value>, String) {
    let mut cmd = folo_cmd();
    cmd.args(args);
    if let Some(cache) = cache {
        cmd.env("FOLO_WEIGHTS_DIR", cache);
    }
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

/// A 1-epoch run of `folo train --json` on a 12-image synthetic dataset (4 val images), trained once for the tests
/// that run at the same time; its folder goes when the last of them lets go.
struct TinyRun {
    root: TestDir,
    /// The dataset (YOLO, classes circle and square).
    data: PathBuf,
    /// The run folder.
    run: PathBuf,
    /// `folo train`'s exit code, records and stderr.
    train: (i32, Vec<Value>, String),
}

fn tiny_run() -> Arc<TinyRun> {
    static RUN: Mutex<Weak<TinyRun>> = Mutex::new(Weak::new());
    let mut shared = RUN.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(run) = shared.upgrade() {
        return run;
    }
    let root = TestDir::new("cli-engine");
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
    let train = folo(&[
        "train",
        "--json",
        "--data",
        s(&data),
        "--imgsz",
        "64",
        "--batch",
        "4",
        "--epochs",
        "1",
        "--workers",
        "2",
        "--close-mosaic",
        "1",
        "--project",
        s(&project),
        "--name",
        "tiny",
    ]);
    let run = Arc::new(TinyRun { run: project.join("tiny"), data, root, train });
    *shared = Arc::downgrade(&run);
    run
}

#[test]
fn train_val_predict_export_with_json() {
    let tiny = tiny_run();
    let (root, data, run) = (&tiny.root, &tiny.data, &tiny.run);
    let project = root.join("runs");
    // A GPU this machine does not have, checked at the end: `folo train` looks at the GPUs first, which takes a
    // while, so it runs alongside the rest.
    let nogpu = {
        let args = [s(data), s(&project)].map(String::from);
        std::thread::spawn(move || {
            let [data, project] = &args;
            let args = ["--imgsz", "64", "--device", "gpu:7", "--project", project, "--name", "nogpu"];
            folo(&[&["train", "--json", "--data", data][..], &args].concat())
        })
    };
    // train: JSON lines, one record per event, a final `done` record.
    let (code, lines, stderr) = &tiny.train;
    assert_eq!(*code, 0, "{stderr}");
    assert!(stderr.contains("epoch 1/1"), "epoch summaries go to stderr: {stderr}");
    let events: Vec<&str> = lines.iter().map(|l| l["event"].as_str().unwrap()).collect();
    assert_eq!(events.first(), Some(&"status"));
    assert_eq!(lines[0]["status"], "running");
    assert_eq!(events.iter().filter(|e| **e == "epoch").count(), 1);
    assert!(events.contains(&"step") && events.contains(&"log"));
    let done = lines.last().unwrap();
    assert_eq!(done["event"], "done");
    assert_eq!(done["status"], "completed");
    assert_eq!(done["epochs_completed"], 1);
    for f in
        ["config.json", "metrics.jsonl", "weights/last.folo", "weights/best.folo", "samples/epoch_001.json"]
    {
        assert!(run.join(f).is_file(), "{f} missing");
    }
    let best = run.join("weights/best.folo");

    // JSON paths use `/` and floats carry no f32 -> f64 noise.
    assert!(!done["run_dir"].as_str().unwrap().contains('\\'), "{done}");
    assert!(!done["weights"]["best"].as_str().unwrap().contains('\\'), "{done}");
    for l in lines {
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
        folo(&["val", "--json", "--model", s(&best), "--data", s(data), "--workers", "1"]);
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
    let done = lines.last().unwrap();
    assert_eq!(done["event"], "done");
    // Timings are rounded to 0.1 µs, so they print with at most 4 decimals (no f64 noise).
    let timings = preds.iter().map(|p| &p["timing"]).chain([&done["mean_timing"], &done["first_timing"]]);
    // The first image (pipeline warm-up) is reported apart and left out of the mean.
    assert_eq!(done["first_timing"], preds[0]["timing"], "{done}");
    if preds.len() > 1 {
        let rest: f64 = preds[1..].iter().map(|p| p["timing"]["total_ms"].as_f64().unwrap()).sum();
        let mean = done["mean_timing"]["total_ms"].as_f64().unwrap();
        assert!((mean - rest / (preds.len() - 1) as f64).abs() < 1e-3, "{done}");
    }
    for (k, v) in timings.flat_map(|t| t.as_object().unwrap()) {
        let decimals = v.to_string().split_once('.').map_or(0, |(_, d)| d.len());
        assert!(v.as_f64().unwrap() >= 0.0 && decimals <= 4, "{k}: {v}");
    }
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
    let (code, lines, _) = folo(&["val", "--json", "--model", s(&best), "--data", s(data), "--conf", "2"]);
    assert_eq!(code, 2);
    assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_config");

    // export: a fused f16 weights file that predicts again.
    let out = root.join("tiny-f16.folo");
    let (code, lines, stderr) = folo(&["export", "--json", "--model", s(&best), "--output", s(&out)]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!((lines[0]["dtype"].as_str(), lines[0]["source"].as_str()), (Some("f16"), Some("ema")));
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
    assert!(r["path"].as_str().unwrap().ends_with("best-decoded-f16.onnx"), "{r}");
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
        folo(&["train", "--json", "--data", s(data), "--lr0", "-1", "--project", s(&project)]);
    assert_eq!((code, lines.last().unwrap()["error"]["code"].as_str()), (2, Some("invalid_config")));
    // a bad config is a usage error and leaves no run folder behind.
    let (code, lines, _) = folo(&[
        "train",
        "--json",
        "--data",
        s(data),
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
    let (code, lines, _) = nogpu.join().unwrap();
    assert_eq!(code, 3);
    let e = &lines.last().unwrap()["error"];
    assert_eq!(e["code"], "device_not_found");
    assert!(e["hint"].as_str().unwrap().contains("cpu"), "{e}");
    assert!(!project.join("nogpu").exists());
}

/// `folo quantize`, then `val` / `predict --precision int8` with the file it writes, and the refusals.
#[test]
fn quantize_then_int8_val_and_predict() {
    let tiny = tiny_run();
    assert_eq!(tiny.train.0, 0, "{}", tiny.train.2);
    let (data, best) = (&tiny.data, tiny.run.join("weights/best.folo"));
    let out = tiny.root.join("tiny-int8.folo");
    let (code, lines, stderr) = folo(&[
        "quantize",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(data),
        "--images",
        "6",
        "--output",
        s(&out),
    ]);
    assert_eq!(code, 0, "{stderr}");
    let r = &lines[0];
    assert_eq!(
        (r["images"].as_u64(), r["method"].as_str(), r["tail"].as_f64()),
        (Some(6), Some("percentile"), Some(1e-4))
    );
    assert!(r["boundaries"].as_u64().unwrap() > 50, "{r}");
    // An existing --output is refused before calibrating.
    let (code, lines, _) =
        folo(&["quantize", "--json", "--model", s(&best), "--data", s(data), "--output", s(&out)]);
    assert_eq!((code, lines[0]["error"]["code"].as_str()), (1, Some("output_exists")));
    // So is an --imgsz the model cannot take (and --method mse runs).
    let (code, lines, _) =
        folo(&["quantize", "--json", "--model", s(&best), "--data", s(data), "--imgsz", "100"]);
    assert_eq!((code, lines[0]["error"]["code"].as_str()), (2, Some("usage")));
    let mse = out.with_file_name("mse-int8.folo");
    let (code, lines, stderr) = folo(&[
        "quantize",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(data),
        "--images",
        "2",
        "--method",
        "mse",
        "--output",
        s(&mse),
    ]);
    assert_eq!(
        (code, lines[0]["method"].as_str(), lines[0]["tail"].as_f64()),
        (0, Some("mse"), None),
        "{stderr}"
    );

    let (code, lines, stderr) =
        folo(&["val", "--json", "--model", s(&out), "--data", s(data), "--precision", "int8"]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!((lines[0]["dtype"].as_str(), lines[0]["device"].as_str()), (Some("int8"), Some("cpu")));
    let image = data.join("images/val").read_dir().unwrap().next().unwrap().unwrap().path();
    let (code, lines, stderr) =
        folo(&["predict", "--json", "--model", s(&out), "--precision", "int8", s(&image)]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines[0]["dtype"], "int8");
    // The same file runs in f32 without the flag.
    let (code, lines, stderr) =
        folo(&["predict", "--json", "--model", s(&out), "--device", "cpu", s(&image)]);
    assert_eq!((code, lines[0]["dtype"].as_str()), (0, Some("f32")), "{stderr}");

    // Weights without a calibration, a GPU device, and --half with --precision are refused.
    let (code, lines, _) =
        folo(&["predict", "--json", "--model", s(&best), "--precision", "int8", s(&image)]);
    assert_eq!((code, lines[0]["error"]["code"].as_str()), (1, Some("invalid_weights")));
    assert!(lines[0]["error"]["hint"].as_str().unwrap().contains("folo quantize"), "{}", lines[0]);
    let (code, lines, _) = folo(&[
        "predict",
        "--json",
        "--model",
        s(&out),
        "--precision",
        "int8",
        "--device",
        "gpu:0",
        s(&image),
    ]);
    assert_eq!((code, lines[0]["error"]["code"].as_str()), (2, Some("usage")));
    let (code, _, _) = folo(&["predict", "--model", s(&out), "--precision", "f16", "--half", s(&image)]);
    assert_eq!(code, 2);
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

/// Epochs of the runs the stop tests stop: far more than a run can finish before its stop lands, however fast
/// training gets (the tests stop every run after its first epoch).
const UNSTOPPED_EPOCHS: &str = "1000";

/// Runs `folo train` with `args` (and `--json`), sends `folo stop` to `run` when the first `epoch` record arrives,
/// and returns every record once the process has exited successfully. A run still going [`STOP_DEADLINE`] after it
/// started fails the test (the guard kills the process), so a trainer that ignores the stop cannot hang it.
fn train_until_stopped(args: &[&str], run: &Path) -> Vec<Value> {
    use std::io::BufRead;
    use std::time::{Duration, Instant};
    let deadline = Instant::now() + STOP_DEADLINE;
    let mut child: ChildGuard = folo_cmd()
        .args(["train", "--json"])
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap()
        .into();
    // Lines arrive on a channel so the wait for the next one can time out; the reader ends when the process does.
    let (tx, rx) = std::sync::mpsc::channel();
    let stdout = child.stdout.take().unwrap();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout).lines().map_while(std::io::Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut records = Vec::new();
    let mut asked = false;
    loop {
        let line = match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(line) => line,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!(
                "folo train {args:?} still running {STOP_DEADLINE:?} after it started (stop sent: {asked}); killed"
            ),
        };
        let v: Value = serde_json::from_str(&line).unwrap();
        if v["event"] == "epoch" && !asked {
            let (code, lines, stderr) = folo(&["stop", "--json", s(run)]);
            assert_eq!(code, 0, "{stderr}");
            assert_eq!(lines[0]["event"], "stop_requested");
            asked = true;
        }
        records.push(v);
    }
    // Its output is closed; it should be gone at once.
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "folo train {args:?} closed its output but did not exit; killed");
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(status.success(), "{status}");
    records
}

/// How long [`train_until_stopped`] lets a run take, start to exit: far more than a stopped run needs.
const STOP_DEADLINE: std::time::Duration = std::time::Duration::from_secs(120);

/// `folo stop` ends a running `folo train` after its current batch (status `stopped`); `--resume` then refuses a
/// changed setting, and `--json-events epoch` leaves the step records out.
#[test]
fn stop_resume_and_epoch_only_records() {
    let root = TestDir::new("cli-stop");
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
    let records = train_until_stopped(
        &[
            "--device",
            "cpu",
            "--data",
            s(&data),
            "--imgsz",
            "64",
            "--batch",
            "4",
            "--epochs",
            UNSTOPPED_EPOCHS,
        ]
        .into_iter()
        .chain(["--workers", "1", "--project", s(&project), "--name", "long"])
        .collect::<Vec<_>>(),
        &run,
    );
    let done = records.last().unwrap();
    assert_eq!(done["event"], "done");
    assert_eq!(done["status"], "stopped", "{done}");
    let finished = done["epochs_completed"].as_u64().unwrap();
    assert!(finished >= 1, "{done}");
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
    let want = format!("--epochs 9 differs from the run's {UNSTOPPED_EPOCHS}");
    assert!(e["message"].as_str().unwrap().contains(&want), "{e}");
    let lines = train_until_stopped(
        &["--resume", s(&last), "--epochs", UNSTOPPED_EPOCHS, "--batch", "4", "--json-events", "epoch"],
        &run,
    );
    assert!(lines.iter().all(|l| l["event"] != "step"), "no step records");
    let done = lines.last().unwrap();
    assert_eq!(done["status"], "stopped", "{done}");
    // The resume went on from the checkpoint: at least one more epoch.
    assert!(done["epochs_completed"].as_u64().unwrap() > finished, "{done}");
    let _ = std::fs::remove_dir_all(&root);
}

/// `--format` picks the labels of a folder that has both YOLO (`data.yaml`) and COCO (`annotations/`) labels, and a
/// resume reads the folder the same way without repeating it (the checkpoint keeps it).
#[test]
fn format_picks_the_labels_and_resume_keeps_it() {
    let root = TestDir::new("cli-format");
    let data = root.join("both");
    let cfg = SyntheticConfig {
        images: 16,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        vary_aspect: false,
        val_fraction: 0.25,
        ..SyntheticConfig::default()
    };
    write_yolo(&cfg, &data, ImageFormat::Png).unwrap();
    // The same images as COCO, next to the YOLO labels, with one class renamed so the two read differently.
    let conv = root.join("conv");
    let (code, _, stderr) = folo(&["dataset", "convert", s(&data), s(&conv), "--to", "coco", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    std::fs::create_dir_all(data.join("annotations")).unwrap();
    for e in std::fs::read_dir(conv.join("annotations")).unwrap().flatten() {
        let text = std::fs::read_to_string(e.path()).unwrap().replace("\"circle\"", "\"ring\"");
        std::fs::write(data.join("annotations").join(e.file_name()), text).unwrap();
    }
    let project = root.join("runs");
    let run = project.join("coco");
    // Runs `folo train` with `args` and stops it after its first epoch; returns the final record.
    let train_one_epoch = |args: &[&str]| {
        let all: Vec<&str> =
            ["--device", "cpu", "--workers", "1"].into_iter().chain(args.iter().copied()).collect();
        train_until_stopped(&all, &run).pop().unwrap()
    };
    let last = train_one_epoch(&[
        "--data",
        s(&data),
        "--format",
        "coco",
        "--imgsz",
        "64",
        "--batch",
        "4",
        "--epochs",
        UNSTOPPED_EPOCHS,
        "--project",
        s(&project),
        "--name",
        "coco",
    ]);
    assert_eq!(last["status"], "stopped", "{last}");
    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["classes"], serde_json::json!(["ring", "square"]), "read as COCO");
    // A checkpoint from before `data_format` (auto): resumed once with --format coco, which its later checkpoints
    // keep, so the next resume needs no --format either.
    let ckpt = run.join("weights/last.folo");
    set_train_key(&ckpt, "data_format", serde_json::json!("auto"));
    let last = train_one_epoch(&["--resume", s(&ckpt), "--format", "coco"]);
    assert_eq!(last["status"], "stopped", "{last}");
    // Resumed without --format: still COCO (as auto, the folder reads as YOLO, with another class list).
    let last = train_one_epoch(&["--resume", s(&ckpt)]);
    assert_eq!(last["status"], "stopped", "{last}");
    let _ = std::fs::remove_dir_all(&root);
}

/// `--resume` without `--device` keeps the run's device, as it keeps `--format` and `--workers`: it used to resume
/// on `auto` (the GPU, for a run started with `--device cpu`). A stored device this machine lacks (`gpu:7`) makes
/// that visible on any machine; `--device cpu` still overrides it.
#[test]
fn resume_keeps_the_runs_device() {
    let root = TestDir::new("cli-resume-device");
    let data = root.join("shapes");
    let cfg = SyntheticConfig { images: 8, imgsz: 64, val_fraction: 0.25, ..SyntheticConfig::default() };
    write_yolo(&cfg, &data, ImageFormat::Png).unwrap();
    let project = root.join("runs");
    let run = project.join("dev");
    // Stopped after its first epoch (a changed `epochs` would not match the checkpoint's schedule).
    let first = train_until_stopped(
        &["--device", "cpu", "--workers", "1", "--data", s(&data), "--imgsz", "64", "--batch", "4"]
            .into_iter()
            .chain(["--epochs", UNSTOPPED_EPOCHS, "--project", s(&project), "--name", "dev"])
            .collect::<Vec<_>>(),
        &run,
    );
    assert_eq!(first.last().unwrap()["status"], "stopped");
    let ckpt = run.join("weights/last.folo");
    set_train_key(&ckpt, "device", serde_json::json!("gpu:7"));
    // Not through `folo()`, which adds --device cpu.
    let resume = |extra: &[&str]| {
        let out = folo_cmd()
            .args(["train", "--json", "--resume", s(&ckpt), "--json-events", "epoch"])
            .args(extra)
            .output()
            .unwrap();
        let stdout = String::from_utf8(out.stdout).unwrap();
        let last: Value = serde_json::from_str(stdout.lines().last().unwrap_or("null")).unwrap();
        (out.status.code().unwrap(), last)
    };
    let (code, last) = resume(&[]);
    assert_eq!((code, &last["error"]["code"]), (3, &serde_json::json!("device_not_found")), "{last}");
    // `--device cpu` overrides it: the run goes on (stopped again after an epoch).
    let lines =
        train_until_stopped(&["--resume", s(&ckpt), "--device", "cpu", "--json-events", "epoch"], &run);
    let last = lines.last().unwrap();
    assert_eq!(last["status"], "stopped", "{last}");
    assert!(lines.iter().any(|l| l["event"] == "epoch"), "it trained");
    let _ = std::fs::remove_dir_all(&root);
}

/// Sets `key` of the `folo.train` config stored in the checkpoint at `path` (its header only; the tensor data, whose
/// offsets are relative to the end of the header, is kept).
fn set_train_key(path: &Path, key: &str, value: Value) {
    let bytes = std::fs::read(path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let meta = header["__metadata__"].as_object_mut().unwrap();
    let mut train: Value = serde_json::from_str(meta["folo.train"].as_str().unwrap()).unwrap();
    train[key] = value;
    meta.insert("folo.train".into(), Value::String(train.to_string()));
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&bytes[8 + n..]);
    std::fs::write(path, out).unwrap();
}

/// An out-of-range value's hint names the command it came from (`folo describe predict --json`, not always
/// `train`'s).
#[test]
fn invalid_config_hints_name_the_command() {
    let root = TestDir::new("cli-config-hints");
    let data = root.join("shapes");
    let cfg =
        SyntheticConfig { images: 4, imgsz: 32, seed: 2, val_fraction: 0.5, ..SyntheticConfig::default() };
    write_yolo(&cfg, &data, ImageFormat::Png).unwrap();
    let model = root.join("m.folo");
    let classes = cfg.classes.len();
    folo_vision::model::FoloModel::new(folo_vision::model::ModelConfig::new("n", classes).unwrap())
        .unwrap()
        .save(&model)
        .unwrap();
    let runs = root.join("runs");
    let image = data.join("images/train").read_dir().unwrap().next().unwrap().unwrap().path();
    let cases: [(&str, Vec<&str>); 3] = [
        ("predict", vec!["predict", "--json", "--model", s(&model), "--conf", "2", s(&image)]),
        ("val", vec!["val", "--json", "--model", s(&model), "--data", s(&data), "--imgsz", "100"]),
        ("train", vec!["train", "--json", "--data", s(&data), "--imgsz", "100", "--project", s(&runs)]),
    ];
    for (command, args) in cases {
        let (code, lines, stderr) = folo(&args);
        let e = &lines.last().unwrap_or_else(|| panic!("{command}: {stderr}"))["error"];
        assert_eq!((code, e["code"].as_str()), (2, Some("invalid_config")), "{command}: {e}");
        let hint = e["hint"].as_str().unwrap_or_default();
        assert!(hint.contains(&format!("folo describe {command} --json")), "{command}: {hint}");
    }
}

/// A 1-epoch `folo train` of `data` into `project/name`, with the weights cache `cache`.
fn tiny_train(
    cache: &Path,
    data: &Path,
    project: &Path,
    name: &str,
    extra: &[&str],
) -> (i32, Vec<Value>, String) {
    let mut args = vec![
        "train",
        "--json",
        "--data",
        s(data),
        "--imgsz",
        "64",
        "--batch",
        "4",
        "--epochs",
        "1",
        "--workers",
        "2",
        "--close-mosaic",
        "0",
        "--project",
        s(project),
        "--name",
        name,
    ];
    args.extend_from_slice(extra);
    folo_with(&args, Some(cache))
}

/// Pretrained-weights plumbing with `--json`: `export --format weights`, `weights list`, name resolution in
/// `train --weights` / `val --model`, and `val --coco` / `--save-json` on a COCO dataset.
#[test]
fn weights_names_export_and_coco_val() {
    let tiny = tiny_run();
    let data = &tiny.data;
    let root = TestDir::new("cli-pretrained");
    let cache = root.join("cache");
    let project = root.join("runs");

    // `weights list`: every published name, none downloaded into an empty cache.
    let (code, lines, stderr) = folo_with(&["weights", "list", "--json"], Some(&cache));
    assert_eq!(code, 0, "{stderr}");
    let names: Vec<&str> = lines.iter().map(|l| l["name"].as_str().unwrap()).collect();
    for name in [
        "folo-n-coco",
        "folo-n-coco-f16",
        "folo-n-coco-v2",
        "folo-n-coco-v2-f16",
        "folo-s-coco",
        "folo-s-coco-f16",
        "folo-n-coco-v3",
        "folo-n-coco-v3-f16",
        "folo-n-coco-v4",
        "folo-n-coco-v4-f16",
    ] {
        assert!(names.contains(&name), "{name} missing from {names:?}");
    }
    assert!(lines.iter().all(|l| l["cached"] == false));
    // JSON paths use `/` on every OS (the cache dir from the environment has `\` on Windows).
    for l in &lines {
        let asset = l["asset"].as_str().unwrap();
        assert_eq!(l["path"].as_str().unwrap(), folo_api::json::path(cache.join(asset)), "{l}");
        assert!(!l["path"].as_str().unwrap().contains('\\'), "{l}");
    }

    // A published name that is not downloaded: not found, with the pull hint.
    let (code, lines, _) = tiny_train(&cache, data, &project, "nope", &["--weights", "folo-n-coco"]);
    assert_eq!(code, 3);
    let err = &lines.last().unwrap()["error"];
    assert_eq!(err["code"], "weights_not_found");
    assert!(err["hint"].as_str().unwrap().contains("folo weights pull folo-n-coco"));
    let (code, lines, _) = folo_with(&["weights", "pull", "no-such-weights", "--json"], Some(&cache));
    assert_eq!(code, 3);
    assert_eq!(lines.last().unwrap()["error"]["code"], "weights_not_found");
    // --weights decides the architecture: an explicit --model alongside it is a usage error.
    let (code, _, _) =
        tiny_train(&cache, data, &project, "both", &["--model", "folo-s", "--weights", "x.folo"]);
    assert_eq!(code, 2);

    // Export the tiny run's EMA as a plain (unfused, f32) weights file.
    let best = tiny.run.join("weights/best.folo");
    let exported = root.join("base-weights.folo");
    let (code, lines, stderr) = folo_with(
        &["export", "--json", "--model", s(&best), "--format", "weights", "--output", s(&exported)],
        Some(&cache),
    );
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(
        (lines[0]["format"].as_str(), lines[0]["fused"].as_bool(), lines[0]["dtype"].as_str()),
        (Some("weights"), Some(false), Some("f32"))
    );
    assert!(std::fs::metadata(&exported).unwrap().len() < std::fs::metadata(&best).unwrap().len());

    // A fused file cannot be exported back to (trainable) weights.
    let fused = root.join("base-f16.folo");
    let (code, _, stderr) =
        folo_with(&["export", "--json", "--model", s(&best), "--output", s(&fused)], Some(&cache));
    assert_eq!(code, 0, "{stderr}");
    let (code, lines, _) = folo_with(
        &[
            "export",
            "--json",
            "--model",
            s(&fused),
            "--format",
            "weights",
            "--output",
            s(&root.join("x.folo")),
        ],
        Some(&cache),
    );
    assert_eq!(code, 2);
    assert_eq!(lines.last().unwrap()["error"]["code"], "usage");

    // Put it in the cache under a published name: list sees it, train / val resolve the name.
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::copy(&exported, cache.join("folo-n-coco.folo")).unwrap();
    let (_, lines, _) = folo_with(&["weights", "list", "--json"], Some(&cache));
    assert!(lines.iter().any(|l| l["name"] == "folo-n-coco" && l["cached"] == true));
    // `pull` of a cached name downloads nothing and reports the file like a download would.
    let (code, lines, stderr) = folo_with(&["weights", "pull", "folo-n-coco", "--json"], Some(&cache));
    assert_eq!(code, 0, "{stderr}");
    let done = lines.last().unwrap();
    assert_eq!((done["event"].as_str(), done["downloaded"].as_bool()), (Some("done"), Some(false)));
    assert_eq!(done["path"].as_str().unwrap(), folo_api::json::path(cache.join("folo-n-coco.folo")));
    assert_eq!(done["bytes"].as_u64(), Some(std::fs::metadata(&exported).unwrap().len()));
    assert_eq!(done["sha256"].as_str().map(str::len), Some(64));
    let (code, lines, stderr) = tiny_train(&cache, data, &project, "tuned", &["--weights", "folo-n-coco"]);
    assert_eq!(code, 0, "{stderr}");
    let transferred = lines.iter().any(|l| {
        l["event"] == "log"
            && l["message"].as_str().is_some_and(|m| m.starts_with("transferred 417 of 417 tensors"))
    });
    assert!(transferred, "every tensor loads from same-class weights: {lines:?}");
    let (code, lines, stderr) = folo_with(
        &["val", "--json", "--model", "folo-n-coco", "--data", s(data), "--workers", "1"],
        Some(&cache),
    );
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines[0]["coco"], Value::Null, "no COCO block without --coco");

    // `val --coco` needs a COCO dataset.
    let (code, lines, _) =
        folo_with(&["val", "--json", "--model", s(&exported), "--data", s(data), "--coco"], Some(&cache));
    assert_eq!(code, 2);
    assert_eq!(lines.last().unwrap()["error"]["code"], "usage");
    // On the same images as COCO: the exact COCOeval block and a results file pycocotools can load.
    let coco = root.join("shapes-coco");
    let (code, _, stderr) =
        folo_with(&["dataset", "convert", s(data), s(&coco), "--to", "coco", "--json"], Some(&cache));
    assert_eq!(code, 0, "{stderr}");
    let results = root.join("dets.json");
    let (code, lines, stderr) = folo_with(
        &[
            "val",
            "--json",
            "--model",
            s(&exported),
            "--data",
            s(&coco),
            "--workers",
            "1",
            "--coco",
            "--save-json",
            s(&results),
        ],
        Some(&cache),
    );
    assert_eq!(code, 0, "{stderr}");
    let c = &lines[0]["coco"];
    assert_eq!(
        (c["images"].as_u64(), c["missing_images"].as_u64(), c["crowd"].as_u64()),
        (Some(4), Some(0), Some(0))
    );
    assert_eq!(c["truths"], lines[0]["boxes"]);
    for k in ["ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large", "ar1", "ar10", "ar100"] {
        let v = c[k].as_f64().unwrap();
        assert!(v == -1.0 || (0.0..=1.0).contains(&v), "{k} = {v}");
    }
    let dets: Value = serde_json::from_str(&std::fs::read_to_string(&results).unwrap()).unwrap();
    let dets = dets.as_array().unwrap();
    assert_eq!(dets.len() as u64, c["detections"].as_u64().unwrap().max(dets.len() as u64));
    for d in dets {
        assert!(d["image_id"].is_u64() && d["category_id"].is_u64() && d["score"].is_f64());
        let b = d["bbox"].as_array().unwrap();
        assert!(b.len() == 4 && b[2].as_f64().unwrap() >= 0.0 && b[3].as_f64().unwrap() >= 0.0);
    }
}

/// A `folo` command with the on-disk caches off (`FOLO_NO_CACHE`), so tests never write into the user's home.
fn folo_cmd() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_folo"));
    c.env("FOLO_NO_CACHE", "1");
    c
}
