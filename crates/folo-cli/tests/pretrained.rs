//! Pretrained-weights plumbing end to end with `--json`: `export --format weights`, `weights list`, name
//! resolution in `train --weights` / `val --model`, and `val --coco` / `--save-json` on a COCO dataset.

use std::path::Path;
use std::process::Command;

use folo_data::ImageFormat;
use folo_data::synthetic::{SyntheticConfig, write_yolo};
use folo_test_support::TestDir;
use serde_json::Value;

/// Runs `folo` with `args` and `FOLO_WEIGHTS_DIR = cache`; engine commands get `--device cpu`.
fn folo(cache: &Path, args: &[&str]) -> (i32, Vec<Value>, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_folo"));
    cmd.args(args).env("FOLO_WEIGHTS_DIR", cache);
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
    folo(cache, &args)
}

#[test]
fn weights_names_export_and_coco_val() {
    let root = TestDir::new("cli-pretrained");
    let cache = root.join("cache");
    let data = root.join("shapes");
    let cfg = SyntheticConfig {
        images: 12,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        seed: 9,
        vary_aspect: false,
        val_fraction: 4.0 / 12.0,
        ..SyntheticConfig::default()
    };
    write_yolo(&cfg, &data, ImageFormat::Png).unwrap();
    let project = root.join("runs");

    // `weights list`: every published name, none downloaded into an empty cache.
    let (code, lines, stderr) = folo(&cache, &["weights", "list", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let names: Vec<&str> = lines.iter().map(|l| l["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"folo-n-coco") && names.contains(&"folo-n-coco-f16"), "{names:?}");
    assert!(lines.iter().all(|l| l["cached"] == false));
    // JSON paths use `/` on every OS (the cache dir from the environment has `\` on Windows).
    for l in &lines {
        let asset = l["asset"].as_str().unwrap();
        assert_eq!(l["path"].as_str().unwrap(), folo_api::json::path(cache.join(asset)), "{l}");
        assert!(!l["path"].as_str().unwrap().contains('\\'), "{l}");
    }

    // A published name that is not downloaded: not found, with the pull hint.
    let (code, lines, _) = tiny_train(&cache, &data, &project, "nope", &["--weights", "folo-n-coco"]);
    assert_eq!(code, 3);
    let err = &lines.last().unwrap()["error"];
    assert_eq!(err["code"], "weights_not_found");
    assert!(err["hint"].as_str().unwrap().contains("folo weights pull folo-n-coco"));
    let (code, lines, _) = folo(&cache, &["weights", "pull", "no-such-weights", "--json"]);
    assert_eq!(code, 3);
    assert_eq!(lines.last().unwrap()["error"]["code"], "weights_not_found");
    // --weights decides the architecture: an explicit --model alongside it is a usage error.
    let (code, _, _) =
        tiny_train(&cache, &data, &project, "both", &["--model", "folo-s", "--weights", "x.folo"]);
    assert_eq!(code, 2);

    // Train a tiny model and export its EMA as a plain (unfused, f32) weights file.
    let (code, _, stderr) = tiny_train(&cache, &data, &project, "base", &[]);
    assert_eq!(code, 0, "{stderr}");
    let best = project.join("base/weights/best.folo");
    let exported = root.join("base-weights.folo");
    let (code, lines, stderr) = folo(
        &cache,
        &["export", "--json", "--model", s(&best), "--format", "weights", "--output", s(&exported)],
    );
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(
        (lines[0]["format"].as_str(), lines[0]["fused"].as_bool(), lines[0]["dtype"].as_str()),
        (Some("weights"), Some(false), Some("f32"))
    );
    assert!(std::fs::metadata(&exported).unwrap().len() < std::fs::metadata(&best).unwrap().len());

    // A fused file cannot be exported back to (trainable) weights.
    let fused = root.join("base-f16.folo");
    let (code, _, stderr) = folo(&cache, &["export", "--json", "--model", s(&best), "--output", s(&fused)]);
    assert_eq!(code, 0, "{stderr}");
    let (code, lines, _) = folo(
        &cache,
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
    );
    assert_eq!(code, 2);
    assert_eq!(lines.last().unwrap()["error"]["code"], "usage");

    // Put it in the cache under a published name: list sees it, train / val resolve the name.
    std::fs::create_dir_all(&cache).unwrap();
    std::fs::copy(&exported, cache.join("folo-n-coco.folo")).unwrap();
    let (_, lines, _) = folo(&cache, &["weights", "list", "--json"]);
    assert!(lines.iter().any(|l| l["name"] == "folo-n-coco" && l["cached"] == true));
    // `pull` of a cached name downloads nothing and reports the file like a download would.
    let (code, lines, stderr) = folo(&cache, &["weights", "pull", "folo-n-coco", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let done = lines.last().unwrap();
    assert_eq!((done["event"].as_str(), done["downloaded"].as_bool()), (Some("done"), Some(false)));
    assert_eq!(done["path"].as_str().unwrap(), folo_api::json::path(cache.join("folo-n-coco.folo")));
    assert_eq!(done["bytes"].as_u64(), Some(std::fs::metadata(&exported).unwrap().len()));
    assert_eq!(done["sha256"].as_str().map(str::len), Some(64));
    let (code, lines, stderr) = tiny_train(&cache, &data, &project, "tuned", &["--weights", "folo-n-coco"]);
    assert_eq!(code, 0, "{stderr}");
    let transferred = lines.iter().any(|l| {
        l["event"] == "log"
            && l["message"].as_str().is_some_and(|m| m.starts_with("transferred 417 of 417 tensors"))
    });
    assert!(transferred, "every tensor loads from same-class weights: {lines:?}");
    let (code, lines, stderr) =
        folo(&cache, &["val", "--json", "--model", "folo-n-coco", "--data", s(&data), "--workers", "1"]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(lines[0]["coco"], Value::Null, "no COCO block without --coco");

    // `val --coco` needs a COCO dataset.
    let (code, lines, _) =
        folo(&cache, &["val", "--json", "--model", s(&exported), "--data", s(&data), "--coco"]);
    assert_eq!(code, 2);
    assert_eq!(lines.last().unwrap()["error"]["code"], "usage");
    // On the same images as COCO: the exact COCOeval block and a results file pycocotools can load.
    let coco = root.join("shapes-coco");
    let (code, _, stderr) =
        folo(&cache, &["dataset", "convert", s(&data), s(&coco), "--to", "coco", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let results = root.join("dets.json");
    let (code, lines, stderr) = folo(
        &cache,
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
    let _ = std::fs::remove_dir_all(&root);
}
