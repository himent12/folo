//! Pretrained-weights plumbing (`export --format weights`, `weights list`, name resolution in
//! `train --weights` / `val --model`) and `val --coco` / `--save-json` on a COCO dataset.

use std::path::Path;

use folo_test_support::TestDir;
use serde_json::Value;

use super::{folo_with, s, tiny_run};

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
        "folo-n-coco-v5",
        "folo-n-coco-v5-f16",
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
    let val = ["val", "--json", "--model", s(&exported), "--data", s(&coco), "--workers", "1"];
    // A request the validator refuses answers that error and creates no folder for its results.
    let refused = root.join("refused").join("dets.json");
    let args = [&val[..], &["--imgsz", "33", "--save-json", s(&refused)]].concat();
    let (code, lines, _) = folo_with(&args, Some(&cache));
    assert_ne!(code, 0);
    assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_config", "{lines:?}");
    assert!(!refused.parent().unwrap().exists(), "a refused request leaves no folder");
    // In a folder that does not exist yet: `val` creates it (examples/coco/README.md runs it this way).
    let results = root.join("eval").join("dets.json");
    let args = [&val[..], &["--coco", "--save-json", s(&results)]].concat();
    let (code, lines, stderr) = folo_with(&args, Some(&cache));
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
