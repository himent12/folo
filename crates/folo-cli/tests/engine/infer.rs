//! Regression tests for `folo predict`, `folo export` and `folo val --coco`.

use std::path::{Path, PathBuf};

use folo_test_support::TestDir;

use super::{folo, s, tiny_run};

/// A directory link (junction on Windows, symlink elsewhere) removed on drop so `TestDir` cleans up even on failure.
struct DirLink(PathBuf);

impl DirLink {
    fn new(target: &Path, link: PathBuf) -> Self {
        #[cfg(windows)]
        {
            let native = |p: &Path| p.to_string_lossy().replace('/', "\\");
            let ok = std::process::Command::new("cmd")
                .args(["/C", "mklink", "/J"])
                .arg(native(&link))
                .arg(native(target))
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|s| s.success());
            assert!(ok, "mklink /J {} -> {}", link.display(), target.display());
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(target, &link).unwrap();
        }
        DirLink(link)
    }
}

impl Drop for DirLink {
    fn drop(&mut self) {
        if std::fs::remove_dir(&self.0).is_err() {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

/// `folo predict` skips a directory link that points to one of its own ancestors (so cycles do not loop or emit
/// 64 duplicates), while still walking non-cyclic repeated links to the same folder (e.g. `latest -> run-3`
/// beside `run-3/`) and non-cyclic external links, as on `main`.
#[test]
fn predict_skips_directory_link_cycles() {
    let tiny = tiny_run();
    let best = tiny.run.join("weights/best.folo");
    let src_img = tiny.data.join("images/val").read_dir().unwrap().next().unwrap().unwrap().path();

    let root = TestDir::new("cli-predict-cycle");
    let input = root.join("input");
    let run3 = input.join("run-3");
    let external = root.join("external");
    std::fs::create_dir_all(&run3).unwrap();
    std::fs::create_dir_all(&external).unwrap();
    std::fs::copy(&src_img, input.join("a.png")).unwrap();
    std::fs::copy(&src_img, run3.join("b.png")).unwrap();
    std::fs::copy(&src_img, external.join("c.png")).unwrap();

    // Real cycle (`loop -> input`), sibling repeated link (`latest -> run-3` beside `run-3/`), and external link.
    let _self_loop = DirLink::new(&input, input.join("loop"));
    let _sibling_link = DirLink::new(&run3, input.join("latest"));
    let _ext_link = DirLink::new(&external, input.join("extra"));

    let (code, lines, stderr) = folo(&["predict", "--json", "--model", s(&best), s(&input)]);
    assert_eq!(code, 0, "{stderr}");
    let preds: Vec<&str> =
        lines.iter().filter(|l| l["event"] == "prediction").map(|l| l["file"].as_str().unwrap()).collect();
    assert_eq!(
        preds.len(),
        4,
        "expected 4 images (a.png, extra/c.png, latest/b.png, run-3/b.png), got {preds:?}"
    );
    for suffix in ["/a.png", "/extra/c.png", "/latest/b.png", "/run-3/b.png"] {
        assert!(preds.iter().any(|p| p.ends_with(suffix)), "missing {suffix} in {preds:?}");
    }
    let done = lines.last().unwrap();
    assert_eq!((done["event"].as_str(), done["images"].as_u64()), (Some("done"), Some(4)), "{done}");
}

/// Inserts string key-value pairs into a `.folo` file's `__metadata__` header without touching tensor bytes.
fn insert_meta_keys(path: &Path, pairs: &[(&str, &str)]) {
    let bytes = std::fs::read(path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let meta = header["__metadata__"].as_object_mut().unwrap();
    for &(k, v) in pairs {
        meta.insert(k.into(), serde_json::Value::String(v.into()));
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&bytes[8 + n..]);
    std::fs::write(path, out).unwrap();
}

/// `folo export --format weights` preserves `folo.metrics`, `folo.created` and unknown `folo.*` keys
/// (docs/FORMAT.md), matching `--format folo`, `--format folo-f16` and `--format onnx`.
#[test]
fn export_weights_preserves_metadata() {
    let tiny = tiny_run();
    let root = TestDir::new("cli-export-weights-meta");
    let src = root.join("best.folo");
    std::fs::copy(tiny.run.join("weights/best.folo"), &src).unwrap();
    insert_meta_keys(&src, &[("folo.custom", "keep-me")]);
    let src_meta = folo_nn::io::ModelFile::open(&src).unwrap().metadata().clone();
    assert!(src_meta.metrics.is_some(), "fixture checkpoint has folo.metrics");
    assert!(src_meta.created.is_some(), "fixture checkpoint has folo.created");

    let out = root.join("weights.folo");
    let (code, _, stderr) =
        folo(&["export", "--json", "--model", s(&src), "--format", "weights", "--output", s(&out)]);
    assert_eq!(code, 0, "{stderr}");

    let file = folo_nn::io::ModelFile::open(&out).unwrap();
    let meta = file.metadata();
    assert_eq!(meta.kind, folo_nn::io::FileKind::Weights);
    assert!(!meta.fused);
    assert_eq!(meta.metrics, src_meta.metrics, "folo.metrics must be preserved");
    assert_eq!(meta.created, src_meta.created, "folo.created must be preserved");
    assert_eq!(meta.extra.get("folo.custom").map(String::as_str), Some("keep-me"));
    for k in ["folo.train", "folo.optim", "folo.sched", "folo.ema", "folo.rng", "folo.epoch", "folo.best"] {
        assert!(!meta.extra.contains_key(k), "checkpoint key `{k}` must be dropped");
    }
}

/// Exporting or quantizing a training checkpoint drops `folo.amp` and `folo.nonfinite_streak` across every
/// export format (`folo`, `folo-f16`, `weights`, `onnx`) and `folo quantize` while preserving unknown `folo.*` keys.
#[test]
fn export_and_quantize_strip_checkpoint_only_keys() {
    let tiny = tiny_run();
    let root = TestDir::new("cli-strip-ckpt-keys");
    let src = root.join("best.folo");
    std::fs::copy(tiny.run.join("weights/best.folo"), &src).unwrap();
    insert_meta_keys(
        &src,
        &[
            (
                "folo.amp",
                r#"{"scale":16384.0,"growth_factor":2.0,"backoff_factor":0.5,"growth_interval":2000,"growth_tracker":0,"skipped_steps":0,"skip_streak":0}"#,
            ),
            ("folo.nonfinite_streak", "2"),
            ("folo.custom", "keep-me"),
        ],
    );

    for (fmt, name) in [("folo", "a-fused.folo"), ("folo-f16", "a-f16.folo"), ("weights", "a-weights.folo")] {
        let out = root.join(name);
        let (code, _, stderr) =
            folo(&["export", "--json", "--model", s(&src), "--format", fmt, "--output", s(&out)]);
        assert_eq!(code, 0, "{fmt}: {stderr}");
        let meta = folo_nn::io::ModelFile::open(&out).unwrap().metadata().clone();
        assert!(!meta.extra.contains_key("folo.amp"), "{fmt} leaked folo.amp");
        assert!(!meta.extra.contains_key("folo.nonfinite_streak"), "{fmt} leaked folo.nonfinite_streak");
        assert_eq!(meta.extra.get("folo.custom").map(String::as_str), Some("keep-me"), "{fmt}");
    }

    let onnx = root.join("a.onnx");
    let (code, _, stderr) =
        folo(&["export", "--json", "--model", s(&src), "--format", "onnx", "--output", s(&onnx)]);
    assert_eq!(code, 0, "onnx: {stderr}");
    let onnx_bytes = std::fs::read(&onnx).unwrap();
    let has = |needle: &[u8]| onnx_bytes.windows(needle.len()).any(|w| w == needle);
    assert!(!has(b"folo.amp"), "onnx leaked folo.amp");
    assert!(!has(b"folo.nonfinite_streak"), "onnx leaked folo.nonfinite_streak");
    assert!(has(b"folo.custom"), "onnx dropped unknown folo.custom");

    let quant = root.join("a-int8.folo");
    let (code, _, stderr) = folo(&[
        "quantize",
        "--json",
        "--model",
        s(&src),
        "--data",
        s(&tiny.data),
        "--images",
        "2",
        "--batch",
        "2",
        "--workers",
        "1",
        "--output",
        s(&quant),
    ]);
    assert_eq!(code, 0, "quantize: {stderr}");
    let qmeta = folo_nn::io::ModelFile::open(&quant).unwrap().metadata().clone();
    assert!(!qmeta.extra.contains_key("folo.amp"), "quantize leaked folo.amp");
    assert!(!qmeta.extra.contains_key("folo.nonfinite_streak"), "quantize leaked folo.nonfinite_streak");
    assert_eq!(qmeta.extra.get("folo.custom").map(String::as_str), Some("keep-me"), "quantize");
}

/// `folo val --coco` / `--save-json` accepts a multi-split COCO dataset whose `val` JSON declares a non-empty
/// `(id, name)` subset of `train`'s categories with every val annotation declared (keeping `coco.detections` and
/// `--save-json` on one filter), and refuses empty categories, undeclared val annotations, and mismatched classes
/// before creating `--save-json`'s output folder.
#[test]
fn val_coco_handles_subset_val_categories_and_fails_early_on_mismatch() {
    let tiny = tiny_run();
    let best = tiny.run.join("weights/best.folo");
    let root = TestDir::new("cli-val-coco-subset");

    let coco = root.join("coco-sub");
    let (code, _, stderr) = folo(&["dataset", "convert", s(&tiny.data), s(&coco), "--to", "coco", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    // Remap ids to non-contiguous 7=circle, 19=square in train and val so (id, name) matching is exercised.
    let remap_ids = |path: &Path| {
        let mut doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let old0 = doc["categories"][0]["id"].as_u64().unwrap();
        let old1 = doc["categories"][1]["id"].as_u64().unwrap();
        doc["categories"][0]["id"] = serde_json::json!(7);
        doc["categories"][1]["id"] = serde_json::json!(19);
        for a in doc["annotations"].as_array_mut().unwrap() {
            let cid = a["category_id"].as_u64().unwrap();
            a["category_id"] = serde_json::json!(if cid == old0 {
                7
            } else if cid == old1 {
                19
            } else {
                cid
            });
        }
        std::fs::write(path, serde_json::to_string(&doc).unwrap()).unwrap();
        doc
    };
    let _train_doc = remap_ids(&coco.join("annotations/instances_train.json"));
    let val_json = coco.join("annotations/instances_val.json");
    let full_val = remap_ids(&val_json);

    // (c) Valid subset: val declares only 19=square and keeps only square annotations.
    let mut sub_val = full_val.clone();
    sub_val["categories"] = serde_json::json!([full_val["categories"][1].clone()]);
    sub_val["annotations"] = serde_json::Value::Array(
        full_val["annotations"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|a| a["category_id"].as_u64() == Some(19))
            .cloned()
            .collect(),
    );
    std::fs::write(&val_json, serde_json::to_string(&sub_val).unwrap()).unwrap();

    let dets_path = root.join("sub-out/dets.json");
    let (code, lines, stderr) = folo(&[
        "val",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&coco),
        "--workers",
        "1",
        "--conf",
        "0.001",
        "--coco",
        "--save-json",
        s(&dets_path),
    ]);
    assert_eq!(code, 0, "{stderr}");
    let c = &lines[0]["coco"];
    assert_eq!((c["images"].as_u64(), c["missing_images"].as_u64()), (Some(4), Some(0)), "{c}");
    let dets: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(&dets_path).unwrap()).unwrap();
    assert_eq!(
        c["detections"].as_u64(),
        Some(dets.len() as u64),
        "coco.detections must match --save-json row count: {c}"
    );
    for d in &dets {
        assert_eq!(d["category_id"].as_u64(), Some(19), "unexpected category_id in {d}");
    }

    let check_refused = |label: &str, data_dir: &Path, out_path: &Path| {
        let (code, lines, _) = folo(&[
            "val",
            "--json",
            "--model",
            s(&best),
            "--data",
            s(data_dir),
            "--workers",
            "1",
            "--coco",
            "--save-json",
            s(out_path),
        ]);
        assert_eq!(code, 1, "{label}: {lines:?}");
        assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_dataset", "{label}: {lines:?}");
        assert!(
            !out_path.parent().unwrap().exists(),
            "{label}: refused categories must fail before creating --save-json dir"
        );
    };

    // (a) Val JSON with `"categories": []` is refused as invalid_dataset (as on main).
    let mut empty_cats = full_val.clone();
    empty_cats["categories"] = serde_json::json!([]);
    std::fs::write(&val_json, serde_json::to_string(&empty_cats).unwrap()).unwrap();
    check_refused("empty categories", &coco, &root.join("never-empty/dets.json"));

    // (b) Val JSON declares only 19=square but holds an annotation with category_id 7 (from train): refused.
    let mut undeclared_ann = full_val.clone();
    undeclared_ann["categories"] = serde_json::json!([full_val["categories"][1].clone()]);
    if !undeclared_ann["annotations"].as_array().unwrap().iter().any(|a| a["category_id"].as_u64() == Some(7))
    {
        let img_id = undeclared_ann["images"][0]["id"].as_u64().unwrap();
        undeclared_ann["annotations"].as_array_mut().unwrap().push(serde_json::json!({
            "id": 9999, "image_id": img_id, "category_id": 7, "bbox": [4.0, 4.0, 10.0, 10.0], "area": 100.0, "iscrowd": 0
        }));
    }
    std::fs::write(&val_json, serde_json::to_string(&undeclared_ann).unwrap()).unwrap();
    check_refused("undeclared annotation category", &coco, &root.join("never-undeclared/dets.json"));

    // Genuine mismatch: YOLO `data.yaml` with different classes alongside COCO annotations fails before creating
    // `--save-json`'s output folder.
    std::fs::write(&val_json, serde_json::to_string(&sub_val).unwrap()).unwrap();
    let mismatch = root.join("mismatch");
    let (code, _, stderr) =
        folo(&["dataset", "convert", s(&tiny.data), s(&mismatch), "--to", "yolo", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    std::fs::create_dir_all(mismatch.join("annotations")).unwrap();
    std::fs::copy(&val_json, mismatch.join("annotations/instances_val.json")).unwrap();
    let yaml = mismatch.join("data.yaml");
    let text =
        std::fs::read_to_string(&yaml).unwrap().replace("circle", "other").replace("square", "another");
    assert!(text.contains("another"), "expected class names in data.yaml: {text}");
    std::fs::write(&yaml, text).unwrap();
    check_refused("yolo/coco mismatch", &mismatch, &root.join("never-mismatch/dets.json"));
}
