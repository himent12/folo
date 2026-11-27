//! Regression tests for `folo predict` and `folo export`.

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
/// (docs/FORMAT.md), drops checkpoint-only keys, and stamps `folo.version` with the exporter's version.
#[test]
fn export_weights_preserves_metadata() {
    let tiny = tiny_run();
    let root = TestDir::new("cli-export-weights-meta");
    let src = root.join("best.folo");
    std::fs::copy(tiny.run.join("weights/best.folo"), &src).unwrap();
    insert_meta_keys(&src, &[("folo.custom", "keep-me"), ("folo.version", "0.0.1")]);
    let src_meta = folo_nn::io::ModelFile::open(&src).unwrap().metadata().clone();
    assert!(src_meta.metrics.is_some(), "fixture checkpoint has folo.metrics");
    assert!(src_meta.created.is_some(), "fixture checkpoint has folo.created");
    assert_eq!(src_meta.version.as_deref(), Some("0.0.1"));

    let out = root.join("weights.folo");
    let (code, _, stderr) =
        folo(&["export", "--json", "--model", s(&src), "--format", "weights", "--output", s(&out)]);
    assert_eq!(code, 0, "{stderr}");

    let file = folo_nn::io::ModelFile::open(&out).unwrap();
    let meta = file.metadata();
    assert_eq!(meta.kind, folo_nn::io::FileKind::Weights);
    assert!(!meta.fused);
    assert_eq!(
        meta.version.as_deref(),
        Some(env!("CARGO_PKG_VERSION")),
        "folo.version must be the exporter's"
    );
    assert_eq!(meta.metrics, src_meta.metrics, "folo.metrics must be preserved");
    assert_eq!(meta.created, src_meta.created, "folo.created must be preserved");
    assert_eq!(meta.extra.get("folo.custom").map(String::as_str), Some("keep-me"));
    for k in ["folo.train", "folo.optim", "folo.sched", "folo.ema", "folo.rng", "folo.epoch", "folo.best"] {
        assert!(!meta.extra.contains_key(k), "checkpoint key `{k}` must be dropped");
    }
}

/// Exporting a mixed-precision checkpoint drops the documented checkpoint-only `folo.amp` key in both `.folo`
/// weights and ONNX `metadata_props`, while preserving unknown `folo.*` keys (including `folo.nonfinite_streak`,
/// docs/FORMAT.md).
#[test]
fn export_strips_amp_and_keeps_nonfinite_streak() {
    let tiny = tiny_run();
    let root = TestDir::new("cli-strip-amp");
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

    let fused = root.join("a-fused.folo");
    let (code, _, stderr) =
        folo(&["export", "--json", "--model", s(&src), "--format", "folo", "--output", s(&fused)]);
    assert_eq!(code, 0, "folo: {stderr}");
    let meta = folo_nn::io::ModelFile::open(&fused).unwrap().metadata().clone();
    assert!(!meta.extra.contains_key("folo.amp"), "folo leaked folo.amp");
    assert_eq!(meta.extra.get("folo.nonfinite_streak").map(String::as_str), Some("2"));
    assert_eq!(meta.extra.get("folo.custom").map(String::as_str), Some("keep-me"));

    let onnx = root.join("a.onnx");
    let (code, _, stderr) =
        folo(&["export", "--json", "--model", s(&src), "--format", "onnx", "--output", s(&onnx)]);
    assert_eq!(code, 0, "onnx: {stderr}");
    let onnx_bytes = std::fs::read(&onnx).unwrap();
    let has = |needle: &[u8]| onnx_bytes.windows(needle.len()).any(|w| w == needle);
    assert!(!has(b"folo.amp"), "onnx leaked folo.amp");
    assert!(has(b"folo.nonfinite_streak"), "onnx dropped unknown folo.nonfinite_streak");
    assert!(has(b"folo.custom"), "onnx dropped unknown folo.custom");
}

/// Invalid thresholds, sizes, `--max-det 0`, `--batch 0`, and non-directory `--save` paths fail immediately with
/// exit code 2 before any model or dataset is opened.
#[test]
fn predict_val_and_export_reject_invalid_args_before_opening_model() {
    let predict_cases: &[(&[&str], &str)] = &[
        (&["predict", "--json", "--model", "none.folo", "a.jpg", "--conf=-0.1"], "--conf"),
        (&["predict", "--json", "--model", "none.folo", "a.jpg", "--conf", "1.5"], "--conf"),
        (&["predict", "--json", "--model", "none.folo", "a.jpg", "--conf", "NaN"], "--conf"),
        (&["predict", "--json", "--model", "none.folo", "a.jpg", "--iou=-0.1"], "--iou"),
        (&["predict", "--json", "--model", "none.folo", "a.jpg", "--iou", "1.5"], "--iou"),
        (&["predict", "--json", "--model", "none.folo", "a.jpg", "--iou", "NaN"], "--iou"),
        (&["predict", "--json", "--model", "none.folo", "a.jpg", "--max-det", "0"], "--max-det"),
        (&["predict", "--json", "--model", "none.folo", "a.jpg", "--imgsz", "33"], "--imgsz"),
        (&["predict", "--json", "--model", "none.folo", "a.jpg", "--imgsz", "0"], "--imgsz"),
        (&["predict", "--json", "--model", "none.folo", "a.jpg", "--imgsz", "8224"], "--imgsz"),
    ];
    for (args, flag) in predict_cases {
        let (code, lines, stderr) = folo(args);
        let e = &lines.last().unwrap_or_else(|| panic!("{args:?}: {stderr}"))["error"];
        assert_eq!((code, e["code"].as_str()), (2, Some("invalid_config")), "{args:?}: {e}");
        assert!(e["message"].as_str().unwrap().contains(flag), "{args:?}: {e}");
        assert!(e["hint"].as_str().unwrap().contains("folo describe predict --json"), "{e}");
    }

    let val_cases: &[(&[&str], &str)] = &[
        (&["val", "--json", "--model", "none.folo", "--data", "none", "--conf=-0.1"], "--conf"),
        (&["val", "--json", "--model", "none.folo", "--data", "none", "--conf", "2"], "--conf"),
        (&["val", "--json", "--model", "none.folo", "--data", "none", "--iou", "1.5"], "--iou"),
        (&["val", "--json", "--model", "none.folo", "--data", "none", "--max-det", "0"], "--max-det"),
        (&["val", "--json", "--model", "none.folo", "--data", "none", "--batch", "0"], "--batch"),
        (&["val", "--json", "--model", "none.folo", "--data", "none", "--imgsz", "33"], "--imgsz"),
    ];
    for (args, flag) in val_cases {
        let (code, lines, stderr) = folo(args);
        let e = &lines.last().unwrap_or_else(|| panic!("{args:?}: {stderr}"))["error"];
        assert_eq!((code, e["code"].as_str()), (2, Some("invalid_config")), "{args:?}: {e}");
        assert!(e["message"].as_str().unwrap().contains(flag), "{args:?}: {e}");
        assert!(e["hint"].as_str().unwrap().contains("folo describe val --json"), "{e}");
    }

    // `val` has no 8192 imgsz ceiling in the engine, so `--imgsz 8224` reaches model lookup instead of `invalid_config`.
    let (code, lines, stderr) =
        folo(&["val", "--json", "--model", "none.folo", "--data", "none", "--imgsz", "8224"]);
    let e = &lines.last().unwrap_or_else(|| panic!("val 8224: {stderr}"))["error"];
    assert_eq!((code, e["code"].as_str()), (3, Some("model_not_found")), "{e}");

    // `--save` pointing to an existing regular file or empty string is a usage error before opening the model.
    let root = TestDir::new("cli-predict-save-file");
    let file = root.join("existing.png");
    std::fs::write(&file, b"x").unwrap();
    for save in [s(&file), ""] {
        let (code, lines, stderr) =
            folo(&["predict", "--json", "--model", "none.folo", "a.jpg", "--save", save]);
        let e = &lines.last().unwrap_or_else(|| panic!("--save {save:?}: {stderr}"))["error"];
        assert_eq!((code, e["code"].as_str()), (2, Some("usage")), "--save {save:?}: {e}");
        assert!(e["message"].as_str().unwrap().contains("--save"), "{e}");
    }

    // `export` rejects `--max-det 0`, `--batch 0`, and bad `--imgsz` before opening the model.
    for (args, flag) in [
        (
            &["export", "--json", "--model", "none.folo", "--format", "onnx", "--max-det", "0"][..],
            "--max-det",
        ),
        (&["export", "--json", "--model", "none.folo", "--format", "onnx", "--batch", "0"], "--batch"),
        (&["export", "--json", "--model", "none.folo", "--format", "onnx", "--imgsz", "33"], "--imgsz"),
    ] {
        let (code, lines, stderr) = folo(args);
        let e = &lines.last().unwrap_or_else(|| panic!("{args:?}: {stderr}"))["error"];
        assert_eq!((code, e["code"].as_str()), (2, Some("usage")), "{args:?}: {e}");
        assert!(e["message"].as_str().unwrap().contains(flag), "{args:?}: {e}");
    }
}

/// Passing an existing directory as `--model` reports that it is a directory (`usage`, exit 2) rather than
/// `model_not_found` ("does not exist").
#[test]
fn model_directory_reports_directory_not_missing() {
    let root = TestDir::new("cli-model-is-dir");
    let dir = root.join("weights_folder");
    std::fs::create_dir_all(&dir).unwrap();
    let cases: &[&[&str]] = &[
        &["predict", "--json", "--model", s(&dir), "a.jpg"],
        &["val", "--json", "--model", s(&dir), "--data", "none"],
        &["export", "--json", "--model", s(&dir)],
    ];
    for args in cases {
        let (code, lines, stderr) = folo(args);
        let e = &lines.last().unwrap_or_else(|| panic!("{args:?}: {stderr}"))["error"];
        assert_eq!((code, e["code"].as_str()), (2, Some("usage")), "{args:?}: {e}");
        let msg = e["message"].as_str().unwrap();
        assert!(msg.contains("directory") && !msg.contains("does not exist"), "{args:?}: {e}");
    }
}

/// Opening a dataset as a bare `data.yaml` in `folo train` derives the run name from its parent folder instead of
/// falling back to literal `dataset`.
#[test]
fn train_with_bare_data_yaml_derives_parent_folder_name() {
    let root = TestDir::new("cli-bare-yaml-train");
    let ds = root.join("pebbles");
    let cfg = folo_data::synthetic::SyntheticConfig {
        images: 4,
        classes: vec!["circle".into()],
        imgsz: 64,
        seed: 7,
        vary_aspect: false,
        val_fraction: 0.25,
        ..Default::default()
    };
    folo_data::synthetic::write_yolo(&cfg, &ds, folo_data::ImageFormat::Png).unwrap();
    let project = root.join("runs");
    let out = super::folo_cmd()
        .args([
            "train",
            "--json",
            "--data",
            "data.yaml",
            "--imgsz",
            "64",
            "--batch",
            "2",
            "--epochs",
            "1",
            "--workers",
            "1",
            "--device",
            "cpu",
            "--project",
            s(&project),
        ])
        .current_dir(&ds)
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert_eq!(out.status.code(), Some(0), "stdout: {stdout}\nstderr: {stderr}");
    let lines: Vec<serde_json::Value> = stdout.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let done = lines.last().unwrap();
    assert_eq!(done["event"], "done", "{done}");
    assert!(
        done["run_dir"].as_str().is_some_and(|p| p.ends_with("/pebbles-n")),
        "expected parent folder name in run_dir, not dataset-n: {done}"
    );
    assert!(project.join("pebbles-n").is_dir());
}

/// A failing `folo export` leaves neither a half-written output file nor any `.tmp` files behind.
#[test]
fn failed_export_leaves_no_partial_output_or_temp_files() {
    let tiny = tiny_run();
    let best = tiny.run.join("weights/best.folo");
    let root = TestDir::new("cli-export-clean-fail");
    let out_dir = root.join("out");
    std::fs::create_dir_all(&out_dir).unwrap();

    // 1. Failing ONNX export (`--head o2o` on an `o2m` model) must leave the output directory completely empty.
    let onnx_out = out_dir.join("bad-o2o.onnx");
    let (code, lines, _) = folo(&[
        "export",
        "--json",
        "--model",
        s(&best),
        "--format",
        "onnx",
        "--head",
        "o2o",
        "--output",
        s(&onnx_out),
    ]);
    assert_ne!(code, 0);
    assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_config", "{lines:?}");
    assert!(!onnx_out.exists(), "failed export must not create target file");
    assert_eq!(std::fs::read_dir(&out_dir).unwrap().count(), 0, "failed export left temp files behind");

    // 2. Corrupt tensor payload (header intact, tensor data truncated) fails cleanly without leaving target or `.tmp`.
    let corrupt = root.join("corrupt.folo");
    let mut bytes = std::fs::read(&best).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    bytes.truncate(8 + header_len + 16);
    std::fs::write(&corrupt, &bytes).unwrap();
    let folo_out = out_dir.join("from-corrupt.folo");
    let (code, lines, _) = folo(&["export", "--json", "--model", s(&corrupt), "--output", s(&folo_out)]);
    assert_eq!(code, 1);
    assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_weights", "{lines:?}");
    assert!(!folo_out.exists());
    assert_eq!(std::fs::read_dir(&out_dir).unwrap().count(), 0, "corrupt export left temp files behind");
}

/// `folo val --coco` / `--save-json` accepts a split JSON declaring a valid subset of the dataset's categories
/// when the model matches the dataset's full class order, and fails fast *before* inference (and without creating
/// `--save-json`'s directory) on empty categories, undeclared annotation category IDs, mismatched model class
/// order, empty splits, or malformed COCO JSON (#422).
#[test]
fn val_coco_subset_categories_and_fail_fast_validation() {
    let tiny = tiny_run();
    let best = tiny.run.join("weights/best.folo");
    let root = TestDir::new("cli-val-coco-failfast");

    // Convert the 2-class (`circle`, `square`) synthetic dataset to COCO.
    let coco = root.join("coco");
    let (code, _, stderr) = folo(&["dataset", "convert", s(&tiny.data), s(&coco), "--to", "coco", "--json"]);
    assert_eq!(code, 0, "{stderr}");

    // Build a COCO copy whose `instances_val.json` declares ONLY the second category (`square`, id 2) while
    // `instances_train.json` declares both `circle` (1) and `square` (2).
    let subset = root.join("coco_subset");
    let (code, _, stderr) =
        folo(&["dataset", "convert", s(&tiny.data), s(&subset), "--to", "coco", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let val_path = subset.join("annotations/instances_val.json");
    let mut val_doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&val_path).unwrap()).unwrap();
    let square_id = val_doc["categories"].as_array().unwrap().iter().find(|c| c["name"] == "square").unwrap()
        ["id"]
        .as_u64()
        .unwrap();
    val_doc["categories"].as_array_mut().unwrap().retain(|c| c["id"].as_u64() == Some(square_id));
    val_doc["annotations"].as_array_mut().unwrap().retain(|a| a["category_id"].as_u64() == Some(square_id));
    std::fs::write(&val_path, val_doc.to_string()).unwrap();

    // 1. Valid subset split with matching model class order (`["circle", "square"]`) succeeds and writes only
    //    `square_id` detections.
    let dets_out = root.join("subset_eval/dets.json");
    let (code, lines, stderr) = folo(&[
        "val",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&subset),
        "--workers",
        "1",
        "--coco",
        "--save-json",
        s(&dets_out),
    ]);
    assert_eq!(code, 0, "{stderr}");
    let coco_block = &lines[0]["coco"];
    assert!(coco_block.is_object(), "{lines:?}");
    let written: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(&dets_out).unwrap()).unwrap();
    assert!(
        written.iter().all(|d| d["category_id"].as_u64() == Some(square_id)),
        "subset val must only write detections for categories declared in the split: {written:?}"
    );

    // 2. Model whose class order differs (`["square", "circle"]`) fails fast before creating `--save-json` dir.
    let swapped = root.join("swapped.folo");
    std::fs::copy(&best, &swapped).unwrap();
    insert_meta_keys(&swapped, &[("folo.classes", r#"["square","circle"]"#)]);
    let refused_swap = root.join("refused_swap/dets.json");
    let (code, lines, _) = folo(&[
        "val",
        "--json",
        "--model",
        s(&swapped),
        "--data",
        s(&subset),
        "--workers",
        "1",
        "--coco",
        "--save-json",
        s(&refused_swap),
    ]);
    assert_eq!(code, 1);
    assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_dataset", "{lines:?}");
    assert!(!refused_swap.parent().unwrap().exists(), "must fail before creating --save-json dir");

    // 3. Empty `categories: []` fails fast before inference and before creating `--save-json` dir.
    let bad_empty = root.join("coco_empty_cats");
    let (code, _, stderr) =
        folo(&["dataset", "convert", s(&tiny.data), s(&bad_empty), "--to", "coco", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let p = bad_empty.join("annotations/instances_val.json");
    let mut doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    doc["categories"] = serde_json::json!([]);
    doc["annotations"] = serde_json::json!([]);
    std::fs::write(&p, doc.to_string()).unwrap();
    let refused_empty = root.join("refused_empty/dets.json");
    let (code, lines, _) = folo(&[
        "val",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&bad_empty),
        "--workers",
        "1",
        "--coco",
        "--save-json",
        s(&refused_empty),
    ]);
    assert_eq!(code, 1);
    assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_dataset", "{lines:?}");
    assert!(!refused_empty.parent().unwrap().exists());

    // 4. Annotation referencing an undeclared `category_id` fails fast before inference.
    let bad_undecl = root.join("coco_undeclared_cat");
    let (code, _, stderr) =
        folo(&["dataset", "convert", s(&tiny.data), s(&bad_undecl), "--to", "coco", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let p = bad_undecl.join("annotations/instances_val.json");
    let mut doc: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    doc["categories"].as_array_mut().unwrap().retain(|c| c["id"].as_u64() == Some(square_id));
    // Keep annotations from both classes (including the removed category id), or inject an undeclared id.
    let first_img_id = doc["images"][0]["id"].as_u64().unwrap();
    doc["annotations"].as_array_mut().unwrap().push(serde_json::json!({
        "id": 9999,
        "image_id": first_img_id,
        "category_id": 999,
        "bbox": [4, 4, 10, 10],
        "area": 100.0,
        "iscrowd": 0
    }));
    std::fs::write(&p, doc.to_string()).unwrap();
    let refused_undecl = root.join("refused_undecl/dets.json");
    let (code, lines, _) = folo(&[
        "val",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&bad_undecl),
        "--workers",
        "1",
        "--coco",
        "--save-json",
        s(&refused_undecl),
    ]);
    assert_eq!(code, 1);
    assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_dataset", "{lines:?}");
    assert!(!refused_undecl.parent().unwrap().exists());

    // 5. Conflicting category names across splits (train calls id 1 "ring", val calls id 1 "circle") fail fast
    //    with `invalid_dataset` even when val declares every category and matches the model's names.
    let bad_conflict = root.join("coco_cat_conflict");
    let (code, _, stderr) =
        folo(&["dataset", "convert", s(&tiny.data), s(&bad_conflict), "--to", "coco", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let train_p = bad_conflict.join("annotations/instances_train.json");
    let train_text = std::fs::read_to_string(&train_p).unwrap().replace("\"circle\"", "\"ring\"");
    std::fs::write(&train_p, train_text).unwrap();
    let refused_conflict = root.join("refused_conflict/dets.json");
    let (code, lines, _) = folo(&[
        "val",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&bad_conflict),
        "--workers",
        "1",
        "--coco",
        "--save-json",
        s(&refused_conflict),
    ]);
    assert_eq!(code, 1);
    assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_dataset", "{lines:?}");
    assert!(!refused_conflict.parent().unwrap().exists());

    // 6. Empty split (`--split test` with no test images) fails fast with `invalid_dataset` and no folder created.
    let refused_split = root.join("refused_split/dets.json");
    let (code, lines, _) = folo(&[
        "val",
        "--json",
        "--model",
        s(&best),
        "--data",
        s(&coco),
        "--split",
        "test",
        "--workers",
        "1",
        "--save-json",
        s(&refused_split),
    ]);
    assert_eq!(code, 1);
    assert_eq!(lines.last().unwrap()["error"]["code"], "invalid_dataset", "{lines:?}");
    assert!(!refused_split.parent().unwrap().exists());
}

/// `folo predict` exposes SAHI tiled inference (`--tile`, `--tile-overlap`, `--no-tile-full-image`, `--tile-batch`,
/// `--tile-merge`) and TTA (`--tta`, `--tta-scales`, `--no-tta-hflip`), validates ranges and mutual exclusion upfront
/// before opening the model, and runs end to end on real CPU weights.
#[test]
fn predict_sahi_tile_and_tta_validation_and_execution() {
    // 1. Upfront invalid_config checks (before opening model).
    let invalid_cases: &[(&[&str], &str)] = &[
        (&["predict", "--json", "--model", "none.folo", "a.jpg", "--tile", "0"], "--tile"),
        (
            &["predict", "--json", "--model", "none.folo", "a.jpg", "--tile", "64", "--tile-overlap", "1.0"],
            "--tile-overlap",
        ),
        (
            &["predict", "--json", "--model", "none.folo", "a.jpg", "--tile", "64", "--tile-overlap=-0.1"],
            "--tile-overlap",
        ),
        (
            &["predict", "--json", "--model", "none.folo", "a.jpg", "--tile", "64", "--tile-batch", "0"],
            "--tile-batch",
        ),
        (&["predict", "--json", "--model", "none.folo", "a.jpg", "--tta", "--tta-scales", "0"], "tta scales"),
        (
            &[
                "predict",
                "--json",
                "--model",
                "none.folo",
                "a.jpg",
                "--imgsz",
                "64",
                "--tta",
                "--tta-scales",
                "200",
            ],
            "8192",
        ),
    ];
    for (args, expected) in invalid_cases {
        let (code, lines, stderr) = folo(args);
        let e = &lines.last().unwrap_or_else(|| panic!("{args:?}: {stderr}"))["error"];
        assert_eq!((code, e["code"].as_str()), (2, Some("invalid_config")), "{args:?}: {e}");
        assert!(e["message"].as_str().unwrap().contains(expected), "{args:?}: {e}");
    }

    // 2. Mutual exclusion and peer-flag requirements are enforced by clap as `usage` (exit 2).
    for args in [
        &["predict", "--json", "--model", "none.folo", "a.jpg", "--tile", "64", "--tta"][..],
        &["predict", "--json", "--model", "none.folo", "a.jpg", "--tile-overlap", "0.2"],
        &["predict", "--json", "--model", "none.folo", "a.jpg", "--tta-scales", "1.0"],
        &["predict", "--json", "--model", "none.folo", "a.jpg", "--no-tta-hflip"],
    ] {
        let (code, lines, stderr) = folo(args);
        let e = &lines.last().unwrap_or_else(|| panic!("{args:?}: {stderr}"))["error"];
        assert_eq!((code, e["code"].as_str()), (2, Some("usage")), "{args:?}: {e}");
    }

    // 3. End-to-end SAHI tiled inference and TTA execution on a real tiny model.
    let tiny = tiny_run();
    let best = tiny.run.join("weights/best.folo");
    let src_img = tiny.data.join("images/val").read_dir().unwrap().next().unwrap().unwrap().path();

    let (code, lines, stderr) = folo(&[
        "predict",
        "--json",
        "--device",
        "cpu",
        "--model",
        s(&best),
        s(&src_img),
        "--conf",
        "0.001",
        "--tile",
        "32",
        "--tile-overlap",
        "0.25",
        "--tile-batch",
        "2",
        "--tile-merge",
        "iou",
        "--no-tile-full-image",
    ]);
    assert_eq!(code, 0, "tile: {stderr}");
    assert_eq!(lines[0]["event"], "prediction", "{lines:?}");
    assert_eq!(lines.last().unwrap()["event"], "done", "{lines:?}");

    // Also verify `--tta-scales` placed BEFORE positional `SOURCES` does not consume the image path.
    let (code, lines, stderr) = folo(&[
        "predict",
        "--json",
        "--device",
        "cpu",
        "--model",
        s(&best),
        "--conf",
        "0.001",
        "--tta",
        "--tta-scales",
        "1.0,1.5",
        "--no-tta-hflip",
        s(&src_img),
    ]);
    assert_eq!(code, 0, "tta before positional source: {stderr}");
    assert_eq!(lines[0]["event"], "prediction", "{lines:?}");
    assert_eq!(lines.last().unwrap()["event"], "done", "{lines:?}");
}
