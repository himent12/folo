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
