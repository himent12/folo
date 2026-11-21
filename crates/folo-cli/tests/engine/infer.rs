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

/// `folo predict` on a folder containing a self-referencing directory link visits each real directory once
/// (instead of looping until the OS symlink/reparse limit and emitting duplicate predictions) while still following
/// a non-cyclic link to an external folder.
#[test]
fn predict_skips_directory_link_cycles() {
    let tiny = tiny_run();
    let best = tiny.run.join("weights/best.folo");
    let src_img = tiny.data.join("images/val").read_dir().unwrap().next().unwrap().unwrap().path();

    let root = TestDir::new("cli-predict-cycle");
    let input = root.join("input");
    let external = root.join("external");
    std::fs::create_dir_all(&input).unwrap();
    std::fs::create_dir_all(&external).unwrap();
    std::fs::copy(&src_img, input.join("a.png")).unwrap();
    std::fs::copy(&src_img, external.join("b.png")).unwrap();

    // Self-loop plus a legitimate non-cyclic link to an external folder.
    let _self_loop = DirLink::new(&input, input.join("loop"));
    let _ext_link = DirLink::new(&external, input.join("extra"));

    let (code, lines, stderr) = folo(&["predict", "--json", "--model", s(&best), s(&input)]);
    assert_eq!(code, 0, "{stderr}");
    let preds: Vec<&str> = lines
        .iter()
        .filter(|l| l["event"] == "prediction")
        .map(|l| l["file"].as_str().unwrap())
        .collect();
    assert_eq!(preds.len(), 2, "expected 2 images (a.png and extra/b.png), got {preds:?}");
    assert!(preds.iter().any(|p| p.ends_with("/a.png")), "{preds:?}");
    assert!(preds.iter().any(|p| p.ends_with("/extra/b.png")), "{preds:?}");
    let done = lines.last().unwrap();
    assert_eq!((done["event"].as_str(), done["images"].as_u64()), (Some("done"), Some(2)), "{done}");
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
