//! Calibrated per-class F1 threshold persistence (`folo val --save-thresholds`) and loading
//! (`folo predict --thresholds`).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use folo_vision::ThresholdCalibration;
use serde_json::{Value, json};

use crate::output::{CliError, ExitCode};

/// Maximum JSON file size accepted by `--thresholds` (prevents accidentally reading weights or images).
const MAX_THRESHOLDS_BYTES: u64 = 4 * 1024 * 1024;

/// Normalizes a path for destination-collision checks (resolving symlinks on the longest existing ancestor and
/// folding `.` / `..`, case-insensitively on Windows).
fn path_identity(path: &Path) -> String {
    let mut abs = PathBuf::new();
    for c in std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()).components() {
        match c {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                abs.pop();
            }
            c => abs.push(c),
        }
    }
    let resolved = abs
        .ancestors()
        .find_map(|a| Some(std::fs::canonicalize(a).ok()?.join(abs.strip_prefix(a).ok()?)))
        .unwrap_or(abs);
    let s = resolved.to_string_lossy().into_owned();
    if cfg!(windows) { s.to_lowercase() } else { s }
}

/// Checks `--save-thresholds <FILE>` upfront before opening the model or running validation, rejecting collisions
/// with `--model` (the resolved weights file) or `--save-json`.
pub(super) fn check_save_thresholds_path(
    path: Option<&Path>,
    model: &Path,
    save_json: Option<&Path>,
) -> Result<(), CliError> {
    let Some(path) = path else { return Ok(()) };
    if path.as_os_str().is_empty() {
        return Err(CliError::new(ExitCode::Usage, "usage", "--save-thresholds file path must not be empty")
            .hint("pass a JSON file path, e.g. --save-thresholds thresholds.json"));
    }
    if std::fs::metadata(path).is_ok_and(|m| m.is_dir()) {
        return Err(CliError::new(
            ExitCode::Usage,
            "usage",
            format!("--save-thresholds {} is a folder, not a file", folo_api::json::path(path)),
        )
        .hint("pass a JSON file path, e.g. --save-thresholds thresholds.json"));
    }
    if path_identity(path) == path_identity(model) {
        return Err(CliError::new(ExitCode::Usage, "usage", "--save-thresholds must differ from --model")
            .hint("pass a separate JSON path, e.g. --save-thresholds thresholds.json"));
    }
    if let Some(json_out) = save_json
        && !json_out.as_os_str().is_empty()
        && path_identity(path) == path_identity(json_out)
    {
        return Err(CliError::new(
            ExitCode::Usage,
            "usage",
            "--save-thresholds and --save-json must point to different files",
        )
        .hint("pass distinct paths, e.g. --save-json detections.json --save-thresholds thresholds.json"));
    }
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty())
        && std::fs::metadata(parent).is_ok_and(|m| !m.is_dir())
    {
        return Err(CliError::new(
            ExitCode::Runtime,
            "io_error",
            format!(
                "cannot create parent folder for {}: {} is not a folder",
                folo_api::json::path(path),
                folo_api::json::path(parent)
            ),
        ));
    }
    Ok(())
}

/// Ensures the parent folder for `--save-thresholds <FILE>` exists (called after config validation, before eval).
pub(super) fn prepare_save_thresholds(
    path: &Path,
    model: &Path,
    save_json: Option<&Path>,
) -> Result<(), CliError> {
    check_save_thresholds_path(Some(path), model, save_json)?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| {
            CliError::new(
                ExitCode::Runtime,
                "io_error",
                format!("cannot create {}: {e}", folo_api::json::path(parent)),
            )
        })?;
    }
    Ok(())
}

/// Formats `cal` as JSON for both `folo val --json` and `--save-thresholds` (preserves exact `f32` operating
/// points and the `folo_vision::ThresholdCalibration` schema, adding human-readable class `name` fields).
pub(super) fn thresholds_json(cal: &ThresholdCalibration, classes: &[String]) -> Value {
    let name = |c: u32| classes.get(c as usize).cloned().unwrap_or_else(|| c.to_string());
    json!({
        "global": {
            "best_conf": cal.global.best_conf,
            "best_f1": cal.global.best_f1,
            "precision": cal.global.precision,
            "recall": cal.global.recall,
        },
        "macro_f1": cal.macro_f1,
        "per_class": cal.per_class.iter().map(|c| json!({
            "class": c.class,
            "name": name(c.class),
            "best_conf": c.best_conf,
            "best_f1": c.best_f1,
            "precision": c.precision,
            "recall": c.recall,
            "truths": c.truths,
            "preds": c.preds,
        })).collect::<Vec<_>>(),
    })
}

/// Atomically writes `cal` as pretty-printed JSON to `path` via a unique temp file + `sync_all` + `rename_over`.
pub(super) fn save_thresholds(
    path: &Path,
    cal: &ThresholdCalibration,
    classes: &[String],
) -> Result<(), CliError> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| {
            CliError::new(
                ExitCode::Runtime,
                "io_error",
                format!("cannot create {}: {e}", folo_api::json::path(parent)),
            )
        })?;
    }
    let mut bytes = serde_json::to_vec_pretty(&thresholds_json(cal, classes)).map_err(|e| {
        CliError::new(ExitCode::Runtime, "internal", format!("serializing thresholds JSON: {e}"))
    })?;
    bytes.push(b'\n');

    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let dst_name = path.file_name().map(|n| n.to_string_lossy().to_lowercase());
    let io = |e: std::io::Error| {
        CliError::new(
            ExitCode::Runtime,
            "io_error",
            format!("cannot write {}: {e}", folo_api::json::path(path)),
        )
    };
    let (tmp, mut f) = loop {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("{}{}-thr-{n}.tmp", folo_nn::io::TEMP_PREFIX, std::process::id());
        if Some(name.to_lowercase()) == dst_name {
            continue;
        }
        let candidate: PathBuf = dir.join(&name);
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&candidate) {
            Ok(f) => break (candidate, f),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(io(e)),
        }
    };
    let res = (|| -> std::io::Result<()> {
        f.write_all(&bytes)?;
        f.sync_all()?;
        drop(f);
        folo_nn::io::rename_over(&tmp, path)
    })();
    if let Err(e) = res {
        let _ = std::fs::remove_file(&tmp);
        return Err(io(e));
    }
    Ok(())
}

/// Loads and validates a saved [`ThresholdCalibration`] JSON file for `folo predict --thresholds`.
pub(super) fn load_thresholds(path: &Path) -> Result<ThresholdCalibration, CliError> {
    if path.as_os_str().is_empty() {
        return Err(CliError::new(ExitCode::Usage, "usage", "--thresholds file path must not be empty")
            .hint("pass a JSON file saved by `folo val --save-thresholds <path>`"));
    }
    let shown = folo_api::json::path(path);
    let meta = std::fs::metadata(path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => CliError::new(
            ExitCode::NotFound,
            "path_not_found",
            format!("--thresholds {shown} does not exist"),
        )
        .hint("generate one with `folo val --model <weights> --data <dataset> --save-thresholds <path>`"),
        _ => CliError::new(ExitCode::Runtime, "io_error", format!("cannot read {shown}: {e}")),
    })?;
    if !meta.is_file() {
        return Err(CliError::new(ExitCode::Usage, "usage", format!("--thresholds {shown} is not a file"))
            .hint("pass a JSON file saved by `folo val --save-thresholds <path>`"));
    }
    if meta.len() > MAX_THRESHOLDS_BYTES {
        return Err(CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!("--thresholds {shown} is too large ({} bytes, max {MAX_THRESHOLDS_BYTES})", meta.len()),
        )
        .hint("pass a JSON file saved by `folo val --save-thresholds <path>`"));
    }
    let bytes = std::fs::read(path)
        .map_err(|e| CliError::new(ExitCode::Runtime, "io_error", format!("cannot read {shown}: {e}")))?;
    let cal: ThresholdCalibration = serde_json::from_slice(&bytes).map_err(|e| {
        CliError::new(
            ExitCode::Usage,
            "invalid_config",
            format!("--thresholds {shown} is not valid ThresholdCalibration JSON: {e}"),
        )
        .hint("pass a JSON file saved by `folo val --save-thresholds <path>`")
    })?;
    cal.validate().map_err(|e| {
        CliError::new(ExitCode::Usage, "invalid_config", format!("--thresholds {shown}: {e}"))
            .hint("regenerate with `folo val --save-thresholds <path>`")
    })?;
    Ok(cal)
}

/// Ensures every class id in `cal.per_class` exists in a model with `num_classes` classes.
pub(super) fn check_thresholds_for_model(
    path: &Path,
    cal: &ThresholdCalibration,
    num_classes: usize,
) -> Result<(), CliError> {
    for c in &cal.per_class {
        if (c.class as usize) >= num_classes {
            return Err(CliError::new(
                ExitCode::Usage,
                "invalid_config",
                format!(
                    "--thresholds {}: class {} is out of range for model with {num_classes} classes (0..{num_classes})",
                    folo_api::json::path(path),
                    c.class
                ),
            )
            .hint("use a thresholds file calibrated for this model's classes"));
        }
    }
    Ok(())
}

/// Concise human-readable lines summarizing calibrated thresholds for `folo val`.
pub(super) fn human_val_lines(
    cal: &ThresholdCalibration,
    classes: &[String],
    saved: Option<&Path>,
) -> Vec<String> {
    let saved_note = saved.map(|p| format!("; saved to {}", folo_api::json::path(p))).unwrap_or_default();
    if cal.is_empty() {
        return if saved.is_some() {
            vec![format!("calibrated thresholds: none (no ground-truth boxes){saved_note}")]
        } else {
            Vec::new()
        };
    }
    let name = |c: u32| classes.get(c as usize).cloned().unwrap_or_else(|| c.to_string());
    let mut lines = vec![format!(
        "calibrated thresholds: global conf {:.3} (F1 {:.3}), macro-F1 {:.3} ({} classes){saved_note}",
        cal.global.best_conf,
        cal.global.best_f1,
        cal.macro_f1,
        cal.per_class.len()
    )];
    const MAX_SHOWN: usize = 8;
    let shown: Vec<String> = cal
        .per_class
        .iter()
        .take(MAX_SHOWN)
        .map(|c| format!("{} {:.3} (F1 {:.3})", name(c.class), c.best_conf, c.best_f1))
        .collect();
    let extra = if cal.per_class.len() > MAX_SHOWN {
        format!(", +{} more", cal.per_class.len() - MAX_SHOWN)
    } else {
        String::new()
    };
    lines.push(format!("  per-class conf: {}{extra}", shown.join(", ")));
    lines
}

/// Metadata emitted in `folo predict --json`'s `done` event when `--thresholds` is active.
pub(super) fn predict_thresholds_json(path: &Path, cal: &ThresholdCalibration) -> Value {
    json!({
        "path": crate::output::record_path(path),
        "active": !cal.is_empty(),
        "global_conf": cal.global.best_conf,
        "macro_f1": cal.macro_f1,
        "per_class_count": cal.per_class.len(),
    })
}

#[cfg(test)]
mod tests {
    use folo_test_support::TestDir;
    use folo_vision::{ClassThreshold, OperatingPoint};

    use super::*;

    fn sample_calibration() -> ThresholdCalibration {
        ThresholdCalibration {
            global: OperatingPoint { best_conf: 0.35, best_f1: 0.85, precision: 0.88, recall: 0.82 },
            per_class: vec![
                ClassThreshold {
                    class: 0,
                    best_conf: 0.25,
                    best_f1: 0.90,
                    precision: 0.91,
                    recall: 0.89,
                    truths: 12,
                    preds: 11,
                },
                ClassThreshold {
                    class: 1,
                    best_conf: 0.55,
                    best_f1: 0.80,
                    precision: 0.85,
                    recall: 0.76,
                    truths: 8,
                    preds: 7,
                },
            ],
            macro_f1: 0.85,
        }
    }

    #[test]
    fn save_and_load_round_trip_and_atomic_overwrite() {
        let dir = TestDir::new("cli-thresholds-roundtrip");
        let path = dir.join("nested").join("thresholds.json");
        let cal = sample_calibration();
        let classes = vec!["circle".into(), "square".into()];

        save_thresholds(&path, &cal, &classes).unwrap();
        let loaded = load_thresholds(&path).unwrap();
        assert_eq!(loaded, cal);

        // File contains readable class names alongside exact operating points.
        let raw: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(raw["per_class"][0]["name"], "circle");
        assert_eq!(raw["per_class"][1]["name"], "square");

        // Atomic overwrite leaves no leftover temp files.
        let mut updated = cal.clone();
        updated.per_class[0].best_conf = 0.42;
        save_thresholds(&path, &updated, &classes).unwrap();
        assert_eq!(load_thresholds(&path).unwrap(), updated);
        let names: Vec<String> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["thresholds.json"]);
    }

    #[test]
    fn load_and_bounds_checks_reject_bad_inputs() {
        let dir = TestDir::new("cli-thresholds-errors");

        // Missing file -> ExitCode::NotFound / path_not_found
        let missing = load_thresholds(&dir.join("missing.json")).unwrap_err();
        assert_eq!((missing.exit, missing.code.as_str()), (ExitCode::NotFound, "path_not_found"));

        // Empty path and directory path -> ExitCode::Usage / usage
        assert_eq!(load_thresholds(Path::new("")).unwrap_err().code, "usage");
        assert_eq!(load_thresholds(dir.path()).unwrap_err().code, "usage");
        let model_file = dir.join("best.folo");
        std::fs::write(&model_file, b"weights").unwrap();
        assert_eq!(
            check_save_thresholds_path(Some(Path::new("")), &model_file, None).unwrap_err().code,
            "usage"
        );
        assert_eq!(
            check_save_thresholds_path(Some(dir.path()), &model_file, None).unwrap_err().code,
            "usage"
        );
        assert_eq!(
            check_save_thresholds_path(Some(&model_file), &model_file, None).unwrap_err().code,
            "usage"
        );
        let shared = dir.join("same.json");
        assert_eq!(
            check_save_thresholds_path(Some(&shared), &model_file, Some(&shared)).unwrap_err().code,
            "usage"
        );

        // Malformed JSON -> invalid_config
        let bad_json = dir.join("bad.json");
        std::fs::write(&bad_json, b"{not json").unwrap();
        let e = load_thresholds(&bad_json).unwrap_err();
        assert_eq!((e.exit, e.code.as_str()), (ExitCode::Usage, "invalid_config"));

        // Out-of-range threshold and duplicate class ids -> invalid_config
        let out_of_range = dir.join("oor.json");
        std::fs::write(
            &out_of_range,
            br#"{"global":{"best_conf":1.5,"best_f1":0.5,"precision":0.5,"recall":0.5},"per_class":[],"macro_f1":0.0}"#,
        )
        .unwrap();
        assert_eq!(load_thresholds(&out_of_range).unwrap_err().code, "invalid_config");

        let dup_cls = dir.join("dup.json");
        std::fs::write(
            &dup_cls,
            br#"{"global":{"best_conf":0.3,"best_f1":0.5,"precision":0.5,"recall":0.5},"per_class":[{"class":0,"best_conf":0.2,"best_f1":0.5,"precision":0.5,"recall":0.5,"truths":5,"preds":5},{"class":0,"best_conf":0.4,"best_f1":0.5,"precision":0.5,"recall":0.5,"truths":5,"preds":5}],"macro_f1":0.5}"#,
        )
        .unwrap();
        assert_eq!(load_thresholds(&dup_cls).unwrap_err().code, "invalid_config");

        // Model class count bounds check
        let cal = sample_calibration();
        check_thresholds_for_model(&bad_json, &cal, 2).unwrap();
        let oob = check_thresholds_for_model(&bad_json, &cal, 1).unwrap_err();
        assert_eq!((oob.exit, oob.code.as_str()), (ExitCode::Usage, "invalid_config"));
        assert!(oob.message.contains("class 1"), "{}", oob.message);
    }
}
