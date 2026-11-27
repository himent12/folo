//! Integration tests for `folo dataset split`, Pascal VOC support across dataset commands, and duplicate /
//! train-val split-leakage reporting in `folo dataset check`.

use std::path::Path;
use std::process::Command;

use folo_test_support::TestDir;
use serde_json::Value;

fn folo(args: &[&str]) -> (i32, Vec<Value>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_folo")).args(args).output().expect("spawn folo");
    let stdout = String::from_utf8(out.stdout).expect("utf8 stdout");
    let stderr = String::from_utf8(out.stderr).expect("utf8 stderr");
    let lines: Vec<Value> = stdout
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str(l).unwrap_or_else(|e| {
                panic!(
                    "invalid json on stdout ({e})\nargs: {args:?}\nline: {l}\nstdout:\n{stdout}\nstderr:\n{stderr}"
                );
            })
        })
        .collect();
    assert!(!lines.is_empty(), "empty stdout for {args:?}\nstderr:\n{stderr}");
    (out.status.code().unwrap_or(-1), lines, stderr)
}

fn folo_one(args: &[&str]) -> (i32, Value, String) {
    let (code, lines, stderr) = folo(args);
    (code, lines.into_iter().last().unwrap(), stderr)
}

fn folo_human(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_folo")).args(args).output().expect("spawn folo");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8(out.stdout).expect("utf8 stdout"),
        String::from_utf8(out.stderr).expect("utf8 stderr"),
    )
}

fn s(p: &Path) -> &str {
    p.to_str().expect("utf-8 test path")
}

#[test]
fn dataset_split_is_deterministic_stratified_and_supports_format_conversion() {
    let dir = TestDir::new("cli-dataset-split");
    let src = dir.join("src_yolo");
    let (code, _, stderr) = folo(&[
        "dataset",
        "synth",
        s(&src),
        "--images",
        "20",
        "--imgsz",
        "64",
        "--classes",
        "circle,square",
        "--seed",
        "1",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}");

    // 3-way split preserving YOLO format.
    let out_a = dir.join("split_a");
    let (code, a, stderr) = folo_one(&[
        "dataset",
        "split",
        s(&src),
        "--out",
        s(&out_a),
        "--val",
        "0.2",
        "--test",
        "0.2",
        "--seed",
        "42",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(a["from"], "yolo");
    assert_eq!(a["to"], "yolo");
    assert_eq!(a["seed"], 42);
    assert_eq!(a["splits"]["train"], 12);
    assert_eq!(a["splits"]["val"], 4);
    assert_eq!(a["splits"]["test"], 4);
    assert_eq!(a["skipped_images"], 0);

    // Same seed reproduces the exact same split counts and boxes.
    let out_b = dir.join("split_b");
    let (code, b, stderr) = folo_one(&[
        "dataset",
        "split",
        s(&src),
        "--out",
        s(&out_b),
        "--val",
        "0.2",
        "--test",
        "0.2",
        "--seed",
        "42",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(a["splits"], b["splits"]);
    assert_eq!(a["boxes"], b["boxes"]);

    // Output passes `folo dataset check` cleanly with zero split leakage.
    let (code, chk, stderr) = folo_one(&["dataset", "check", s(&out_a), "--json"]);
    assert_eq!(code, 0, "{stderr}: {chk}");
    assert_eq!(chk["clean"], true);
    assert_eq!(chk["counts"]["split_leakage"], 0);

    // Split directly into COCO and Pascal VOC formats.
    let out_coco = dir.join("split_coco");
    let (code, c, stderr) = folo_one(&[
        "dataset",
        "split",
        s(&src),
        "--out",
        s(&out_coco),
        "--val",
        "0.25",
        "--to",
        "coco",
        "--seed",
        "9",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(c["to"], "coco");
    assert_eq!(c["splits"]["train"], 15);
    assert_eq!(c["splits"]["val"], 5);
    assert_eq!(c["splits"]["test"], 0);

    let out_voc = dir.join("split_voc");
    let (code, v, stderr) = folo_one(&[
        "dataset",
        "split",
        s(&src),
        "--out",
        s(&out_voc),
        "--val",
        "0.25",
        "--to",
        "voc",
        "--seed",
        "9",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(v["to"], "voc");
    assert_eq!(v["splits"]["train"], 15);
    assert_eq!(v["splits"]["val"], 5);
    assert!(out_voc.join("Annotations").is_dir());
    assert!(out_voc.join("JPEGImages").is_dir());
}

#[test]
fn dataset_voc_convert_stats_check_and_split_round_trip() {
    let dir = TestDir::new("cli-dataset-voc");
    let src = dir.join("yolo");
    let (code, _, stderr) = folo(&[
        "dataset",
        "synth",
        s(&src),
        "--images",
        "12",
        "--imgsz",
        "64",
        "--classes",
        "circle,square",
        "--seed",
        "3",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}");

    // Convert YOLO -> Pascal VOC.
    let voc = dir.join("voc");
    let (code, conv, stderr) = folo_one(&["dataset", "convert", s(&src), s(&voc), "--to", "voc", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!((conv["from"].as_str(), conv["to"].as_str()), (Some("yolo"), Some("voc")));
    assert_eq!(conv["splits"]["train"].as_u64().unwrap() + conv["splits"]["val"].as_u64().unwrap(), 12);

    // `stats` and `check` detect Pascal VOC automatically and with explicit `--format voc`.
    let (code, st, stderr) = folo_one(&["dataset", "stats", s(&voc), "--json"]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(st["format"], "voc");
    let (code, st_explicit, stderr) = folo_one(&["dataset", "stats", s(&voc), "--format", "voc", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(st_explicit["format"], "voc");
    assert_eq!(st["splits"], st_explicit["splits"]);

    let (code, chk, stderr) = folo_one(&["dataset", "check", s(&voc), "--json"]);
    assert_eq!(code, 0, "{stderr}: {chk}");
    assert_eq!(chk["format"], "voc");
    assert_eq!(chk["clean"], true);

    // Split a Pascal VOC dataset keeping VOC by default, and convert VOC -> YOLO.
    let voc_split = dir.join("voc_split");
    let (code, sp, stderr) = folo_one(&[
        "dataset",
        "split",
        s(&voc),
        "--out",
        s(&voc_split),
        "--val",
        "0.25",
        "--seed",
        "5",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!((sp["from"].as_str(), sp["to"].as_str()), (Some("voc"), Some("voc")));
    assert_eq!((sp["splits"]["train"].as_u64(), sp["splits"]["val"].as_u64()), (Some(9), Some(3)));

    let back_yolo = dir.join("back_yolo");
    let (code, back, stderr) =
        folo_one(&["dataset", "convert", s(&voc), s(&back_yolo), "--from", "voc", "--to", "yolo", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!((back["from"].as_str(), back["to"].as_str()), (Some("voc"), Some("yolo")));
}

#[test]
fn dataset_check_surfaces_duplicate_and_train_val_split_leakage() {
    let dir = TestDir::new("cli-dataset-leakage");
    let ds = dir.join("leaked");
    let (code, _, stderr) =
        folo(&["dataset", "synth", s(&ds), "--images", "10", "--imgsz", "64", "--seed", "2", "--json"]);
    assert_eq!(code, 0, "{stderr}");

    // Copy one train image + label verbatim into val to create cross-split leakage.
    let train_img = std::fs::read_dir(ds.join("images/train")).unwrap().next().unwrap().unwrap().path();
    let stem = train_img.file_stem().unwrap().to_str().unwrap();
    let ext = train_img.extension().unwrap().to_str().unwrap();
    let train_lbl = ds.join(format!("labels/train/{stem}.txt"));
    std::fs::copy(&train_img, ds.join(format!("images/val/leaked_copy.{ext}"))).unwrap();
    if train_lbl.is_file() {
        std::fs::copy(&train_lbl, ds.join("labels/val/leaked_copy.txt")).unwrap();
    }

    // JSON output reports `split_leakage >= 1` and a `duplicate_image` warning whose message flags split leakage.
    let (code, lines, stderr) = folo(&["dataset", "check", s(&ds), "--json"]);
    assert_eq!(code, 1, "leaked dataset must exit 1: {stderr}");
    let chk = &lines[0];
    assert_eq!(chk["clean"], false);
    assert!(chk["counts"]["split_leakage"].as_u64().unwrap_or(0) >= 1, "expected split_leakage >= 1: {chk}");
    assert!(
        chk["counts"]["by_kind"]["duplicate_image"].as_u64().unwrap_or(0) >= 1,
        "expected duplicate_image >= 1: {chk}"
    );
    let issues = chk["issues"].as_array().unwrap();
    assert!(
        issues.iter().any(|i| {
            i["kind"] == "duplicate_image"
                && i["message"].as_str().is_some_and(|m| m.starts_with("split leakage ("))
        }),
        "expected split leakage issue in {issues:?}"
    );
    assert_eq!(lines.last().unwrap()["error"]["code"], "dataset_issues");

    // Human output includes an explicit cross-split duplicate leakage summary.
    let (code, stdout, _stderr) = folo_human(&["dataset", "check", s(&ds)]);
    assert_eq!(code, 1);
    assert!(stdout.contains("cross-split duplicate leakage"), "stdout:\n{stdout}");
    assert!(stdout.contains("remove duplicate copies"), "stdout:\n{stdout}");
}

#[test]
fn dataset_split_val_plus_test_one_has_zero_train_and_voc_unreadable_split_fails() {
    let dir = TestDir::new("cli-dataset-split-edge");
    let src = dir.join("src");
    let (code, _, stderr) =
        folo(&["dataset", "synth", s(&src), "--images", "10", "--imgsz", "64", "--seed", "1", "--json"]);
    assert_eq!(code, 0, "{stderr}");

    // `--val 0.7 --test 0.3` sums to 1.0 in decimal (despite IEEE-754 remainder); train must stay 0.
    let out_vt = dir.join("val_test_only");
    let (code, vt, stderr) = folo_one(&[
        "dataset",
        "split",
        s(&src),
        "--out",
        s(&out_vt),
        "--val",
        "0.7",
        "--test",
        "0.3",
        "--to",
        "coco",
        "--seed",
        "1",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(vt["splits"]["train"].as_u64(), Some(0), "{vt}");
    assert_eq!(vt["splits"]["val"].as_u64(), Some(7), "{vt}");
    assert_eq!(vt["splits"]["test"].as_u64(), Some(3), "{vt}");

    // An unreadable VOC `ImageSets/Main/val.txt` (invalid UTF-8) fails both `convert` and `split` with
    // `invalid_dataset` before writing output instead of silently dropping the entire val split.
    let voc = dir.join("voc");
    let (code, _, stderr) = folo_one(&["dataset", "convert", s(&src), s(&voc), "--to", "voc", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    std::fs::write(voc.join("ImageSets/Main/val.txt"), [0xff, 0xfe, 0xfd]).unwrap();

    let conv_out = dir.join("voc_conv_fail");
    let (code, err, _) = folo_one(&["dataset", "convert", s(&voc), s(&conv_out), "--to", "yolo", "--json"]);
    assert_eq!((code, err["error"]["code"].as_str()), (1, Some("invalid_dataset")), "{err}");
    assert!(!conv_out.exists(), "failed convert must not create output dir");

    let split_out = dir.join("voc_split_fail");
    let (code, err, _) =
        folo_one(&["dataset", "split", s(&voc), "--out", s(&split_out), "--val", "0.2", "--json"]);
    assert_eq!((code, err["error"]["code"].as_str()), (1, Some("invalid_dataset")), "{err}");
    assert!(!split_out.exists(), "failed split must not create output dir");
}

#[test]
fn dataset_split_validates_ratios_and_output_folder() {
    let dir = TestDir::new("cli-dataset-split-errors");
    let src = dir.join("src");
    let (code, _, stderr) =
        folo(&["dataset", "synth", s(&src), "--images", "8", "--imgsz", "64", "--seed", "1", "--json"]);
    assert_eq!(code, 0, "{stderr}");

    // Bad ratios fail upfront with `invalid_config` (exit 2) before opening any dataset.
    for args in [
        &["dataset", "split", "nonexistent", "--out", s(&dir.join("o1")), "--val", "1.5", "--json"][..],
        &[
            "dataset",
            "split",
            "nonexistent",
            "--out",
            s(&dir.join("o2")),
            "--val",
            "0.6",
            "--test",
            "0.5",
            "--json",
        ],
        &["dataset", "split", "nonexistent", "--out", s(&dir.join("o3")), "--val=-0.1", "--json"],
    ] {
        let (code, v, _) = folo_one(args);
        assert_eq!((code, v["error"]["code"].as_str()), (2, Some("invalid_config")), "{args:?}: {v}");
    }

    // Output inside input is refused as `usage` (exit 2).
    let (code, v, _) =
        folo_one(&["dataset", "split", s(&src), "--out", s(&src.join("nested")), "--val", "0.25", "--json"]);
    assert_eq!((code, v["error"]["code"].as_str()), (2, Some("usage")), "{v}");

    // Non-empty output is refused as `output_exists` (exit 1).
    let non_empty = dir.join("busy");
    std::fs::create_dir_all(&non_empty).unwrap();
    std::fs::write(non_empty.join("keep.txt"), b"x").unwrap();
    let (code, v, _) =
        folo_one(&["dataset", "split", s(&src), "--out", s(&non_empty), "--val", "0.25", "--json"]);
    assert_eq!((code, v["error"]["code"].as_str()), (1, Some("output_exists")), "{v}");
}
