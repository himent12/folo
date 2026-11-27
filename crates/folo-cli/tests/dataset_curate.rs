//! Integration tests for `folo dataset merge`, `folo dataset filter`, and `folo dataset check --perceptual`
//! (64-bit `dHash` near-duplicate and cross-split leakage detection).

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
                panic!("invalid json on stdout ({e})\nargs: {args:?}\nline: {l}\nstderr:\n{stderr}")
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

fn s(p: &Path) -> &str {
    p.to_str().expect("utf-8 test path")
}

/// Builds a textured 64x64 RGB image (distinct gradients + block pattern so its 64-bit dHash is rich and stable
/// across JPEG recompression).
fn textured_image(seed: u8) -> image::RgbImage {
    image::RgbImage::from_fn(64, 64, |x, y| {
        let r = ((x * 4) as u8).wrapping_add(seed.wrapping_mul(17));
        let g = ((y * 4) as u8).wrapping_add(((x / 8) as u8).wrapping_mul(29));
        let b = (((x ^ y) * 3) as u8).wrapping_add(seed.wrapping_mul(43));
        image::Rgb([r, g, b])
    })
}

#[test]
fn dataset_merge_unifies_classes_remaps_resolves_collisions_and_splits() {
    let dir = TestDir::new("cli-dataset-merge");
    let ds_a = dir.join("ds_a");
    let ds_b = dir.join("ds_b");
    let (code, _, stderr) = folo(&[
        "dataset",
        "synth",
        s(&ds_a),
        "--images",
        "6",
        "--imgsz",
        "64",
        "--classes",
        "circle,square",
        "--seed",
        "1",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}");
    let (code, _, stderr) = folo(&[
        "dataset",
        "synth",
        s(&ds_b),
        "--images",
        "6",
        "--imgsz",
        "64",
        "--classes",
        "Circle,star",
        "--seed",
        "2",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}");

    let merged = dir.join("merged");
    let (code, rec, stderr) = folo_one(&[
        "dataset",
        "merge",
        s(&ds_a),
        s(&ds_b),
        "--out",
        s(&merged),
        "--normalize",
        "case-and-whitespace",
        "--remap",
        "star=triangle",
        "--splits",
        "stratified",
        "--val",
        "0.25",
        "--seed",
        "7",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}: {rec}");
    assert_eq!(rec["format"], "yolo");
    let classes: Vec<&str> = rec["classes"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    // `circle` and `Circle` unified via case-and-whitespace; `star` remapped to `triangle`.
    assert_eq!(classes, vec!["circle", "square", "triangle"]);
    let train = rec["splits"]["train"].as_u64().unwrap();
    let val = rec["splits"]["val"].as_u64().unwrap();
    assert_eq!(train + val, 12);
    assert!(val >= 2, "{rec}");
    // Both synthetic datasets share filenames (`000000.jpg`, ...), so collisions are renamed automatically.
    assert!(rec["renamed_collisions"].as_u64().unwrap() > 0, "{rec}");

    let (code, chk, stderr) = folo_one(&["dataset", "check", s(&merged), "--json"]);
    assert_eq!(code, 0, "{stderr}: {chk}");
    assert_eq!(chk["clean"], true);

    // Unmatched `--remap` and `--drop-class` rules are surfaced in JSON and human warnings.
    let merged_unmatched = dir.join("merged_unmatched");
    let (code, rec, stderr) = folo_one(&[
        "dataset",
        "merge",
        s(&ds_a),
        s(&ds_b),
        "--out",
        s(&merged_unmatched),
        "--remap",
        "missing=round",
        "--drop-class",
        "ghost",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}: {rec}");
    assert_eq!(rec["unmatched_remaps"], serde_json::json!(["missing"]));
    assert_eq!(rec["unmatched_drops"], serde_json::json!(["ghost"]));
    assert!(stderr.contains("unmatched --remap"), "{stderr}");
    assert!(stderr.contains("unmatched --drop-class"), "{stderr}");

    // Upfront validation: `--val` without `--splits stratified` and normalized remap conflicts fail with
    // exit 2 (`invalid_config`) before opening any input dataset.
    let (code, err, _) =
        folo_one(&["dataset", "merge", s(&ds_a), "--out", s(&dir.join("bad_out")), "--val", "0.2", "--json"]);
    assert_eq!(code, 2);
    assert_eq!(err["error"]["code"], "invalid_config");

    let (code, err, _) = folo_one(&[
        "dataset",
        "merge",
        "nonexistent_ds",
        "--out",
        s(&dir.join("bad_out")),
        "--normalize",
        "case-and-whitespace",
        "--remap",
        "car=auto",
        "--remap",
        "Car=truck",
        "--json",
    ]);
    assert_eq!(code, 2);
    assert_eq!(err["error"]["code"], "invalid_config");
}

#[test]
fn dataset_filter_filters_classes_splits_and_converts_format() {
    let dir = TestDir::new("cli-dataset-filter");
    let src = dir.join("src");
    let (code, _, stderr) = folo(&[
        "dataset",
        "synth",
        s(&src),
        "--images",
        "12",
        "--imgsz",
        "64",
        "--classes",
        "circle,square,triangle",
        "--seed",
        "3",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}");

    let filtered = dir.join("filtered_coco");
    let (code, rec, stderr) = folo_one(&[
        "dataset",
        "filter",
        s(&src),
        "--out",
        s(&filtered),
        "--include-class",
        "circle",
        "--drop-empty-images",
        "--to",
        "coco",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}: {rec}");
    assert_eq!(rec["from"], "yolo");
    assert_eq!(rec["to"], "coco");
    assert_eq!(rec["classes"], serde_json::json!(["circle"]));
    assert!(rec["boxes"].as_u64().unwrap() > 0, "{rec}");

    let (code, chk, stderr) = folo_one(&["dataset", "check", s(&filtered), "--json"]);
    assert_eq!(code, 0, "{stderr}: {chk}");
    assert_eq!(chk["clean"], true);

    // Upfront validation: `--min-box-area > 1.0` fails with exit 2 (`invalid_config`).
    let (code, err, _) = folo_one(&[
        "dataset",
        "filter",
        s(&src),
        "--out",
        s(&dir.join("bad_out")),
        "--min-box-area",
        "1.5",
        "--json",
    ]);
    assert_eq!(code, 2);
    assert_eq!(err["error"]["code"], "invalid_config");

    // Unmatched class filter names are surfaced in JSON (and human warnings) so typos are caught.
    let typo_out = dir.join("typo_out");
    let (code, rec, stderr) =
        folo_one(&["dataset", "filter", s(&src), "--out", s(&typo_out), "--exclude-class", "quda", "--json"]);
    assert_eq!(code, 0, "{stderr}: {rec}");
    assert_eq!(rec["unmatched_classes"], serde_json::json!(["quda"]));

    // Filtering out every image reports that no images matched the filter (not "no readable images").
    let empty_out = dir.join("empty_out");
    let (code, err, _) = folo_one(&[
        "dataset",
        "filter",
        s(&src),
        "--out",
        s(&empty_out),
        "--min-image-width",
        "65",
        "--json",
    ]);
    assert_eq!(code, 1);
    assert_eq!(err["error"]["code"], "invalid_dataset");
    let msg = err["error"]["message"].as_str().unwrap();
    assert!(msg.contains("matched the filter"), "{err}");
    assert!(!msg.contains("no readable images"), "{err}");
}

#[test]
fn dataset_check_perceptual_detects_recompressed_split_leakage() {
    let dir = TestDir::new("cli-dataset-perceptual");
    for sub in ["images/train", "images/val", "labels/train", "labels/val"] {
        std::fs::create_dir_all(dir.join(sub)).unwrap();
    }
    std::fs::write(dir.join("data.yaml"), "names:\n  0: obj\ntrain: images/train\nval: images/val\n")
        .unwrap();

    // Save the same textured scene as PNG in train and recompressed JPEG in val (different bytes/SHA-256!).
    let img = textured_image(7);
    img.save(dir.join("images/train/scene_a.png")).unwrap();
    let mut jpeg_buf = std::io::Cursor::new(Vec::new());
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg_buf, 75);
    enc.encode_image(&img).unwrap();
    std::fs::write(dir.join("images/val/scene_a_recomp.jpg"), jpeg_buf.into_inner()).unwrap();

    // Add one distinct unrelated image in train.
    textured_image(200).save(dir.join("images/train/other.png")).unwrap();

    for lbl in ["labels/train/scene_a.txt", "labels/val/scene_a_recomp.txt", "labels/train/other.txt"] {
        std::fs::write(dir.join(lbl), "0 0.5 0.5 0.3 0.3\n").unwrap();
    }

    // Exact-byte check sees no split leakage because PNG and JPEG have different bytes.
    let (code, plain, stderr) = folo_one(&["dataset", "check", s(&dir), "--json"]);
    assert_eq!(code, 0, "{stderr}: {plain}");
    assert_eq!(plain["clean"], true);
    assert_eq!(plain["counts"]["split_leakage"], 0);
    assert!(plain.get("perceptual").is_none());

    // Perceptual dHash check catches the recompressed cross-split near-duplicate leakage!
    let (code, lines, _) =
        folo(&["dataset", "check", s(&dir), "--perceptual", "--dhash-threshold", "6", "--json"]);
    assert_eq!(code, 1, "perceptual split leakage should exit 1: {lines:?}");
    let report = &lines[0];
    assert_eq!(report["clean"], false);
    assert!(report["counts"]["split_leakage"].as_u64().unwrap() >= 1, "{report}");
    assert_eq!(report["perceptual"]["threshold"], 6);
    assert!(report["perceptual"]["near_duplicate_leakage"].as_u64().unwrap() >= 1, "{report}");
    assert_eq!(lines[1]["error"]["code"], "dataset_issues");

    // Human summary distinguishes perceptual near-duplicate leakage from exact-byte duplicates.
    let human = Command::new(env!("CARGO_BIN_EXE_folo"))
        .args(["dataset", "check", s(&dir), "--perceptual", "--dhash-threshold", "6"])
        .output()
        .expect("spawn folo");
    assert_eq!(human.status.code(), Some(1));
    let human_out = String::from_utf8(human.stdout).unwrap();
    assert!(human_out.contains("perceptually near-duplicate images across splits"), "{human_out}");
    assert!(!human_out.contains("identical image content across splits"), "{human_out}");

    // `--perceptual` conflicts with `--headers-only` (exit 2, `usage`).
    let (code, err, _) = folo_one(&["dataset", "check", s(&dir), "--perceptual", "--headers-only", "--json"]);
    assert_eq!(code, 2);
    assert_eq!(err["error"]["code"], "usage");
}

#[test]
fn dataset_export_coco_modes_splits_and_errors() {
    let dir = TestDir::new("cli-dataset-export-coco");
    let src = dir.join("src");
    let (code, _, stderr) = folo(&[
        "dataset",
        "synth",
        s(&src),
        "--images",
        "10",
        "--classes",
        "circle,square",
        "--imgsz",
        "64",
        "--seed",
        "13",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}");

    // 1. Copy mode (default) with pretty JSON and description; round-trips through `dataset check` and `stats`.
    let out_copy = dir.join("coco_copy");
    let (code, rec, stderr) = folo_one(&[
        "dataset",
        "export",
        s(&src),
        "--out",
        s(&out_copy),
        "--format",
        "coco",
        "--pretty",
        "--description",
        "sprint 14 export",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}: {rec}");
    assert_eq!(rec["from"], "yolo");
    assert_eq!(rec["format"], "coco");
    assert_eq!(rec["images"], "copy");
    assert_eq!(rec["pretty"], true);
    assert_eq!(rec["categories"], 2);
    assert_eq!(
        rec["splits"]["train"].as_u64().unwrap() + rec["splits"]["val"].as_u64().unwrap(),
        10,
        "{rec}"
    );
    assert!(rec["boxes"].as_u64().unwrap() > 0, "{rec}");
    assert_eq!(rec["exported_splits"].as_array().unwrap().len(), 2, "{rec}");
    let train_json_path = out_copy.join("annotations/instances_train.json");
    let train_raw = std::fs::read_to_string(&train_json_path).unwrap();
    assert!(train_raw.contains('\n'), "pretty JSON should contain newlines");
    let train_json: serde_json::Value = serde_json::from_str(&train_raw).unwrap();
    assert_eq!(train_json["info"]["description"], "sprint 14 export");
    let (code, chk, stderr) = folo_one(&["dataset", "check", s(&out_copy), "--json"]);
    assert_eq!(code, 0, "{stderr}: {chk}");
    assert_eq!(chk["clean"], true);
    assert_eq!(chk["format"], "coco");

    // 2. Link mode with mixed `--split val --split test` (empty `test` split is skipped, `val` exported).
    let out_link = dir.join("coco_link");
    let (code, rec, stderr) = folo_one(&[
        "dataset",
        "export",
        s(&src),
        "--out",
        s(&out_link),
        "--images",
        "link",
        "--split",
        "val",
        "--split",
        "test",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}: {rec}");
    assert_eq!(rec["images"], "link");
    let splits = rec["exported_splits"].as_array().unwrap();
    assert_eq!(splits.len(), 1, "{rec}");
    assert_eq!(splits[0]["split"], "val");
    assert!(
        splits[0]["linked_images"].as_u64().unwrap() + splits[0]["copied_images"].as_u64().unwrap()
            == splits[0]["images"].as_u64().unwrap(),
        "{rec}"
    );
    assert!(!out_link.join("annotations/instances_train.json").exists());
    assert!(out_link.join("annotations/instances_val.json").is_file());

    // 3. Reference mode writes annotations only without creating `images/`.
    let out_ref = dir.join("coco_ref");
    let (code, rec, stderr) =
        folo_one(&["dataset", "export", s(&src), "--out", s(&out_ref), "--images", "reference", "--json"]);
    assert_eq!(code, 0, "{stderr}: {rec}");
    assert_eq!(rec["images"], "reference");
    assert!(!out_ref.join("images").exists(), "reference mode must not copy images");
    assert!(rec["exported_splits"][0]["referenced_images"].as_u64().unwrap() > 0, "{rec}");

    // 4. Error cases: non-empty `--out` (`output_exists`, exit 1), `--out` inside input (`usage`, exit 2),
    // and missing requested split (`invalid_dataset`, exit 1).
    let (code, err, _) = folo_one(&["dataset", "export", s(&src), "--out", s(&out_copy), "--json"]);
    assert_eq!(code, 1);
    assert_eq!(err["error"]["code"], "output_exists");

    let (code, err, _) = folo_one(&["dataset", "export", s(&src), "--out", s(&src.join("nested")), "--json"]);
    assert_eq!(code, 2);
    assert_eq!(err["error"]["code"], "usage");

    let out_missing = dir.join("coco_missing_split");
    let (code, err, _) =
        folo_one(&["dataset", "export", s(&src), "--out", s(&out_missing), "--split", "test", "--json"]);
    assert_eq!(code, 1);
    assert_eq!(err["error"]["code"], "invalid_dataset");
    assert!(!out_missing.join(".folo-unfinished").exists(), "failed export must not leave unfinished marker");

    // A `--out` traversing through a missing intermediate directory inside `src` and back out via `..` writes
    // only to the resolved destination without creating the stray intermediate directory inside `src`.
    let stray_via_src = src.join("stray_missing").join("..").join("..").join("coco_via_dotdot");
    let (code, rec, stderr) = folo_one(&[
        "dataset",
        "export",
        s(&src),
        "--out",
        s(&stray_via_src),
        "--images",
        "reference",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}: {rec}");
    assert!(!src.join("stray_missing").exists(), "stray intermediate dir inside source must not be created");
    assert!(dir.join("coco_via_dotdot/annotations/instances_train.json").is_file());
}
