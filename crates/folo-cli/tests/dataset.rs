//! `folo dataset synth | stats | check | convert` end to end, on small synthetic datasets in temp folders.

use std::path::{Path, PathBuf};
use std::process::Command;

use folo_test_support::TestDir;
use serde_json::Value;

fn folo(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_folo")).args(args).output().unwrap();
    (
        out.status.code().unwrap(),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

fn json_lines(stdout: &str) -> Vec<Value> {
    stdout.lines().map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("not JSON: {l}: {e}"))).collect()
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// Writes a 40-image synthetic dataset and returns its path.
fn synth(t: &TestDir, name: &str, extra: &[&str]) -> PathBuf {
    let dir = t.join(name);
    let mut args =
        vec!["dataset", "synth", s(&dir), "--images", "40", "--imgsz", "96", "--seed", "4", "--json"];
    args.extend_from_slice(extra);
    let (code, stdout, stderr) = folo(&args);
    assert_eq!(code, 0, "{stderr}");
    let lines = json_lines(&stdout);
    let done = lines.last().unwrap();
    assert_eq!(done["event"], "done");
    assert_eq!(done["images"], 40);
    let last_progress = &lines[lines.len() - 2];
    assert_eq!(last_progress["event"], "progress");
    assert_eq!(last_progress["done"], 40);
    assert_eq!(last_progress["total"], 40);
    dir
}

#[test]
fn synth_then_stats_and_a_clean_check() {
    let t = TestDir::new("cli-synth");
    let dir =
        synth(&t, "shapes", &["--classes", "circle,box,star", "--val-frac", "0.25", "--image-format", "png"]);
    assert!(dir.join("images/train/000000.png").is_file());

    let (code, stdout, _) = folo(&["dataset", "stats", s(&dir), "--json"]);
    assert_eq!(code, 0);
    let v = &json_lines(&stdout)[0];
    assert_eq!(v["format"], "yolo");
    assert_eq!(v["classes"], serde_json::json!(["circle", "box", "star"]));
    assert_eq!(v["splits"], serde_json::json!({ "train": 30, "val": 10, "test": 0 }));
    let st = &v["stats"];
    for k in
        ["boxes", "empty_images", "broken_labels", "class_counts", "box_size_hist", "boxes_per_image_hist"]
    {
        assert!(st.get(k).is_some(), "stats.{k}");
    }
    let counts: u64 = st["class_counts"].as_array().unwrap().iter().map(|c| c.as_u64().unwrap()).sum();
    assert_eq!(counts, st["boxes"].as_u64().unwrap());
    assert_eq!(st["box_size_hist"]["edges"].as_array().unwrap().len(), 8);

    let (code, stdout, _) = folo(&["dataset", "stats", s(&dir)]);
    assert_eq!(code, 0);
    assert!(
        stdout.contains("box size") && stdout.contains("boxes per image") && stdout.contains('█'),
        "{stdout}"
    );

    let (code, stdout, stderr) = folo(&["dataset", "check", s(&dir), "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let lines = json_lines(&stdout);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0]["clean"], true);
    assert_eq!(lines[0]["images"], 40);
    assert_eq!(lines[0]["counts"]["error"], 0);

    // The output folder now holds a dataset: a second synth into it is refused.
    let (code, stdout, _) = folo(&["dataset", "synth", s(&dir), "--images", "2", "--json"]);
    assert_eq!(code, 1);
    assert_eq!(json_lines(&stdout).last().unwrap()["error"]["code"], "output_exists");
}

#[test]
fn check_reports_each_issue_and_exits_1() {
    let t = TestDir::new("cli-check");
    let dir = synth(&t, "broken", &[]);
    std::fs::write(dir.join("labels/train/000001.txt"), "0 0.5 0.5 0.2 0.2\n9 0.5 0.5 0.1 0.1\n0 0.5\n")
        .unwrap();
    std::fs::write(dir.join("images/train/000002.jpg"), b"not a jpeg at all").unwrap();
    std::fs::copy(dir.join("images/train/000003.jpg"), dir.join("images/train/000003b.jpg")).unwrap();
    // Cut mid-scan (the dogfood's `head -c 2000`): it still decodes, with grey rows.
    let jpg = std::fs::read(dir.join("images/train/000004.jpg")).unwrap();
    std::fs::write(dir.join("images/train/000004.jpg"), &jpg[..jpg.len() * 2 / 3]).unwrap();

    let (code, stdout, stderr) = folo(&["dataset", "check", s(&dir), "--json", "--fix-dry-run"]);
    assert_eq!(code, 1, "{stderr}");
    let lines = json_lines(&stdout);
    assert_eq!(lines.len(), 2, "report, then the error envelope");
    assert_eq!(lines[1]["error"]["code"], "dataset_issues");
    let r = &lines[0];
    assert_eq!(r["clean"], false);
    let issues = r["issues"].as_array().unwrap();
    let find = |kind: &str| {
        issues.iter().find(|i| i["kind"] == kind).unwrap_or_else(|| panic!("no {kind} in {issues:?}"))
    };
    let overflow = find("class_overflow");
    assert_eq!(overflow["file"], "labels/train/000001.txt");
    assert_eq!(overflow["line"], 2);
    assert_eq!(overflow["severity"], "error");
    assert!(overflow["fix"].as_str().unwrap().contains("line 2"));
    assert_eq!(find("broken_label")["line"], 3);
    let corrupt = |file: &str| {
        issues
            .iter()
            .find(|i| i["kind"] == "corrupt_image" && i["file"] == file)
            .unwrap_or_else(|| panic!("no corrupt_image for {file} in {issues:?}"))
    };
    let undecodable = corrupt("images/train/000002.jpg")["message"].as_str().unwrap();
    assert!(undecodable.starts_with("cannot decode: ") && !undecodable.contains(" at :"), "{undecodable}");
    assert_eq!(undecodable, undecodable.trim_end());
    assert!(corrupt("images/train/000004.jpg")["message"].as_str().unwrap().contains("truncated JPEG"));
    let dup = find("duplicate_image");
    assert_eq!(dup["file"], "images/train/000003b.jpg");
    assert_eq!(dup["message"], "same bytes as images/train/000003.jpg");
    assert!(dup["line"].is_null());
    // Each issue parses as the API's issue object.
    for i in issues {
        let _: folo_server::api::DatasetIssue = serde_json::from_value(i.clone()).unwrap();
    }
    for i in issues {
        assert!(i["kind"].is_string() && i["file"].is_string() && i["message"].is_string(), "{i}");
    }
    assert_eq!(r["counts"]["by_kind"]["class_overflow"], 1);
    assert!(!r["root"].as_str().unwrap().contains('\\'), "JSON paths use /: {}", r["root"]);

    // --min-severity error lists only errors; the counts still cover everything.
    let (code, stdout, _) = folo(&["dataset", "check", s(&dir), "--json", "--min-severity", "error"]);
    assert_eq!(code, 1);
    let only = &json_lines(&stdout)[0];
    let listed = only["issues"].as_array().unwrap();
    assert!(!listed.is_empty() && listed.iter().all(|i| i["severity"] == "error"), "{listed:?}");
    assert_eq!(only["counts"], r["counts"]);

    // Without --fix-dry-run there are no fixes; human output names the file and line.
    let (code, stdout, _) = folo(&["dataset", "check", s(&dir), "--json"]);
    assert_eq!(code, 1);
    assert!(json_lines(&stdout)[0]["issues"].as_array().unwrap().iter().all(|i| i.get("fix").is_none()));
    let (code, stdout, stderr) = folo(&["dataset", "check", s(&dir), "--fix-dry-run"]);
    assert_eq!(code, 1);
    assert!(
        stdout.contains("labels/train/000001.txt:2") && stdout.contains("would delete line 2"),
        "{stdout}"
    );
    assert!(stderr.contains("error(s)"), "{stderr}");
}

#[test]
fn convert_round_trips_through_coco() {
    let t = TestDir::new("cli-convert");
    let dir = synth(&t, "yolo", &["--classes", "ring,hexagon"]);
    let coco = t.join("coco");
    let (code, stdout, stderr) = folo(&["dataset", "convert", s(&dir), s(&coco), "--to", "coco", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let v = &json_lines(&stdout)[0];
    assert_eq!((v["from"].as_str(), v["to"].as_str()), (Some("yolo"), Some("coco")));
    assert!(coco.join("annotations/instances_train.json").is_file());

    let stats = |p: &Path| {
        let (code, stdout, _) = folo(&["dataset", "stats", s(p), "--json"]);
        assert_eq!(code, 0);
        json_lines(&stdout).remove(0)
    };
    let (a, b) = (stats(&dir), stats(&coco));
    assert_eq!(b["format"], "coco");
    assert_eq!(a["classes"], b["classes"]);
    assert_eq!(a["splits"], b["splits"]);
    assert_eq!(a["stats"]["class_counts"], b["stats"]["class_counts"]);
    assert_eq!(a["stats"]["boxes_per_image_hist"], b["stats"]["boxes_per_image_hist"]);

    let back = t.join("back");
    let (code, _, stderr) = folo(&["dataset", "convert", s(&coco), s(&back), "--to", "yolo"]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(stats(&back)["stats"]["class_counts"], a["stats"]["class_counts"]);
    let (code, stdout, _) = folo(&["dataset", "convert", s(&dir), s(&back), "--to", "coco", "--json"]);
    assert_eq!(code, 1);
    assert_eq!(json_lines(&stdout)[0]["error"]["code"], "output_exists");
}

#[test]
fn dataset_errors_use_stable_codes() {
    let t = TestDir::new("cli-errors");
    let missing = t.join("nope");
    for cmd in ["stats", "check"] {
        let (code, stdout, _) = folo(&["dataset", cmd, s(&missing), "--json"]);
        assert_eq!(code, 3, "{cmd}");
        assert_eq!(json_lines(&stdout)[0]["error"]["code"], "path_not_found");
    }
    let empty = t.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let (code, stdout, _) = folo(&["dataset", "stats", s(&empty), "--json"]);
    assert_eq!(code, 1);
    assert_eq!(json_lines(&stdout)[0]["error"]["code"], "unknown_format");
    let out = t.join("o");
    for bad in [&["--classes", "a,b,c,d,e,f,g,h,i"][..], &["--classes", "a,a"], &["--images", "0"]] {
        let mut args = vec!["dataset", "synth", s(&out), "--json"];
        args.extend_from_slice(bad);
        let (code, stdout, _) = folo(&args);
        assert_eq!(code, 2, "{bad:?}");
        assert_eq!(json_lines(&stdout)[0]["error"]["code"], "usage");
    }
    assert!(!out.exists(), "nothing is written on a usage error");
    let (code, _, _) = folo(&["dataset", "convert", s(&empty), s(&out)]);
    assert_eq!(code, 2, "--to is required");
}
