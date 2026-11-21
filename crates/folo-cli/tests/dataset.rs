//! `folo dataset synth | stats | check | convert` end to end, on small synthetic datasets in temp folders.

use std::path::{Path, PathBuf};
use std::process::Command;

use folo_test_support::TestDir;
use serde_json::Value;

fn folo(args: &[&str]) -> (i32, String, String) {
    let out = folo_cmd().args(args).output().unwrap();
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

/// A synth stopped part way (issue #316) leaves its folder marked unfinished: `stats` and `check` refuse it, the
/// same command run again says what to do and deletes nothing, and run again after deleting the folder it writes
/// the whole dataset.
#[test]
fn synth_after_a_stopped_one() {
    let t = TestDir::new("cli-synth-again");
    let dir = t.join("NEW");
    // What a synth stopped after its first image leaves.
    for d in ["images/train", "labels/train"] {
        std::fs::create_dir_all(dir.join(d)).unwrap();
    }
    std::fs::write(dir.join(folo_data::unfinished::MARKER), "").unwrap();
    std::fs::write(dir.join("images/train/000000.jpg"), "half").unwrap();
    let hint = |err: &serde_json::Value| {
        err["hint"].as_str().unwrap().contains("delete the folder and run the command")
    };
    for cmd in ["stats", "check"] {
        let (code, stdout, _) = folo(&["dataset", cmd, s(&dir), "--json"]);
        let err = &json_lines(&stdout).pop().unwrap()["error"];
        assert_eq!((code, err["code"].as_str()), (1, Some("dataset_unfinished")), "{err}");
        assert!(hint(err), "{err}");
    }
    let synth = ["dataset", "synth", s(&dir), "--images", "8", "--imgsz", "32", "--json"];
    let (code, stdout, _) = folo(&synth);
    let err = &json_lines(&stdout).pop().unwrap()["error"];
    assert_eq!((code, err["code"].as_str()), (1, Some("output_exists")), "{err}");
    assert!(hint(err), "{err}");
    assert!(
        dir.join("images/train/000000.jpg").is_file() && dir.join(folo_data::unfinished::MARKER).is_file()
    );

    std::fs::remove_dir_all(&dir).unwrap();
    let (code, _, stderr) = folo(&synth);
    assert_eq!(code, 0, "{stderr}");
    assert!(!dir.join(folo_data::unfinished::MARKER).exists());
    let (code, stdout, _) = folo(&["dataset", "stats", s(&dir), "--json"]);
    assert_eq!(code, 0);
    assert_eq!(json_lines(&stdout)[0]["splits"], serde_json::json!({ "train": 6, "val": 2, "test": 0 }));
}

/// Output folders `main` wrote into: the current folder (`synth .`) and a 250-character name.
#[test]
fn synth_into_the_current_folder_and_a_long_name() {
    let t = TestDir::new("cli-synth-here");
    let here = t.join("here");
    std::fs::create_dir(&here).unwrap();
    let out = folo_cmd()
        .args(["dataset", "synth", ".", "--images", "4", "--imgsz", "32", "--json"])
        .current_dir(&here)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stdout));
    assert!(here.join("data.yaml").is_file() && !here.join(folo_data::unfinished::MARKER).exists());

    let long = t.join("n".repeat(250));
    let (code, stdout, _) = folo(&["dataset", "synth", s(&long), "--images", "4", "--imgsz", "32", "--json"]);
    assert_eq!(code, 0, "{stdout}");
    assert!(long.join("data.yaml").is_file());
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

#[test]
fn check_finds_an_orphan_label_from_a_relative_root_and_absolute_list() {
    let t = TestDir::new("cli-check-relative");
    let dir = t.join("ds");
    let (code, _, stderr) = folo(&["dataset", "synth", s(&dir), "--images", "4", "--imgsz", "64", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    // The train split is a list of absolute image paths, and the dataset is named relative to the current folder.
    std::fs::create_dir(dir.join("set")).unwrap();
    for sub in ["images", "labels"] {
        std::fs::rename(dir.join(sub), dir.join("set").join(sub)).unwrap();
    }
    let list: String = std::fs::read_dir(dir.join("set/images/train"))
        .unwrap()
        .map(|e| format!("{}\n", e.unwrap().path().display()))
        .collect();
    std::fs::write(dir.join("train.txt"), list).unwrap();
    let yaml = std::fs::read_to_string(dir.join("data.yaml")).unwrap();
    let yaml =
        yaml.replace("train: images/train", "train: train.txt").replace("val: images/", "val: set/images/");
    std::fs::write(dir.join("data.yaml"), yaml).unwrap();
    std::fs::write(dir.join("set/labels/train/gone.txt"), "0 0.5 0.5 0.2 0.2\n").unwrap();
    let out = folo_cmd().args(["dataset", "check", "ds", "--json"]).current_dir(t.path()).output().unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let issues = json_lines(&stdout)[0]["issues"].as_array().unwrap().clone();
    let orphans: Vec<_> = issues.iter().filter(|i| i["kind"] == "missing_image").collect();
    assert_eq!(orphans.len(), 1, "{issues:?}");
    assert!(orphans[0]["file"].as_str().unwrap().ends_with("gone.txt"), "{issues:?}");
}

/// Labels beside the images, below a folder called `images`, with an image list naming them by absolute path: the
/// dataset reads its box by absolute path and as a bare `data.yaml` from inside it.
#[test]
fn labels_beside_listed_images_read_the_same_from_any_working_folder() {
    let t = TestDir::new("cli-labels-beside");
    let src = synth(&t, "src", &["--val-frac", "0.25", "--image-format", "png"]);
    let dir = t.join("images/birds");
    std::fs::create_dir_all(dir.join("train")).unwrap();
    let image = dir.join("train/a.png");
    std::fs::copy(src.join("images/train/000000.png"), &image).unwrap();
    std::fs::write(dir.join("train/a.txt"), "0 0.5 0.5 0.2 0.2\n").unwrap();
    std::fs::write(dir.join("train.txt"), format!("{}\n", s(&image))).unwrap();
    std::fs::write(dir.join("data.yaml"), "train: train.txt\nnames: [bird]\n").unwrap();

    let (code, stdout, stderr) = folo(&["dataset", "stats", s(&dir), "--json"]);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(json_lines(&stdout)[0]["stats"]["boxes"], 1, "by absolute path");

    let out =
        folo_cmd().args(["dataset", "stats", "data.yaml", "--json"]).current_dir(&dir).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(json_lines(&stdout)[0]["stats"]["boxes"], 1, "as data.yaml from inside");
}

/// A `folo` command with the on-disk caches off (`FOLO_NO_CACHE`), so tests never write into the user's home.
fn folo_cmd() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_folo"));
    c.env("FOLO_NO_CACHE", "1");
    c
}
