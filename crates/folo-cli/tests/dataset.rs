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
    assert!(stdout.contains("id  class"), "{stdout}");
    assert!(
        stdout.contains(" 0  circle") && stdout.contains(" 1  box") && stdout.contains(" 2  star"),
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

/// Two class names sharing a 24-character prefix stay distinguishable in human `folo dataset stats` (via class
/// index and ellipsis) while `--json` keeps the full names.
#[test]
fn stats_disambiguates_long_truncated_class_names_in_human_output() {
    let t = TestDir::new("cli-stats-long-names");
    let a = "industrial_safety_helmet_yellow";
    let b = "industrial_safety_helmet_white";
    let dir = synth(&t, "long_names", &["--classes", &format!("{a},{b}")]);

    let (code, stdout, stderr) = folo(&["dataset", "stats", s(&dir), "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let v = &json_lines(&stdout)[0];
    assert_eq!(v["classes"], serde_json::json!([a, b]), "JSON keeps full class names");

    let (code, stdout, stderr) = folo(&["dataset", "stats", s(&dir)]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains(" 0  industrial_safety_helme…"), "{stdout}");
    assert!(stdout.contains(" 1  industrial_safety_helme…"), "{stdout}");
    assert!(!stdout.contains(a) && !stdout.contains(b), "long labels are truncated in human table: {stdout}");
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
    assert_eq!(only["min_severity"], "error");
    assert_eq!(
        only["filtered_out"].as_u64().unwrap() + listed.len() as u64,
        only["counts"]["total"].as_u64().unwrap()
    );

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

/// `--min-severity` filters both human and `--json` output across `info`, `warning`/`warn`, and `error`,
/// while keeping crisp per-severity counts and exit code semantics intact.
#[test]
fn check_min_severity_filters_human_and_json_with_per_severity_counts() {
    let t = TestDir::new("cli-check-min-severity");
    let dir = synth(&t, "mixed", &[]);
    // One error (class overflow), one warning (duplicate image), and one info (empty background image).
    std::fs::write(dir.join("labels/train/000000.txt"), "9 0.5 0.5 0.2 0.2\n").unwrap();
    std::fs::copy(dir.join("images/train/000001.jpg"), dir.join("images/train/000001_dup.jpg")).unwrap();
    std::fs::write(dir.join("labels/train/000002.txt"), "").unwrap();

    // JSON default (info): includes error, warning, info; filtered_out == 0.
    let (code, stdout, _) = folo(&["dataset", "check", s(&dir), "--json"]);
    assert_eq!(code, 1);
    let r_all = &json_lines(&stdout)[0];
    let counts = &r_all["counts"];
    let errs = counts["error"].as_u64().unwrap();
    let warns = counts["warning"].as_u64().unwrap();
    let infos = counts["info"].as_u64().unwrap();
    let total = counts["total"].as_u64().unwrap();
    assert!(errs >= 1 && warns >= 1 && infos >= 1, "{counts}");
    assert_eq!(total, errs + warns + infos, "{counts}");
    assert_eq!(r_all["min_severity"], "info");
    assert_eq!(r_all["filtered_out"], 0);
    assert_eq!(r_all["issues"].as_array().unwrap().len() as u64, total);

    // JSON `--min-severity warn` (alias of `warning`): excludes info, keeps error + warning.
    let (code, stdout, _) = folo(&["dataset", "check", s(&dir), "--json", "--min-severity", "warn"]);
    assert_eq!(code, 1);
    let r_warn = &json_lines(&stdout)[0];
    let warn_issues = r_warn["issues"].as_array().unwrap();
    assert_eq!(r_warn["min_severity"], "warning");
    assert_eq!(r_warn["counts"], *counts);
    assert_eq!(r_warn["filtered_out"], infos);
    assert_eq!(warn_issues.len() as u64, errs + warns);
    assert!(warn_issues.iter().all(|i| i["severity"] != "info"), "{warn_issues:?}");

    // JSON `--min-severity error`: only error issues; counts unchanged; filtered_out == warns + infos.
    let (code, stdout, _) = folo(&["dataset", "check", s(&dir), "--json", "--min-severity", "error"]);
    assert_eq!(code, 1);
    let r_err = &json_lines(&stdout)[0];
    let err_issues = r_err["issues"].as_array().unwrap();
    assert_eq!(r_err["min_severity"], "error");
    assert_eq!(r_err["counts"], *counts);
    assert_eq!(r_err["filtered_out"], warns + infos);
    assert_eq!(err_issues.len() as u64, errs);
    assert!(err_issues.iter().all(|i| i["severity"] == "error"), "{err_issues:?}");

    // Human default: lists error + warning, hides info lines, shows all three per-severity counts.
    let (code, stdout, _) = folo(&["dataset", "check", s(&dir)]);
    assert_eq!(code, 1);
    assert!(stdout.contains("error    class_overflow"), "{stdout}");
    assert!(stdout.contains("warning  duplicate_image"), "{stdout}");
    assert!(!stdout.contains("info     empty_image"), "{stdout}");
    assert!(stdout.contains(&format!("{errs} error(s), {warns} warning(s), {infos} info(s):")), "{stdout}");
    assert!(stdout.contains("--min-severity info"), "{stdout}");

    // Human `--min-severity error`: hides warnings too and reports how many warnings were hidden.
    let (code, stdout, _) = folo(&["dataset", "check", s(&dir), "--min-severity", "error"]);
    assert_eq!(code, 1);
    assert!(stdout.contains("error    class_overflow"), "{stdout}");
    assert!(!stdout.contains("warning  duplicate_image"), "{stdout}");
    assert!(!stdout.contains("info     empty_image"), "{stdout}");
    assert!(stdout.contains(&format!("{errs} error(s), {warns} warning(s), {infos} info(s):")), "{stdout}");
    assert!(stdout.contains(&format!("({warns} warning(s) hidden by --min-severity error)")), "{stdout}");

    // Human `--min-severity info`: lists error, warning, AND info lines.
    let (code, stdout, _) = folo(&["dataset", "check", s(&dir), "--min-severity", "info"]);
    assert_eq!(code, 1);
    assert!(stdout.contains("error    class_overflow"), "{stdout}");
    assert!(stdout.contains("warning  duplicate_image"), "{stdout}");
    assert!(stdout.contains("info     empty_image"), "{stdout}");
    assert!(stdout.contains(&format!("{errs} error(s), {warns} warning(s), {infos} info(s):")), "{stdout}");

    // Human `--min-severity warn` alias works identically to `warning`.
    let (code, stdout, _) = folo(&["dataset", "check", s(&dir), "--min-severity", "warn"]);
    assert_eq!(code, 1);
    assert!(stdout.contains("error    class_overflow"), "{stdout}");
    assert!(stdout.contains("warning  duplicate_image"), "{stdout}");
    assert!(!stdout.contains("info     empty_image"), "{stdout}");
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

/// Naming a dataset by a bare file in the working directory (`data.yaml`, `instances_val.json`) or under a single
/// `annotations/` folder (`annotations/instances_val.json`) still sets `root` to `.` and reports issue paths
/// relative to it, rather than leaving `root` empty and emitting full absolute paths.
#[test]
fn bare_yaml_and_coco_paths_report_dot_root_and_relative_issue_files() {
    let t = TestDir::new("cli-bare-root");
    let yolo = t.join("yolo");
    let (code, _, stderr) = folo(&["dataset", "synth", s(&yolo), "--images", "4", "--imgsz", "64", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    std::fs::write(yolo.join("labels/train/000000.txt"), "99 0.5 0.5 0.2 0.2\n").unwrap();

    let run = |dir: &Path, args: &[&str]| {
        let out = folo_cmd().args(args).current_dir(dir).output().unwrap();
        (out.status.code().unwrap(), String::from_utf8(out.stdout).unwrap())
    };

    let (code, stdout) = run(&yolo, &["dataset", "stats", "data.yaml", "--json"]);
    assert_eq!(code, 0, "{stdout}");
    assert_eq!(json_lines(&stdout)[0]["root"], ".");

    let (code, stdout) = run(&yolo, &["dataset", "check", "data.yaml", "--json"]);
    assert_eq!(code, 1, "{stdout}");
    let r = &json_lines(&stdout)[0];
    assert_eq!(r["root"], ".");
    let overflow = r["issues"].as_array().unwrap().iter().find(|i| i["kind"] == "class_overflow").unwrap();
    assert_eq!(overflow["file"], "labels/train/000000.txt");

    // COCO under `annotations/instances_val.json` and bare `instances_val.json` in the working directory.
    let coco = t.join("coco");
    let (code, _, stderr) = folo(&["dataset", "convert", s(&yolo), s(&coco), "--to", "coco", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let val_img = std::fs::read_dir(coco.join("images/val")).unwrap().next().unwrap().unwrap().path();
    std::fs::write(&val_img, b"not an image").unwrap();
    let expected_val = format!("images/val/{}", val_img.file_name().unwrap().to_str().unwrap());

    let (code, stdout) = run(&coco, &["dataset", "stats", "annotations/instances_val.json", "--json"]);
    assert_eq!(code, 0, "{stdout}");
    assert_eq!(json_lines(&stdout)[0]["root"], ".");

    let (code, stdout) = run(&coco, &["dataset", "check", "annotations/instances_val.json", "--json"]);
    assert_eq!(code, 1, "{stdout}");
    let r = &json_lines(&stdout)[0];
    assert_eq!(r["root"], ".");
    let corrupt = r["issues"].as_array().unwrap().iter().find(|i| i["kind"] == "corrupt_image").unwrap();
    assert_eq!(corrupt["file"], expected_val);

    std::fs::copy(coco.join("annotations/instances_val.json"), coco.join("instances_val.json")).unwrap();
    let (code, stdout) = run(&coco, &["dataset", "check", "instances_val.json", "--json"]);
    assert_eq!(code, 1, "{stdout}");
    let r = &json_lines(&stdout)[0];
    assert_eq!(r["root"], ".");
    let corrupt = r["issues"].as_array().unwrap().iter().find(|i| i["kind"] == "corrupt_image").unwrap();
    assert_eq!(corrupt["file"], expected_val);
}

/// A `convert` stopped during image copying holds `.folo-unfinished` and `images/train/` with no `labels/`,
/// `annotations/` or `data.yaml` yet: `stats` and `check` must report `dataset_unfinished`, not `unknown_format`.
#[test]
fn interrupted_convert_folder_reports_dataset_unfinished() {
    let t = TestDir::new("cli-unfinished-convert");
    let dir = t.join("partial");
    std::fs::create_dir_all(dir.join("images/train")).unwrap();
    std::fs::write(dir.join(folo_data::unfinished::MARKER), "").unwrap();
    std::fs::write(dir.join("images/train/000000.png"), "half").unwrap();
    for cmd in ["stats", "check"] {
        let (code, stdout, _) = folo(&["dataset", cmd, s(&dir), "--json"]);
        let err = &json_lines(&stdout).pop().unwrap()["error"];
        assert_eq!((code, err["code"].as_str()), (1, Some("dataset_unfinished")), "{cmd}: {err}");
    }
}

/// `folo dataset convert` counts only skipped images in `skipped_images`, not broken non-image COCO entries
/// (like a malformed category or an annotation without `image_id`).
#[test]
fn convert_does_not_count_non_image_coco_broken_labels_as_skipped_images() {
    let t = TestDir::new("cli-convert-skipped");
    let src = synth(&t, "src", &["--val-frac", "0.25", "--image-format", "png"]);
    let img_a = src.join("images/train/000000.png");
    let img_b = src.join("images/train/000001.png");

    // Case 1: only non-image broken entries (bad category + annotation without `image_id`) -> 0 skipped images.
    let coco1 = t.join("coco_non_image_broken");
    std::fs::create_dir_all(coco1.join("images/train")).unwrap();
    std::fs::create_dir_all(coco1.join("annotations")).unwrap();
    std::fs::copy(&img_a, coco1.join("images/train/a.png")).unwrap();
    let doc1 = serde_json::json!({
        "images": [{ "id": 1, "file_name": "a.png", "width": 96, "height": 96 }],
        "annotations": [
            { "id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0 },
            { "id": 2, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0 }
        ],
        "categories": [
            { "id": 1, "name": "circle" },
            { "name": "missing_id" }
        ]
    });
    std::fs::write(coco1.join("annotations/instances_train.json"), doc1.to_string()).unwrap();

    let out1 = t.join("out1");
    let (code, stdout, stderr) = folo(&["dataset", "convert", s(&coco1), s(&out1), "--to", "yolo", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let v1 = &json_lines(&stdout)[0];
    assert_eq!(v1["splits"], serde_json::json!({ "train": 1, "val": 0, "test": 0 }));
    assert_eq!(
        v1["skipped_images"], 0,
        "non-image broken COCO entries must not inflate skipped_images: {v1}"
    );

    // Case 2: add a second image whose own annotation has a broken bbox plus a malformed `images[]` row missing `id`
    // -> exactly 2 skipped images (still excluding the bad category and annotation without `image_id`).
    let coco2 = t.join("coco_with_skipped_images");
    std::fs::create_dir_all(coco2.join("images/train")).unwrap();
    std::fs::create_dir_all(coco2.join("annotations")).unwrap();
    std::fs::copy(&img_a, coco2.join("images/train/a.png")).unwrap();
    std::fs::copy(&img_b, coco2.join("images/train/b.png")).unwrap();
    let doc2 = serde_json::json!({
        "images": [
            { "id": 1, "file_name": "a.png", "width": 96, "height": 96 },
            { "id": 2, "file_name": "b.png", "width": 96, "height": 96 },
            { "file_name": "missing_id.png", "width": 96, "height": 96 }
        ],
        "annotations": [
            { "id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0 },
            { "id": 2, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0 },
            { "id": 3, "image_id": 2, "category_id": 1, "bbox": [10, 10, -5, 20], "iscrowd": 0 }
        ],
        "categories": [
            { "id": 1, "name": "circle" },
            { "name": "missing_id" }
        ]
    });
    std::fs::write(coco2.join("annotations/instances_train.json"), doc2.to_string()).unwrap();

    let out2 = t.join("out2");
    let (code, stdout, stderr) = folo(&["dataset", "convert", s(&coco2), s(&out2), "--to", "yolo", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let v2 = &json_lines(&stdout)[0];
    assert_eq!(v2["splits"], serde_json::json!({ "train": 1, "val": 0, "test": 0 }));
    assert_eq!(v2["skipped_images"], 2, "skipped image row + broken-annotation image only: {v2}");
}

/// A directory link (junction on Windows, symlink elsewhere) removed on drop so `TestDir` cleans up cleanly.
struct DirLink(PathBuf);

impl DirLink {
    fn new(target: &Path, link: PathBuf) -> Self {
        #[cfg(windows)]
        {
            let native = |p: &Path| p.to_string_lossy().replace('/', "\\");
            let ok = Command::new("cmd")
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

/// `folo dataset convert` refuses an output inside (or equal to / containing) the input dataset or any of its
/// image/source folders, including through `..`, symlinks/junctions, and YOLO `path:` indirection.
#[test]
fn convert_refuses_output_inside_input_dataset_or_image_folder() {
    let t = TestDir::new("cli-convert-inside");
    let ds = t.join("ds");
    let (code, _, stderr) = folo(&["dataset", "synth", s(&ds), "--images", "4", "--imgsz", "32", "--json"]);
    assert_eq!(code, 0, "{stderr}");

    let mut targets = vec![
        ds.clone(),
        t.path().to_path_buf(),
        ds.join("images"),
        ds.join("images/out"),
        ds.join("images/train/out"),
        ds.join("nested_out"),
        t.join("scratch/../ds/images/out"),
    ];
    if cfg!(windows) {
        targets.push(t.join("DS/IMAGES/out"));
    }
    for target in targets {
        let (code, stdout, _) = folo(&["dataset", "convert", s(&ds), s(&target), "--to", "coco", "--json"]);
        let err = &json_lines(&stdout).pop().unwrap()["error"];
        assert_eq!((code, err["code"].as_str()), (2, Some("usage")), "{}: {err}", target.display());
        assert!(err["hint"].as_str().unwrap().contains("outside"), "{err}");
    }
    assert!(!ds.join("images/out").exists(), "must not create output inside input");
    assert!(!ds.join("nested_out").exists(), "must not create output inside input");

    // Symlink / junction into `ds/images`: `alias/out` writes into `ds/images/out` on every OS, and on Unix
    // `alias/../out` resolves `alias` before `..` and writes into `ds/out` (Windows resolves `..` lexically before
    // junctions, so `alias/../out` resolves to `<t>/out` outside `ds`).
    let alias = t.join("alias");
    let _link = DirLink::new(&ds.join("images"), alias.clone());
    let via_link_in = alias.join("out");
    let (code, stdout, _) = folo(&["dataset", "convert", s(&ds), s(&via_link_in), "--to", "coco", "--json"]);
    let err = &json_lines(&stdout).pop().unwrap()["error"];
    assert_eq!((code, err["code"].as_str()), (2, Some("usage")), "{err}");
    assert!(!ds.join("images/out").exists(), "link/out must not write inside ds/images");
    #[cfg(unix)]
    {
        let via_link_up = alias.join("../out");
        let (code, stdout, _) =
            folo(&["dataset", "convert", s(&ds), s(&via_link_up), "--to", "coco", "--json"]);
        let err = &json_lines(&stdout).pop().unwrap()["error"];
        assert_eq!((code, err["code"].as_str()), (2, Some("usage")), "{err}");
        assert!(!ds.join("out").exists(), "symlink/../out must not write inside ds");
    }

    // YOLO `path:` pointing to another folder still protects the supplied config folder too.
    let config = t.join("config");
    std::fs::create_dir_all(&config).unwrap();
    std::fs::write(
        config.join("data.yaml"),
        "path: ../ds\ntrain: images/train\nval: images/val\nnames: [circle, square, triangle]\n",
    )
    .unwrap();
    for out_path in [config.join("out"), ds.join("from_config_out")] {
        let (code, stdout, _) =
            folo(&["dataset", "convert", s(&config), s(&out_path), "--to", "coco", "--json"]);
        let err = &json_lines(&stdout).pop().unwrap()["error"];
        assert_eq!((code, err["code"].as_str()), (2, Some("usage")), "{}: {err}", out_path.display());
        assert!(!out_path.exists());
    }

    // Even when the target already exists as an empty folder inside the input, refuse it and leave it empty.
    let empty_inside = ds.join("images/empty_out");
    std::fs::create_dir_all(&empty_inside).unwrap();
    let (code, stdout, _) = folo(&["dataset", "convert", s(&ds), s(&empty_inside), "--to", "yolo", "--json"]);
    let err = &json_lines(&stdout).pop().unwrap()["error"];
    assert_eq!((code, err["code"].as_str()), (2, Some("usage")), "{err}");
    assert!(std::fs::read_dir(&empty_inside).unwrap().next().is_none());

    // Bare `data.yaml` from inside `ds` converting to `images/out` is refused too.
    let out = folo_cmd()
        .args(["dataset", "convert", "data.yaml", "images/out", "--to", "coco", "--json"])
        .current_dir(&ds)
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let err = &json_lines(&stdout).pop().unwrap()["error"];
    assert_eq!((out.status.code(), err["code"].as_str()), (Some(2), Some("usage")), "{err}");
    assert!(!ds.join("images/out").exists());
}

/// On Windows, `synth` and `convert` to a nonexistent drive report `path_not_found` with an output-path hint
/// rather than the missing-input-dataset hint.
#[cfg(windows)]
#[test]
fn synth_and_convert_to_missing_drive_give_output_path_hint() {
    let Some(drive) = ('D'..='Z').rev().find(|d| !Path::new(&format!("{d}:\\")).exists()) else {
        return;
    };
    let bad_out = format!("{drive}:/folo-no-such-drive/out");
    let (code, stdout, _) = folo(&["dataset", "synth", &bad_out, "--images", "2", "--imgsz", "32", "--json"]);
    let err = &json_lines(&stdout).pop().unwrap()["error"];
    assert_eq!((code, err["code"].as_str()), (3, Some("path_not_found")), "{err}");
    let hint = err["hint"].as_str().unwrap_or("");
    assert!(hint.contains("output") && !hint.contains("data.yaml"), "synth hint: {err}");

    let t = TestDir::new("cli-convert-bad-drive");
    let ds = t.join("ds");
    let (code, _, stderr) = folo(&["dataset", "synth", s(&ds), "--images", "4", "--imgsz", "32", "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let (code, stdout, _) = folo(&["dataset", "convert", s(&ds), &bad_out, "--to", "coco", "--json"]);
    let err = &json_lines(&stdout).pop().unwrap()["error"];
    assert_eq!((code, err["code"].as_str()), (3, Some("path_not_found")), "{err}");
    let hint = err["hint"].as_str().unwrap_or("");
    assert!(hint.contains("output") && !hint.contains("data.yaml"), "convert hint: {err}");
}

/// Helper to create a file symlink when the OS/account allows it (always on Unix; on Windows, skipped only if
/// unprivileged accounts lack SeCreateSymbolicLinkPrivilege / Developer Mode).
fn try_file_symlink(target: &Path, link: &Path) -> bool {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link).unwrap();
        true
    }
    #[cfg(windows)]
    {
        match std::os::windows::fs::symlink_file(target, link) {
            Ok(()) => true,
            Err(e) if e.raw_os_error() == Some(1314) => false,
            Err(e) => panic!("symlink_file {} -> {}: {e}", link.display(), target.display()),
        }
    }
}

/// `folo dataset check` flags dangling symlinks / Windows directory junctions with clear diagnostics instead of
/// silently skipping them or misreporting a dangling label symlink as a background image (#382).
#[test]
fn check_flags_dangling_symlinks_portably() {
    let t = TestDir::new("cli-check-dangling");
    let ds = t.join("ds");
    let (code, _, stderr) = folo(&["dataset", "synth", s(&ds), "--images", "4", "--imgsz", "64", "--json"]);
    assert_eq!(code, 0, "{stderr}");

    // 1. Dangling directory link (junction on Windows, symlink on Unix) inside `images/train/` and as the dataset
    // path itself.
    let doomed_dir = t.join("doomed_dir");
    std::fs::create_dir(&doomed_dir).unwrap();
    let _dangling_sub = DirLink::new(&doomed_dir, ds.join("images/train/broken_sub"));
    let _dangling_root = DirLink::new(&doomed_dir, t.join("broken_ds_link"));
    std::fs::remove_dir(&doomed_dir).unwrap();

    // Passing a dangling symlink as the dataset path reports `path_not_found` (exit 3) naming the broken link.
    let (code, stdout, _) = folo(&["dataset", "check", s(&t.join("broken_ds_link")), "--json"]);
    let err = &json_lines(&stdout).pop().unwrap()["error"];
    assert_eq!((code, err["code"].as_str()), (3, Some("path_not_found")), "{err}");
    assert!(err["message"].as_str().unwrap().contains("dangling symlink"), "{err}");

    // Inside `images/train/`, the dangling subdirectory link is flagged rather than silently skipped.
    let (code, stdout, _) = folo(&["dataset", "check", s(&ds), "--json", "--fix-dry-run"]);
    assert_eq!(code, 1);
    let r = &json_lines(&stdout)[0];
    let issues = r["issues"].as_array().unwrap();
    let sub = issues
        .iter()
        .find(|i| i["file"] == "images/train/broken_sub")
        .unwrap_or_else(|| panic!("expected dangling dir issue in {issues:?}"));
    assert!(sub["message"].as_str().unwrap().contains("dangling symlink"), "{sub}");
    assert!(sub["fix"].as_str().unwrap().contains("symlink"), "{sub}");

    // 2. Dangling image, paired label, and orphan label file symlinks (always on Unix; on Windows when enabled).
    let dangling_img = ds.join("images/train/ghost.jpg");
    if try_file_symlink(&t.join("no_such_image.jpg"), &dangling_img) {
        let lbl = ds.join("labels/train/000000.txt");
        std::fs::remove_file(&lbl).unwrap();
        assert!(try_file_symlink(&t.join("no_such_label.txt"), &lbl));
        // Orphan dangling label symlink (no matching image) must be reported as `broken_label`, never dropped.
        let orphan_lbl = ds.join("labels/train/dangling_orphan.txt");
        assert!(try_file_symlink(&t.join("no_such_orphan.txt"), &orphan_lbl));

        let (code, stdout, _) = folo(&["dataset", "check", s(&ds), "--json", "--fix-dry-run"]);
        assert_eq!(code, 1);
        let r = &json_lines(&stdout)[0];
        let issues = r["issues"].as_array().unwrap();

        let img_issue = issues
            .iter()
            .find(|i| i["file"] == "images/train/ghost.jpg")
            .unwrap_or_else(|| panic!("expected dangling image issue in {issues:?}"));
        assert_eq!(img_issue["kind"], "missing_image", "{img_issue}");
        assert_eq!(img_issue["severity"], "error", "{img_issue}");
        assert!(img_issue["message"].as_str().unwrap().contains("dangling symlink"), "{img_issue}");

        let lbl_issue = issues
            .iter()
            .find(|i| i["file"] == "labels/train/000000.txt")
            .unwrap_or_else(|| panic!("expected dangling label issue in {issues:?}"));
        assert_eq!(lbl_issue["kind"], "broken_label", "{lbl_issue}");
        assert_eq!(lbl_issue["severity"], "error", "{lbl_issue}");
        assert!(lbl_issue["message"].as_str().unwrap().contains("dangling symlink"), "{lbl_issue}");

        let orphan_issues: Vec<_> =
            issues.iter().filter(|i| i["file"] == "labels/train/dangling_orphan.txt").collect();
        assert_eq!(orphan_issues.len(), 1, "orphan dangling label must appear once: {issues:?}");
        assert_eq!(orphan_issues[0]["kind"], "broken_label", "{orphan_issues:?}");
        assert!(
            orphan_issues[0]["message"].as_str().unwrap().contains("dangling symlink"),
            "{orphan_issues:?}"
        );
        // Must NOT also misreport `000000.jpg` as a background image (`empty_image`).
        assert!(
            !issues.iter().any(|i| i["kind"] == "empty_image" && i["file"] == "images/train/000000.jpg"),
            "dangling label symlink must not be misreported as empty_image: {issues:?}"
        );
    }

    // 3. Empty `folo dataset synth ""` inside an existing dataset must be refused with exit 2 and delete nothing.
    let empty_out = folo_cmd()
        .args(["dataset", "synth", "", "--images", "2", "--imgsz", "32", "--json"])
        .current_dir(&ds)
        .output()
        .unwrap();
    let stdout = String::from_utf8(empty_out.stdout).unwrap();
    assert_eq!(empty_out.status.code(), Some(2), "{stdout}");
    assert_eq!(json_lines(&stdout).last().unwrap()["error"]["code"], "usage");
    assert!(
        ds.join("data.yaml").is_file() && ds.join("images/train").is_dir(),
        "cwd dataset must stay intact"
    );
}

/// `folo dataset check` on COCO evaluates boxes against the real image dimensions (matching the loader) when the
/// JSON's declared `width`/`height` disagree with the actual image (#422).
#[test]
fn check_coco_reconciles_boxes_against_real_image_dimensions() {
    let t = TestDir::new("cli-check-coco-dims");
    let src = t.join("src");
    let (code, _, stderr) = folo(&[
        "dataset",
        "synth",
        s(&src),
        "--images",
        "4",
        "--imgsz",
        "64",
        "--image-format",
        "png",
        "--json",
    ]);
    assert_eq!(code, 0, "{stderr}");
    let real_png = src.join("images/train/000000.png");

    let coco = t.join("coco");
    std::fs::create_dir_all(coco.join("images/train")).unwrap();
    std::fs::create_dir_all(coco.join("annotations")).unwrap();
    std::fs::copy(&real_png, coco.join("images/train/small_decl.png")).unwrap();
    std::fs::copy(&real_png, coco.join("images/train/large_decl.png")).unwrap();

    // Both images are actually 64x64:
    // - `small_decl.png` declares 32x32 with a box at [40, 40, 10, 10] (valid in 64x64, outside 32x32).
    // - `large_decl.png` declares 128x128 with a box at [80, 80, 10, 10] (outside 64x64, inside 128x128).
    let doc = serde_json::json!({
        "images": [
            { "id": 1, "file_name": "small_decl.png", "width": 32, "height": 32 },
            { "id": 2, "file_name": "large_decl.png", "width": 128, "height": 128 }
        ],
        "annotations": [
            { "id": 1, "image_id": 1, "category_id": 1, "bbox": [40, 40, 10, 10], "iscrowd": 0 },
            { "id": 2, "image_id": 2, "category_id": 1, "bbox": [80, 80, 10, 10], "iscrowd": 0 }
        ],
        "categories": [
            { "id": 1, "name": "circle" }
        ]
    });
    std::fs::write(coco.join("annotations/instances_train.json"), doc.to_string()).unwrap();

    for extra in [&[][..], &["--headers-only"][..]] {
        let mut args = vec!["dataset", "check", s(&coco), "--json"];
        args.extend_from_slice(extra);
        let (code, stdout, _) = folo(&args);
        assert_eq!(code, 1, "{extra:?}");
        let r = &json_lines(&stdout)[0];
        let issues = r["issues"].as_array().unwrap();
        let kinds_of = |file: &str| -> Vec<&str> {
            issues.iter().filter(|i| i["file"] == file).map(|i| i["kind"].as_str().unwrap()).collect()
        };

        // 64x64 declared as 32x32: size mismatch warning ONLY — no false zero_size_box or empty_image.
        let small = kinds_of("images/train/small_decl.png");
        assert_eq!(small, vec!["other"], "extra={extra:?}, issues={issues:?}");

        // 64x64 declared as 128x128: size mismatch warning PLUS the zero_size_box and empty_image the loader sees.
        let large = kinds_of("images/train/large_decl.png");
        assert!(large.contains(&"other"), "missing size mismatch: {large:?}");
        assert!(large.contains(&"zero_size_box"), "missing zero_size_box: {large:?}");
        assert!(large.contains(&"empty_image"), "missing empty_image: {large:?}");
    }

    // Normal loader (`stats`) agrees: 1 box kept (from small_decl), 1 empty image (large_decl).
    let (code, stdout, stderr) = folo(&["dataset", "stats", s(&coco), "--json"]);
    assert_eq!(code, 0, "{stderr}");
    let st = &json_lines(&stdout)[0]["stats"];
    assert_eq!((st["boxes"].as_u64(), st["empty_images"].as_u64()), (Some(1), Some(1)), "{st}");
}

/// `folo dataset convert` fails loudly (instead of silently dropping a split) when a COCO split annotation file is
/// corrupt JSON or lacks an `images` array, and writes no partial output directory.
#[test]
fn convert_refuses_corrupt_coco_split_annotation_file() {
    let t = TestDir::new("cli-convert-corrupt-coco");
    let src = t.join("synthetic");
    let (code, _, stderr) = folo(&[
        "dataset",
        "synth",
        s(&src),
        "--images",
        "8",
        "--imgsz",
        "64",
        "--val-frac",
        "0.5",
        "--seed",
        "13",
    ]);
    assert_eq!(code, 0, "{stderr}");

    let coco = t.join("coco");
    let (code, _, stderr) = folo(&["dataset", "convert", s(&src), s(&coco), "--to", "coco"]);
    assert_eq!(code, 0, "{stderr}");

    // Case 1: truncated/invalid JSON in instances_val.json.
    std::fs::write(coco.join("annotations/instances_val.json"), "{\"images\":[").unwrap();
    let out1 = t.join("converted_bad_json");
    let (code, stdout, _) = folo(&["dataset", "convert", s(&coco), s(&out1), "--to", "yolo", "--json"]);
    let err = &json_lines(&stdout).pop().unwrap()["error"];
    assert_eq!((code, err["code"].as_str()), (1, Some("invalid_dataset")), "{err}");
    let msg = err["message"].as_str().unwrap();
    assert!(
        msg.contains("annotations/instances_val.json") && msg.contains("invalid JSON"),
        "expected filename and parse error: {msg}"
    );
    assert!(!out1.exists(), "must not publish a partial output dataset");

    // Case 2: valid JSON but missing `images` array (not a COCO file).
    std::fs::write(coco.join("annotations/instances_val.json"), "{}").unwrap();
    let out2 = t.join("converted_no_images");
    let (code, stdout, _) = folo(&["dataset", "convert", s(&coco), s(&out2), "--to", "yolo", "--json"]);
    let err = &json_lines(&stdout).pop().unwrap()["error"];
    assert_eq!((code, err["code"].as_str()), (1, Some("invalid_dataset")), "{err}");
    let msg = err["message"].as_str().unwrap();
    assert!(
        msg.contains("annotations/instances_val.json") && msg.contains("no `images` array"),
        "expected filename and missing-images error: {msg}"
    );
    assert!(!out2.exists(), "must not publish a partial output dataset");
}

/// A `folo` command with the on-disk caches off (`FOLO_NO_CACHE`), so tests never write into the user's home.
fn folo_cmd() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_folo"));
    c.env("FOLO_NO_CACHE", "1");
    c
}
