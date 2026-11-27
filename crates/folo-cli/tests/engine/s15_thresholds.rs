//! End-to-end CLI integration tests for Sprint 15 per-class F1 threshold calibration (`folo val --save-thresholds`
//! -> `folo predict --thresholds`, `folo describe`, and error handling).

use std::process::Command;

use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::{folo, folo_cmd, s, tiny_run};

#[test]
fn val_save_thresholds_and_predict_thresholds_round_trip() {
    let t = tiny_run();
    let weights = t.run.join("weights").join("best.folo");
    let scratch = TestDir::new("cli-s15-thresholds");
    let cal_path = scratch.join("calibrated").join("thresholds.json");

    // 1. `folo val --json --save-thresholds` emits calibrated `thresholds`, `thresholds_saved`, and writes JSON.
    let (code, recs, err) = folo(&[
        "val",
        "--model",
        s(&weights),
        "--data",
        s(&t.data),
        "--imgsz",
        "64",
        "--save-thresholds",
        s(&cal_path),
        "--json",
    ]);
    assert_eq!(code, 0, "val failed: {err}");
    assert_eq!(recs.len(), 1);
    let rec = &recs[0];
    assert!(rec["thresholds"].is_object(), "val --json must emit thresholds: {rec}");
    assert!(rec["thresholds"]["global"]["best_conf"].is_f64(), "{rec}");
    assert!(rec["thresholds"]["macro_f1"].is_f64(), "{rec}");
    let per_class = rec["thresholds"]["per_class"].as_array().unwrap();
    assert!(!per_class.is_empty(), "expected per-class operating points: {rec}");
    assert!(rec["thresholds_saved"].as_str().unwrap().ends_with("thresholds.json"), "{rec}");
    assert!(cal_path.is_file(), "thresholds file not written");

    let saved: Value = serde_json::from_slice(&std::fs::read(&cal_path).unwrap()).unwrap();
    assert_eq!(saved["global"], rec["thresholds"]["global"]);
    assert_eq!(saved["per_class"], rec["thresholds"]["per_class"]);
    assert_eq!(saved["macro_f1"], rec["thresholds"]["macro_f1"]);

    // Human output concisely reports calibrated thresholds and saved path.
    let human = folo_cmd()
        .args([
            "val",
            "--model",
            s(&weights),
            "--data",
            s(&t.data),
            "--imgsz",
            "64",
            "--device",
            "cpu",
            "--save-thresholds",
            s(&cal_path),
        ])
        .output()
        .unwrap();
    assert!(human.status.success(), "{}", String::from_utf8_lossy(&human.stderr));
    let human_stdout = String::from_utf8(human.stdout).unwrap();
    assert!(human_stdout.contains("calibrated thresholds:"), "{human_stdout}");

    // 2. Round-trip `folo predict --thresholds` across single image, directory batch, tiled/SAHI, TTA, and ByteTrack.
    let val_dir = t.data.join("images").join("val");
    let first_img =
        std::fs::read_dir(&val_dir).unwrap().map(|e| e.unwrap().path()).find(|p| p.is_file()).unwrap();

    for (label, extra) in [
        ("single", vec![s(&first_img)]),
        ("batch", vec![s(&val_dir)]),
        ("tile", vec!["--tile", "32", "--tile-overlap", "0.2", s(&first_img)]),
        ("tta", vec!["--tta", "--tta-scales", "1.0", "--no-tta-hflip", s(&first_img)]),
        ("track", vec!["--track", s(&val_dir)]),
    ] {
        let mut args =
            vec!["predict", "--model", s(&weights), "--imgsz", "64", "--thresholds", s(&cal_path), "--json"];
        args.extend(extra);
        let (code, recs, err) = folo(&args);
        assert_eq!(code, 0, "predict ({label}) failed: {err}");
        let done = recs.last().unwrap();
        assert_eq!(done["event"], "done", "{label}: {done}");
        assert!(done["thresholds"].is_object(), "{label}: missing thresholds metadata: {done}");
        assert_eq!(done["thresholds"]["active"], true, "{label}: {done}");
        assert_eq!(done["thresholds"]["per_class_count"], per_class.len(), "{label}: {done}");
    }

    // 3. Deterministic detection filtering: setting both classes to 0.0 keeps detections while setting one or both
    //    to 1.0 suppresses detections of that class.
    let zero_path = scratch.join("zero.json");
    let one_path = scratch.join("one.json");
    let class0_only_path = scratch.join("class0_only.json");
    std::fs::write(
        &zero_path,
        serde_json::to_vec(&json!({
            "global": {"best_conf": 0.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5},
            "per_class": [
                {"class": 0, "best_conf": 0.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5, "truths": 4, "preds": 4},
                {"class": 1, "best_conf": 0.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5, "truths": 4, "preds": 4},
            ],
            "macro_f1": 0.5,
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &one_path,
        serde_json::to_vec(&json!({
            "global": {"best_conf": 1.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5},
            "per_class": [
                {"class": 0, "best_conf": 1.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5, "truths": 4, "preds": 4},
                {"class": 1, "best_conf": 1.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5, "truths": 4, "preds": 4},
            ],
            "macro_f1": 0.5,
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &class0_only_path,
        serde_json::to_vec(&json!({
            "global": {"best_conf": 1.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5},
            "per_class": [
                {"class": 0, "best_conf": 0.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5, "truths": 4, "preds": 4},
                {"class": 1, "best_conf": 1.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5, "truths": 4, "preds": 4},
            ],
            "macro_f1": 0.5,
        }))
        .unwrap(),
    )
    .unwrap();

    let (c0, r0, e0) = folo(&[
        "predict",
        "--model",
        s(&weights),
        "--imgsz",
        "64",
        "--thresholds",
        s(&zero_path),
        s(&first_img),
        "--json",
    ]);
    assert_eq!(c0, 0, "{e0}");
    let boxes_zero = r0[0]["boxes"].as_array().unwrap();
    assert!(!boxes_zero.is_empty(), "conf=0.0 thresholds should emit detections");

    let (c1, r1, e1) = folo(&[
        "predict",
        "--model",
        s(&weights),
        "--imgsz",
        "64",
        "--thresholds",
        s(&one_path),
        s(&first_img),
        "--json",
    ]);
    assert_eq!(c1, 0, "{e1}");
    assert!(r1[0]["boxes"].as_array().unwrap().is_empty(), "conf=1.0 thresholds should suppress all boxes");

    let (c2, r2, e2) = folo(&[
        "predict",
        "--model",
        s(&weights),
        "--imgsz",
        "64",
        "--thresholds",
        s(&class0_only_path),
        s(&first_img),
        "--json",
    ]);
    assert_eq!(c2, 0, "{e2}");
    let boxes_c0 = r2[0]["boxes"].as_array().unwrap();
    assert!(!boxes_c0.is_empty(), "class 0 at conf=0.0 should emit boxes");
    assert!(boxes_c0.iter().all(|b| b["class"] == 0), "class 1 at conf=1.0 must be suppressed: {boxes_c0:?}");
}

#[test]
fn thresholds_cli_errors_and_describe_schema() {
    let t = tiny_run();
    let weights = t.run.join("weights").join("best.folo");
    let scratch = TestDir::new("cli-s15-thresholds-err");
    let val_dir = t.data.join("images").join("val");
    let first_img =
        std::fs::read_dir(&val_dir).unwrap().map(|e| e.unwrap().path()).find(|p| p.is_file()).unwrap();

    // Missing thresholds file -> exit 3 (path_not_found).
    let (code, recs, _) = folo(&[
        "predict",
        "--model",
        s(&weights),
        "--thresholds",
        s(&scratch.join("missing.json")),
        s(&first_img),
        "--json",
    ]);
    assert_eq!(code, 3);
    assert_eq!(recs[0]["error"]["code"], "path_not_found");

    // Directory passed to --thresholds -> exit 2 (usage).
    let (code, recs, _) = folo(&[
        "predict",
        "--model",
        s(&weights),
        "--thresholds",
        s(scratch.path()),
        s(&first_img),
        "--json",
    ]);
    assert_eq!(code, 2);
    assert_eq!(recs[0]["error"]["code"], "usage");

    // Invalid JSON -> exit 2 (invalid_config).
    let bad_json = scratch.join("bad.json");
    std::fs::write(&bad_json, b"{invalid").unwrap();
    let (code, recs, _) =
        folo(&["predict", "--model", s(&weights), "--thresholds", s(&bad_json), s(&first_img), "--json"]);
    assert_eq!(code, 2);
    assert_eq!(recs[0]["error"]["code"], "invalid_config");

    // Out-of-bounds class id for 2-class model -> exit 2 (invalid_config).
    let oob = scratch.join("oob.json");
    std::fs::write(
        &oob,
        serde_json::to_vec(&json!({
            "global": {"best_conf": 0.3, "best_f1": 0.5, "precision": 0.5, "recall": 0.5},
            "per_class": [{"class": 5, "best_conf": 0.3, "best_f1": 0.5, "precision": 0.5, "recall": 0.5, "truths": 4, "preds": 4}],
            "macro_f1": 0.5,
        }))
        .unwrap(),
    )
    .unwrap();
    let (code, recs, _) =
        folo(&["predict", "--model", s(&weights), "--thresholds", s(&oob), s(&first_img), "--json"]);
    assert_eq!(code, 2);
    assert_eq!(recs[0]["error"]["code"], "invalid_config");

    // Combining --conf and --thresholds fails with exit 2.
    let out = Command::new(env!("CARGO_BIN_EXE_folo"))
        .args([
            "predict",
            "--model",
            s(&weights),
            "--conf",
            "0.25",
            "--thresholds",
            s(&oob),
            s(&first_img),
            "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));

    // Directory passed to val --save-thresholds fails upfront with exit 2 (usage).
    let (code, recs, _) = folo(&[
        "val",
        "--model",
        s(&weights),
        "--data",
        s(&t.data),
        "--save-thresholds",
        s(scratch.path()),
        "--json",
    ]);
    assert_eq!(code, 2);
    assert_eq!(recs[0]["error"]["code"], "usage");

    // Passing --model path or the same path as --save-json to --save-thresholds is refused upfront with exit 2
    // without modifying the weights file.
    let weights_before = std::fs::metadata(&weights).unwrap().len();
    let (code, recs, _) = folo(&[
        "val",
        "--model",
        s(&weights),
        "--data",
        s(&t.data),
        "--save-thresholds",
        s(&weights),
        "--json",
    ]);
    assert_eq!(code, 2);
    assert_eq!(recs[0]["error"]["code"], "usage");
    assert_eq!(std::fs::metadata(&weights).unwrap().len(), weights_before);

    let colliding = scratch.join("both.json");
    let (code, recs, _) = folo(&[
        "val",
        "--model",
        s(&weights),
        "--data",
        s(&t.data),
        "--save-json",
        s(&colliding),
        "--save-thresholds",
        s(&colliding),
        "--json",
    ]);
    assert_eq!(code, 2);
    assert_eq!(recs[0]["error"]["code"], "usage");

    // `folo describe` exposes `--save-thresholds` on `val` and `--thresholds` (conflicting with `--conf`) on `predict`.
    let (code, recs, err) = folo(&["describe", "val", "--json"]);
    assert_eq!(code, 0, "{err}");
    let val_args = recs[0]["args"].as_array().unwrap();
    let save_thr = val_args.iter().find(|a| a["long"] == "--save-thresholds").expect("val --save-thresholds");
    assert_eq!(save_thr["type"], "path");

    let (code, recs, err) = folo(&["describe", "predict", "--json"]);
    assert_eq!(code, 0, "{err}");
    let pred_args = recs[0]["args"].as_array().unwrap();
    let thr = pred_args.iter().find(|a| a["long"] == "--thresholds").expect("predict --thresholds");
    assert_eq!(thr["type"], "path");
    assert!(
        thr["conflicts_with"].as_array().unwrap().iter().any(|v| v == "--conf"),
        "expected --thresholds to conflict with --conf: {thr}"
    );
    let conf = pred_args.iter().find(|a| a["long"] == "--conf").expect("predict --conf");
    assert!(
        conf["conflicts_with"].as_array().unwrap().iter().any(|v| v == "--thresholds"),
        "expected symmetric conflict on --conf: {conf}"
    );
}
