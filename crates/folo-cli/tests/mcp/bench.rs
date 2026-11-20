//! The benchmark tools on the real store over stdio: a published model (a random folo-n standing in for the
//! release asset) entered on a benchmark of a synthetic dataset, followed with `bench_entry`'s `wait` to its score.
//! The other contestant, a file of predictions, and the refusals are folo-mcp's `tools/bench_tests.rs`.

use serde_json::json;

use super::harness::{ALL_TOOLS, check_tools, checked, init_old, spawn_with_weights, tmp_dir};

#[test]
fn a_published_model_runs_a_benchmark() {
    let dir = tmp_dir("bench");
    let weights = dir.join("weights");
    std::fs::create_dir_all(&weights).unwrap();
    let mut m = folo_vision::FoloModel::with_rng(
        folo_vision::ModelConfig::new("n", 2).unwrap(),
        &mut folo_core::rng::Rng::seed(9),
    )
    .unwrap();
    m.set_classes(vec!["circle".into(), "square".into()]).unwrap();
    m.set_imgsz(64);
    m.save(weights.join("folo-n-coco.folo")).unwrap();
    let home = dir.join("home");
    let mut c =
        spawn_with_weights(&["--data-dir", home.to_str().unwrap(), "--device", "cpu"], None, &weights);
    init_old(&mut c);
    let schemas = check_tools(&mut c, &ALL_TOOLS);
    let ds = c.ok(
        "create_synthetic_dataset",
        json!({ "name": "shapes", "images": 20, "imgsz": 64, "classes": ["circle", "square"], "seed": 3 }),
    );
    let plan = json!({ "conditions": [{ "condition": "haze", "severities": [2] }] });
    let b = checked(
        &mut c,
        &schemas,
        "create_benchmark",
        json!({ "name": "shapes", "dataset_id": ds["id"], "plan": plan }),
    );
    let started = checked(
        &mut c,
        &schemas,
        "start_bench_entry",
        json!({ "benchmark_id": b["id"], "model_id": "folo-n-coco", "name": "published" }),
    );
    assert_eq!((started["total"].as_u64(), started["result"].is_null()), (Some(8), true), "{started}");
    let ids = json!({ "benchmark_id": b["id"], "entry_id": started["entry_id"], "wait": 50 });
    let mut entry = checked(&mut c, &schemas, "bench_entry", ids.clone());
    // A slow machine: one more wait (each is under the minute a client gives a call).
    if entry["result"].is_null() {
        entry = checked(&mut c, &schemas, "bench_entry", ids);
    }
    assert_eq!((entry["status"].as_str(), entry["done"].as_u64()), (Some("done"), Some(8)), "{entry}");
    let r = &entry["result"];
    assert_eq!(r["axes"].as_array().unwrap().len(), 5, "{r}");
    assert_eq!(r["conditions"][0]["condition"], "haze", "{r}");
    assert!(r["ms_per_image"].as_f64().unwrap() > 0.0, "{r}");
    let shown = checked(&mut c, &schemas, "get_benchmark", json!({ "benchmark_id": b["id"] }));
    assert_eq!(
        (shown["leaderboard"][0]["name"].as_str(), shown["leaderboard"][0]["kind"].as_str()),
        (Some("published"), Some("model"))
    );
    c.finish();
}
