//! `GET /api/runs/{id}/cli` of runs made under older recipes: the command repeats the `auto` rule and the warmup
//! floor the run trained with, read from its `config.json` or, for a cloud run (no `config.json` here), from its
//! checkpoint's `folo.train`.

use std::collections::BTreeMap;
use std::path::Path;

use folo_server::api::*;
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::harness::{app, get, ok, store};

/// A stopped run folder as a server writes it (`run.json`), for a dataset at `data`; no training needed.
fn write_run(dir: &Path, id: &str, data: &Path) {
    std::fs::create_dir_all(dir.join("weights")).unwrap();
    let config: TrainConfig =
        serde_json::from_value(json!({ "dataset_id": "ds_legacy", "epochs": 3, "imgsz": 64 })).unwrap();
    let run = json!({
        "id": id, "name": id, "status": "stopped", "config": config, "created": "2026-11-01T00:00:00Z",
        "epoch": 1, "epochs": 3, "step": 1, "steps_per_epoch": 1, "device": "cpu",
    });
    let data = folo_api::json::path(data);
    let file = json!({ "run": run, "dataset": { "id": "ds_legacy", "name": "legacy", "path": data, "format": "yolo" } });
    std::fs::write(dir.join("run.json"), file.to_string()).unwrap();
}

/// A checkpoint-shaped file whose header carries `folo.train` (the cloud mirror copies only `weights/*.folo`).
fn write_checkpoint(path: &Path, train: &Value) {
    let meta = BTreeMap::from([("folo.train".to_string(), train.to_string())]);
    let t = folo_core::Tensor::from_vec(vec![0f32], [1]).unwrap();
    folo_nn::io::safetensors::write_file(path, Some(&meta), &[("w".into(), t)]).unwrap();
}

/// A short `auto` run made under rule 3 with a `config.json` from before `warmup_floor` repeats with both
/// `--set auto_rule=3` and `--set warmup_floor=100`; a cloud run with no `config.json` reads the same keys from
/// `weights/last.folo` (missing `auto_rule`: rule 1; its own floor 0: no `--set`).
#[tokio::test(flavor = "multi_thread")]
async fn old_runs_repeat_with_their_recipe() {
    let t = TestDir::new("runs-cli-recipe");
    let home = t.join("home");
    let runs = home.join("runs");
    let data = t.join("data");
    std::fs::create_dir_all(&data).unwrap();

    let legacy = runs.join("run_legacy");
    write_run(&legacy, "run_legacy", &data);
    let config = json!({ "train": { "dataset_id": "ds_legacy", "epochs": 3, "auto_rule": 3 } });
    std::fs::write(legacy.join("config.json"), config.to_string()).unwrap();

    let cloud = runs.join("run_cloud");
    write_run(&cloud, "run_cloud", &data);
    write_checkpoint(&cloud.join("weights").join("last.folo"), &json!({ "epochs": 3, "warmup_floor": 0 }));

    let s = store(&home);
    let app = app(&s);
    let sets = |c: &CliCommand| -> Vec<String> {
        c.argv.windows(2).filter(|w| w[0] == "--set").map(|w| w[1].clone()).collect()
    };
    let c: CliCommand = ok(&app, get("/api/runs/run_legacy/cli")).await;
    assert_eq!(sets(&c), ["auto_rule=3", "warmup_floor=100"], "{}", c.cli);
    let c: CliCommand = ok(&app, get("/api/runs/run_cloud/cli")).await;
    assert_eq!(sets(&c), ["auto_rule=1"], "{}", c.cli);
}
