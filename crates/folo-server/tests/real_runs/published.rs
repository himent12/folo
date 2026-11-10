//! Published weights as models, fine-tuned by name.

use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use folo_server::RealStore;
use folo_server::api::*;
use folo_test_support::TestDir;
use serde_json::json;

use super::harness::{Events, app, dataset, err, get, ok, post_json, store, tiny};

/// Published weights downloaded with `folo weights pull` list as models (id = the published name, `source:
/// published`) when the store is given the weights cache, and fine-tune by name; the run takes the weights'
/// architecture whatever `arch` says. A published name that is not downloaded is `model_not_found` with the pull
/// command as the hint.
#[tokio::test(flavor = "multi_thread")]
async fn published_weights_list_as_models_and_fine_tune_by_name() {
    let t = TestDir::new("runs-published");
    let home = t.join("home");
    let cache = t.join("weights");
    std::fs::create_dir_all(&cache).unwrap();
    let mut m = folo_vision::FoloModel::with_rng(
        folo_nn::io::ModelConfig::new("n", 3).unwrap(),
        &mut folo_core::rng::Rng::seed(9),
    )
    .unwrap();
    m.set_classes(vec!["a".into(), "b".into(), "c".into()]).unwrap();
    m.save(cache.join("folo-n-coco.folo")).unwrap();

    // Without the cache the name is unknown, and the hint says how to get it there.
    let plain = store(&home);
    let e = err(&self::app(&plain), get("/api/models/folo-n-coco"), StatusCode::NOT_FOUND, "model_not_found")
        .await;
    assert!(e.hint.unwrap().contains("folo weights pull folo-n-coco"));
    drop(plain);

    let s = Arc::new(
        RealStore::open(home.clone()).recheck_every(Duration::ZERO).device("cpu").published_weights(&cache),
    );
    let app = app(&s);
    let models = ok::<ModelList>(&app, get("/api/models")).await.models;
    let published: Vec<&Model> = models.iter().filter(|m| m.source == Some(ModelSource::Published)).collect();
    assert_eq!(published.len(), 1, "{models:?}");
    let p = published[0];
    assert_eq!((p.id.as_str(), p.name.as_str(), p.arch), ("folo-n-coco", "folo-n-coco", Arch::N));
    assert!(p.about.as_deref().is_some_and(|a| a.contains("COCO")) && p.run_id.is_none());
    assert_eq!(p.path.as_deref(), Some(folo_api::json::path(cache.join("folo-n-coco.folo")).as_str()));
    assert_eq!(ok::<Model>(&app, get("/api/models/folo-n-coco")).await, *p);
    // Downloadable but not downloaded.
    let e = err(&app, get("/api/models/folo-n-coco-f16"), StatusCode::NOT_FOUND, "model_not_found").await;
    assert!(e.message.contains("not downloaded"), "{}", e.message);
    assert!(e.hint.unwrap().contains("folo weights pull folo-n-coco-f16"));

    // A fine-tune by name, with a mismatching `arch`: the run trains folo-n.
    let ds = dataset(&app, "pub", 5).await;
    let mut cfg = tiny(&ds, 1);
    cfg["weights"] = json!("folo-n-coco");
    cfg["arch"] = json!("folo-s");
    let est: RunEstimate = ok(&app, post_json("/api/runs/estimate", cfg.clone())).await;
    let cli = est.cli.unwrap();
    assert!(cli.contains(" --weights folo-n-coco ") && !cli.contains("--model"), "{cli}");
    // tiny: close_mosaic 1 on a 1-epoch run, capped at half the run.
    assert_eq!(est.close_mosaic, 0);
    let run: Run = ok(&app, post_json("/api/runs", cfg)).await;
    assert_eq!((run.config.arch, run.config.weights.as_deref()), (Arch::N, Some("folo-n-coco")));
    // The run's command names the published weights and the dataset's folder on this machine.
    let c: CliCommand = ok(&app, get(&format!("/api/runs/{}/cli", run.id))).await;
    let at = |flag: &str| c.argv.iter().position(|a| a == flag).map(|i| c.argv[i + 1].as_str());
    assert_eq!((at("--weights"), at("--head")), (Some("folo-n-coco"), Some("o2m")), "{}", c.cli);
    assert!(at("--data").is_some_and(|d| std::path::Path::new(d).is_dir()), "{}", c.cli);
    let mut sse = Events::open(&app, &run.id).await;
    let events = sse.until_status(&["completed", "failed"]).await;
    assert_eq!(events.last().unwrap().1["status"], "completed", "{events:?}");
    let done: Run = ok(&app, get(&format!("/api/runs/{}", run.id))).await;
    assert_eq!(done.resolved.unwrap().close_mosaic, Some(0));
    let model: Model = ok(&app, get(&format!("/api/models/{}", done.model_id.unwrap()))).await;
    assert!(model.path.unwrap().ends_with("/weights/best.folo"));
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", run.id))).await;
    let transferred = |l: &&LogLine| l.message.contains("transferred") && l.message.contains("folo-n-coco");
    assert!(logs.lines.iter().any(|l| transferred(&l)), "{:?}", logs.lines);
}
