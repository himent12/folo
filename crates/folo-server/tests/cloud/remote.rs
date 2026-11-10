//! A `remote` connection: a FOLO server used as it is, and worker runs whose answer was lost.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use folo_server::api::*;
use folo_server::store::MetricsQuery;
use folo_server::{Options, RealStore, Server, Store, Token};
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::harness::{
    Live, busy_once, dataset, epochs, fast, finished, log_of, logged, lose_answer, no_gpu, phase, resume,
    run_of, until,
};

#[test]
fn a_remote_server_is_used_as_it_is_and_keeps_the_dataset() {
    no_gpu();
    let t = TestDir::new("cloud-remote");
    let worker = Arc::new(RealStore::open(t.join("worker")).recheck_every(Duration::ZERO).device("cpu"));
    let options = Options { token: Token::new("remote-token-0123456789"), ..Options::default() };
    let live = Live::start(Server::new(worker.clone(), options).router());
    let store = Arc::new(
        RealStore::open_with_cloud(t.join("home"), folo_cloud::providers(), fast())
            .recheck_every(Duration::ZERO),
    );
    let ds = dataset(&store);
    let values = BTreeMap::from([
        ("url".to_string(), live.url.clone()),
        ("token".to_string(), "remote-token-0123456789".to_string()),
    ]);
    let conn =
        store.connect_cloud(ConnectCloud { provider: "remote".into(), name: String::new(), values }).unwrap();
    assert_eq!(conn.values["token"], "••••6789");
    let offers = store.cloud_offers(&conn.id, false).unwrap().offers;
    assert_eq!((offers.len(), offers[0].id.as_str(), offers[0].price_per_hour), (1, "cpu", None));
    let config: TrainConfig = {
        serde_json::from_value(json!({ "dataset_id": ds.id, "epochs": 1, "batch": 4, "imgsz": 64,
                                       "compute": { "connection_id": conn.id, "offer_id": "cpu" } }))
        .unwrap()
    };
    for n in 0..2 {
        let run = store.create_run(config.clone()).unwrap();
        assert_eq!(phase(&run), CloudPhase::Booting, "nothing to provision");
        let done = finished(&store, &run.id);
        assert_eq!(done.status, RunStatus::Completed, "{:?}", log_of(&store, &run.id));
        let c = done.cloud.unwrap();
        assert!(c.instance_id.is_none() && c.cost_usd.is_none());
        assert_eq!(logged(&store, &run.id, "the worker already has this dataset"), n == 1);
    }
    assert!(store.cloud_instances().unwrap().is_empty(), "no ledger record");
    assert_eq!((worker.list_datasets().unwrap().len(), worker.list_runs().unwrap().len()), (1, 2));
    drop(store);
    drop(live);
}

#[test]
fn a_run_resumed_on_a_remote_server_does_not_take_its_old_worker_run_for_the_new_one() {
    no_gpu();
    let t = TestDir::new("cloud-remote-resume");
    let worker = Arc::new(RealStore::open(t.join("worker")).recheck_every(Duration::ZERO).device("cpu"));
    let options = Options { token: Token::new("remote-token-0123456789"), ..Options::default() };
    let busy = Arc::new(AtomicBool::new(false));
    let router = Server::new(worker.clone(), options)
        .router()
        .layer(axum::middleware::from_fn_with_state(busy.clone(), busy_once));
    let live = Live::start(router);
    let store = Arc::new(
        RealStore::open_with_cloud(t.join("home"), folo_cloud::providers(), fast())
            .recheck_every(Duration::ZERO),
    );
    let ds = dataset(&store);
    let values = BTreeMap::from([
        ("url".to_string(), live.url.clone()),
        ("token".to_string(), "remote-token-0123456789".to_string()),
    ]);
    let conn =
        store.connect_cloud(ConnectCloud { provider: "remote".into(), name: String::new(), values }).unwrap();
    let config: TrainConfig = serde_json::from_value(json!({
        "dataset_id": ds.id, "epochs": 2, "batch": 2, "imgsz": 64, "augment": { "close_mosaic": 1 },
        "compute": { "connection_id": conn.id, "offer_id": "cpu" }
    }))
    .unwrap();
    // Stopped after its first epoch: the worker keeps that run, stopped, under the run's name.
    worker.hold_runs_after_epochs(true);
    let run = store.create_run(config).unwrap();
    until("epoch 1 here", || run_of(&store, &run.id).best);
    store.stop_run(&run.id).unwrap();
    assert_eq!(finished(&store, &run.id).status, RunStatus::Stopped);
    worker.hold_runs_after_epochs(false);

    // Resumed there, and the first request for the new worker run never reaches the worker.
    busy.store(true, Ordering::SeqCst);
    resume(&store, &run.id, json!({}));
    let done = finished(&store, &run.id);
    assert!(!busy.load(Ordering::SeqCst), "the request was refused once");
    assert_eq!(done.status, RunStatus::Completed, "{:?}", log_of(&store, &run.id));
    assert_eq!(worker.list_runs().unwrap().len(), 2, "a new worker run");
    let m = store.run_metrics(&run.id, &MetricsQuery::default()).unwrap();
    assert_eq!(epochs(&m), [1, 2]);
    drop(store);
    drop(live);
}

/// A control plane connected to a `remote` worker that loses its answer to the next `POST /api/runs` while
/// `lose` is set ([`lose_answer`]).
struct Lossy {
    store: Arc<RealStore>,
    worker: Arc<RealStore>,
    lose: Arc<AtomicBool>,
    ds: Dataset,
    conn: String,
    home: PathBuf,
    _live: Live,
}

impl Lossy {
    pub(super) fn new(t: &TestDir) -> Lossy {
        let worker = Arc::new(RealStore::open(t.join("worker")).recheck_every(Duration::ZERO).device("cpu"));
        let options = Options { token: Token::new("remote-token-0123456789"), ..Options::default() };
        let lose = Arc::new(AtomicBool::new(false));
        let router = Server::new(worker.clone(), options)
            .router()
            .layer(axum::middleware::from_fn_with_state((lose.clone(), worker.clone()), lose_answer));
        let live = Live::start(router);
        let home = t.join("home");
        let store = Arc::new(
            RealStore::open_with_cloud(home.clone(), folo_cloud::providers(), fast())
                .recheck_every(Duration::ZERO),
        );
        let ds = dataset(&store);
        let values = BTreeMap::from([
            ("url".to_string(), live.url.clone()),
            ("token".to_string(), "remote-token-0123456789".to_string()),
        ]);
        let conn = store
            .connect_cloud(ConnectCloud { provider: "remote".into(), name: String::new(), values })
            .unwrap()
            .id;
        Lossy { store, worker, lose, ds, conn, home, _live: live }
    }

    /// A one-epoch run there, from `weights`.
    fn config(&self, weights: Option<&str>) -> TrainConfig {
        serde_json::from_value(json!({
            "dataset_id": self.ds.id, "epochs": 1, "batch": 2, "imgsz": 64, "augment": { "close_mosaic": 1 },
            "weights": weights, "compute": { "connection_id": self.conn, "offer_id": "cpu" }
        }))
        .unwrap()
    }
}

#[test]
fn a_worker_run_that_completed_before_its_lost_answer_was_found_is_not_trained_again() {
    no_gpu();
    let t = TestDir::new("cloud-lost-start");
    let l = Lossy::new(&t);
    // The worker took the run and it completed there before anyone asked again: it is that run, not a second.
    l.lose.store(true, Ordering::SeqCst);
    let run = l.store.create_run(l.config(None)).unwrap();
    let done = finished(&l.store, &run.id);
    assert!(!l.lose.load(Ordering::SeqCst), "the answer was lost once");
    assert_eq!(done.status, RunStatus::Completed, "{:?}", log_of(&l.store, &run.id));
    assert_eq!(l.worker.list_runs().unwrap().len(), 1, "trained once");
    assert_eq!(epochs(&l.store.run_metrics(&run.id, &MetricsQuery::default()).unwrap()), [1]);
}

#[test]
fn a_worker_run_that_failed_before_its_lost_answer_was_found_is_not_posted_again() {
    no_gpu();
    let t = TestDir::new("cloud-lost-failed");
    let l = Lossy::new(&t);
    let (store, worker, home) = (&l.store, &l.worker, &l.home);
    let run = store.create_run(l.config(None)).unwrap();
    assert_eq!(finished(store, &run.id).status, RunStatus::Completed);
    // Weights that list (their header reads) and do not load (a tensor of the model is not there under its name):
    // the worker takes the run, and it fails there.
    let best = std::fs::read(home.join("runs").join(&run.id).join("weights").join("best.folo")).unwrap();
    let n = u64::from_le_bytes(best[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&best[8..8 + n]).unwrap();
    let tensors: Value = serde_json::from_str(header.trim_end()).unwrap();
    let name = tensors.as_object().unwrap().keys().find(|k| k.starts_with("backbone.")).unwrap().clone();
    let renamed = format!("{}~", &name[..name.len() - 1]);
    let header = header.replacen(&format!("\"{name}\""), &format!("\"{renamed}\""), 1);
    let file = [&best[..8], header.as_bytes(), &best[8 + n..]].concat();
    std::fs::create_dir_all(home.join("models")).unwrap();
    std::fs::write(home.join("models").join("broken.folo"), file).unwrap();
    let models = store.list_models().unwrap();
    let broken =
        models.iter().find(|m| m.name == "broken").unwrap_or_else(|| panic!("{models:?}")).id.clone();
    l.lose.store(true, Ordering::SeqCst);
    let run = store.create_run(l.config(Some(&broken))).unwrap();
    let done = finished(store, &run.id);
    assert!(!l.lose.load(Ordering::SeqCst), "the answer was lost once");
    let w = worker.list_runs().unwrap();
    assert_eq!(w.len(), 2, "the failed run is not posted again: {w:?}");
    let failed: Vec<&Run> = w.iter().filter(|r| r.status == RunStatus::Failed).collect();
    assert_eq!(failed.len(), 1, "{w:?}");
    assert_eq!(
        (done.status, &done.error_code),
        (RunStatus::Failed, &failed[0].error_code),
        "{:?}",
        log_of(store, &run.id)
    );
}
