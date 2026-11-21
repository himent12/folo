//! The benchmark tools end to end on a synthetic dataset: in process on a real store and through a `folo serve` on
//! loopback (proxy mode), with the same calls, plus the mock (which has no benchmarks) and argument checks. A
//! model's entry is run by the end-to-end tests (`crates/folo-cli/tests/mcp/real_store.rs`), which can write one.

use std::path::Path;
use std::sync::Arc;

use folo_api::SyntheticDataset;
use folo_server::{RealStore, Store};
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::{Progress, find};
use crate::proxy::HttpStore;
use crate::server::FoloMcp;
use crate::{Options, errors::Error};

fn options(data_dir: &Path) -> Options {
    Options {
        data_dir: data_dir.to_path_buf(),
        read_only: false,
        confine: false,
        device: "cpu".into(),
        published_weights: None,
    }
}

/// A store on `home` with a synthetic dataset of 20 images of circles and squares (4 in val); its id.
fn store(home: &Path) -> (Arc<RealStore>, String) {
    let store = Arc::new(RealStore::open(home.to_path_buf()).device("cpu"));
    let req = SyntheticDataset {
        name: "shapes".into(),
        images: 20,
        classes: vec!["circle".into(), "square".into()],
        imgsz: 64,
        seed: 3,
    };
    let ds = store.create_synthetic(req).unwrap();
    (store, ds.id)
}

async fn call(s: &FoloMcp, tool: &str, args: Value) -> Result<Value, Error> {
    let args = args.as_object().unwrap().clone();
    Ok(s.inner.call(find(tool).unwrap(), args, &mut Progress::none()).await?.value)
}

async fn ok(s: &FoloMcp, tool: &str, args: Value) -> Value {
    call(s, tool, args).await.unwrap_or_else(|e| panic!("{tool}: {} {}", e.body.code, e.body.message))
}

async fn err(s: &FoloMcp, tool: &str, args: Value) -> Error {
    call(s, tool, args).await.err().unwrap_or_else(|| panic!("{tool} should fail"))
}

/// The truth of benchmark `id`'s unchanged images as predictions, one record a line, in `file`.
fn truth(store: &RealStore, id: &str, file: &Path) {
    let page = store.benches().unwrap().images(id, 0, 1000, None).unwrap();
    let lines: Vec<String> = page
        .images
        .iter()
        .map(|im| {
            let boxes: Vec<Value> = im
                .boxes
                .iter()
                .map(|b| json!({ "class": b.class, "score": 0.9, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2 }))
                .collect();
            json!({ "image": im.id, "boxes": boxes }).to_string()
        })
        .collect();
    std::fs::write(file, lines.join("\n")).unwrap();
}

/// The journey on `s` (whose data dir is `home`; the benchmarks are `store`'s): the catalog, a benchmark from
/// `dataset`, an entry of predictions from a file, its score, the leaderboard, the refusals, and the delete.
async fn journey(s: &FoloMcp, store: &RealStore, dataset: &str, home: &Path) {
    let catalog = ok(s, "bench_catalog", json!({})).await;
    assert!(catalog["conditions"].as_array().unwrap().len() >= 12, "{catalog}");
    assert_eq!(ok(s, "list_benchmarks", json!({})).await["total"], 0);

    let plan = json!({ "conditions": [{ "condition": "haze", "severities": [2] }] });
    let both = json!({ "name": "x", "dataset_id": dataset, "label_job_id": "lb_1" });
    assert_eq!(err(s, "create_benchmark", both).await.body.code, "invalid_arguments");
    let bad = json!({ "name": "x", "dataset_id": dataset, "plan": { "conditions": [{ "condition": "glare", "severities": [1] }] } });
    let e = err(s, "create_benchmark", bad).await;
    assert!(e.body.code == "invalid_arguments" && e.body.message.contains("glare"), "{e:?}");
    let made =
        ok(s, "create_benchmark", json!({ "name": "shapes", "dataset_id": dataset, "plan": plan })).await;
    let b = made["id"].as_str().unwrap().to_string();
    assert_eq!((made["images"].as_u64(), made["conditions"].as_u64()), (Some(4), Some(1)), "{made}");
    // A plan argument is the plan's conditions only (no agent-design fields).
    let extra = json!({ "name": "x", "dataset_id": dataset, "plan": { "conditions": [], "rationale": "r" } });
    assert_eq!(err(s, "create_benchmark", extra).await.body.code, "invalid_arguments");

    // A file of predictions, relative to the data dir, read by this process and scored at once.
    truth(store, &b, &home.join("truth.jsonl"));
    let args = |more: Value| {
        let mut a = json!({ "benchmark_id": b, "predictions_file": "truth.jsonl" });
        a.as_object_mut().unwrap().extend(more.as_object().unwrap().clone());
        a
    };
    assert_eq!(err(s, "start_bench_entry", args(json!({}))).await.body.code, "invalid_arguments");
    let e = err(s, "start_bench_entry", args(json!({ "name": "t", "model_id": "m_1" }))).await;
    assert_eq!(e.body.code, "invalid_arguments");
    let started = ok(s, "start_bench_entry", args(json!({ "name": "truth" }))).await;
    assert_eq!((started["status"].as_str(), started["done"].as_u64()), (Some("done"), Some(4)), "{started}");
    let e_id = started["entry_id"].as_str().unwrap().to_string();
    let ids = json!({ "benchmark_id": b, "entry_id": e_id });
    let entry = ok(s, "bench_entry", ids.clone()).await;
    assert_eq!(entry["result"]["accuracy"]["map50_95"].as_f64(), Some(1.0), "{entry}");
    assert_eq!(entry["result"]["score"], started["score"], "{entry}");
    // The condition was left out: robustness is not measured, and the score is partial.
    assert_eq!(
        (entry["result"]["partial"].as_bool(), entry["result"]["axes"][1]["value"].clone()),
        (Some(true), Value::Null)
    );
    let long = json!({ "benchmark_id": b, "entry_id": e_id, "wait": 51 });
    assert_eq!(err(s, "bench_entry", long).await.body.code, "invalid_arguments");
    let e = err(s, "stop_bench_entry", ids).await;
    assert!(
        e.body.code == "bench_entry_not_active"
            && e.body.hint.as_deref().unwrap_or_default().contains("bench_entry"),
        "{e:?}"
    );
    let e = err(s, "start_bench_entry", json!({ "benchmark_id": b, "model_id": "m_nope" })).await;
    assert!(
        e.body.code == "model_not_found"
            && e.body.hint.as_deref().unwrap_or_default().contains("list_models"),
        "{e:?}"
    );

    let shown = ok(s, "get_benchmark", json!({ "benchmark_id": b })).await;
    assert_eq!(shown["leaderboard"].as_array().unwrap().len(), 1, "{shown}");
    assert_eq!(
        (shown["leaderboard"][0]["name"].as_str(), shown["leaderboard"][0]["stale"].as_bool()),
        (Some("truth"), Some(false))
    );
    assert_eq!(ok(s, "list_benchmarks", json!({})).await["items"][0]["entries"], 1);
    assert_eq!(ok(s, "delete_benchmark", json!({ "benchmark_id": b })).await["deleted"], true);
    let e = err(s, "get_benchmark", json!({ "benchmark_id": b })).await;
    assert!(
        e.body.code == "bench_not_found"
            && e.body.hint.as_deref().unwrap_or_default().contains("list_benchmarks"),
        "{e:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn benchmarks_in_process() {
    let home = TestDir::new("mcp-bench");
    let (store, dataset) = store(home.path());
    let s = FoloMcp::new(store.clone(), None, options(home.path()));
    journey(&s, &store, &dataset, home.path()).await;
    store.shutdown();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn benchmarks_through_a_server() {
    let home = TestDir::new("mcp-bench-proxy");
    let (store, dataset) = store(&home.join("server"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = folo_server::Server::new(store.clone(), folo_server::Options::default());
    let serving = tokio::spawn(server.serve(listener, async move {
        let _ = stopped.await;
    }));
    let http = Arc::new(HttpStore::new(&url, None));
    let client = home.join("client");
    std::fs::create_dir_all(&client).unwrap();
    let s = FoloMcp::new(http.clone(), Some(http), options(&client));
    journey(&s, &store, &dataset, &client).await;
    let _ = stop.send(());
    serving.await.unwrap().unwrap();
    store.shutdown();
}

#[tokio::test]
async fn the_mock_has_no_benchmarks() {
    let home = TestDir::new("mcp-bench-mock");
    let s = FoloMcp::new(Arc::new(folo_server::MockStore::new()), None, options(home.path()));
    for tool in ["bench_catalog", "list_benchmarks"] {
        assert_eq!(err(&s, tool, json!({})).await.body.code, "not_implemented", "{tool}");
    }
}

/// `get_benchmark` stays within the result limit whatever the names: 80 class names of 64 four-byte characters, a
/// 4,000-character description, 50 entries with long names.
#[test]
fn a_benchmark_fits_the_result_limit() {
    use super::results::{MAX_RESULT_BYTES, json_len};
    let wide = |n: usize| "\u{1F600}".repeat(n);
    let row = |i: usize, name: String| crate::types::BenchEntryRow {
        id: format!("be_{i}"),
        name,
        kind: "model".into(),
        status: folo_api::BenchEntryStatus::Done,
        score: Some(50.0),
        partial: false,
        map50_95: Some(0.5),
        ci95: Some([0.4, 0.6]),
        stale: false,
    };
    let detail = |classes: Vec<String>, description: String, names: &dyn Fn(usize) -> String| {
        crate::types::BenchDetail {
            id: "bn_1".into(),
            name: "b".into(),
            description,
            source: folo_api::BenchSource::Dataset { dataset_id: "ds_1".into(), split: None },
            source_name: "d".into(),
            classes,
            images: 5000,
            boxes: 1,
            plan: vec![],
            seed: 0,
            fingerprint: "f".repeat(64),
            active: false,
            leaderboard: (0..50).map(|i| row(i, names(i))).collect(),
        }
    };
    // A usual benchmark is left as it is.
    let mut small = detail(vec!["cat".into(), "dog".into()], "a pond".into(), &|i| format!("model {i}"));
    assert!(!super::bench::fit_detail(&mut small));
    assert_eq!((small.leaderboard.len(), small.description.as_str()), (50, "a pond"));
    // A long description goes before any row of the leaderboard.
    let mut wordy = detail(vec!["cat".into()], wide(9000), &|i| format!("model {i}"));
    assert!(json_len(&wordy) > MAX_RESULT_BYTES);
    assert!(super::bench::fit_detail(&mut wordy));
    assert_eq!((wordy.leaderboard.len(), wordy.description.chars().count()), (50, 201));
    // A huge one also loses rows, then its class names are cut; every class stays.
    let classes = (0..80).map(|i| format!("{i:02}{}", wide(62))).collect();
    let mut big = detail(classes, wide(4000), &|_| wide(100));
    assert!(json_len(&big) > MAX_RESULT_BYTES);
    assert!(super::bench::fit_detail(&mut big));
    assert!(json_len(&big) <= MAX_RESULT_BYTES - 1024, "{}", json_len(&big));
    assert_eq!(big.classes.len(), 80);
    assert!(big.classes[7].starts_with("07"), "{}", big.classes[7]);
}

/// A server that accepts and never answers (the connection waits in the listener's backlog): the wait ends at its
/// grace with the entry's hint, not after the HTTP client's 10-minute timeout, and leaves no request running.
#[test]
fn a_wait_ends_when_the_server_stops_answering() {
    use std::time::Duration;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let http = HttpStore::new(&url, None).assume_features(vec![folo_api::FEATURE_VISION_BENCH.to_string()]);
    let b = crate::bench::Benchmarks::Http(Arc::new(http));
    let poll = Duration::from_millis(10);
    let e = super::bench::wait_state(&b, "bn_1", "be_1", poll, Duration::ZERO, Duration::from_millis(50))
        .unwrap_err();
    assert_eq!(e.body.code, "server_unreachable", "{e:?}");
    assert!(e.body.hint.as_deref().unwrap_or_default().contains("bench_entry"), "{e:?}");
}

/// `delete_benchmark` is the one tool that removes files, and the only one a client is told is destructive.
#[test]
fn only_delete_benchmark_is_destructive() {
    let destructive: Vec<&str> = super::TOOLS
        .iter()
        .filter(|t| t.tool().annotations.is_some_and(|a| a.destructive_hint == Some(true)))
        .map(|t| t.name)
        .collect();
    assert_eq!(destructive, ["delete_benchmark"]);
}
