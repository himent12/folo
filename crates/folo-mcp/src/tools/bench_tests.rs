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
    // Class names that alone pass the limit are cut, every class stays, and no row is lost to them.
    let classes = (0..80).map(|i| format!("{i:02}{}", wide(148))).collect();
    let mut named = detail(classes, "a pond".into(), &|i| format!("model {i}"));
    assert!(json_len(&named) > MAX_RESULT_BYTES);
    assert!(super::bench::fit_detail(&mut named));
    assert!(json_len(&named) <= MAX_RESULT_BYTES - 1024, "{}", json_len(&named));
    assert_eq!((named.classes.len(), named.leaderboard.len()), (80, 50));
    assert!((0..80).all(|i| named.classes[i].starts_with(&format!("{i:02}"))));
    // Long entry names are cut before any row goes.
    let classes = (0..80).map(|i| format!("{i:02}{}", wide(22))).collect();
    let mut ranked = detail(classes, "a pond".into(), &|_| wide(100));
    assert!(json_len(&ranked) > MAX_RESULT_BYTES);
    assert!(super::bench::fit_detail(&mut ranked));
    assert!(json_len(&ranked) <= MAX_RESULT_BYTES - 1024, "{}", json_len(&ranked));
    assert_eq!((ranked.leaderboard.len(), ranked.leaderboard[49].name.chars().count()), (50, 41));
    // A huge one (control characters take six bytes each in JSON) has all its names cut, then loses rows from the end.
    let control = |n: usize| "\u{1}".repeat(n);
    let classes = (0..80).map(|i| format!("{i:02}{}", control(62))).collect();
    let mut big = detail(classes, control(4000), &|_| control(100));
    big.name = control(500);
    big.source_name = control(500);
    assert!(json_len(&big) > MAX_RESULT_BYTES);
    assert!(super::bench::fit_detail(&mut big));
    assert!(json_len(&big) <= MAX_RESULT_BYTES - 1024, "{}", json_len(&big));
    assert_eq!(big.classes.len(), 80);
    assert!(big.classes[7].starts_with("07"), "{}", big.classes[7]);
    assert!(!big.leaderboard.is_empty() && big.leaderboard.len() < 50, "{}", big.leaderboard.len());
}

/// `bench_entry` stays within the result limit with every class missing under a very long name.
#[test]
fn an_entry_fits_the_result_limit() {
    use super::results::{MAX_RESULT_BYTES, json_len};
    let wide = |n: usize| "\u{1F600}".repeat(n);
    let state = |missing_classes: Vec<String>, error: Option<String>| crate::types::BenchEntryState {
        benchmark_id: "bn_1".into(),
        entry_id: "be_1".into(),
        name: "m".into(),
        status: folo_api::BenchEntryStatus::Failed,
        done: 0,
        total: 8,
        condition: None,
        severity: None,
        images: 4,
        conditions: true,
        missing_classes,
        error,
        hint: None,
        result: None,
        summary_score: None,
    };
    let mut small = state(vec!["cat".into()], Some("no such file".into()));
    assert!(!super::bench::fit_state(&mut small));
    assert_eq!((small.missing_classes[0].as_str(), small.error.as_deref()), ("cat", Some("no such file")));
    let classes = (0..80).map(|i| format!("{i:02}{}", wide(148))).collect();
    let mut big = state(classes, Some(wide(20_000)));
    assert!(json_len(&big) > MAX_RESULT_BYTES);
    assert!(super::bench::fit_state(&mut big));
    assert!(json_len(&big) <= MAX_RESULT_BYTES - 1024, "{}", json_len(&big));
    assert_eq!(big.missing_classes.len(), 80);
    assert!((0..80).all(|i| big.missing_classes[i].starts_with(&format!("{i:02}"))));
    // The summary keeps the way to the whole text, whatever it had to shorten for it.
    big.name = wide(100);
    let summary = super::bench::fitted_summary(&big, true);
    assert!(summary.chars().count() <= super::results::MAX_SUMMARY_CHARS, "{summary}");
    assert!(summary.ends_with("/entries/{entry_id} has it whole)"), "{summary}");

    // Large-vocabulary benchmarks (>1000 missing classes) trim trailing `missing_classes` so `bench_entry`
    // never fails with `413 result_too_large`.
    let huge_classes = (0..1_500).map(|i| format!("category_name_long_{i:04}")).collect();
    let mut huge = state(huge_classes, None);
    assert!(json_len(&huge) > MAX_RESULT_BYTES);
    assert!(super::bench::fit_state(&mut huge));
    assert!(json_len(&huge) <= MAX_RESULT_BYTES - 1024, "{}", json_len(&huge));
    assert!(!huge.missing_classes.is_empty() && huge.missing_classes.len() < 1_500);
    let huge_summary = super::bench::fitted_summary(&huge, true);
    assert!(huge_summary.ends_with("/entries/{entry_id} has it whole)"), "{huge_summary}");
    assert!(super::results::out(&huge, huge_summary).is_ok());
}

/// With `--confine` every benchmark tool is refused, in process and through a server, before it touches a file or
/// the network (the address below has no server). In embedded mode, `build` does not even open `<data_dir>/bench/`.
#[tokio::test]
async fn a_confined_session_refuses_the_benchmark_tools() {
    let home = TestDir::new("mcp-bench-confined");
    // Place an incomplete benchmark dir (no `bench.json`) that `Benches::open` would delete on startup.
    let canary = home.join("bench").join("bn_unfinished");
    std::fs::create_dir_all(&canary).unwrap();
    let confined = || Options { confine: true, ..options(home.path()) };
    let real_store = Arc::new(crate::open_real_store(&confined()));
    assert!(canary.is_dir(), "`folo mcp --confine` must not read or clean up the `bench/` folder at startup");
    assert_eq!(real_store.benches().err().unwrap().body.code, "not_supported");
    let built = FoloMcp::new(real_store, None, confined());
    let http = Arc::new(HttpStore::new("http://127.0.0.1:9", None));
    let through = FoloMcp::new(http.clone(), Some(http), confined());
    let bench = json!({ "benchmark_id": "bn_1" });
    let entry = json!({ "benchmark_id": "bn_1", "entry_id": "be_1" });
    for s in [&built, &through] {
        for (tool, args) in [
            ("bench_catalog", json!({})),
            ("list_benchmarks", json!({})),
            ("get_benchmark", bench.clone()),
            ("create_benchmark", json!({ "name": "x", "dataset_id": "ds_1" })),
            ("delete_benchmark", bench.clone()),
            (
                "start_bench_entry",
                json!({ "benchmark_id": "bn_1", "predictions_file": "p.jsonl", "name": "p" }),
            ),
            ("bench_entry", entry.clone()),
            ("stop_bench_entry", entry.clone()),
            (
                "compare_bench_entries",
                json!({ "benchmark_id": "bn_1", "baseline_entry_id": "be_1", "candidate_entry_id": "be_2" }),
            ),
        ] {
            assert_eq!(err(s, tool, args).await.body.code, "not_supported", "{tool}");
        }
    }
    assert!(canary.is_dir(), "refused benchmark calls must not touch `bench/` either");
    built.shutdown_store().await;
    // With --read-only too, a mutating tool is refused as read-only first: neither reaches a file.
    let both = Options { read_only: true, ..confined() };
    let both = FoloMcp::new(Arc::new(folo_server::MockStore::new()), None, both);
    let create = json!({ "name": "x", "dataset_id": "ds_1" });
    assert_eq!(err(&both, "create_benchmark", create).await.body.code, "read_only");
    assert_eq!(err(&both, "get_benchmark", bench).await.body.code, "not_supported");
}

/// Runs `f` on drop so an unwind or timeout always releases a fake server or worker.
struct Defer<F: FnOnce()>(Option<F>);

impl<F: FnOnce()> Drop for Defer<F> {
    fn drop(&mut self) {
        if let Some(f) = self.0.take() {
            f();
        }
    }
}

/// Runs `call` on a thread and fails the test if it has not answered in a minute (a call that lost its deadline would
/// hold far longer); `release` runs inside the scope before the worker is joined (on timeout or panic) so no thread
/// can outlive the test or deadlock teardown.
fn bounded<T: Send>(call: impl FnOnce() -> T + Send, release: impl FnOnce()) -> T {
    std::thread::scope(|scope| {
        let mut release = Defer(Some(release));
        let (done, answer) = std::sync::mpsc::channel();
        scope.spawn(move || {
            let _ = done.send(call());
        });
        match answer.recv_timeout(std::time::Duration::from_secs(60)) {
            Ok(r) => {
                release.0.take();
                r
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                panic!("the bounded call panicked");
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if let Some(f) = release.0.take() {
                    f();
                }
                panic!("the call did not end by its deadline");
            }
        }
    })
}

/// A server that stops answering ends the wait at its grace, with the entry's hint: at the first request of a session
/// (the feature probe) and, with the features known, at the request for the entry.
#[test]
fn a_wait_ends_when_the_server_stops_answering() {
    use std::time::Duration;
    for known in [false, true] {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let http = HttpStore::new(&url, None);
        let http =
            if known { http.assume_features(vec![folo_api::FEATURE_VISION_BENCH.to_string()]) } else { http };
        let b = crate::bench::Benchmarks::Http(Arc::new(http));
        let (poll, grace) = (Duration::from_millis(10), Duration::from_millis(50));
        // The connections wait in the listener's backlog; closing it ends a call still held.
        let r = bounded(
            || super::bench::wait_state(&b, "bn_1", "be_1", poll, Duration::ZERO, grace),
            move || drop(listener),
        );
        let e = r.unwrap_err();
        assert_eq!(e.body.code, "server_unreachable", "known features: {known}, {e:?}");
        assert!(e.body.hint.as_deref().unwrap_or_default().contains("bench_entry"), "{e:?}");
    }
}

/// Answers every request with `running` for the first `active` requests and then `ended`, recording the total
/// requests in `total_reads` and requests after the entry ended in `after_end`.
fn serve_entry(
    listener: std::net::TcpListener,
    (running, active): (&str, usize),
    ended: &str,
    total_reads: &std::sync::atomic::AtomicUsize,
    after_end: &std::sync::atomic::AtomicUsize,
    closing: &std::sync::atomic::AtomicBool,
) {
    use std::io::{BufRead, Write};
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::Duration;
    let mut reads = 0_usize;
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { return };
        if closing.load(Relaxed) {
            return;
        }
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
        let mut head = std::io::BufReader::new(stream.try_clone().unwrap());
        let mut line = String::new();
        while head.read_line(&mut line).is_ok_and(|n| n > 0) && line != "\r\n" {
            line.clear();
        }
        reads += 1;
        total_reads.store(reads, Relaxed);
        after_end.store(reads.saturating_sub(active + 1), Relaxed);
        let body = if reads <= active { running } else { ended };
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
    }
}

/// `wait` on a real store and through a fake server: `wait: 0` polls once and returns right away (distinct from
/// `wait: 1`), `wait: 1` holds for its length, and an entry that ends early during a long wait returns promptly
/// without asking the server again.
#[test]
fn a_wait_follows_an_entry_until_it_ends() {
    use folo_api::{BenchContestantSpec, BenchEntryStatus, BenchPrediction, CreateBench, StartBenchEntry};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};
    let home = TestDir::new("mcp-bench-wait");
    let (store, dataset) = store(home.path());
    let _shutdown = Defer(Some(|| store.shutdown()));
    let b = crate::bench::Benchmarks::Store(store.clone());
    let req = CreateBench {
        name: "shapes".into(),
        description: String::new(),
        source: folo_api::BenchSource::Dataset { dataset_id: dataset, split: None },
        max_images: None,
        seed: 0,
        plan: None,
    };
    let bench = b.create_bench(req).unwrap().id;
    let page = store.benches().unwrap().images(&bench, 0, 1000, None).unwrap();
    let predictions = page
        .images
        .iter()
        .map(|im| BenchPrediction { image: im.id.clone(), condition: None, severity: None, boxes: vec![] })
        .collect();
    let contestant = BenchContestantSpec::External { scored: true, open: true, predictions };
    let req = StartBenchEntry { contestant, name: Some("open".into()), max_images: None, conditions: None };
    let entry = b.start_entry(&bench, req).unwrap().id;

    let s = FoloMcp::new(store.clone(), None, options(home.path()));
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let args = |wait: u32| json!({ "benchmark_id": bench, "entry_id": entry, "wait": wait });
    // Finishing or stopping (and shutting down the store) guarantees an in-flight call terminates even if finish fails.
    let release = || {
        if b.finish(&bench, &entry).is_err() {
            let _ = b.stop_entry(&bench, &entry);
        }
        store.shutdown();
    };
    let now = bounded(|| rt.block_on(ok(&s, "bench_entry", args(0))), release);
    assert_eq!(now["status"], "running", "{now}");
    let start = Instant::now();
    let held = bounded(|| rt.block_on(ok(&s, "bench_entry", args(1))), release);
    assert_eq!(held["status"], "running", "{held}");
    // The tool polls every 100 ms and does not start a sleep that would pass the end of its wait.
    assert!(start.elapsed() >= Duration::from_millis(900), "{:?}", start.elapsed());

    // The store's own JSON of the entry, open and then stopped, served by a fake server.
    let running = serde_json::to_string(&b.entry(&bench, &entry).unwrap()).unwrap();
    let stopped = b.stop_entry(&bench, &entry).unwrap();
    assert_eq!(stopped.status, BenchEntryStatus::Stopped);
    let stopped = serde_json::to_string(&stopped).unwrap();
    drop(_shutdown);

    // Zero-wait through the fake server makes exactly one read (proving `wait: 0` does not poll like `wait: 1`).
    let run_fake = |active: usize,
                    wait: Duration|
     -> (Result<crate::types::BenchEntryState, Error>, usize, usize, Duration) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let http = HttpStore::new(&format!("http://{addr}"), None)
            .assume_features(vec![folo_api::FEATURE_VISION_BENCH.to_string()]);
        let via = crate::bench::Benchmarks::Http(Arc::new(http));
        let (poll, grace) = (Duration::from_millis(5), Duration::from_secs(1));
        let total_reads = AtomicUsize::new(0);
        let after_end = AtomicUsize::new(0);
        let closing = AtomicBool::new(false);
        let close = || {
            closing.store(true, Ordering::Relaxed);
            let _ = std::net::TcpStream::connect(addr);
        };
        let t0 = Instant::now();
        let r = std::thread::scope(|scope| {
            // Drop guard is created inside the scope so even if creating the worker thread or running `bounded`
            // panics, the fake server's accept loop is always released before `scope` joins it.
            scope.spawn(|| {
                serve_entry(listener, (&running, active), &stopped, &total_reads, &after_end, &closing)
            });
            let _close = Defer(Some(close));
            bounded(|| super::bench::wait_state(&via, &bench, &entry, poll, wait, grace), close)
        });
        (r, total_reads.load(Ordering::Relaxed), after_end.load(Ordering::Relaxed), t0.elapsed())
    };

    let (zero_res, zero_reads, _, _) = run_fake(10, Duration::ZERO);
    assert_eq!(zero_res.unwrap().status, BenchEntryStatus::Running);
    assert_eq!(zero_reads, 1, "wait: 0 must read the entry exactly once");

    let long = Duration::from_secs(30);
    let (ended, _, after_end, elapsed) = run_fake(3, long);
    assert_eq!(ended.unwrap().status, BenchEntryStatus::Stopped);
    assert_eq!(after_end, 0, "the wait asked again after the entry ended");
    assert!(
        elapsed < Duration::from_secs(5),
        "early completion must return promptly instead of holding until the {long:?} deadline ({elapsed:?})"
    );
}

/// `delete_benchmark` removes frozen files and entry results, so it is marked destructive (the full
/// annotation table is checked in `tools::tests::tool_annotations_match_tool_semantics`).
#[test]
fn delete_benchmark_is_marked_destructive() {
    let ann = find("delete_benchmark").unwrap().tool().annotations.unwrap();
    assert_eq!(ann.destructive_hint, Some(true));
}

/// `create_benchmark` reports the benchmark's true class count (>100 from a server with larger vocabularies),
/// matching `list_benchmarks` rather than truncating at `BenchDetail`'s 100-class preview cap.
#[tokio::test]
async fn create_benchmark_reports_true_num_classes_above_100() {
    use std::io::{BufRead, Read, Write};

    let home = TestDir::new("mcp-bench-105-classes");
    let classes: Vec<String> = (0..105).map(|i| format!("cls_{i}")).collect();
    let bench = json!({
        "id": "bn_many",
        "name": "many-classes",
        "description": "",
        "classes": classes,
        "created": "2026-11-20T00:00:00Z",
        "source": { "kind": "dataset", "dataset_id": "ds_1", "split": "val" },
        "source_name": "many",
        "images": 1,
        "boxes": 1,
        "class_boxes": vec![0u64; 105],
        "plan": { "conditions": [] },
        "seed": 0,
        "fingerprint": "fp",
        "entries": 0,
        "best": null,
        "active": false
    });
    let created_body = bench.to_string();
    let listed_body = json!({ "benchmarks": [bench] }).to_string();

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let srv = std::thread::spawn(move || {
        for (status, body) in [("201 Created", created_body.as_str()), ("200 OK", listed_body.as_str())] {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
            let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
            let mut content_len = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" || line == "\n" {
                    break;
                }
                if let Some((k, v)) = line.split_once(':')
                    && k.trim().eq_ignore_ascii_case("content-length")
                {
                    content_len = v.trim().parse().unwrap_or(0);
                }
            }
            if content_len > 0 {
                let mut discard = vec![0u8; content_len];
                let _ = reader.read_exact(&mut discard);
            }
            let _ = write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });

    let http = Arc::new(
        HttpStore::new(&format!("http://{addr}"), None)
            .assume_features(vec![folo_api::FEATURE_VISION_BENCH.to_string()]),
    );
    let s = FoloMcp::new(http.clone(), Some(http), options(home.path()));
    let made = ok(
        &s,
        "create_benchmark",
        json!({
            "name": "many-classes",
            "dataset_id": "ds_1",
            "plan": { "conditions": [] }
        }),
    )
    .await;
    assert_eq!(made["num_classes"], 105, "{made}");
    let listed = ok(&s, "list_benchmarks", json!({})).await;
    assert_eq!(listed["items"][0]["num_classes"], 105, "{listed}");
    srv.join().unwrap();
}

/// `start_bench_entry` keeps the server's headline score in `BenchEntryBrief` even when fetching the separate
/// `/result` breakdown fails after `/finish` succeeded.
#[tokio::test]
async fn start_bench_entry_preserves_score_when_result_fetch_fails() {
    use std::io::{BufRead, Read, Write};

    let home = TestDir::new("mcp-bench-score-fallback");
    let pred_file = home.join("preds.jsonl");
    std::fs::write(&pred_file, r#"{"image":"1","boxes":[]}"#).unwrap();

    let open_entry = json!({
        "id": "be_1",
        "bench_id": "bn_1",
        "name": "ext",
        "contestant": { "kind": "external" },
        "status": "running",
        "created": "2026-11-20T00:00:00Z",
        "finished": null,
        "fingerprint": "fp",
        "images": 1,
        "conditions": false,
        "scored": true,
        "progress": { "done": 1, "total": 1, "condition": null, "severity": null },
        "score": null,
        "map50_95": null,
        "ci95": null,
        "speed": null,
        "missing_classes": [],
        "error": null,
        "hint": null
    })
    .to_string();
    let done_entry = json!({
        "id": "be_1",
        "bench_id": "bn_1",
        "name": "ext",
        "contestant": { "kind": "external" },
        "status": "done",
        "created": "2026-11-20T00:00:00Z",
        "finished": "2026-11-20T00:00:01Z",
        "fingerprint": "fp",
        "images": 1,
        "conditions": false,
        "scored": true,
        "progress": { "done": 1, "total": 1, "condition": null, "severity": null },
        "score": { "total": 78.4, "partial": true, "axes": [] },
        "map50_95": 0.78,
        "ci95": [0.7, 0.85],
        "speed": null,
        "missing_classes": [],
        "error": null,
        "hint": null
    })
    .to_string();
    let err_503 = r#"{"error":{"code":"internal","message":"result temporarily unavailable","hint":null}}"#;

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let srv = std::thread::spawn(move || {
        for (status, body) in [
            ("201 Created", open_entry.as_str()),
            ("200 OK", done_entry.as_str()),
            ("503 Service Unavailable", err_503),
        ] {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
            let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
            let mut content_len = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" || line == "\n" {
                    break;
                }
                if let Some((k, v)) = line.split_once(':')
                    && k.trim().eq_ignore_ascii_case("content-length")
                {
                    content_len = v.trim().parse().unwrap_or(0);
                }
            }
            if content_len > 0 {
                let mut discard = vec![0u8; content_len];
                let _ = reader.read_exact(&mut discard);
            }
            let _ = write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });

    let http = Arc::new(
        HttpStore::new(&format!("http://{addr}"), None)
            .assume_features(vec![folo_api::FEATURE_VISION_BENCH.to_string()]),
    );
    let s = FoloMcp::new(http.clone(), Some(http), options(home.path()));
    let res = ok(
        &s,
        "start_bench_entry",
        json!({
            "benchmark_id": "bn_1",
            "predictions_file": "preds.jsonl",
            "name": "ext"
        }),
    )
    .await;
    srv.join().unwrap();
    assert_eq!(res["status"], "done", "{res}");
    assert!((res["score"].as_f64().unwrap() - 78.4).abs() < 1e-4, "{res}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn create_benchmark_generated_embedded_proxy_and_old_server() {
    let home = TestDir::new("mcp-bench-generated");
    let store = Arc::new(RealStore::open(home.path().to_path_buf()).device("cpu"));
    let embedded = FoloMcp::new(store.clone(), None, options(home.path()));

    let plan = json!({ "conditions": [{ "condition": "haze", "severities": [2] }] });
    // Conflicting `generated.images` vs `max_images` and `split` with `generated` are rejected.
    let e = err(
        &embedded,
        "create_benchmark",
        json!({ "name": "bad1", "generated": { "images": 4 }, "max_images": 6 }),
    )
    .await;
    assert_eq!(e.body.code, "invalid_arguments");
    let e = err(
        &embedded,
        "create_benchmark",
        json!({ "name": "bad2", "generated": { "images": 4 }, "split": "val" }),
    )
    .await;
    assert_eq!(e.body.code, "invalid_arguments");

    // `create_benchmark` schema enforces mutually exclusive sources, `split` dependency, and generated class bounds.
    let create_schema = crate::tools::find("create_benchmark").unwrap().input_schema();
    assert_eq!(create_schema["oneOf"].as_array().unwrap().len(), 3, "{create_schema:?}");
    assert_eq!(create_schema["dependentRequired"]["split"], json!(["dataset_id"]));
    let classes_schema = &create_schema["$defs"]["GeneratedBenchSourceArgs"]["properties"]["classes"];
    assert_eq!(classes_schema["minItems"], json!(1));
    assert_eq!(classes_schema["maxItems"], json!(8));
    assert_eq!(classes_schema["uniqueItems"], json!(true));

    // `generated.images: 0` blames `images`, not `max_images`.
    let e = err(&embedded, "create_benchmark", json!({ "name": "zero", "generated": { "images": 0 } })).await;
    assert_eq!(e.body.code, "invalid_arguments");
    assert!(e.body.message.contains("images"), "{}", e.body.message);
    assert!(!e.body.message.contains("max_images"), "{}", e.body.message);

    // Embedded generated benchmark creation.
    let g1 = ok(
        &embedded,
        "create_benchmark",
        json!({
            "name": "gen-embedded",
            "generated": { "images": 4, "classes": ["circle", "square"], "imgsz": 64 },
            "seed": 11,
            "plan": plan
        }),
    )
    .await;
    assert_eq!((g1["images"].as_u64(), g1["num_classes"].as_u64()), (Some(4), Some(2)), "{g1}");
    let d1 = ok(&embedded, "get_benchmark", json!({ "benchmark_id": g1["id"] })).await;

    // Proxy mode through a real `folo serve` produces the exact same fingerprint for the same spec & seed.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = folo_server::Server::new(store.clone(), folo_server::Options::default());
    let serving = tokio::spawn(server.serve(listener, async move {
        let _ = stopped.await;
    }));
    let http = Arc::new(HttpStore::new(&url, None));
    let proxied = FoloMcp::new(http.clone(), Some(http), options(home.path()));
    let g2 = ok(
        &proxied,
        "create_benchmark",
        json!({
            "name": "gen-proxied",
            "generated": { "images": 4, "classes": ["circle", "square"], "imgsz": 64 },
            "seed": 11,
            "plan": plan
        }),
    )
    .await;
    let d2 = ok(&proxied, "get_benchmark", json!({ "benchmark_id": g2["id"] })).await;
    assert_eq!(d1["fingerprint"], d2["fingerprint"], "embedded and proxied generated benchmarks must match");
    let _ = stop.send(());
    serving.await.unwrap().unwrap();

    // Older proxied server without `bench_generated_source` fails cleanly with `not_implemented`.
    let old_http = Arc::new(
        HttpStore::new("http://127.0.0.1:9", None)
            .assume_features(vec![folo_api::FEATURE_VISION_BENCH.to_string()]),
    );
    let old_mcp = FoloMcp::new(old_http.clone(), Some(old_http), options(home.path()));
    let e = err(
        &old_mcp,
        "create_benchmark",
        json!({ "name": "gen-old", "generated": { "images": 4, "imgsz": 64 } }),
    )
    .await;
    assert_eq!(e.body.code, "not_implemented");
    store.shutdown();
}
