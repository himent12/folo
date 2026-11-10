//! Restarts and shutdown.

use std::time::Duration;

use folo_server::api::*;
use folo_server::{RealStore, Store};
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::harness::{
    Step, TIMEOUT, bench, dataset, fixture, get, held, loop_ended, model, ok, post, small_plan,
};

#[tokio::test(flavor = "multi_thread")]
async fn entries_and_loops_left_active_fail_at_the_next_start() {
    let t = TestDir::new("bench-restart");
    let home = t.join("home");
    let (bench_id, entry_id) = {
        let f = fixture(&home);
        let app = &f.app;
        let ds = dataset(app, "shapes", 10).await;
        let b = bench(app, json!({ "name": "b", "source": { "kind": "dataset", "dataset_id": ds.id }, "plan": small_plan() })).await;
        // Shutting down ends an agent call in flight: the entry fails as the server stopped.
        f.agent.script(vec![Step::Hold]);
        let e: BenchEntry = ok(
            app,
            post(
                &format!("/api/bench/benchmarks/{}/entries", b.id),
                json!({ "contestant": { "kind": "agent", "agent": "claude" } }),
            ),
        )
        .await;
        f.store.shutdown();
        let s: BenchEntry = ok(app, get(&format!("/api/bench/benchmarks/{}/entries/{}", b.id, e.id))).await;
        assert_eq!(
            (s.status, s.error.as_deref()),
            (BenchEntryStatus::Failed, Some("the server stopped before the entry finished"))
        );
        (b.id, e.id)
    };
    // A process that died left an entry running and a loop running with a finished generation.
    let dir = home.join("bench").join(&bench_id);
    let path = dir.join("entries").join(format!("{entry_id}.json"));
    let mut file: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    file["entry"]["status"] = json!("running");
    file["entry"]["error"] = Value::Null;
    file["entry"]["finished"] = Value::Null;
    std::fs::write(&path, file.to_string()).unwrap();
    let lp = json!({ "version": 1, "loop": {
        "id": "bl_1", "bench_id": bench_id, "name": "l", "status": "running", "phase": "training",
        "created": "2026-11-09T00:00:00Z", "finished": null,
        "config": { "teacher": "m_1", "pool": { "kind": "label_job", "job_id": "lb_1" } },
        "generations": [{ "generation": 0, "labels": null, "run_id": null, "model_id": "m_1", "model_name": "t",
                          "entry_id": entry_id, "score": 40.0, "map50_95": 0.3, "ci95": [0.2, 0.4] }],
        "verdict": null, "error": null } });
    std::fs::create_dir_all(dir.join("loops")).unwrap();
    std::fs::write(dir.join("loops").join("bl_1.json"), lp.to_string()).unwrap();
    std::fs::write(dir.join("loops").join("broken.json"), "{").unwrap();
    // And a benchmark it was making when it died: images copied, no bench.json.
    let unfinished = home.join("bench").join("bn_unfinished");
    std::fs::create_dir_all(unfinished.join("images")).unwrap();
    std::fs::write(unfinished.join("images").join("a.jpg"), b"x").unwrap();

    let f = fixture(&home);
    assert!(!unfinished.exists(), "an unfinished benchmark's folder is swept at start");
    let e: BenchEntry =
        ok(&f.app, get(&format!("/api/bench/benchmarks/{bench_id}/entries/{entry_id}"))).await;
    assert_eq!(
        (e.status, e.error.as_deref()),
        (BenchEntryStatus::Failed, Some("the server stopped before the entry finished"))
    );
    assert!(e.hint.is_some() && e.finished.is_some());
    let l: BenchLoop = ok(&f.app, get(&format!("/api/bench/benchmarks/{bench_id}/loops/bl_1"))).await;
    assert_eq!(
        (l.status, l.phase, l.error.as_deref()),
        (BenchLoopStatus::Failed, None, Some("the server stopped"))
    );
    assert_eq!(l.generations.len(), 1, "its finished generations are kept");
    let loops: BenchLoopList = ok(&f.app, get(&format!("/api/bench/benchmarks/{bench_id}/loops"))).await;
    assert_eq!(loops.loops.len(), 1, "an unreadable loop file is skipped");
    assert!(dir.join("loops").join("broken.json").is_file(), "and never overwritten");
    // Written back: the next start reads them as failed too.
    let saved: Value =
        serde_json::from_slice(&std::fs::read(dir.join("loops").join("bl_1.json")).unwrap()).unwrap();
    assert_eq!(saved["loop"]["status"], "failed");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stop_that_was_accepted_survives_the_server_stopping() {
    let t = TestDir::new("bench-stopping");
    let home = t.join("home");
    let src = |ds: &Dataset| json!({ "kind": "dataset", "dataset_id": ds.id });
    // An entry stopped while it runs is `stopping`; the server stops before its work ends: it is `stopped`, not
    // `failed` ("start it again") as an entry nobody stopped is.
    let (bench_id, entry_id, teacher, pool) = {
        let f = fixture(&home);
        let app = &f.app;
        let ds = dataset(app, "shapes", 20).await;
        let b = bench(app, json!({ "name": "b", "source": src(&ds), "plan": small_plan() })).await;
        let m = model(app, &home, "teacher", &["circle", "square"], 7).await;
        f.store.hold_bench_entries(true);
        let entries = format!("/api/bench/benchmarks/{}/entries", b.id);
        let e: BenchEntry =
            ok(app, post(&entries, json!({ "contestant": { "kind": "model", "model_id": m } }))).await;
        held(app, &b.id, &e.id).await;
        let s: BenchEntry = ok(app, post(&format!("{entries}/{}/stop", e.id), json!({}))).await;
        assert_eq!(s.status, BenchEntryStatus::Stopping);
        f.store.shutdown();
        let s: BenchEntry = ok(app, get(&format!("{entries}/{}", e.id))).await;
        assert_eq!((s.status, s.error.as_deref()), (BenchEntryStatus::Stopped, None), "{s:?}");
        (b.id, e.id, m, json!({ "kind": "dataset", "dataset_id": ds.id, "split": "train" }))
    };
    // A process that died with it `stopping`: it is `stopped` at the next start.
    let dir = home.join("bench").join(&bench_id);
    let path = dir.join("entries").join(format!("{entry_id}.json"));
    let mut file: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    file["entry"]["status"] = json!("stopping");
    file["entry"]["finished"] = Value::Null;
    std::fs::write(&path, file.to_string()).unwrap();
    let loop_id = {
        let f = fixture(&home);
        let app = &f.app;
        let e: BenchEntry =
            ok(app, get(&format!("/api/bench/benchmarks/{bench_id}/entries/{entry_id}"))).await;
        assert_eq!((e.status, e.error.as_deref()), (BenchEntryStatus::Stopped, None), "{e:?}");
        assert!(e.finished.is_some());
        // The same for a loop: stopped while its teacher's entry is held, then the server stops.
        f.store.hold_bench_entries(true);
        let loops = format!("/api/bench/benchmarks/{bench_id}/loops");
        let student = json!({ "epochs": 1, "imgsz": 64, "batch": 4, "device": "cpu" });
        let l: BenchLoop =
            ok(app, post(&loops, json!({ "teacher": teacher, "pool": pool, "student": student }))).await;
        let t0 = std::time::Instant::now();
        let gen0 = loop {
            let now: BenchLoop = ok(app, get(&format!("{loops}/{}", l.id))).await;
            if let Some(id) = now.generations.first().and_then(|g| g.entry_id.clone()) {
                break id;
            }
            assert!(t0.elapsed() < TIMEOUT, "{now:?}");
            tokio::time::sleep(Duration::from_millis(10)).await;
        };
        held(app, &bench_id, &gen0).await;
        let s: BenchLoop = ok(app, post(&format!("{loops}/{}/stop", l.id), json!({}))).await;
        assert_eq!(s.status, BenchLoopStatus::Stopping);
        f.store.shutdown();
        let s: BenchLoop = ok(app, get(&format!("{loops}/{}", l.id))).await;
        assert_eq!((s.status, s.error.as_deref()), (BenchLoopStatus::Stopped, None), "{s:?}");
        l.id
    };
    let path = dir.join("loops").join(format!("{loop_id}.json"));
    let mut file: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    file["loop"]["status"] = json!("stopping");
    file["loop"]["finished"] = Value::Null;
    std::fs::write(&path, file.to_string()).unwrap();
    let f = fixture(&home);
    let l: BenchLoop = ok(&f.app, get(&format!("/api/bench/benchmarks/{bench_id}/loops/{loop_id}"))).await;
    assert_eq!((l.status, l.error.as_deref()), (BenchLoopStatus::Stopped, None), "{l:?}");
    assert!(l.finished.is_some());
}

#[tokio::test(flavor = "multi_thread")]
async fn benchmarks_and_loops_keep_their_order_after_a_restart() {
    let t = TestDir::new("bench-order");
    let home = t.join("home");
    let (benches, bench_id, loops) = {
        let f = fixture(&home);
        let app = &f.app;
        let ds = dataset(app, "shapes", 20).await;
        let src = json!({ "kind": "dataset", "dataset_id": ds.id, "split": "val" });
        // Made one right after the other (many in the same second): each is later than the one before.
        let mut made: Vec<Bench> = Vec::new();
        for i in 0..12 {
            let b = bench(app, json!({ "name": format!("b{i}"), "source": src, "max_images": 1, "plan": { "conditions": [] } }))
                .await;
            if let Some(prev) = made.last() {
                assert!(b.created > prev.created, "{} then {}", prev.created, b.created);
            }
            made.push(b);
        }
        let list: BenchList = ok(app, get("/api/bench/benchmarks")).await;
        let benches: Vec<String> = list.benchmarks.iter().map(|b| b.id.clone()).collect();
        assert_eq!(benches, made.iter().rev().map(|b| b.id.clone()).collect::<Vec<_>>(), "newest first");
        let bench_id = made[0].id.clone();
        let teacher = model(app, &home, "teacher", &["circle", "square"], 7).await;
        let loops = format!("/api/bench/benchmarks/{bench_id}/loops");
        let pool = json!({ "kind": "dataset", "dataset_id": ds.id, "split": "train" });
        let mut started: Vec<BenchLoop> = Vec::new();
        for _ in 0..4 {
            let l: BenchLoop = ok(app, post(&loops, json!({ "teacher": teacher, "pool": pool }))).await;
            if let Some(prev) = started.last() {
                assert!(l.created > prev.created, "{} then {}", prev.created, l.created);
            }
            started.push(l);
        }
        for l in &started {
            loop_ended(app, &bench_id, &l.id).await;
        }
        let list: BenchLoopList = ok(app, get(&loops)).await;
        let loops: Vec<String> = list.loops.iter().map(|l| l.id.clone()).collect();
        assert_eq!(loops, started.iter().rev().map(|l| l.id.clone()).collect::<Vec<_>>(), "newest first");
        (benches, bench_id, loops)
    };
    let f = fixture(&home);
    let list: BenchList = ok(&f.app, get("/api/bench/benchmarks")).await;
    assert_eq!(list.benchmarks.iter().map(|b| b.id.clone()).collect::<Vec<_>>(), benches);
    let list: BenchLoopList = ok(&f.app, get(&format!("/api/bench/benchmarks/{bench_id}/loops"))).await;
    assert_eq!(list.loops.iter().map(|l| l.id.clone()).collect::<Vec<_>>(), loops);
}

#[test]
fn shutdown_returns_with_an_idle_worker() {
    // The worker waits for entries; shutdown must wake it however their timing falls.
    let t = TestDir::new("bench-shutdown");
    for i in 0..25 {
        let home = t.join(format!("home{i}"));
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let store = RealStore::open(home).device("cpu");
            store.shutdown();
            drop(store);
            let _ = tx.send(());
        });
        rx.recv_timeout(Duration::from_secs(30)).expect("shutdown returned");
    }
}
