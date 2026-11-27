//! End-to-end tests for MCP `compare_bench_entries`, `folo_info` CPU SIMD parity, and `train` spot-preemption
//! argument validation.

use std::path::Path;
use std::sync::Arc;

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

async fn call(s: &FoloMcp, tool: &str, args: Value) -> Result<Value, Error> {
    let args = args.as_object().unwrap().clone();
    Ok(s.inner.call(find(tool).unwrap(), args, &mut Progress::none()).await?.value)
}

async fn ok(s: &FoloMcp, tool: &str, args: Value) -> Value {
    call(s, tool, args).await.unwrap_or_else(|e| panic!("{tool} failed: {e:?}"))
}

async fn err(s: &FoloMcp, tool: &str, args: Value) -> Error {
    call(s, tool, args).await.expect_err(&format!("{tool} succeeded"))
}

/// `compare_bench_entries` reports head-to-head score, axis, accuracy, and condition deltas in process and
/// through a loopback `folo serve`, flags unmatched conditions & identical entries, and refuses unfinished or
/// missing entries cleanly.
#[tokio::test]
async fn compare_bench_entries_reports_deltas_and_refuses_unfinished() {
    let home = TestDir::new("mcp-bench-compare");
    let store = Arc::new(RealStore::open(home.to_path_buf()).device("cpu"));
    let embedded = FoloMcp::new(store.clone(), None, options(home.path()));

    let plan = json!({ "conditions": [{ "condition": "haze", "severities": [2], "weight": 1.0 }] });
    let bench = ok(
        &embedded,
        "create_benchmark",
        json!({
            "name": "compare-smoke",
            "generated": { "images": 4, "classes": ["circle", "square"], "imgsz": 64 },
            "seed": 7,
            "plan": plan
        }),
    )
    .await;
    let bench_id = bench["id"].as_str().unwrap().to_string();
    let imgs = store.benches().unwrap().images(&bench_id, 0, 10, None).unwrap().images;

    // Baseline: perfect boxes on unchanged images only (no haze condition -> partial, unmatched condition).
    let base_file = home.join("base.jsonl");
    let mut base_lines = Vec::new();
    for img in &imgs {
        let boxes: Vec<_> = img
            .boxes
            .iter()
            .map(
                |b| json!({ "class": b.class, "score": 0.9, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2 }),
            )
            .collect();
        base_lines.push(json!({ "image": img.id, "boxes": boxes }).to_string());
    }
    std::fs::write(&base_file, base_lines.join("\n") + "\n").unwrap();

    // Candidate: perfect boxes on unchanged AND haze severity 2.
    let cand_file = home.join("cand.jsonl");
    let mut cand_lines = base_lines.clone();
    for img in &imgs {
        let boxes: Vec<_> = img
            .boxes
            .iter()
            .map(|b| json!({ "class": b.class, "score": 0.95, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2 }))
            .collect();
        cand_lines
            .push(json!({ "image": img.id, "condition": "haze", "severity": 2, "boxes": boxes }).to_string());
    }
    std::fs::write(&cand_file, cand_lines.join("\n") + "\n").unwrap();

    let e_base = ok(
        &embedded,
        "start_bench_entry",
        json!({ "benchmark_id": bench_id, "predictions_file": "base.jsonl", "name": "baseline" }),
    )
    .await;
    let e_cand = ok(
        &embedded,
        "start_bench_entry",
        json!({ "benchmark_id": bench_id, "predictions_file": "cand.jsonl", "name": "candidate" }),
    )
    .await;
    let base_id = e_base["entry_id"].as_str().unwrap().to_string();
    let cand_id = e_cand["entry_id"].as_str().unwrap().to_string();

    let cmp = ok(
        &embedded,
        "compare_bench_entries",
        json!({ "benchmark_id": bench_id, "baseline_entry_id": base_id, "candidate_entry_id": cand_id }),
    )
    .await;
    assert_eq!(cmp["benchmark_id"], bench_id);
    assert_eq!(cmp["baseline"]["entry_id"], base_id);
    assert_eq!(cmp["candidate"]["entry_id"], cand_id);
    assert_eq!(cmp["axes"].as_array().unwrap().len(), 5);
    assert_eq!(cmp["conditions"][0]["condition"], "haze");
    assert_eq!(cmp["conditions"][0]["match_kind"], "candidate_only");
    assert!(cmp["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("partial")));

    // Comparing an entry with itself emits a comparability warning and zero score delta.
    let self_cmp = ok(
        &embedded,
        "compare_bench_entries",
        json!({ "benchmark_id": bench_id, "baseline_entry_id": cand_id, "candidate_entry_id": cand_id }),
    )
    .await;
    assert_eq!(self_cmp["score_delta"].as_f64().unwrap(), 0.0);
    assert!(
        self_cmp["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("same entry"))
    );

    // Missing entry -> bench_entry_not_found.
    let missing = err(
        &embedded,
        "compare_bench_entries",
        json!({ "benchmark_id": bench_id, "baseline_entry_id": base_id, "candidate_entry_id": "be_missing" }),
    )
    .await;
    assert_eq!(missing.body.code, "bench_entry_not_found");

    // Unfinished (stopped) entry -> 409 bench_entry_not_done.
    let open = store
        .benches()
        .unwrap()
        .start_entry(
            &*store,
            &bench_id,
            folo_api::StartBenchEntry {
                contestant: folo_api::BenchContestantSpec::External {
                    scored: true,
                    open: true,
                    predictions: vec![],
                },
                name: Some("unfinished".into()),
                max_images: None,
                conditions: None,
            },
        )
        .unwrap();
    store.benches().unwrap().stop_entry(&bench_id, &open.id).unwrap();
    let not_done = err(
        &embedded,
        "compare_bench_entries",
        json!({ "benchmark_id": bench_id, "baseline_entry_id": base_id, "candidate_entry_id": open.id }),
    )
    .await;
    assert_eq!(not_done.body.code, "bench_entry_not_done");
    let not_done_hint = not_done.body.hint.as_deref().unwrap_or("");
    assert!(
        not_done_hint.contains("rerun")
            && !not_done_hint.contains("wait for")
            && !not_done_hint.contains("<compare terminal>"),
        "stopped entry compare hint should suggest inspect/rerun, not waiting: {not_done_hint}"
    );

    // Through a loopback `folo serve` matches embedded comparison.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = folo_server::Server::new(store.clone(), folo_server::Options::default());
    let serving = tokio::spawn(server.serve(listener, async move {
        let _ = stopped.await;
    }));
    let http = Arc::new(HttpStore::new(&url, None));
    let proxied = FoloMcp::new(http.clone(), Some(http), options(home.path()));
    let cmp_http = ok(
        &proxied,
        "compare_bench_entries",
        json!({ "benchmark_id": bench_id, "baseline_entry_id": base_id, "candidate_entry_id": cand_id }),
    )
    .await;
    assert_eq!(cmp_http["score_delta"], cmp["score_delta"]);
    assert_eq!(cmp_http["conditions"], cmp["conditions"]);

    // `folo_info` exposes CPU SIMD capabilities including `avx512_vnni` and `int8_kernel` in both modes.
    for s in [&embedded, &proxied] {
        let info = ok(s, "folo_info", json!({})).await;
        let simd = &info["devices"]["cpu"]["simd"];
        assert!(simd["avx512_vnni"].is_boolean(), "{simd}");
        assert!(!simd["int8_kernel"].as_str().unwrap().is_empty(), "{simd}");
    }

    // `train` rejects `max_preempt_retries` on local runs with `invalid_config`.
    let bad_local =
        err(&embedded, "train", json!({ "dataset_id": "ds_missing", "max_preempt_retries": 2 })).await;
    assert_eq!(bad_local.body.code, "invalid_config");

    let _ = stop.send(());
    serving.await.unwrap().unwrap();
    store.shutdown();
}
