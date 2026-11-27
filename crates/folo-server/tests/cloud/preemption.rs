//! Spot / preemptible worker automatic checkpoint resume: mid-run preemption recovery, bounded retry budget
//! exhaustion, stop-request cancellation during recovery backoff, manual termination exclusion, and preemption
//! before any checkpoint is synced.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use folo_server::Store;
use folo_server::api::*;
use folo_server::cloud::Tuning;
use folo_server::store::MetricsQuery;
use folo_test_support::TestDir;

use super::harness::{
    Call, Fault, TestCloud, cloud_run, connect, dataset, epochs, fast, finished, log_of, logged, no_gpu,
    phase, plane, record, run_of, steps, until,
};

#[test]
fn a_preempted_worker_auto_reprovisions_and_resumes_from_synced_checkpoint() {
    no_gpu();
    let t = TestDir::new("cloud-spot-resume");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);

    cloud.hold_epochs(true);
    let run = store.create_run(cloud_run(&ds, &conn.id, 3)).unwrap();
    let events = record(&store, &run.id);

    // Wait until epoch 1's checkpoint has synced locally, then simulate abrupt spot preemption of pod-1.
    until("first checkpoint synced", || run_of(&store, &run.id).cloud.and_then(|c| c.synced));
    let first_instance = run_of(&store, &run.id).cloud.and_then(|c| c.instance_id).unwrap();
    cloud.kill(0);

    // Release epoch holds once the replacement worker has launched so it trains epochs 2 and 3 to completion.
    until("replacement worker launched", || (cloud.calls(Call::Launch) >= 2).then_some(()));
    cloud.hold_epochs(false);

    let done = finished(&store, &run.id);
    let events = events.join().unwrap();
    assert_eq!(done.status, RunStatus::Completed, "{:?}", log_of(&store, &run.id));
    assert_eq!(done.epoch, 3);
    assert_eq!(phase(&done), CloudPhase::Released);
    assert_eq!(cloud.calls(Call::Launch), 2, "initial worker + 1 replacement");
    assert_eq!(cloud.live(), 0, "both instances released");

    // Preemption & auto-resume attempt surfaced clearly in run logs and checkpoint uploaded to pod-2.
    assert!(logged(&store, &run.id, "worker preempted or lost"), "{:?}", log_of(&store, &run.id));
    assert!(logged(&store, &run.id, "attempt 1/3"), "{:?}", log_of(&store, &run.id));
    assert!(logged(&store, &run.id, "uploading the checkpoint weights/last.folo"));

    // Cloud phase stream returned to Provisioning once for auto-resume (filter to Cloud events so interleaved
    // log lines do not affect transition counting) and cleared silence.
    let cloud_phases: Vec<CloudPhase> = events
        .iter()
        .filter_map(|e| match e {
            RunEvent::Cloud(c) => Some(c.phase),
            _ => None,
        })
        .collect();
    let provisioning_transitions = cloud_phases
        .windows(2)
        .filter(|w| w[0] != CloudPhase::Provisioning && w[1] == CloudPhase::Provisioning)
        .count();
    assert_eq!(
        provisioning_transitions, 1,
        "expected return to provisioning on preemption: {cloud_phases:?}"
    );
    let final_cloud = done.cloud.as_ref().unwrap();
    assert_ne!(final_cloud.instance_id.as_deref(), Some(first_instance.as_str()));
    assert_eq!(final_cloud.unanswered_since, None);
    assert!(final_cloud.cost_usd.is_some());

    // Metrics across both workers are continuous and duplicate-free (2 steps/epoch * 3 epochs = 6 steps).
    let m = store.run_metrics(&run.id, &MetricsQuery::default()).unwrap();
    assert_eq!((steps(&m), epochs(&m)), ((1..=6).collect::<Vec<_>>(), vec![1, 2, 3]));

    // Both instances are recorded as Terminated in the ledger and the final model & checkpoint exist.
    let instances = store.cloud_instances().unwrap();
    assert_eq!(instances.len(), 2);
    assert!(instances.iter().all(|i| i.status == CloudInstanceStatus::Terminated), "{instances:?}");
    let model = done.model_id.as_deref().unwrap();
    assert_eq!(store.get_model(model).unwrap().run_id.as_deref(), Some(run.id.as_str()));
    let last_model = format!("m_{}-last", run.id.trim_start_matches("run_"));
    assert_eq!(store.get_model(&last_model).unwrap().run_id.as_deref(), Some(run.id.as_str()));
}

#[test]
fn preemption_auto_resume_exhausts_bounded_retry_budget() {
    no_gpu();
    let t = TestDir::new("cloud-spot-budget");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, Tuning { max_preemption_retries: 2, ..fast() });
    let ds = dataset(&store);
    let conn = connect(&store);

    // Use 5 epochs (> 1 + max_preemption_retries) so worker 3 is guaranteed to pause at EpochHold even if both
    // worker 1 and worker 2 complete and sync an epoch before being killed.
    cloud.hold_epochs(true);
    let run = store.create_run(cloud_run(&ds, &conn.id, 5)).unwrap();

    // 1st preemption (after epoch 1 checkpoint syncs) -> retry 1/2.
    until("first checkpoint synced", || run_of(&store, &run.id).cloud.and_then(|c| c.synced));
    cloud.kill(0);

    // 2nd preemption once worker 2 reaches Training -> retry 2/2.
    until("worker 2 training", || {
        (cloud.calls(Call::Launch) >= 2 && phase(&run_of(&store, &run.id)) == CloudPhase::Training)
            .then_some(())
    });
    cloud.kill(1);

    // 3rd preemption once worker 3 reaches Training -> budget exhausted, no 4th worker launched.
    until("worker 3 training", || {
        (cloud.calls(Call::Launch) >= 3 && phase(&run_of(&store, &run.id)) == CloudPhase::Training)
            .then_some(())
    });
    cloud.kill(2);

    let done = finished(&store, &run.id);
    assert_eq!(
        (done.status, done.error_code.as_deref()),
        (RunStatus::Failed, Some("cloud_instance_lost")),
        "{:?}",
        log_of(&store, &run.id)
    );
    let err = done.error.as_deref().unwrap();
    assert!(err.contains("exhausted all 2 attempt(s)"), "{err}");
    assert_eq!(cloud.calls(Call::Launch), 3, "must not launch past max_preemption_retries");
    assert_eq!(cloud.live(), 0);
    assert!(t.join("home").join("runs").join(&run.id).join("weights").join("last.folo").is_file());
    let instances = store.cloud_instances().unwrap();
    assert_eq!(instances.len(), 3);
    assert!(instances.iter().all(|i| i.status == CloudInstanceStatus::Terminated), "{instances:?}");
}

#[test]
fn stop_request_during_preemption_backoff_cancels_recovery_promptly() {
    no_gpu();
    let t = TestDir::new("cloud-spot-stop");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(
        &t.join("home"),
        &cloud,
        Tuning { max_preemption_retries: 3, preemption_backoff: Duration::from_secs(60), ..fast() },
    );
    let ds = dataset(&store);
    let conn = connect(&store);

    cloud.hold_epochs(true);
    let run = store.create_run(cloud_run(&ds, &conn.id, 3)).unwrap();
    until("first checkpoint synced", || run_of(&store, &run.id).cloud.and_then(|c| c.synced));
    cloud.kill(0);

    // Wait until the executor enters preemption recovery backoff, then request stop.
    until("preemption recovery logged", || {
        logged(&store, &run.id, "auto-resuming from synced checkpoint").then_some(())
    });
    let t0 = Instant::now();
    store.stop_run(&run.id).unwrap();

    let done = finished(&store, &run.id);
    assert!(t0.elapsed() < Duration::from_secs(30), "stop waited through backoff: {:?}", t0.elapsed());
    assert_eq!(done.status, RunStatus::Stopped, "{:?}", log_of(&store, &run.id));
    assert_eq!(cloud.calls(Call::Launch), 1, "must not launch a replacement after stop");
    assert_eq!(cloud.live(), 0);
    assert!(logged(&store, &run.id, "stopped during preemption recovery"), "{:?}", log_of(&store, &run.id));
    assert!(t.join("home").join("runs").join(&run.id).join("weights").join("last.folo").is_file());
}

#[test]
fn manual_termination_and_preemption_without_checkpoint_do_not_auto_resume() {
    no_gpu();
    let t = TestDir::new("cloud-spot-exclusions");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);

    // 1. Manual termination via `terminate_cloud_instance` after a checkpoint synced must NOT auto-resume.
    cloud.hold_epochs(true);
    let run1 = store.create_run(cloud_run(&ds, &conn.id, 3)).unwrap();
    until("run1 checkpoint synced", || run_of(&store, &run1.id).cloud.and_then(|c| c.synced));
    let inst1 = run_of(&store, &run1.id).cloud.and_then(|c| c.instance_id).unwrap();
    store.terminate_instance(&inst1).unwrap();
    let done1 = finished(&store, &run1.id);
    assert_eq!(
        (done1.status, done1.error_code.as_deref()),
        (RunStatus::Failed, Some("cloud_instance_lost")),
        "{:?}",
        log_of(&store, &run1.id)
    );
    assert!(done1.error.as_deref().unwrap().contains("terminated by hand"));
    assert_eq!(cloud.calls(Call::Launch), 1, "manual termination must not auto-resume");
    assert_eq!(cloud.live(), 0);

    // 2. Unexpected preemption before any `weights/last.folo` finishes copying must NOT auto-resume.
    cloud.gates.weights.store(true, Ordering::SeqCst);
    let run2 = store.create_run(cloud_run(&ds, &conn.id, 3)).unwrap();
    until("checkpoint download started and held", || {
        (cloud.gates.weights_held.load(Ordering::SeqCst) > 0).then_some(())
    });
    cloud.kill(1);
    cloud.gates.weights.store(false, Ordering::SeqCst);
    let done2 = finished(&store, &run2.id);
    assert_eq!(
        (done2.status, done2.error_code.as_deref()),
        (RunStatus::Failed, Some("cloud_instance_lost")),
        "{:?}",
        log_of(&store, &run2.id)
    );
    assert!(done2.error.as_deref().unwrap().contains("no synced checkpoint"));
    assert_eq!(cloud.calls(Call::Launch), 2, "no replacement without a synced checkpoint");
    assert_eq!(cloud.live(), 0);
}

#[test]
fn reaper_terminates_superseded_preempted_instance_while_replacement_trains() {
    no_gpu();
    let t = TestDir::new("cloud-spot-reaper");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(
        &t.join("home"),
        &cloud,
        Tuning { release_wait: Duration::from_millis(80), reaper_every: Duration::from_millis(50), ..fast() },
    );
    let ds = dataset(&store);
    let conn = connect(&store);

    cloud.hold_epochs(true);
    let run = store.create_run(cloud_run(&ds, &conn.id, 3)).unwrap();
    until("first checkpoint synced", || run_of(&store, &run.id).cloud.and_then(|c| c.synced));
    let first_id = run_of(&store, &run.id).cloud.and_then(|c| c.instance_id).unwrap();

    // Refuse Terminate while pod-1 is preempted so pod-1 stays `Terminating` after `release_for_retry` times out.
    cloud.fail(Call::Terminate, Fault::Fail(folo_cloud::ErrorKind::Unavailable), usize::MAX);
    cloud.kill(0);

    // Wait until the replacement worker (pod-2) is actively training (run still holds `cloud.lock`!).
    until("replacement worker training", || {
        (cloud.calls(Call::Launch) >= 2 && phase(&run_of(&store, &run.id)) == CloudPhase::Training)
            .then_some(())
    });
    let first_now = store.cloud_instances().unwrap().into_iter().find(|i| i.id == first_id).unwrap();
    assert_eq!(first_now.status, CloudInstanceStatus::Terminating, "pod-1 still awaiting confirmation");

    // Re-enable Terminate while pod-2 is STILL held mid-training: the reaper must confirm pod-1's termination
    // without waiting for pod-2 to finish!
    cloud.clear(Call::Terminate);
    until("reaper confirms superseded pod-1 terminated while pod-2 runs", || {
        store
            .cloud_instances()
            .unwrap()
            .into_iter()
            .find(|i| i.id == first_id && i.status == CloudInstanceStatus::Terminated)
    });
    assert!(run_of(&store, &run.id).status.is_active(), "replacement worker still training");

    // Let pod-2 finish and verify clean completion.
    cloud.hold_epochs(false);
    let done = finished(&store, &run.id);
    assert_eq!(done.status, RunStatus::Completed, "{:?}", log_of(&store, &run.id));
    assert_eq!(cloud.live(), 0);
}

#[test]
fn preemption_during_replacement_sync_auto_reprovisions_again() {
    no_gpu();
    let t = TestDir::new("cloud-spot-sync-preempt");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);

    cloud.hold_epochs(true);
    let run = store.create_run(cloud_run(&ds, &conn.id, 3)).unwrap();
    until("first checkpoint synced", || run_of(&store, &run.id).cloud.and_then(|c| c.synced));

    // Hold uploads on the replacement worker and wait until an upload request actually arrives at pod-2 before
    // killing it, proving the worker died mid-upload.
    cloud.gates.uploads_held.store(0, Ordering::SeqCst);
    cloud.gates.uploads.store(true, Ordering::SeqCst);
    cloud.kill(0);
    until("replacement worker upload held", || {
        (cloud.calls(Call::Launch) >= 2 && cloud.gates.uploads_held.load(Ordering::SeqCst) > 0).then_some(())
    });
    cloud.kill(1);
    cloud.gates.uploads.store(false, Ordering::SeqCst);

    // Sync failure on a lost instance must escalate to Preempted -> launch pod-3 -> complete cleanly.
    until("third worker launched", || (cloud.calls(Call::Launch) >= 3).then_some(()));
    cloud.hold_epochs(false);
    let done = finished(&store, &run.id);
    assert_eq!(done.status, RunStatus::Completed, "{:?}", log_of(&store, &run.id));
    assert_eq!(done.epoch, 3);
    assert_eq!(cloud.calls(Call::Launch), 3, "pod-1 + preempted-in-sync pod-2 + pod-3");
    assert_eq!(cloud.live(), 0);
    assert!(logged(&store, &run.id, "attempt 2/3"), "{:?}", log_of(&store, &run.id));
}

#[test]
fn restart_during_preemption_recovery_continues_auto_resume() {
    no_gpu();
    let t = TestDir::new("cloud-spot-restart-backoff");
    let cloud = TestCloud::new(&t.join("workers"));
    let home1 = t.join("home1");
    let store1 = plane(
        &home1,
        &cloud,
        Tuning { max_preemption_retries: 3, preemption_backoff: Duration::from_secs(60), ..fast() },
    );
    let ds = dataset(&store1);
    let conn = connect(&store1);

    cloud.hold_epochs(true);
    let run = store1.create_run(cloud_run(&ds, &conn.id, 3)).unwrap();
    until("first checkpoint synced", || run_of(&store1, &run.id).cloud.and_then(|c| c.synced));
    cloud.kill(0);

    // Wait until preemption recovery persisted `recovering_preemption: true` and entered backoff.
    let cloud_json = home1.join("runs").join(&run.id).join("cloud.json");
    until("recovering_preemption persisted", || {
        let ok_log = logged(&store1, &run.id, "auto-resuming from synced checkpoint");
        let ok_json = std::fs::read_to_string(&cloud_json)
            .ok()
            .is_some_and(|s| s.contains("\"recovering_preemption\": true"));
        (ok_log && ok_json).then_some(())
    });

    // Snapshot home1 into home2 (skipping advisory lock files) to simulate an abrupt process crash mid-backoff,
    // then stop & shut down store1 cleanly so only store2 drives recovery.
    let home2 = t.join("home2");
    copy_dir_without_locks(&home1, &home2);
    store1.stop_run(&run.id).unwrap();
    let _ = finished(&store1, &run.id);
    store1.shutdown();
    drop(store1);

    // Reopen on home2 with fast tuning: `Mode::Recover` must detect `recovering_preemption: true`, launch the
    // replacement worker, resume from weights/last.folo, and finish epochs 2..=3.
    cloud.hold_epochs(false);
    let store2 = plane(&home2, &cloud, fast());
    let done = finished(&store2, &run.id);
    assert_eq!(done.status, RunStatus::Completed, "{:?}", log_of(&store2, &run.id));
    assert_eq!(done.epoch, 3);
    assert_eq!(cloud.calls(Call::Launch), 2);
    assert_eq!(cloud.live(), 0);
    assert!(
        logged(&store2, &run.id, "restarted during preemption recovery"),
        "{:?}",
        log_of(&store2, &run.id)
    );
    store2.shutdown();
}

#[test]
fn per_run_max_preempt_retries_override_and_status_surface() {
    no_gpu();
    let t = TestDir::new("cloud-spot-per-run");
    let cloud = TestCloud::new(&t.join("workers"));
    // Server default is 3, but per-run `max_preempt_retries` overrides it to 1 and 0.
    let store = plane(&t.join("home"), &cloud, Tuning { max_preemption_retries: 3, ..fast() });
    let ds = dataset(&store);
    let conn = connect(&store);

    // Local run rejects `max_preempt_retries`; out-of-range (> 20) is rejected too.
    let local_cfg: TrainConfig = serde_json::from_value(serde_json::json!({
        "dataset_id": ds.id, "epochs": 1, "imgsz": 64, "batch": 2, "device": "cpu"
    }))
    .unwrap();
    let e = store
        .create_run_with_options(
            local_cfg,
            folo_server::store::RunOptions { max_preempt_retries: Some(2), ..Default::default() },
        )
        .unwrap_err();
    assert_eq!(e.body.code, "invalid_config");

    let e = store
        .create_run_with_options(
            cloud_run(&ds, &conn.id, 3),
            folo_server::store::RunOptions { max_preempt_retries: Some(21), ..Default::default() },
        )
        .unwrap_err();
    assert_eq!(e.body.code, "invalid_config");

    // Per-run `max_preempt_retries: Some(1)` recovers once then exhausts at attempt 1 (not server default 3).
    cloud.hold_epochs(true);
    let run1 = store
        .create_run_with_options(
            cloud_run(&ds, &conn.id, 4),
            folo_server::store::RunOptions { max_preempt_retries: Some(1), ..Default::default() },
        )
        .unwrap();
    until("run1 checkpoint synced", || run_of(&store, &run1.id).cloud.and_then(|c| c.synced));
    let first_inst = run_of(&store, &run1.id).cloud.and_then(|c| c.instance_id).unwrap();
    cloud.kill(0);

    until("run1 replacement worker training", || {
        (cloud.calls(Call::Launch) >= 2 && phase(&run_of(&store, &run1.id)) == CloudPhase::Training)
            .then_some(())
    });
    let (_, pre1) = store.get_run_with_preemption(&run1.id).unwrap();
    let pre1 = pre1.expect("cloud run reports preemption status");
    assert_eq!((pre1.retries, pre1.max_retries, pre1.recovering), (1, 1, false));
    assert_eq!(pre1.preempted_instances, vec![first_inst]);

    // Second preemption exhausts the per-run limit of 1.
    cloud.kill(1);
    let done1 = finished(&store, &run1.id);
    assert_eq!((done1.status, done1.error_code.as_deref()), (RunStatus::Failed, Some("cloud_instance_lost")));
    assert!(done1.error.as_deref().unwrap().contains("exhausted all 1 attempt(s)"), "{:?}", done1.error);
    assert_eq!(cloud.calls(Call::Launch), 2);

    // Per-run `max_preempt_retries: Some(0)` disables automatic recovery on the very first preemption.
    let run0 = store
        .create_run_with_options(
            cloud_run(&ds, &conn.id, 3),
            folo_server::store::RunOptions { max_preempt_retries: Some(0), ..Default::default() },
        )
        .unwrap();
    until("run0 checkpoint synced", || run_of(&store, &run0.id).cloud.and_then(|c| c.synced));
    cloud.kill(2);
    let done0 = finished(&store, &run0.id);
    assert_eq!((done0.status, done0.error_code.as_deref()), (RunStatus::Failed, Some("cloud_instance_lost")));
    let err0 = done0.error.as_deref().unwrap();
    assert!(err0.contains("the weights copied so far stay") && !err0.contains("exhausted"), "{err0}");
    assert_eq!(cloud.calls(Call::Launch), 3, "zero retries must not launch a replacement");
    let (_, pre0) = store.get_run_with_preemption(&run0.id).unwrap();
    assert_eq!(pre0.unwrap().max_retries, 0);

    // HTTP `POST /api/runs` with `checkpoint` + `compute` + `max_preempt_retries: 0` forwards `0`
    // (does not drop it back to server default 3), and `MockStore` neither advertises `cloud_spot_resume`
    // nor accepts `max_preempt_retries`.
    let models_dir = t.join("home").join("models");
    std::fs::create_dir_all(&models_dir).unwrap();
    std::fs::copy(
        t.join("home").join("runs").join(&run0.id).join("weights").join("last.folo"),
        models_dir.join("ckpt-spot.folo"),
    )
    .unwrap();
    let ckpt_model = store
        .list_models()
        .unwrap()
        .into_iter()
        .find(|m| m.name == "ckpt-spot")
        .expect("copied checkpoint discovered as model");

    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let app = folo_server::Server::new(store.clone(), folo_server::Options::default()).router();
    let (status, body) = rt.block_on(async {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let offer_id = cloud_run(&ds, &conn.id, 3).compute.unwrap().offer_id;
        let req_body = serde_json::json!({
            "checkpoint": ckpt_model.id,
            "dataset_id": ds.id,
            "epochs": 3,
            "imgsz": 64,
            "batch": 2,
            "device": "auto",
            "compute": { "connection_id": conn.id, "offer_id": offer_id },
            "max_preempt_retries": 0
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/runs")
            .header("host", "127.0.0.1")
            .header("content-type", "application/json")
            .body(Body::from(req_body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice::<serde_json::Value>(&bytes).unwrap())
    });
    assert_eq!(status, axum::http::StatusCode::CREATED, "{body}");
    let cont_id = body["id"].as_str().unwrap().to_string();
    let (_, cont_pre) = store.get_run_with_preemption(&cont_id).unwrap();
    assert_eq!(
        cont_pre.unwrap().max_retries,
        0,
        "checkpoint continuation must preserve max_preempt_retries: 0"
    );
    until("continued run training", || {
        (cloud.calls(Call::Launch) >= 4 && phase(&run_of(&store, &cont_id)) == CloudPhase::Training)
            .then_some(())
    });
    cloud.kill(3);
    let cont_done = finished(&store, &cont_id);
    assert_eq!(
        (cont_done.status, cont_done.error_code.as_deref()),
        (RunStatus::Failed, Some("cloud_instance_lost"))
    );
    assert_eq!(
        cloud.calls(Call::Launch),
        4,
        "checkpoint continuation with 0 retries must not launch a replacement"
    );

    let mock_app = folo_server::Server::new(
        std::sync::Arc::new(folo_server::MockStore::new()),
        folo_server::Options::default(),
    )
    .router();
    let (h_body, m_status, m_body) = rt.block_on(async {
        use axum::body::Body;
        use axum::http::Request;
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let h_req =
            Request::builder().uri("/api/health").header("host", "127.0.0.1").body(Body::empty()).unwrap();
        let h_resp = mock_app.clone().oneshot(h_req).await.unwrap();
        let h_bytes = h_resp.into_body().collect().await.unwrap().to_bytes();
        let h_body: serde_json::Value = serde_json::from_slice(&h_bytes).unwrap();

        let r_body = serde_json::json!({
            "dataset_id": "ds_001",
            "epochs": 1,
            "imgsz": 64,
            "batch": 2,
            "compute": { "connection_id": "cc_1", "offer_id": "o1" },
            "max_preempt_retries": 0
        });
        let r_req = Request::builder()
            .method("POST")
            .uri("/api/runs")
            .header("host", "127.0.0.1")
            .header("content-type", "application/json")
            .body(Body::from(r_body.to_string()))
            .unwrap();
        let r_resp = mock_app.oneshot(r_req).await.unwrap();
        let m_status = r_resp.status();
        let m_bytes = r_resp.into_body().collect().await.unwrap().to_bytes();
        (h_body, m_status, serde_json::from_slice::<serde_json::Value>(&m_bytes).unwrap())
    });
    let mock_features: Vec<&str> =
        h_body["features"].as_array().unwrap().iter().filter_map(|v| v.as_str()).collect();
    assert!(
        !mock_features.contains(&folo_server::store::FEATURE_CLOUD_SPOT_RESUME),
        "MockStore must not advertise cloud_spot_resume: {mock_features:?}"
    );
    assert_eq!(m_status, axum::http::StatusCode::NOT_IMPLEMENTED, "{m_body}");
    assert_eq!(m_body["error"]["code"], "not_implemented");
    store.shutdown();
}

fn copy_dir_without_locks(src: &std::path::Path, dst: &std::path::Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let s = name.to_string_lossy();
        if s == "cloud.lock" || s == ".lock" || s.ends_with(".lock") {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        if entry.file_type().unwrap().is_dir() {
            copy_dir_without_locks(&from, &to);
        } else {
            std::fs::copy(&from, &to).unwrap();
        }
    }
}
