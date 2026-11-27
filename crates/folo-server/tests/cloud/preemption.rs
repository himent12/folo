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
