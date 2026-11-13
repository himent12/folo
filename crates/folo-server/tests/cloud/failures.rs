//! Failures and releases: a removed connection, boot failures, a worker that is lost, panics, refused
//! terminations, crash recovery, shutdown and the deadline.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::time::Duration;

use folo_cloud::{ErrorKind, LaunchSpec, Provider, Values};
use folo_server::Store;
use folo_server::api::*;
use folo_server::cloud::Tuning;
use folo_server::store::MetricsQuery;
use folo_test_support::TestDir;
use serde_json::json;

use super::harness::{
    Call, Fault, SECRET, TestCloud, cloud_run, connect, dataset, epochs, fast, finished, log_of, logged,
    no_gpu, offer, phase, plane, run_of, steps, until,
};

#[test]
fn a_connection_removed_while_a_run_starts_has_nothing_launched_on_it() {
    no_gpu();
    let t = TestDir::new("cloud-disconnect-race");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    // The run is asked for, and while its offer is looked up the connection is removed: nothing uses it yet.
    cloud.hold(Call::Offers);
    let creating = {
        let (store, config) = (store.clone(), cloud_run(&ds, &conn.id, 1));
        std::thread::spawn(move || store.create_run(config))
    };
    until("the offers asked for", || (cloud.calls(Call::Offers) == 1).then_some(()));
    store.disconnect_cloud(&conn.id).unwrap();
    cloud.release(Call::Offers);
    let run = creating.join().unwrap().unwrap();
    let done = finished(&store, &run.id);
    // An instance launched now could not be terminated: without its connection, nothing can reach the provider.
    assert_eq!(done.status, RunStatus::Failed, "{:?}", log_of(&store, &run.id));
    assert_eq!(done.error_code.as_deref(), Some("connection_not_found"));
    assert_eq!((cloud.calls(Call::Launch), cloud.live()), (0, 0));
    assert!(store.cloud_instances().unwrap().is_empty());
}

#[test]
fn boot_failures_name_the_check_and_release() {
    no_gpu();
    let t = TestDir::new("cloud-boot");
    let cloud = TestCloud::new(&t.join("workers"));
    let boot = |name: &str, tuning: Tuning| {
        let store = plane(&t.join(name), &cloud, tuning);
        let ds = dataset(&store);
        let conn = connect(&store);
        let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
        let done = finished(&store, &run.id);
        assert_eq!(done.error_code.as_deref(), Some("cloud_boot_failed"), "{name}: {:?}", done.error);
        assert_eq!(cloud.live(), 0, "{name}: released");
        done.error.unwrap()
    };
    cloud.starting.store(u32::MAX, Ordering::SeqCst);
    let e = boot("timeout", Tuning { boot_timeout: Duration::from_millis(300), ..fast() });
    assert!(e.contains("did not start within"), "{e}");
    cloud.starting.store(1, Ordering::SeqCst);
    cloud.gates.old.store(true, Ordering::SeqCst);
    let e = boot("old", fast());
    assert!(
        e.contains("cannot train cloud runs") && e.contains("uploads, run_cursors, run_checkpoint"),
        "{e}"
    );
    cloud.gates.old.store(false, Ordering::SeqCst);
    let e = boot("no-gpu", Tuning { require_gpu: true, ..fast() });
    assert!(e.contains("no GPU FOLO can use"), "{e}");
}

#[test]
fn an_instance_that_gives_up_before_its_worker_answers_fails_the_boot_at_once() {
    no_gpu();
    let t = TestDir::new("cloud-setup");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    // The instance runs and its worker does not answer, as while (and when) its setup fails...
    cloud.gates.down.store(true, Ordering::SeqCst);
    let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
    until("the instance runs", || logged(&store, &run.id, "the instance runs").then_some(()));
    assert_eq!(phase(&run_of(&store, &run.id)), CloudPhase::Booting);
    // ...and then it terminates itself, as the bootstrap script does when a step fails.
    cloud.kill(0);
    let done = finished(&store, &run.id);
    assert_eq!((done.status, done.error_code.as_deref()), (RunStatus::Failed, Some("cloud_boot_failed")));
    let e = done.error.clone().unwrap();
    assert!(e.contains("its setup failed") && e.contains("installs FOLO from http"), "{e}");
    assert!(e.contains("log at Test cloud"), "{e}");
    // Said as soon as the provider is believed, not at the boot timeout (whose message is "did not answer within").
    assert!(!e.contains("did not answer within"), "{e}");
    assert_eq!(phase(&done), CloudPhase::Released);
    // It failed while booting, and the run says so (Studio marks that phase, not the ones after it).
    assert_eq!(done.cloud.as_ref().unwrap().ended_in, Some(CloudPhase::Booting));
    assert_eq!(store.cloud_instances().unwrap()[0].status, CloudInstanceStatus::Terminated);
}

#[test]
fn a_stop_that_finds_the_worker_gone_loses_the_run_instead_of_collecting() {
    no_gpu();
    let t = TestDir::new("cloud-stop-lost");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    cloud.hold_epochs(true);
    let run = store.create_run(cloud_run(&ds, &conn.id, 2)).unwrap();
    until("the first checkpoint copied", || run_of(&store, &run.id).cloud.and_then(|c| c.synced));
    // The instance is gone (it terminated itself at its deadline while nothing followed the run), and then the
    // run is asked to stop: there is nothing to stop or to collect from.
    cloud.kill(0);
    store.stop_run(&run.id).unwrap();
    let lost = finished(&store, &run.id);
    assert_eq!(
        (lost.status, lost.error_code.as_deref()),
        (RunStatus::Failed, Some("cloud_instance_lost")),
        "{:?}",
        lost.error
    );
    assert!(lost.error.unwrap().contains("resume the run"));
    assert!(t.join("home").join("runs").join(&run.id).join("weights").join("last.folo").is_file());
}

#[test]
fn a_checkpoint_fetched_just_before_the_worker_is_lost_is_kept() {
    no_gpu();
    let t = TestDir::new("cloud-copy-lost");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    cloud.hold_epochs(true);
    cloud.gates.cut_after_weights.store(true, Ordering::SeqCst);
    let run = store.create_run(cloud_run(&ds, &conn.id, 2)).unwrap();
    // The checkpoint of epoch 1 arrives whole, and after it nothing from the worker does...
    let weights = t.join("home").join("runs").join(&run.id).join("weights");
    // (Staged, or already moved into place by the next poll: the window of the staged file is one poll long.)
    until("the checkpoint fetched", || {
        let names = std::fs::read_dir(&weights).ok()?;
        names
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .any(|n| n.starts_with(".last.folo.copy.") || n == "last.folo")
            .then_some(())
    });
    // The run says since when its worker is silent, and stops saying so when it answers again...
    let silent = || run_of(&store, &run.id).cloud.and_then(|c| c.unanswered_since);
    until("the silence on the run", silent);
    cloud.gates.cut_after_weights.store(false, Ordering::SeqCst);
    cloud.gates.cut.store(false, Ordering::SeqCst);
    until("the worker answers again", || silent().is_none().then_some(()));
    // ...until its instance is gone.
    cloud.kill(0);
    let lost = finished(&store, &run.id);
    assert_eq!(lost.error_code.as_deref(), Some("cloud_instance_lost"), "{:?}", log_of(&store, &run.id));
    assert!(weights.join("last.folo").is_file(), "the checkpoint fetched is kept, to resume from");
    let cloud = lost.cloud.unwrap();
    assert!(cloud.synced.is_some());
    // An ended run's silence is said by how it ended.
    assert_eq!(cloud.unanswered_since, None);
}

#[test]
fn a_panic_still_releases_the_instance() {
    no_gpu();
    let t = TestDir::new("cloud-panic");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    cloud.fail(Call::State, Fault::Panic, 1);
    let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
    let done = finished(&store, &run.id);
    assert_eq!((done.status, done.error_code.as_deref()), (RunStatus::Failed, Some("internal")));
    assert!(done.error.as_deref().unwrap().contains("panicked"));
    assert_eq!((cloud.live(), phase(&done)), (0, CloudPhase::Released));
}

#[test]
fn a_refused_termination_is_retried_until_it_succeeds() {
    no_gpu();
    let t = TestDir::new("cloud-terminate");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    cloud.fail(Call::Terminate, Fault::Fail(ErrorKind::Unavailable), usize::MAX);
    let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
    let done = finished(&store, &run.id);
    assert_eq!(done.status, RunStatus::Completed, "the run does not wait for its instance");
    assert!(logged(&store, &run.id, "the instance is not terminated yet"));
    let i = store.cloud_instances().unwrap().remove(0);
    assert_eq!(i.status, CloudInstanceStatus::Terminating);
    assert!(i.error.unwrap().contains("refuses Terminate"));
    assert!(
        cloud.live() == 1 && store.disconnect_cloud(&conn.id).unwrap_err().body.code == "connection_in_use"
    );
    cloud.clear(Call::Terminate);
    until("the reaper's termination", || {
        (store.cloud_instances().unwrap()[0].status == CloudInstanceStatus::Terminated).then_some(())
    });
    assert_eq!(cloud.live(), 0);
    assert!(store.cloud_instances().unwrap()[0].error.is_none());
}

#[test]
fn a_restart_releases_what_a_crash_left_and_follows_a_training_run_again() {
    no_gpu();
    let t = TestDir::new("cloud-restart");
    let home = t.join("home");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&home, &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);

    // A run that trains when this server goes away: the next one follows it to the end.
    cloud.hold_epochs(true);
    let run = store.create_run(cloud_run(&ds, &conn.id, 2)).unwrap();
    until("epoch 1 here", || run_of(&store, &run.id).best);
    drop(store);
    // Meanwhile a crash left two `launching` records: one whose launch created an instance, one whose did not.
    let values = Values::from(BTreeMap::from([("api_key".to_string(), SECRET.to_string())]));
    let spec = |name: &str| LaunchSpec {
        name: name.into(),
        offer_id: "cpu@test".into(),
        disk_gb: 20,
        token: "t".into(),
        lifetime_s: 60,
        worker_base_url: "http://x".into(),
    };
    cloud.launch(&values, &spec("folo-run_gone-a1b2")).unwrap();
    let record = |id: &str, name: &str| {
        let i = json!({ "id": id, "connection_id": conn.id, "provider": "test", "provider_ref": null, "name": name,
                        "run_id": "run_gone", "offer": offer("cpu@test", CloudAvailability::High),
                        "status": "launching", "created": "2026-11-09T00:00:00Z" });
        std::fs::write(home.join("cloud").join("instances").join(format!("{id}.json")), i.to_string())
            .unwrap();
    };
    record("ci_dead", "folo-run_gone-a1b2");
    record("ci_none", "folo-run_gone-c3d4");
    assert_eq!(cloud.live(), 2, "the training run's worker and the orphan");

    let store = plane(&home, &cloud, fast());
    until("the run followed again", || {
        logged(&store, &run.id, "following the run on its worker again").then_some(())
    });
    until("the orphan terminated", || {
        let all = store.cloud_instances().unwrap();
        let status = |id: &str| all.iter().find(|i| i.id == id).map(|i| (i.status, i.cost_usd));
        // The launch that created nothing ends at no cost (its record names no time limit: nothing to watch).
        let ended = (CloudInstanceStatus::Terminated, Some(0.0));
        (status("ci_dead").is_some_and(|s| s.0 == CloudInstanceStatus::Terminated)
            && status("ci_none") == Some(ended))
        .then_some(())
    });
    cloud.hold_epochs(false);
    let done = finished(&store, &run.id);
    assert_eq!(done.status, RunStatus::Completed, "{:?}", log_of(&store, &run.id));
    let m = store.run_metrics(&run.id, &MetricsQuery::default()).unwrap();
    assert_eq!((steps(&m), epochs(&m)), ((1..=4).collect::<Vec<_>>(), vec![1, 2]));
    let worker = cloud.worker(0);
    let w = worker.list_runs().unwrap().remove(0);
    assert_eq!(m, worker.run_metrics(&w.id, &MetricsQuery::default()).unwrap(), "a complete history");
    assert_eq!(cloud.live(), 0);
}

#[test]
fn a_shutdown_before_training_releases_the_instance() {
    no_gpu();
    let t = TestDir::new("cloud-shutdown");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    // The instance never finishes starting.
    cloud.starting.store(u32::MAX, Ordering::SeqCst);
    let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
    until("the instance launched", || (cloud.live() == 1).then_some(()));
    store.shutdown();
    assert_eq!(cloud.live(), 0, "released at the shutdown, not at the next start");
    let stopped = run_of(&store, &run.id);
    assert_eq!((stopped.status, phase(&stopped)), (RunStatus::Stopped, CloudPhase::Released));
    assert!(
        logged(&store, &run.id, "this server shuts down while the run is"),
        "{:?}",
        log_of(&store, &run.id)
    );
    assert_eq!(store.cloud_instances().unwrap()[0].status, CloudInstanceStatus::Terminated);
}

#[test]
fn the_deadline_stops_the_run_and_releases_its_instance() {
    no_gpu();
    let t = TestDir::new("cloud-deadline");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, Tuning { max_run: Some(Duration::from_secs(2)), ..fast() });
    let ds = dataset(&store);
    let conn = connect(&store);
    cloud.hold_epochs(true);
    let run = store.create_run(cloud_run(&ds, &conn.id, 3)).unwrap();
    let done = finished(&store, &run.id);
    assert_eq!(done.status, RunStatus::Stopped, "{:?}", log_of(&store, &run.id));
    assert!(logged(&store, &run.id, "the time limit of 2.0 s"), "{:?}", log_of(&store, &run.id));
    assert_eq!(cloud.live(), 0);
}
