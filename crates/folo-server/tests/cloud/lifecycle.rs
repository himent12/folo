//! Whole runs, resumes, stops before training, and refused and lost launches.

use std::sync::atomic::Ordering;

use folo_cloud::ErrorKind;
use folo_server::Store;
use folo_server::api::*;
use folo_server::store::MetricsQuery;
use folo_test_support::TestDir;
use serde_json::json;

use super::harness::{
    Call, Fault, TestCloud, cloud_run, connect, dataset, epochs, fast, finished, log_of, logged, no_gpu,
    phase, plane, record, resume, run_of, steps, until,
};

#[test]
fn a_cloud_run_trains_on_an_instance_mirrors_it_and_releases_it() {
    no_gpu();
    let t = TestDir::new("cloud-run");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    assert_eq!((conn.account.as_deref(), conn.balance_usd), (Some("tester@example.com"), Some(12.5)));

    cloud.hold(Call::Launch);
    let run = store.create_run(cloud_run(&ds, &conn.id, 2)).unwrap();
    assert_eq!((run.status, run.queue_position), (RunStatus::Queued, None));
    let c = run.cloud.clone().unwrap();
    assert_eq!(
        (c.phase, c.provider.as_str(), c.offer.id.as_str()),
        (CloudPhase::Provisioning, "test", "cpu@test")
    );
    let events = record(&store, &run.id);
    cloud.release(Call::Launch);
    let done = finished(&store, &run.id);
    let events = events.join().unwrap();
    assert_eq!(done.status, RunStatus::Completed, "{:?}", log_of(&store, &run.id));

    // Phases in order, in the events and in the run; the status stays `queued` until the worker trains.
    let mut phases: Vec<CloudPhase> = Vec::new();
    let mut running_at = None;
    for ev in &events {
        match ev {
            RunEvent::Cloud(c) if phases.last() != Some(&c.phase) => phases.push(c.phase),
            RunEvent::Status(s) if s.status == RunStatus::Running => running_at = Some(phases.clone()),
            _ => {}
        }
    }
    assert_eq!(phases, CloudPhase::ALL);
    assert_eq!(running_at.unwrap().last(), Some(&CloudPhase::Training));
    let c = done.cloud.clone().unwrap();
    assert_eq!(c.phase, CloudPhase::Released);
    assert!(c.instance_id.is_some() && c.deadline.is_some() && c.synced.is_some() && c.cost_usd.is_some());

    // The worker trained the run without `compute`, on its own device, with the dataset sent as FOLO reads it.
    let worker = cloud.worker(0);
    let w = worker.list_runs().unwrap().remove(0);
    assert!(w.config.compute.is_none() && w.config.device == "auto" && w.name.contains(&run.id), "{w:?}");
    assert_eq!((done.device.as_str(), done.steps_per_epoch), (w.device.as_str(), w.steps_per_epoch));
    assert_eq!(done.resolved, w.resolved);
    // Its history, here: every step and epoch, every log line, every sample (with this dataset's ids).
    let all = MetricsQuery::default();
    let (local, remote) =
        (store.run_metrics(&run.id, &all).unwrap(), worker.run_metrics(&w.id, &all).unwrap());
    assert_eq!((steps(&local), epochs(&local)), ((1..=4).collect::<Vec<_>>(), vec![1, 2]));
    assert_eq!(local, remote);
    let here = log_of(&store, &run.id);
    let mut rest = here.iter();
    for line in log_of(&worker, &w.id) {
        assert!(rest.any(|l| *l == line), "the worker's line `{}` is not in the run's log", line.message);
    }
    for phase_line in
        ["launching Test GPU at Test cloud", "uploading the dataset tiny", "copied weights/last.folo"]
    {
        assert!(here.iter().any(|l| l.message.contains(phase_line)), "no `{phase_line}` in {here:?}");
    }
    assert!(
        here.iter().any(|l| l.message.starts_with("terminated the instance; it cost about $")),
        "{here:?}"
    );
    for epoch in [1, 2] {
        let (a, b) = (
            store.run_samples(&run.id, Some(epoch)).unwrap(),
            worker.run_samples(&w.id, Some(epoch)).unwrap(),
        );
        assert_eq!(a.samples.len(), b.samples.len());
        for (x, y) in a.samples.iter().zip(&b.samples) {
            assert_eq!((&x.image_id, &x.truth, &x.pred), (&y.image_id, &y.truth, &y.pred));
            assert_eq!(x.dataset_id, ds.id);
            store.image(&ds.id, &x.image_id, None).unwrap();
        }
    }
    // The weights are the run's models.
    let model = done.model_id.clone().unwrap();
    for id in [model.clone(), format!("{model}-last")] {
        assert_eq!(store.get_model(&id).unwrap().run_id.as_deref(), Some(run.id.as_str()));
    }
    // And nothing else is next to them: a copy that was under way when the run ended leaves no staged file.
    let weights = t.join("home").join("runs").join(&run.id).join("weights");
    until("only the two weights files", || {
        let mut names: Vec<String> = std::fs::read_dir(&weights)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        (names == ["best.folo", "last.folo"]).then_some(())
    });
    // Nothing is left at the provider; the ledger says what it cost.
    assert_eq!(cloud.live(), 0);
    let instances = store.cloud_instances().unwrap();
    assert_eq!(instances.len(), 1);
    let i = &instances[0];
    assert_eq!(
        (i.status, i.run_id.as_deref(), i.provider_ref.as_deref()),
        (CloudInstanceStatus::Terminated, Some(run.id.as_str()), Some("pod-1"))
    );
    assert!(i.cost_usd.is_some() && i.ended.is_some() && i.name.starts_with(&format!("folo-{}-", run.id)));
    assert_eq!(cloud.calls(Call::Terminate), 1);
}

#[test]
fn a_lost_or_stopped_run_resumes_on_a_new_instance_and_then_here() {
    no_gpu();
    let t = TestDir::new("cloud-resume");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    cloud.hold_epochs(true);
    let run = store.create_run(cloud_run(&ds, &conn.id, 3)).unwrap();

    // The worker dies after its first checkpoint was copied: the run is lost, not the checkpoint.
    until("the first checkpoint copied", || run_of(&store, &run.id).cloud.and_then(|c| c.synced));
    cloud.kill(0);
    let lost = finished(&store, &run.id);
    assert_eq!((lost.status, lost.error_code.as_deref()), (RunStatus::Failed, Some("cloud_instance_lost")));
    let dir = t.join("home").join("runs").join(&run.id);
    assert!(dir.join("weights").join("last.folo").is_file(), "the checkpoint was collected");
    assert_eq!(cloud.live(), 0);

    // On a new instance, from the checkpoint: the step numbers go on. Stopped there, it is collected and released.
    let resumed = resume(&store, &run.id, json!({}));
    assert_eq!((resumed.status, phase(&resumed)), (RunStatus::Queued, CloudPhase::Provisioning));
    until("epoch 2 here", || {
        Some(run_of(&store, &run.id)).filter(|r| {
            r.best.is_some_and(|_| r.epoch >= 2)
                && store.run_metrics(&run.id, &MetricsQuery::default()).unwrap().epochs.len() == 2
        })
    });
    store.stop_run(&run.id).unwrap();
    let again = finished(&store, &run.id);
    assert_eq!((again.status, again.epoch, phase(&again)), (RunStatus::Stopped, 2, CloudPhase::Released));
    assert_eq!(store.cloud_instances().unwrap().len(), 2);
    assert!(logged(&store, &run.id, "uploading the checkpoint weights/last.folo"));

    // Here: `compute: null` resumes on this machine, as a run of this machine.
    cloud.hold_epochs(false);
    let here = resume(&store, &run.id, json!({ "compute": null }));
    assert!(here.config.compute.is_none() && here.cloud.is_none());
    let done = finished(&store, &run.id);
    assert_eq!((done.status, done.epoch), (RunStatus::Completed, 3));
    let m = store.run_metrics(&run.id, &MetricsQuery::default()).unwrap();
    assert_eq!((steps(&m), epochs(&m)), ((1..=6).collect::<Vec<_>>(), vec![1, 2, 3]));
    assert_eq!(cloud.live(), 0);
}

#[test]
fn a_stop_before_training_releases_at_once() {
    no_gpu();
    let t = TestDir::new("cloud-early-stop");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    let stopped = |phase_now: CloudPhase, hold: &dyn Fn(), go: &dyn Fn()| {
        hold();
        let run = store.create_run(cloud_run(&ds, &conn.id, 2)).unwrap();
        until(&format!("{phase_now:?}"), || (phase(&run_of(&store, &run.id)) == phase_now).then_some(()));
        let r = store.stop_run(&run.id).unwrap();
        assert_eq!(r.status, RunStatus::Stopping);
        go();
        let done = finished(&store, &run.id);
        assert_eq!((done.status, done.model_id.as_deref()), (RunStatus::Stopped, None));
        assert!(logged(&store, &run.id, "stopped before the worker trained"));
        assert_eq!(cloud.live(), 0, "released while {phase_now:?}");
    };
    stopped(CloudPhase::Provisioning, &|| cloud.hold(Call::Launch), &|| cloud.release(Call::Launch));
    stopped(CloudPhase::Booting, &|| cloud.hold(Call::State), &|| cloud.release(Call::State));
    let g = cloud.gates.clone();
    stopped(CloudPhase::Syncing, &|| g.uploads.store(true, Ordering::SeqCst), &|| {
        g.uploads.store(false, Ordering::SeqCst)
    });
    let all = store.cloud_instances().unwrap();
    assert_eq!(all.len(), 3);
    assert!(all.iter().all(|i| i.status == CloudInstanceStatus::Terminated), "{all:?}");
}

#[test]
fn refused_and_lost_launches() {
    no_gpu();
    let t = TestDir::new("cloud-launch");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);

    // Refused: no instance, and no record of one.
    cloud.fail(Call::Launch, Fault::Fail(ErrorKind::Refused), 1);
    let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
    let done = finished(&store, &run.id);
    assert_eq!((done.status, done.error_code.as_deref()), (RunStatus::Failed, Some("cloud_refused")));
    assert!(done.cloud.unwrap().instance_id.is_none());
    assert!(store.cloud_instances().unwrap().is_empty());

    // Created, but the answer was lost: found by its name and used.
    cloud.fail(Call::Launch, Fault::LostAnswer, 1);
    let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
    let done = finished(&store, &run.id);
    assert_eq!(done.status, RunStatus::Completed, "{:?}", log_of(&store, &run.id));
    assert!(logged(&store, &run.id, "but the instance exists; using it"));
    assert_eq!(cloud.calls(Call::Launch), 2, "one instance, not a second launch");
    assert_eq!(cloud.live(), 0);

    // Lost, and the provider cannot say whether it exists: the run fails, and the release looks it up again.
    cloud.fail(Call::Launch, Fault::LostAnswer, 1);
    cloud.fail(Call::Find, Fault::Fail(ErrorKind::Unavailable), 1);
    let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
    let done = finished(&store, &run.id);
    assert_eq!(done.error_code.as_deref(), Some("cloud_unavailable"));
    until("the lost launch released", || (cloud.live() == 0).then_some(()));
}

#[test]
fn a_lost_launch_that_the_provider_lists_late_is_still_terminated() {
    no_gpu();
    let t = TestDir::new("cloud-late-listing");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&t.join("home"), &cloud, fast());
    let ds = dataset(&store);
    let conn = connect(&store);
    // The launch created an instance and its answer was lost, and the provider does not list the instance yet.
    cloud.fail(Call::Launch, Fault::LostAnswer, 1);
    cloud.fail(Call::Find, Fault::Hidden, usize::MAX);
    let run = store.create_run(cloud_run(&ds, &conn.id, 1)).unwrap();
    let done = finished(&store, &run.id);
    assert_eq!(done.error_code.as_deref(), Some("cloud_unavailable"));
    assert_eq!(cloud.live(), 1);
    // Not finding it is no proof that there is none: its record stays, so it goes once it is listed.
    let all = store.cloud_instances().unwrap();
    assert_eq!(all.len(), 1, "the record of the launch is kept");
    assert_eq!(all[0].status, CloudInstanceStatus::Terminating);
    cloud.clear(Call::Find);
    let i = until("the instance terminated", || {
        store.cloud_instances().unwrap().into_iter().find(|i| i.status == CloudInstanceStatus::Terminated)
    });
    assert_eq!((cloud.live(), i.error), (0, None));
}
