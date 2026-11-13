//! Instances whose launch's answer was lost: listed late, never listed, and the watchdog.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use folo_cloud::{LaunchSpec, Provider, Values};
use folo_server::api::*;
use folo_server::cloud::Tuning;
use folo_server::{RealStore, Store};
use folo_test_support::TestDir;
use serde_json::json;

use super::harness::{Call, SECRET, TestCloud, connect, fast, no_gpu, offer, plane, until};

/// Unix seconds as the ledger writes times (`YYYY-MM-DDTHH:MM:SSZ`).
fn rfc3339(secs: f64) -> String {
    let secs = secs.floor() as i64;
    let (days, tod) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let (era, doe) = (z.div_euclid(146_097), z.rem_euclid(146_097));
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", tod / 3600, (tod / 60) % 60, tod % 60)
}

fn unix_now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64()
}

/// Seconds an instance's watchdog adds to its run's deadline at most, as the ledger counts them: its setup's start
/// (15 minutes), the grace and the time it keeps trying to terminate itself.
const WATCHDOG_AFTER_DEADLINE: f64 =
    900.0 + (folo_cloud::bootstrap::GRACE_S + folo_cloud::bootstrap::TERMINATE_TRIES_S) as f64;

/// Writes the `launching` record of an instance named `name` whose launch's answer was lost (no provider id),
/// written `age` seconds ago, with the run's deadline `deadline_in` seconds from now.
fn lost_launch(home: &Path, conn: &str, id: &str, name: &str, age: f64, deadline_in: f64) {
    let now = unix_now();
    let i = json!({ "id": id, "connection_id": conn, "provider": "test", "provider_ref": null, "name": name,
                    "run_id": null, "offer": offer("cpu@test", CloudAvailability::High), "status": "launching",
                    "created": rfc3339(now - age), "deadline": rfc3339(now + deadline_in) });
    let dir = home.join("cloud").join("instances");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(format!("{id}.json")), i.to_string()).unwrap();
}

fn instance(store: &RealStore, id: &str) -> Option<CloudInstance> {
    store.cloud_instances().unwrap().into_iter().find(|i| i.id == id)
}

#[test]
fn instances_listed_after_the_wait_are_still_terminated() {
    no_gpu();
    let t = TestDir::new("cloud-listed-late");
    let home = t.join("home");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&home, &cloud, Tuning { unseen_every: Duration::from_millis(100), ..fast() });
    let conn = connect(&store);
    let values = Values::from(BTreeMap::from([("api_key".to_string(), SECRET.to_string())]));
    // Launched (and their answers lost) 4 minutes, 6 minutes and 2 hours ago, and one whose watchdog stops it in
    // half a minute; the provider does not list any of them yet.
    let cases = [
        ("ci_four", 240.0, 3600.0),
        ("ci_six", 360.0, 3600.0),
        ("ci_hours", 7200.0, 3600.0),
        ("ci_last", 7200.0, 30.0 - WATCHDOG_AFTER_DEADLINE),
    ];
    for (id, age, deadline_in) in cases {
        let name = format!("folo-run_late-{id}");
        cloud.hide(&name, true);
        let spec = LaunchSpec {
            name: name.clone(),
            offer_id: "cpu@test".into(),
            disk_gb: 20,
            token: "t".into(),
            lifetime_s: 3600,
            worker_base_url: "http://x".into(),
        };
        cloud.launch(&values, &spec).unwrap();
        lost_launch(&home, &conn.id, id, &name, age, deadline_in);
    }
    // The one launched 4 minutes ago is waited for; the others end, as launches that created nothing.
    let ended = until("the records looked at", || {
        let all = store.cloud_instances().unwrap();
        let alive = |id: &str| all.iter().find(|i| i.id == id).map(|i| i.status.is_alive());
        let others = ["ci_six", "ci_hours", "ci_last"];
        (alive("ci_four") == Some(true) && others.iter().all(|i| alive(i) != Some(true))).then_some(all)
    });
    assert_eq!(cloud.live(), 4);

    // Then the provider lists them: every one is terminated, and costs what it ran from its launch.
    for (id, ..) in cases {
        cloud.hide(&format!("folo-run_late-{id}"), false);
    }
    until("every instance terminated", || (cloud.live() == 0).then_some(()));
    for (id, age, _) in cases {
        let i = until(id, || instance(&store, id).filter(|i| i.status == CloudInstanceStatus::Terminated));
        let ran = age / 3600.0 * 0.36;
        assert!(i.cost_usd.unwrap() >= ran - 1e-3 && i.error.is_none(), "{id}: {i:?}");
    }
    // Before they were found, nothing was counted for them: nobody had seen them.
    for i in ended.iter().filter(|i| i.id != "ci_four") {
        assert_eq!((i.ended.as_ref(), i.cost_usd), (Some(&i.created), Some(0.0)), "{i:?}");
    }
}

#[test]
fn an_instance_never_listed_ends_at_no_cost_and_is_still_looked_for_after_its_watchdog_is_due() {
    no_gpu();
    let t = TestDir::new("cloud-never-listed");
    let home = t.join("home");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&home, &cloud, Tuning { unseen_every: Duration::from_millis(100), ..fast() });
    let (kept, removed) = (connect(&store), connect(&store));
    // Two launches whose answers were lost 6 minutes ago, and nothing the provider ever lists: one whose watchdog
    // was due a second ago, one an hour from now on a connection that is removed meanwhile.
    lost_launch(&home, &kept.id, "ci_kept", "folo-run_never-a", 360.0, -1.0 - WATCHDOG_AFTER_DEADLINE);
    lost_launch(&home, &removed.id, "ci_gone", "folo-run_never-b", 360.0, 3600.0);
    for id in ["ci_kept", "ci_gone"] {
        let i = until(id, || instance(&store, id).filter(|i| !i.status.is_alive()));
        assert_eq!(
            (i.status, i.ended.as_ref(), i.cost_usd),
            (CloudInstanceStatus::Terminated, Some(&i.created), Some(0.0))
        );
    }
    // Ended, so its connection can go; the looks for it then say why they cannot be made.
    store.disconnect_cloud(&removed.id).unwrap();
    let gone = until("the failed look", || instance(&store, "ci_gone").filter(|i| i.error.is_some()));
    let e = gone.error.unwrap();
    assert!(e.contains("never listed") && e.contains(&removed.id), "{e}");
    assert_eq!(gone.status, CloudInstanceStatus::Terminated);
    // The other's watchdog is due already, and it is still looked for (a watchdog can fail): its record names the
    // end of the watch, a week after its launch.
    until("looked for again", || (cloud.calls_with(Call::Find, "folo-run_never-a") >= 2).then_some(()));
    let file = home.join("cloud").join("instances").join("ci_kept.json");
    assert!(std::fs::read_to_string(&file).unwrap().contains("unseen_until"), "still watched");
    let kept_i = instance(&store, "ci_kept").unwrap();
    assert_eq!(
        (kept_i.status, kept_i.cost_usd, kept_i.error),
        (CloudInstanceStatus::Terminated, Some(0.0), None)
    );
    store.disconnect_cloud(&kept.id).unwrap();
}

#[test]
fn a_failed_look_for_a_never_found_instance_is_made_again_at_the_next_sweep() {
    no_gpu();
    let t = TestDir::new("cloud-look-failed");
    let home = t.join("home");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&home, &cloud, fast());
    let conn = connect(&store);
    // Never found: the first look (its 1st find) ends it, the next one (the 2nd) fails once.
    let name = "folo-run_flaky-a1b2";
    cloud.script_find(name, &[2], &[]);
    lost_launch(&home, &conn.id, "ci_flaky", name, 360.0, 3600.0);
    // Not ten minutes later: at the next sweep.
    until("a third look", || (cloud.calls_with(Call::Find, name) >= 3).then_some(()));
    let i = until("the error cleared", || instance(&store, "ci_flaky").filter(|i| i.error.is_none()));
    assert_eq!((i.status, i.cost_usd), (CloudInstanceStatus::Terminated, Some(0.0)));
}

#[test]
fn an_instance_listed_after_its_watchdog_was_due_is_still_found_and_terminated() {
    no_gpu();
    let t = TestDir::new("cloud-listed-later");
    let home = t.join("home");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&home, &cloud, fast());
    let conn = connect(&store);
    let name = "folo-run_later-a1b2";
    let values = Values::from(BTreeMap::from([("api_key".to_string(), SECRET.to_string())]));
    let spec = LaunchSpec {
        name: name.into(),
        offer_id: "cpu@test".into(),
        disk_gb: 20,
        token: "t".into(),
        lifetime_s: 3600,
        worker_base_url: "http://x".into(),
    };
    cloud.launch(&values, &spec).unwrap();
    // Launched two hours ago; its watchdog was due a minute ago and did not stop it. Not listed at first.
    cloud.script_find(name, &[], &[1]);
    lost_launch(&home, &conn.id, "ci_later", name, 7200.0, -60.0 - WATCHDOG_AFTER_DEADLINE);
    until("the instance terminated", || (cloud.live() == 0).then_some(()));
    let i = until("its record", || {
        instance(&store, "ci_later").filter(|i| i.status == CloudInstanceStatus::Terminated)
    });
    assert!(i.cost_usd.unwrap() >= 2.0 * 0.36 - 1e-3, "costs what it ran from its launch: {i:?}");
}

#[test]
fn the_watch_ends_only_with_a_last_look_that_succeeds() {
    no_gpu();
    let t = TestDir::new("cloud-last-look");
    let home = t.join("home");
    let cloud = TestCloud::new(&t.join("workers"));
    let store = plane(&home, &cloud, fast());
    let conn = connect(&store);
    // Written eight days ago, its watchdog long due: the watch is over as soon as it starts. The first find ends
    // it; the last look (the 2nd) fails, so it is made again (the 3rd) before the record goes.
    let name = "folo-run_old-a1b2";
    cloud.script_find(name, &[2], &[]);
    lost_launch(&home, &conn.id, "ci_old", name, 8.0 * 86_400.0, -8.0 * 86_400.0);
    until("the record pruned", || instance(&store, "ci_old").is_none().then_some(()));
    assert_eq!(cloud.calls_with(Call::Find, name), 3, "looked at last, and again after the failed look");
}
