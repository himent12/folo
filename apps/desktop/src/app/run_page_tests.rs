//! A run's page after the run: its whole history once it ends, and a cloud run's phases and device
//! as they happened.

use std::sync::Arc;

use super::tests::{drive, open, open_with};
use crate::api::mock::MockBackend;
use crate::api::{CloudPhase, RunStatus};
use crate::cloud::{PhaseState, panel_cost, panel_warning, phase_states};
use crate::store::Load;

#[test]
fn a_run_watched_live_shows_its_whole_history_once_it_ends() {
    let (mut app, window) = open();
    let id = drive(&mut app, window, |this, _, cx| {
        let id = this.store.runs.ready().unwrap().iter().find(|r| !r.status.is_active()).unwrap().id.clone();
        this.ensure_run_data(&id, cx);
        id
    });
    let all = drive(&mut app, window, |this, _, _| this.store.api.metrics(&id).unwrap().steps.len());
    // What the stream gave a page watching it live: every fifth step (the server sends ~10 a second).
    drive(&mut app, window, |this, _, cx| {
        let Some(Load::Ready(d)) = this.store.run_data.get_mut(&id) else { panic!("loaded") };
        let thinned: Vec<_> = d.metrics.steps.iter().step_by(5).cloned().collect();
        d.metrics.steps = thinned;
        this.run_ended(&id, RunStatus::Completed, cx);
    });
    drive(&mut app, window, |this, _, _| {
        let Some(Load::Ready(d)) = this.store.run_data.get(&id) else { panic!("still shown") };
        assert_eq!(d.metrics.steps.len(), all, "the file's every step, not the stream's");
    });
}

#[test]
fn a_cloud_run_that_failed_uploading_did_not_train() {
    let (mut app, window) = open_with(Arc::new(MockBackend::with_cloud_runs(4200, true)));
    drive(&mut app, window, |this, _, _| {
        use PhaseState::*;
        let mut r = this.store.runs.ready().unwrap().iter().find(|r| r.cloud.is_some()).unwrap().clone();
        let mut c = r.cloud.clone().unwrap();
        let marks = |c: &crate::api::RunCloud, r: &crate::api::Run| -> Vec<PhaseState> {
            phase_states(c, r, false).into_iter().map(|(_, s)| s).collect()
        };
        // Released after the upload failed: syncing failed, training and collecting never reached.
        c.phase = CloudPhase::Released;
        c.ended_in = Some(CloudPhase::Syncing);
        (r.status, r.error_code) = (RunStatus::Failed, Some("cloud_sync_failed".into()));
        assert_eq!(marks(&c, &r), [Done, Done, Failed, Skipped, Skipped, Done, Done]);
        // The copy back failed: collecting is where.
        c.ended_in = Some(CloudPhase::Collecting);
        assert_eq!(marks(&c, &r), [Done, Done, Done, Done, Failed, Done, Done]);
        // Stopped while it booted: booting skipped, the rest never reached.
        c.ended_in = Some(CloudPhase::Booting);
        (r.status, r.error_code) = (RunStatus::Stopped, None);
        assert_eq!(marks(&c, &r), [Done, Skipped, Skipped, Skipped, Skipped, Done, Done]);
        // A server that doesn't say where it ended, or a completed run: every phase done.
        c.ended_in = None;
        assert!(marks(&c, &r).iter().all(|s| *s == Done));
        // Live: done up to the current one.
        c.phase = CloudPhase::Training;
        assert_eq!(marks(&c, &r), [Done, Done, Done, Now, Next, Next, Next]);
        // Its device until the worker reports one: the GPU once, not "<gpu> at <provider>".
        let provider = this.provider_name(&c.provider);
        r.device = format!("{} at {provider}", c.offer.gpu);
        let said = this.cloud_device(&r, &c);
        assert_eq!(said, format!("{provider} · {}", crate::ui::device_label(&c.offer.gpu)));
        assert!(!said.contains(" at "), "{said}");
    });
}

#[test]
fn a_history_asked_for_before_the_end_is_replaced_by_the_final_one() {
    // The page opened just before the run ended: its history request was answered then (held
    // here), and lands after the end.
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let held = api.hold("metrics run_52e8");
    app.update(|cx| window.update(cx, |this, _, cx| this.ensure_run_data("run_52e8", cx)).unwrap());
    app.run_until_held(1);
    held.wait_held();
    let key = ("run_52e8".to_string(), "metrics");
    let (asked, final_read) = app.update(|cx| {
        window
            .update(cx, |this, _, cx| {
                let asked = this.history_tickets.get(&key).copied();
                this.run_ended("run_52e8", RunStatus::Completed, cx);
                (asked, this.history_tickets.get(&key).copied())
            })
            .unwrap()
    });
    assert_ne!(
        asked, final_read,
        "the end asks for the history again, and the earlier answer no longer counts"
    );
    drop(held);
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        // The final read installed it (the earlier answer's ticket no longer counts).
        let Some(Load::Ready(_)) = this.store.run_data.get("run_52e8") else { panic!("loaded") };
    });
}

#[test]
fn a_final_read_that_fails_does_not_leave_the_history_loading() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let held = api.hold("metrics run_52e8");
    app.update(|cx| window.update(cx, |this, _, cx| this.ensure_run_data("run_52e8", cx)).unwrap());
    app.run_until_held(1);
    held.wait_held();
    // The final read fails (the server went away for a moment).
    api.fail_metrics_once();
    app.update(|cx| {
        window.update(cx, |this, _, cx| this.run_ended("run_52e8", RunStatus::Completed, cx)).unwrap()
    });
    drop(held);
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| {
        assert!(
            matches!(this.store.run_data.get("run_52e8"), Some(Load::Failed(_))),
            "a failure, not loading for good"
        );
    });
}

#[test]
fn a_released_runs_instance_not_confirmed_terminated_says_it_may_still_be_billed() {
    use PhaseState::*;
    // RunPod hasn't confirmed the termination of the training run's instance (the demo's +trouble).
    let (mut app, window) = open_with(Arc::new(MockBackend::with_cloud_runs(4200, true).cloud_trouble()));
    drive(&mut app, window, |this, _, cx| this.ensure_cloud(cx));
    drive(&mut app, window, |this, _, _| {
        let r = this.store.runs.ready().unwrap().iter().find(|r| r.id == "run_c2e1").unwrap().clone();
        let c = r.cloud.clone().unwrap();
        assert_eq!(c.phase, CloudPhase::Released);
        let i = this.released_instance(&c).expect("its instance in the ledger").clone();
        assert!(i.status.is_alive(), "not confirmed terminated");
        // Releasing is still under way, released still to come.
        let marks: Vec<_> = phase_states(&c, &r, true).into_iter().map(|(_, s)| s).collect();
        assert_eq!(&marks[5..], [Now, Next]);
        // The words promise no more than the app knows, and lead to the instance.
        let (_, words, to_settings) = panel_warning(&c, true, false, 0).unwrap();
        assert_eq!(words, "Termination not confirmed: the instance may still be billed.");
        assert!(to_settings);
        // Its cost is the instance's own, still counting.
        assert_eq!(panel_cost(&c, Some(&i)), format!("{} so far", crate::ui::fmt_cost(i.cost_usd.unwrap())));
        // Confirmed: final, no warning, every phase done.
        let mut gone = i.clone();
        gone.status = crate::api::CloudInstanceStatus::Terminated;
        assert_eq!(panel_cost(&c, Some(&gone)), crate::ui::fmt_cost(i.cost_usd.unwrap()));
        assert!(panel_warning(&c, false, false, 0).is_none());
        // The ledger couldn't be read: that is said, not "terminated".
        let (_, words, _) = panel_warning(&c, false, true, 0).unwrap();
        assert_eq!(words, "Could not check that the instance is terminated.");
    });
}

#[test]
fn a_cloud_run_whose_worker_is_silent_says_so_with_the_time() {
    let (mut app, window) = open_with(Arc::new(MockBackend::with_cloud_runs(4200, true).cloud_silent()));
    drive(&mut app, window, |this, _, _| {
        let r = this.store.runs.ready().unwrap().iter().find(|r| r.id == "run_c2e1").unwrap().clone();
        let c = r.cloud.clone().unwrap();
        assert_eq!(c.phase, CloudPhase::Training);
        let since = c.unanswered_since.as_deref().and_then(crate::api::time::parse).expect("silent");
        let (mark, words, to_settings) = panel_warning(&c, false, false, since).unwrap();
        assert_eq!(mark, "icons/wifi-off.svg");
        assert!(!to_settings);
        let at = crate::api::time::local_deadline(since, since);
        assert_eq!(
            words,
            format!(
                "The worker has not answered since {at}. FOLO keeps asking; the run may still be training or waiting there."
            )
        );
        // A silent worker comes first: it is what the run is waiting on.
        assert_eq!(panel_warning(&c, true, true, since).unwrap().0, "icons/wifi-off.svg");
    });
}
