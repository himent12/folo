//! A run's page after the run: its whole history once it ends, and a cloud run's phases and device
//! as they happened.

use std::sync::Arc;

use super::tests::{drive, open, open_with};
use crate::api::mock::MockBackend;
use crate::api::{CloudPhase, RunStatus};
use crate::cloud::{PhaseState, phase_states};
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
        let mut r = this.store.runs.ready().unwrap().iter().find(|r| r.cloud.is_some()).unwrap().clone();
        let mut c = r.cloud.clone().unwrap();
        c.phase = CloudPhase::Released;
        (r.status, r.error_code, r.step, r.epoch) =
            (RunStatus::Failed, Some("cloud_sync_failed".into()), 0, 0);
        let states: Vec<PhaseState> = phase_states(&c, &r).into_iter().map(|(_, s)| s).collect();
        use PhaseState::*;
        assert_eq!(states, [Done, Done, Failed, Skipped, Skipped, Done, Done]);
        // Trained, then the copy back failed: collecting is where.
        (r.step, r.epoch) = (400, 3);
        let states: Vec<PhaseState> = phase_states(&c, &r).into_iter().map(|(_, s)| s).collect();
        assert_eq!(states, [Done, Done, Done, Done, Failed, Done, Done]);
        // Completed: every phase done; live: done up to the current one.
        (r.status, r.error_code) = (RunStatus::Completed, None);
        assert!(phase_states(&c, &r).iter().all(|(_, s)| *s == Done));
        c.phase = CloudPhase::Training;
        let states: Vec<PhaseState> = phase_states(&c, &r).into_iter().map(|(_, s)| s).collect();
        assert_eq!(states, [Done, Done, Done, Now, Next, Next, Next]);
        // Its device until the worker reports one: the GPU once, not "<gpu> at <provider>".
        let provider = this.provider_name(&c.provider);
        r.device = format!("{} at {provider}", c.offer.gpu);
        let said = this.cloud_device(&r, &c);
        assert_eq!(said, format!("{provider} · {}", crate::ui::device_label(&c.offer.gpu)));
        assert!(!said.contains(" at "), "{said}");
    });
}
