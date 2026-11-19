//! Unit tests of the simulated runs: checkpoints of stopped runs, the curves, the live clock, stopping and the phase
//! a cloud run ended in.

use folo_api::CloudPhase;

use super::*;
use crate::mock::cloud::{PHASE_S, offers};

fn config(epochs: u32) -> TrainConfig {
    serde_json::from_value(serde_json::json!({ "dataset_id": "ds", "epochs": epochs })).unwrap()
}

fn user_run(offset_s: f64) -> SimRun {
    let cfg = config(10);
    let curve = Curve::new(3, 0.5, 1, &cfg, 10);
    let clock = LiveClock::new(1.0, 0.0, 0.0, false, offset_s);
    let t = Timeline::Live(Mutex::new(clock));
    SimRun::new("run_x".into(), cfg, vec!["a".into()], 10, curve, "cpu".into(), Some("m_x".into()), true, t)
}

#[test]
fn stopped_runs_keep_a_checkpoint_after_an_epoch() {
    // Stopped mid-epoch 1: nothing validated, no model.
    let early = user_run(0.5);
    assert!(early.stop());
    assert_eq!(early.snap().status, RunStatus::Stopping);
    early.pass_time(Duration::from_secs_f64(STOP_S + 0.1));
    let snap = early.snap();
    assert_eq!(snap.status, RunStatus::Stopped);
    assert_eq!(early.current_model(&snap), None);
    // Stopped in epoch 3: the last checkpoint is a model.
    let later = user_run(2.5);
    assert_eq!(later.current_model(&later.snap()), None, "running runs expose no model yet");
    assert!(later.stop());
    later.pass_time(Duration::from_secs_f64(STOP_S + 0.1));
    let snap = later.snap();
    assert_eq!(snap.status, RunStatus::Stopped);
    assert_eq!(later.to_api().model_id.as_deref(), Some("m_x"));
}

#[test]
fn curves_are_plausible() {
    let c = Curve::new(1, 0.7, 3, &config(100), 50);
    // Warmup then cosine.
    assert!(c.lr(1) < c.lr(100));
    assert!((c.lr(150) - 0.01).abs() < 1e-4);
    assert!(c.lr(5000) < 0.0002);
    // Loss decays.
    let e1 = c.epoch(1, 1.0);
    let e100 = c.epoch(100, 1.0);
    assert!(e100.train_loss.total < e1.train_loss.total * 0.6);
    assert!(
        (e100.train_loss.total - (e100.train_loss.box_ + e100.train_loss.cls + e100.train_loss.dfl)).abs()
            < 1e-3
    );
    // mAP rises and saturates near quality.
    assert!(c.val(1).map50_95 < 0.1);
    assert!((c.val(100).map50_95 - 0.7).abs() < 0.04);
    assert!(c.val(100).map50 > c.val(100).map50_95);
    assert_eq!(c.val(5).per_class.len(), 3);
    let best = c.best(100).unwrap();
    assert!(best.epoch > 60);
}

#[test]
fn live_clock_phases_and_stop() {
    let cfg = config(10);
    let curve = Curve::new(2, 0.5, 1, &cfg, 10);
    let mut clock = LiveClock::new(1.0, 2.0, 3.0, true, 0.0);
    let t0 = clock.origin;
    let at = |c: &mut LiveClock, s: f64| c.snap(&curve, t0 + Duration::from_secs_f64(s));
    assert_eq!(at(&mut clock, 1.0).status, RunStatus::Queued);
    let s = at(&mut clock, 4.55);
    assert_eq!((s.status, s.step), (RunStatus::Running, 25));
    assert_eq!(at(&mut clock, 12.5).status, RunStatus::Completed);
    // Loops after the hold.
    let s = at(&mut clock, 15.5);
    assert_eq!((s.status, s.cycle), (RunStatus::Queued, 1));
    // A stopped looping run restarts after the hold, too.
    clock.stop_at = Some(0.5);
    assert_eq!(at(&mut clock, 18.0).status, RunStatus::Stopped);
    let s = at(&mut clock, 21.0);
    assert_eq!((s.status, s.cycle), (RunStatus::Queued, 2));
}

#[test]
fn stop_sequence() {
    let cfg = config(10);
    let curve = Curve::new(2, 0.5, 1, &cfg, 10);
    let mut clock = LiveClock::new(1.0, 0.0, 100.0, false, 0.0);
    let t0 = clock.origin;
    clock.stop_at = Some(5.0);
    let s = clock.snap(&curve, t0 + Duration::from_secs_f64(5.5));
    assert_eq!((s.status, s.step), (RunStatus::Stopping, 50));
    let s = clock.snap(&curve, t0 + Duration::from_secs_f64(8.0));
    assert_eq!((s.status, s.step), (RunStatus::Stopped, 50));
    assert!(s.finished.is_some());
}

/// A cloud run as the store creates one (3 phases of [`PHASE_S`] before it trains 3 epochs of 6 s, from 6 s to 24 s,
/// then collecting and releasing), stopped at `stop` and its instance lost at `lost` (seconds after its creation): its
/// last status and every `cloud.ended_in` it showed, read every 0.25 s for 100 s from the first of them.
fn cloud_end(stop: Option<f64>, lost: Option<f64>) -> (RunStatus, Vec<CloudPhase>) {
    let cfg = config(3);
    let curve = Curve::new(3, 0.5, 1, &cfg, 10);
    let mut clock = LiveClock::new(6.0, 3.0 * PHASE_S, 0.0, false, 0.0).with_tail(2.0 * PHASE_S);
    (clock.stop_at, clock.lost_at) = (stop, lost);
    let t0 = clock.origin;
    let t = Timeline::Live(Mutex::new(clock));
    let run = SimRun::new("run_x".into(), cfg, vec!["a".into()], 10, curve, "cpu".into(), None, true, t)
        .with_cloud(SimCloud::new("run_x", "cc_x", offers().remove(0), 1.0));
    let Timeline::Live(c) = &run.timeline else { unreachable!() };
    // From the first of the two on (the clock knows them in advance).
    let from = [stop, lost].into_iter().flatten().reduce(f64::min).unwrap_or(0.0);
    let (mut status, mut seen) = (RunStatus::Queued, Vec::new());
    for i in 0..400 {
        let snap = c.lock().unwrap().snap(&run.curve, t0 + Duration::from_secs_f64(from + i as f64 * 0.25));
        status = snap.status;
        if let Some(p) = run.cloud_at(&snap).unwrap().ended_in
            && seen.last() != Some(&p)
        {
            seen.push(p);
        }
    }
    (status, seen)
}

#[test]
fn a_cloud_run_ends_in_the_phase_its_end_was_decided_in() {
    use CloudPhase::*;
    use RunStatus::{Completed, Failed, Stopped};
    for (k, phase) in [Provisioning, Booting, Syncing].into_iter().enumerate() {
        let mid = (k as f64 + 0.5) * PHASE_S;
        assert_eq!(cloud_end(Some(mid), None), (Stopped, vec![phase]));
        assert_eq!(cloud_end(None, Some(mid)), (Failed, vec![phase]));
    }
    // A lost instance cuts the training short; a stop ends it on the worker, which hands its weights over.
    assert_eq!(cloud_end(None, Some(10.0)), (Failed, vec![Training]));
    assert_eq!(cloud_end(Some(10.0), None), (Stopped, vec![Collecting]));
    // Lost while it releases after a stop: the stop decided where it ended, even when the training would have started
    // between the two.
    assert_eq!(cloud_end(Some(5.9), Some(6.5)), (Failed, vec![Syncing]));
    assert_eq!(cloud_end(Some(10.0), Some(10.5)), (Failed, vec![Collecting]));
    // Lost while it collects, and never stopped.
    assert_eq!(cloud_end(None, Some(25.0)), (Failed, vec![Collecting]));
    assert_eq!(cloud_end(None, None), (Completed, vec![Collecting]));
}
