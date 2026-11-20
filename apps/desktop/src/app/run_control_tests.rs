//! Stop and Resume on a run's page (Studio's #389, Desktop's side): a stop's answer that comes after
//! the stream moved the run on never puts the older state back, the toast says only what is known,
//! and demo data offers no Resume it would refuse.

use std::sync::Arc;

use folo_ui::{HeadlessApp, WindowHandle};

use super::tests::{drive, open, open_with};
use super::{FoloApp, Route};
use crate::api::mock::MockBackend;
use crate::api::{Run, RunEvent, RunStatus, StatusEvent};
use crate::store::Load;

/// The demo's live run.
const LIVE: &str = "run_a19f";

fn titles(this: &FoloApp) -> Vec<String> {
    this.toasts.iter().map(|t| t.title.to_string()).collect()
}

/// Runs `f` on the app, then lets every fetch answer but the `held` ones the test holds back.
fn step<R>(
    app: &mut HeadlessApp,
    window: WindowHandle<FoloApp>,
    held: usize,
    f: impl FnOnce(&mut FoloApp, &mut folo_ui::Context<FoloApp>) -> R,
) -> R {
    let r = app.update(|cx| window.update(cx, |this, _, cx| f(this, cx)).expect("the window is open"));
    app.run_until_held(held);
    r
}

#[test]
fn a_stop_answered_after_the_stream_said_the_run_completed_leaves_it_completed() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    // The stop's answer is held, and so are the run-list reads after it: the end's, and the stop's.
    let stop = api.hold("stop");
    let lists = (api.hold("runs"), api.hold("runs"));
    let tx = step(&mut app, window, 1, |this, cx| {
        let (tx, rx) = std::sync::mpsc::channel();
        this.subscriptions.insert(LIVE.into(), crate::api::Subscription::new(rx).0);
        this.stop_run(LIVE.into(), cx);
        tx
    });
    stop.wait_held();
    // A stop after the last batch: the stream says the run completed before the answer comes.
    step(&mut app, window, 2, |this, cx| {
        assert_eq!(this.store.run(LIVE).unwrap().status, RunStatus::Stopping, "shown stopping at once");
        tx.send(RunEvent::Status(StatusEvent::new(RunStatus::Completed, None))).unwrap();
        this.drain_streams(cx);
    });
    stop.release();
    // Until the answer is in: the run-list read it asks for is then held. Bounded only to fail clearly
    // (as `Held::wait_held` is) if that read never comes.
    let asked = std::time::Instant::now();
    while !lists.1.is_held() {
        assert!(
            asked.elapsed() < std::time::Duration::from_secs(10),
            "the stop's answer asked for no run list"
        );
        app.run_until_held(2);
        std::thread::yield_now();
    }
    let name = step(&mut app, window, 2, |this, _| {
        let r = this.store.run(LIVE).unwrap();
        assert_eq!(r.status, RunStatus::Completed, "the stream's state stays");
        let titles = titles(this);
        assert!(titles.contains(&format!("{} finished", r.name)), "{titles:?}");
        assert!(!titles.iter().any(|t| t.starts_with("Stop")), "no stop toast over the end's: {titles:?}");
        r.name.clone()
    });
    drop(lists);
    app.run_until_idle();
    drive(&mut app, window, |this, _, _| assert_eq!(titles(this), [format!("{name} finished")]));
}

#[test]
fn a_stop_answered_with_nothing_newer_shown_applies() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api);
    drive(&mut app, window, |this, _, cx| this.stop_run(LIVE.into(), cx));
    drive(&mut app, window, |this, _, _| {
        let r = this.store.run(LIVE).unwrap();
        assert_eq!(r.status, RunStatus::Stopped);
        assert_eq!(titles(this), [format!("Stopped {}", r.name)]);
        assert_eq!(this.toasts[0].body.as_deref(), Some("Last checkpoint kept."));
    });
}

/// The answer comes while the run is shown running or queued again: a list read from before the server
/// took the stop, or the run resumed after the stream said it stopped. The answer is older than that
/// and says nothing then (with "X finished" already shown, a "Stopping X" beside it contradicted it).
#[test]
fn a_stop_answered_while_the_run_is_shown_active_again_adds_no_toast() {
    for again in [RunStatus::Running, RunStatus::Queued] {
        let api = Arc::new(MockBackend::new(4200, true));
        let (mut app, window) = open_with(api.clone());
        let stop = api.hold("stop");
        let list = api.hold("runs");
        step(&mut app, window, 1, |this, cx| this.stop_run(LIVE.into(), cx));
        stop.wait_held();
        step(&mut app, window, 1, |this, _| {
            let runs = this.store.runs.ready_mut().unwrap();
            runs.iter_mut().find(|r| r.id == LIVE).unwrap().status = again;
        });
        stop.release();
        // Until the answer is in: the run-list read it asks for is then held.
        let asked = std::time::Instant::now();
        while !list.is_held() {
            assert!(
                asked.elapsed() < std::time::Duration::from_secs(10),
                "the stop's answer asked for no run list"
            );
            app.run_until_held(1);
            std::thread::yield_now();
        }
        step(&mut app, window, 1, |this, _| {
            assert_eq!(this.store.run(LIVE).unwrap().status, again, "what is shown stays");
            assert!(titles(this).is_empty(), "{again:?}: {:?}", titles(this));
        });
        drop(list);
        app.run_until_idle();
    }
}

/// The toast for `status`, with or without a validated epoch.
fn toast(status: RunStatus, kept: bool, dequeued: bool) -> Option<(String, Option<&'static str>)> {
    let api = MockBackend::new(4200, true);
    let mut r: Run = crate::api::Backend::run(&api, LIVE).unwrap();
    r.status = status;
    r.model_id = None;
    if !kept {
        r.best = None;
    }
    super::actions::stop_toast(&r, dequeued)
}

#[test]
fn a_stop_under_way_with_no_finished_epoch_does_not_yet_say_nothing_is_saved() {
    let (title, body) = toast(RunStatus::Stopping, false, false).unwrap();
    assert_eq!(title, "Stopping shapes-hard-m");
    // An epoch whose last batch is done when the stop comes still validates and saves.
    assert_eq!(
        body,
        Some(
            "Finishing the current batch. No epoch had finished yet. Unless one finishes before it stops, \
             nothing is saved."
        )
    );
    assert!(toast(RunStatus::Stopping, true, false).unwrap().1.unwrap().contains("checkpoint is kept"));
    // Stopped: a checkpoint is claimed only when one is known, and "nothing is saved" never (a stream
    // opened again after the stop does not replay the epoch that finished meanwhile).
    assert_eq!(toast(RunStatus::Stopped, true, false).unwrap().1, Some("Last checkpoint kept."));
    assert_eq!(toast(RunStatus::Stopped, false, false).unwrap().1, None);
    assert_eq!(toast(RunStatus::Stopped, true, true).unwrap().1, Some("It keeps what it trained so far."));
    assert_eq!(toast(RunStatus::Stopped, false, true).unwrap().1, None);
    // Ended (the end had its own toast), or anything else: no stop toast.
    for status in [RunStatus::Completed, RunStatus::Failed, RunStatus::Running, RunStatus::Queued] {
        assert_eq!(toast(status, true, false), None, "{status:?}");
    }
}

/// The focus keys a page draws, once what it asked for has come.
fn drawn_keys(app: &mut HeadlessApp, window: WindowHandle<FoloApp>) -> Vec<String> {
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(app, window, |this, _, _| this.focus.borrow().keys().map(|k| k.to_string()).collect())
}

#[test]
fn demo_data_offers_no_resume_and_says_why() {
    // The built-in demo, like `folo serve --mock`, says `mock` in its health and refuses a resume.
    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        assert!(this.store.health.ready().is_some_and(|h| h.mock));
        this.navigate(Route::Run("run_0b5e".into()), w, cx);
    });
    let keys = drawn_keys(&mut app, window);
    assert!(keys.iter().any(|k| k == "run-resume-disabled"), "{keys:?}");
    assert!(!keys.iter().any(|k| k == "run-resume"), "{keys:?}");
    // A real server's health: Resume is offered.
    drive(&mut app, window, |this, _, _| {
        let Load::Ready(h) = &mut this.store.health else { panic!("health is in") };
        h.mock = false;
    });
    let keys = drawn_keys(&mut app, window);
    assert!(keys.iter().any(|k| k == "run-resume"), "{keys:?}");
}
