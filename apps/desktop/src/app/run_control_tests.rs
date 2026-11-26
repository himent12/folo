//! Stop and Resume on a run's page (Studio's #389, Desktop's side): a stop's answer that comes after
//! the stream moved the run on never puts the older state back, the toast says only what is known,
//! and demo data offers no Resume it would refuse.

use std::sync::Arc;

use folo_ui::{HeadlessApp, WindowHandle};

use super::tests::{drive, open, open_with, wait_for_held};
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
    // Until the answer is in: the run-list read it asks for is then held.
    wait_for_held(&mut app, window, &lists.1, false, "the stop's answer asked for no run list");
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

/// Review round 3: a run-list read asked after the stop and before the end, answered after the end,
/// still showed the run active, so the late stop answer applied over it again ("Stopping X" beside
/// "X finished"). The end drops the answers asked for before it.
#[test]
fn a_run_list_asked_before_the_end_does_not_undo_it() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = open_with(api.clone());
    let stop = api.hold("stop");
    // The run-list reads, in order: the refresh's (before the end), the end's, the stop answer's.
    let (before, end, after) = (api.hold("runs"), api.hold("runs"), api.hold("runs"));
    let tx = step(&mut app, window, 1, |this, cx| {
        let (tx, rx) = std::sync::mpsc::channel();
        this.subscriptions.insert(LIVE.into(), crate::api::Subscription::new(rx).0);
        this.stop_run(LIVE.into(), cx);
        tx
    });
    stop.wait_held();
    step(&mut app, window, 2, |this, cx| this.refresh_runs(cx));
    before.wait_held();
    step(&mut app, window, 3, |this, cx| {
        tx.send(RunEvent::Status(StatusEvent::new(RunStatus::Completed, None))).unwrap();
        this.drain_streams(cx);
    });
    end.wait_held();
    // The read from before the end lands (the demo has the run stopped by then; a server, stopping).
    before.release();
    app.run_until_held(2);
    stop.release();
    wait_for_held(&mut app, window, &after, false, "the stop's answer asked for no run list");
    step(&mut app, window, 2, |this, _| {
        assert_eq!(this.store.run(LIVE).unwrap().status, RunStatus::Completed, "the end stays");
        let titles = titles(this);
        assert!(!titles.iter().any(|t| t.starts_with("Stop")), "no stop toast over the end's: {titles:?}");
    });
    drop((end, after));
    app.run_until_idle();
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
        wait_for_held(&mut app, window, &list, false, "the stop's answer asked for no run list");
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

#[test]
fn clone_config_preserves_gpu1_when_available_and_blocks_when_unavailable() {
    use crate::api::GpuInfo;

    let (mut app, window) = open();
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());

    let mut cfg = drive(&mut app, window, |this, _, _| this.store.runs.ready().unwrap()[0].config.clone());
    cfg.device = "gpu:1".into();

    // 1. On a 2-GPU server, cloning `gpu:1` preserves `"gpu:1"` in the submitted config.
    drive(&mut app, window, {
        let cfg = cfg.clone();
        move |this, w, cx| {
            let sys = this.store.system.ready_mut().unwrap();
            let g0 = sys.gpus[0].clone();
            sys.gpus = vec![g0.clone(), GpuInfo { name: "RTX 4090 Secondary".into(), ..g0 }];
            this.open_new_run_from(&cfg, None, w, cx);
        }
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.nr_device_gone(), None, "gpu:1 is available on a 2-GPU server");
        let out = this.form_config(cx).expect("form config built");
        assert_eq!(out.device, "gpu:1", "cloned gpu:1 must not silently revert to auto");
    });

    // 2. On a 1-GPU server, cloning `gpu:1` flags it unavailable and blocks submission until changed.
    let runs_before = drive(&mut app, window, |this, w, cx| {
        let sys = this.store.system.ready_mut().unwrap();
        sys.gpus.truncate(1);
        this.store.last_gpu_count = Some(1);
        this.open_new_run_from(&cfg, None, w, cx);
        this.store.runs.ready().unwrap().len()
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| {
        let why = this.nr_device_gone().expect("gpu:1 unavailable on 1-GPU server");
        assert!(why.contains("GPU 1"), "message names GPU 1: {why}");
        this.submit_new_run(w, cx);
        assert!(!this.new_run.as_ref().unwrap().submitting, "submission blocked while device is unavailable");
        // Switching to Auto clears custom_device and lifts the block.
        this.new_run.as_mut().unwrap().pick_device(0);
        assert_eq!(this.nr_device_gone(), None);
        assert_eq!(this.form_config(cx).unwrap().device, "auto");
        assert_eq!(this.store.runs.ready().unwrap().len(), runs_before);
    });

    // 3. Unknown device string is also flagged unavailable instead of silently falling back to auto.
    let mut bad = cfg;
    bad.device = "mps".into();
    drive(&mut app, window, move |this, w, cx| {
        this.open_new_run_from(&bad, None, w, cx);
        let why = this.nr_device_gone().expect("unknown device blocks submission");
        assert!(why.contains("mps"), "{why}");
        assert_eq!(this.form_config(cx).unwrap().device, "mps");
    });
}

#[test]
fn remote_export_offers_download_and_local_export_offers_show_in_folder() {
    use std::sync::Arc;

    use crate::api::http::HttpBackend;
    use crate::api::{ApiResult, Backend, ExportResult};
    use crate::app::ServerMode;

    // 1. HttpBackend remote detection & download URL formatting.
    let remote_http = HttpBackend::new("http://192.168.1.50:7878");
    assert!(remote_http.is_remote());
    assert_eq!(
        remote_http.export_download_url("mdl 1", "onnx").as_deref(),
        Some("http://192.168.1.50:7878/api/models/mdl%201/weights?format=onnx")
    );
    assert_eq!(
        remote_http.export_download_url("mdl_1", "folo-f16").as_deref(),
        Some("http://192.168.1.50:7878/api/models/mdl_1/weights?format=f16")
    );
    assert!(!HttpBackend::new("http://127.0.0.1:7878").is_remote());

    // Minimal wrapper over MockBackend to control export path, remote flag, and download URL.
    struct ExportStub {
        inner: crate::api::mock::MockBackend,
        remote: bool,
        path: String,
        download_url: Option<String>,
    }
    impl Backend for ExportStub {
        fn describe(&self) -> String {
            self.inner.describe()
        }
        fn now(&self) -> i64 {
            self.inner.now()
        }
        fn health(&self) -> ApiResult<crate::api::Health> {
            self.inner.health()
        }
        fn system(&self) -> ApiResult<crate::api::SystemResponse> {
            self.inner.system()
        }
        fn datasets(&self) -> ApiResult<Vec<crate::api::Dataset>> {
            self.inner.datasets()
        }
        fn dataset(&self, id: &str) -> ApiResult<crate::api::Dataset> {
            self.inner.dataset(id)
        }
        fn images(&self, id: &str, q: &crate::api::ImageQuery) -> ApiResult<crate::api::ImagePage> {
            self.inner.images(id, q)
        }
        fn image(&self, d: &str, i: &str, t: Option<u32>) -> ApiResult<crate::api::ImageData> {
            self.inner.image(d, i, t)
        }
        fn models(&self) -> ApiResult<Vec<crate::api::Model>> {
            self.inner.models()
        }
        fn model_eval(&self, id: &str) -> ApiResult<crate::api::ModelEval> {
            self.inner.model_eval(id)
        }
        fn export_model(&self, _: &str, _: &str) -> ApiResult<ExportResult> {
            Ok(ExportResult { path: self.path.clone(), bytes: 4096 })
        }
        fn is_remote(&self) -> bool {
            self.remote
        }
        fn export_download_url(&self, _: &str, _: &str) -> Option<String> {
            self.download_url.clone()
        }
        fn runs(&self) -> ApiResult<Vec<crate::api::Run>> {
            self.inner.runs()
        }
        fn run(&self, id: &str) -> ApiResult<crate::api::Run> {
            self.inner.run(id)
        }
        fn create_run(&self, c: crate::api::TrainConfig) -> ApiResult<crate::api::Run> {
            self.inner.create_run(c)
        }
        fn stop_run(&self, id: &str) -> ApiResult<crate::api::Run> {
            self.inner.stop_run(id)
        }
        fn delete_run(&self, id: &str) -> ApiResult<()> {
            self.inner.delete_run(id)
        }
        fn delete_dataset(&self, id: &str, f: bool) -> ApiResult<()> {
            self.inner.delete_dataset(id, f)
        }
        fn metrics(&self, id: &str) -> ApiResult<crate::api::RunMetrics> {
            self.inner.metrics(id)
        }
        fn samples(&self, id: &str, e: Option<u32>) -> ApiResult<crate::api::RunSamples> {
            self.inner.samples(id, e)
        }
        fn logs(&self, id: &str) -> ApiResult<crate::api::RunLogs> {
            self.inner.logs(id)
        }
        fn events(&self, id: &str) -> ApiResult<crate::api::Subscription> {
            self.inner.events(id)
        }
        fn predict(&self, req: &crate::api::PredictRequest) -> ApiResult<crate::api::PredictResponse> {
            self.inner.predict(req)
        }
    }

    // 2. Remote export keeps POSIX server path unmangled and offers Download (not Show in folder).
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        this.set_backend(
            Arc::new(ExportStub {
                inner: crate::api::mock::MockBackend::new(4200, true),
                remote: true,
                path: "/srv/folo/exports/round4.onnx".into(),
                download_url: Some("http://gpu-box:7878/api/models/m1/weights?format=onnx".into()),
            }),
            ServerMode::Local,
            cx,
        );
        this.export_model("m1".into(), "onnx", cx);
    });
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let toast = this.toasts.last().expect("export toast shown");
        assert_eq!(toast.title.as_ref(), "Export complete");
        assert!(
            toast.body.as_deref().unwrap_or("").starts_with("/srv/folo/exports/round4.onnx"),
            "remote POSIX path preserved verbatim: {:?}",
            toast.body
        );
        assert_eq!(
            toast.action.as_ref().map(|(lbl, _)| lbl.as_ref()),
            Some("Download"),
            "remote export offers Download, not Show in folder"
        );
    });

    // 3. Local export whose output exists on disk offers Show in folder.
    let dir = folo_test_support::TestDir::new("export-local");
    let local_file = dir.path().join("round4.onnx");
    std::fs::write(&local_file, b"onnx").unwrap();
    let local_path = local_file.to_string_lossy().into_owned();
    drive(&mut app, window, move |this, _, cx| {
        this.toasts.clear();
        this.set_backend(
            Arc::new(ExportStub {
                inner: crate::api::mock::MockBackend::new(4200, true),
                remote: false,
                path: local_path,
                download_url: Some("http://127.0.0.1:7878/api/models/m1/weights?format=onnx".into()),
            }),
            ServerMode::Local,
            cx,
        );
        this.export_model("m1".into(), "onnx", cx);
    });
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let toast = this.toasts.last().expect("local export toast shown");
        assert_eq!(
            toast.action.as_ref().map(|(lbl, _)| lbl.as_ref()),
            Some("Show in folder"),
            "existing local export offers Show in folder"
        );
    });
}
