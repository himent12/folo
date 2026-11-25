//! Connecting from Settings and losing the local server, with a fake connector: no socket is opened
//! and no `folo serve` is started.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::tests::{drive, open};
use super::{Connector, ServerMode};
use crate::api::Backend;
use crate::api::http::{Probe, Spawned};
use crate::api::mock::MockBackend;
use crate::store::Load;

fn demo(_: &str) -> Arc<dyn Backend> {
    Arc::new(MockBackend::new(4200, true))
}

/// Nothing answers anywhere; the data dir's server is at 127.0.0.1:4100.
fn found_local(_: std::time::Duration) -> Result<Spawned, String> {
    Ok(Spawned::Running("http://127.0.0.1:4100".into()))
}

fn titles(this: &super::FoloApp) -> Vec<String> {
    this.toasts.iter().map(|t| t.title.to_string()).collect()
}

#[test]
fn a_typed_address_nothing_answers_says_where_the_app_went_instead() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        this.connector = Connector { probe: |_| Probe::Nothing, start: found_local, backend: demo };
        this.server_url.update(cx, |i, cx| i.set_text("http://127.0.0.1:1", cx));
        this.switch_source(ServerMode::Local, cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.server_url.read(cx).text(), "http://127.0.0.1:4100");
        assert_eq!(this.local_url.as_deref(), Some("http://127.0.0.1:4100"));
        assert!(titles(this).contains(&"Nothing answered at 127.0.0.1:1".to_string()), "{:?}", titles(this));
    });
}

static PAGE_STARTS: AtomicUsize = AtomicUsize::new(0);

/// #382 (seen in #365): with automatic start-up on, a typed address that answered with a web page
/// was taken for nothing there, so the app went to the local server and said "Nothing answered".
/// It is connected as typed, as with start-up off, and its own answer says what is wrong. At launch
/// such an answer still falls through to the local server.
#[test]
fn a_typed_address_that_answers_but_not_as_folo_is_kept_and_starts_nothing() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        this.connector = Connector {
            probe: |_| Probe::Answered,
            start: |t| {
                PAGE_STARTS.fetch_add(1, Ordering::SeqCst);
                found_local(t)
            },
            backend: demo,
        };
        this.server_url.update(cx, |i, cx| i.set_text("http://127.0.0.1:4100/subpath", cx));
        this.switch_source(ServerMode::Local, cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(PAGE_STARTS.load(Ordering::SeqCst), 0, "no local server in its place");
        assert_eq!(this.server_url.read(cx).text(), "http://127.0.0.1:4100/subpath");
        assert_eq!(this.connected_url.as_deref(), Some("http://127.0.0.1:4100/subpath"));
        assert_eq!(this.local_url, None);
        assert!(!titles(this).iter().any(|t| t.starts_with("Nothing answered")), "{:?}", titles(this));
        // At launch the same answer falls through, as on main: the app starts (here finds) its own.
        this.autoconnect("http://127.0.0.1:7878".into(), true, super::Reason::Launch, cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert_eq!(PAGE_STARTS.load(Ordering::SeqCst), 1);
        assert_eq!(this.local_url.as_deref(), Some("http://127.0.0.1:4100"));
    });
}

static REMOTE_STARTS: AtomicUsize = AtomicUsize::new(0);

#[test]
fn another_machines_address_is_kept_and_no_local_server_takes_its_place() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        this.connector = Connector {
            probe: |_| Probe::Nothing,
            start: |t| {
                REMOTE_STARTS.fetch_add(1, Ordering::SeqCst);
                found_local(t)
            },
            backend: demo,
        };
        this.server_url.update(cx, |i, cx| i.set_text("http://192.168.9.9:7878", cx));
        this.switch_source(ServerMode::Local, cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(REMOTE_STARTS.load(Ordering::SeqCst), 0, "no local server for another machine's address");
        assert_eq!(this.server_url.read(cx).text(), "http://192.168.9.9:7878");
        assert_eq!(this.local_url, None);
        assert_eq!(this.server_mode, ServerMode::Local);
    });
}

static BAD_STARTS: AtomicUsize = AtomicUsize::new(0);

/// Review round 3: this machine's address with a port that can't be used started (or found) the
/// local server and wrote its address over the typed one, so Settings never said what was wrong.
#[test]
fn a_local_address_that_is_not_one_is_kept_and_starts_nothing() {
    for typed in ["http://127.0.0.1:99999", "localhost:abc", "http://[::1]:0"] {
        let (mut app, window) = open();
        drive(&mut app, window, move |this, _, cx| {
            assert!(this.spawn_server, "the default: start a local server when nothing answers");
            this.connector = Connector {
                probe: |_| Probe::Nothing,
                start: |t| {
                    BAD_STARTS.fetch_add(1, Ordering::SeqCst);
                    found_local(t)
                },
                backend: |u| Arc::new(crate::api::http::HttpBackend::new(u)),
            };
            this.server_url.update(cx, |i, cx| i.set_text(typed, cx));
            this.switch_source(ServerMode::Local, cx);
        });
        drive(&mut app, window, move |this, _, cx| {
            assert_eq!(BAD_STARTS.load(Ordering::SeqCst), 0, "no local server for {typed:?}");
            assert_eq!(this.server_url.read(cx).text(), typed);
            assert_eq!(this.local_url, None);
            assert_eq!(this.connected_url.as_deref(), Some(typed));
            assert!(this.store.api.health().unwrap_err().is_bad_address(), "{typed:?}");
        });
        // d4-bench-settings-4: the "Can't reach folo serve, retrying…" pill went up for it, though
        // nothing is sent and no retry can help; the sidebar says "Not a server address".
        drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
        drive(&mut app, window, move |this, _, _| {
            assert!(this.store.unreachable() && this.store.bad_address(), "{typed:?}");
            assert!(!this.lost_banner_shown(), "no retrying pill for {typed:?}");
        });
    }
}

/// The local server stops answering (the window that started it closed, or it crashed): the 2 s
/// refresh sees it, and the app finds or starts one again instead of retrying a dead port.
fn lose_the_server(local: Option<&str>) -> (folo_ui::HeadlessApp, folo_ui::WindowHandle<super::FoloApp>) {
    lose_the_server_with(local, Connector { probe: |_| Probe::Nothing, start: found_local, backend: demo })
}

fn lose_the_server_with(
    local: Option<&str>,
    connector: Connector,
) -> (folo_ui::HeadlessApp, folo_ui::WindowHandle<super::FoloApp>) {
    let (mut app, window) = open();
    let local = local.map(str::to_string);
    drive(&mut app, window, move |this, _, cx| {
        this.connector = connector;
        this.server_mode = ServerMode::Local;
        this.local_url = local;
        this.server_url.update(cx, |i, cx| i.set_text("http://127.0.0.1:4000", cx));
        this.store.api = Arc::new(MockBackend::new(4200, true).failing());
        this.store.system = Load::Idle;
        this.ensure_core(cx);
    });
    (app, window)
}

#[test]
fn a_lost_local_server_is_found_or_started_again() {
    let (mut app, window) = lose_the_server(Some("http://127.0.0.1:4000"));
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.server_url.read(cx).text(), "http://127.0.0.1:4100");
        assert_eq!(this.local_url.as_deref(), Some("http://127.0.0.1:4100"), "and again next time");
        assert!(!this.store.unreachable());
        assert!(titles(this).contains(&"folo serve stopped".to_string()), "{:?}", titles(this));
    });
}

/// The 2 s refresh, unreachable again.
fn refresh(app: &mut folo_ui::HeadlessApp, window: folo_ui::WindowHandle<super::FoloApp>) {
    drive(app, window, |this, _, cx| {
        this.store.system = Load::Idle;
        this.ensure_core(cx);
    });
}

static FAILED_STARTS: AtomicUsize = AtomicUsize::new(0);

#[test]
fn a_recovery_that_fails_tries_again_later_and_says_so_once() {
    let (mut app, window) = lose_the_server_with(
        Some("http://127.0.0.1:4000"),
        Connector {
            probe: |_| Probe::Nothing,
            start: |_| {
                FAILED_STARTS.fetch_add(1, Ordering::SeqCst);
                Err("folo serve did not start in time".into())
            },
            backend: |_| Arc::new(MockBackend::new(4200, true).failing()),
        },
    );
    assert_eq!(FAILED_STARTS.load(Ordering::SeqCst), 1);
    // The status already reads unreachable when the next refreshes fail (health answered first): they
    // still count. The first failure waits two refreshes, then tries again.
    refresh(&mut app, window);
    refresh(&mut app, window);
    assert_eq!(FAILED_STARTS.load(Ordering::SeqCst), 1, "waits");
    refresh(&mut app, window);
    assert_eq!(FAILED_STARTS.load(Ordering::SeqCst), 2, "tries again");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.local_url.as_deref(), Some("http://127.0.0.1:4000"));
        assert_eq!(this.backend_gen, 0, "nothing to switch to: the backend stays");
        assert!(this.connecting.is_none());
        assert_eq!(this.recover_wait, 4, "and waits longer");
        let said = titles(this).iter().filter(|t| *t == "Couldn't start folo serve").count();
        assert_eq!(said, 1, "{:?}", titles(this));
    });
}

#[test]
fn a_slow_local_server_is_kept_as_it_is() {
    let (mut app, window) = lose_the_server_with(
        Some("http://127.0.0.1:4000"),
        Connector { probe: |_| Probe::Folo, start: found_local, backend: demo },
    );
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.backend_gen, 0, "no backend switch: live views and drafts stay");
        assert_eq!(this.server_url.read(cx).text(), "http://127.0.0.1:4000");
        assert_eq!(this.local_url.as_deref(), Some("http://127.0.0.1:4000"));
        assert!(this.connecting.is_none() && !this.hold_loading);
        assert!(!titles(this).contains(&"folo serve stopped".to_string()));
    });
}

#[test]
fn a_server_serving_the_local_data_dir_is_the_local_one_again() {
    let (mut app, window) = open();
    let connect = |app: &mut folo_ui::HeadlessApp, url: &'static str| {
        drive(app, window, move |this, _, cx| {
            this.connector = Connector { probe: |_| Probe::Folo, start: found_local, backend: demo };
            this.server_url.update(cx, |i, cx| i.set_text(url, cx));
            this.switch_source(ServerMode::Local, cx);
        })
    };
    // Connect on the local server's address again (or back from the demo): its data dir says it is ours.
    drive(&mut app, window, |this, _, _| this.local_dir = Some("D:/data".into()));
    connect(&mut app, "127.0.0.1:4100");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.local_url, None, "not before it says what it serves");
        this.note_data_dir(Some("D:/data"));
        assert_eq!(this.local_url.as_deref(), Some("127.0.0.1:4100"));
    });
    // Another server at an address the local one once had: not ours to start.
    connect(&mut app, "127.0.0.1:4100");
    drive(&mut app, window, |this, _, _| {
        this.note_data_dir(Some("D:/other"));
        assert_eq!(this.local_url, None);
        assert_eq!(this.local_dir.as_deref(), Some("D:/data"));
    });
    // Nor another machine's server that happens to name the same folder.
    connect(&mut app, "http://192.168.9.9:7878");
    drive(&mut app, window, |this, _, _| {
        this.note_data_dir(Some("D:/data"));
        assert_eq!(this.local_url, None);
    });
}

#[test]
fn a_good_refresh_ends_the_backoff() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, _| {
        this.recover_tries = 3;
        this.recover_wait = 7;
    });
    refresh(&mut app, window);
    drive(&mut app, window, |this, _, _| assert_eq!((this.recover_tries, this.recover_wait), (0, 0)));
}

#[test]
fn the_lost_server_answering_the_data_dirs_address_is_kept_as_it_is() {
    // The probe gave up, but starting finds the data dir's server at the very same address.
    let (mut app, window) = lose_the_server_with(
        Some("http://127.0.0.1:4100"),
        Connector { probe: |_| Probe::Nothing, start: found_local, backend: demo },
    );
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.backend_gen, 0, "no backend switch");
        assert!(this.connecting.is_none());
        assert_eq!(this.local_url.as_deref(), Some("http://127.0.0.1:4100"));
    });
}

#[test]
fn a_server_found_at_its_address_is_only_retried() {
    let (mut app, window) = lose_the_server(None);
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.server_url.read(cx).text(), "http://127.0.0.1:4000");
        assert!(this.store.unreachable(), "the pill says it is retrying");
        assert!(this.connecting.is_none());
    });
}

#[test]
fn a_server_that_never_answered_was_not_lost() {
    assert_eq!(super::render::lost_text(false), "Can't reach folo serve, retrying…");
    assert_eq!(super::render::lost_text(true), "Lost connection to folo serve, retrying…");
    // `reached` follows the first answer of each server.
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, _| assert!(this.reached, "the demo answered"));
    drive(&mut app, window, |this, _, cx| {
        this.set_backend(Arc::new(MockBackend::new(4200, true).failing()), ServerMode::Local, cx)
    });
    // What the next frame asks for.
    drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.unreachable());
        assert!(this.lost_banner_shown(), "a server that is down is retried, and the pill says so");
        assert!(!this.reached, "a new server starts as not reached");
    });
}

#[test]
fn a_cpu_only_server_does_not_take_gpu_0_from_the_next_one() {
    // Server A answered `/system` with no GPU: New run disables GPU 0 there.
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, _| {
        if let Load::Ready(s) = &mut this.store.system {
            s.gpus.clear();
        }
        this.store.last_gpu_count = Some(0);
        assert!(this.nr_no_gpu().is_some());
    });
    // Switched to server B: before (and when) its `/system` fails to answer, A's "no GPU" is gone.
    drive(&mut app, window, |this, _, cx| {
        this.set_backend(Arc::new(MockBackend::new(4200, true).failing()), ServerMode::Local, cx);
        assert_eq!(this.store.last_gpu_count, None);
        assert_eq!(this.nr_no_gpu(), None, "B's GPU 0 is offered until B says otherwise");
    });
    drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.system.ready().is_none(), "B's /system failed");
        assert_eq!(this.nr_no_gpu(), None);
    });
    // A same-server refresh keeps the last answer (rev_user round 1): B with a GPU, then asked again.
    drive(&mut app, window, |this, _, cx| {
        this.set_backend(Arc::new(MockBackend::new(4200, true)), ServerMode::Local, cx)
    });
    drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.store.last_gpu_count.map(|n| n > 0), Some(true));
        this.store.system = Load::Loading;
        assert_eq!(this.nr_no_gpu(), None);
    });
}

#[test]
fn the_reason_the_local_server_did_not_start_stays_on_the_error_panels() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        this.connector = Connector {
            probe: |_| Probe::Nothing,
            start: |_| Err("FOLO_BIN names D:\\nope\\folo.exe, which isn't a file".into()),
            backend: |_| Arc::new(MockBackend::new(4200, true).failing()),
        };
        this.autoconnect("http://127.0.0.1:1".into(), true, super::Reason::Launch, cx);
    });
    drive(&mut app, window, |this, _, cx| this.ensure_core(cx));
    drive(&mut app, window, |this, _, _| {
        assert_eq!(
            this.start_error.as_deref(),
            Some("FOLO_BIN names D:\\nope\\folo.exe, which isn't a file")
        );
        assert!(this.store.unreachable());
    });
}

#[test]
fn one_server_and_one_folder_by_any_of_their_names() {
    use super::connection::{same_dir, same_server};
    assert!(same_server("localhost:4100", "http://127.0.0.1:4100/"));
    assert!(same_server("http://[::1]:4100", "127.0.0.1:4100"));
    assert!(!same_server("localhost:4100", "127.0.0.1:4101"));
    assert!(!same_server("http://192.168.9.9:4100", "127.0.0.1:4100"));
    assert!(same_dir("D:/folo/data/", "D:/folo/data"));
    assert_eq!(same_dir("D:/Data", r"d:\data"), cfg!(windows));
    assert!(!same_dir("D:/data", "D:/other"));
}

#[test]
fn another_server_on_the_local_ones_port_is_not_ours() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, _| {
        this.server_mode = ServerMode::Local;
        this.local_url = Some("http://127.0.0.1:4100".into());
        this.note_data_dir(Some("D:/data"));
        this.note_data_dir(Some(if cfg!(windows) { "D:/Data/" } else { "D:/data/" }));
        assert_eq!(this.local_url.as_deref(), Some("http://127.0.0.1:4100"), "the same folder");
        this.note_data_dir(Some("D:/other"));
        assert_eq!(this.local_url, None);
        assert_eq!(this.local_dir.as_deref(), Some("D:/data"), "the local data dir stays known");
    });
}

/// Opened before its server: every list failed. The server answers: every list is read again, the run
/// list too (it kept "Could not reach folo serve" on Runs and Overview until Retry).
#[test]
fn every_list_that_failed_is_read_again_once_the_server_answers() {
    let (mut app, window) = open();
    let down =
        crate::api::ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:18110");
    drive(&mut app, window, |this, _, cx| {
        this.store.health = Load::Failed(down.clone());
        this.store.datasets = Load::Failed(down.clone());
        this.store.models = Load::Failed(down.clone());
        this.store.runs = Load::Failed(down.clone());
        // The 2 s refresh, now answered.
        this.store.system = Load::Idle;
        this.ensure_core(cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(!this.store.unreachable());
        assert!(this.store.datasets.ready().is_some(), "datasets");
        assert!(this.store.models.ready().is_some(), "models");
        assert!(this.store.runs.ready().is_some(), "runs: {:?}", this.store.runs.error());
    });
}

/// A run stream's sender: `drop` it for a stream that ended.
fn stream_of(this: &mut super::FoloApp, id: &str) -> std::sync::mpsc::Sender<crate::api::RunEvent> {
    let (tx, rx) = std::sync::mpsc::channel();
    this.subscriptions.insert(id.into(), crate::api::Subscription::new(rx).0);
    tx
}

fn finished_toasts(this: &super::FoloApp, id: &str) -> usize {
    let title = format!("{} finished", this.store.run(id).expect("the run is listed").name);
    titles(this).iter().filter(|t| **t == title).count()
}

/// A run's stream went silent and timed out after the run had ended (the list already says so): the
/// end it never delivered is handled now (final history, models, the toast), not dropped (the page
/// kept its first chart point and no final numbers for good).
#[test]
fn a_stream_that_ends_silently_after_its_run_finished_still_ends_the_run() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        let id = "run_52e8";
        assert!(!this.store.run(id).unwrap().status.is_active(), "a finished run");
        drop(stream_of(this, id));
        this.drain_streams(cx);
        assert_eq!(finished_toasts(this, id), 1);
        assert!(!this.subscriptions.contains_key(id));
    });
}

/// A run's stream dropped mid-run, opening it again failed (the server away a moment), and the run
/// ended before the next list refresh: nothing subscribes to an ended run, so the list's word that it
/// ended is taken as its end (final history, the toast). Before, the page kept the history of the
/// moment the stream dropped for good.
#[test]
fn a_run_that_ends_while_its_lost_stream_cannot_reopen_still_ends() {
    use crate::api::RunStatus;
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = super::tests::open_with(api.clone());
    let id = "run_52e8";
    let all = drive(&mut app, window, |this, _, cx| {
        this.ensure_run_data(id, cx);
        this.store.api.metrics(id).unwrap().steps.len()
    });
    drive(&mut app, window, |this, _, cx| {
        // Seen training, its history what the stream thinned.
        let r = this.store.runs.ready_mut().unwrap().iter_mut().find(|r| r.id == id).unwrap();
        r.status = RunStatus::Running;
        let Some(Load::Ready(d)) = this.store.run_data.get_mut(id) else { panic!("loaded") };
        d.metrics.steps = d.metrics.steps.iter().step_by(5).cloned().collect();
        api.fail_events_once();
        drop(stream_of(this, id));
        this.drain_streams(cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert!(!this.subscriptions.contains_key(id), "the stream did not open again");
        assert_eq!(finished_toasts(this, id), 0);
        // The 5 s refresh: the run has ended.
        this.refresh_runs(cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert_eq!(finished_toasts(this, id), 1, "{:?}", titles(this));
        let Some(Load::Ready(d)) = this.store.run_data.get(id) else { panic!("still shown") };
        assert_eq!(d.metrics.steps.len(), all, "the final history");
        assert!(this.stream_lost.is_empty());
    });
}

/// A stream gone silent until after its run ended: a loaded log and the latest predictions missed
/// what it never delivered, and are read again (the log kept the lines up to the silence, and the
/// samples the epoch it was at, under a Latest picker offering the final epoch).
#[test]
fn a_silent_end_reads_the_log_and_latest_samples_again() {
    let (mut app, window) = open();
    let id = "run_52e8";
    drive(&mut app, window, |this, _, cx| {
        // What it held of the latest (the epoch the stream went silent at).
        let samples = this.store.api.samples(id, None).unwrap();
        this.store.samples.insert(id.into(), Load::Ready(samples));
        this.store.install_logs(id, Ok(crate::api::RunLogs { lines: Vec::new(), total: 0 }));
        drop(stream_of(this, id));
        this.drain_streams(cx);
        assert!(
            matches!(this.store.samples.get(id), Some(Load::Idle)),
            "the latest samples are asked for again"
        );
    });
    drive(&mut app, window, |this, _, _| {
        let Some(Load::Ready(hist)) = this.store.logs.get(id) else { panic!("the log reloaded") };
        assert!(!hist.lines.is_empty(), "the lines the stream never delivered");
    });
}

/// Pages that failed while the server was away (a dataset, the benchmarks) ask again once it answers:
/// before, they kept "Could not reach folo serve" until Retry, while the lists around them recovered.
#[test]
fn a_page_that_failed_while_the_server_was_away_loads_once_it_is_back() {
    let (mut app, window) = open();
    let down =
        crate::api::ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:18110");
    let ds = drive(&mut app, window, |this, _, cx| {
        let ds = this.store.datasets.ready().unwrap()[0].id.clone();
        this.store.health = Load::Failed(down.clone());
        this.store.dataset.insert(ds.clone(), Load::Failed(down.clone()));
        this.bench.list = Load::Failed(down.clone());
        this.store.system = Load::Idle;
        this.ensure_core(cx);
        ds
    });
    drive(&mut app, window, |this, _, cx| {
        assert!(!this.store.unreachable());
        assert!(this.bench.list.error().is_none(), "the benchmark list is asked for again");
        this.ensure_dataset(&ds, cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.dataset.get(&ds).and_then(|l| l.ready()).is_some(), "the dataset loads");
    });
}

/// A request sent while the server was away that fails only after it answers again (a timeout still
/// running at the reconnect): its failure is cleared too, and the page asks again (it kept "Could not
/// reach folo serve" with the server healthy, nothing asking again).
#[test]
fn a_request_that_fails_after_the_server_is_back_is_asked_again() {
    let (mut app, window) = open();
    let down =
        crate::api::ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:18110");
    let ds = drive(&mut app, window, |this, _, cx| {
        let ds = this.store.datasets.ready().unwrap()[0].id.clone();
        this.store.health = Load::Failed(down.clone());
        this.store.dataset.insert(ds.clone(), Load::Loading);
        let (e, key) = (down.clone(), ds.clone());
        this.fetch(
            cx,
            move |_| Err::<(), _>(e),
            move |this, r, _| {
                if let Err(e) = r {
                    this.store.dataset.insert(key, Load::Failed(e));
                }
            },
        );
        // Back before that answer is in.
        this.store.note_reachability(None);
        this.reconnected(cx);
        ds
    });
    drive(&mut app, window, |this, _, cx| {
        assert!(
            this.store.dataset.get(&ds).is_none_or(|l| l.error().is_none()),
            "the late failure is cleared"
        );
        this.ensure_dataset(&ds, cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(this.store.dataset.get(&ds).and_then(|l| l.ready()).is_some(), "the dataset loads");
    });
}

/// A finished benchmark entry's result that failed while the server was away is asked for again once
/// it answers (its page has no Retry: it said "The result didn't load" for good).
#[test]
fn a_benchmark_result_that_failed_while_the_server_was_away_is_asked_again() {
    let (mut app, window) = open();
    let down =
        crate::api::ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:18110");
    let key = ("bench".to_string(), "entry".to_string());
    drive(&mut app, window, |this, _, cx| {
        this.store.health = Load::Failed(down.clone());
        this.bench.results.insert(key.clone(), Load::Failed(down.clone()));
        this.store.system = Load::Idle;
        this.ensure_core(cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(!this.store.unreachable());
        assert!(!this.bench.results.contains_key(&key), "the result is asked for again when shown");
    });
}

/// A run ended unseen (its stream went silent) and its final history failed to load once: the partial
/// history shown is replaced at the next run-list refresh, and the end is announced once.
#[test]
fn a_final_history_that_failed_is_read_again_at_the_next_refresh() {
    let api = Arc::new(MockBackend::new(4200, true));
    let (mut app, window) = super::tests::open_with(api.clone());
    let id = "run_52e8";
    let all = drive(&mut app, window, |this, _, cx| {
        this.ensure_run_data(id, cx);
        this.store.api.metrics(id).unwrap().steps.len()
    });
    drive(&mut app, window, |this, _, cx| {
        let Some(Load::Ready(d)) = this.store.run_data.get_mut(id) else { panic!("loaded") };
        d.metrics.steps = d.metrics.steps.iter().step_by(5).cloned().collect();
        api.fail_metrics_once();
        drop(stream_of(this, id));
        this.drain_streams(cx);
    });
    drive(&mut app, window, |this, _, cx| {
        // The 5 s refresh.
        this.refresh_runs(cx);
    });
    drive(&mut app, window, |this, _, _| {
        let Some(Load::Ready(d)) = this.store.run_data.get(id) else { panic!("still shown") };
        assert_eq!(d.metrics.steps.len(), all, "the final history");
        assert_eq!(finished_toasts(this, id), 1, "{:?}", titles(this));
        assert!(this.history_retry.is_empty());
    });
}

/// A delivered end, its disconnect seen a tick later: handled once.
#[test]
fn a_delivered_end_is_handled_once() {
    use crate::api::{RunEvent, RunStatus, StatusEvent};
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        let id = "run_52e8";
        let tx = stream_of(this, id);
        tx.send(RunEvent::Status(StatusEvent::new(RunStatus::Completed, None))).unwrap();
        this.drain_streams(cx);
        drop(tx);
        this.drain_streams(cx);
        assert_eq!(finished_toasts(this, id), 1, "{:?}", titles(this));
    });
}

#[test]
fn images_that_failed_while_the_server_was_away_load_once_it_answers() {
    use crate::api::ApiError;
    use crate::store::{ImgKey, ImgKind, ImgSlot};
    let (mut app, window) = open();
    let key = ImgKey { dataset: "ds_3b91".into(), image: "img_03417".into(), kind: ImgKind::Thumb };
    // A thumbnail asked for while the server is away; nothing else saw the outage.
    drive(&mut app, window, |this, _, cx| {
        this.store.api = Arc::new(MockBackend::new(4200, true).failing());
        assert!(this.store.images_px.begin(&key));
        let down = ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7926");
        this.load_cached_image(key.clone(), move |_: &dyn Backend| Err(down), cx);
    });
    drive(&mut app, window, |this, _, _| {
        assert!(matches!(this.store.images_px.get(&key), Some(ImgSlot::Failed)));
    });
    drive(&mut app, window, |this, _, cx| {
        assert!(this.store.unreachable(), "the image's failure has the outage checked, and seen");
        // The 2 s refresh, now answered.
        this.store.api = Arc::new(MockBackend::new(4200, true));
        this.store.system = Load::Idle;
        this.ensure_core(cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert!(!this.store.unreachable());
        assert!(this.store.images_px.get(&key).is_none(), "the failure is forgotten");
        assert!(this.image("ds_3b91", "img_03417", ImgKind::Thumb, cx).is_none(), "asked for again");
    });
    drive(&mut app, window, |this, _, _| {
        assert!(matches!(this.store.images_px.get(&key), Some(ImgSlot::Ready(_))), "the tile shows");
    });
}

#[test]
fn an_image_that_failed_with_the_server_up_is_asked_for_again_a_few_times() {
    use std::time::Duration;

    use crate::api::ApiError;
    use crate::store::{ImgKey, ImgKind, ImgSlot};
    let (mut app, window) = open();
    let key = ImgKey { dataset: "ds_3b91".into(), image: "img_03417".into(), kind: ImgKind::Full };
    // A body cut off after its headers, then headers that time out, while `/system` answers: no
    // outage, so no reconnect clears them (nor resets their retries).
    let fail = |app: &mut folo_ui::HeadlessApp, code: &'static str| {
        drive(app, window, |this, _, cx| {
            assert!(this.store.images_px.begin(&key));
            let cut = ApiError::new(code, "folo serve's answer was cut off");
            this.load_cached_image(key.clone(), move |_: &dyn Backend| Err(cut), cx);
        })
    };
    let failed = |app: &mut folo_ui::HeadlessApp| {
        drive(app, window, |this, _, _| matches!(this.store.images_px.get(&key), Some(ImgSlot::Failed)))
    };
    for (wait, code) in [(2, "bad_response"), (8, "server_unreachable"), (32, "server_unreachable")] {
        fail(&mut app, code);
        assert!(failed(&mut app));
        // The `/system` probe a timed-out image asks for is answered.
        assert!(failed(&mut app));
        assert!(!drive(&mut app, window, |this, _, _| this.store.unreachable()), "no outage");
        app.advance_clock(Duration::from_secs(wait) - Duration::from_millis(1));
        assert!(failed(&mut app), "{wait} s not up yet");
        app.advance_clock(Duration::from_millis(1));
        assert!(
            drive(&mut app, window, |this, _, _| this.store.images_px.get(&key).is_none()),
            "after {wait} s"
        );
    }
    // A fourth failure in a row (a corrupt file) waits for a Retry instead of polling for ever.
    fail(&mut app, "server_unreachable");
    app.advance_clock(Duration::from_secs(3600));
    assert!(failed(&mut app));
    assert!(
        drive(&mut app, window, |this, _, _| this.image_failed("ds_3b91", "img_03417", ImgKind::Full)),
        "it says so"
    );
    // The inspector's Retry: asked for again by the next draw, it shows.
    drive(&mut app, window, |this, _, cx| {
        this.retry_failed_image("ds_3b91", "img_03417", cx);
        assert!(this.image("ds_3b91", "img_03417", ImgKind::Full, cx).is_none());
    });
    drive(&mut app, window, |this, _, _| {
        assert!(matches!(this.store.images_px.get(&key), Some(ImgSlot::Ready(_))));
        assert!(!this.image_failed("ds_3b91", "img_03417", ImgKind::Full));
    });
}

#[test]
fn a_thumbnail_past_its_retries_loads_again_once_the_inspector_loads_its_full_size() {
    use std::time::Duration;

    use crate::api::ApiError;
    use crate::store::{ImgKey, ImgKind, ImgSlot};
    let (mut app, window) = open();
    let ds = "ds_3b91";
    let key = |im: &str, kind| ImgKey { dataset: ds.into(), image: im.into(), kind };
    let slot = |app: &mut folo_ui::HeadlessApp, im: &str, kind| {
        drive(app, window, |this, _, _| match this.store.images_px.get(&key(im, kind)) {
            None => "none",
            Some(ImgSlot::Loading) => "loading",
            Some(ImgSlot::Ready(_)) => "ready",
            Some(ImgSlot::Failed) => "failed",
        })
    };
    let show = |app: &mut folo_ui::HeadlessApp, im: &str, kind| {
        drive(app, window, |this, _, cx| assert!(this.image(ds, im, kind, cx).is_none()));
        assert_eq!(slot(app, im, kind), "ready", "{im} {kind:?}");
    };
    // The grid's thumbnail is cut off four times in a row (`/system` answering): no more retries.
    let exhaust = |app: &mut folo_ui::HeadlessApp, im: &str| {
        for wait in [2, 8, 32, 0] {
            drive(app, window, |this, _, cx| {
                assert!(this.store.images_px.begin(&key(im, ImgKind::Thumb)));
                let cut = ApiError::new("bad_response", "folo serve's answer was cut off");
                this.load_cached_image(key(im, ImgKind::Thumb), move |_: &dyn Backend| Err(cut), cx);
            });
            assert_eq!(slot(app, im, ImgKind::Thumb), "failed");
            app.advance_clock(Duration::from_secs(wait));
        }
        app.advance_clock(Duration::from_secs(3600));
        assert_eq!(slot(app, im, ImgKind::Thumb), "failed", "it waits");
    };
    let im = "img_03417";
    exhaust(&mut app, im);
    // d1-datasets, x4-soak: its grid tile says "Image unavailable" instead of looking as if it
    // still loads (the inspector it opens has the Retry).
    assert!(drive(&mut app, window, |this, _, _| this.thumb_unavailable(ds, im)));
    // The images answer again; the inspector opens on it and its full size loads.
    show(&mut app, im, ImgKind::Full);
    // So the thumbnail is asked for again by the grid's next draw, and shows.
    assert_eq!(slot(&mut app, im, ImgKind::Thumb), "none");
    assert!(!drive(&mut app, window, |this, _, _| this.thumb_unavailable(ds, im)), "loading again");
    show(&mut app, im, ImgKind::Thumb);
    // A thumbnail that fails after its full size loaded: the inspector offers Retry for it.
    let im = "img_00001";
    show(&mut app, im, ImgKind::Full);
    exhaust(&mut app, im);
    assert_eq!(slot(&mut app, im, ImgKind::Full), "ready");
    drive(&mut app, window, |this, _, cx| {
        assert!(this.image_failed(ds, im, ImgKind::Thumb) && !this.image_failed(ds, im, ImgKind::Full));
        this.retry_failed_image(ds, im, cx);
    });
    show(&mut app, im, ImgKind::Thumb);
}

#[test]
fn a_retry_timer_left_from_before_a_reconnect_leaves_the_next_failure_alone() {
    use std::time::Duration;

    use crate::api::ApiError;
    use crate::store::{ImgKey, ImgKind, ImgSlot};
    let (mut app, window) = open();
    let key = ImgKey { dataset: "ds_3b91".into(), image: "img_03417".into(), kind: ImgKind::Thumb };
    let fail = |app: &mut folo_ui::HeadlessApp| {
        drive(app, window, |this, _, cx| {
            assert!(this.store.images_px.begin(&key));
            let cut = ApiError::new("bad_response", "folo serve's answer was cut off");
            this.load_cached_image(key.clone(), move |_: &dyn Backend| Err(cut), cx);
        })
    };
    let failed = |app: &mut folo_ui::HeadlessApp| {
        drive(app, window, |this, _, _| matches!(this.store.images_px.get(&key), Some(ImgSlot::Failed)))
    };
    // A failure at 0 s (its retry due at 2 s); a reconnect at 1 s asks for it again, and it fails
    // again (its retry due at 3 s).
    fail(&mut app);
    assert!(failed(&mut app));
    app.advance_clock(Duration::from_secs(1));
    drive(&mut app, window, |this, _, cx| this.reconnected(cx));
    assert!(!failed(&mut app));
    fail(&mut app);
    assert!(failed(&mut app));
    app.advance_clock(Duration::from_secs(1));
    assert!(failed(&mut app), "the timer from before the reconnect does nothing");
    app.advance_clock(Duration::from_secs(1));
    assert!(!failed(&mut app), "its own timer asks again");
}

#[test]
fn a_huge_full_size_image_is_held_at_the_cap_and_its_thumbnail_as_sent() {
    use crate::api::ImageData;
    use crate::store::{ImageCache, ImgKey, ImgKind, ImgSlot};
    let (mut app, window) = open();
    let key = |kind| ImgKey { dataset: "ds_3b91".into(), image: "img_03417".into(), kind };
    let wide = || ImageData::Rgba { width: 5000, height: 10, pixels: vec![9; 5000 * 10 * 4] };
    drive(&mut app, window, |this, _, cx| {
        for kind in [ImgKind::Full, ImgKind::Thumb] {
            assert!(this.store.images_px.begin(&key(kind)));
            this.load_cached_image(key(kind), move |_: &dyn Backend| Ok(wide()), cx);
        }
    });
    let side = |app: &mut folo_ui::HeadlessApp, kind| {
        drive(app, window, |this, _, _| match this.store.images_px.get(&key(kind)) {
            Some(ImgSlot::Ready(img)) => {
                let s = img.size(0);
                (s.width.0 as u32, s.height.0 as u32)
            }
            _ => panic!("{kind:?} not loaded"),
        })
    };
    assert_eq!(side(&mut app, ImgKind::Full), (ImageCache::FULL_MAX_SIDE, 8));
    assert_eq!(side(&mut app, ImgKind::Thumb), (5000, 10), "a thumbnail is the server's size");
}

/// A thumbnail whose body is cut off midway over real HTTP reads as `server_unreachable`
/// (`HttpBackend::body_failed`); with `/system` answering that only probes, so it still backs off
/// 2, 8 and 32 s, waits after the fourth failure, and the inspector's Retry loads it.
#[test]
fn a_thumbnail_cut_off_midway_backs_off_then_waits_for_a_retry() {
    use std::io::{Read, Write};
    use std::sync::Mutex;
    use std::time::Duration;

    use crate::api::http::HttpBackend;
    use crate::store::{ImgKey, ImgKind, ImgSlot};
    // Announces 1000 bytes, sends 10, hangs up: four times.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for _ in 0..4 {
            let Ok((mut sock, _)) = listener.accept() else { return };
            let mut req = Vec::new();
            let mut buf = [0u8; 1024];
            while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                match sock.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => req.extend_from_slice(&buf[..n]),
                }
            }
            let head = "HTTP/1.1 200 OK\r\ncontent-type: image/jpeg\r\ncontent-length: 1000\r\nconnection: close\r\n\r\n";
            let _ = sock.write_all(head.as_bytes());
            let _ = sock.write_all(b"\xff\xd8\xff\xe0abcdef");
            let _ = sock.flush();
        }
    });
    let http = Arc::new(HttpBackend::new(&url));
    let codes = Arc::new(Mutex::new(Vec::<String>::new()));
    let (mut app, window) = open();
    let (ds, im) = ("ds_3b91", "img_03417");
    let key = ImgKey { dataset: ds.into(), image: im.into(), kind: ImgKind::Thumb };
    let fetch = |app: &mut folo_ui::HeadlessApp| {
        let (http, codes) = (http.clone(), codes.clone());
        drive(app, window, |this, _, cx| {
            assert!(this.store.images_px.begin(&key));
            let call = move |_: &dyn Backend| {
                let r = http.image(ds, im, Some(128));
                if let Err(e) = &r {
                    codes.lock().unwrap().push(e.code.clone());
                }
                r
            };
            this.load_cached_image(key.clone(), call, cx);
        })
    };
    let failed = |app: &mut folo_ui::HeadlessApp| {
        drive(app, window, |this, _, _| matches!(this.store.images_px.get(&key), Some(ImgSlot::Failed)))
    };
    for wait in [2, 8, 32] {
        fetch(&mut app);
        assert!(failed(&mut app));
        // The `/system` probe the outage-looking failure asked for is answered: no outage.
        assert!(failed(&mut app));
        assert!(!drive(&mut app, window, |this, _, _| this.store.unreachable()), "no outage");
        app.advance_clock(Duration::from_secs(wait) - Duration::from_millis(1));
        assert!(failed(&mut app), "{wait} s not up yet");
        app.advance_clock(Duration::from_millis(1));
        assert!(
            drive(&mut app, window, |this, _, _| this.store.images_px.get(&key).is_none()),
            "asked for again after {wait} s"
        );
    }
    fetch(&mut app);
    app.advance_clock(Duration::from_secs(3600));
    assert!(failed(&mut app), "the fourth failure waits");
    assert!(drive(&mut app, window, |this, _, _| this.image_failed(ds, im, ImgKind::Thumb)));
    assert_eq!(*codes.lock().unwrap(), ["server_unreachable"; 4], "a cut body is the server out of reach");
    // The inspector's Retry asks the app's own server (answering now): the thumbnail shows.
    drive(&mut app, window, |this, _, cx| {
        this.retry_failed_image(ds, im, cx);
        assert!(this.image(ds, im, ImgKind::Thumb, cx).is_none());
    });
    drive(&mut app, window, |this, _, _| {
        assert!(matches!(this.store.images_px.get(&key), Some(ImgSlot::Ready(_))));
        assert!(!this.image_failed(ds, im, ImgKind::Thumb));
    });
}
