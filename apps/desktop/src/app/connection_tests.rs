//! Connecting from Settings and losing the local server, with a fake connector: no socket is opened
//! and no `folo serve` is started.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::tests::{drive, open};
use super::{Connector, ServerMode};
use crate::api::Backend;
use crate::api::http::Spawned;
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
        this.connector = Connector { probe: |_| false, start: found_local, backend: demo };
        this.server_url.update(cx, |i, cx| i.set_text("http://127.0.0.1:1", cx));
        this.switch_source(ServerMode::Local, cx);
    });
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.server_url.read(cx).text(), "http://127.0.0.1:4100");
        assert_eq!(this.local_url.as_deref(), Some("http://127.0.0.1:4100"));
        assert!(titles(this).contains(&"Nothing answered at 127.0.0.1:1".to_string()), "{:?}", titles(this));
    });
}

static REMOTE_STARTS: AtomicUsize = AtomicUsize::new(0);

#[test]
fn another_machines_address_is_kept_and_no_local_server_takes_its_place() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        this.connector = Connector {
            probe: |_| false,
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

/// The local server stops answering (the window that started it closed, or it crashed): the 2 s
/// refresh sees it, and the app finds or starts one again instead of retrying a dead port.
fn lose_the_server(local: Option<&str>) -> (folo_ui::HeadlessApp, folo_ui::WindowHandle<super::FoloApp>) {
    let (mut app, window) = open();
    let local = local.map(str::to_string);
    drive(&mut app, window, move |this, _, cx| {
        this.connector = Connector { probe: |_| false, start: found_local, backend: demo };
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
        assert!(!this.reached, "a new server starts as not reached");
    });
}

#[test]
fn the_reason_the_local_server_did_not_start_stays_on_the_error_panels() {
    let (mut app, window) = open();
    drive(&mut app, window, |this, _, cx| {
        this.connector = Connector {
            probe: |_| false,
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
