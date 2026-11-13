//! The data source: switching between the built-in demo and folo serve, and connecting to (or
//! starting) a server.

use std::sync::Arc;

use folo_ui::Context;

use crate::api::*;
use crate::store::*;
use crate::ui::*;

use super::{FoloApp, ServerMode};

/// The most refreshes (2 s apart) a failed recovery waits before the next attempt.
const RECOVER_WAIT_MAX: u32 = 30;

/// Who asked for a connection ([`FoloApp::autoconnect`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reason {
    /// The app starting up (`$FOLO_SERVER` or the default address).
    Launch,
    /// Connect or Enter in Settings, or picking `folo serve` there.
    Typed,
    /// The local server stopped answering ([`FoloApp::connection_lost`]).
    Recover,
}

/// How the app reaches servers: asking an address whether a `folo serve` answers, starting a local one,
/// and the backend for an address. Real HTTP and processes; the tests put fakes in.
#[derive(Clone, Copy)]
pub struct Connector {
    pub probe: fn(&str) -> bool,
    pub start: fn(std::time::Duration) -> Result<crate::api::http::Spawned, String>,
    pub backend: fn(&str) -> Arc<dyn Backend>,
}

impl Default for Connector {
    fn default() -> Self {
        use crate::api::http::{HttpBackend, probe, spawn_server};
        Self { probe, start: spawn_server, backend: |url| Arc::new(HttpBackend::new(url)) }
    }
}

impl FoloApp {
    /// Swap the data source; everything reloads from the new backend.
    pub fn set_backend(&mut self, api: Arc<dyn Backend>, mode: ServerMode, cx: &mut Context<Self>) {
        self.backend_gen += 1;
        // Estimates and speeds are the old server's (and a pending answer will be dropped).
        self.throughput = Load::Idle;
        self.bench_busy = false;
        self.bench_note = None;
        self.nr_estimate = None;
        self.nr_estimate_last = None;
        self.nr_estimate_gen += 1;
        // A draft's dataset belongs to the old server, and so do the ids deleted there.
        self.new_run_draft = None;
        self.deleted.clear();
        // Model ids are per server: loaded local models belong to the old one.
        crate::local::forget_all();
        if !api.local_inference() {
            self.play_target = None;
        }
        self.store.api = api;
        self.server_mode = mode;
        self.clear_bench();
        self.subscriptions.clear();
        self.subscribing.clear();
        self.store.clear_pending();
        *self.ds_filter_cache.borrow_mut() = None;
        *self.ds_split_counts.borrow_mut() = None;
        // Ids are only unique per server: cached pixels (and failures) belong to the old one. The
        // textures are freed after this update, when no window is mid-draw, so every window's atlas
        // is reached (this often runs inside a click handler).
        let mut textures = self.store.images_px.clear();
        textures.extend(self.play_image.take().map(|p| p.image));
        cx.defer(move |cx| {
            for img in textures {
                cx.drop_image(img, None);
            }
        });
        self.connecting = None;
        self.hold_loading = false;
        // Set again by the caller when this is the local server; the reason one didn't start is old.
        self.local_url = None;
        self.start_error = None;
        self.reached = false;
        self.store.gpu_util.clear();
        self.play_pred = Load::Idle;
        self.play_pred_pending = false;
        self.play_model = None;
        self.play_model_run = None;
        self.play_model_gone = None;
        self.retry_all(cx);
    }

    /// The server answers again after it was unreachable: an open dialog's "could not reach"
    /// error is out of date (its next try may well work), and the estimate is asked for again.
    pub fn reconnected(&mut self, cx: &mut Context<Self>) {
        let stale = |e: &Option<ApiError>| e.as_ref().is_some_and(|e| e.code == "server_unreachable");
        if let Some(f) = self.new_run.as_mut()
            && stale(&f.error)
        {
            f.error = None;
        }
        if matches!(&self.nr_estimate, Some((_, Load::Failed(e))) if e.code == "server_unreachable") {
            // The dialog asks again on its next frame.
            self.nr_estimate = None;
        }
        if let Some(f) = self.synth.as_mut()
            && stale(&f.error)
        {
            f.error = None;
        }
        for e in [
            self.bench.new_bench.as_mut().map(|f| &mut f.error),
            self.bench.new_loop.as_mut().map(|f| &mut f.error),
            self.bench.design.as_mut().map(|f| &mut f.error),
        ]
        .into_iter()
        .flatten()
        {
            if stale(e) {
                *e = None;
            }
        }
        // The Playground's prediction that couldn't reach it runs again, as Retry would.
        if self.play_pred.error().is_some_and(|e| e.code == "server_unreachable") {
            self.rerun_predict(cx);
        }
        // Back: read the lists now (the server may have restarted on another data dir).
        self.refresh_lists(cx);
        cx.notify();
    }

    /// Connect to a running `folo serve`, or start a local `folo serve` (the real engine, bare
    /// until datasets and runs are added) when `spawn` is on; when the data dir already has a server,
    /// connect to that one instead. If neither is reachable, stay on the HTTP backend so the
    /// unreachable state is shown instead of silent demo data, and keep the reason the local server
    /// didn't start for the error panels. A server this app started runs until the app quits, whatever
    /// it connects to later: it may be training. `why` says who asked: a connection that lands
    /// somewhere else than the address typed in Settings, or one that brings a lost local server back,
    /// says so.
    pub fn autoconnect(&mut self, url: String, spawn: bool, why: Reason, cx: &mut Context<Self>) {
        use crate::api::http::Spawned;
        let connector = self.connector;
        // A recovery keeps what is on screen (the server may only be slow); a launch or a switch shows
        // the pages loading.
        self.hold_loading = why != Reason::Recover;
        if why != Reason::Recover {
            self.recover_tries = 0;
            self.recover_wait = 0;
        }
        self.connecting = Some(format!("Connecting to {}", url.trim_start_matches("http://")).into());
        // Only the newest attempt may install its result; older ones bail out at each step, and a
        // server an obsolete attempt started is dropped (stopped) with it.
        self.connect_seq += 1;
        let seq = self.connect_seq;
        cx.spawn(async move |this, cx| {
            let current = |this: &Self| this.connect_seq == seq;
            let probe = url.clone();
            let found = cx.background_executor().spawn(async move { (connector.probe)(&probe) }).await;
            if found {
                this.update(cx, |this, cx| {
                    if !current(this) {
                        return;
                    }
                    if why == Reason::Recover {
                        // Still there, only slow (a loaded PC): same server, so nothing is reset (live
                        // views, drafts); the next refresh that answers reconnects as usual.
                        this.connecting = None;
                        this.recovered();
                        cx.notify();
                        return;
                    }
                    this.set_backend((connector.backend)(&url), ServerMode::Local, cx);
                    // The local server again (Connect pressed on its address, or back from the demo): it
                    // stays the one to bring back. Any other server found at an address isn't ours to start.
                    let base = |u: &str| crate::api::http::HttpBackend::new(u).base().to_string();
                    if this.last_local.as_deref().is_some_and(|l| base(l) == base(&url)) {
                        this.local_url = Some(url.clone());
                    }
                })
                .ok();
                return;
            }
            let mut spawn_err = None;
            if spawn {
                let go_on = this
                    .update(cx, |this, cx| {
                        if current(this) {
                            this.connecting = Some("Starting folo serve…".into());
                            cx.notify();
                        }
                        current(this)
                    })
                    .unwrap_or(false);
                if !go_on {
                    return;
                }
                let started = cx
                    .background_executor()
                    .spawn(async move { (connector.start)(std::time::Duration::from_secs(15)) })
                    .await;
                match started {
                    Ok(spawned) => {
                        this.update(cx, |this, cx| {
                            if !current(this) {
                                return; // A started server drops here: the stray server stops.
                            }
                            let (local, started) = match spawned {
                                Spawned::Started(proc_) => {
                                    let local = proc_.url.clone();
                                    this.server_proc = Some(proc_);
                                    (local, true)
                                }
                                // The data dir's server (another window's, or one started by hand): ours stays.
                                Spawned::Running(local) => (local, false),
                            };
                            this.server_url.update(cx, |i, cx| i.set_text(local.clone(), cx));
                            this.set_backend((connector.backend)(&local), ServerMode::Local, cx);
                            this.local_url = Some(local.clone());
                            this.last_local = Some(local.clone());
                            this.recovered();
                            this.said_where(why, &url, &local, started, cx);
                        })
                        .ok();
                        return;
                    }
                    Err(e) => spawn_err = Some(e),
                }
            }
            this.update(cx, |this, cx| {
                if !current(this) {
                    return;
                }
                this.set_backend((connector.backend)(&url), ServerMode::Local, cx);
                let mut say = true;
                if why == Reason::Recover {
                    // Not back yet: try again later, waiting longer each time (`connection_lost`), and
                    // say so once.
                    this.local_url = Some(url.clone());
                    this.recover_tries += 1;
                    this.recover_wait = (1u32 << this.recover_tries.min(5)).min(RECOVER_WAIT_MAX);
                    say = this.recover_tries == 1;
                }
                // Say why there is no local server; the pages show the unreachable state, and their
                // error panels keep the reason after the toast has gone.
                if let Some(e) = &spawn_err {
                    this.start_error = Some(e.clone().into());
                }
                if let Some(e) = spawn_err.filter(|_| say) {
                    this.toast(
                        Tone::Danger,
                        "Couldn't start folo serve",
                        Some(format!("{e}. Start one yourself, or pick Built-in demo in Settings.").into()),
                        cx,
                    );
                }
            })
            .ok();
        })
        .detach();
        cx.notify();
    }

    /// After a connection landed on a local server instead of the address asked for: say so when a
    /// person typed that address, or when the local server had stopped.
    fn said_where(&mut self, why: Reason, asked: &str, local: &str, started: bool, cx: &mut Context<Self>) {
        let host = |u: &str| u.trim().trim_start_matches("http://").trim_end_matches('/').to_string();
        let (title, body) = match why {
            Reason::Launch => return,
            Reason::Typed if host(asked) == host(local) => return,
            Reason::Typed => (
                format!("Nothing answered at {}", host(asked)),
                format!(
                    "Connected to the folo serve {} at {} instead.",
                    if started { "FOLO started" } else { "already serving this data" },
                    host(local)
                ),
            ),
            Reason::Recover => (
                "folo serve stopped".to_string(),
                if started {
                    format!("FOLO started it again at {}.", host(local))
                } else {
                    format!("Connected to the folo serve now serving this data, at {}.", host(local))
                },
            ),
        };
        self.toast(Tone::Info, title, Some(body.into()), cx);
    }

    /// The periodic refresh found the server unreachable (each time it does). When it is the local one
    /// (started by this app, or another window's found through the data dir) and starting one is on,
    /// find or start it again, as at launch: else this window would retry a port nobody will ever
    /// listen on again. One attempt at a time; after a failed one, the next waits for more refreshes.
    pub fn connection_lost(&mut self, cx: &mut Context<Self>) {
        if self.server_mode != ServerMode::Local || !self.spawn_server || self.connecting.is_some() {
            return;
        }
        let Some(url) = self.local_url.clone() else {
            return;
        };
        if self.recover_wait > 0 {
            self.recover_wait -= 1;
            return;
        }
        self.autoconnect(url, true, Reason::Recover, cx);
    }

    /// The local server answers again: the next loss starts over without waiting.
    fn recovered(&mut self) {
        self.recover_tries = 0;
        self.recover_wait = 0;
    }

    /// Settings › Data source.
    pub fn switch_source(&mut self, mode: ServerMode, cx: &mut Context<Self>) {
        match mode {
            ServerMode::Mock => {
                // Cancels any connection attempt still in flight. A server this app started keeps
                // running (it may be training) until the app quits.
                self.connect_seq += 1;
                self.set_backend(
                    Arc::new(crate::api::mock::MockBackend::new(4180, false)),
                    ServerMode::Mock,
                    cx,
                );
            }
            ServerMode::Local => {
                let url = self.server_url.read(cx).text().to_string();
                // Only this machine's server can be started here: a typo in another machine's address
                // must say so, not land on the local data.
                let spawn = self.spawn_server && crate::api::http::is_loopback(&url);
                self.autoconnect(url, spawn, Reason::Typed, cx);
            }
        }
    }
}
