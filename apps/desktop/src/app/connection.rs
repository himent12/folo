//! The data source: switching between the built-in demo and folo serve, and connecting to (or
//! starting) a server.

use std::sync::Arc;

use folo_ui::Context;

use crate::api::*;
use crate::store::*;
use crate::ui::*;

use super::{FoloApp, ServerMode};

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
        self.store.gpu_util.clear();
        self.play_pred = Load::Idle;
        self.play_pred_pending = false;
        self.play_model = None;
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
        cx.notify();
    }

    /// Connect to a running `folo serve`, or start a local `folo serve` (the real engine, bare
    /// until datasets and runs are added) when `spawn` is on; when the data dir already has a server,
    /// connect to that one instead. If neither is reachable, stay on the HTTP backend so the
    /// unreachable state is shown instead of silent demo data. A server this app started runs until
    /// the app quits, whatever it connects to later: it may be training.
    pub fn autoconnect(&mut self, url: String, spawn: bool, cx: &mut Context<Self>) {
        use crate::api::http::{HttpBackend, Spawned, spawn_server};
        self.hold_loading = true;
        self.connecting = Some(format!("Connecting to {}", url.trim_start_matches("http://")).into());
        // Only the newest attempt may install its result; older ones bail out at each step, and a
        // server an obsolete attempt started is dropped (stopped) with it.
        self.connect_seq += 1;
        let seq = self.connect_seq;
        cx.spawn(async move |this, cx| {
            let current = |this: &Self| this.connect_seq == seq;
            let probe = url.clone();
            let found = cx
                .background_executor()
                .spawn(async move { HttpBackend::new(&probe).health().is_ok() })
                .await;
            if found {
                this.update(cx, |this, cx| {
                    if current(this) {
                        this.set_backend(Arc::new(HttpBackend::new(&url)), ServerMode::Local, cx);
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
                    .spawn(async move { spawn_server(std::time::Duration::from_secs(15)) })
                    .await;
                match started {
                    Ok(spawned) => {
                        this.update(cx, |this, cx| {
                            if !current(this) {
                                return; // A started server drops here: the stray server stops.
                            }
                            let url = match spawned {
                                Spawned::Started(proc_) => {
                                    let url = proc_.url.clone();
                                    this.server_proc = Some(proc_);
                                    url
                                }
                                // The data dir's server (another window's, or one started by hand): ours stays.
                                Spawned::Running(url) => url,
                            };
                            this.server_url.update(cx, |i, cx| i.set_text(url.clone(), cx));
                            this.set_backend(Arc::new(HttpBackend::new(&url)), ServerMode::Local, cx);
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
                this.set_backend(Arc::new(HttpBackend::new(&url)), ServerMode::Local, cx);
                // Say why there is no local server; the pages show the unreachable state.
                if let Some(e) = spawn_err {
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
                let spawn = self.spawn_server;
                self.autoconnect(url, spawn, cx);
            }
        }
    }
}
