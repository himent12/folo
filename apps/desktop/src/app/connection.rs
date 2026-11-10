//! The data source: switching between the built-in demo and folo serve, and connecting to (or
//! starting) a server.

use std::sync::Arc;

use gpui::Context;

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
        // A draft's dataset belongs to the old server.
        self.new_run_draft = None;
        self.new_run_draft_ds = None;
        // Model ids are per server: loaded local models belong to the old one.
        crate::local::forget_all();
        if !api.local_inference() {
            self.play_target = None;
        }
        self.store.api = api;
        self.server_mode = mode;
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

    /// Find a data source: a running `folo serve`, else one we start (`folo serve --mock`),
    /// else the built-in demo data. Runs off the UI thread; the app shows "Connecting…" meanwhile.
    pub fn autoconnect(&mut self, url: String, spawn: bool, cx: &mut Context<Self>) {
        use crate::api::http::{HttpBackend, spawn_mock_server};
        self.hold_loading = true;
        self.connecting = Some(format!("Connecting to {}", url.trim_start_matches("http://")).into());
        // Only the newest attempt may install its result; older ones bail out at each step, and a
        // server an obsolete attempt started is dropped (stopped) with it.
        self.connect_seq += 1;
        let seq = self.connect_seq;
        let live_step = 4180;
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
            if spawn {
                let go_on = this
                    .update(cx, |this, cx| {
                        if current(this) {
                            this.connecting = Some("Starting folo serve --mock…".into());
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
                    .spawn(async move { spawn_mock_server(std::time::Duration::from_secs(10)) })
                    .await;
                if let Ok(proc_) = started {
                    this.update(cx, |this, cx| {
                        if !current(this) {
                            return; // proc_ drops here: the stray server stops.
                        }
                        let backend = HttpBackend::new(&proc_.url);
                        let url = proc_.url.clone();
                        this.server_proc = Some(proc_);
                        this.server_url.update(cx, |i, cx| i.set_text(url, cx));
                        this.set_backend(Arc::new(backend), ServerMode::Local, cx);
                    })
                    .ok();
                    return;
                }
            }
            this.update(cx, |this, cx| {
                if !current(this) {
                    return;
                }
                this.set_backend(
                    Arc::new(crate::api::mock::MockBackend::new(live_step, false)),
                    ServerMode::Mock,
                    cx,
                );
                this.toast(
                    Tone::Info,
                    "Using built-in demo data",
                    Some("No folo serve was found. Start one and pick it in Settings › Connection.".into()),
                    cx,
                );
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
                // Cancels any connection attempt still in flight.
                self.connect_seq += 1;
                self.server_proc = None;
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
