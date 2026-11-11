//! What commands do on the server: generate or add a dataset, resume, reorder, stop and export.

use std::rc::Rc;

use folo_ui::{AppContext, Context, Focusable, Window};

use crate::api::*;
use crate::input::TextInput;
use crate::ui::*;

use super::forms::{SYNTH_CLASSES, SYNTH_IMGSZ, SynthForm, parse_images};
use super::{FoloApp, Route, ServerMode, act};

impl FoloApp {
    // ---------- actions ----------

    /// Generate synthetic: a small dialog against folo serve; the built-in demo explains itself.
    pub fn open_synth(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.server_mode == ServerMode::Mock {
            self.toast(
                Tone::Info,
                "Synthetic datasets need folo serve",
                Some("Run `folo dataset synth`, or connect to folo serve in Settings.".into()),
                cx,
            );
            return;
        }
        self.remember_focus(window, cx);
        let taken: Vec<String> = self
            .store
            .datasets
            .ready()
            .map(|d| d.iter().map(|d| d.name.clone()).collect())
            .unwrap_or_default();
        let default_name = (1..)
            .map(|n| if n == 1 { "shapes".to_string() } else { format!("shapes-{n}") })
            .find(|n| !taken.contains(n));
        let name = cx.new(|cx| {
            let mut i = TextInput::new("shapes", cx);
            i.set_text(default_name.unwrap_or_default(), cx);
            i
        });
        let images = cx.new(|cx| {
            let mut i = TextInput::new("1000", cx);
            i.set_text("1000", cx);
            i
        });
        window.focus(&name.read(cx).focus_handle(cx));
        self.synth_seq += 1;
        self.synth = Some(SynthForm {
            id: self.synth_seq,
            name,
            images,
            imgsz: 1,
            classes: 0,
            submitting: false,
            error: None,
        });
        cx.notify();
    }

    pub fn close_synth(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.synth = None;
        self.restore_focus(window);
        cx.notify();
    }

    pub fn submit_synth(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        let Some(f) = self.synth.as_mut() else { return };
        if f.submitting {
            return;
        }
        let name = f.name.read(cx).text().trim().to_string();
        let images = parse_images(&f.images.read(cx).text());
        let (Some(images), false) = (images, name.is_empty()) else {
            f.error = Some(if name.is_empty() {
                ApiError::new("invalid_name", "Give the dataset a name.")
            } else {
                ApiError::new("invalid_body", "Images: a whole number from 1 to 100,000.")
            });
            cx.notify();
            return;
        };
        f.submitting = true;
        f.error = None;
        let form = f.id;
        let req = SyntheticDataset {
            name,
            images,
            classes: SYNTH_CLASSES[f.classes].iter().map(|s| s.to_string()).collect(),
            imgsz: SYNTH_IMGSZ[f.imgsz],
            seed: 0,
        };
        cx.notify();
        self.fetch(
            cx,
            move |api| api.synth_dataset(&req),
            move |this, r, cx| {
                let mine = this.synth.as_ref().is_some_and(|f| f.id == form);
                match r {
                    Ok(ds) => {
                        if mine {
                            this.synth = None;
                            // Back to whatever opened the dialog, on the next frame (needs a window).
                            this.restore_pending = true;
                        }
                        this.dataset_added(ds, "Generated", cx);
                    }
                    Err(e) if mine => {
                        if let Some(f) = this.synth.as_mut() {
                            f.submitting = false;
                            f.error = Some(e);
                        }
                    }
                    // Its dialog is gone (dismissed, or another one open): say so instead.
                    Err(e) => this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx),
                }
            },
        );
    }

    /// Add dataset: pick a folder and register it with folo serve.
    pub fn add_dataset_folder(&mut self, cx: &mut Context<Self>) {
        if self.server_mode == ServerMode::Mock {
            self.toast(
                Tone::Info,
                "Registering folders needs folo serve",
                Some("The built-in demo has a fixed set of datasets.".into()),
                cx,
            );
            return;
        }
        let rx = cx.prompt_for_paths(folo_ui::PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Add dataset".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(p) = paths.into_iter().next()
            {
                this.update(cx, |this, cx| {
                    let path = p.to_string_lossy().to_string();
                    let folder = p.file_name().map_or(path.clone(), |n| n.to_string_lossy().to_string());
                    this.toast(
                        Tone::Info,
                        format!("Indexing {folder}"),
                        Some("Reading every image header…".into()),
                        cx,
                    );
                    let req = RegisterDataset { name: None, path, format: Default::default() };
                    this.fetch(
                        cx,
                        move |api| api.register_dataset(&req),
                        |this, r, cx| match r {
                            Ok(ds) => this.dataset_added(ds, "Added", cx),
                            Err(e) => this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx),
                        },
                    );
                })
                .ok();
            }
        })
        .detach();
    }

    /// A new dataset from the server: list it first and say so, with a way to open it.
    fn dataset_added(&mut self, ds: crate::api::Dataset, verb: &str, cx: &mut Context<Self>) {
        let (id, name) = (ds.id.clone(), ds.name.clone());
        let body = format!("{} images · {}", int(ds.splits.total()), ds.classes.join(", "));
        // Registering a folder twice answers with the dataset it already is: say so, and leave
        // the list's order alone.
        // Unknown while the list isn't loaded: neither "added" nor "already added" is claimed then.
        let existed = self.store.datasets.ready().map(|l| l.iter().any(|d| d.id == ds.id));
        if let Some(list) = self.store.datasets.ready_mut() {
            if let Some(slot) = list.iter_mut().find(|d| d.id == ds.id) {
                *slot = ds;
            } else {
                list.insert(0, ds);
            }
        }
        self.toast_with(
            if existed == Some(true) { Tone::Info } else { Tone::Accent },
            match existed {
                Some(true) => format!("{name} is already added"),
                Some(false) => format!("{verb} {name}"),
                None => format!("{name} is ready"),
            },
            Some(body.into()),
            Some((
                "Open".into(),
                Rc::new(move |this: &mut Self, w: &mut Window, cx: &mut Context<Self>| {
                    this.navigate(Route::Dataset(id.clone()), w, cx)
                }),
            )),
            cx,
        );
    }

    /// Resume a stopped or failed run from its last checkpoint.
    pub fn resume_run(&mut self, id: String, cx: &mut Context<Self>) {
        self.fetch(
            cx,
            move |api| api.resume_run(&id),
            |this, r, cx| match r {
                Ok(run) => {
                    let (rid, name) = (run.id.clone(), run.name.clone());
                    if let Some(runs) = this.store.runs.ready_mut()
                        && let Some(slot) = runs.iter_mut().find(|r| r.id == rid)
                    {
                        *slot = run;
                    }
                    // Resumed steps replace the earlier ones. Retire everything tied to the old
                    // attempt (stream, in-flight history, buffered events), then open a fresh
                    // stream; the dashboard reloads history once it is open.
                    this.subscriptions.remove(&rid);
                    *this.sub_attempt.entry(rid.clone()).or_default() += 1;
                    this.subscribing.remove(&rid);
                    this.store.reset_for_resume(&rid);
                    this.history_ticket(&rid, "metrics");
                    this.history_ticket(&rid, "logs");
                    this.history_ticket(&rid, "samples");
                    this.resubscribe(rid.clone(), cx);
                    this.toast(
                        Tone::Accent,
                        format!("Resuming {name}"),
                        Some("From its last finished epoch, with the optimizer state.".into()),
                        cx,
                    );
                }
                Err(e) => this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx),
            },
        );
    }

    /// Moves a waiting run in the training queue (servers with `queue_control`).
    pub fn move_queued(&mut self, id: String, action: crate::api::QueueAction, cx: &mut Context<Self>) {
        let req = crate::api::QueueRequest { action: Some(action), position: None };
        self.fetch(
            cx,
            move |api| api.queue_run(&id, req),
            move |this, r, cx| match r {
                Ok(run) => {
                    let name = run.name.clone();
                    let started = run.status != RunStatus::Queued;
                    if let Some(runs) = this.store.runs.ready_mut()
                        && let Some(slot) = runs.iter_mut().find(|r| r.id == run.id)
                    {
                        *slot = run;
                    }
                    let title =
                        if started { format!("Started {name}") } else { format!("{name} starts next") };
                    this.toast(Tone::Neutral, title, None, cx);
                }
                Err(e) => {
                    let msg = if e.code == "run_not_queued" {
                        "It has already left the queue.".to_string()
                    } else {
                        e.message.clone()
                    };
                    this.toast(Tone::Danger, "Couldn't move the run", Some(msg.into()), cx);
                }
            },
        );
    }

    pub fn stop_run(&mut self, id: String, cx: &mut Context<Self>) {
        self.confirm_stop = None;
        let rid = id.clone();
        // Shown as stopping right away; the status it had comes back if the server refuses.
        let mut prev = None;
        if let Some(runs) = self.store.runs.ready_mut()
            && let Some(r) = runs.iter_mut().find(|r| r.id == id)
        {
            prev = Some(std::mem::replace(&mut r.status, RunStatus::Stopping));
        }
        self.fetch(
            cx,
            move |api| api.stop_run(&rid),
            move |this, r, cx| match r {
                Ok(run) => {
                    // Keep the event subscription: the server may answer `stopping` and still send
                    // the last steps, a final validation and the terminal status.
                    let name = run.name.clone();
                    let epoch = run.epoch;
                    let stopping = run.status == RunStatus::Stopping;
                    if let Some(runs) = this.store.runs.ready_mut()
                        && let Some(slot) = runs.iter_mut().find(|r| r.id == run.id)
                    {
                        *slot = run;
                    }
                    let (title, body) = if stopping {
                        (
                            format!("Stopping {name}"),
                            "Finishing the current step; the last checkpoint is kept.".to_string(),
                        )
                    } else {
                        (format!("Stopped {name}"), format!("Last checkpoint kept (epoch {epoch})."))
                    };
                    this.toast(Tone::Neutral, title, Some(body.into()), cx);
                }
                Err(e) => {
                    // Unless events moved it on meanwhile, so Stop is offered again.
                    if let Some(prev) = prev
                        && let Some(runs) = this.store.runs.ready_mut()
                        && let Some(r) =
                            runs.iter_mut().find(|r| r.id == id && r.status == RunStatus::Stopping)
                    {
                        r.status = prev;
                    }
                    this.toast(Tone::Danger, "Could not stop the run", Some(e.message.into()), cx)
                }
            },
        );
        cx.notify();
    }

    pub fn export_model(&mut self, id: String, format: &'static str, cx: &mut Context<Self>) {
        self.export_open = false;
        self.fetch(
            cx,
            move |api| api.export_model(&id, format),
            |this, r, cx| match r {
                Ok(res) => {
                    let (v, u) = bytes(res.bytes);
                    let path = res.path.clone();
                    this.toast_with(
                        Tone::Success,
                        "Export complete",
                        Some(format!("{} · {v} {u}", display_path(&res.path)).into()),
                        Some((
                            "Show in folder".into(),
                            act(move |_, _, cx| cx.reveal_path(std::path::Path::new(&path))),
                        )),
                        cx,
                    );
                }
                Err(e) => this.toast(Tone::Danger, "Export failed", Some(e.message.into()), cx),
            },
        );
    }
}
