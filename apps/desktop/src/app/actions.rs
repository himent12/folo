//! What commands do on the server: generate or add a dataset, resume, reorder, stop and export.

use std::rc::Rc;

use folo_ui::{AppContext, Context, Focusable, Window};

use crate::api::*;
use crate::input::TextInput;
use crate::ui::*;

use super::forms::{SYNTH_CLASSES, SYNTH_IMGSZ, SynthForm, parse_images};
use super::lists::List;
use super::{DeleteTarget, FoloApp, Route, ServerMode, act};

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
            sent_name: String::new(),
            error: None,
        });
        cx.notify();
    }

    pub fn close_synth(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        // Closed while the server generates: the job goes on, and its result comes as a toast
        // later; say so now rather than let it vanish without a word.
        if let Some(f) = self.synth.take().filter(|f| f.submitting) {
            self.toast(
                Tone::Info,
                format!("Generating {} in the background", f.sent_name),
                Some("It's added to Datasets when it's ready.".into()),
                cx,
            );
        }
        self.restore_focus(window);
        cx.notify();
    }

    pub fn submit_synth(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        let Some(f) = self.synth.as_mut() else { return };
        if f.submitting {
            return;
        }
        let name = f.name.read(cx).text().trim().to_string();
        // Each field says what's wrong under itself, and Generate is off until both are right
        // (Enter included).
        let images = parse_images(&f.images.read(cx).text());
        let (Some(images), false) = (images, name.is_empty()) else { return };
        f.submitting = true;
        f.sent_name = name.clone();
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
            if let Some(p) = picked(&this, rx.await, cx) {
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

    /// A new dataset from the server: list it where the server lists it and say so, with a way to
    /// open it.
    pub(super) fn dataset_added(&mut self, ds: crate::api::Dataset, verb: &str, cx: &mut Context<Self>) {
        let (id, name) = (ds.id.clone(), ds.name.clone());
        // A folder registered again after its dataset was deleted can come back under its id.
        self.deleted.remove(&id);
        let body = format!("{} · {}", counted(ds.splits.total(), "image", "images"), ds.classes.join(", "));
        // Registering a folder twice answers with the dataset it already is: say so, and leave
        // the list's order alone.
        // Unknown while the list isn't loaded: neither "added" nor "already added" is claimed then.
        let existed = self.store.datasets.ready().map(|l| l.iter().any(|d| d.id == ds.id));
        // Registered, but nothing in it can train: a warning, not a success.
        let unusable = ds.splits.total() == 0;
        if let Some(list) = self.store.datasets.ready_mut() {
            upsert_dataset(list, ds);
        }
        self.changed_here(List::Datasets, cx);
        let (tone, title, body) = match existed {
            _ if unusable => (
                Tone::Warning,
                format!("{name} has no usable images"),
                "FOLO added it but can't read any of its images with their labels. Its page says what's wrong."
                    .to_string(),
            ),
            Some(true) => (Tone::Info, format!("{name} is already added"), body),
            Some(false) => (Tone::Accent, format!("{verb} {name}"), body),
            None => (Tone::Accent, format!("{name} is ready"), body),
        };
        self.toast_with(
            tone,
            title,
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
        // A run with no finished epoch starts over (the server keeps its config): said as that.
        let again = self.store.run(&id).is_some_and(|r| !r.can_continue());
        self.fetch(
            cx,
            move |api| api.resume_run(&id),
            move |this, r, cx| match r {
                Ok(run) => {
                    let (rid, name) = (run.id.clone(), run.name.clone());
                    if let Some(runs) = this.store.runs.ready_mut()
                        && let Some(slot) = runs.iter_mut().find(|r| r.id == rid)
                    {
                        *slot = run;
                    }
                    this.changed_here(List::Runs, cx);
                    // Resumed steps replace the earlier ones. Retire everything tied to the old
                    // attempt (stream, in-flight history, buffered events), then open a fresh
                    // stream; the dashboard reloads history once it is open.
                    this.subscriptions.remove(&rid);
                    *this.sub_attempt.entry(rid.clone()).or_default() += 1;
                    this.subscribing.remove(&rid);
                    this.stream_lost.remove(&rid);
                    this.history_retry.remove(&rid);
                    this.store.reset_for_resume(&rid);
                    this.history_ticket(&rid, "metrics");
                    this.history_ticket(&rid, "logs");
                    this.history_ticket(&rid, "samples");
                    this.resubscribe(rid.clone(), cx);
                    let (title, body) = if again {
                        (format!("Starting {name} again"), "No epoch had finished, so it starts over.")
                    } else {
                        (
                            format!("Resuming {name}"),
                            "From its last finished epoch, with the optimizer state.",
                        )
                    };
                    this.toast(Tone::Accent, title, Some(body.into()), cx);
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
                    this.changed_here(List::Runs, cx);
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

    /// Deletes what the confirmation named; on success its page (if open) gives way to the list.
    pub fn delete_confirmed(&mut self, cx: &mut Context<Self>) {
        let Some(target) = self.confirm_delete.take() else { return };
        let name = match &target {
            DeleteTarget::Run(id) => self.store.run(id).map_or(id.clone(), |r| r.name.clone()),
            DeleteTarget::Dataset(id) => {
                self.store.dataset_summary(id).map_or(id.clone(), |d| d.name.clone())
            }
        };
        // A generated set's folder goes too when asked (and still allowed: no run started on it since).
        let files = matches!(&target, DeleteTarget::Dataset(id) if self.delete_files && self.dataset_files_deletable(id));
        let images = match &target {
            DeleteTarget::Dataset(id) => self.store.dataset_summary(id).map(|d| {
                counted(d.splits.train + d.splits.val + d.splits.test, "generated image", "generated images")
            }),
            DeleteTarget::Run(_) => None,
        };
        let call = target.clone();
        self.fetch(
            cx,
            move |api| match &call {
                DeleteTarget::Run(id) => api.delete_run(id),
                DeleteTarget::Dataset(id) => api.delete_dataset(id, files),
            },
            move |this, r, cx| match r {
                Ok(()) => match &target {
                    DeleteTarget::Run(id) => {
                        this.forget_run(id, cx);
                        this.toast(Tone::Info, format!("Deleted {name}"), None, cx);
                    }
                    DeleteTarget::Dataset(id) => {
                        this.forget_dataset(id, cx);
                        // Studio's words.
                        let body = if files {
                            format!("{name} and its {} were deleted.", images.as_deref().unwrap_or("files"))
                        } else {
                            format!("{name} is no longer registered. Files on disk were not touched.")
                        };
                        this.toast(Tone::Info, "Dataset removed", Some(body.into()), cx);
                    }
                },
                Err(e) => {
                    let verb = if matches!(target, DeleteTarget::Run(_)) { "delete" } else { "remove" };
                    let body = removal_refused(&e, files);
                    this.toast(Tone::Danger, format!("Couldn't {verb} {name}"), Some(body.into()), cx)
                }
            },
        );
        cx.notify();
    }

    /// Drops a deleted run (and the models it made, fetched again) from what the app holds, and
    /// leaves its page (or one of its models') for the list.
    fn forget_run(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some(r) = self.store.run(id) {
            self.deleted.insert(id.to_string(), r.name.clone());
        }
        if let Some(runs) = self.store.runs.ready_mut() {
            runs.retain(|r| r.id != id);
        }
        self.changed_here(List::Runs, cx);
        self.store.run_data.remove(id);
        self.store.samples.remove(id);
        self.store.forget_logs(id);
        self.forget_history(id);
        let made: Vec<String> = self
            .store
            .models
            .ready()
            .into_iter()
            .flatten()
            .filter(|m| m.run_id.as_deref() == Some(id))
            .map(|m| {
                self.deleted.insert(m.id.clone(), m.name.clone());
                m.id.clone()
            })
            .collect();
        if self.play_model_made_by(id, &made) {
            let name = self.play_model.as_ref().and_then(|m| self.deleted.get(m)).cloned();
            self.drop_play_model(name);
        }
        // Gone from the list now, which is fetched again (an answer asked for before the delete still
        // lists them, and is dropped). Playground takes the best model left at once.
        if let Some(models) = self.store.models.ready_mut() {
            models.retain(|m| !made.contains(&m.id));
        }
        self.reconcile_play_model(cx);
        self.changed_here(List::Models, cx);
        // An open select may list it, or its models: its highlight is an index into that list.
        self.menu_open = None;
        self.leave_deleted();
    }

    /// The open page of something just deleted gives way to its list (Back skips it too).
    fn leave_deleted(&mut self) {
        if let Some((_, list)) = super::navigation::deleted_route(&self.deleted, &self.route) {
            self.route_pending = Some(list);
        }
    }

    /// Drops an unregistered dataset from what the app holds, and leaves its page for the list.
    fn forget_dataset(&mut self, id: &str, cx: &mut Context<Self>) {
        if let Some(d) = self.store.dataset_summary(id) {
            self.deleted.insert(id.to_string(), d.name.clone());
        }
        self.ds_recheck.remove(id);
        if self.inspector.as_ref().is_some_and(|i| i.dataset == id) {
            self.take_inspector(cx);
            self.inspector_return = None;
        }
        if let Some(list) = self.store.datasets.ready_mut() {
            list.retain(|d| d.id != id);
        }
        self.changed_here(List::Datasets, cx);
        self.store.dataset.remove(id);
        self.store.images.remove(id);
        self.store.previews.remove(id);
        // An open select may list it (New run's datasets): its highlight is an index into the list.
        // The forms hold their dataset by id, so the choice in them stays what it was.
        self.menu_open = None;
        self.leave_deleted();
    }

    /// Takes a run that never started out of the queue: the server deletes it (docs/API.md), so it
    /// leaves the list too, as in Studio, instead of staying behind as a stopped run.
    pub fn remove_queued(&mut self, id: String, cx: &mut Context<Self>) {
        self.confirm_stop = None;
        let name = self.store.run(&id).map_or(id.clone(), |r| r.name.clone());
        let rid = id.clone();
        self.fetch(
            cx,
            move |api| api.delete_run(&rid),
            move |this, r, cx| match r {
                Ok(()) => {
                    this.forget_run(&id, cx);
                    this.toast(
                        Tone::Neutral,
                        format!("Removed {name} from the queue"),
                        Some("It hadn't started, so nothing was lost.".into()),
                        cx,
                    );
                }
                Err(e) => {
                    this.toast(Tone::Danger, format!("Couldn't remove {name}"), Some(e.message.into()), cx)
                }
            },
        );
        cx.notify();
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
        // An answer asked for before would show it running again (and offer Stop).
        self.list_tickets.changed(List::Runs);
        self.fetch(
            cx,
            move |api| api.stop_run(&rid),
            move |this, r, cx| match r {
                Ok(run) => {
                    // Keep the event subscription: the server may answer `stopping` and still send
                    // the last steps, a final validation and the terminal status.
                    let name = run.name.clone();
                    let epoch = run.epoch;
                    // No checkpoint before the first epoch is validated: nothing is claimed then.
                    let kept = run.can_continue();
                    let stopping = run.status == RunStatus::Stopping;
                    let dequeued = prev == Some(RunStatus::Queued) && run.cloud.is_none();
                    if let Some(runs) = this.store.runs.ready_mut()
                        && let Some(slot) = runs.iter_mut().find(|r| r.id == run.id)
                    {
                        *slot = run;
                    }
                    this.changed_here(List::Runs, cx);
                    let nothing = "No epoch had finished, so nothing is saved.".to_string();
                    let (title, body) = if dequeued {
                        // A resumed run that waited to continue: it keeps what it trained.
                        (
                            format!("Removed {name} from the queue"),
                            if kept { "It keeps what it trained so far.".to_string() } else { nothing },
                        )
                    } else if stopping {
                        (
                            format!("Stopping {name}"),
                            if kept {
                                "Finishing the current batch; the last finished epoch's checkpoint is kept."
                                    .to_string()
                            } else {
                                format!("Finishing the current batch. {nothing}")
                            },
                        )
                    } else if kept {
                        (format!("Stopped {name}"), format!("Last checkpoint kept (epoch {epoch})."))
                    } else {
                        (format!("Stopped {name}"), nothing)
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

/// The first path a file dialog answered with; `None` when it was cancelled. A dialog that could
/// not open (off screen they are refused: `FOLO_UI_OFFSCREEN=1`) says why in a toast.
pub(super) fn picked(
    this: &folo_ui::WeakEntity<FoloApp>,
    answer: Result<anyhow::Result<Option<Vec<std::path::PathBuf>>>, folo_ui::oneshot::Canceled>,
    cx: &mut folo_ui::AsyncApp,
) -> Option<std::path::PathBuf> {
    match answer {
        Ok(Ok(paths)) => paths?.into_iter().next(),
        Ok(Err(e)) => {
            this.update(cx, |this, cx| {
                this.toast(Tone::Danger, "The file dialog didn't open", Some(e.to_string().into()), cx)
            })
            .ok();
            None
        }
        Err(_) => None,
    }
}

/// Puts a dataset the server answered with into the list in the server's order (oldest first,
/// docs/API.md: a new one last), or replaces it in place when it is already there, so the list
/// reads the same now and after a restart.
fn upsert_dataset(list: &mut Vec<crate::api::Dataset>, ds: crate::api::Dataset) {
    match list.iter_mut().find(|d| d.id == ds.id) {
        Some(slot) => *slot = ds,
        None => list.push(ds),
    }
}

/// What a refused removal says: the server's reason, then the way forward in the app's words. A
/// file deletion refused before anything changed (another dataset may read that folder, or could
/// not be checked: `dataset_in_use`; a folder FOLO did not write: `invalid_query`) can still
/// unregister the dataset alone, which the server's hint names by its API flag (`files=true`),
/// meaningless here: the switch in the dialog is what the person knows. Any other error keeps its
/// own hint (a run active refuses unregistering too; a folder that failed to delete was already
/// unregistered; `unauthorized` names the token to fix).
fn removal_refused(e: &ApiError, files: bool) -> String {
    let mut body = e.message.clone();
    let way = if files && matches!(e.code.as_str(), "dataset_in_use" | "invalid_query") {
        Some("To unregister it and keep its files, remove it again with “Also delete” off.".to_string())
    } else {
        e.hint.clone().filter(|h| !h.contains("files=true") && !h.contains("/api/"))
    };
    if let Some(way) = way {
        if !body.ends_with(['.', '!', '?']) {
            body.push('.');
        }
        body.push(' ');
        let mut c = way.chars();
        body.extend(c.next().map(|f| f.to_ascii_uppercase()));
        body.push_str(c.as_str());
    }
    body
}

#[cfg(test)]
mod tests {
    use crate::api::{ApiError, Backend};

    #[test]
    fn a_refused_file_deletion_says_how_to_unregister_alone() {
        let e = ApiError::new(
            "dataset_in_use",
            "the files of broken could not be checked, so they may be in that folder",
        )
        .with_hint("fix broken first, or unregister this dataset without files=true");
        let body = super::removal_refused(&e, true);
        assert!(body.starts_with("the files of broken could not be checked"), "{body}");
        assert!(body.contains("“Also delete” off"), "{body}");
        assert!(!body.contains("files=true"), "API words stay out: {body}");
        // Unregistering alone failed: the server's hint, unless it speaks the API.
        let e = ApiError::new("run_active", "run a is queued or training")
            .with_hint("wait for it to end, or stop it first");
        assert_eq!(
            super::removal_refused(&e, false),
            "run a is queued or training. Wait for it to end, or stop it first"
        );
        let e = ApiError::new("run_active", "run 'a' is still active")
            .with_hint("stop it first: POST /api/runs/a/stop");
        assert_eq!(super::removal_refused(&e, false), "run 'a' is still active");
        // The server away: nothing to change in the dialog.
        let e = ApiError::new("server_unreachable", "Lost connection to folo serve.");
        assert_eq!(super::removal_refused(&e, true), "Lost connection to folo serve.");
        // Errors where unregistering alone would not help keep their own way forward: a run
        // refuses it too; a folder that failed to delete was already unregistered; a token.
        let e = ApiError::new("run_active", "run a is queued or training")
            .with_hint("wait for it to end, or stop it first");
        assert_eq!(
            super::removal_refused(&e, true),
            "run a is queued or training. Wait for it to end, or stop it first"
        );
        let e = ApiError::new("internal", "broken was unregistered, but deleting 'D:/x' failed: denied");
        assert!(!super::removal_refused(&e, true).contains("remove it again"));
        let e = ApiError::new("unauthorized", "folo serve wants a token")
            .with_hint("enter the server's token in Settings");
        assert_eq!(
            super::removal_refused(&e, true),
            "folo serve wants a token. Enter the server's token in Settings"
        );
        // A folder FOLO did not write is refused before anything changed: unregistering still can.
        let e = ApiError::new("invalid_query", "'D:/photos' is not a folder FOLO wrote into its data dir")
            .with_hint("unregister it without files=true");
        assert!(super::removal_refused(&e, true).ends_with("with “Also delete” off."));
    }

    #[test]
    fn a_new_dataset_goes_where_the_server_lists_it() {
        let api = crate::api::mock::MockBackend::new(4200, true);
        let all = api.datasets().unwrap();
        let mut list = all[..2].to_vec();
        super::upsert_dataset(&mut list, all[2].clone());
        let ids: Vec<&str> = list.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, [&all[0].id, &all[1].id, &all[2].id].map(String::as_str));
        // Added again (a folder registered twice): replaced where it is.
        super::upsert_dataset(&mut list, crate::api::Dataset { name: "again".into(), ..all[0].clone() });
        assert_eq!((list.len(), list[0].name.as_str()), (3, "again"));
    }
}
