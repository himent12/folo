//! The New run dialog's state: opening it (fresh, from a run, or as the kept draft), the config it
//! makes, the server's estimate for it, and submitting it. `crate::overlays` draws it.

use std::rc::Rc;
use std::time::Duration;

use folo_ui::point;
use folo_ui::{App, AppContext, Context, Focusable, Window, px};

use crate::api::*;
use crate::input::{InputEvent, TextInput};
use crate::store::*;
use crate::ui::*;

use super::forms::{
    ARCHS, BATCHES, DEVICES, IMGSZ, NewRunForm, OPTIMIZERS, again_run_name, auto_close_mosaic, batch_index,
    carry_over, free_run_name, parse_epochs, parse_lr, run_name_error,
};
use super::{FoloApp, Route, RunFilter};

impl FoloApp {
    /// The form to keep as a draft when the dialog closes without starting a run: none while a
    /// submission is pending (that run is being created; keeping it would allow a duplicate).
    pub(super) fn keep_draft(&mut self, mut f: NewRunForm) -> Option<NewRunForm> {
        if f.submitting {
            return None;
        }
        f.error = None;
        Some(f)
    }

    /// The New run form's dataset; none before the list is in, or once it was deleted.
    pub fn nr_dataset(&self) -> Option<&Dataset> {
        self.store.dataset_summary(&self.new_run.as_ref()?.dataset)
    }

    /// The New run form's dataset was removed (while it was open, or before it was opened for it, from
    /// a cached page or a run's Train again): the list is in without it, and the form asks for another
    /// before it starts.
    pub fn nr_dataset_gone(&self) -> bool {
        self.new_run.is_some() && self.store.datasets.ready().is_some() && self.nr_dataset().is_none()
    }

    /// What New run calls its removed dataset: the name it was deleted under, or the one a run trained
    /// on it recorded, else "That dataset".
    pub fn nr_gone_name(&self) -> String {
        let Some(id) = self.new_run.as_ref().map(|f| &f.dataset) else { return String::new() };
        let recorded = || {
            let runs = self.store.runs.ready()?;
            runs.iter()
                .filter(|r| r.config.dataset_id == *id)
                .find_map(|r| r.dataset.as_ref())
                .map(|d| d.name.clone())
        };
        self.deleted.get(id).cloned().or_else(recorded).unwrap_or_else(|| "That dataset".into())
    }

    pub fn close_new_run(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.new_run_draft = self.new_run.take().and_then(|f| self.keep_draft(f));
        self.restore_focus(window);
        cx.notify();
    }

    /// The New run dialog prefilled from an earlier run; `note` explains any adjustment.
    pub fn open_new_run_from(
        &mut self,
        config: &TrainConfig,
        note: Option<String>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.open_new_run(Some(&config.dataset_id), window, cx);
        let idx = |list: &[u32], v: u32| list.iter().position(|x| *x == v);
        let runs = self.store.runs.ready().map(Vec::as_slice).unwrap_or_default();
        let name = again_run_name(&config.name, runs);
        let cloud_on = self.cloud_on();
        if let Some(f) = self.new_run.as_mut() {
            f.arch = ARCHS.iter().position(|a| *a == config.arch).unwrap_or(f.arch);
            f.epochs = config.epochs as f32;
            f.batch = batch_index(config.batch);
            f.imgsz = idx(&IMGSZ, config.imgsz).unwrap_or(f.imgsz);
            f.device = DEVICES.iter().position(|d| *d == config.device).unwrap_or(0);
            f.prefill_optimizer(config.optimizer);
            f.lr0 = config.lr0;
            let (ei, li) = (f.epochs_input.clone(), f.lr_input.clone());
            let (ci, cv) = (f.close_input.clone(), config.augment.close_mosaic.to_string());
            ci.update(cx, |i, cx| i.set_text(cv, cx));
            let (ev, lv) = (config.epochs.to_string(), format!("{}", config.lr0));
            ei.update(cx, |i, cx| i.set_text(ev, cx));
            li.update(cx, |i, cx| i.set_text(lv, cx));
            f.mosaic = config.augment.mosaic;
            f.mixup = config.augment.mixup;
            f.fliplr = config.augment.fliplr;
            f.close_mosaic = config.augment.close_mosaic as f32;
            // That run's default still follows the epochs; a value it set by hand stays.
            f.close_auto = config.augment.close_mosaic == auto_close_mosaic(config.epochs);
            f.scale = config.augment.scale;
            f.amp = config.amp;
            f.e2e = config.head == Head::O2o;
            f.note = note;
            f.weights = config.weights.clone();
            f.base = Some(config.clone());
            // A cloud run's connection and offer, when they still exist (checked once the
            // connections and offers are in), and its time limit as it was.
            if let Some(c) = config.compute.as_ref().filter(|_| cloud_on) {
                f.run_on = Some(c.connection_id.clone());
                f.want_offer = Some(c.offer_id.clone());
                f.max_hours = c.max_hours;
                f.hours_edited = true;
                let (hi, hv) = (f.hours_input.clone(), crate::ui::hours_value(c.max_hours));
                hi.update(cx, |i, cx| i.set_text(hv, cx));
            }
            // Named after that run, not suggested from the choices: it stays as it is.
            f.name_suggested = None;
            let input = f.name.clone();
            input.update(cx, |i, cx| i.set_text(name, cx));
        }
        self.sync_compute(cx);
        cx.notify();
    }

    pub fn open_new_run(&mut self, dataset: Option<&str>, window: &mut Window, cx: &mut Context<Self>) {
        // Nothing to train on until the dataset list is in (Ctrl+N included).
        if self.store.datasets.ready().is_none_or(|d| d.is_empty()) {
            return;
        }
        if self.new_run.is_none() {
            self.remember_focus(window, cx);
        }
        // A plain New run picks up a draft left by Esc or Cancel; one for a specific dataset (or
        // prefilled from a run) starts fresh.
        // Its dataset deleted meanwhile (or the list unloaded), the draft is dropped.
        if let Some(draft) = self.new_run_draft.take()
            && self.new_run.is_none()
            && dataset.is_none()
            && self.store.dataset_summary(&draft.dataset).is_some()
        {
            let fh = draft.name.read(cx).focus_handle(cx);
            self.new_run = Some(draft);
            self.nr_recheck_dataset(cx);
            window.focus(&fh);
            cx.notify();
            return;
        }
        let list = self.store.datasets.ready().map(Vec::as_slice).unwrap_or_default();
        // A dataset asked for by id that is gone (removed since its page or run was read) is never
        // swapped for another in silence: the form opens on it, says it was removed and asks for a
        // choice. Only a plain New run starts on the first dataset.
        let (ds_id, ds_name) = match dataset {
            Some(id) => match list.iter().find(|d| d.id == id) {
                Some(ds) => (ds.id.clone(), ds.name.clone()),
                None => (id.to_string(), self.deleted.get(id).cloned().unwrap_or_else(|| "run".into())),
            },
            None => match list.first() {
                Some(ds) => (ds.id.clone(), ds.name.clone()),
                None => return,
            },
        };
        // The last started run's choices, else the defaults (s, 640 px, 100 epochs).
        let (arch, imgsz, epochs) = self.new_run_last.unwrap_or((1, 2, 100.0));
        let runs = self.store.runs.ready().map(Vec::as_slice).unwrap_or_default();
        let suggested = free_run_name(&ds_name, ARCHS[arch], runs);
        let name = cx.new(|cx| {
            let mut i = TextInput::new("run name", cx);
            i.set_text(suggested.clone(), cx);
            i
        });
        let fh = name.read(cx).focus_handle(cx);
        let epochs_input = cx.new(|cx| {
            let mut i = TextInput::new("100", cx);
            i.set_text(format!("{epochs:.0}"), cx);
            i
        });
        let close = auto_close_mosaic(epochs.round() as u32);
        let lr_input = cx.new(|cx| {
            let mut i = TextInput::new("0.01", cx);
            i.set_text("0.010", cx);
            i
        });
        let close_input = cx.new(|cx| {
            let mut i = TextInput::new("10", cx);
            i.set_text(close.to_string(), cx);
            i
        });
        let hours_input = cx.new(|cx| {
            let mut i = TextInput::new("12", cx);
            i.set_text(crate::ui::hours_value(folo_api::DEFAULT_MAX_HOURS), cx);
            i
        });
        let subs = vec![
            cx.subscribe(&name, |this: &mut Self, _, ev: &InputEvent, _| {
                let InputEvent::Changed(s) = ev;
                // Any other text than the suggestion is the user's: it stays as typed.
                if let Some(f) = this.new_run.as_mut()
                    && f.name_suggested.as_deref().is_some_and(|n| n != s.as_ref())
                {
                    f.name_suggested = None;
                }
            }),
            cx.subscribe(&hours_input, |this: &mut Self, _, ev: &InputEvent, cx| {
                let InputEvent::Changed(s) = ev;
                if let Some(f) = this.new_run.as_mut() {
                    match crate::cloud::parse_hours(s) {
                        Some(v) => {
                            // Any other value than the one the form put there is the user's.
                            if crate::ui::hours_value(v) != crate::ui::hours_value(f.max_hours) {
                                f.hours_edited = true;
                            }
                            f.max_hours = v;
                            f.hours_invalid = false;
                        }
                        None => f.hours_invalid = true,
                    }
                    f.compute_error.take_if(|(c, _)| *c == crate::cloud::ComputeField::StopAfter);
                    cx.notify();
                }
            }),
            cx.subscribe(&close_input, |this: &mut Self, _, ev: &InputEvent, cx| {
                let InputEvent::Changed(s) = ev;
                if let Some(f) = this.new_run.as_mut() {
                    // Any other text than the default the form put there is the user's.
                    if s.trim() != format!("{:.0}", f.close_mosaic) {
                        f.close_auto = false;
                    }
                    f.check_close(s);
                    cx.notify();
                }
            }),
            cx.subscribe(&epochs_input, |this: &mut Self, _, ev: &InputEvent, cx| {
                let InputEvent::Changed(s) = ev;
                if let Some(f) = this.new_run.as_mut() {
                    match parse_epochs(s) {
                        Some(v) => {
                            f.epochs = v as f32;
                            f.epochs_invalid = false;
                        }
                        None => f.epochs_invalid = true,
                    }
                    // Close mosaic follows the epochs while it is the default; a value set by
                    // hand is bounded by them: recheck it against the new count.
                    if let Some(v) = f.follow_epochs() {
                        f.close_input.clone().update(cx, |i, cx| i.set_text(v, cx));
                    } else if !f.close_auto {
                        let close = f.close_input.read(cx).text();
                        f.check_close(&close);
                    }
                    cx.notify();
                }
            }),
            cx.subscribe(&lr_input, |this: &mut Self, _, ev: &InputEvent, cx| {
                let InputEvent::Changed(s) = ev;
                if let Some(f) = this.new_run.as_mut() {
                    match parse_lr(s) {
                        Some(v) => {
                            f.lr0 = v;
                            f.lr_invalid = false;
                        }
                        None => {
                            f.lr_invalid = true;
                            // The rate lives under Advanced: keep the group open while it's being
                            // fixed, so the first valid keystroke doesn't fold the field away.
                            f.advanced = true;
                        }
                    }
                    cx.notify();
                }
            }),
        ];
        self.new_run = Some(NewRunForm {
            name,
            name_suggested: Some(suggested),
            dataset: ds_id,
            arch,
            epochs,
            batch: batch_index(16),
            imgsz,
            device: 0,
            optimizer: 0,
            optimizer_unknown: false,
            lr0: 0.01,
            mosaic: 1.0,
            mixup: 0.0,
            fliplr: 0.5,
            close_mosaic: close as f32,
            scale: 0.5,
            note: None,
            weights: None,
            weights_name: None,
            base: None,
            epochs_input,
            lr_input,
            close_input,
            close_invalid: false,
            close_auto: true,
            amp: false,
            e2e: false,
            advanced: false,
            run_on: None,
            offer: None,
            max_hours: folo_api::DEFAULT_MAX_HOURS,
            hours_input,
            hours_invalid: false,
            hours_edited: false,
            compute_error: None,
            want_offer: None,
            epochs_invalid: false,
            lr_invalid: false,
            _subs: subs,
            submitting: false,
            error: None,
        });
        self.nr_recheck_dataset(cx);
        window.focus(&fh);
        cx.notify();
    }

    /// Suggests the name again for the form's dataset and architecture, unless it was typed.
    pub fn resuggest_run_name(&mut self, cx: &mut Context<Self>) {
        let Some(f) = self.new_run.as_mut() else { return };
        let Some(ds) = self.store.dataset_summary(&f.dataset) else { return };
        let runs = self.store.runs.ready().map(Vec::as_slice).unwrap_or_default();
        let name = free_run_name(&ds.name, ARCHS[f.arch], runs);
        if f.name_suggested.as_ref().is_none_or(|s| *s == name) {
            return;
        }
        f.name_suggested = Some(name.clone());
        f.name.clone().update(cx, |i, cx| i.set_text(name, cx));
    }

    /// The New run form as the config it would submit (its name as typed), or None without a
    /// form or dataset. Submitting and the server's estimate both use it.
    pub fn form_config(&self, cx: &App) -> Option<TrainConfig> {
        let form = self.new_run.as_ref()?;
        let compute = crate::cloud::form_compute(
            self.cloud_on(),
            form.run_on.as_deref(),
            form.offer.as_deref(),
            form.max_hours,
        );
        // On a cloud instance mixed precision is checked when it boots, not against this machine.
        let amp_ok = compute.is_some() || self.amp_blocked(DEVICES[form.device]).is_none();
        let ds = self.store.dataset_summary(&form.dataset)?;
        Some(carry_over(
            form.base.as_ref(),
            TrainConfig {
                name: form.name.read(cx).text().trim().to_string(),
                dataset_id: ds.id.clone(),
                arch: ARCHS[form.arch],
                epochs: form.epochs.round() as u32,
                batch: BATCHES[form.batch],
                imgsz: IMGSZ[form.imgsz],
                // The worker picks its GPU.
                device: if compute.is_some() { "auto".into() } else { DEVICES[form.device].into() },
                optimizer: OPTIMIZERS[form.optimizer],
                lr0: form.lr0,
                weights: form.weights.clone(),
                augment: Augment {
                    mosaic: form.mosaic,
                    mixup: form.mixup,
                    fliplr: form.fliplr,
                    close_mosaic: form.close_mosaic.round() as u32,
                    scale: form.scale,
                    ..Augment::default()
                },
                seed: 0,
                // Only where it can run: not the CPU, nor a server that says it can't.
                amp: form.amp && amp_ok,
                head: if form.e2e { Head::O2o } else { Head::O2m },
                compute,
            },
        ))
    }

    /// Why New run can't train on its dataset: one with no training image (every image of the
    /// folder failed to read), refused by the server only after Start otherwise.
    pub fn dataset_blocked(&self) -> Option<String> {
        let f = self.new_run.as_ref()?;
        let ds = self.store.dataset_summary(&f.dataset)?;
        (ds.splits.train == 0).then(|| format!("{} has no training images. Pick another dataset.", ds.name))
    }

    /// The form's dataset has no training image by the list: ask the server for it again (its
    /// page indexes the folder afresh), so images added since lift the block.
    pub fn nr_recheck_dataset(&mut self, cx: &mut Context<Self>) {
        let Some(id) = self.new_run.as_ref().map(|f| f.dataset.clone()) else { return };
        if self.store.dataset_summary(&id).is_none_or(|d| d.splits.train > 0) {
            return;
        }
        self.fetch(
            cx,
            move |api| api.dataset(&id),
            |this, r, cx| {
                let Ok(ds) = r else { return };
                if let Some(slot) =
                    this.store.datasets.ready_mut().and_then(|l| l.iter_mut().find(|d| d.id == ds.id))
                {
                    *slot = ds;
                    cx.notify();
                }
            },
        );
    }

    /// Why New run can't send its config yet because of its compute choice: a connection chosen
    /// whose offers are still out, no offer picked, or a Stop after that doesn't parse.
    pub fn compute_blocked(&self) -> Option<&'static str> {
        let f = self.new_run.as_ref()?;
        let conn = f.run_on.as_deref().filter(|_| self.cloud_on())?;
        if f.hours_invalid {
            Some("Fix the highlighted fields first")
        } else if f.offer.is_some() {
            None
        } else if self.cloud.offers.get(conn).is_none_or(|l| l.is_loading()) {
            Some("Loading offers…")
        } else {
            Some("Choose a GPU first")
        }
    }

    /// Asks the server how long the New run form's config would take (servers with
    /// `run_estimate`), once per distinct config.
    pub fn ensure_estimate(&mut self, cx: &mut Context<Self>) {
        if !self.store.features.iter().any(|f| f == "run_estimate") {
            return;
        }
        // A value being typed that doesn't parse yet: no estimate for the last good one.
        // Nor for a cloud run without its offer yet (the config would be this machine's).
        if self.new_run.as_ref().is_some_and(|f| {
            f.epochs_invalid
                || f.close_invalid
                || (f.lr_invalid && f.optimizer != 0)
                || f.optimizer_blocks().is_some()
        }) || self.compute_blocked().is_some()
        {
            // Cancel a pending ask; an unanswered key would otherwise stay "loading" for good.
            self.nr_estimate_gen += 1;
            if matches!(self.nr_estimate, Some((_, Load::Loading))) {
                self.nr_estimate = None;
            }
            return;
        }
        let Some(mut config) = self.form_config(cx) else { return };
        // The name doesn't change the estimate (and an empty one mustn't fail it).
        config.name = "estimate".into();
        let key = serde_json::to_string(&config).unwrap_or_default();
        if self.nr_estimate.as_ref().is_some_and(|(k, _)| *k == key) {
            return;
        }
        self.nr_estimate = Some((key.clone(), Load::Loading));
        self.nr_estimate_gen += 1;
        let generation = self.nr_estimate_gen;
        // Settled first (250 ms; at once for the frozen scenes), so a slider drag asks once for
        // the value it stops on rather than for every step.
        if self.frozen {
            self.request_estimate(key, config, cx);
            return;
        }
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(Duration::from_millis(250)).await;
            this.update(cx, |this, cx| {
                if this.nr_estimate_gen == generation {
                    this.request_estimate(key, config, cx);
                }
            })
            .ok();
        })
        .detach();
    }

    fn request_estimate(&mut self, key: String, config: TrainConfig, cx: &mut Context<Self>) {
        self.fetch(
            cx,
            move |api| api.estimate_run(&config),
            move |this, r, cx| {
                if this.nr_estimate.as_ref().is_none_or(|(k, _)| *k != key) {
                    return;
                }
                if let Ok(e) = &r {
                    this.nr_estimate_last = Some(e.clone());
                }
                this.nr_estimate = Some((key, r.map_or_else(Load::Failed, Load::Ready)));
                // Stop after follows the estimate (until typed).
                this.sync_compute(cx);
                cx.notify();
            },
        );
    }

    pub fn submit_new_run(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let config = self.form_config(cx);
        let compute_blocked = self.compute_blocked().is_some();
        let dataset_blocked = self.dataset_blocked().is_some();
        let weights_gone = self.nr_weights_gone().is_some();
        let Some(form) = self.new_run.as_mut() else { return };
        // Auto sets its own rate, so an unparsed rate entry doesn't block it.
        let lr_invalid = form.lr_invalid && form.optimizer != 0;
        if form.submitting
            || form.epochs_invalid
            || lr_invalid
            || form.close_invalid
            || form.optimizer_blocks().is_some()
            || compute_blocked
            || dataset_blocked
            || weights_gone
        {
            return;
        }
        let Some(config) = config else { return };
        if let Some(why) = run_name_error(&config.name) {
            form.error = Some(ApiError::new("invalid_name", why));
            // The error shows at the top of the form, by the Name field: go there.
            let name_fh = form.name.read(cx).focus_handle(cx);
            window.focus(&name_fh);
            self.dialog_scroll.set_offset(point(px(0.), px(0.)));
            cx.notify();
            return;
        }
        form.submitting = true;
        // Which form this request belongs to (its name input, kept by a restored draft): the
        // dialog can be dismissed and another run submitted before this one answers.
        let form_id = form.name.entity_id();
        let _ = window;
        self.fetch(
            cx,
            move |api| api.create_run(config),
            move |this, r, cx| {
                let mine = this.new_run.as_ref().is_some_and(|f| f.name.entity_id() == form_id);
                match r {
                    Ok(run) => {
                        let id = run.id.clone();
                        let name = run.name.clone();
                        if let Some(runs) = this.store.runs.ready_mut() {
                            runs.insert(0, run);
                        }
                        this.changed_here(super::lists::List::Runs, cx);
                        this.subscribe_active(cx);
                        let run = this.store.run(&id).cloned();
                        let cloud = run.as_ref().and_then(|r| Some(this.cloud_device(r, r.cloud.as_ref()?)));
                        // The server answers `queued` first even when it trains at once: say
                        // "Queued" only when it waits behind another run.
                        let runs = this.store.runs.ready().map(|r| r.as_slice()).unwrap_or_default();
                        let wait = run.as_ref().filter(|r| waits_to_start(r, runs)).map(|r| {
                            match (r.queue_position, training_now(runs, &r.id)) {
                                (Some(p), _) if p > 1 => {
                                    format!("{}: it starts after the runs ahead of it.", queue_text(Some(p)))
                                }
                                (_, Some(a)) => format!("It starts when {} finishes.", a.name),
                                _ => "It starts when the current run finishes.".to_string(),
                            }
                        });
                        if let Some(on) = cloud {
                            // It waits on its instance, not in the queue.
                            this.toast(
                                Tone::Accent,
                                format!("Launching {name}"),
                                Some(format!("On {on}: the instance starts, then the data uploads.").into()),
                                cx,
                            );
                            this.refresh_instances(cx);
                        } else if let Some(wait) = wait {
                            // The run's page opens next, so the action is the queue it waits in.
                            this.toast_with(
                                Tone::Info,
                                format!("Queued {name}"),
                                Some(wait.into()),
                                Some((
                                    "View queue".into(),
                                    Rc::new(
                                        move |this: &mut Self, w: &mut Window, cx: &mut Context<Self>| {
                                            this.navigate(Route::Runs, w, cx);
                                            this.run_filter = RunFilter::Active;
                                        },
                                    ),
                                )),
                                cx,
                            );
                        } else {
                            this.toast(
                                Tone::Accent,
                                format!("Started {name}"),
                                Some("Live metrics stream into the dashboard.".into()),
                                cx,
                            );
                        }
                        // An earlier, dismissed submission leaves the form now open alone.
                        if mine {
                            // The next New run starts from these choices.
                            this.new_run_last = this.new_run.as_ref().map(|f| (f.arch, f.imgsz, f.epochs));
                            this.new_run = None;
                            this.overlay_return = None;
                            let prev = std::mem::replace(&mut this.route, Route::Run(id));
                            this.history.push(prev);
                        }
                    }
                    Err(e) => {
                        if mine && let Some(f) = this.new_run.as_mut() {
                            f.submitting = false;
                            // A refused compute choice shows on its control (Run on, GPU, Stop
                            // after); a GPU that went away or is out of stock gets fresh offers.
                            match crate::cloud::compute_error_field(&e).filter(|_| f.run_on.is_some()) {
                                Some(field) => {
                                    let conn = f.run_on.clone();
                                    f.compute_error = Some((field, e.message));
                                    if field == crate::cloud::ComputeField::Gpu
                                        && let Some(conn) = conn
                                    {
                                        this.ensure_offers(&conn, true, cx);
                                    }
                                }
                                None => f.error = Some(e),
                            }
                            // The error leads the form: scroll it into view.
                            this.dialog_scroll.set_offset(point(px(0.), px(0.)));
                        } else {
                            // Its form is gone: say so where it can be seen.
                            this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx);
                        }
                    }
                }
            },
        );
        cx.notify();
    }

    /// Why Mixed precision can't be used for a run on `device` (an entry of `DEVICES`), if it can't:
    /// the CPU, or the server's own answer (`capabilities.amp`, the message such a run would fail
    /// with). An older server without capabilities says nothing; the run's failure tells then.
    pub fn amp_blocked(&self, device: &str) -> Option<String> {
        if device == "cpu" {
            return Some("Off on the CPU: it needs a GPU with tensor cores".into());
        }
        let caps = self.store.capabilities.as_ref()?;
        // The tensor cores' own short reason when there is one ("off: <GPU> has no cooperative
        // matrices", "off (not requested: …)"); else the trainer's full message.
        (!caps.amp.available).then(|| match (&caps.tensor_cores.reason, &caps.amp.reason) {
            (Some(tc), _) => format!("Not available: tensor cores {tc}"),
            (None, Some(amp)) => amp.clone(),
            (None, None) => "Mixed precision isn't available on this server".into(),
        })
    }
}
