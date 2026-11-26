//! Inline form controllers for Agent Labeling: New job, Add folder, Export dataset, and Esc/Ctrl+Enter.

use folo_ui::{AppContext, Context, Focusable, Window};

use super::actions::{LABEL_VAL_FRACTIONS, validate_new_job};
use super::state::{AddFolderForm, ExportDatasetForm, NewLabelJobForm};
use crate::api::*;
use crate::app::{FoloApp, Route, act};
use crate::input::TextInput;
use crate::ui::Tone;

impl FoloApp {
    // ---------- New labeling job form ----------

    pub fn open_new_label_job(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.label_on() {
            return;
        }
        self.label.form_seq += 1;
        let seq = self.label.form_seq;
        let name = cx.new(|cx| TextInput::new("e.g. street-scenes", cx));
        let classes = cx.new(|cx| TextInput::new("car, person, bicycle", cx));
        let path = cx.new(|cx| TextInput::new("Optional folder path on the server", cx));
        let instructions =
            cx.new(|cx| TextInput::new("Optional instructions for the agent (what to include or skip)", cx));
        let fh = name.read(cx).focus_handle(cx);
        window.focus(&fh);
        self.label.new_job = Some(NewLabelJobForm {
            seq,
            name,
            classes,
            path,
            instructions,
            submitting: false,
            touched: false,
            error: None,
        });
        cx.notify();
    }

    pub fn close_new_label_job(&mut self, cx: &mut Context<Self>) {
        self.label.new_job = None;
        cx.notify();
    }

    pub fn browse_new_label_folder(&mut self, cx: &mut Context<Self>) {
        let Some(form) = self.label.new_job.as_ref() else { return };
        let path_input = form.path.clone();
        let seq = form.seq;
        let rx = cx.prompt_for_paths(folo_ui::PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Select image folder".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(p) = paths.into_iter().next()
            {
                let text = p.to_string_lossy().to_string();
                let _ = this.update(cx, |this, cx| {
                    if this.label.new_job.as_ref().is_some_and(|f| f.seq == seq) {
                        path_input.update(cx, |i, cx| i.set_text(text, cx));
                        cx.notify();
                    }
                });
            }
        })
        .detach();
    }

    pub fn submit_new_label_job(&mut self, cx: &mut Context<Self>) {
        let Some(form) = self.label.new_job.as_mut() else { return };
        if form.submitting {
            return;
        }
        form.touched = true;
        let name = form.name.read(cx).text().trim().to_string();
        let classes_raw = form.classes.read(cx).text().to_string();
        let path_raw = form.path.read(cx).text().trim().to_string();
        let instructions = form.instructions.read(cx).text().trim().to_string();
        let classes = match validate_new_job(&name, &classes_raw, &instructions) {
            Ok(c) => c,
            Err(msg) => {
                form.error = Some(ApiError::new("invalid_body", msg));
                cx.notify();
                return;
            }
        };
        form.submitting = true;
        form.error = None;
        let seq = form.seq;
        let lgen = self.label.epoch;
        let req =
            CreateLabelJob { name, classes, instructions, path: (!path_raw.is_empty()).then_some(path_raw) };
        cx.notify();
        self.fetch(
            cx,
            move |api| api.create_label_job(&req),
            move |this, r, cx| {
                if this.label.epoch != lgen {
                    return;
                }
                let mine = this.label.new_job.as_ref().is_some_and(|f| f.seq == seq);
                match r {
                    Ok(job) => {
                        let id = job.id.clone();
                        let title = job.name.clone();
                        let n_cls = job.classes.len();
                        let had_list = this.label.jobs.ready().is_some();
                        this.upsert_label_job(job);
                        if !had_list {
                            this.refresh_label_jobs(cx);
                        }
                        if mine {
                            this.label.new_job = None;
                            this.route_pending = Some(Route::LabelJob(id));
                        }
                        this.toast(
                            Tone::Accent,
                            format!("Created {title}"),
                            Some(format!("{n_cls} {}", if n_cls == 1 { "class" } else { "classes" }).into()),
                            cx,
                        );
                    }
                    Err(e) if mine => {
                        if let Some(f) = this.label.new_job.as_mut() {
                            f.submitting = false;
                            f.error = Some(e);
                        }
                    }
                    Err(e) => this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx),
                }
            },
        );
    }

    // ---------- Add folder to existing job ----------

    pub fn open_add_label_folder(&mut self, job_id: &str, window: &mut Window, cx: &mut Context<Self>) {
        self.label.export_ds = None;
        self.label.form_seq += 1;
        let seq = self.label.form_seq;
        let path = cx.new(|cx| TextInput::new("Folder or image path on the server", cx));
        let fh = path.read(cx).focus_handle(cx);
        window.focus(&fh);
        self.label.add_folder =
            Some(AddFolderForm { job: job_id.to_string(), seq, path, submitting: false, error: None });
        cx.notify();
    }

    pub fn browse_add_label_folder(&mut self, cx: &mut Context<Self>) {
        let Some(form) = self.label.add_folder.as_ref() else { return };
        let path_input = form.path.clone();
        let seq = form.seq;
        let rx = cx.prompt_for_paths(folo_ui::PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Select image folder".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(p) = paths.into_iter().next()
            {
                let text = p.to_string_lossy().to_string();
                let _ = this.update(cx, |this, cx| {
                    if this.label.add_folder.as_ref().is_some_and(|f| f.seq == seq) {
                        path_input.update(cx, |i, cx| i.set_text(text, cx));
                        cx.notify();
                    }
                });
            }
        })
        .detach();
    }

    pub fn submit_add_label_folder(&mut self, cx: &mut Context<Self>) {
        let Some(form) = self.label.add_folder.as_mut() else { return };
        if form.submitting {
            return;
        }
        let path = form.path.read(cx).text().trim().to_string();
        if path.is_empty() {
            form.error = Some(ApiError::new("invalid_body", "Enter a folder path."));
            cx.notify();
            return;
        }
        form.submitting = true;
        form.error = None;
        let job_id = form.job.clone();
        let seq = form.seq;
        let lgen = self.label.epoch;
        let seen_job_seq = self.label.job_event_seq.get(&job_id).copied().unwrap_or(0);
        let req = ImportLabelImages { path };
        let jid = job_id.clone();
        cx.notify();
        self.fetch(
            cx,
            move |api| api.import_label_images(&jid, &req),
            move |this, r, cx| {
                if this.label.epoch != lgen {
                    return;
                }
                let mine = this.label.add_folder.as_ref().is_some_and(|f| f.seq == seq);
                match r {
                    Ok(res) => {
                        if mine {
                            this.label.add_folder = None;
                        }
                        let added_n = res.added.len();
                        let skipped_n = res.skipped.len();
                        let cur_seq = this.label.job_event_seq.get(&job_id).copied().unwrap_or(0);
                        if cur_seq == seen_job_seq {
                            this.upsert_label_job(res.job);
                        }
                        this.reload_label_images(&job_id, cx);
                        let body =
                            if skipped_n > 0 { Some(format!("{skipped_n} skipped").into()) } else { None };
                        this.toast(
                            Tone::Success,
                            format!("Added {} {}", added_n, if added_n == 1 { "image" } else { "images" }),
                            body,
                            cx,
                        );
                    }
                    Err(e) if mine => {
                        if let Some(f) = this.label.add_folder.as_mut() {
                            f.submitting = false;
                            f.error = Some(e);
                        }
                    }
                    Err(e) => this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx),
                }
            },
        );
    }

    // ---------- Export / Register as Dataset ----------

    pub fn open_export_label_dataset(&mut self, job: &LabelJob, window: &mut Window, cx: &mut Context<Self>) {
        self.label.add_folder = None;
        self.label.form_seq += 1;
        let seq = self.label.form_seq;
        let default_name = job.name.clone();
        let name = cx.new(|cx| {
            let mut i = TextInput::new("Dataset name", cx);
            i.set_text(default_name, cx);
            i
        });
        let fh = name.read(cx).focus_handle(cx);
        window.focus(&fh);
        self.label.export_ds = Some(ExportDatasetForm {
            job: job.id.clone(),
            seq,
            name,
            val_idx: 1, // 20%
            submitting: false,
            error: None,
        });
        cx.notify();
    }

    pub fn submit_export_label_dataset(&mut self, cx: &mut Context<Self>) {
        let Some(form) = self.label.export_ds.as_mut() else { return };
        if form.submitting {
            return;
        }
        let name = form.name.read(cx).text().trim().to_string();
        if name.is_empty() {
            form.error = Some(ApiError::new("invalid_body", "Enter a dataset name."));
            cx.notify();
            return;
        }
        form.submitting = true;
        form.error = None;
        let job_id = form.job.clone();
        let seq = form.seq;
        let lgen = self.label.epoch;
        let val_fraction = LABEL_VAL_FRACTIONS[form.val_idx.min(LABEL_VAL_FRACTIONS.len() - 1)];
        let req = LabelDatasetRequest { name: Some(name), val_fraction, seed: 0 };
        let jid = job_id.clone();
        cx.notify();
        self.fetch(
            cx,
            move |api| api.create_label_dataset(&jid, &req),
            move |this, r, cx| {
                if this.label.epoch != lgen {
                    return;
                }
                let mine = this.label.export_ds.as_ref().is_some_and(|f| f.seq == seq);
                match r {
                    Ok(ds) => {
                        if mine {
                            this.label.export_ds = None;
                        }
                        let ds_id = ds.id.clone();
                        let ds_name = ds.name.clone();
                        let total = ds.splits.total();
                        if let Some(list) = this.store.datasets.ready_mut() {
                            match list.iter_mut().find(|d| d.id == ds.id) {
                                Some(slot) => *slot = ds,
                                None => list.push(ds),
                            }
                        }
                        if let Some(jobs) = this.label.jobs.ready_mut()
                            && let Some(job) = jobs.iter_mut().find(|j| j.id == job_id)
                        {
                            this.label.jobs_seq += 1;
                            job.dataset_id = Some(ds_id.clone());
                        }
                        this.refresh_label_jobs(cx);
                        this.toast_with(
                            Tone::Success,
                            format!("Exported dataset {ds_name}"),
                            Some(format!("{total} images registered and ready to train.").into()),
                            Some((
                                "Open dataset".into(),
                                act(move |this, w, cx| this.navigate(Route::Dataset(ds_id.clone()), w, cx)),
                            )),
                            cx,
                        );
                    }
                    Err(e) if mine => {
                        if let Some(f) = this.label.export_ds.as_mut() {
                            f.submitting = false;
                            f.error = Some(e);
                        }
                    }
                    Err(e) => this.toast(Tone::Danger, e.message, e.hint.map(Into::into), cx),
                }
            },
        );
    }

    /// Closes an inline labeling form or delete confirmation on `Esc`.
    pub fn dismiss_label_form(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        if self.label.new_job.take().is_some()
            || self.label.add_folder.take().is_some()
            || self.label.export_ds.take().is_some()
            || self.label.confirm_delete_job.take().is_some()
            || self.label.confirm_delete_image.take().is_some()
        {
            window.focus(&self.root_focus);
            cx.notify();
            true
        } else if self.label.selected_box.take().is_some() {
            cx.notify();
            true
        } else {
            false
        }
    }

    /// Submits the open inline labeling form on `Ctrl+Enter`.
    pub fn submit_label_form(&mut self, cx: &mut Context<Self>) -> bool {
        if self.label.new_job.is_some() {
            self.submit_new_label_job(cx);
            true
        } else if self.label.add_folder.is_some() {
            self.submit_add_label_folder(cx);
            true
        } else if self.label.export_ds.is_some() {
            self.submit_export_label_dataset(cx);
            true
        } else {
            false
        }
    }
}
