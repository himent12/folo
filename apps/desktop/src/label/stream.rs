//! Live SSE event stream and periodic status polling for Agent Labeling.

use std::sync::mpsc::TryRecvError;

use folo_ui::Context;

use super::load::route_label;
use super::state::PendingImageEvent;
use crate::api::*;
use crate::app::{FoloApp, Route};
use crate::store::Load;
use crate::ui::Tone;

impl FoloApp {
    /// Called from the app's 100 ms pump: applies live events for the open labeling job, keeps its
    /// SSE stream connected while the job page is shown, and polls the jobs list while on
    /// `Route::LabelJobs` so running jobs show live progress.
    pub fn label_pump(&mut self, tick: u64, cx: &mut Context<Self>) -> bool {
        if !self.label_on() {
            self.label.events = None;
            self.label.sub_for = None;
            return false;
        }

        // Poll job list every 5 s while viewing the list (especially when an agent run is active).
        if matches!(self.route, Route::LabelJobs)
            && tick.is_multiple_of(50)
            && self.label.jobs.ready().is_some()
        {
            self.refresh_label_jobs(cx);
        }

        let open = route_label(&self.route).map(str::to_string);
        if self.label.events.as_ref().is_some_and(|(id, _)| Some(id) != open.as_ref()) {
            self.label.events = None;
        }
        let Some(id) = open else {
            self.label.sub_for = None;
            return false;
        };

        let mut changed = false;
        let mut ended = false;
        let mut reload = false;
        loop {
            let ev = match self.label.events.as_ref().map(|(_, rx)| rx.try_recv()) {
                Some(Ok(ev)) => ev,
                Some(Err(TryRecvError::Empty)) | None => break,
                Some(Err(TryRecvError::Disconnected)) => {
                    ended = true;
                    break;
                }
            };
            changed = true;
            match ev {
                LabelEvent::Job(job) => {
                    let was_active =
                        self.label_job_by_id(&job.id).is_some_and(|prev| prev.status.is_active());
                    let finished = was_active && !job.status.is_active();
                    if finished && let Some(run) = &job.run {
                        match run.outcome {
                            Some(LabelOutcome::Completed) => {
                                self.toast(
                                    Tone::Success,
                                    format!("{} finished labeling", job.name),
                                    Some(format!("{} of {} labeled", run.done, run.total).into()),
                                    cx,
                                );
                            }
                            Some(LabelOutcome::Failed) => {
                                self.toast(
                                    Tone::Danger,
                                    format!("{} stopped on error", job.name),
                                    run.error.clone().map(Into::into),
                                    cx,
                                );
                            }
                            Some(LabelOutcome::Stopped) => {
                                self.toast(Tone::Neutral, format!("Stopped {}", job.name), None, cx);
                            }
                            None => {}
                        }
                    }
                    self.upsert_label_job_from_event(*job);
                }
                LabelEvent::Image(img) => {
                    *self.label.image_event_seq.entry((id.clone(), img.id.clone())).or_insert(0) += 1;
                    if self.label.images_reqs.contains_key(&id) {
                        self.label
                            .pending_image_events
                            .entry(id.clone())
                            .or_default()
                            .push(PendingImageEvent::Updated(img.clone()));
                    }
                    self.invalidate_stale_box_selection(&id, &img);
                    self.upsert_label_image(&id, img);
                    self.reconcile_selected_label_image(&id);
                }
                LabelEvent::Removed(r) => {
                    if self.label.images_reqs.contains_key(&id) {
                        self.label
                            .pending_image_events
                            .entry(id.clone())
                            .or_default()
                            .push(PendingImageEvent::Removed(r.id.clone()));
                    }
                    if let Some(list) = self.label.images.get_mut(&id).and_then(|l| l.ready_mut()) {
                        list.retain(|i| i.id != r.id);
                    }
                    if self.label.selected_image.as_deref() == Some(r.id.as_str()) {
                        self.label.selected_image = None;
                        self.label.selected_box = None;
                        self.label.hover_box = None;
                    } else if self.label.selected_box.as_ref().is_some_and(|s| s.image_id == r.id) {
                        self.label.selected_box = None;
                        self.label.hover_box = None;
                    }
                    self.reconcile_selected_label_image(&id);
                }
                LabelEvent::Reload => {
                    reload = true;
                }
                LabelEvent::Log(line) => {
                    if self.label.logs_reqs.contains_key(&id) {
                        self.label.pending_log_events.entry(id.clone()).or_default().push(line.clone());
                    }
                    if let Some(Load::Ready(lines)) = self.label.logs.get_mut(&id) {
                        lines.push(line);
                        if lines.len() > 500 {
                            let drop_n = lines.len() - 500;
                            lines.drain(..drop_n);
                        }
                    }
                }
            }
        }

        if ended {
            self.label.events = None;
            self.refresh_label_jobs(cx);
        }
        if reload {
            self.reload_label_images(&id, cx);
            self.reload_label_logs(&id, cx);
            self.refresh_label_jobs(cx);
        }

        let job_exists = !self.label.removed.contains(&id)
            && self.label.jobs.ready().is_none_or(|l| l.iter().any(|j| j.id == id));
        let first = self.label.sub_for.as_deref() != Some(id.as_str());
        if job_exists
            && self.label.events.is_none()
            && self.label.subscribing.is_none()
            && (first || tick.is_multiple_of(20))
        {
            self.subscribe_label(id, cx);
        }
        changed
    }

    fn subscribe_label(&mut self, id: String, cx: &mut Context<Self>) {
        self.label.subscribing = Some(id.clone());
        self.label.sub_for = Some(id.clone());
        let lgen = self.label.epoch;
        let jid = id.clone();
        self.fetch(
            cx,
            move |api| api.label_events(&jid),
            move |this, r, cx| {
                if this.label.epoch != lgen {
                    return;
                }
                if this.label.subscribing.as_deref() == Some(id.as_str()) {
                    this.label.subscribing = None;
                }
                match r {
                    Ok(sub) if route_label(&this.route) == Some(id.as_str()) => {
                        // Subscribe first, then snapshot images + logs + job summary so events arriving
                        // during the reload buffer and reconcile cleanly with zero gap.
                        this.label.events = Some((id.clone(), sub));
                        this.reload_label_images(&id, cx);
                        this.reload_label_logs(&id, cx);
                        this.refresh_label_jobs(cx);
                    }
                    Err(e) if e.code == "label_job_not_found" || e.status == Some(404) => {
                        this.label.jobs_seq += 1;
                        if let Some(list) = this.label.jobs.ready_mut() {
                            list.retain(|j| j.id != id);
                        }
                        this.label.images.remove(&id);
                        this.label.logs.remove(&id);
                        this.refresh_label_jobs(cx);
                    }
                    _ => {}
                }
            },
        );
    }
}
