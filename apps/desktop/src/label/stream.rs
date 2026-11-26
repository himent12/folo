//! Live SSE event stream and periodic status polling for Agent Labeling.

use std::sync::mpsc::TryRecvError;

use folo_ui::Context;

use super::load::route_label;
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
                    self.upsert_label_job(*job);
                }
                LabelEvent::Image(img) => {
                    let selected_matches = self.label.selected_image.as_deref() == Some(img.id.as_str());
                    if selected_matches
                        && let Some(sel_box) = self.label.selected_box
                        && sel_box >= img.boxes.len()
                    {
                        self.label.selected_box = None;
                    }
                    self.upsert_label_image(&id, img);
                    self.reconcile_selected_label_image(&id);
                }
                LabelEvent::Removed(r) => {
                    if let Some(list) = self.label.images.get_mut(&id).and_then(|l| l.ready_mut()) {
                        list.retain(|i| i.id != r.id);
                    }
                    if self.label.selected_image.as_deref() == Some(r.id.as_str()) {
                        self.label.selected_image = None;
                        self.label.selected_box = None;
                    }
                    self.reconcile_selected_label_image(&id);
                }
                LabelEvent::Reload => {
                    reload = true;
                }
                LabelEvent::Log(line) => match self.label.logs.get_mut(&id) {
                    Some(Load::Ready(lines)) => {
                        lines.push(line);
                        if lines.len() > 500 {
                            let drop_n = lines.len() - 500;
                            lines.drain(..drop_n);
                        }
                    }
                    _ => {
                        self.label.logs.insert(id.clone(), Load::Ready(vec![line]));
                    }
                },
            }
        }

        if ended {
            self.label.events = None;
            reload = true;
        }
        if reload {
            self.reload_label_images(&id, cx);
            self.refresh_label_jobs(cx);
        }

        let first = self.label.sub_for.as_deref() != Some(id.as_str());
        if self.label.events.is_none()
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
        let jid = id.clone();
        self.fetch(
            cx,
            move |api| api.label_events(&jid),
            move |this, r, _| {
                if this.label.subscribing.as_deref() == Some(id.as_str()) {
                    this.label.subscribing = None;
                }
                if let Ok(sub) = r
                    && route_label(&this.route) == Some(id.as_str())
                {
                    this.label.events = Some((id, sub));
                }
            },
        );
    }
}
