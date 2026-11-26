//! Loading and caching for Agent Labeling: jobs, agents, paginated images, logs, and decoded pixels.

use std::sync::Arc;

use folo_ui::{Context, RenderImage};

use super::state::{LabelImageFilter, LabelState, PendingImageEvent};
use crate::api::*;
use crate::app::{FoloApp, Route};
use crate::store::{ImgKey, ImgKind, ImgSlot, Load};

/// The job id of a labeling route, if the route is a job detail page.
pub fn route_label(route: &Route) -> Option<&str> {
    match route {
        Route::LabelJob(id) => Some(id),
        _ => None,
    }
}

/// Cache namespace for decoded labeling images inside `store.images_px`.
pub(super) fn label_img_namespace(job_id: &str) -> String {
    format!("label:{job_id}")
}

impl FoloApp {
    /// Whether the connected server advertises `agent_labeling`.
    pub fn label_on(&self) -> bool {
        self.store.features.iter().any(|f| f == FEATURE_AGENT_LABELING)
    }

    /// True once `/api/system` has answered without `agent_labeling`.
    pub fn label_unsupported(&self) -> bool {
        self.store.system.ready().is_some() && !self.label_on()
    }

    /// Surfacing `/api/system` failure while waiting to discover `agent_labeling`.
    pub fn label_blocked_error(&self) -> Option<&ApiError> {
        if self.label_on() || self.label_unsupported() {
            return None;
        }
        self.store.system.error()
    }

    /// Look up a labeling job by id from the loaded list.
    pub fn label_job_by_id(&self, id: &str) -> Option<&LabelJob> {
        self.label.jobs.ready()?.iter().find(|j| j.id == id)
    }

    /// Display name of a labeling job (for breadcrumbs and confirmations).
    pub fn label_job_name(&self, id: &str) -> String {
        self.label_job_by_id(id).map_or_else(|| id.to_string(), |j| j.name.clone())
    }

    /// Loads the labeling jobs list and available coding agents once each.
    pub fn ensure_label_jobs(&mut self, cx: &mut Context<Self>) {
        if !self.label_on() {
            return;
        }
        if self.label.jobs.needs_fetch() {
            self.label.jobs = Load::Loading;
            self.refresh_label_jobs(cx);
        }
        if self.label.agents.needs_fetch() {
            self.label.agents = Load::Loading;
            let lgen = self.label.epoch;
            self.fetch(
                cx,
                |api| api.label_agents(),
                move |this, r, _| {
                    if this.label.epoch != lgen {
                        return;
                    }
                    if let Ok(agents) = &r {
                        this.sync_default_label_agent(agents);
                    }
                    this.label.agents = r.into();
                },
            );
        }
    }

    /// Picks the first available agent if none is selected yet (or the selected one is unavailable).
    pub(crate) fn sync_default_label_agent(&mut self, agents: &[LabelAgent]) {
        let valid = self.label.agent_id.is_some_and(|id| agents.iter().any(|a| a.id == id && a.available));
        if !valid && let Some(first) = agents.iter().find(|a| a.available).or_else(|| agents.first()) {
            self.label.agent_id = Some(first.id);
            self.label.model_idx = 0;
            self.label.effort = first.default_effort;
            self.label.fast = false;
        }
    }

    /// Reloads the labeling jobs list in place without regressing jobs updated by newer SSE events.
    pub fn refresh_label_jobs(&mut self, cx: &mut Context<Self>) {
        if !self.label_on() {
            return;
        }
        let lgen = self.label.epoch;
        let seen_seqs = self.label.job_event_seq.clone();
        self.fetch(
            cx,
            |api| api.label_jobs(),
            move |this, r, _| {
                if this.label.epoch != lgen {
                    return;
                }
                match r {
                    Ok(mut list) => {
                        list.retain(|j| !this.label.removed.contains(&j.id));
                        if let Some(prev) = this.label.jobs.ready() {
                            for job in &mut list {
                                let before = seen_seqs.get(&job.id).copied().unwrap_or(0);
                                let now = this.label.job_event_seq.get(&job.id).copied().unwrap_or(0);
                                if now > before
                                    && let Some(newer) = prev.iter().find(|p| p.id == job.id)
                                {
                                    *job = newer.clone();
                                }
                            }
                        }
                        this.label.jobs = Load::Ready(list);
                    }
                    Err(e) => {
                        if this.label.jobs.ready().is_none() {
                            this.label.jobs = Load::Failed(e);
                        }
                    }
                }
            },
        );
    }

    /// Loads a job's images, logs, and agents.
    pub fn ensure_label_job(&mut self, id: &str, cx: &mut Context<Self>) {
        self.ensure_label_jobs(cx);
        if !self.label_on() {
            return;
        }
        if self.label.images.get(id).is_none_or(|l| l.needs_fetch()) {
            self.label.images.insert(id.to_string(), Load::Loading);
            self.reload_label_images(id, cx);
        }
        if self.label.logs.get(id).is_none_or(|l| l.needs_fetch()) {
            self.label.logs.insert(id.to_string(), Load::Loading);
            self.reload_label_logs(id, cx);
        }
    }

    /// Reloads a job's log history in place, replaying any live log lines received while in flight.
    pub fn reload_label_logs(&mut self, id: &str, cx: &mut Context<Self>) {
        if !self.label_on() {
            return;
        }
        self.label.logs_seq += 1;
        let seq = self.label.logs_seq;
        let lgen = self.label.epoch;
        self.label.logs_reqs.insert(id.to_string(), seq);
        self.label.pending_log_events.insert(id.to_string(), Vec::new());
        let jid = id.to_string();
        let key = id.to_string();
        self.fetch(
            cx,
            move |api| api.label_logs(&jid),
            move |this, r, _| {
                if this.label.epoch != lgen || this.label.logs_reqs.get(&key) != Some(&seq) {
                    return;
                }
                this.label.logs_reqs.remove(&key);
                let buffered = this.label.pending_log_events.remove(&key).unwrap_or_default();
                match r {
                    Ok(mut res) => {
                        for line in buffered {
                            let dup = res.lines.last().is_some_and(|l| {
                                l.t == line.t && l.level == line.level && l.message == line.message
                            });
                            if !dup {
                                res.lines.push(line);
                            }
                        }
                        if res.lines.len() > 500 {
                            let drop_n = res.lines.len() - 500;
                            res.lines.drain(..drop_n);
                        }
                        this.label.logs.insert(key, Load::Ready(res.lines));
                    }
                    Err(e) => {
                        if this.label.logs.get(&key).is_none_or(|l| l.ready().is_none()) {
                            this.label.logs.insert(key, Load::Failed(e));
                        }
                    }
                }
            },
        );
    }

    /// Loads every image of a labeling job (paging in chunks of 1,000, restarting if concurrent
    /// edits shift `total`, and replaying live SSE image events that arrived while paging).
    pub fn reload_label_images(&mut self, id: &str, cx: &mut Context<Self>) {
        if !self.label_on() {
            return;
        }
        self.label.images_seq += 1;
        let seq = self.label.images_seq;
        let lgen = self.label.epoch;
        self.label.images_reqs.insert(id.to_string(), seq);
        self.label.pending_image_events.insert(id.to_string(), Vec::new());
        let jid = id.to_string();
        let key = id.to_string();
        self.fetch(
            cx,
            move |api| {
                for attempt in 0..2 {
                    let mut all = Vec::new();
                    let mut offset = 0u32;
                    let mut expected_total: Option<u64> = None;
                    let mut shifted = false;
                    loop {
                        let page = api.label_images(&jid, offset, 1000)?;
                        match expected_total {
                            None => expected_total = Some(page.total),
                            Some(prev) if prev != page.total && attempt == 0 => {
                                shifted = true;
                                break;
                            }
                            _ => {}
                        }
                        let n = page.images.len();
                        all.extend(page.images);
                        offset = offset.saturating_add(n as u32);
                        if n == 0 || (all.len() as u64) >= page.total || offset >= 10_000 {
                            break;
                        }
                    }
                    if !shifted {
                        return Ok(all);
                    }
                }
                Ok(Vec::new())
            },
            move |this, r, _| {
                if this.label.epoch != lgen || this.label.images_reqs.get(&key) != Some(&seq) {
                    return;
                }
                this.label.images_reqs.remove(&key);
                let buffered = this.label.pending_image_events.remove(&key).unwrap_or_default();
                match r {
                    Ok(mut imgs) => {
                        for ev in buffered {
                            match ev {
                                PendingImageEvent::Updated(im) => {
                                    match imgs.iter_mut().find(|x| x.id == im.id) {
                                        Some(slot) => *slot = im,
                                        None => imgs.push(im),
                                    }
                                }
                                PendingImageEvent::Removed(rid) => imgs.retain(|x| x.id != rid),
                            }
                        }
                        let on_this_job = route_label(&this.route) == Some(key.as_str());
                        if on_this_job && let Some(sel) = this.label.selected_box.as_ref() {
                            let still_valid =
                                imgs.iter().find(|im| im.id == sel.image_id).is_some_and(|im| {
                                    sel.index < im.boxes.len()
                                        && label_box_revision(&im.boxes) == sel.revision
                                });
                            if !still_valid {
                                this.label.selected_box = None;
                                this.label.hover_box = None;
                            }
                        }
                        this.label.images.insert(key.clone(), Load::Ready(imgs));
                        if on_this_job {
                            this.reconcile_selected_label_image(&key);
                        }
                    }
                    Err(e) => {
                        if this.label.images.get(&key).is_none_or(|l| l.ready().is_none()) {
                            this.label.images.insert(key, Load::Failed(e));
                        }
                    }
                }
            },
        );
    }

    /// Retries a failed job images fetch.
    pub fn retry_label_job(&mut self, id: &str, cx: &mut Context<Self>) {
        self.label.images.remove(id);
        self.label.logs.remove(id);
        self.ensure_label_job(id, cx);
        self.refresh_label_jobs(cx);
        cx.notify();
    }

    /// Fetches and caches a labeling image's decoded pixels using the app's shared [`crate::store::ImageCache`].
    pub fn label_image_px(
        &mut self,
        job_id: &str,
        image_id: &str,
        kind: ImgKind,
        cx: &mut Context<Self>,
    ) -> Option<Arc<RenderImage>> {
        let key = ImgKey { dataset: label_img_namespace(job_id), image: image_id.to_string(), kind };
        match self.store.images_px.get(&key) {
            Some(ImgSlot::Ready(img)) => return Some(img.clone()),
            Some(_) => {
                self.store.image_queue.touch(&key);
                return None;
            }
            None => {}
        }
        if self.hold_loading || !self.store.images_px.begin(&key) {
            return None;
        }
        let thumb = (kind == ImgKind::Thumb).then_some(256);
        let (jid, iid) = (job_id.to_string(), image_id.to_string());
        self.load_cached_image(key, move |api| api.label_image(&jid, &iid, thumb), cx);
        None
    }

    /// Whether a labeling image failed to decode or fetch.
    pub fn label_image_failed(&self, job_id: &str, image_id: &str, kind: ImgKind) -> bool {
        let key = ImgKey { dataset: label_img_namespace(job_id), image: image_id.to_string(), kind };
        self.store.images_px.failed(&key)
    }

    /// Clears all labeling state (on backend switch or full retry), advancing `label.gen` so
    /// in-flight callbacks from before the reset cannot write into the fresh state.
    pub fn clear_label(&mut self) {
        let epoch = self.label.epoch + 1;
        let form_seq = self.label.form_seq;
        let images_seq = self.label.images_seq;
        let logs_seq = self.label.logs_seq;
        let mutating_seq = self.label.mutating_seq;
        let starting_seq = self.label.starting_seq;
        self.label = LabelState {
            epoch,
            form_seq,
            images_seq,
            logs_seq,
            mutating_seq,
            starting_seq,
            ..LabelState::default()
        };
    }

    /// Resets transient per-job UI state when navigating between routes.
    pub fn after_label_navigate(&mut self) {
        self.label.confirm_delete_job = None;
        self.label.confirm_delete_image = None;
        self.label.review_error = None;
        self.label.start_error = None;
        self.label.selected_box = None;
        self.label.hover_box = None;
        self.label.zoom = 1.0;
        self.label.pan = (0., 0.);
        self.label.scope = LabelScope::Unlabeled;
        if matches!(self.route, Route::LabelJobs) {
            self.label.list_stale = true;
            self.label.add_folder = None;
            self.label.export_ds = None;
        } else if let Route::LabelJob(id) = &self.route {
            self.label.new_job = None;
            if self.label.add_folder.as_ref().is_some_and(|f| f.job != *id) {
                self.label.add_folder = None;
            }
            if self.label.export_ds.as_ref().is_some_and(|f| f.job != *id) {
                self.label.export_ds = None;
            }
            self.label.img_filter = LabelImageFilter::All;
            self.label.class_filter = None;
            let jid = id.clone();
            self.reconcile_selected_label_image(&jid);
        } else {
            self.label.new_job = None;
            self.label.add_folder = None;
            self.label.export_ds = None;
        }
    }

    /// Upserts a job snapshot into `self.label.jobs` (ignoring jobs deleted in this session).
    pub(super) fn upsert_label_job(&mut self, job: LabelJob) {
        if self.label.removed.contains(&job.id) {
            return;
        }
        if let Some(list) = self.label.jobs.ready_mut() {
            match list.iter_mut().find(|j| j.id == job.id) {
                Some(slot) => *slot = job,
                None => list.insert(0, job),
            }
        }
    }

    /// Applies an authoritative SSE `LabelEvent::Job` snapshot and bumps its per-job event sequence
    /// so older HTTP responses cannot regress it.
    pub(super) fn upsert_label_job_from_event(&mut self, job: LabelJob) {
        if self.label.removed.contains(&job.id) {
            return;
        }
        *self.label.job_event_seq.entry(job.id.clone()).or_insert(0) += 1;
        self.upsert_label_job(job);
    }

    /// Clears `selected_box` if a live update changed the box array of the currently selected image.
    pub(super) fn invalidate_stale_box_selection(&mut self, job_id: &str, image: &LabelImage) {
        if route_label(&self.route) != Some(job_id) {
            return;
        }
        if let Some(sel) = self.label.selected_box.as_ref()
            && sel.image_id == image.id
            && (sel.index >= image.boxes.len() || label_box_revision(&image.boxes) != sel.revision)
        {
            self.label.selected_box = None;
            self.label.hover_box = None;
        }
    }

    /// Upserts an image snapshot into `self.label.images[job_id]`.
    pub(super) fn upsert_label_image(&mut self, job_id: &str, image: LabelImage) {
        if let Some(list) = self.label.images.get_mut(job_id).and_then(|l| l.ready_mut()) {
            match list.iter_mut().find(|i| i.id == image.id) {
                Some(slot) => *slot = image,
                None => list.push(image),
            }
        }
    }
}
