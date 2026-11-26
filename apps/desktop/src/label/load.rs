//! Loading and caching for Agent Labeling: jobs, agents, paginated images, logs, and decoded pixels.

use std::sync::Arc;

use folo_ui::{Context, RenderImage};

use super::state::{LabelImageFilter, LabelState};
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
            self.fetch(
                cx,
                |api| api.label_agents(),
                |this, r, _| {
                    if let Ok(agents) = &r {
                        this.sync_default_label_agent(agents);
                    }
                    this.label.agents = r.into();
                },
            );
        }
    }

    /// Picks the first available agent if none is selected yet (or the selected one is unavailable).
    pub(super) fn sync_default_label_agent(&mut self, agents: &[LabelAgent]) {
        let valid = self.label.agent_id.is_some_and(|id| agents.iter().any(|a| a.id == id && a.available));
        if !valid && let Some(first) = agents.iter().find(|a| a.available).or_else(|| agents.first()) {
            self.label.agent_id = Some(first.id);
            self.label.model_idx = 0;
            self.label.effort = first.default_effort;
            self.label.fast = false;
        }
    }

    /// Reloads the labeling jobs list in place.
    pub fn refresh_label_jobs(&mut self, cx: &mut Context<Self>) {
        if !self.label_on() {
            return;
        }
        self.fetch(
            cx,
            |api| api.label_jobs(),
            |this, r, _| match r {
                Ok(mut list) => {
                    list.retain(|j| !this.label.removed.contains(&j.id));
                    this.label.jobs = Load::Ready(list);
                }
                Err(e) => {
                    if this.label.jobs.ready().is_none() {
                        this.label.jobs = Load::Failed(e);
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
            let jid = id.to_string();
            let key = id.to_string();
            self.fetch(
                cx,
                move |api| api.label_logs(&jid),
                move |this, r, _| {
                    this.label.logs.insert(key, r.map(|l| l.lines).into());
                },
            );
        }
    }

    /// Loads every image of a labeling job (paging through up to 10,000 images in chunks of 1,000).
    pub fn reload_label_images(&mut self, id: &str, cx: &mut Context<Self>) {
        if !self.label_on() {
            return;
        }
        self.label.images_seq += 1;
        let seq = self.label.images_seq;
        self.label.images_reqs.insert(id.to_string(), seq);
        let jid = id.to_string();
        let key = id.to_string();
        self.fetch(
            cx,
            move |api| {
                let mut all = Vec::new();
                let mut offset = 0u32;
                loop {
                    let page = api.label_images(&jid, offset, 1000)?;
                    let n = page.images.len();
                    all.extend(page.images);
                    offset = offset.saturating_add(n as u32);
                    if n == 0 || (all.len() as u64) >= page.total || offset >= 10_000 {
                        break;
                    }
                }
                Ok(all)
            },
            move |this, r, _| {
                if this.label.images_reqs.get(&key) != Some(&seq) {
                    return;
                }
                match r {
                    Ok(imgs) => {
                        this.label.images.insert(key.clone(), Load::Ready(imgs));
                        this.reconcile_selected_label_image(&key);
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

    /// Clears all labeling state (on backend switch or full retry).
    pub fn clear_label(&mut self) {
        let form_seq = self.label.form_seq;
        let images_seq = self.label.images_seq;
        self.label = LabelState { form_seq, images_seq, ..LabelState::default() };
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

    /// Upserts a job snapshot into `self.label.jobs` (keeping newest-first order).
    pub(super) fn upsert_label_job(&mut self, job: LabelJob) {
        if let Some(list) = self.label.jobs.ready_mut() {
            match list.iter_mut().find(|j| j.id == job.id) {
                Some(slot) => *slot = job,
                None => list.insert(0, job),
            }
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
