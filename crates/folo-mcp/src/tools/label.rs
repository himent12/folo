//! The labeling tools: agents, jobs, images, runs, a view of an image with its boxes, box edits, review and the
//! dataset export. The operations are [`crate::label::Labels`]'s (shared with `folo label`); this file adds path
//! resolution, `--confine`, the blocking pool and the one-line summaries.

use base64::Engine as _;
use folo_api::{LabelDatasetRequest, LabelJobStatus, LabelOutcome, StartLabeling};
use rmcp::model::ContentBlock;

use super::results::{Output, fit_page, out, page_of, page_window, short};
use crate::errors::{Error, invalid_arguments};
use crate::label::{BoxEdit, Labels};
use crate::server::Inner;
use crate::types::*;

/// Longest text beside a view (one line per box).
const MAX_VIEW_TEXT: usize = 16_000;

impl Inner {
    /// Runs labeling calls on the blocking pool (they read files or make HTTP requests).
    async fn labels<T, F>(&self, f: F) -> Result<T, Error>
    where
        T: Send + 'static,
        F: FnOnce(&Labels) -> Result<T, Error> + Send + 'static,
    {
        let labels = self.labels.clone();
        tokio::task::spawn_blocking(move || f(&labels)).await.map_err(super::join)?
    }

    /// Input paths of images to add, resolved as every input path is (relative to `--data-dir`; with
    /// `--confine`, inside it), in the form the server takes.
    async fn label_paths(&self, paths: Vec<String>) -> Result<Vec<String>, Error> {
        let mut out = Vec::with_capacity(paths.len());
        for p in paths {
            out.push(folo_api::json::path(self.resolve(p).await?));
        }
        Ok(out)
    }

    /// With `--confine` in process: fails unless every image of job `id` is inside the data dir (a run and an
    /// export read them all, a view one of them). Through a server, it reads them itself.
    async fn confine_job(&self, id: &str) -> Result<(), Error> {
        if !self.opts.confine {
            return Ok(());
        }
        let (paths, id) = (self.paths.clone(), id.to_string());
        self.labels(move |l| match l.files(&id)? {
            Some(files) => paths.check_files(files.iter().map(|f| f.as_path())),
            None => Ok(()),
        })
        .await
    }

    pub(super) async fn list_label_agents(&self, _: NoArgs) -> Result<Output, Error> {
        let r = self.labels(|l| l.agent_list()).await?;
        let parts: Vec<String> = r
            .agents
            .iter()
            .map(|a| match (a.available, &a.default_model) {
                (true, model) => format!(
                    "{}: available{}, default {}/{}",
                    a.id.as_str(),
                    a.version.as_deref().map(|v| format!(" ({v})")).unwrap_or_default(),
                    model.as_deref().unwrap_or("its own model"),
                    a.default_effort.as_str()
                ),
                (false, _) => format!("{}: not installed", a.id.as_str()),
            })
            .collect();
        out(&r, parts.join("; "))
    }

    pub(super) async fn list_label_jobs(&self, a: PageArgs) -> Result<Output, Error> {
        let jobs = self.labels(|l| l.job_list()).await?;
        let page = page_of(jobs, a.cursor.as_deref(), a.limit)?;
        let names: Vec<String> = page
            .items
            .iter()
            .take(5)
            .map(|j| {
                format!("{} {} ({}/{} labeled)", j.id, short(&j.name, 40), j.counts.labeled, j.counts.images)
            })
            .collect();
        let summary = match page.total {
            0 => "no labeling jobs; create_label_job makes one".to_string(),
            n => format!("{n} labeling job(s): {}", names.join(", ")),
        };
        out(&page, summary)
    }

    pub(super) async fn create_label_job(&self, a: CreateLabelJobArgs) -> Result<Output, Error> {
        let paths = self.label_paths(a.paths).await?;
        let (name, classes, instructions) = (a.name, a.classes, a.instructions);
        let r = self.labels(move |l| l.create_job(name, classes, instructions, &paths)).await?;
        let summary =
            format!("{}: {} images ({} skipped); start_labeling labels them", r.job_id, r.images, r.skipped);
        out(&r, summary)
    }

    pub(super) async fn add_label_images(&self, a: AddLabelImagesArgs) -> Result<Output, Error> {
        if a.paths.is_empty() {
            return Err(invalid_arguments("paths must name at least one folder or image file"));
        }
        let paths = self.label_paths(a.paths).await?;
        let id = a.job_id;
        let r = self.labels(move |l| l.add_paths(&id, &paths)).await?;
        let summary =
            format!("added {} ({} skipped); {} has {} images", r.added, r.skipped, r.job_id, r.images);
        out(&r, summary)
    }

    pub(super) async fn start_labeling(&self, a: StartLabelingArgs) -> Result<Output, Error> {
        self.confine_job(&a.job_id).await?;
        let id = a.job_id;
        let req = StartLabeling {
            agent: a.agent,
            model: a.model,
            effort: a.effort,
            scope: a.scope,
            image_ids: a.image_ids,
            note: a.note,
            batch: a.batch,
            parallel: a.parallel,
        };
        let r = self.labels(move |l| l.start_run(&id, req)).await?;
        let summary = match &r.run {
            Some(run) => format!(
                "{} ({}, {}) labels {} images of {}; follow it with label_status",
                run.agent.as_str(),
                run.model.as_deref().unwrap_or("its default model"),
                run.effort.as_str(),
                run.total,
                r.job_id
            ),
            None => format!("{}: {}", r.job_id, status_name(r.status)),
        };
        out(&r, summary)
    }

    pub(super) async fn stop_labeling(&self, a: JobIdArgs) -> Result<Output, Error> {
        let id = a.job_id;
        let r = self.labels(move |l| l.stop_run(&id)).await?;
        let summary =
            format!("{}: {} (calls in flight are ended; done images stay)", r.job_id, status_name(r.status));
        out(&r, summary)
    }

    pub(super) async fn label_status(&self, a: LabelStatusArgs) -> Result<Output, Error> {
        if a.log_lines > 50 {
            return Err(invalid_arguments(format!("log_lines must be in 0..=50, got {}", a.log_lines)));
        }
        let id = a.job_id;
        let lines = a.log_lines as usize;
        let r = self.labels(move |l| l.status(&id, lines)).await?;
        let c = r.counts;
        let mut summary = format!(
            "{} {}: {}/{} labeled, {} failed, {} reviewed",
            r.id,
            status_name(r.status),
            c.labeled,
            c.images,
            c.failed,
            r.reviewed
        );
        if let Some(run) = &r.run {
            let state = match run.outcome {
                None => "running".to_string(),
                Some(LabelOutcome::Completed) => "completed".into(),
                Some(LabelOutcome::Stopped) => "stopped".into(),
                Some(LabelOutcome::Failed) => {
                    format!("failed: {}", short(run.error.as_deref().unwrap_or("see log"), 120))
                }
            };
            summary.push_str(&format!(
                "; run {} {}/{} ({} failed), {state}",
                run.agent.as_str(),
                run.done,
                run.total,
                run.failed
            ));
        }
        let u = r.usage;
        summary.push_str(&format!("; {} in / {} out tokens", u.input_tokens, u.output_tokens));
        if let Some(cost) = u.cost_usd {
            summary.push_str(&format!(", ${cost:.3}"));
        }
        out(&r, summary)
    }

    pub(super) async fn list_label_images(&self, a: ListLabelImagesArgs) -> Result<Output, Error> {
        let (offset, limit) = page_window(a.cursor.as_deref(), a.limit)?;
        let (id, status, reviewed, boxes) = (a.job_id, a.status, a.reviewed, a.include_boxes);
        let (items, total) =
            self.labels(move |l| l.image_page(&id, status, reviewed, boxes, offset, limit)).await?;
        let page = fit_page(items, offset, limit, total);
        let summary = format!("{} of {total} images (from {})", page.items.len(), offset + 1);
        out(&page, summary)
    }

    pub(super) async fn view_label_image(&self, a: ViewLabelImageArgs) -> Result<Output, Error> {
        if !(256..=1568).contains(&a.size) {
            return Err(invalid_arguments(format!("size must be in 256..=1568, got {}", a.size)));
        }
        self.confine_job(&a.job_id).await?;
        let (id, image_id, size) = (a.job_id, a.image_id, a.size);
        let (r, jpeg) = self.labels(move |l| l.view(&id, image_id.as_deref(), size)).await?;
        let text = match &r.image {
            Some(v) => short(&crate::label::view_text(&r.job_id, v, &r.classes, r.to_review), MAX_VIEW_TEXT),
            None => format!("{}: nothing left to review (every labeled image is reviewed)", r.job_id),
        };
        let mut o = out(&r, "")?;
        o.summary = text;
        if let Some(jpeg) = jpeg {
            let b64 = base64::engine::general_purpose::STANDARD.encode(&jpeg);
            o.links.push(ContentBlock::image(b64, "image/jpeg"));
        }
        Ok(o)
    }

    pub(super) async fn set_label_boxes(&self, a: SetLabelBoxesArgs) -> Result<Output, Error> {
        let edit = BoxEdit { boxes: a.boxes, remove: a.remove, relabel: a.relabel, add: a.add };
        let (id, image_id) = (a.job_id, a.image_id);
        let (v, to_review) = self
            .labels(move |l| {
                let v = l.set_boxes(&id, &image_id, &edit)?;
                Ok((v, l.to_review(&id)?))
            })
            .await?;
        let summary = format!(
            "image {}: {} boxes saved (by hand, reviewed); {to_review} left to review",
            v.id,
            v.boxes.len()
        );
        out(&v, summary)
    }

    pub(super) async fn review_label_images(&self, a: ReviewLabelImagesArgs) -> Result<Output, Error> {
        if a.image_ids.is_empty() || a.image_ids.len() > 500 {
            return Err(invalid_arguments("image_ids must name 1 to 500 images"));
        }
        let (id, ids, reviewed) = (a.job_id, a.image_ids, a.reviewed);
        let r = self.labels(move |l| l.review(&id, &ids, reviewed)).await?;
        let mut summary =
            format!("marked {} {}", r.updated, if reviewed { "reviewed" } else { "not reviewed" });
        if let Some(f) = r.failed.first() {
            summary.push_str(&format!(" ({} failed, e.g. {}: {})", r.failed.len(), f.id, f.code));
        }
        summary.push_str(&format!("; {} left to review", r.to_review));
        out(&r, summary)
    }

    pub(super) async fn export_label_dataset(&self, a: ExportLabelDatasetArgs) -> Result<Output, Error> {
        self.confine_job(&a.job_id).await?;
        let id = a.job_id;
        let req = LabelDatasetRequest { name: a.name, val_fraction: a.val_fraction, seed: a.seed };
        let r = self.labels(move |l| l.export(&id, req)).await?;
        let summary = format!(
            "dataset {} '{}': {} train, {} val images; train it with train {{ \"dataset_id\": \"{}\" }}",
            r.dataset_id,
            short(&r.name, 60),
            r.train,
            r.val,
            r.dataset_id
        );
        out(&r, summary)
    }
}

fn status_name(s: LabelJobStatus) -> &'static str {
    match s {
        LabelJobStatus::Idle => "idle",
        LabelJobStatus::Running => "running",
        LabelJobStatus::Stopping => "stopping",
    }
}
