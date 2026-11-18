//! The labeling tools: agents, jobs, images, runs, a view of an image with its boxes, box edits, review and the
//! dataset export. The operations are [`crate::label::Labels`]'s (shared with `folo label`); this file adds path
//! handling, `--confine`, the blocking pool, the hints in MCP words and the one-line summaries.

use base64::Engine as _;
use folo_api::{LabelDatasetRequest, LabelJobStatus, LabelOutcome, StartLabeling};
use http::StatusCode;
use rmcp::model::ContentBlock;

use super::results::{Output, fit_page, out, page_of, page_window, short};
use crate::errors::{Error, invalid_arguments};
use crate::label::hints::{self, Front};
use crate::label::{BoxEdit, Labels, looks_absolute};
use crate::server::Inner;
use crate::types::*;

/// Longest text beside a view (the box list is at most [`crate::label::VIEW_BOXES`] lines).
const MAX_VIEW_TEXT: usize = 16_000;

impl Inner {
    /// Runs labeling calls on the blocking pool (they read files or make HTTP requests); errors get MCP hints.
    async fn labels<T, F>(&self, f: F) -> Result<T, Error>
    where
        T: Send + 'static,
        F: FnOnce(&Labels) -> Result<T, Error> + Send + 'static,
    {
        let labels = self.labels.clone();
        let r = tokio::task::spawn_blocking(move || f(&labels)).await.map_err(super::join)?;
        r.map_err(|e| hints::decorate(e, Front::Mcp))
    }

    /// Input paths of images to add, in the form the server takes. In process they resolve as every input path
    /// does (relative to `--data-dir`; with `--confine`, inside it). Through a server they are paths on its
    /// machine, passed on as given, so they must be absolute.
    async fn label_paths(&self, paths: Vec<String>) -> Result<Vec<String>, Error> {
        if paths.is_empty() {
            // Nothing is read: an empty job is fine even with --confine through a server.
            return Ok(paths);
        }
        if self.labels.is_remote() {
            self.confine_reads("adding images")?;
            if let Some(p) = paths.iter().find(|p| !looks_absolute(p)) {
                let what = if p.starts_with('\\') { "has no drive letter" } else { "is relative" };
                return Err(invalid_arguments(format!(
                    "`{}` {what}: through `folo serve`, give the full path of the folder on the server's machine",
                    short(p, 80)
                )));
            }
            return Ok(paths);
        }
        let mut out = Vec::with_capacity(paths.len());
        for p in paths {
            out.push(folo_api::json::path(self.resolve(p).await?));
        }
        Ok(out)
    }

    /// With `--confine` through a server: refuses `doing` (a call that has the server read image files, which
    /// the confinement cannot check there).
    fn confine_reads(&self, doing: &str) -> Result<(), Error> {
        if self.opts.confine && self.labels.is_remote() {
            return Err(Error::new(
                StatusCode::NOT_IMPLEMENTED,
                "not_supported",
                format!("{doing} is not available with --confine through `folo serve` (it reads image files there)"),
            )
            .hint("run `folo mcp --confine` on the data dir itself (no server), or without --confine"));
        }
        Ok(())
    }

    /// With `--confine`: fails unless every image of job `id` is inside the data dir (in process; a job with
    /// one outside is neither shown nor changed).
    async fn confine_job(&self, id: &str) -> Result<(), Error> {
        if !self.opts.confine {
            return Ok(());
        }
        let (c, id) = (self.confinement.clone(), id.to_string());
        self.labels(move |l| c.label_job(l, &id)).await
    }

    pub(super) async fn list_label_jobs(&self, a: PageArgs) -> Result<Output, Error> {
        let c = self.confinement.clone();
        // With --confine, jobs with images outside the data dir are not listed.
        let jobs = self
            .labels(move |l| {
                let jobs = l.job_list()?;
                Ok(jobs.into_iter().filter(|j| c.label_job(l, &j.id).is_ok()).collect::<Vec<_>>())
            })
            .await?;
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
        if let Some(id) = a.job_id {
            if a.name.is_some() || a.classes.is_some() {
                return Err(invalid_arguments(
                    "name and classes are a new job's: leave them out with job_id",
                ));
            }
            if a.paths.is_empty() {
                return Err(invalid_arguments("paths must name at least one folder or image file"));
            }
            let paths = self.label_paths(a.paths).await?;
            self.confine_job(&id).await?;
            let r = self.labels(move |l| l.add_paths(&id, &paths)).await?;
            let summary =
                format!("added {} ({} skipped); {} has {} images", r.added, r.skipped, r.job_id, r.images);
            return out(&r, summary);
        }
        let (Some(name), Some(classes)) = (a.name, a.classes) else {
            return Err(invalid_arguments("a new job needs name and classes (or give job_id to add images)"));
        };
        let paths = self.label_paths(a.paths).await?;
        let instructions = a.instructions;
        let labels = self.labels.clone();
        let r = tokio::task::spawn_blocking(move || labels.create_job(name, classes, instructions, &paths))
            .await
            .map_err(super::join)?
            .map_err(|f| hints::create_failed(f, Front::Mcp))?;
        let summary =
            format!("{}: {} images ({} skipped); start_labeling labels them", r.job_id, r.images, r.skipped);
        out(&r, summary)
    }

    pub(super) async fn start_labeling(&self, a: StartLabelingArgs) -> Result<Output, Error> {
        self.confine_reads("labeling")?;
        self.confine_job(&a.job_id).await?;
        let (id, agent) = (a.job_id, a.agent);
        let effort = a.effort.map_or("default", |e| e.as_str());
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
        let summary = format!(
            "{} ({}, {effort} effort) labels {} images of {}; follow it with label_status",
            agent.as_str(),
            r.model.as_deref().unwrap_or("its default model"),
            r.total,
            r.job_id
        );
        out(&r, summary)
    }

    pub(super) async fn stop_labeling(&self, a: JobIdArgs) -> Result<Output, Error> {
        self.confine_job(&a.job_id).await?;
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
        self.confine_job(&a.job_id).await?;
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
                None => "running",
                Some(LabelOutcome::Completed) => "completed",
                Some(LabelOutcome::Stopped) => "stopped",
                Some(LabelOutcome::Failed) => "failed (see log)",
            };
            summary.push_str(&format!(
                "; run {} {}/{} ({} failed), {state}",
                run.agent.as_str(),
                run.done,
                run.total,
                run.failed
            ));
        }
        let u = &r.usage;
        summary.push_str(&format!("; {} in / {} out tokens", u.input_tokens, u.output_tokens));
        if let Some(cost) = u.cost_usd {
            summary.push_str(&format!(", ${cost:.3}"));
        }
        out(&r, summary)
    }

    pub(super) async fn list_label_images(&self, a: ListLabelImagesArgs) -> Result<Output, Error> {
        let (offset, limit) = page_window(a.cursor.as_deref(), a.limit)?;
        self.confine_job(&a.job_id).await?;
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
        if a.from == 0 {
            return Err(invalid_arguments("from is a box number, 1 or more"));
        }
        self.confine_reads("viewing an image")?;
        self.confine_job(&a.job_id).await?;
        let (id, image_id, size, from) = (a.job_id, a.image_id, a.size, a.from);
        let (r, jpeg) = self.labels(move |l| l.view(&id, image_id.as_deref(), size, from)).await?;
        let text = match &r.image {
            Some(v) => crate::label::view_text(&r.job_id, v, &r.classes, r.to_review, MAX_VIEW_TEXT),
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
        self.confine_job(&a.job_id).await?;
        let edit = BoxEdit {
            expect: a.expect,
            size: a.size,
            boxes: a.boxes,
            relabel: a.relabel,
            remove: a.remove,
            add: a.add,
        };
        let (id, image_id) = (a.job_id, a.image_id);
        let v = self.labels(move |l| l.set_boxes(&id, &image_id, &edit)).await?;
        let summary = format!(
            "image {}: {} boxes saved (by hand, reviewed); revision {}",
            v.id, v.num_boxes, v.revision
        );
        out(&v, summary)
    }

    pub(super) async fn review_label_images(&self, a: ReviewLabelImagesArgs) -> Result<Output, Error> {
        self.confine_job(&a.job_id).await?;
        let (id, ids, reviewed, expect) = (a.job_id, a.image_ids, a.reviewed, a.expect);
        let r = self.labels(move |l| l.review(&id, &ids, reviewed, &expect)).await?;
        let mut summary =
            format!("marked {} {}", r.updated, if reviewed { "reviewed" } else { "not reviewed" });
        if let Some(f) = r.failed.first() {
            summary.push_str(&format!(" ({} failed, e.g. {}: {})", r.failed_total, f.id, f.code));
        }
        if let Some(n) = r.to_review {
            summary.push_str(&format!("; {n} left to review"));
        }
        out(&r, summary)
    }

    pub(super) async fn export_label_dataset(&self, a: ExportLabelDatasetArgs) -> Result<Output, Error> {
        self.confine_reads("exporting a dataset")?;
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
