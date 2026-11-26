//! Interactive in-process Agent Labeling fixtures behind [`MockBench`] / [`super::mock::MockBackend`].

use std::sync::Arc;
use std::sync::mpsc::channel;
use std::time::Duration;

use crate::api::mock::DatasetFixture;
use crate::api::time;
use crate::api::*;
use crate::synth::{Scene, hash};

use super::MockBench;

/// One procedural image inside a mock labeling job.
#[derive(Clone)]
pub(super) struct MockLabelImage {
    pub image: LabelImage,
    pub seed: u64,
    pub classes: u32,
}

impl MockLabelImage {
    pub fn scene(&self) -> Scene {
        Scene::generate(self.seed, self.classes.max(1), 8, false)
    }
}

/// Builds deterministic procedural images for a seeded or imported mock labeling job.
pub(super) fn seed_job_images(
    job_id: &str,
    classes: usize,
    labeled: u64,
    pending: u64,
    failed: u64,
) -> (Vec<MockLabelImage>, LabelCounts, Vec<LogLine>) {
    let seed = hash(job_id);
    let total = labeled + pending + failed;
    let mut out = Vec::with_capacity(total as usize);
    let mut boxes_total = 0u64;
    for i in 0..total {
        let s = seed ^ i.wrapping_mul(0x9E37);
        let scene = Scene::generate(s, classes.max(1) as u32, 8, false);
        let (status, boxes, source, reviewed, error) = if i < labeled {
            let b = scene.boxes(640, 480);
            boxes_total += b.len() as u64;
            let rev = i < labeled / 3;
            (LabelImageStatus::Labeled, b, Some(LabelSource::Agent), rev, None)
        } else if i < labeled + pending {
            (LabelImageStatus::Pending, Vec::new(), None, false, None)
        } else {
            (
                LabelImageStatus::Failed,
                Vec::new(),
                None,
                false,
                Some("Agent reply dropped: no valid JSON object found.".into()),
            )
        };
        out.push(MockLabelImage {
            image: LabelImage {
                id: format!("li_{:04}", i + 1),
                file: format!("frame_{:04}.jpg", i + 1),
                width: 640,
                height: 480,
                status,
                boxes,
                source,
                error,
                reviewed,
            },
            seed: s,
            classes: classes.max(1) as u32,
        });
    }
    let counts = LabelCounts { images: total, pending, labeled, failed, boxes: boxes_total };
    let logs = vec![
        LogLine {
            t: time::format(crate::api::mock::MOCK_NOW - 3600),
            level: LogLevel::Info,
            message: format!("job {job_id}: loaded {total} images ({classes} classes)"),
        },
        LogLine {
            t: time::format(crate::api::mock::MOCK_NOW - 1800),
            level: LogLevel::Info,
            message: format!("claude/sonnet: labeled {labeled} images ({boxes_total} boxes)"),
        },
    ];
    (out, counts, logs)
}

fn recount(images: &[MockLabelImage]) -> LabelCounts {
    let mut counts = LabelCounts { images: images.len() as u64, ..Default::default() };
    for m in images {
        match m.image.status {
            LabelImageStatus::Pending | LabelImageStatus::Labeling => counts.pending += 1,
            LabelImageStatus::Labeled => {
                counts.labeled += 1;
                counts.boxes += m.image.boxes.len() as u64;
            }
            LabelImageStatus::Failed => counts.failed += 1,
        }
    }
    counts
}

fn find_job<'a>(jobs: &'a [LabelJob], id: &str) -> ApiResult<&'a LabelJob> {
    jobs.iter()
        .find(|j| j.id == id)
        .ok_or_else(|| ApiError::new("label_job_not_found", format!("No labeling job {id}")))
}

fn find_job_mut<'a>(jobs: &'a mut [LabelJob], id: &str) -> ApiResult<&'a mut LabelJob> {
    jobs.iter_mut()
        .find(|j| j.id == id)
        .ok_or_else(|| ApiError::new("label_job_not_found", format!("No labeling job {id}")))
}

fn validate_job_name(name: &str) -> ApiResult<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err(ApiError::new("invalid_body", "Job name is required."));
    }
    if trimmed.chars().count() > 100 || trimmed.contains(['\n', '\r']) {
        return Err(ApiError::new("invalid_body", "Job name must be one line of at most 100 characters."));
    }
    Ok(trimmed.to_string())
}

fn validate_classes(classes: &[String]) -> ApiResult<Vec<String>> {
    if classes.is_empty() || classes.len() > 80 {
        return Err(ApiError::new("invalid_body", "Provide between 1 and 80 classes."));
    }
    let mut out = Vec::with_capacity(classes.len());
    let mut seen = std::collections::HashSet::new();
    for raw in classes {
        let c = raw.trim();
        if c.is_empty() || c.chars().count() > 64 || c.contains([',', ':', '\n', '\r']) {
            return Err(ApiError::new("invalid_body", format!("Invalid class name `{c}`.")));
        }
        if !seen.insert(c.to_lowercase()) {
            return Err(ApiError::new("invalid_body", format!("Duplicate class `{c}`.")));
        }
        out.push(c.to_string());
    }
    Ok(out)
}

impl MockBench {
    fn label_job(&self, id: &str, datasets: &[DatasetFixture], models: &[Model]) -> ApiResult<LabelJob> {
        self.with(datasets, models, |d, _, _| Ok(find_job(&d.jobs, id)?.clone()))
    }

    pub fn create_label_job(
        &self,
        req: &CreateLabelJob,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<LabelJob> {
        let name = validate_job_name(&req.name)?;
        let classes = validate_classes(&req.classes)?;
        if req.instructions.chars().count() > 4000 {
            return Err(ApiError::new("invalid_body", "Instructions must be at most 4000 characters."));
        }
        if let Some(p) = req.path.as_deref()
            && p.trim().is_empty()
        {
            return Err(ApiError::new("invalid_body", "Folder path cannot be blank."));
        }
        self.with(datasets, models, |d, _, now| {
            d.seq += 1;
            let id = format!("lb_{:04x}", (hash(&name) ^ d.seq) & 0xFFFF);
            let pending = if req.path.is_some() { 12 } else { 0 };
            let (imgs, counts, logs) = seed_job_images(&id, classes.len(), 0, pending, 0);
            let job = LabelJob {
                id: id.clone(),
                name,
                classes,
                instructions: req.instructions.trim().to_string(),
                created: time::format(now),
                status: LabelJobStatus::Idle,
                counts,
                run: None,
                usage: LabelUsage::default(),
                dataset_id: None,
            };
            d.jobs.insert(0, job.clone());
            d.label_images.insert(id.clone(), imgs);
            d.label_logs.insert(id, logs);
            Ok(job)
        })
    }

    pub fn delete_label_job(&self, id: &str, datasets: &[DatasetFixture], models: &[Model]) -> ApiResult<()> {
        self.with(datasets, models, |d, _, _| {
            let idx = d
                .jobs
                .iter()
                .position(|j| j.id == id)
                .ok_or_else(|| ApiError::new("label_job_not_found", format!("No labeling job {id}")))?;
            if d.jobs[idx].status.is_active() {
                return Err(ApiError::new("label_job_active", "Stop the active agent run first."));
            }
            d.jobs.remove(idx);
            d.label_images.remove(id);
            d.label_logs.remove(id);
            Ok(())
        })
    }

    pub fn import_label_images(
        &self,
        id: &str,
        req: &ImportLabelImages,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<LabelImagesAdded> {
        let path = req.path.trim().trim_matches('"').trim_matches('\'');
        if path.is_empty() {
            return Err(ApiError::new("invalid_body", "Folder path is required."));
        }
        self.with(datasets, models, |d, _, now| {
            let classes = find_job(&d.jobs, id)?.classes.len().max(1) as u32;
            let imgs = d.label_images.entry(id.to_string()).or_default();
            let start = imgs.len() as u64;
            let mut added = Vec::new();
            for k in 0..8u64 {
                let n = start + k + 1;
                let s = hash(&format!("{id}:{path}:{n}"));
                let im = LabelImage {
                    id: format!("li_{n:04}"),
                    file: format!("import_{n:04}.jpg"),
                    width: 640,
                    height: 480,
                    status: LabelImageStatus::Pending,
                    boxes: Vec::new(),
                    source: None,
                    error: None,
                    reviewed: false,
                };
                added.push(im.clone());
                imgs.push(MockLabelImage { image: im, seed: s, classes });
            }
            let counts = recount(imgs);
            let job = find_job_mut(&mut d.jobs, id)?;
            job.counts = counts;
            let snap = job.clone();
            d.label_logs.entry(id.to_string()).or_default().push(LogLine {
                t: time::format(now),
                level: LogLevel::Info,
                message: format!("added {} images from {path}", added.len()),
            });
            Ok(LabelImagesAdded { added, skipped: Vec::new(), job: snap })
        })
    }

    pub fn label_images(
        &self,
        id: &str,
        offset: u32,
        limit: u32,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<LabelImagePage> {
        self.with(datasets, models, |d, _, _| {
            let _ = find_job(&d.jobs, id)?;
            let imgs = d.label_images.get(id).map(Vec::as_slice).unwrap_or(&[]);
            let total = imgs.len() as u64;
            let lim = if limit == 0 { 100 } else { limit.min(1000) } as usize;
            let images = imgs.iter().skip(offset as usize).take(lim).map(|m| m.image.clone()).collect();
            Ok(LabelImagePage { total, images })
        })
    }

    pub fn label_image_pixels(
        &self,
        job_id: &str,
        image_id: &str,
        thumb: Option<u32>,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<ImageData> {
        let src = self.with(datasets, models, |d, _, _| {
            let _ = find_job(&d.jobs, job_id)?;
            d.label_images
                .get(job_id)
                .and_then(|imgs| imgs.iter().find(|m| m.image.id == image_id))
                .cloned()
                .ok_or_else(|| ApiError::new("image_not_found", format!("No image {image_id}")))
        })?;
        let (mut w, mut h) = (src.image.width, src.image.height);
        if let Some(t) = thumb {
            let s = t as f32 / w.max(h) as f32;
            if s < 1.0 {
                w = ((w as f32 * s).round() as u32).max(1);
                h = ((h as f32 * s).round() as u32).max(1);
            }
        }
        let pixels = src.scene().render(w, h);
        Ok(ImageData::Rgba { width: w, height: h, pixels })
    }

    pub fn update_label_image(
        &self,
        job_id: &str,
        image_id: &str,
        req: &UpdateLabelImage,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<LabelImage> {
        if req.boxes.is_none() && req.reviewed.is_none() {
            return Err(ApiError::new("invalid_body", "Provide boxes or reviewed."));
        }
        self.with(datasets, models, |d, _, _| {
            let num_classes = find_job(&d.jobs, job_id)?.classes.len() as u32;
            let imgs = d
                .label_images
                .get_mut(job_id)
                .ok_or_else(|| ApiError::new("image_not_found", format!("No image {image_id}")))?;
            let item = imgs
                .iter_mut()
                .find(|m| m.image.id == image_id)
                .ok_or_else(|| ApiError::new("image_not_found", format!("No image {image_id}")))?;
            if item.image.status == LabelImageStatus::Labeling {
                return Err(ApiError::new("label_job_active", "The agent is labeling this image right now."));
            }
            if let Some(exp) = &req.expect
                && label_box_revision(&item.image.boxes) != *exp
            {
                return Err(ApiError {
                    status: Some(409),
                    code: "label_boxes_changed".into(),
                    message: "This image's boxes changed since you opened it.".into(),
                    hint: Some("Reloaded the latest boxes from the server.".into()),
                });
            }
            if let Some(boxes) = &req.boxes {
                for b in boxes {
                    if b.class >= num_classes {
                        return Err(ApiError::new("invalid_body", "Box class out of range."));
                    }
                }
                item.image.boxes = boxes.clone();
                item.image.status = LabelImageStatus::Labeled;
                item.image.source = Some(LabelSource::Manual);
                item.image.error = None;
                item.image.reviewed = req.reviewed.unwrap_or(true);
            } else if let Some(rev) = req.reviewed {
                if item.image.status != LabelImageStatus::Labeled {
                    return Err(ApiError::new(
                        "invalid_body",
                        "Only a labeled image can be marked reviewed.",
                    ));
                }
                item.image.reviewed = rev;
            }
            let updated = item.image.clone();
            let counts = recount(imgs);
            if let Some(job) = d.jobs.iter_mut().find(|j| j.id == job_id) {
                job.counts = counts;
            }
            Ok(updated)
        })
    }

    pub fn delete_label_image(
        &self,
        job_id: &str,
        image_id: &str,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<()> {
        self.with(datasets, models, |d, _, _| {
            let _ = find_job(&d.jobs, job_id)?;
            let imgs = d
                .label_images
                .get_mut(job_id)
                .ok_or_else(|| ApiError::new("image_not_found", format!("No image {image_id}")))?;
            let pos = imgs
                .iter()
                .position(|m| m.image.id == image_id)
                .ok_or_else(|| ApiError::new("image_not_found", format!("No image {image_id}")))?;
            if imgs[pos].image.status == LabelImageStatus::Labeling {
                return Err(ApiError::new("label_job_active", "The agent is labeling this image right now."));
            }
            imgs.remove(pos);
            let counts = recount(imgs);
            if let Some(job) = d.jobs.iter_mut().find(|j| j.id == job_id) {
                job.counts = counts;
            }
            Ok(())
        })
    }

    pub fn start_labeling(
        &self,
        id: &str,
        req: &StartLabeling,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<LabelJob> {
        self.with(datasets, models, |d, _, now| {
            let agent = d
                .agents
                .iter()
                .find(|a| a.id == req.agent)
                .cloned()
                .ok_or_else(|| ApiError::new("invalid_body", "Unknown agent."))?;
            if !agent.available {
                return Err(ApiError::new("not_supported", format!("{} is not installed.", agent.name)));
            }
            let job_active = find_job(&d.jobs, id)?.status.is_active();
            if job_active {
                return Err(ApiError::new("label_job_active", "An agent run is already active."));
            }
            let imgs = d.label_images.entry(id.to_string()).or_default();
            let targets: Vec<usize> = if let Some(ids) = &req.image_ids {
                imgs.iter().enumerate().filter(|(_, m)| ids.contains(&m.image.id)).map(|(i, _)| i).collect()
            } else {
                imgs.iter()
                    .enumerate()
                    .filter(|(_, m)| match req.scope {
                        LabelScope::Unlabeled => {
                            matches!(m.image.status, LabelImageStatus::Pending | LabelImageStatus::Failed)
                        }
                        LabelScope::All => true,
                    })
                    .map(|(i, _)| i)
                    .collect()
            };
            if targets.is_empty() {
                return Err(ApiError::new("invalid_body", "Nothing to label in this scope."));
            }
            // In demo mode, label the target images deterministically so the workflow completes cleanly.
            let mut labeled_now = 0u64;
            for idx in targets {
                let m = &mut imgs[idx];
                m.image.boxes = m.scene().boxes(m.image.width, m.image.height);
                m.image.status = LabelImageStatus::Labeled;
                m.image.source = Some(LabelSource::Agent);
                m.image.error = None;
                m.image.reviewed = false;
                labeled_now += 1;
            }
            let counts = recount(imgs);
            let usage = LabelUsage {
                calls: labeled_now.div_ceil(4),
                input_tokens: labeled_now * 620,
                cached_input_tokens: 0,
                output_tokens: labeled_now * 95,
                cost_usd: (req.agent == LabelAgentId::Claude).then_some(labeled_now as f64 * 0.0007),
            };
            let effort = req.effort.unwrap_or(agent.default_effort);
            let model = req.model.clone().or(agent.default_model);
            let run = LabelRun {
                id: format!("lr_{:04x}", (hash(id) ^ d.seq) & 0xFFFF),
                agent: req.agent,
                model: model.clone(),
                effort,
                fast: req.fast,
                started: time::format(now),
                finished: Some(time::format(now)),
                total: labeled_now,
                done: labeled_now,
                failed: 0,
                outcome: Some(LabelOutcome::Completed),
                error: None,
                hint: None,
                usage,
            };
            d.seq += 1;
            let job = find_job_mut(&mut d.jobs, id)?;
            job.counts = counts;
            job.usage.add(&usage);
            job.run = Some(run);
            job.status = LabelJobStatus::Idle;
            let snap = job.clone();
            d.label_logs.entry(id.to_string()).or_default().push(LogLine {
                t: time::format(now),
                level: LogLevel::Info,
                message: format!(
                    "{}/{}: labeled {} images",
                    req.agent.as_str(),
                    model.as_deref().unwrap_or("default"),
                    labeled_now
                ),
            });
            Ok(snap)
        })
    }

    pub fn stop_labeling(
        &self,
        id: &str,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<LabelJob> {
        self.with(datasets, models, |d, _, now| {
            let job = find_job_mut(&mut d.jobs, id)?;
            if !job.status.is_active() {
                return Err(ApiError::new("label_job_not_active", "No agent run is active."));
            }
            job.status = LabelJobStatus::Idle;
            if let Some(r) = job.run.as_mut() {
                r.outcome = Some(LabelOutcome::Stopped);
                r.finished = Some(time::format(now));
            }
            Ok(job.clone())
        })
    }

    pub fn label_logs(&self, id: &str, datasets: &[DatasetFixture], models: &[Model]) -> ApiResult<RunLogs> {
        self.with(datasets, models, |d, _, _| {
            let _ = find_job(&d.jobs, id)?;
            let lines = d.label_logs.get(id).cloned().unwrap_or_default();
            let total = lines.len() as u64;
            Ok(RunLogs { lines, total })
        })
    }

    pub fn label_events(
        this: Arc<Self>,
        id: &str,
        datasets: Vec<DatasetFixture>,
        models: Vec<Model>,
    ) -> ApiResult<LabelSubscription> {
        let initial = this.label_job(id, &datasets, &models)?;
        let (tx, rx) = channel();
        let (sub, cancel) = LabelSubscription::new(rx);
        let _ = tx.send(LabelEvent::Job(Box::new(initial.clone())));
        if this.frozen {
            return Ok(sub);
        }
        let jid = id.to_string();
        let mut last_job = initial;
        let mut last_log_len = this
            .with(&datasets, &models, |d, _, _| Ok(d.label_logs.get(&jid).map_or(0, Vec::len)))
            .unwrap_or(0);
        std::thread::spawn(move || {
            while !cancel.load(std::sync::atomic::Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(500));
                if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                let snap = this.with(&datasets, &models, |d, _, _| {
                    let j = find_job(&d.jobs, &jid)?.clone();
                    let logs = d.label_logs.get(&jid).cloned().unwrap_or_default();
                    Ok((j, logs))
                });
                let Ok((job, logs)) = snap else { break };
                if job != last_job {
                    last_job = job.clone();
                    if tx.send(LabelEvent::Job(Box::new(job))).is_err() {
                        return;
                    }
                }
                if logs.len() > last_log_len {
                    for line in &logs[last_log_len..] {
                        if tx.send(LabelEvent::Log(line.clone())).is_err() {
                            return;
                        }
                    }
                    last_log_len = logs.len();
                }
            }
        });
        Ok(sub)
    }

    pub fn create_label_dataset(
        &self,
        id: &str,
        req: &LabelDatasetRequest,
        datasets: &[DatasetFixture],
        models: &[Model],
    ) -> ApiResult<Dataset> {
        if !(0.0..=0.5).contains(&req.val_fraction) || !req.val_fraction.is_finite() {
            return Err(ApiError::new("invalid_body", "val_fraction must be between 0 and 0.5."));
        }
        self.with(datasets, models, |d, _, now| {
            let job = find_job_mut(&mut d.jobs, id)?;
            if job.status.is_active() {
                return Err(ApiError::new("label_job_active", "Wait for the agent run to finish."));
            }
            if job.counts.labeled == 0 {
                return Err(ApiError::new("invalid_dataset", "Label at least one image first."));
            }
            let name =
                req.name.as_deref().map(str::trim).filter(|s| !s.is_empty()).unwrap_or(&job.name).to_string();
            let labeled = job.counts.labeled;
            let val = ((labeled as f32 * req.val_fraction).round() as u64).min(labeled.saturating_sub(1));
            let train = labeled - val;
            d.seq += 1;
            let ds_id = format!("ds_{:04x}", (hash(&name) ^ d.seq) & 0xFFFF);
            job.dataset_id = Some(ds_id.clone());
            let ds = Dataset {
                id: ds_id,
                name: name.clone(),
                path: format!("D:/folo/datasets/{name}"),
                format: DatasetFormat::Yolo,
                classes: job.classes.clone(),
                splits: Splits { train, val, test: 0 },
                created: time::format(now),
                stats: None,
                revision: None,
            };
            d.label_logs.entry(id.to_string()).or_default().push(LogLine {
                t: time::format(now),
                level: LogLevel::Info,
                message: format!("exported dataset {name} ({train} train, {val} val)"),
            });
            Ok(ds)
        })
    }
}

pub(super) fn seed_default_label_data(
    jobs: &mut [LabelJob],
) -> (std::collections::HashMap<String, Vec<MockLabelImage>>, std::collections::HashMap<String, Vec<LogLine>>)
{
    let mut images_map = std::collections::HashMap::new();
    let mut logs_map = std::collections::HashMap::new();
    for job in jobs.iter_mut() {
        let labeled = job.counts.labeled;
        let pending = job.counts.pending.saturating_sub(2);
        let failed = job.counts.pending - pending;
        let (imgs, counts, logs) = seed_job_images(&job.id, job.classes.len(), labeled, pending, failed);
        job.counts = counts;
        images_map.insert(job.id.clone(), imgs);
        logs_map.insert(job.id.clone(), logs);
    }
    (images_map, logs_map)
}
