//! The labeling operations the MCP tools and `folo label` share, built on the calls of [`Labels`]: each answers
//! with a result of [`crate::types::label`]. Their errors carry no hint: each front end adds its own
//! ([`super::hints`]), naming its tools or commands.

use std::collections::{BTreeSet, HashMap, HashSet};

use folo_api::{
    BBox, CreateLabelJob, LabelDatasetRequest, LabelImage, LabelImageStatus, ReviewLabelImages,
    StartLabeling, UpdateLabelImage, label_box_revision,
};
use folo_server::label::LabelImageQuery;
use http::StatusCode;

use super::Labels;
use super::compact::{self, frame, from_view, to_view};
use crate::errors::{Error, invalid_arguments};
use crate::types::label::*;

/// Most images `list_label_images` asks the server for at once.
const PAGE: u64 = 1000;
/// Most images one review marks.
pub const MAX_REVIEW: usize = 500;
/// Longest image id a call may name.
const MAX_IMAGE_ID: usize = 128;

/// `set_label_boxes`: all the boxes, or edits of the current ones by their 1-based number, in pixels of a view.
#[derive(Clone, Debug, Default)]
pub struct BoxEdit {
    /// The revision of the boxes the edit was made on; required.
    pub expect: Option<String>,
    /// The frame of the coordinates (the `size` of the view they were read on); required with coordinates.
    pub size: Option<[u32; 2]>,
    /// All the boxes, replacing the image's.
    pub boxes: Option<Vec<PxBox>>,
    /// `[number, class]`: boxes to give another class.
    pub relabel: Vec<[u32; 2]>,
    /// Numbers of boxes to delete.
    pub remove: Vec<u32>,
    /// Boxes to append.
    pub add: Vec<PxBox>,
}

impl BoxEdit {
    fn edits(&self) -> bool {
        !(self.remove.is_empty() && self.relabel.is_empty() && self.add.is_empty())
    }

    /// Whether the edit has coordinates, which mean nothing without the frame they were read in.
    fn coordinates(&self) -> bool {
        !self.add.is_empty() || self.boxes.as_ref().is_some_and(|b| !b.is_empty())
    }
}

/// `create_label_job` that failed: the error, and the job it made anyway when an earlier path was added (so the
/// caller can add the rest to it).
#[derive(Debug)]
pub struct CreateFailed {
    /// What went wrong.
    pub error: Error,
    /// The job that was made with the paths before the one that failed.
    pub made: Option<String>,
}

/// `409 label_boxes_changed`: the image's boxes are not those of revision `expect`.
pub fn boxes_changed(image_id: &str, expect: &str, now: &str) -> Error {
    Error::new(
        StatusCode::CONFLICT,
        "label_boxes_changed",
        format!("image '{image_id}' has other boxes than revision {expect} (now {now})"),
    )
}

/// An `invalid_arguments` error unless `id` can be an image id: not empty, short, and no `,` (the server's list
/// separator).
fn check_image_id(id: &str) -> Result<(), Error> {
    if id.is_empty() || id.len() > MAX_IMAGE_ID || id.contains(',') {
        return Err(invalid_arguments(format!("`{}` is not an image id", crate::tools::short(id, 40))));
    }
    Ok(())
}

/// Checks `frame` (`set_label_boxes`' `size`) against a `width` × `height` image: it must be the frame of a view
/// of that image, i.e. what [`frame`] gives for its long side (so the `size` a view returned always passes,
/// rounding and the one-pixel minimum side included).
fn check_frame(f: [u32; 2], width: u32, height: u32) -> Result<(), Error> {
    let [fw, fh] = f;
    if fw == 0 || fh == 0 || fw > 8192 || fh > 8192 || frame(width, height, fw.max(fh)) != f {
        let example = frame(width, height, DEFAULT_VIEW_SIZE);
        return Err(invalid_arguments(format!(
            "size [{fw}, {fh}] does not have the image's shape ({width}x{height}): give the size of the view the \
             boxes were read on, e.g. [{}, {}]",
            example[0], example[1]
        )));
    }
    Ok(())
}

impl Labels {
    /// The job with its review count and its last `log_lines` log lines (`label_status`).
    pub fn status(&self, id: &str, log_lines: usize) -> Result<LabelStatusResult, Error> {
        self.status_of(self.job(id)?, log_lines)
    }

    /// [`Self::status`] of a job already read: its run and counts are those of `job`, not of a later read (a
    /// waiter reports the run it saw end, not the one that may have started since).
    pub fn status_of(&self, job: folo_api::LabelJob, log_lines: usize) -> Result<LabelStatusResult, Error> {
        let id = job.id.as_str();
        let reviewed = self.count(id, Some(LabelImageStatus::Labeled), Some(true))?;
        let log = if log_lines == 0 { Vec::new() } else { self.logs(id, log_lines)? };
        Ok(LabelStatusResult {
            id: job.id,
            name: job.name,
            status: job.status,
            classes: job.classes,
            counts: job.counts,
            reviewed,
            run: job.run.map(Into::into),
            usage: job.usage,
            dataset_id: job.dataset_id,
            log: log.iter().map(compact::log_line).collect(),
        })
    }

    /// Images of job `id` with this status and review mark.
    fn count(
        &self,
        id: &str,
        status: Option<LabelImageStatus>,
        reviewed: Option<bool>,
    ) -> Result<u64, Error> {
        let q = LabelImageQuery { status, reviewed, limit: 0, ..LabelImageQuery::default() };
        Ok(self.images(id, &q)?.total)
    }

    /// Labeled images of job `id` that are not reviewed.
    pub fn to_review(&self, id: &str) -> Result<u64, Error> {
        self.count(id, Some(LabelImageStatus::Labeled), Some(false))
    }

    /// Image `image_id` of job `id` (exactly that id).
    pub fn image_record(&self, id: &str, image_id: &str) -> Result<LabelImage, Error> {
        check_image_id(image_id)?;
        let q = LabelImageQuery {
            ids: Some(HashSet::from([image_id.to_string()])),
            limit: 1,
            ..LabelImageQuery::default()
        };
        let found = self.images(id, &q)?.images.into_iter().find(|i| i.id == image_id);
        found.ok_or_else(|| {
            Error::new(
                StatusCode::NOT_FOUND,
                "image_not_found",
                format!("labeling job '{id}' has no image '{image_id}'"),
            )
        })
    }

    /// The next image to review: the first labeled one that is not reviewed.
    fn next_to_review(&self, id: &str) -> Result<Option<LabelImage>, Error> {
        let q = LabelImageQuery {
            status: Some(LabelImageStatus::Labeled),
            reviewed: Some(false),
            limit: 1,
            ..LabelImageQuery::default()
        };
        Ok(self.images(id, &q)?.images.pop())
    }

    /// A page of job `id`'s images as summaries (`list_label_images`): `offset` and at most `limit` of them.
    pub fn image_page(
        &self,
        id: &str,
        status: Option<LabelImageStatus>,
        reviewed: Option<bool>,
        include_boxes: bool,
        offset: u64,
        limit: u64,
    ) -> Result<(Vec<LabelImageSummary>, u64), Error> {
        let q = LabelImageQuery { status, reviewed, ids: None, offset, limit: limit.min(PAGE) };
        let page = self.images(id, &q)?;
        Ok((page.images.iter().map(|i| compact::image_summary(i, include_boxes)).collect(), page.total))
    }

    /// Image `image_id` of job `id` (else the next to review), with the JPEG of its view: at most `size` pixels on
    /// the long side, every box drawn and numbered, boxes listed from number `from` (`view_label_image`). No image
    /// when nothing is left to review.
    pub fn view(
        &self,
        id: &str,
        image_id: Option<&str>,
        size: u32,
        from: u32,
    ) -> Result<(LabelImageViewResult, Option<Vec<u8>>), Error> {
        let job = self.job(id)?;
        let image = match image_id {
            Some(image_id) => Some(self.image_record(id, image_id)?),
            None => self.next_to_review(id)?,
        };
        let to_review = self.to_review(id)?;
        let mut result =
            LabelImageViewResult { job_id: job.id, classes: job.classes, to_review, image: None };
        let Some(image) = image else { return Ok((result, None)) };
        let f = frame(image.width, image.height, size);
        let view = compact::image_view(&image, f, from);
        let jpeg = self.image(id, &image.id, size)?;
        let draw: Vec<_> = image
            .boxes
            .iter()
            .map(|b| (b.class, compact::view_corners(b, (image.width, image.height), f)))
            .collect();
        let drawn = super::view::render(&jpeg.bytes, f, &draw)?;
        result.image = Some(view);
        Ok((result, Some(drawn)))
    }

    /// Changes image `image_id`'s boxes (`set_label_boxes`). Saved as boxes drawn by hand: the image becomes
    /// `labeled`, `source: manual` and reviewed, as Studio's save does. The change is made on the revision the
    /// caller saw (`expect`), else on the boxes read here; a server with `label_revisions` checks it under the job's
    /// lock, an older one only here (a change in between is then overwritten). The result lists the boxes in the
    /// edit's frame.
    pub fn set_boxes(&self, id: &str, image_id: &str, edit: &BoxEdit) -> Result<LabelImageView, Error> {
        if edit.boxes.is_some() == edit.edits() {
            return Err(invalid_arguments(
                "give `boxes` (all of them), or edits of the current ones (`relabel`, `remove`, `add`)",
            ));
        }
        let Some(expect) = edit.expect.as_deref() else {
            return Err(invalid_arguments(
                "give `expect`: the `revision` of the boxes the edit was made on, as view_label_image showed it",
            ));
        };
        if edit.coordinates() && edit.size.is_none() {
            return Err(invalid_arguments(
                "give `size`: the [width, height] of the view the boxes were read on, as view_label_image returned \
                 it (coordinates of another view would be saved at the wrong scale)",
            ));
        }
        let classes = self.job(id)?.classes.len() as u32;
        let image = self.image_record(id, image_id)?;
        let now = label_box_revision(&image.boxes);
        if expect != now {
            return Err(boxes_changed(image_id, expect, &now));
        }
        let size = (image.width, image.height);
        let f = match edit.size {
            Some(f) => {
                check_frame(f, image.width, image.height)?;
                f
            }
            None => frame(image.width, image.height, DEFAULT_VIEW_SIZE),
        };
        let class_ok = |what: &str, c: u32| {
            if c < classes {
                Ok(())
            } else {
                Err(invalid_arguments(format!(
                    "{what}: class {c} is not one of the job's {classes} classes (0 to {})",
                    classes.saturating_sub(1)
                )))
            }
        };
        // A new box: a class of the job, inside the frame, at least a pixel of the image wide and high.
        let new_box = |what: &str, p: &PxBox| -> Result<BBox, Error> {
            class_ok(what, p[0])?;
            let [_, x1, y1, x2, y2] = *p;
            if x1 >= x2 || y1 >= y2 || x2 > f[0] || y2 > f[1] {
                return Err(invalid_arguments(format!(
                    "{what}: {p:?} is not [class, x1, y1, x2, y2] with x1 < x2 <= {} and y1 < y2 <= {} (pixels \
                     of the {}x{} view)",
                    f[0], f[1], f[0], f[1]
                )));
            }
            let b = from_view(p, size, f);
            if b.x2 - b.x1 < 1.0 || b.y2 - b.y1 < 1.0 {
                return Err(invalid_arguments(format!("{what}: {p:?} is less than a pixel of the image")));
            }
            Ok(b)
        };
        let boxes = match &edit.boxes {
            Some(all) => {
                if all.len() > folo_server::label::prompt::MAX_BOXES {
                    return Err(invalid_arguments(format!(
                        "an image takes at most {} boxes",
                        folo_server::label::prompt::MAX_BOXES
                    )));
                }
                // A box sent back as the view shows it keeps its stored pixels (no re-rounding, and a narrow box
                // that rounds to nothing in the view can be sent back as shown).
                let mut shown: HashMap<PxBox, Vec<usize>> = HashMap::new();
                for (k, b) in image.boxes.iter().enumerate().rev() {
                    shown.entry(to_view(b, size, f)).or_default().push(k);
                }
                // Stored boxes that look the same in this view but differ: which ones a shorter list keeps is
                // not known, so it is refused rather than guessed.
                let mut sent: HashMap<&PxBox, usize> = HashMap::new();
                all.iter().for_each(|p| *sent.entry(p).or_default() += 1);
                for (p, ks) in &shown {
                    let n = sent.get(p).copied().unwrap_or(0);
                    let same = ks.iter().all(|&k| image.boxes[k] == image.boxes[ks[0]]);
                    if n > 0 && n < ks.len() && !same {
                        let mut numbers: Vec<usize> = ks.iter().map(|k| k + 1).collect();
                        numbers.sort_unstable();
                        return Err(invalid_arguments(format!(
                            "boxes {numbers:?} all show as {p:?} in this {}x{} view but differ, and {n} of them \
                             were sent: remove the others by number (`remove`), or send boxes of a larger view",
                            f[0], f[1]
                        )));
                    }
                }
                let mut out = Vec::with_capacity(all.len());
                for (i, p) in all.iter().enumerate() {
                    match shown.get_mut(p).and_then(Vec::pop) {
                        Some(k) => out.push(image.boxes[k]),
                        None => out.push(new_box(&format!("box {}", i + 1), p)?),
                    }
                }
                out
            }
            None => {
                // Edits refer to the numbers of the current boxes, which keep their exact pixels.
                let n = image.boxes.len() as u32;
                let number = |what: &str, k: u32| {
                    if (1..=n).contains(&k) {
                        Ok(k as usize - 1)
                    } else {
                        Err(invalid_arguments(format!(
                            "{what}: image {image_id} has no box {k} (it has {n})"
                        )))
                    }
                };
                let mut relabeled = HashMap::new();
                for [k, class] in &edit.relabel {
                    let what = format!("relabel {k}");
                    let i = number(&what, *k)?;
                    class_ok(&what, *class)?;
                    if relabeled.insert(i, *class).is_some() {
                        return Err(invalid_arguments(format!("{what}: box {k} is relabeled twice")));
                    }
                }
                let mut gone = BTreeSet::new();
                for k in &edit.remove {
                    let i = number(&format!("remove {k}"), *k)?;
                    if relabeled.contains_key(&i) {
                        return Err(invalid_arguments(format!("box {k} is both relabeled and removed")));
                    }
                    gone.insert(i);
                }
                let added = edit
                    .add
                    .iter()
                    .enumerate()
                    .map(|(i, p)| new_box(&format!("add {}", i + 1), p))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut boxes: Vec<BBox> = image
                    .boxes
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !gone.contains(i))
                    .map(|(i, b)| BBox { class: relabeled.get(&i).copied().unwrap_or(b.class), ..*b })
                    .collect();
                boxes.extend(added);
                boxes
            }
        };
        let req = UpdateLabelImage { boxes: Some(boxes), reviewed: None, expect: Some(now) };
        let saved = self.update_image(id, image_id, req)?;
        Ok(compact::image_view(&saved, f, 1))
    }

    /// Marks images reviewed (or not): Studio's "Looks right" (`review_label_images`), each optionally on the
    /// revision its boxes were checked on. One save on a server with `label_revisions` (and in process); an
    /// older server gets one request per image, with the revisions checked here first (racy).
    pub fn review(
        &self,
        id: &str,
        image_ids: &[String],
        reviewed: bool,
        expect: &std::collections::BTreeMap<String, String>,
    ) -> Result<ReviewLabelImagesResult, Error> {
        if image_ids.is_empty() || image_ids.len() > MAX_REVIEW {
            return Err(invalid_arguments(format!("name 1 to {MAX_REVIEW} images")));
        }
        for i in image_ids {
            check_image_id(i)?;
        }
        // A revision for an image that is not reviewed here is a mistake (a typo'd id, or the wrong list), and
        // would be silently ignored.
        let named: HashSet<&str> = image_ids.iter().map(String::as_str).collect();
        if let Some(stray) = expect.keys().find(|k| !named.contains(k.as_str())) {
            return Err(invalid_arguments(format!(
                "expect names image '{}', which is not in image_ids",
                crate::tools::short(stray, 40)
            )));
        }
        let req = ReviewLabelImages { image_ids: image_ids.to_vec(), reviewed, expect: expect.clone() };
        if let Some(r) = self.review_batch(id, &req)? {
            return Ok(ReviewLabelImagesResult {
                updated: r.updated,
                failed: r.failed,
                failed_total: r.failed_total,
                to_review: Some(r.to_review),
            });
        }
        // A job that is not there is one error, not one per image.
        self.job(id)?;
        let (mut updated, mut failed, mut failed_total) = (0u64, Vec::new(), 0u64);
        let mut seen = HashSet::new();
        for image_id in image_ids.iter().filter(|i| seen.insert(i.as_str())) {
            let done = (|| {
                if let Some(e) = expect.get(image_id) {
                    let now = label_box_revision(&self.image_record(id, image_id)?.boxes);
                    if *e != now {
                        return Err(boxes_changed(image_id, e, &now));
                    }
                }
                let req = UpdateLabelImage { boxes: None, reviewed: Some(reviewed), expect: None };
                self.update_image(id, image_id, req)
            })();
            match done {
                Ok(_) => updated += 1,
                Err(e) => {
                    failed_total += 1;
                    if failed.len() < 20 {
                        failed.push(ReviewFailure {
                            id: image_id.clone(),
                            code: e.body.code,
                            message: crate::tools::short(&e.body.message, 200),
                        });
                    }
                }
            }
        }
        // The marks are saved: a count that fails now is left out, not an error.
        Ok(ReviewLabelImagesResult { updated, failed, failed_total, to_review: self.to_review(id).ok() })
    }

    /// A new job, with the images of `paths` (absolute paths on the server's machine) added (`create_label_job`).
    pub fn create_job(
        &self,
        name: String,
        classes: Vec<String>,
        instructions: String,
        paths: &[String],
    ) -> Result<LabelImagesAddedResult, CreateFailed> {
        let failed = |error, made| CreateFailed { error, made };
        let job = self
            .create(CreateLabelJob { name, classes, instructions, path: None })
            .map_err(|e| failed(e, None))?;
        let mut adds = Vec::with_capacity(paths.len());
        for p in paths {
            match self.add_path(&job.id, p) {
                Ok(a) => adds.push(a),
                // Nothing added yet: no job is left behind for a path that is not there.
                Err(e) if adds.iter().all(|a| a.added.is_empty()) => {
                    let _ = self.delete(&job.id);
                    return Err(failed(e, None));
                }
                Err(e) => return Err(failed(e, Some(job.id))),
            }
        }
        let images = adds.last().map_or(0, |a| a.job.counts.images);
        Ok(compact::added(&job.id, &adds, images))
    }

    /// Adds the images of `paths` (folders or image files on the server's machine) to job `id`.
    pub fn add_paths(&self, id: &str, paths: &[String]) -> Result<LabelImagesAddedResult, Error> {
        let mut adds = Vec::with_capacity(paths.len());
        for p in paths {
            adds.push(self.add_path(id, p)?);
        }
        let images = match adds.last() {
            Some(a) => a.job.counts.images,
            None => self.job(id)?.counts.images,
        };
        Ok(compact::added(id, &adds, images))
    }

    /// Starts an agent run (`start_labeling`).
    pub fn start_run(&self, id: &str, req: StartLabeling) -> Result<LabelRunStarted, Error> {
        Ok((&self.start(id, req)?).into())
    }

    /// Stops the active run (`stop_labeling`).
    pub fn stop_run(&self, id: &str) -> Result<LabelJobState, Error> {
        let job = self.stop(id)?;
        Ok(LabelJobState { job_id: job.id, status: job.status })
    }

    /// Writes and registers the dataset of the job's labeled images (`export_label_dataset`).
    pub fn export(&self, id: &str, req: LabelDatasetRequest) -> Result<LabelDatasetResult, Error> {
        let d = self.dataset(id, req)?;
        Ok(LabelDatasetResult {
            dataset_id: d.id,
            name: d.name,
            path: d.path,
            train: d.splits.train,
            val: d.splits.val,
        })
    }

    /// The agents as summaries (`folo label agents`).
    pub fn agent_list(&self) -> Result<LabelAgentsResult, Error> {
        Ok(LabelAgentsResult { agents: self.agents()?.into_iter().map(compact::agent).collect() })
    }

    /// The jobs as summaries, newest first (`list_label_jobs`).
    pub fn job_list(&self) -> Result<Vec<LabelJobSummary>, Error> {
        Ok(self.jobs()?.into_iter().map(compact::job).collect())
    }
}
