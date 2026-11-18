//! The labeling operations the MCP tools and `folo label` share, built on the calls of [`Labels`]: each answers
//! with a result of [`crate::types::label`].

use std::collections::{BTreeSet, HashSet};

use folo_api::{
    CreateLabelJob, LabelDatasetRequest, LabelImage, LabelImageStatus, StartLabeling, UpdateLabelImage,
};
use folo_server::label::LabelImageQuery;
use http::StatusCode;

use super::Labels;
use super::compact::{self, from_rel};
use crate::errors::{Error, invalid_arguments};
use crate::types::label::*;

/// Most images `list_label_images` asks the server for at once.
const PAGE: u64 = 1000;
/// Most failures `review_label_images` names.
const FAILED_FIRST: usize = 20;

/// `set_label_boxes`: all the boxes, or edits of the current ones by their 1-based number.
#[derive(Clone, Debug, Default)]
pub struct BoxEdit {
    /// All the boxes, replacing the image's.
    pub boxes: Option<Vec<RelBox>>,
    /// Numbers of boxes to delete.
    pub remove: Vec<u32>,
    /// `[number, class]`: boxes to give another class.
    pub relabel: Vec<[u32; 2]>,
    /// Boxes to append.
    pub add: Vec<RelBox>,
}

impl BoxEdit {
    fn edits(&self) -> bool {
        !(self.remove.is_empty() && self.relabel.is_empty() && self.add.is_empty())
    }
}

/// An `invalid_arguments` error unless `r` is a box: corners in 0..=1000, `x1 < x2` and `y1 < y2`.
fn check_rel(r: &RelBox) -> Result<(), Error> {
    let [_, x1, y1, x2, y2] = *r;
    if x2 > 1000 || y2 > 1000 || x1 >= x2 || y1 >= y2 {
        return Err(invalid_arguments(format!(
            "box {r:?} is not [class, x1, y1, x2, y2] with 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000"
        )));
    }
    Ok(())
}

impl Labels {
    /// The job with its review count and its last `log_lines` log lines (`label_status`).
    pub fn status(&self, id: &str, log_lines: usize) -> Result<LabelStatusResult, Error> {
        let job = self.job(id)?;
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

    /// Image `image_id` of job `id`.
    pub fn image_record(&self, id: &str, image_id: &str) -> Result<LabelImage, Error> {
        let q = LabelImageQuery {
            ids: Some(HashSet::from([image_id.to_string()])),
            limit: 1,
            ..LabelImageQuery::default()
        };
        self.images(id, &q)?.images.pop().ok_or_else(|| {
            Error::new(
                StatusCode::NOT_FOUND,
                "image_not_found",
                format!("labeling job '{id}' has no image '{image_id}'"),
            )
            .hint("list_label_images (MCP) or `folo label images` lists a job's images")
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
    /// the long side, its boxes drawn and numbered (`view_label_image`). No image when nothing is left to review.
    pub fn view(
        &self,
        id: &str,
        image_id: Option<&str>,
        size: u32,
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
        let view = compact::image_view(&image);
        let jpeg = self.image(id, &image.id, size)?;
        let drawn = super::view::render(&jpeg.bytes, &view.boxes)?;
        result.image = Some(view);
        Ok((result, Some(drawn)))
    }

    /// Changes image `image_id`'s boxes (`set_label_boxes`). Saved as boxes drawn by hand: the image becomes
    /// `labeled`, `source: manual` and reviewed, as Studio's save does.
    pub fn set_boxes(&self, id: &str, image_id: &str, edit: &BoxEdit) -> Result<LabelImageView, Error> {
        if edit.boxes.is_some() == edit.edits() {
            return Err(invalid_arguments(
                "give `boxes` (all of them), or edits of the current ones (`remove`, `relabel`, `add`)",
            ));
        }
        for r in edit.boxes.iter().flatten().chain(&edit.add) {
            check_rel(r)?;
        }
        let image = self.image_record(id, image_id)?;
        let (w, h) = (image.width, image.height);
        let boxes = match &edit.boxes {
            Some(all) => all.iter().map(|r| from_rel(r, w, h)).collect(),
            None => {
                // Edits refer to the numbers of the current boxes, which keep their exact pixels.
                let mut boxes = image.boxes;
                let n = boxes.len() as u32;
                let number = |k: u32| {
                    if (1..=n).contains(&k) {
                        Ok(k as usize - 1)
                    } else {
                        Err(invalid_arguments(format!("image {image_id} has no box {k} (it has {n})")))
                    }
                };
                for [k, class] in &edit.relabel {
                    boxes[number(*k)?].class = *class;
                }
                let gone: BTreeSet<usize> =
                    edit.remove.iter().map(|k| number(*k)).collect::<Result<_, _>>()?;
                let mut i = 0;
                boxes.retain(|_| {
                    i += 1;
                    !gone.contains(&(i - 1))
                });
                boxes.extend(edit.add.iter().map(|r| from_rel(r, w, h)));
                boxes
            }
        };
        let saved =
            self.update_image(id, image_id, UpdateLabelImage { boxes: Some(boxes), reviewed: None })?;
        Ok(compact::image_view(&saved))
    }

    /// Marks images reviewed (or not): Studio's "Looks right" (`review_label_images`).
    pub fn review(
        &self,
        id: &str,
        image_ids: &[String],
        reviewed: bool,
    ) -> Result<ReviewLabelImagesResult, Error> {
        // A job that is not there is one error, not one per image.
        self.job(id)?;
        let (mut updated, mut failed) = (0u64, Vec::new());
        let mut seen = HashSet::new();
        for image_id in image_ids.iter().filter(|i| seen.insert(i.as_str())) {
            let req = UpdateLabelImage { boxes: None, reviewed: Some(reviewed) };
            match self.update_image(id, image_id, req) {
                Ok(_) => updated += 1,
                Err(e) if failed.len() < FAILED_FIRST => failed.push(ReviewFailure {
                    id: image_id.clone(),
                    code: e.body.code,
                    message: crate::tools::short(&e.body.message, 200),
                }),
                Err(_) => {}
            }
        }
        Ok(ReviewLabelImagesResult { updated, failed, to_review: self.to_review(id)? })
    }

    /// A new job, with the images of `paths` (absolute paths on the server's machine) added (`create_label_job`).
    pub fn create_job(
        &self,
        name: String,
        classes: Vec<String>,
        instructions: String,
        paths: &[String],
    ) -> Result<LabelImagesAddedResult, Error> {
        let job = self.create(CreateLabelJob { name, classes, instructions, path: None })?;
        let mut adds = Vec::with_capacity(paths.len());
        for p in paths {
            match self.add_path(&job.id, p) {
                Ok(a) => adds.push(a),
                // Nothing added yet: no job is left behind for a path that is not there.
                Err(e) if adds.iter().all(|a| a.added.is_empty()) => {
                    let _ = self.delete(&job.id);
                    return Err(e);
                }
                Err(e) => {
                    let hint = format!(
                        "job {} was made with the images of the paths before it; fix this path and add it with                          add_label_images",
                        job.id
                    );
                    return Err(e.hint(hint));
                }
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
        let job = self.start(id, req)?;
        Ok(LabelRunStarted { job_id: job.id, status: job.status, run: job.run.map(Into::into) })
    }

    /// Stops the active run (`stop_labeling`).
    pub fn stop_run(&self, id: &str) -> Result<LabelRunStarted, Error> {
        let job = self.stop(id)?;
        Ok(LabelRunStarted { job_id: job.id, status: job.status, run: job.run.map(Into::into) })
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

    /// The agents as summaries (`list_label_agents`).
    pub fn agent_list(&self) -> Result<LabelAgentsResult, Error> {
        Ok(LabelAgentsResult { agents: self.agents()?.into_iter().map(compact::agent).collect() })
    }

    /// The jobs as summaries, newest first (`list_label_jobs`).
    pub fn job_list(&self) -> Result<Vec<LabelJobSummary>, Error> {
        Ok(self.jobs()?.into_iter().map(compact::job).collect())
    }
}
