//! A job's images: adding uploads and folders, listing and serving them, editing their boxes, removing them.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use folo_api::{
    LabelEvent, LabelImage, LabelImagePage, LabelImageStatus, LabelImagesAdded, LabelReviewFailure,
    LabelReviewResult, LabelSource, RemovedImage, ReviewLabelImages, SkippedImage, UpdateLabelImage,
    label_box_revision,
};
use rayon::prelude::*;

use crate::error::Result;
use crate::index::{read_image_with, write_atomic};
use crate::store::ImageData;
use crate::util::fnv1a;

use super::job::{Job, Record, stem_key, unique_file_name};
use super::validate::check_boxes;
use super::{BULK_IMAGES, Labeler, MAX_IMAGES, Upload, folder_path, invalid, io_error, path_not_found};

impl Labeler {
    /// `POST /api/label/jobs/{id}/images` (multipart): stores the uploads in the job's folder.
    pub fn add_uploads(&self, id: &str, uploads: Vec<Upload>) -> Result<LabelImagesAdded> {
        static N: AtomicU64 = AtomicU64::new(0);
        let job = self.job(id)?;
        let dir = job.images_dir();
        std::fs::create_dir_all(&dir).map_err(|e| io_error("creating", &dir, e))?;
        // Written and probed in parallel before the lock is taken (rayon preserves input order); named for good
        // under the lock.
        let pid = std::process::id();
        let results: Vec<std::result::Result<Candidate, SkippedImage>> = uploads
            .into_par_iter()
            .map(|upload| {
                if !folo_data::is_image_path(Path::new(&upload.name)) {
                    return Err(SkippedImage {
                        file: upload.name,
                        reason: "not a JPEG, PNG, BMP or WebP file".into(),
                    });
                }
                let tmp = dir.join(format!(".upload-{pid}-{}", N.fetch_add(1, Ordering::Relaxed)));
                let probed = std::fs::write(&tmp, &upload.bytes)
                    .map_err(|e| format!("could not be stored: {e}"))
                    .and_then(|()| {
                        folo_data::probe_size(&tmp).map_err(|_| "not a readable image".to_string())
                    });
                match probed {
                    Ok(size) => Ok(Candidate {
                        name: upload.name,
                        source: Source::Upload { tmp, hash: fnv1a(&upload.bytes) },
                        sha256: super::dataset::sha256_bytes(&upload.bytes),
                        size,
                    }),
                    Err(reason) => {
                        let _ = std::fs::remove_file(&tmp);
                        Err(SkippedImage { file: upload.name, reason })
                    }
                }
            })
            .collect();
        let (mut staged, mut skipped) = (Vec::new(), Vec::new());
        for res in results {
            match res {
                Ok(c) => staged.push(c),
                Err(s) => skipped.push(s),
            }
        }
        self.insert(&job, staged, skipped)
    }

    /// `POST /api/label/jobs/{id}/images` (JSON): adds the images of a folder on this machine, in place.
    pub fn add_folder(&self, id: &str, path: &str) -> Result<LabelImagesAdded> {
        let job = self.job(id)?;
        let Some(folder) = folder_path(path)? else {
            return Err(invalid("path must not be blank"));
        };
        if !folder.exists() {
            return Err(path_not_found(&folder));
        }
        self.add_folder_to(&job, &folder)
    }

    pub(super) fn add_folder_to(&self, job: &Job, folder: &Path) -> Result<LabelImagesAdded> {
        // Known files and the room left are read up front so that neither is probed for nothing; `insert` has the
        // last word on both.
        let (known, room) = {
            let state = job.lock();
            let known: HashSet<PathBuf> =
                state.file.images.iter().filter_map(|r| r.external.clone()).collect();
            (known, MAX_IMAGES.saturating_sub(state.file.images.len()))
        };
        // The other files' extensions, a few, say why a folder without images added nothing.
        let mut others = std::collections::BTreeSet::new();
        let walked: Vec<PathBuf> = walkdir::WalkDir::new(folder)
            .sort_by_file_name()
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter(|e| {
                let image = folo_data::is_image_path(e.path());
                if !image && others.len() < 4 {
                    others.insert(
                        e.path().extension().map(|x| x.to_string_lossy().to_lowercase()).unwrap_or_default(),
                    );
                }
                image
            })
            .map(|e| e.into_path())
            .collect();
        let found = walked.len();
        let (staged, mut skipped, over) = probe_folder_files(walked, &known, room);
        if over > 0 {
            skipped.push(SkippedImage {
                file: folder.display().to_string(),
                reason: format!("{over} more images were left out: a job holds at most {MAX_IMAGES}"),
            });
        }
        if found == 0 {
            skipped.push(nothing_found(folder, &others));
        }
        self.insert(job, staged, skipped)
    }

    /// Adds probed images to the job under its lock: duplicates and images past the limit go to `skipped`.
    fn insert(
        &self,
        job: &Job,
        staged: Vec<Candidate>,
        mut skipped: Vec<SkippedImage>,
    ) -> Result<LabelImagesAdded> {
        // The job may have been deleted while the files were probed: its staged uploads go with it.
        let mut state = job.live().inspect_err(|_| staged.iter().for_each(|c| c.source.discard()))?;
        let images = &state.file.images;
        let mut taken: HashSet<String> = images.iter().map(|r| stem_key(&r.file)).collect();
        // What the job has, by what makes an image the same one: an upload's bytes, a folder image's path.
        let mut uploads: HashMap<u64, String> =
            images.iter().filter_map(|r| Some((r.hash?, r.file.clone()))).collect();
        let mut folders: HashSet<PathBuf> = images.iter().filter_map(|r| r.external.clone()).collect();
        let (first_new, next_image) = (images.len(), state.file.next_image);
        for candidate in staged {
            let Candidate { name, source, sha256, size: (width, height) } = candidate;
            let reason = if let Source::Upload { hash, .. } = &source
                && let Some(same) = uploads.get(hash)
            {
                Some(format!("the job already has this image ({same})"))
            } else if matches!(&source, Source::Folder(path) if folders.contains(path)) {
                Some("the job already has this file".to_string())
            } else if state.file.images.len() >= MAX_IMAGES {
                Some(format!("a job holds at most {MAX_IMAGES} images"))
            } else {
                None
            };
            if let Some(reason) = reason {
                source.discard();
                skipped.push(SkippedImage { file: name, reason });
                continue;
            }
            let file = unique_file_name(&name, &mut taken);
            let (external, hash) = match source {
                Source::Folder(path) => {
                    folders.insert(path.clone());
                    (Some(path), None)
                }
                Source::Upload { tmp, hash } => {
                    if let Err(e) = std::fs::rename(&tmp, job.images_dir().join(&file)) {
                        let _ = std::fs::remove_file(&tmp);
                        taken.remove(&stem_key(&file));
                        skipped
                            .push(SkippedImage { file: name, reason: format!("could not be stored: {e}") });
                        continue;
                    }
                    uploads.insert(hash, file.clone());
                    (None, Some(hash))
                }
            };
            let id = state.file.next_image.to_string();
            state.file.next_image += 1;
            state.file.images.push(Record {
                id,
                file,
                external,
                hash,
                sha256: Some(sha256),
                width,
                height,
                status: LabelImageStatus::Pending,
                boxes: Vec::new(),
                source: None,
                error: None,
                reviewed: false,
            });
        }
        let added: Vec<LabelImage> = state.file.images[first_new..].iter().map(|r| state.image(r)).collect();
        if !added.is_empty() {
            if let Err(e) = job.save(&state) {
                // Not on disk: the job stays as it was, without the files stored for the new images.
                for r in state.file.images.drain(first_new..) {
                    if r.external.is_none() {
                        let _ = std::fs::remove_file(job.image_path(&r));
                    }
                }
                state.file.next_image = next_image;
                return Err(e);
            }
            if added.len() > BULK_IMAGES {
                job.emit(LabelEvent::Reload);
            } else {
                added.iter().cloned().for_each(|image| job.emit(LabelEvent::Image(image)));
            }
            job.emit_job(&state);
        }
        Ok(LabelImagesAdded { added, skipped, job: state.job() })
    }

    /// `GET /api/label/jobs/{id}/images` with only a status: [`Labeler::query`] without the other filters.
    pub fn images(
        &self,
        id: &str,
        status: Option<LabelImageStatus>,
        offset: u64,
        limit: u64,
    ) -> Result<LabelImagePage> {
        self.query(id, &LabelImageQuery { status, offset, limit, ..LabelImageQuery::default() })
    }

    /// `GET /api/label/jobs/{id}/images`: the images that pass every filter of `q`, paged.
    pub fn query(&self, id: &str, q: &LabelImageQuery) -> Result<LabelImagePage> {
        let (offset, limit) = (q.offset, q.limit);
        let job = self.job(id)?;
        let at = q.path.as_deref().map(|p| images_at(&job, p)).transpose()?;
        let state = job.live()?;
        let matching = state.file.images.iter().filter(|r| {
            q.status.is_none_or(|s| s == state.status(r))
                && q.reviewed.is_none_or(|v| v == state.reviewed(r))
                && q.ids.as_ref().is_none_or(|ids| ids.contains(&r.id))
                && at.as_ref().is_none_or(|ids| ids.contains(&r.id))
        });
        let (mut total, mut images) = (0u64, Vec::new());
        for record in matching {
            if total >= offset && (images.len() as u64) < limit {
                images.push(state.image(record));
            }
            total += 1;
        }
        Ok(LabelImagePage { total, images })
    }

    /// `GET /api/label/jobs/{id}/images/{image_id}`: the file's bytes, or a JPEG thumbnail whose long side is
    /// `thumb` (cached under the data dir's `cache/thumbs/<job id>`).
    pub fn image(&self, id: &str, image_id: &str, thumb: Option<u32>) -> Result<ImageData> {
        let job = self.job(id)?;
        let path = {
            let state = job.live()?;
            job.image_path(state.record(image_id)?)
        };
        let cache = thumb.map(|_| self.thumbs.join(id));
        read_image_with(&path, thumb, cache.as_deref(), |file, bytes| {
            // Not after the job was deleted (its thumbnails went with it).
            if !job.lock().deleted
                && let Err(e) = write_atomic(file, bytes)
            {
                tracing::warn!("caching thumbnail {}: {e}", file.display());
            }
        })
    }

    /// `PUT /api/label/jobs/{id}/images/{image_id}`: boxes drawn by hand, and whether a person checked the labels.
    pub fn update_image(&self, id: &str, image_id: &str, req: UpdateLabelImage) -> Result<LabelImage> {
        let job = self.job(id)?;
        let mut state = job.live()?;
        let classes = state.file.classes.len();
        let record = state.record(image_id)?;
        if let Some(expect) = &req.expect {
            check_revision(image_id, &record.boxes, expect)?;
        }
        let boxes = match &req.boxes {
            Some(boxes) => Some(check_boxes(boxes, classes, (record.width, record.height))?),
            None if req.reviewed.is_none() => return Err(invalid("give `boxes`, `reviewed` or both")),
            None => {
                // Report `409 label_job_active` first when the image is in an agent call (or export),
                // without pulling a waiting image out of the active run before checking eligibility.
                state.check_editable(image_id)?;
                // Only labels can be checked: an image without them is saved with its boxes (none: a background image).
                if record.status != LabelImageStatus::Labeled {
                    return Err(invalid(format!(
                        "image '{image_id}' has no labels to mark reviewed: save its boxes instead"
                    )));
                }
                None
            }
        };
        // What the change replaces, to put back when it cannot be saved ([`State::claim`] takes the image out of
        // a run waiting for it).
        let old = record.clone();
        let waiting = state.active.as_ref().is_some_and(|a| a.waiting.contains(image_id));
        let total = state.file.run.as_ref().map(|r| r.total);
        state.claim(image_id)?;
        let record = state.record_mut(image_id).expect("the record was just read");
        if let Some(boxes) = boxes {
            record.boxes = boxes;
            record.status = LabelImageStatus::Labeled;
            record.source = Some(LabelSource::Manual);
            record.error = None;
            record.reviewed = true;
        }
        if let Some(reviewed) = req.reviewed {
            record.reviewed = reviewed;
        }
        if let Err(e) = job.save(&state) {
            *state.record_mut(image_id).expect("the record was just read") = old;
            if let (Some(run), Some(total)) = (state.file.run.as_mut(), total) {
                run.total = total;
            }
            if let Some(active) = state.active.as_mut().filter(|_| waiting) {
                active.waiting.insert(image_id.to_owned());
            }
            return Err(e);
        }
        let image = state.image(state.record(image_id)?);
        job.emit(LabelEvent::Image(image.clone()));
        job.emit_job(&state);
        Ok(image)
    }

    /// `POST /api/label/jobs/{id}/review`: marks images reviewed (or not) with one save of the job. Each image is
    /// checked as `PUT` checks it (it exists, has labels to be right, is not in an agent call, has the expected
    /// revision); those that fail are reported and left as they were, the rest are marked.
    pub fn review(&self, id: &str, req: &ReviewLabelImages) -> Result<LabelReviewResult> {
        if req.image_ids.len() > MAX_REVIEW {
            return Err(invalid(format!("review at most {MAX_REVIEW} images at a time")));
        }
        // A revision for an image not in the list (a typo'd id, the wrong list) would be silently ignored.
        if let Some(stray) = req.expect.keys().find(|k| !req.image_ids.contains(k)) {
            let stray: String = stray.chars().take(40).collect();
            return Err(invalid(format!("expect names image '{stray}', which is not in image_ids")));
        }
        let job = self.job(id)?;
        let mut state = job.live()?;
        let total = state.file.run.as_ref().map(|r| r.total);
        let (mut changed, mut failed, mut failed_total) = (Vec::new(), Vec::new(), 0u64);
        let mut seen = HashSet::new();
        for image_id in req.image_ids.iter().filter(|i| seen.insert(i.as_str())) {
            let checked = state.record(image_id).and_then(|r| {
                if let Some(expect) = req.expect.get(image_id) {
                    check_revision(image_id, &r.boxes, expect)?;
                }
                state.check_editable(image_id)?;
                if r.status != LabelImageStatus::Labeled {
                    return Err(invalid(format!("image '{image_id}' has no labels to mark reviewed")));
                }
                Ok(r.reviewed)
            });
            let waiting = state.active.as_ref().is_some_and(|a| a.waiting.contains(image_id.as_str()));
            match checked.and_then(|old| state.claim(image_id).map(|()| old)) {
                Ok(old) => {
                    state.record_mut(image_id).expect("the record was just read").reviewed = req.reviewed;
                    changed.push((image_id.clone(), old, waiting));
                }
                Err(e) => {
                    failed_total += 1;
                    if failed.len() < FAILED_FIRST {
                        let message = e.body.message.chars().take(200).collect();
                        failed.push(LabelReviewFailure { id: image_id.clone(), code: e.body.code, message });
                    }
                }
            }
        }
        if !changed.is_empty()
            && let Err(e) = job.save(&state)
        {
            for (image_id, old, waiting) in &changed {
                state.record_mut(image_id).expect("the record was just read").reviewed = *old;
                if let Some(active) = state.active.as_mut().filter(|_| *waiting) {
                    active.waiting.insert(image_id.clone());
                }
            }
            if let (Some(run), Some(total)) = (state.file.run.as_mut(), total) {
                run.total = total;
            }
            return Err(e);
        }
        if changed.len() > BULK_IMAGES {
            job.emit(LabelEvent::Reload);
        } else {
            for (image_id, _, _) in &changed {
                job.emit(LabelEvent::Image(state.image(state.record(image_id)?)));
            }
        }
        if !changed.is_empty() {
            job.emit_job(&state);
        }
        let to_review = state
            .file
            .images
            .iter()
            .filter(|r| state.status(r) == LabelImageStatus::Labeled && !state.reviewed(r))
            .count() as u64;
        Ok(LabelReviewResult { updated: changed.len() as u64, failed, failed_total, to_review })
    }

    /// `DELETE /api/label/jobs/{id}/images/{image_id}`: an upload's file goes too, a folder image's never.
    pub fn delete_image(&self, id: &str, image_id: &str) -> Result<()> {
        let job = self.job(id)?;
        let mut state = job.live()?;
        let pos = state
            .file
            .images
            .iter()
            .position(|r| r.id == image_id)
            .ok_or_else(|| crate::error::Error::not_found("image", image_id))?;
        state.check_editable(image_id)?;
        let waiting = state.active.as_ref().is_some_and(|a| a.waiting.contains(image_id));
        let record = state.file.images[pos].clone();
        // Persist a copy with the image removed before mutating in-memory state, so a failed save leaves
        // `state.file.images`, `active.waiting`, and `run.total` unchanged.
        let mut next = state.file.clone();
        next.images.remove(pos);
        if waiting && let Some(run) = next.run.as_mut() {
            run.total = run.total.saturating_sub(1);
        }
        job.save_file(&state, &next)?;
        state.claim(image_id)?;
        state.file.images.remove(pos);
        if record.external.is_none() {
            let _ = std::fs::remove_file(job.image_path(&record));
        }
        job.emit(LabelEvent::Removed(RemovedImage { id: record.id }));
        job.emit_job(&state);
        Ok(())
    }
}

/// Most images one `POST /api/label/jobs/{id}/review` marks.
pub const MAX_REVIEW: usize = 1000;
/// Most failures a review names.
const FAILED_FIRST: usize = 20;

/// `409 label_boxes_changed` unless `boxes` have the revision `expect`.
fn check_revision(image_id: &str, boxes: &[folo_api::BBox], expect: &str) -> Result<()> {
    let now = label_box_revision(boxes);
    if now == expect {
        return Ok(());
    }
    Err(crate::error::Error::conflict(
        "label_boxes_changed",
        format!("image '{image_id}' has other boxes than revision {expect} (now {now})"),
    ))
}

/// Filter and page of `GET /api/label/jobs/{id}/images` ([`Labeler::query`]).
#[derive(Clone, Debug, Default)]
pub struct LabelImageQuery {
    /// Only images with this status (as shown: `labeling` while in an agent call).
    pub status: Option<LabelImageStatus>,
    /// Only reviewed (`true`) or not reviewed (`false`) images.
    pub reviewed: Option<bool>,
    /// Only these image ids (in the job's order, not this list's).
    pub ids: Option<HashSet<String>>,
    /// Only the image whose file is this file, however either path spells it (`..`, letter case on Windows, a link
    /// to a folder); a path that does not exist on this machine is no image's.
    pub path: Option<PathBuf>,
    /// Skip this many matching images.
    pub offset: u64,
    /// Return at most this many.
    pub limit: u64,
}

/// An image about to be added to a job.
struct Candidate {
    name: String,
    source: Source,
    sha256: String,
    size: (u32, u32),
}

enum Source {
    /// A file added in place.
    Folder(PathBuf),
    /// An upload, written to a temporary file in the job's `images/`.
    Upload { tmp: PathBuf, hash: u64 },
}

impl Source {
    /// Drops an upload that is not added after all.
    fn discard(&self) {
        if let Source::Upload { tmp, .. } = self {
            let _ = std::fs::remove_file(tmp);
        }
    }
}

/// The ids of `job`'s images whose file is the file at `path` (an absolute path; both resolved, so any spelling of
/// the same path matches; a second hard link under another name does not). The files are resolved without holding
/// the job's lock.
fn images_at(job: &Job, path: &Path) -> Result<HashSet<String>> {
    let Some(path) = Some(path).filter(|p| p.is_absolute()).and_then(|p| std::fs::canonicalize(p).ok())
    else {
        return Ok(HashSet::new());
    };
    let files: Vec<_> = job.live()?.file.images.iter().map(|r| (r.id.clone(), job.image_path(r))).collect();
    let same = |f: &Path| std::fs::canonicalize(f).is_ok_and(|f| f == path);
    Ok(files.into_iter().filter(|(_, f)| same(f)).map(|(id, _)| id).collect())
}

/// Why a folder (or a file) added nothing: a file is not an image, a folder has none (and which other files it has).
fn nothing_found(path: &Path, others: &std::collections::BTreeSet<String>) -> SkippedImage {
    if path.is_file() {
        let file = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
        return SkippedImage { file, reason: "not a JPEG, PNG, BMP or WebP file".into() };
    }
    let mut reason = "no JPEG, PNG, BMP or WebP images in this folder or its subfolders".to_string();
    let kinds: Vec<String> = others.iter().filter(|x| !x.is_empty()).map(|x| format!(".{x}")).collect();
    if !kinds.is_empty() {
        reason.push_str(&format!(" (it has {} files)", kinds.join(", ")));
    }
    SkippedImage { file: path.display().to_string(), reason }
}

/// Probes unknown image paths in bounded parallel waves of size `room - staged.len()`, preserving exact input
/// order for both `staged` and `skipped` (including interleaved `known` files) while never probing files past
/// `room`.
fn probe_folder_files(
    walked: Vec<PathBuf>,
    known: &HashSet<PathBuf>,
    room: usize,
) -> (Vec<Candidate>, Vec<SkippedImage>, usize) {
    let mut staged = Vec::new();
    let mut skipped = Vec::new();
    let mut over = 0usize;
    let mut iter = walked.into_iter().peekable();

    while iter.peek().is_some() {
        let need = room.saturating_sub(staged.len());
        if need == 0 {
            for path in iter {
                if known.contains(&path) {
                    let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
                    skipped.push(SkippedImage { file: name, reason: "the job already has this file".into() });
                } else {
                    over += 1;
                }
            }
            break;
        }
        // Collect a wave containing at most `need` unknown paths (preserving any interleaved known files).
        enum Slot {
            Known(String),
            Probe(usize),
        }
        let mut wave_slots = Vec::new();
        let mut to_probe: Vec<(PathBuf, String)> = Vec::with_capacity(need.min(1024));
        while to_probe.len() < need {
            let Some(path) = iter.next() else { break };
            let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
            if known.contains(&path) {
                wave_slots.push(Slot::Known(name));
            } else {
                let idx = to_probe.len();
                to_probe.push((path, name));
                wave_slots.push(Slot::Probe(idx));
            }
        }
        let mut probed: Vec<Option<((u32, u32), String)>> = to_probe
            .par_iter()
            .map(|(p, _)| {
                let size = folo_data::probe_size(p).ok()?;
                let sha256 = super::dataset::sha256_file(p).ok()?;
                Some((size, sha256))
            })
            .collect();
        let mut to_probe: Vec<Option<(PathBuf, String)>> = to_probe.into_iter().map(Some).collect();
        for slot in wave_slots {
            match slot {
                Slot::Known(name) => {
                    skipped.push(SkippedImage { file: name, reason: "the job already has this file".into() });
                }
                Slot::Probe(i) => {
                    let (path, name) = to_probe[i].take().expect("unique probe index");
                    match probed[i].take() {
                        Some((size, sha256)) => {
                            staged.push(Candidate { name, source: Source::Folder(path), sha256, size })
                        }
                        None => {
                            skipped.push(SkippedImage { file: name, reason: "not a readable image".into() })
                        }
                    }
                }
            }
        }
    }
    (staged, skipped, over)
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_test_support::TestDir;

    /// Bounded parallel folder probing matches sequential order across interleaved known, corrupt, valid, and
    /// over-limit images, and pulls in extra waves when corrupt files leave room open.
    #[test]
    fn parallel_folder_probe_preserves_order_and_room_semantics() {
        let dir = TestDir::new("label-probe-order");
        let img = folo_data::Image { width: 8, height: 6, data: vec![128u8; 8 * 6 * 3] };
        let jpeg = folo_data::encode(&img, folo_data::ImageFormat::Jpeg(85)).unwrap();

        let p = |name: &str| dir.join(name);
        std::fs::write(p("01-known.jpg"), &jpeg).unwrap();
        std::fs::write(p("02-bad.jpg"), b"not an image").unwrap();
        std::fs::write(p("03-good1.jpg"), &jpeg).unwrap();
        std::fs::write(p("04-known2.jpg"), &jpeg).unwrap();
        std::fs::write(p("05-good2.jpg"), &jpeg).unwrap();
        std::fs::write(p("06-over.jpg"), &jpeg).unwrap();
        std::fs::write(p("07-known-after-full.jpg"), &jpeg).unwrap();

        let walked = vec![
            p("01-known.jpg"),
            p("02-bad.jpg"),
            p("03-good1.jpg"),
            p("04-known2.jpg"),
            p("05-good2.jpg"),
            p("06-over.jpg"),
            p("07-known-after-full.jpg"),
        ];
        let known = HashSet::from([p("01-known.jpg"), p("04-known2.jpg"), p("07-known-after-full.jpg")]);

        let (staged, skipped, over) = probe_folder_files(walked, &known, 2);
        assert_eq!(
            staged.iter().map(|c| (c.name.as_str(), c.size)).collect::<Vec<_>>(),
            [("03-good1.jpg", (8, 6)), ("05-good2.jpg", (8, 6))]
        );
        assert_eq!(
            skipped.iter().map(|s| (s.file.as_str(), s.reason.as_str())).collect::<Vec<_>>(),
            [
                ("01-known.jpg", "the job already has this file"),
                ("02-bad.jpg", "not a readable image"),
                ("04-known2.jpg", "the job already has this file"),
                ("07-known-after-full.jpg", "the job already has this file"),
            ]
        );
        assert_eq!(over, 1);
    }

    #[test]
    fn delete_image_leaves_memory_unchanged_when_save_fails() {
        let dir = TestDir::new("label-delete-save-fail");
        let labeler = Labeler::open(dir.path());
        let job = labeler
            .create(folo_api::CreateLabelJob {
                name: "test".into(),
                classes: vec!["cat".into()],
                instructions: String::new(),
                path: None,
            })
            .unwrap();
        let img = folo_data::Image { width: 8, height: 6, data: vec![128u8; 8 * 6 * 3] };
        let jpeg = folo_data::encode(&img, folo_data::ImageFormat::Jpeg(85)).unwrap();
        let added = labeler.add_uploads(&job.id, vec![Upload { name: "a.jpg".into(), bytes: jpeg }]).unwrap();
        assert_eq!(added.added.len(), 1);
        let page = labeler.query(&job.id, &LabelImageQuery { limit: 10, ..Default::default() }).unwrap();
        assert_eq!(page.total, 1);
        let image_id = page.images[0].id.clone();
        let upload_path = dir.join("label").join(&job.id).join("images").join(&page.images[0].file);
        assert!(upload_path.is_file());

        // Replace `job.json` with a directory so `write_atomic` cannot rename over it on any OS.
        let job_json = dir.join("label").join(&job.id).join("job.json");
        std::fs::remove_file(&job_json).unwrap();
        std::fs::create_dir(&job_json).unwrap();

        let err = labeler.delete_image(&job.id, &image_id).unwrap_err();
        assert_eq!(err.body.code, "internal", "{err:?}");

        // Restore cleanup-friendly state and verify in-memory job and uploaded image stayed intact.
        let after = labeler.query(&job.id, &LabelImageQuery { limit: 10, ..Default::default() }).unwrap();
        assert_eq!(after.total, 1, "failed save must not drop image from memory");
        assert_eq!(after.images[0].id, image_id);
        assert!(upload_path.is_file(), "failed save must not delete uploaded image file");
        std::fs::remove_dir(&job_json).unwrap();
    }

    #[test]
    fn delete_queued_image_preserves_run_on_save_failure_and_succeeds_on_retry() {
        let dir = TestDir::new("label-delete-queued-save-fail");
        let labeler = Labeler::open(dir.path());
        let job_info = labeler
            .create(folo_api::CreateLabelJob {
                name: "queued".into(),
                classes: vec!["cat".into()],
                instructions: String::new(),
                path: None,
            })
            .unwrap();
        let img = folo_data::Image { width: 8, height: 6, data: vec![128u8; 8 * 6 * 3] };
        let jpeg_a = folo_data::encode(&img, folo_data::ImageFormat::Jpeg(85)).unwrap();
        let img2 = folo_data::Image { width: 8, height: 6, data: vec![64u8; 8 * 6 * 3] };
        let jpeg_b = folo_data::encode(&img2, folo_data::ImageFormat::Jpeg(85)).unwrap();
        let added = labeler
            .add_uploads(
                &job_info.id,
                vec![
                    Upload { name: "a.jpg".into(), bytes: jpeg_a },
                    Upload { name: "b.jpg".into(), bytes: jpeg_b },
                ],
            )
            .unwrap();
        assert_eq!(added.added.len(), 2);
        let page = labeler.query(&job_info.id, &LabelImageQuery { limit: 10, ..Default::default() }).unwrap();
        let id_a = page.images[0].id.clone();
        let id_b = page.images[1].id.clone();
        let upload_a = dir.join("label").join(&job_info.id).join("images").join(&page.images[0].file);
        assert!(upload_a.is_file());

        // Simulate an active run with both images still waiting in the queue.
        let job = labeler.job(&job_info.id).unwrap();
        {
            let mut state = job.lock();
            state.file.run = Some(folo_api::LabelRun {
                id: "run_test".into(),
                agent: folo_api::LabelAgentId::Claude,
                model: None,
                effort: folo_api::LabelEffort::Low,
                fast: false,
                started: "2026-01-01T00:00:00Z".into(),
                finished: None,
                total: 2,
                done: 0,
                failed: 0,
                outcome: None,
                error: None,
                hint: None,
                usage: folo_api::LabelUsage::default(),
            });
            state.active = Some(super::super::job::Active {
                stopping: false,
                cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                waiting: HashSet::from([id_a.clone(), id_b.clone()]),
                in_call: HashSet::new(),
                thread: None,
            });
        }

        // Replace `job.json` with a directory so persisting the deletion fails.
        let job_json = dir.join("label").join(&job_info.id).join("job.json");
        std::fs::remove_file(&job_json).unwrap();
        std::fs::create_dir(&job_json).unwrap();

        let err = labeler.delete_image(&job_info.id, &id_a).unwrap_err();
        assert_eq!(err.body.code, "internal", "{err:?}");
        {
            let state = job.lock();
            assert_eq!(state.file.images.len(), 2);
            assert!(state.active.as_ref().unwrap().waiting.contains(&id_a));
            assert_eq!(state.file.run.as_ref().unwrap().total, 2);
        }
        assert!(upload_a.is_file(), "failed save must not delete uploaded image file");

        // Remove the blocking directory and retry: image, waiting entry, run.total, and upload all update.
        std::fs::remove_dir(&job_json).unwrap();
        labeler.delete_image(&job_info.id, &id_a).unwrap();
        {
            let state = job.lock();
            assert_eq!(state.file.images.len(), 1);
            assert_eq!(state.file.images[0].id, id_b);
            let active = state.active.as_ref().unwrap();
            assert!(!active.waiting.contains(&id_a));
            assert!(active.waiting.contains(&id_b));
            assert_eq!(state.file.run.as_ref().unwrap().total, 1);
        }
        assert!(!upload_a.exists(), "retry must delete uploaded image file");
    }
}
