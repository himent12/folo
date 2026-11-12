//! A job's images: adding uploads and folders, listing and serving them, editing their boxes, removing them.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use folo_api::{
    LabelEvent, LabelImage, LabelImagePage, LabelImageStatus, LabelImagesAdded, LabelSource, RemovedImage,
    SkippedImage, UpdateLabelImage,
};

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
        // Written and probed before the lock is taken; named for good under it.
        let (mut staged, mut skipped) = (Vec::new(), Vec::new());
        for upload in uploads {
            let skip = |reason: &str| SkippedImage { file: upload.name.clone(), reason: reason.into() };
            if !folo_data::is_image_path(Path::new(&upload.name)) {
                skipped.push(skip("not a JPEG, PNG, BMP or WebP file"));
                continue;
            }
            let tmp =
                dir.join(format!(".upload-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
            let probed = std::fs::write(&tmp, &upload.bytes)
                .map_err(|e| format!("could not be stored: {e}"))
                .and_then(|()| folo_data::probe_size(&tmp).map_err(|_| "not a readable image".to_string()));
            match probed {
                Ok(size) => staged.push(Candidate {
                    name: upload.name,
                    source: Source::Upload { tmp, hash: fnv1a(&upload.bytes) },
                    size,
                }),
                Err(reason) => {
                    let _ = std::fs::remove_file(&tmp);
                    skipped.push(skip(&reason));
                }
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
        let files = walkdir::WalkDir::new(folder)
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
            .map(|e| e.into_path());
        let (mut staged, mut skipped, mut over) = (Vec::new(), Vec::new(), 0usize);
        let mut found = 0usize;
        for path in files {
            found += 1;
            let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
            if known.contains(&path) {
                skipped.push(SkippedImage { file: name, reason: "the job already has this file".into() });
            } else if staged.len() >= room {
                over += 1;
            } else {
                match folo_data::probe_size(&path) {
                    Ok(size) => staged.push(Candidate { name, source: Source::Folder(path), size }),
                    Err(_) => {
                        skipped.push(SkippedImage { file: name, reason: "not a readable image".into() })
                    }
                }
            }
        }
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
            let Candidate { name, source, size: (width, height) } = candidate;
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

    /// `GET /api/label/jobs/{id}/images`.
    pub fn images(
        &self,
        id: &str,
        status: Option<LabelImageStatus>,
        offset: u64,
        limit: u64,
    ) -> Result<LabelImagePage> {
        let job = self.job(id)?;
        let state = job.live()?;
        let matching = state.file.images.iter().filter(|r| status.is_none_or(|s| s == state.status(r)));
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
        let boxes = match &req.boxes {
            Some(boxes) => Some(check_boxes(boxes, classes, (record.width, record.height))?),
            None if req.reviewed.is_none() => return Err(invalid("give `boxes`, `reviewed` or both")),
            // Only labels can be right: an image without them is saved with its boxes (none: a background image).
            None if req.reviewed == Some(true) && record.status != LabelImageStatus::Labeled => {
                return Err(invalid(format!(
                    "image '{image_id}' has no labels to mark reviewed: save its boxes instead"
                )));
            }
            None => None,
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

    /// `DELETE /api/label/jobs/{id}/images/{image_id}`: an upload's file goes too, a folder image's never.
    pub fn delete_image(&self, id: &str, image_id: &str) -> Result<()> {
        let job = self.job(id)?;
        let mut state = job.live()?;
        let record = state.record(image_id)?.clone();
        state.claim(image_id)?;
        state.file.images.retain(|r| r.id != image_id);
        job.save(&state)?;
        if record.external.is_none() {
            let _ = std::fs::remove_file(job.image_path(&record));
        }
        job.emit(LabelEvent::Removed(RemovedImage { id: record.id }));
        job.emit_job(&state);
        Ok(())
    }
}

/// An image about to be added to a job.
struct Candidate {
    name: String,
    source: Source,
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
