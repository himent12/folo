//! Exporting a job's labeled images as a YOLO dataset, and handing its images to benchmarks.

use axum::http::StatusCode;
use folo_api::{Dataset, LabelDatasetRequest, LabelImageStatus, LogLevel};

use crate::error::{Error, Result};
use crate::util::{claim_dir, slug};

use super::job::Exporting;
use super::validate::check_name;
use super::{JobImage, JobImages, Labeler, WrittenDataset, dataset, invalid, io_error};

impl Labeler {
    /// Writes the job's labeled images as a YOLO dataset into a new folder under `<data dir>/datasets`
    /// (the first half of `POST /api/label/jobs/{id}/dataset`; the store registers it).
    pub fn write_dataset(&self, id: &str, req: &LabelDatasetRequest) -> Result<WrittenDataset> {
        if !(0.0..=0.5).contains(&req.val_fraction) {
            return Err(invalid("val_fraction must be in 0..=0.5"));
        }
        let export = Exporting::begin(self.job(id)?)?;
        let job = &export.0;
        let (name, classes, items) = {
            let state = job.lock();
            let name = match req.name.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
                Some(name) => check_name(name)?,
                None => state.file.name.clone(),
            };
            let items: Vec<dataset::Item> = state
                .file
                .images
                .iter()
                .filter(|r| r.status == LabelImageStatus::Labeled)
                .map(|r| dataset::Item {
                    source: job.image_path(r),
                    file: r.file.clone(),
                    sha256: r.sha256.clone(),
                    width: r.width,
                    height: r.height,
                    boxes: r.boxes.clone(),
                })
                .collect();
            (name, state.file.classes.clone(), items)
        };
        if items.is_empty() {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_dataset",
                format!("labeling job '{id}' has no labeled image yet"),
            )
            .hint("label images first: POST /api/label/jobs/{id}/start, or draw boxes by hand"));
        }
        // Images added from a folder are read in place: a folder moved or renamed since leaves them behind. Said as
        // that, before any folder is made, rather than as a failure to write the dataset.
        let missing: Vec<&dataset::Item> = items.iter().filter(|i| !i.source.is_file()).collect();
        if let Some(first) = missing.first() {
            let what = match missing.len() {
                1 => "1 labeled image is".to_string(),
                n => format!("{n} labeled images are"),
            };
            return Err(Error::new(
                StatusCode::NOT_FOUND,
                "image_not_found",
                format!("{what} no longer on disk, such as {}", first.source.display()),
            )
            .hint("put the folder back where it was, or remove those images from the job"));
        }
        let path = claim_dir(&self.datasets, &slug(&name, "labeled"))
            .map_err(|e| io_error("creating a folder in", &self.datasets, e))?;
        if let Err(e) =
            dataset::write(&path, &classes, &items, req.val_fraction, req.seed, dataset::PROVENANCE)
        {
            let _ = std::fs::remove_dir_all(&path);
            return Err(match e {
                dataset::WriteError::Missing(src) => Error::new(
                    StatusCode::NOT_FOUND,
                    "image_not_found",
                    format!("1 labeled image is no longer on disk, such as {}", src.display()),
                )
                .hint("put the folder back where it was, or remove those images from the job"),
                dataset::WriteError::Unreadable { file, path, reason } => Error::new(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    "image_decode_failed",
                    format!("labeled image '{file}' ({}) cannot be decoded: {reason}", path.display()),
                )
                .hint("restore the original image file, or remove the corrupt image from the job"),
                dataset::WriteError::DimensionsChanged { file, path, expected, actual } => Error::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_dataset",
                    format!(
                        "labeled image '{file}' ({}) changed size on disk: expected {}x{}, found {}x{}",
                        path.display(),
                        expected.0,
                        expected.1,
                        actual.0,
                        actual.1
                    ),
                )
                .hint("restore the original image, or remove it from the job, add the new image and label it again"),
                dataset::WriteError::ContentChanged { file, path } => Error::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_dataset",
                    format!(
                        "labeled image '{file}' ({}) changed on disk since it was added to the job",
                        path.display()
                    ),
                )
                .hint("restore the original image, or remove it from the job, add the new image and label it again"),
                dataset::WriteError::Io(e) => io_error("writing the dataset", &path, e),
            });
        }
        Ok(WrittenDataset { name, path, export })
    }

    /// Remembers the registered `dataset` on the job `written` was made from (the second half of
    /// `POST /api/label/jobs/{id}/dataset`), and lets the job go. The dataset exists either way: a job file that
    /// cannot be written is logged, not an error of the request.
    pub fn set_dataset(&self, written: WrittenDataset, dataset: &Dataset) {
        let job = &written.export.0;
        let mut state = job.lock();
        state.file.dataset_id = Some(dataset.id.clone());
        let message = format!(
            "dataset '{}' made: {} train and {} val images",
            dataset.name, dataset.splits.train, dataset.splits.val
        );
        job.log(&mut state, LogLevel::Info, message);
        if let Err(e) = job.save(&state) {
            tracing::warn!(job = %job.id, "{e}");
        }
        job.emit_job(&state);
    }

    /// The images of job `id` with their files and boxes, in the order they were added: every image, or only the
    /// `labeled` ones (with the boxes they are labeled with). What a benchmark (its truth) or a self-improvement
    /// loop (its pool, whose boxes are never read) is made from.
    pub fn job_images(&self, id: &str, labeled_only: bool) -> Result<JobImages> {
        let job = self.job(id)?;
        let state = job.live()?;
        let images = state
            .file
            .images
            .iter()
            .filter(|r| !labeled_only || r.status == LabelImageStatus::Labeled)
            .map(|r| JobImage {
                file: r.file.clone(),
                path: job.image_path(r),
                sha256: r.sha256.clone(),
                width: r.width,
                height: r.height,
                boxes: if r.status == LabelImageStatus::Labeled { r.boxes.clone() } else { Vec::new() },
            })
            .collect();
        Ok(JobImages { name: state.file.name.clone(), classes: state.file.classes.clone(), images })
    }
}
