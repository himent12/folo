//! Agent labeling for the MCP tools and `folo label` (docs/MCP.md and docs/CLI.md, Labeling).
//!
//! [`Labels`] reaches a data dir's labeling jobs with one set of calls, either in process (the store's
//! [`Labeler`](folo_server::label::Labeler), as `folo mcp` embedded and `folo label` without a server) or through
//! a running `folo serve` (`/api/label`, docs/API.md). This file holds those calls, one per endpoint; `ops` builds
//! the tools' operations on them (status, a view of an image, box edits, review), `compact` turns the API's
//! records into the small results of [`crate::types::label`], and `view` draws an image's boxes, numbered.

mod compact;
mod ops;
pub mod view;

use std::path::PathBuf;
use std::sync::Arc;

use folo_api::{
    CreateLabelJob, Dataset, ImportLabelImages, LabelAgent, LabelAgentList, LabelDatasetRequest, LabelImage,
    LabelImagePage, LabelImageStatus, LabelImagesAdded, LabelJob, LabelJobList, LogLine, RunLogs,
    StartLabeling, UpdateLabelImage,
};
use folo_client::encode;
use folo_server::Store;
use folo_server::label::LabelImageQuery;
use folo_server::store::ImageData;

pub use compact::{from_rel, to_rel, view_text};
pub use ops::BoxEdit;

use crate::errors::Error;
use crate::proxy::{HttpStore, store_error};

/// Where labeling calls go. Cheap to clone.
#[derive(Clone)]
pub enum Labels {
    /// The store's labeler, in this process.
    Store(Arc<dyn Store>),
    /// A running `folo serve`.
    Http(Arc<HttpStore>),
}

impl Labels {
    /// `GET /api/label/agents`: probes the CLIs (cached by the server for 60 s).
    pub fn agents(&self) -> Result<Vec<LabelAgent>, Error> {
        match self {
            Labels::Store(s) => Ok(s.labeler()?.agents()),
            Labels::Http(h) => Ok(get::<LabelAgentList>(h, "/label/agents")?.agents),
        }
    }

    /// `GET /api/label/jobs`: newest first.
    pub fn jobs(&self) -> Result<Vec<LabelJob>, Error> {
        match self {
            Labels::Store(s) => Ok(s.labeler()?.list()),
            Labels::Http(h) => Ok(get::<LabelJobList>(h, "/label/jobs")?.jobs),
        }
    }

    /// `GET /api/label/jobs/{id}`.
    pub fn job(&self, id: &str) -> Result<LabelJob, Error> {
        match self {
            Labels::Store(s) => s.labeler()?.get(id),
            Labels::Http(h) => get(h, &job_path(id, "")),
        }
    }

    /// `POST /api/label/jobs`.
    pub fn create(&self, req: CreateLabelJob) -> Result<LabelJob, Error> {
        match self {
            Labels::Store(s) => s.labeler()?.create(req),
            Labels::Http(h) => h.client().post("/label/jobs", &req).map_err(store_error),
        }
    }

    /// `DELETE /api/label/jobs/{id}`.
    pub fn delete(&self, id: &str) -> Result<(), Error> {
        match self {
            Labels::Store(s) => s.labeler()?.delete(id),
            Labels::Http(h) => h.client().delete(&job_path(id, "")).map_err(store_error),
        }
    }

    /// `POST /api/label/jobs/{id}/images` with a folder or an image file on the server's machine.
    pub fn add_path(&self, id: &str, path: &str) -> Result<LabelImagesAdded, Error> {
        match self {
            Labels::Store(s) => s.labeler()?.add_folder(id, path),
            Labels::Http(h) => {
                let body = ImportLabelImages { path: path.to_string() };
                h.client().post(&job_path(id, "/images"), &body).map_err(store_error)
            }
        }
    }

    /// `GET /api/label/jobs/{id}/images` (the server caps `limit` at 1000). The `reviewed` and `ids` filters need
    /// a server that lists `label_image_filters`; an older one (which would ignore them) is asked for every image
    /// of the status, and they are applied here.
    pub fn images(&self, id: &str, q: &LabelImageQuery) -> Result<LabelImagePage, Error> {
        match self {
            Labels::Store(s) => s.labeler()?.query(id, q),
            Labels::Http(h) => {
                let filtered = q.reviewed.is_some() || q.ids.is_some();
                if filtered && !h.has_feature(folo_api::FEATURE_LABEL_IMAGE_FILTERS)? {
                    return filter_here(h, id, q);
                }
                let mut path = images_path(id, q.status, q.offset, q.limit);
                if let Some(r) = q.reviewed {
                    path.push_str(&format!("&reviewed={r}"));
                }
                if let Some(ids) = &q.ids {
                    let ids: Vec<String> = ids.iter().map(|i| encode(i)).collect();
                    path.push_str(&format!("&ids={}", ids.join(",")));
                }
                get(h, &path)
            }
        }
    }

    /// `GET /api/label/jobs/{id}/images/{image_id}?thumb=…`: a JPEG whose long side is at most `thumb`.
    pub fn image(&self, id: &str, image_id: &str, thumb: u32) -> Result<ImageData, Error> {
        match self {
            Labels::Store(s) => s.labeler()?.image(id, image_id, Some(thumb)),
            Labels::Http(h) => {
                let path = format!("{}?thumb={thumb}", image_path(id, image_id));
                let (_, bytes) = h.client().get_bytes(&path).map_err(store_error)?;
                Ok(ImageData { content_type: "image/jpeg", bytes, immutable: false })
            }
        }
    }

    /// `PUT /api/label/jobs/{id}/images/{image_id}`.
    pub fn update_image(&self, id: &str, image_id: &str, req: UpdateLabelImage) -> Result<LabelImage, Error> {
        match self {
            Labels::Store(s) => s.labeler()?.update_image(id, image_id, req),
            Labels::Http(h) => h.client().put_json(&image_path(id, image_id), &req).map_err(store_error),
        }
    }

    /// `POST /api/label/jobs/{id}/start`.
    pub fn start(&self, id: &str, req: StartLabeling) -> Result<LabelJob, Error> {
        match self {
            Labels::Store(s) => s.labeler()?.start(id, req),
            Labels::Http(h) => h.client().post(&job_path(id, "/start"), &req).map_err(store_error),
        }
    }

    /// `POST /api/label/jobs/{id}/stop`.
    pub fn stop(&self, id: &str) -> Result<LabelJob, Error> {
        match self {
            Labels::Store(s) => s.labeler()?.stop(id),
            Labels::Http(h) => h.client().post_empty(&job_path(id, "/stop")).map_err(store_error),
        }
    }

    /// `GET /api/label/jobs/{id}/logs`: the last `limit` lines, oldest first.
    pub fn logs(&self, id: &str, limit: usize) -> Result<Vec<LogLine>, Error> {
        match self {
            Labels::Store(s) => s.labeler()?.logs(id, limit),
            Labels::Http(h) => {
                Ok(get::<RunLogs>(h, &format!("{}?limit={}", job_path(id, "/logs"), limit.max(1)))?.lines)
            }
        }
    }

    /// `POST /api/label/jobs/{id}/dataset`: writes and registers the dataset.
    pub fn dataset(&self, id: &str, req: LabelDatasetRequest) -> Result<Dataset, Error> {
        match self {
            Labels::Store(s) => s.label_dataset(id, req),
            Labels::Http(h) => h.client().post(&job_path(id, "/dataset"), &req).map_err(store_error),
        }
    }

    /// The files of every image of job `id`, in process only (`None` through a server, which reads them itself):
    /// what `--confine` checks before a run or an export reads them.
    pub fn files(&self, id: &str) -> Result<Option<Vec<PathBuf>>, Error> {
        match self {
            Labels::Store(s) => {
                Ok(Some(s.labeler()?.job_images(id, false)?.images.into_iter().map(|i| i.path).collect()))
            }
            Labels::Http(_) => Ok(None),
        }
    }
}

/// `/label/jobs/{id}/images?offset=…&limit=…[&status=…]`.
fn images_path(id: &str, status: Option<LabelImageStatus>, offset: u64, limit: u64) -> String {
    let mut path = format!("{}?offset={offset}&limit={limit}", job_path(id, "/images"));
    if let Some(s) = status {
        path.push_str(&format!("&status={}", s.as_str()));
    }
    path
}

/// [`Labels::images`] on a server without the `reviewed` and `ids` filters: every image of `q.status`, 1000 per
/// request, filtered and paged here.
fn filter_here(h: &HttpStore, id: &str, q: &LabelImageQuery) -> Result<LabelImagePage, Error> {
    const PAGE: u64 = 1000;
    let mut all = Vec::new();
    loop {
        let page: LabelImagePage = get(h, &images_path(id, q.status, all.len() as u64, PAGE))?;
        let last = page.images.len() < PAGE as usize;
        all.extend(page.images);
        if last || all.len() as u64 >= page.total {
            break;
        }
    }
    all.retain(|i| {
        q.reviewed.is_none_or(|r| i.reviewed == r) && q.ids.as_ref().is_none_or(|ids| ids.contains(&i.id))
    });
    let total = all.len() as u64;
    let images = all.into_iter().skip(q.offset as usize).take(q.limit as usize).collect();
    Ok(LabelImagePage { total, images })
}

fn get<T: serde::de::DeserializeOwned>(h: &HttpStore, path: &str) -> Result<T, Error> {
    h.client().get(path).map_err(store_error)
}

/// `/label/jobs/{id}{rest}`.
fn job_path(id: &str, rest: &str) -> String {
    format!("/label/jobs/{}{rest}", encode(id))
}

fn image_path(id: &str, image_id: &str) -> String {
    format!("/label/jobs/{}/images/{}", encode(id), encode(image_id))
}
