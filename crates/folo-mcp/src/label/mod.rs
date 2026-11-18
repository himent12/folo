//! Agent labeling for the MCP tools and `folo label` (docs/MCP.md and docs/CLI.md, Labeling).
//!
//! [`Labels`] reaches a data dir's labeling jobs with one set of calls: in process (a store's
//! [`Labeler`](folo_server::label::Labeler) as `folo mcp` embeds it, or a bare one as `folo label` opens it), or
//! through a running `folo serve` (`/api/label`, docs/API.md). This file holds those calls, one per endpoint;
//! `ops` builds the tools' operations on them (status, a view of an image, box edits, review), `compact` turns the
//! API's records into the small results of [`crate::types::label`] and boxes into pixels of a view, `view` draws
//! an image's boxes, numbered, and `hints` words each front end's hints.

mod compact;
pub mod hints;
mod ops;
pub mod view;

use std::path::PathBuf;
use std::sync::Arc;

use folo_api::{
    CreateLabelJob, Dataset, ImportLabelImages, LabelAgent, LabelAgentList, LabelDatasetRequest, LabelImage,
    LabelImagePage, LabelImageStatus, LabelImagesAdded, LabelJob, LabelJobList, LabelReviewResult, LogLine,
    ReviewLabelImages, RunLogs, StartLabeling, UpdateLabelImage,
};
use folo_client::encode;
use folo_server::Store;
use folo_server::label::{LabelImageQuery, Labeler};
use folo_server::store::ImageData;

pub use compact::{VIEW_BOXES, frame, from_view, to_view, view_text};
pub use ops::{BoxEdit, CreateFailed, MAX_REVIEW, boxes_changed};

use crate::errors::Error;
use crate::proxy::{HttpStore, store_error};

/// Where labeling calls go. Cheap to clone.
#[derive(Clone)]
pub enum Labels {
    /// A store's labeler, in this process.
    Store(Arc<dyn Store>),
    /// A labeler alone, in this process (`folo label` on a data dir no server owns). It cannot register a
    /// dataset: [`Labels::dataset`] needs a store.
    Local(Arc<Labeler>),
    /// A running `folo serve`.
    Http(Arc<HttpStore>),
}

/// One labeling call's way: a labeler here, or a server.
enum Via<'a> {
    Here(&'a Labeler),
    Http(&'a HttpStore),
}

impl Labels {
    fn via(&self) -> Result<Via<'_>, Error> {
        Ok(match self {
            Labels::Store(s) => Via::Here(s.labeler()?),
            Labels::Local(l) => Via::Here(l),
            Labels::Http(h) => Via::Http(h),
        })
    }

    /// Whether the calls go to a `folo serve` (which reads the image files on its machine).
    pub fn is_remote(&self) -> bool {
        matches!(self, Labels::Http(_))
    }

    /// `GET /api/label/agents`: probes the CLIs (cached by the server for 60 s).
    pub fn agents(&self) -> Result<Vec<LabelAgent>, Error> {
        match self.via()? {
            Via::Here(l) => Ok(l.agents()),
            Via::Http(h) => Ok(get::<LabelAgentList>(h, "/label/agents")?.agents),
        }
    }

    /// `GET /api/label/jobs`: newest first.
    pub fn jobs(&self) -> Result<Vec<LabelJob>, Error> {
        match self.via()? {
            Via::Here(l) => Ok(l.list()),
            Via::Http(h) => Ok(get::<LabelJobList>(h, "/label/jobs")?.jobs),
        }
    }

    /// `GET /api/label/jobs/{id}`.
    pub fn job(&self, id: &str) -> Result<LabelJob, Error> {
        match self.via()? {
            Via::Here(l) => l.get(id),
            Via::Http(h) => get(h, &job_path(id, "")),
        }
    }

    /// `POST /api/label/jobs`.
    pub fn create(&self, req: CreateLabelJob) -> Result<LabelJob, Error> {
        match self.via()? {
            Via::Here(l) => l.create(req),
            Via::Http(h) => h.client().post("/label/jobs", &req).map_err(store_error),
        }
    }

    /// `DELETE /api/label/jobs/{id}`.
    pub fn delete(&self, id: &str) -> Result<(), Error> {
        match self.via()? {
            Via::Here(l) => l.delete(id),
            Via::Http(h) => h.client().delete(&job_path(id, "")).map_err(store_error),
        }
    }

    /// `POST /api/label/jobs/{id}/images` with a folder or an image file on the server's machine.
    pub fn add_path(&self, id: &str, path: &str) -> Result<LabelImagesAdded, Error> {
        match self.via()? {
            Via::Here(l) => l.add_folder(id, path),
            Via::Http(h) => {
                let body = ImportLabelImages { path: path.to_string() };
                h.client().post(&job_path(id, "/images"), &body).map_err(store_error)
            }
        }
    }

    /// `GET /api/label/jobs/{id}/images` (the server caps `limit` at 1000). The `reviewed` and `ids` filters need
    /// a server that lists `label_image_filters`; an older one (which would ignore them) is read page by page and
    /// they are applied here.
    pub fn images(&self, id: &str, q: &LabelImageQuery) -> Result<LabelImagePage, Error> {
        match self.via()? {
            Via::Here(l) => l.query(id, q),
            Via::Http(h) => {
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
        match self.via()? {
            Via::Here(l) => l.image(id, image_id, Some(thumb)),
            Via::Http(h) => {
                let path = format!("{}?thumb={thumb}", image_path(id, image_id));
                let (_, bytes) = h.client().get_bytes(&path).map_err(store_error)?;
                Ok(ImageData { content_type: "image/jpeg", bytes, immutable: false })
            }
        }
    }

    /// `PUT /api/label/jobs/{id}/images/{image_id}`.
    pub fn update_image(&self, id: &str, image_id: &str, req: UpdateLabelImage) -> Result<LabelImage, Error> {
        match self.via()? {
            Via::Here(l) => l.update_image(id, image_id, req),
            Via::Http(h) => h.client().put_json(&image_path(id, image_id), &req).map_err(store_error),
        }
    }

    /// `POST /api/label/jobs/{id}/review` (one save); `None` from a server without `label_revisions`.
    pub fn review_batch(
        &self,
        id: &str,
        req: &ReviewLabelImages,
    ) -> Result<Option<LabelReviewResult>, Error> {
        match self.via()? {
            Via::Here(l) => l.review(id, req).map(Some),
            Via::Http(h) if h.has_feature(folo_api::FEATURE_LABEL_REVISIONS)? => {
                h.client().post(&job_path(id, "/review"), req).map(Some).map_err(store_error)
            }
            Via::Http(_) => Ok(None),
        }
    }

    /// `POST /api/label/jobs/{id}/start`.
    pub fn start(&self, id: &str, req: StartLabeling) -> Result<LabelJob, Error> {
        match self.via()? {
            Via::Here(l) => l.start(id, req),
            Via::Http(h) => h.client().post(&job_path(id, "/start"), &req).map_err(store_error),
        }
    }

    /// `POST /api/label/jobs/{id}/stop`.
    pub fn stop(&self, id: &str) -> Result<LabelJob, Error> {
        match self.via()? {
            Via::Here(l) => l.stop(id),
            Via::Http(h) => h.client().post_empty(&job_path(id, "/stop")).map_err(store_error),
        }
    }

    /// `GET /api/label/jobs/{id}/logs`: the last `limit` lines, oldest first.
    pub fn logs(&self, id: &str, limit: usize) -> Result<Vec<LogLine>, Error> {
        match self.via()? {
            Via::Here(l) => l.logs(id, limit),
            Via::Http(h) => {
                Ok(get::<RunLogs>(h, &format!("{}?limit={}", job_path(id, "/logs"), limit.max(1)))?.lines)
            }
        }
    }

    /// `POST /api/label/jobs/{id}/dataset`: writes and registers the dataset (not on [`Labels::Local`]).
    pub fn dataset(&self, id: &str, req: LabelDatasetRequest) -> Result<Dataset, Error> {
        match self {
            Labels::Store(s) => s.label_dataset(id, req),
            Labels::Local(_) => Err(Error::internal("a dataset is registered through a store")),
            Labels::Http(h) => h.client().post(&job_path(id, "/dataset"), &req).map_err(store_error),
        }
    }

    /// The files of every image of job `id`, in process only (`None` through a server, which reads them itself):
    /// what `--confine` checks before a tool reads them.
    pub fn files(&self, id: &str) -> Result<Option<Vec<PathBuf>>, Error> {
        match self.via()? {
            Via::Here(l) => Ok(Some(l.job_images(id, false)?.images.into_iter().map(|i| i.path).collect())),
            Via::Http(_) => Ok(None),
        }
    }
}

/// Whether `p` is an absolute path on some machine (`/…`, `\\…`, `C:/…`, `C:\…`): a path a server takes as
/// given, whatever the OS of the caller. `\photos` is not: on Windows it is on the current drive, which is the
/// caller's to know, not the server's.
pub fn looks_absolute(p: &str) -> bool {
    let b = p.as_bytes();
    p.starts_with('/')
        || p.starts_with(r"\\")
        || (b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b[2] == b'/' || b[2] == b'\\'))
}

/// `/label/jobs/{id}/images?offset=…&limit=…[&status=…]`.
fn images_path(id: &str, status: Option<LabelImageStatus>, offset: u64, limit: u64) -> String {
    let mut path = format!("{}?offset={offset}&limit={limit}", job_path(id, "/images"));
    if let Some(s) = status {
        path.push_str(&format!("&status={}", s.as_str()));
    }
    path
}

/// Images per request of [`filter_here`] (small in tests, so they cross page boundaries).
#[cfg(not(test))]
const FALLBACK_PAGE: u64 = 1000;
#[cfg(test)]
const FALLBACK_PAGE: u64 = 2;

/// [`Labels::images`] on a server without the `reviewed` and `ids` filters: the images of `q.status`, read
/// page by page and filtered as they come, keeping only the page asked for (and stopping once every id asked
/// for is found).
fn filter_here(h: &HttpStore, id: &str, q: &LabelImageQuery) -> Result<LabelImagePage, Error> {
    let (mut read, mut total, mut found) = (0u64, 0u64, 0usize);
    let mut images = Vec::new();
    loop {
        let page: LabelImagePage = get(h, &images_path(id, q.status, read, FALLBACK_PAGE))?;
        let n = page.images.len() as u64;
        read += n;
        for i in page.images {
            let keep = q.reviewed.is_none_or(|r| i.reviewed == r)
                && q.ids.as_ref().is_none_or(|ids| ids.contains(&i.id));
            if keep {
                if total >= q.offset && (images.len() as u64) < q.limit {
                    images.push(i);
                }
                total += 1;
                found += 1;
            }
        }
        let all_found = q.ids.as_ref().is_some_and(|ids| found >= ids.len());
        if n < FALLBACK_PAGE || read >= page.total || all_found {
            break;
        }
    }
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

#[cfg(test)]
mod tests {
    #[test]
    fn absolute_paths_of_either_os_are_recognised() {
        for p in ["/data/x", r"\\nas\x", "C:/x", r"d:\photos"] {
            assert!(super::looks_absolute(p), "{p}");
        }
        for p in ["photos", "./x", "C:x", r"\photos", ""] {
            assert!(!super::looks_absolute(p), "{p}");
        }
    }
}
