//! Agent labeling HTTP endpoints (`/api/label/*`, docs/API.md).

use std::io::BufReader;
use std::sync::mpsc::channel;

use crate::api::*;

use super::HttpBackend;
use super::streams::parse_sse_while;
use super::transport::seg;

impl HttpBackend {
    pub(super) fn create_label_job_for(&self, req: &CreateLabelJob) -> ApiResult<LabelJob> {
        if req.path.is_some() { self.post_slow("/label/jobs", req) } else { self.post("/label/jobs", req) }
    }

    pub(super) fn delete_label_job_for(&self, id: &str) -> ApiResult<()> {
        self.delete(&format!("/label/jobs/{}", seg(id)))
    }

    pub(super) fn import_label_images_for(
        &self,
        id: &str,
        req: &ImportLabelImages,
    ) -> ApiResult<LabelImagesAdded> {
        self.post_slow(&format!("/label/jobs/{}/images", seg(id)), req)
    }

    pub(super) fn label_images_for(&self, id: &str, offset: u32, limit: u32) -> ApiResult<LabelImagePage> {
        let want = if limit == 0 { 100 } else { limit } as u64;
        let mut images = Vec::new();
        let mut total;
        let mut cur = offset as u64;
        loop {
            let chunk = (want - images.len() as u64).min(1000);
            let path = format!("/label/jobs/{}/images?offset={cur}&limit={chunk}", seg(id));
            let page: LabelImagePage = self.get(&path)?;
            total = page.total;
            let n = page.images.len() as u64;
            images.extend(page.images);
            cur += n;
            if n == 0 || images.len() as u64 >= want || cur >= total {
                break;
            }
        }
        Ok(LabelImagePage { total, images })
    }

    pub(super) fn label_image_for(
        &self,
        job_id: &str,
        image_id: &str,
        thumb: Option<u32>,
    ) -> ApiResult<ImageData> {
        let mut path = format!("/label/jobs/{}/images/{}", seg(job_id), seg(image_id));
        if let Some(t) = thumb {
            path.push_str(&format!("?thumb={t}"));
        }
        let mut resp = self.agent.get(&self.url(&path)).call().map_err(|e| self.unreachable(&e))?;
        if resp.status().as_u16() >= 400 {
            return Err(self
                .decode::<serde_json::Value>(resp)
                .err()
                .unwrap_or_else(|| ApiError::new("http_error", "label image request failed")));
        }
        let bytes = resp
            .body_mut()
            .with_config()
            .limit(64 << 20)
            .read_to_vec()
            .map_err(|e| self.body_failed(e, ""))?;
        Ok(ImageData::Encoded(bytes))
    }

    pub(super) fn update_label_image_for(
        &self,
        job_id: &str,
        image_id: &str,
        req: &UpdateLabelImage,
    ) -> ApiResult<LabelImage> {
        self.put(&format!("/label/jobs/{}/images/{}", seg(job_id), seg(image_id)), req)
    }

    pub(super) fn delete_label_image_for(&self, job_id: &str, image_id: &str) -> ApiResult<()> {
        self.delete(&format!("/label/jobs/{}/images/{}", seg(job_id), seg(image_id)))
    }

    pub(super) fn start_labeling_for(&self, id: &str, req: &StartLabeling) -> ApiResult<LabelJob> {
        self.post(&format!("/label/jobs/{}/start", seg(id)), req)
    }

    pub(super) fn stop_labeling_for(&self, id: &str) -> ApiResult<LabelJob> {
        self.post_empty(&format!("/label/jobs/{}/stop", seg(id)))
    }

    pub(super) fn label_logs_for(&self, id: &str) -> ApiResult<RunLogs> {
        self.get(&format!("/label/jobs/{}/logs?limit=500", seg(id)))
    }

    pub(super) fn label_events_for(&self, id: &str) -> ApiResult<LabelSubscription> {
        let url = self.url(&format!("/label/jobs/{}/events", seg(id)));
        let resp = self
            .stream_agent
            .get(&url)
            .header("Accept", "text/event-stream")
            .call()
            .map_err(|e| self.unreachable(&e))?;
        if resp.status().as_u16() >= 400 {
            return Err(self
                .decode::<serde_json::Value>(resp)
                .err()
                .unwrap_or_else(|| ApiError::new("http_error", "labeling event stream failed")));
        }
        let (tx, rx) = channel();
        let (sub, cancel) = LabelSubscription::new(rx);
        let reader = BufReader::new(resp.into_body().into_reader());
        std::thread::spawn(move || {
            parse_sse_while(
                reader,
                || !cancel.load(std::sync::atomic::Ordering::Relaxed),
                |name, data| match LabelEvent::from_sse(name, data) {
                    Some(Ok(ev)) => tx.send(ev).is_ok(),
                    _ => true,
                },
            );
        });
        Ok(sub)
    }

    pub(super) fn create_label_dataset_for(&self, id: &str, req: &LabelDatasetRequest) -> ApiResult<Dataset> {
        self.post_slow(&format!("/label/jobs/{}/dataset", seg(id)), req)
    }
}
