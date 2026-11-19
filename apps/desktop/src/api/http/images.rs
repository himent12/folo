//! Image pages and image files: a dataset's images, and a benchmark's under a condition.

use crate::api::*;

use super::HttpBackend;
use super::transport::seg;

impl HttpBackend {
    /// [`Backend::images`]'s answer.
    pub(super) fn images_for(&self, id: &str, q: &ImageQuery) -> ApiResult<ImagePage> {
        // The server caps a page at 1000; fetch pages until the requested limit is met.
        let want = if q.limit == 0 { 100 } else { q.limit } as u64;
        let mut images = Vec::new();
        let mut total;
        let mut offset = q.offset as u64;
        loop {
            let limit = (want - images.len() as u64).min(1000);
            let mut path = format!("/datasets/{}/images?offset={offset}&limit={limit}", seg(id));
            if let Some(s) = q.split {
                path.push_str(&format!("&split={}", s.as_str()));
            }
            if let Some(c) = q.class {
                path.push_str(&format!("&class={c}"));
            }
            let page: ImagePage = self.get(&path)?;
            total = page.total;
            let n = page.images.len() as u64;
            images.extend(page.images);
            offset += n;
            if n == 0 || images.len() as u64 >= want || offset >= total {
                break;
            }
        }
        Ok(ImagePage { total, images })
    }

    /// [`Backend::image`]'s answer.
    pub(super) fn image_for(
        &self,
        dataset_id: &str,
        image_id: &str,
        thumb: Option<u32>,
    ) -> ApiResult<ImageData> {
        let mut path = format!("/datasets/{}/images/{}", seg(dataset_id), seg(image_id));
        if let Some(t) = thumb {
            path.push_str(&format!("?thumb={t}"));
        }
        let mut resp = self.agent.get(&self.url(&path)).call().map_err(|e| self.unreachable(&e))?;
        if resp.status().as_u16() >= 400 {
            return Err(self
                .decode::<serde_json::Value>(resp)
                .err()
                .unwrap_or_else(|| ApiError::new("http_error", "image request failed")));
        }
        let bytes =
            resp.body_mut().with_config().limit(64 << 20).read_to_vec().map_err(|e| self.body_failed(e))?;
        Ok(ImageData::Encoded(bytes))
    }
    /// [`Backend::bench_image`]'s answer.
    pub(super) fn bench_image_for(
        &self,
        id: &str,
        image_id: &str,
        condition: Option<(&str, u8)>,
        thumb: Option<u32>,
    ) -> ApiResult<ImageData> {
        let mut path = format!("/bench/benchmarks/{}/images/{}", seg(id), seg(image_id));
        let mut q = Vec::new();
        if let Some((c, s)) = condition {
            q.push(format!("condition={}", seg(c)));
            q.push(format!("severity={s}"));
        }
        if let Some(t) = thumb {
            q.push(format!("thumb={t}"));
        }
        if !q.is_empty() {
            path.push('?');
            path.push_str(&q.join("&"));
        }
        let mut resp = self.agent.get(&self.url(&path)).call().map_err(|e| self.unreachable(&e))?;
        if resp.status().as_u16() >= 400 {
            return Err(self
                .decode::<serde_json::Value>(resp)
                .err()
                .unwrap_or_else(|| ApiError::new("http_error", "benchmark image request failed")));
        }
        let bytes =
            resp.body_mut().with_config().limit(64 << 20).read_to_vec().map_err(|e| self.body_failed(e))?;
        Ok(ImageData::Encoded(bytes))
    }
}
