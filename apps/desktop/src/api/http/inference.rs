//! Inference: a model's weights for local inference (revalidated by ETag), and predictions on the
//! server (a dataset image, or an uploaded file as multipart).

use std::time::Duration;

use crate::api::*;

use super::HttpBackend;
use super::transport::seg;

impl HttpBackend {
    /// [`Backend::model_weights`]'s answer.
    pub(super) fn model_weights_for(&self, id: &str, etag: Option<&str>) -> ApiResult<Option<Weights>> {
        let url = self.url(&format!("/models/{}/weights?format=f16", seg(id)));
        // The first request of a checkpoint makes its f16 export on the server: allow for it. A
        // revalidation (a copy is already loaded) gives up on a silent server after 15 s, so the
        // copy runs instead of every local prediction waiting on it.
        let headers = if etag.is_some() { 15 } else { 300 };
        let mut req = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(2)))
            .timeout_recv_response(Some(Duration::from_secs(headers)))
            .timeout_global(Some(Duration::from_secs(600)))
            .build()
            .new_agent()
            .get(&url);
        if let Some(e) = etag {
            req = req.header("If-None-Match", e);
        }
        let mut resp = req.call().map_err(|e| self.unreachable(&e))?;
        match resp.status().as_u16() {
            304 => return Ok(None),
            s if s >= 400 => {
                return Err(self.decode::<serde_json::Value>(resp).err().unwrap_or_else(|| {
                    ApiError::new("http_error", format!("folo serve answered HTTP {s}"))
                }));
            }
            _ => {}
        }
        let etag = resp.headers().get("etag").and_then(|v| v.to_str().ok()).unwrap_or_default().to_string();
        // A download cut or stalled midway is the server out of reach (callers holding older
        // weights fall back to them), not a malformed answer.
        let bytes = resp
            .body_mut()
            .with_config()
            .limit(2 << 30)
            .read_to_vec()
            .map_err(|e| self.body_failed(e, ""))?;
        Ok(Some(Weights { bytes, etag }))
    }

    /// [`Backend::predict`]'s answer.
    pub(super) fn predict_for(&self, req: &PredictRequest) -> ApiResult<PredictResponse> {
        match &req.input {
            PredictInput::Dataset { dataset_id, image_id } => {
                #[derive(serde::Serialize)]
                struct JsonPredictBody<'a> {
                    #[serde(flatten)]
                    base: folo_api::PredictRequest,
                    #[serde(skip_serializing_if = "Option::is_none")]
                    per_class_conf: Option<&'a std::collections::BTreeMap<u32, f32>>,
                    #[serde(skip_serializing_if = "std::ops::Not::not")]
                    conf_inclusive: bool,
                }
                let body = JsonPredictBody {
                    base: folo_api::PredictRequest {
                        model_id: req.model_id.clone(),
                        dataset_id: dataset_id.clone(),
                        image_id: image_id.clone(),
                        conf: req.conf,
                        iou: req.iou,
                        max_det: req.max_det,
                    },
                    per_class_conf: req.per_class_conf.as_ref().filter(|m| !m.is_empty()),
                    conf_inclusive: req.conf_inclusive,
                };
                self.post("/predict", &body)
            }
            PredictInput::File { name, bytes } => {
                let boundary = format!("folo-{:016x}", crate::synth::hash(name) ^ bytes.len() as u64);
                let mut body = Vec::with_capacity(bytes.len() + 1024);
                let field = |body: &mut Vec<u8>, k: &str, v: &str| {
                    body.extend_from_slice(
                        format!(
                            "--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n"
                        )
                        .as_bytes(),
                    );
                };
                field(&mut body, "model_id", &req.model_id);
                field(&mut body, "conf", &req.conf.to_string());
                field(&mut body, "iou", &req.iou.to_string());
                field(&mut body, "max_det", &req.max_det.to_string());
                if let Some(map) = req.per_class_conf.as_ref().filter(|m| !m.is_empty())
                    && let Ok(encoded) = serde_json::to_string(map)
                {
                    field(&mut body, "per_class_conf", &encoded);
                }
                if req.conf_inclusive {
                    field(&mut body, "conf_inclusive", "true");
                }
                let safe_name = name.replace(['"', '\r', '\n'], "_");
                body.extend_from_slice(
                    format!(
                        "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{safe_name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
                    )
                    .as_bytes(),
                );
                body.extend_from_slice(bytes);
                body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
                let resp = self
                    .agent
                    .post(&self.url("/predict"))
                    .header("Content-Type", &format!("multipart/form-data; boundary={boundary}"))
                    .send(&body[..])
                    .map_err(|e| self.unreachable(&e))?;
                self.decode(resp)
            }
        }
    }
}
