//! `/api/predict`: predictions from an upload or a dataset image.

use axum::Json;
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use folo_api::PredictRequest;

use crate::AppState;
use crate::error::{ApiJson, Error, Result};
use crate::store::{PredictInput, PredictSource};

use super::blocking;

// ---- inference ---------------------------------------------------------------------------------------------------

pub(super) fn field_err(name: &str, e: impl std::fmt::Display) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "invalid_body", format!("field `{name}`: {e}"))
}

async fn parse_multipart(mut mp: Multipart) -> Result<PredictInput> {
    let mut image = None;
    let (mut model_id, mut dataset_id, mut image_id) = (None, None, None);
    let (mut conf, mut iou, mut max_det) =
        (folo_api::default_conf(), folo_api::default_iou(), folo_api::default_max_det());
    while let Some(field) = mp.next_field().await.map_err(|e| Error::multipart("multipart", e))? {
        let name = field.name().unwrap_or("").to_string();
        if name == "image" {
            image = Some(field.bytes().await.map_err(|e| Error::multipart("image", e))?.to_vec());
            continue;
        }
        let text = field.text().await.map_err(|e| Error::multipart(&name, e))?;
        let text = text.trim().to_string();
        match name.as_str() {
            "model_id" => model_id = Some(text),
            "dataset_id" => dataset_id = Some(text),
            "image_id" => image_id = Some(text),
            "conf" => conf = text.parse().map_err(|e| field_err("conf", e))?,
            "iou" => iou = text.parse().map_err(|e| field_err("iou", e))?,
            "max_det" => max_det = text.parse().map_err(|e| field_err("max_det", e))?,
            _ => {}
        }
    }
    let model_id = model_id.ok_or_else(|| field_err("model_id", "missing"))?;
    let source = match (image, dataset_id, image_id) {
        (Some(bytes), _, _) if !bytes.is_empty() => PredictSource::Upload(bytes),
        (_, Some(dataset_id), Some(image_id)) => PredictSource::Dataset { dataset_id, image_id },
        _ => return Err(field_err("image", "missing or empty (send a file, or dataset_id + image_id)")),
    };
    Ok(PredictInput { model_id, source, conf, iou, max_det })
}

fn check_thresholds(p: &PredictInput) -> Result<()> {
    if !(0.0..=1.0).contains(&p.conf) {
        return Err(field_err("conf", "must be in 0..=1"));
    }
    if !(0.0..=1.0).contains(&p.iou) {
        return Err(field_err("iou", "must be in 0..=1"));
    }
    if p.max_det == 0 {
        return Err(field_err("max_det", "must be at least 1"));
    }
    Ok(())
}

/// True for a `multipart/form-data` request (the endpoints that take uploads also take JSON).
pub(super) fn is_multipart(req: &Request) -> bool {
    let ct = req.headers().get(header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("");
    ct.to_ascii_lowercase().starts_with("multipart/form-data")
}

pub(super) async fn predict(State(st): State<AppState>, req: Request) -> Result<Response> {
    let input = if is_multipart(&req) {
        let mp = Multipart::from_request(req, &st).await?;
        parse_multipart(mp).await?
    } else {
        let ApiJson(r) = ApiJson::<PredictRequest>::from_request(req, &st).await?;
        PredictInput {
            model_id: r.model_id,
            source: PredictSource::Dataset { dataset_id: r.dataset_id, image_id: r.image_id },
            conf: r.conf,
            iou: r.iou,
            max_det: r.max_det,
        }
    };
    check_thresholds(&input)?;
    Ok(Json(blocking(&st, move |s| s.predict(input)).await?).into_response())
}

pub(super) async fn predict_stream() -> Error {
    Error::not_implemented("streaming inference over WebSocket")
        .hint("use POST /api/predict per frame for now")
}
