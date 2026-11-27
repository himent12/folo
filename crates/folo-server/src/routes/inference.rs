//! `/api/predict`: predictions from an upload or a dataset image.

use axum::Json;
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use folo_api::PredictRequest;

use crate::AppState;
use crate::error::{ApiJson, Error, Result};
use crate::store::{
    PredictInput, PredictOptions, PredictSource, TileConfig, TtaConfig, check_predict_options,
};

use super::blocking;

// ---- inference ---------------------------------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
struct PredictJsonBody {
    #[serde(flatten)]
    base: PredictRequest,
    #[serde(default)]
    tile: Option<serde_json::Value>,
    #[serde(default)]
    tta: Option<TtaConfig>,
}

/// Deserializes an optional `tile` JSON value into [`TileConfig`], inheriting `default_max_det` when the payload
/// omits `"max_det"` so top-level `max_det` applies consistently across tiled and standard predictions.
fn parse_tile_value(raw: Option<serde_json::Value>, default_max_det: u32) -> Result<Option<TileConfig>> {
    let Some(v) = raw else { return Ok(None) };
    let explicit_max_det = v.as_object().is_some_and(|m| m.contains_key("max_det"));
    let mut cfg: TileConfig = serde_json::from_value(v).map_err(|e| field_err("tile", e))?;
    if !explicit_max_det {
        cfg.max_det = default_max_det as usize;
    }
    Ok(Some(cfg))
}

pub(super) fn field_err(name: &str, e: impl std::fmt::Display) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "invalid_body", format!("field `{name}`: {e}"))
}

async fn parse_multipart(mut mp: Multipart) -> Result<PredictInput> {
    let mut image = None;
    let (mut model_id, mut dataset_id, mut image_id) = (None, None, None);
    let (mut conf, mut iou, mut max_det) =
        (folo_api::default_conf(), folo_api::default_iou(), folo_api::default_max_det());
    let (mut tile_raw, mut tta): (Option<serde_json::Value>, Option<TtaConfig>) = (None, None);
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
            "tile" if !text.is_empty() => {
                tile_raw = Some(serde_json::from_str(&text).map_err(|e| field_err("tile", e))?);
            }
            "tta" if !text.is_empty() => {
                tta = Some(serde_json::from_str(&text).map_err(|e| field_err("tta", e))?);
            }
            _ => {}
        }
    }
    let model_id =
        model_id.filter(|m| !m.is_empty()).ok_or_else(|| field_err("model_id", "missing or empty"))?;
    let source = resolve_predict_source(image, dataset_id, image_id)?;
    let tile = parse_tile_value(tile_raw, max_det)?;
    Ok(PredictInput { model_id, source, conf, iou, max_det, options: PredictOptions { tta, tile } })
}

/// Validates and selects the single image source of a `POST /api/predict` request (multipart or JSON).
fn resolve_predict_source(
    image: Option<Vec<u8>>,
    dataset_id: Option<String>,
    image_id: Option<String>,
) -> Result<PredictSource> {
    match (image, dataset_id, image_id) {
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => Err(field_err(
            "image",
            "cannot be combined with dataset_id or image_id (send either a file or dataset_id + image_id)",
        )),
        (Some(bytes), None, None) if !bytes.is_empty() => Ok(PredictSource::Upload(bytes)),
        (Some(_), None, None) => Err(field_err("image", "must not be empty")),
        (None, Some(ds), Some(im)) if !ds.is_empty() && !im.is_empty() => {
            Ok(PredictSource::Dataset { dataset_id: ds, image_id: im })
        }
        (None, Some(_), _) | (None, _, Some(_)) => Err(field_err(
            "dataset_id",
            "dataset_id and image_id must both be non-empty when predicting on a dataset image",
        )),
        (None, None, None) => {
            Err(field_err("image", "missing or empty (send a file, or dataset_id + image_id)"))
        }
    }
}

fn parse_json_predict(body: PredictJsonBody) -> Result<PredictInput> {
    let r = body.base;
    if r.model_id.trim().is_empty() {
        return Err(field_err("model_id", "missing or empty"));
    }
    if r.dataset_id.trim().is_empty() || r.image_id.trim().is_empty() {
        return Err(field_err(
            "dataset_id",
            "dataset_id and image_id must both be non-empty when predicting on a dataset image",
        ));
    }
    let tile = parse_tile_value(body.tile, r.max_det)?;
    Ok(PredictInput {
        model_id: r.model_id,
        source: PredictSource::Dataset { dataset_id: r.dataset_id, image_id: r.image_id },
        conf: r.conf,
        iou: r.iou,
        max_det: r.max_det,
        options: PredictOptions { tta: body.tta, tile },
    })
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
    check_predict_options(&p.options)?;
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
        let ApiJson(r) = ApiJson::<PredictJsonBody>::from_request(req, &st).await?;
        parse_json_predict(r)?
    };
    check_thresholds(&input)?;
    Ok(Json(blocking(&st, move |s| s.predict(input)).await?).into_response())
}

pub(super) async fn predict_stream() -> Error {
    Error::not_implemented("streaming inference over WebSocket")
        .hint("use POST /api/predict per frame for now")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_and_json_predict_reject_blank_conflicting_empty_and_partial_sources() {
        assert!(matches!(
            resolve_predict_source(Some(vec![1, 2, 3]), None, None).unwrap(),
            PredictSource::Upload(b) if b == [1, 2, 3]
        ));
        assert!(matches!(
            resolve_predict_source(None, Some("ds_1".into()), Some("0".into())).unwrap(),
            PredictSource::Dataset { dataset_id, image_id } if dataset_id == "ds_1" && image_id == "0"
        ));

        for (image, ds, im, expected) in [
            (Some(vec![1]), Some("ds_1".into()), Some("0".into()), "cannot be combined"),
            (Some(vec![]), Some("ds_1".into()), Some("0".into()), "cannot be combined"),
            (Some(vec![1]), Some("ds_1".into()), None, "cannot be combined"),
            (Some(vec![]), None, None, "must not be empty"),
            (None, Some("ds_1".into()), None, "must both be non-empty"),
            (None, None, Some("0".into()), "must both be non-empty"),
            (None, Some("".into()), Some("0".into()), "must both be non-empty"),
            (None, Some("ds_1".into()), Some("".into()), "must both be non-empty"),
            (None, None, None, "missing or empty"),
        ] {
            let err = resolve_predict_source(image, ds, im).unwrap_err();
            assert_eq!((err.status, err.body.code.as_str()), (StatusCode::BAD_REQUEST, "invalid_body"));
            assert!(err.body.message.contains(expected), "{err}");
        }

        // JSON `POST /api/predict` rejects blank `model_id`, `dataset_id`, and `image_id` with `400 invalid_body`
        // identically to multipart instead of falling through to `404 *_not_found`.
        for (model_id, dataset_id, image_id) in [
            ("", "ds_1", "0"),
            ("   ", "ds_1", "0"),
            ("m_1", "", "0"),
            ("m_1", "   ", "0"),
            ("m_1", "ds_1", ""),
            ("m_1", "ds_1", "   "),
        ] {
            let req = PredictJsonBody {
                base: PredictRequest {
                    model_id: model_id.into(),
                    dataset_id: dataset_id.into(),
                    image_id: image_id.into(),
                    conf: 0.25,
                    iou: 0.7,
                    max_det: 100,
                },
                tile: None,
                tta: None,
            };
            let err = parse_json_predict(req).unwrap_err();
            assert_eq!(
                (err.status, err.body.code.as_str()),
                (StatusCode::BAD_REQUEST, "invalid_body"),
                "{model_id:?}/{dataset_id:?}/{image_id:?}"
            );
        }

        // Non-blank padded IDs keep their exact spelling so JSON HTTP matches embedded MCP lookup semantics.
        let padded = parse_json_predict(PredictJsonBody {
            base: PredictRequest {
                model_id: " m_1 ".into(),
                dataset_id: " ds_1 ".into(),
                image_id: " 0 ".into(),
                conf: 0.25,
                iou: 0.7,
                max_det: 100,
            },
            tile: None,
            tta: None,
        })
        .unwrap();
        assert_eq!(padded.model_id, " m_1 ");
        assert!(matches!(
            padded.source,
            PredictSource::Dataset { dataset_id, image_id } if dataset_id == " ds_1 " && image_id == " 0 "
        ));

        // Omitting `tile.max_det` inherits top-level `max_det`; workload check caps excessive tile grids.
        let inherited = parse_tile_value(Some(serde_json::json!({ "tile_width": 64, "tile_height": 64 })), 5)
            .unwrap()
            .unwrap();
        assert_eq!(inherited.max_det, 5);
        let explicit = parse_tile_value(
            Some(serde_json::json!({ "tile_width": 64, "tile_height": 64, "max_det": 12 })),
            5,
        )
        .unwrap()
        .unwrap();
        assert_eq!(explicit.max_det, 12);
        let workload_err = crate::store::check_predict_workload(
            &PredictOptions { tile: Some(TileConfig::new(8)), tta: None },
            640,
            Some((1024, 1024)),
        )
        .unwrap_err();
        assert_eq!(
            (workload_err.status, workload_err.body.code.as_str()),
            (StatusCode::BAD_REQUEST, "invalid_config")
        );
    }
}
