//! Models, prediction (JSON and multipart), evaluation curves and validation.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::api::*;
use folo_server::{Options, RealStore, Server};
use folo_test_support::TestDir;
use serde_json::json;

use super::harness::{err, get, mock, ok, post_json, send};

#[tokio::test]
async fn models() {
    let (server, _) = mock();
    let app = server.router();
    let list: ModelList = ok(&app, get("/api/models")).await;
    assert!((5..=6).contains(&list.models.len()));
    let created: Vec<_> = list.models.iter().map(|m| m.created.clone()).collect();
    let mut sorted = created.clone();
    sorted.sort_by(|a, b| b.cmp(a));
    assert_eq!(created, sorted, "newest first");
    let m = &list.models[0];
    let one: Model = ok(&app, get(&format!("/api/models/{}", m.id))).await;
    assert_eq!(&one, m);
    let ex: ExportResult =
        ok(&app, post_json(&format!("/api/models/{}/export", m.id), json!({ "format": "onnx" }))).await;
    assert!(ex.path.ends_with(".onnx") && ex.bytes > 0);
    err(
        &app,
        post_json(&format!("/api/models/{}/export", m.id), json!({ "format": "tflite" })),
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
    )
    .await;
    err(&app, get("/api/models/m_none"), StatusCode::NOT_FOUND, "model_not_found").await;

    // Head and training settings: shapes-s is an o2o model, r6-n was trained with AMP, its f16 export has no
    // `folo.train`; each matches its run's config.
    let by_id = |id: &str| list.models.iter().find(|m| m.id == id).unwrap();
    assert_eq!(by_id("m_31ab").head, Head::O2o);
    assert_eq!(by_id("m_53d1").trained_with, Some(TrainedWith { amp: true, imgsz: 640, epochs: 80 }));
    assert_eq!((by_id("m_53d2").head, by_id("m_53d2").trained_with), (Head::O2m, None));
    for m in list.models.iter().filter(|m| m.trained_with.is_some()) {
        let run: Run = ok(&app, get(&format!("/api/runs/{}", m.run_id.as_deref().unwrap()))).await;
        let t = m.trained_with.unwrap();
        assert_eq!(
            (m.head, t.amp, t.imgsz, t.epochs),
            (run.config.head, run.config.amp, run.config.imgsz, run.epochs)
        );
    }
    // Stored dtype and fusing: run checkpoints are unfused f32, the f16 export is fused f16; every model has both.
    assert_eq!((by_id("m_53d2").precision, by_id("m_53d2").fused), (Some(ModelPrecision::F16), Some(true)));
    for m in list.models.iter().filter(|m| m.id != "m_53d2") {
        assert_eq!((m.precision, m.fused), (Some(ModelPrecision::F32), Some(false)), "{}", m.id);
    }
    let raw: serde_json::Value = ok(&app, get("/api/models/m_53d2")).await;
    assert_eq!((&raw["precision"], &raw["fused"]), (&json!("f16"), &json!(true)));
}

#[tokio::test]
async fn predict_json_and_multipart() {
    let (server, _) = mock();
    let app = server.router();
    let models: ModelList = ok(&app, get("/api/models")).await;
    let model = models.models.iter().find(|m| m.name == "traffic-s-best").unwrap();
    let ds: DatasetList = ok(&app, get("/api/datasets")).await;
    let traffic = ds.datasets.iter().find(|d| d.name == "traffic-cams").unwrap();

    let req = json!({ "model_id": model.id, "dataset_id": traffic.id, "image_id": "000010", "conf": 0.3 });
    let p: PredictResponse = ok(&app, post_json("/api/predict", req.clone())).await;
    assert_eq!((p.image.width, p.image.height), (1280, 720));
    assert!(p.boxes.iter().all(|b| b.score >= 0.3 && b.x1 <= b.x2 && b.y1 <= b.y2));
    assert!(p.boxes.windows(2).all(|w| w[0].score >= w[1].score));
    assert!(p.timing.total_ms > 0.0);
    let again: PredictResponse = ok(&app, post_json("/api/predict", req)).await;
    assert_eq!(again, p, "deterministic");

    // Multipart upload of a JPEG we fetched from the server.
    let (_, _, jpg) = send(&app, get(&format!("/api/datasets/{}/images/000003", traffic.id))).await;
    let multipart = |bytes: &[u8], max_det: &str| {
        let boundary = "XfoloBoundary";
        let mut body = Vec::new();
        for (name, value) in [("model_id", model.id.as_str()), ("conf", "0.1"), ("max_det", max_det)] {
            body.extend_from_slice(
                format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                    .as_bytes(),
            );
        }
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(bytes);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        Request::post("/api/predict")
            .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
            .body(Body::from(body))
            .unwrap()
    };
    let up: PredictResponse = ok(&app, multipart(&jpg, "300")).await;
    assert_eq!((up.image.width, up.image.height), (1280, 720));
    assert!(!up.boxes.is_empty());
    let up2: PredictResponse = ok(&app, multipart(&jpg, "1")).await;
    assert_eq!(up2.boxes.len(), 1);
    assert_eq!(up2.boxes[0], up.boxes[0], "same bytes, same answer");
    err(&app, multipart(b"not an image", "300"), StatusCode::UNPROCESSABLE_ENTITY, "image_decode_failed")
        .await;

    err(
        &app,
        post_json(
            "/api/predict",
            json!({ "model_id": "m_none", "dataset_id": traffic.id, "image_id": "000001" }),
        ),
        StatusCode::NOT_FOUND,
        "model_not_found",
    )
    .await;
    err(
        &app,
        post_json(
            "/api/predict",
            json!({ "model_id": model.id, "dataset_id": traffic.id, "image_id": "000001", "conf": 2 }),
        ),
        StatusCode::BAD_REQUEST,
        "invalid_body",
    )
    .await;
    err(&app, get("/api/predict/stream"), StatusCode::NOT_IMPLEMENTED, "not_implemented").await;
}

#[tokio::test]
async fn predictions_stay_within_the_model_classes() {
    let (server, _) = mock();
    let app = server.router();
    // A shapes model (circle / square / triangle) on traffic images: it may only answer with its own classes.
    let model: Model = ok(&app, get("/api/models/m_31aa")).await;
    assert_eq!(model.classes.len(), 3);
    for i in 0..20 {
        let req = json!({ "model_id": "m_31aa", "dataset_id": "ds_7c2f", "image_id": format!("{i:06}"), "conf": 0.0 });
        let p: PredictResponse = ok(&app, post_json("/api/predict", req)).await;
        for b in &p.boxes {
            assert!((b.class as usize) < model.classes.len(), "{b:?}");
            assert_eq!(b.name, model.classes[b.class as usize]);
        }
    }
    // A matching model keeps the dataset's classes.
    let req = json!({ "model_id": "m_53d1", "dataset_id": "ds_3b9d", "image_id": "000700", "conf": 0.0 });
    let p: PredictResponse = ok(&app, post_json("/api/predict", req)).await;
    assert!(p.boxes.iter().any(|b| b.name == "body"));
}

#[tokio::test]
async fn model_eval_curves_and_confusion() {
    let (server, _) = mock();
    let app = server.router();
    let models: ModelList = ok(&app, get("/api/models")).await;
    for m in &models.models {
        let ev: ModelEval = ok(&app, get(&format!("/api/models/{}/eval", m.id))).await;
        let n = m.classes.len();
        assert!(ev.images > 0, "{}", m.id);
        assert_eq!((ev.pr.len(), ev.per_class.len()), (n, n));
        for (c, curve) in ev.pr.iter().enumerate() {
            assert_eq!(curve.class as usize, c);
            assert_eq!((curve.recall.len(), curve.precision.len()), (101, 101));
            assert_eq!((curve.recall[0], curve.recall[100]), (0.0, 1.0));
            assert!(curve.precision.iter().all(|p| (0.0..=1.0).contains(p)));
            assert!(curve.precision.windows(2).all(|w| w[0] >= w[1]), "interpolated precision never rises");
            let mean = curve.precision.iter().sum::<f32>() / 101.0;
            assert!((curve.ap50 - mean).abs() < 1e-3);
            assert_eq!(ev.per_class[c].map50, curve.ap50);
            assert!(curve.ap50 > 0.2 && curve.ap50 < 1.0, "{} class {c}: {}", m.id, curve.ap50);
        }
        assert_eq!(ev.confusion.len(), n + 1);
        assert!(ev.confusion.iter().all(|row| row.len() == n + 1));
        assert_eq!(ev.confusion[n][n], 0);
        let truth: u32 = ev.confusion[..n].iter().flatten().sum();
        let diag: u32 = (0..n).map(|c| ev.confusion[c][c]).sum();
        assert!(truth > 0 && diag as f32 > 0.5 * truth as f32, "{}: mostly diagonal", m.id);
        for c in 0..n {
            let row = &ev.confusion[c];
            assert!((0..n).filter(|&j| j != c).all(|j| row[c] >= row[j]), "{} row {c}: {row:?}", m.id);
        }
        let again: ModelEval = ok(&app, get(&format!("/api/models/{}/eval", m.id))).await;
        assert_eq!(again, ev, "deterministic");
    }
    err(&app, get("/api/models/m_none/eval"), StatusCode::NOT_FOUND, "model_not_found").await;
}

#[tokio::test]
async fn val_endpoint() {
    let (server, _) = mock();
    let app = server.router();
    let req = json!({ "model_id": "m_31aa", "dataset_id": "ds_5a1e" });
    let r: ValReport = ok(&app, post_json("/api/val", req.clone())).await;
    assert_eq!((r.split, r.images, r.imgsz), (Split::Val, 200, 640));
    assert_eq!(r.val.per_class.len(), 3);
    assert!(r.val.map50 > 0.0 && r.val.map50 <= 1.0 && r.val.map50_95 <= r.val.map50);
    assert!(r.timing.total_s > 0.0);
    // Deterministic.
    let again: ValReport = ok(&app, post_json("/api/val", req)).await;
    assert_eq!(again, r);
    let train: ValReport = ok(
        &app,
        post_json("/api/val", json!({ "model_id": "m_31aa", "dataset_id": "ds_5a1e", "split": "train" })),
    )
    .await;
    assert_eq!(train.images, 800);
    let req = json!({ "model_id": "m_31aa", "dataset_id": "ds_5a1e", "split": "test" });
    err(&app, post_json("/api/val", req), StatusCode::BAD_REQUEST, "invalid_dataset").await;
    let req = json!({ "model_id": "nope", "dataset_id": "ds_5a1e" });
    err(&app, post_json("/api/val", req), StatusCode::NOT_FOUND, "model_not_found").await;
    let req = json!({ "model_id": "m_31aa", "dataset_id": "ds_5a1e", "conf": 2.0 });
    err(&app, post_json("/api/val", req), StatusCode::BAD_REQUEST, "invalid_body").await;

    let dir = TestDir::new("val-real");
    let real = Server::new(Arc::new(RealStore::open(dir.to_path_buf())), Options::default()).router();
    let req = json!({ "model_id": "m_1", "dataset_id": "ds_1" });
    err(&real, post_json("/api/val", req), StatusCode::NOT_FOUND, "model_not_found").await;
}
