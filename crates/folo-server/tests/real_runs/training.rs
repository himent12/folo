//! A whole run: train over HTTP, evaluate, export, predict, fine-tune and reload after a restart.

use std::path::Path;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use folo_server::Store;
use folo_server::api::*;
use folo_test_support::TestDir;
use serde_json::{Value, json};

use super::harness::{Events, app, count, dataset, delete, err, get, ok, post_json, send, store, tiny};

#[tokio::test(flavor = "multi_thread")]
async fn train_evaluate_predict_and_reload() {
    let t = TestDir::new("runs-train");
    let home = t.join("home");
    let s = store(&home);
    let app = app(&s);
    let ds = dataset(&app, "tiny", 6).await;
    assert_eq!(ds.splits, Splits { train: 5, val: 1, test: 0 });

    // Validation against the dataset and the engine's limits.
    err(
        &app,
        post_json("/api/runs", json!({ "dataset_id": "ds_none" })),
        StatusCode::NOT_FOUND,
        "dataset_not_found",
    )
    .await;
    let e = err(
        &app,
        post_json("/api/runs", json!({ "dataset_id": ds.id, "imgsz": 48 })),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    assert!(e.message.contains("imgsz"), "{}", e.message);
    let e = err(
        &app,
        post_json("/api/runs", json!({ "dataset_id": ds.id, "optimizer": "lion" })),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    assert!(e.message.contains("optimizer"), "{}", e.message);
    err(
        &app,
        post_json("/api/runs", json!({ "dataset_id": ds.id, "epochs": 0 })),
        StatusCode::BAD_REQUEST,
        "invalid_config",
    )
    .await;
    let w = json!({ "dataset_id": ds.id, "weights": "m_none" });
    err(&app, post_json("/api/runs", w), StatusCode::NOT_FOUND, "model_not_found").await;
    err(
        &app,
        post_json("/api/runs", json!({ "epochs": 2 })),
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
    )
    .await;
    assert!(ok::<RunList>(&app, get("/api/runs")).await.runs.is_empty(), "rejected configs leave no run");

    // A 2-epoch run, watched over SSE from the start.
    let (status, b) = send(&app, post_json("/api/runs", tiny(&ds, 2))).await;
    assert_eq!(status, StatusCode::CREATED, "{}", String::from_utf8_lossy(&b));
    let run: Run = serde_json::from_slice(&b).unwrap();
    assert_eq!(run.status, RunStatus::Queued);
    assert_eq!(run.name, "tiny-n");
    assert_eq!((run.epochs, run.config.imgsz), (2, 64));
    assert!(run.device.contains("cpu"), "{}", run.device);
    let mut sse = Events::open(&app, &run.id).await;
    let first = sse.next().await;
    assert_eq!(first.0, "status");
    let events = sse.until_status(&["completed", "failed"]).await;
    let last = &events.last().unwrap().1;
    assert_eq!(last["status"], "completed", "{last}: {events:?}");
    assert_eq!(count(&events, "epoch"), 2);
    assert!(count(&events, "step") >= 2 && count(&events, "log") >= 1);
    // Every validated epoch says which confidence its precision / recall are at, not only the best one's.
    let conf_ok = |c: &Value| c.as_f64().is_some_and(|c| (0.0..=1.0).contains(&c));
    for (_, v) in events.iter().filter(|(n, _)| n == "epoch") {
        assert!(conf_ok(&v["val"]["best_f1_conf"]), "SSE epoch {v}");
    }
    // Steps and the trainer's statuses carry the run's length (2 batches of up to 4 out of 5 images, 2 epochs).
    for (name, v) in events.iter().filter(|(n, v)| n == "step" || (n == "status" && v["status"] != "queued"))
    {
        assert_eq!((v["steps_per_epoch"].as_u64(), v["epochs"].as_u64()), (Some(2), Some(2)), "{name} {v}");
    }

    let done: Run = ok(&app, get(&format!("/api/runs/{}", run.id))).await;
    assert_eq!(done.status, RunStatus::Completed);
    assert_eq!((done.epoch, done.steps_per_epoch), (2, 2));
    assert_eq!(done.step, 4);
    assert!(done.started.is_some() && done.finished.is_some() && done.eta_s.is_none());
    let best = done.best.unwrap();
    let model_id = done.model_id.clone().unwrap();
    assert_eq!(model_id, format!("m_{}", run.id.trim_start_matches("run_")));
    for f in
        ["run.json", "config.json", "metrics.jsonl", "log.jsonl", "weights/best.folo", "weights/last.folo"]
    {
        assert!(home.join("runs").join(&run.id).join(f).is_file(), "{f}");
    }

    // History: every step, both epochs, the log.
    let m: RunMetrics = ok(&app, get(&format!("/api/runs/{}/metrics", run.id))).await;
    assert_eq!(m.steps.iter().map(|s| s.step).collect::<Vec<_>>(), [1, 2, 3, 4]);
    assert_eq!(m.epochs.iter().map(|e| e.epoch).collect::<Vec<_>>(), [1, 2]);
    assert!(m.epochs.iter().all(|e| e.val.best_f1_conf.is_some()), "/metrics {:?}", m.epochs);
    let jsonl = std::fs::read_to_string(home.join("runs").join(&run.id).join("metrics.jsonl")).unwrap();
    let lines: Vec<Value> = jsonl.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let epoch_lines: Vec<&Value> = lines.iter().filter(|v| v["event"] == "epoch").collect();
    assert_eq!(epoch_lines.len(), 2);
    for (v, e) in epoch_lines.iter().zip(&m.epochs) {
        assert!(conf_ok(&v["val"]["best_f1_conf"]), "metrics.jsonl {v}");
        assert_eq!(v["val"]["best_f1_conf"].as_f64().map(|c| c as f32), e.val.best_f1_conf);
    }
    let best_epoch = &m.epochs[best.epoch as usize - 1];
    assert_eq!(best_epoch.val.map50_95, best.map50_95);
    let thin: RunMetrics = ok(&app, get(&format!("/api/runs/{}/metrics?every=3", run.id))).await;
    assert_eq!(thin.steps.iter().map(|s| s.step).collect::<Vec<_>>(), [3, 4]);
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", run.id))).await;
    assert_eq!(logs.lines[0].message, "queued");
    assert!(logs.lines.iter().any(|l| l.message.contains("finished 2 epochs")), "{:?}", logs.lines);
    let two: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs?limit=2", run.id))).await;
    assert_eq!(two.lines.len(), 2);
    assert_eq!(two.lines[1], *logs.lines.last().unwrap());

    // Validation samples: the validation images, with dataset image ids.
    let samples: RunSamples = ok(&app, get(&format!("/api/runs/{}/samples", run.id))).await;
    assert_eq!(samples.epoch, 2);
    assert_eq!(samples.samples.len(), 1);
    let val: ImagePage = ok(&app, get(&format!("/api/datasets/{}/images?split=val", ds.id))).await;
    for (s, im) in samples.samples.iter().zip(&val.images) {
        assert_eq!((s.image_id.as_str(), s.dataset_id.as_str()), (im.id.as_str(), ds.id.as_str()));
        assert_eq!((s.width, s.height), (64, 64));
        assert_eq!(s.truth.len(), im.boxes.len());
        assert!(s.pred.iter().all(|p| p.score >= 0.25));
    }
    let e1: RunSamples = ok(&app, get(&format!("/api/runs/{}/samples?epoch=1", run.id))).await;
    assert_eq!(e1.epoch, 1);
    let uri = format!("/api/runs/{}/samples?epoch=3", run.id);
    err(&app, get(&uri), StatusCode::NOT_FOUND, "epoch_not_available").await;

    // Its models: best and last.
    let models: ModelList = ok(&app, get("/api/models")).await;
    let ids: Vec<&str> = models.models.iter().map(|m| m.id.as_str()).collect();
    assert!(
        ids.contains(&model_id.as_str()) && ids.contains(&format!("{model_id}-last").as_str()),
        "{ids:?}"
    );
    let model: Model = ok(&app, get(&format!("/api/models/{model_id}"))).await;
    assert_eq!((model.arch, model.imgsz, model.name.as_str()), (Arch::N, 64, "tiny-n-best"));
    assert_eq!(model.classes, ["circle", "square"]);
    assert_eq!(model.run_id.as_deref(), Some(run.id.as_str()));
    assert!(model.params > 2_000_000 && model.bytes > 0 && model.flops_g > 0.0);
    assert_eq!(model.metrics.unwrap().map50_95, best.map50_95);
    // The best-F1 confidence of the best epoch, through the checkpoint's `folo.metrics`.
    assert!(best_epoch.val.best_f1_conf.is_some());
    assert_eq!(model.metrics.unwrap().best_f1_conf, best_epoch.val.best_f1_conf);
    let resolved = done.resolved.clone().expect("a finished run says what it trained with");
    assert_eq!((resolved.optimizer, resolved.accumulate), (ResolvedOptimizer::Musgd, 16));
    // The head comes from `folo.config`, `trained_with` from the checkpoint's `folo.train`.
    let trained = TrainedWith { amp: false, imgsz: 64, epochs: 2 };
    assert_eq!((model.head, model.trained_with), (Head::O2m, Some(trained)));
    let last: Model = ok(&app, get(&format!("/api/models/{model_id}-last"))).await;
    assert_eq!(last.metrics.unwrap().map50_95, m.epochs[1].val.map50_95);
    assert_eq!((last.head, last.trained_with), (Head::O2m, Some(trained)));
    // Checkpoints are unfused f32 (`folo.dtype`, `folo.fused`).
    for m in [&model, &last] {
        assert_eq!((m.precision, m.fused), (Some(ModelPrecision::F32), Some(false)), "{}", m.id);
    }

    // The models folder reads the same metadata: the checkpoint copied there keeps `trained_with`; an o2o weights
    // file (no `folo.train`) has none.
    let zoo = home.join("models");
    std::fs::create_dir_all(&zoo).unwrap();
    std::fs::copy(home.join("runs").join(&run.id).join("weights/best.folo"), zoo.join("copied.folo"))
        .unwrap();
    let mut config = folo_nn::io::ModelConfig::new("n", 2).unwrap();
    config.head = "o2o".into();
    let mut o2o = folo_vision::FoloModel::with_rng(config, &mut folo_core::rng::Rng::seed(3)).unwrap();
    o2o.set_classes(vec!["circle".into(), "square".into()]).unwrap();
    o2o.save(zoo.join("nms-free.folo")).unwrap();
    // Precision and fusing from the header: an f16 export; files whose `folo.dtype` is missing or unknown, or whose
    // `folo.fused` is unknown, still list with that field left out.
    let weights = std::fs::read(zoo.join("nms-free.folo")).unwrap();
    let mut half = Vec::new();
    let file = folo_nn::io::ModelFile::from_bytes(weights.clone()).unwrap();
    file.export_to(&mut half, folo_nn::io::FloatDType::F16).unwrap();
    std::fs::write(zoo.join("half.folo"), half).unwrap();
    let edited = [
        ("bf16", "folo.dtype", Some("BF16")),
        ("no-dtype", "folo.dtype", None),
        ("odd-fused", "folo.fused", Some("maybe")),
    ];
    for (name, key, value) in edited {
        std::fs::write(zoo.join(format!("{name}.folo")), with_metadata(&weights, key, value)).unwrap();
    }
    let listed: ModelList = ok(&app, get("/api/models")).await;
    let by_name = |n: &str| listed.models.iter().find(|m| m.name == n).unwrap_or_else(|| panic!("{n}"));
    let copied = by_name("copied");
    assert_eq!(
        (copied.head, copied.trained_with, copied.run_id.as_deref()),
        (Head::O2m, Some(trained), None)
    );
    let nms_free = by_name("nms-free");
    assert_eq!((nms_free.head, nms_free.trained_with), (Head::O2o, None));
    let f32 = Some(ModelPrecision::F32);
    for (name, precision, fused) in [
        ("copied", f32, Some(false)),
        ("nms-free", f32, Some(false)),
        ("half", Some(ModelPrecision::F16), Some(true)),
        ("bf16", None, Some(false)),
        ("no-dtype", None, Some(false)),
        ("odd-fused", f32, None),
    ] {
        let m = by_name(name);
        assert_eq!((m.precision, m.fused), (precision, fused), "{name}");
        // Unknown is left out of the JSON, not sent as `null`.
        let raw: Value = ok(&app, get(&format!("/api/models/{}", m.id))).await;
        assert_eq!(raw.get("precision").is_some(), precision.is_some(), "{name}: {raw}");
        assert_eq!(raw.get("fused").is_some(), fused.is_some(), "{name}: {raw}");
    }
    for p in ["copied", "nms-free", "half", "bf16", "no-dtype", "odd-fused"].map(|n| format!("{n}.folo")) {
        std::fs::remove_file(zoo.join(&p)).unwrap();
    }
    err(&app, get("/api/models/m_none"), StatusCode::NOT_FOUND, "model_not_found").await;

    // Its weights: the checkpoint as stored, and the f16 export (cached under cache/weights/).
    let (status, body) = send(&app, get(&format!("/api/models/{model_id}/weights"))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, std::fs::read(home.join("runs").join(&run.id).join("weights/best.folo")).unwrap());
    let (status, f16) = send(&app, get(&format!("/api/models/{model_id}/weights?format=f16"))).await;
    assert_eq!(status, StatusCode::OK);
    let file = folo_nn::io::ModelFile::from_bytes(f16.clone()).unwrap();
    assert_eq!(file.metadata().dtype, folo_nn::io::FloatDType::F16);
    assert!(file.metadata().fused && f16.len() < body.len() / 2, "{} vs {}", f16.len(), body.len());
    let cached = || {
        std::fs::read_dir(home.join("cache/weights"))
            .map(|d| {
                d.flatten()
                    .filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{model_id}.")))
                    .count()
            })
            .unwrap_or(0)
    };
    assert_eq!(cached(), 1);

    // Eval on the validation split (cached), export.
    let eval: ModelEval = ok(&app, get(&format!("/api/models/{model_id}/eval"))).await;
    assert_eq!(eval.images, 1);
    // What was evaluated: the run's dataset, its validation split, and the confusion matrix's thresholds.
    assert_eq!(
        (eval.dataset_id.as_deref(), eval.dataset_name.as_deref()),
        (Some(ds.id.as_str()), Some("tiny"))
    );
    assert_eq!((eval.split, eval.split_fallback.as_ref()), (Some(Split::Val), None));
    assert_eq!((eval.confusion_conf, eval.confusion_iou), (Some(0.25), Some(0.45)));
    assert_eq!(eval.confusion.len(), 3);
    assert!(eval.confusion.iter().all(|r| r.len() == 3) && eval.confusion[2][2] == 0);
    for c in &eval.pr {
        assert_eq!((c.recall.len(), c.precision.len()), (101, 101));
        assert!(c.precision.windows(2).all(|w| w[0] >= w[1]), "the envelope never rises");
    }
    let again: ModelEval = ok(&app, get(&format!("/api/models/{model_id}/eval"))).await;
    assert_eq!(again, eval);
    assert!(home.join("cache/eval").join(format!("{model_id}.json")).is_file());
    // POST /api/val: the same split and thresholds as /eval give the same numbers; other splits and errors.
    let v: ValReport =
        ok(&app, post_json("/api/val", json!({ "model_id": model_id, "dataset_id": ds.id }))).await;
    assert_eq!((v.split, v.images, v.imgsz), (Split::Val, 1, 64));
    assert_eq!((v.dataset_id.as_str(), v.dataset_name.as_deref()), (ds.id.as_str(), Some("tiny")));
    assert_eq!(v.val.per_class, eval.per_class);
    // Precision and recall come with their confidence threshold, the same in both.
    assert!(eval.best_f1_conf.is_some_and(|c| (0.0..=1.0).contains(&c)), "{:?}", eval.best_f1_conf);
    assert_eq!(v.val.best_f1_conf, eval.best_f1_conf);
    assert!(v.timing.total_s > 0.0 && v.timing.per_image_ms > 0.0);
    let body =
        json!({ "model_id": model_id, "dataset_id": ds.id, "split": "train", "imgsz": 96, "conf": 0.25 });
    let t: ValReport = ok(&app, post_json("/api/val", body)).await;
    assert_eq!((t.split, t.images, t.imgsz), (Split::Train, 5, 96));
    for (body, status, code) in [
        (
            json!({ "model_id": model_id, "dataset_id": ds.id, "split": "test" }),
            StatusCode::BAD_REQUEST,
            "invalid_dataset",
        ),
        (
            json!({ "model_id": model_id, "dataset_id": ds.id, "imgsz": 100 }),
            StatusCode::BAD_REQUEST,
            "invalid_body",
        ),
        (
            json!({ "model_id": model_id, "dataset_id": ds.id, "iou": 1.5 }),
            StatusCode::BAD_REQUEST,
            "invalid_body",
        ),
        (json!({ "model_id": "m_none", "dataset_id": ds.id }), StatusCode::NOT_FOUND, "model_not_found"),
        (
            json!({ "model_id": model_id, "dataset_id": "ds_none" }),
            StatusCode::NOT_FOUND,
            "dataset_not_found",
        ),
    ] {
        err(&app, post_json("/api/val", body), status, code).await;
    }

    let x: ExportResult =
        ok(&app, post_json(&format!("/api/models/{model_id}/export"), json!({ "format": "folo-f16" }))).await;
    assert_eq!(Path::new(&x.path), home.join("exports").join("tiny-n-best-f16.folo"));
    assert_eq!(std::fs::metadata(&x.path).unwrap().len(), x.bytes);
    let x2: ExportResult =
        ok(&app, post_json(&format!("/api/models/{model_id}/export"), json!({ "format": "folo-f16" }))).await;
    assert_ne!(x2.path, x.path, "exports never overwrite");
    let onnx: ExportResult =
        ok(&app, post_json(&format!("/api/models/{model_id}/export"), json!({ "format": "onnx" }))).await;
    assert_eq!(Path::new(&onnx.path), home.join("exports").join("tiny-n-best.onnx"));
    let bytes = std::fs::read(&onnx.path).unwrap();
    assert_eq!(bytes.len() as u64, onnx.bytes);
    // A ModelProto: field 1 (ir_version) = 8 first.
    assert_eq!(&bytes[..2], &[0x08, 8]);

    // Predict: a dataset image (JSON) and an upload (multipart).
    let p = json!({ "model_id": model_id, "dataset_id": ds.id, "image_id": "0", "conf": 0.001 });
    let r: PredictResponse = ok(&app, post_json("/api/predict", p)).await;
    assert_eq!(r.model_id, model_id);
    assert_eq!((r.image.width, r.image.height, r.device.as_str()), (64, 64, "cpu"));
    assert!(!r.boxes.is_empty(), "conf 0.001 keeps low-score boxes");
    assert!(r.boxes.windows(2).all(|w| w[0].score >= w[1].score));
    assert!(r.boxes.iter().all(|b| b.score >= 0.001 && ["circle", "square"].contains(&b.name.as_str())));
    let ti = r.timing;
    assert!(ti.inference_ms > 0.0 && ti.total_ms >= ti.preprocess_ms + ti.inference_ms);
    let png = std::fs::read(home.join("datasets/tiny").join(&val.images[0].file)).unwrap();
    let r2: PredictResponse =
        ok(&app, multipart(&model_id, Some(("image.jpg", png)), &[("conf", "0.001")])).await;
    assert_eq!((r2.image.width, r2.image.height), (64, 64));
    assert!(!r2.boxes.is_empty());
    // The model's first prediction since it was loaded says its timing includes the warm-up; the next one does not.
    assert!(r.cold && !r2.cold, "cold: first {}, second {}", r.cold, r2.cold);
    let bad = multipart(&model_id, Some(("x.jpg", b"not an image".to_vec())), &[]);
    err(&app, bad, StatusCode::UNPROCESSABLE_ENTITY, "image_decode_failed").await;
    let p = json!({ "model_id": "m_none", "dataset_id": ds.id, "image_id": "0" });
    err(&app, post_json("/api/predict", p), StatusCode::NOT_FOUND, "model_not_found").await;

    // Fine-tuning from the run's model.
    let mut ft = tiny(&ds, 1);
    ft["weights"] = json!(model_id);
    ft["name"] = json!("tuned");
    let tuned: Run = ok(&app, post_json("/api/runs", ft)).await;
    let mut sse = Events::open(&app, &tuned.id).await;
    let events = sse.until_status(&["completed", "failed"]).await;
    assert_eq!(events.last().unwrap().1["status"], "completed");
    let logs: RunLogs = ok(&app, get(&format!("/api/runs/{}/logs", tuned.id))).await;
    assert!(logs.lines.iter().any(|l| l.message.starts_with("transferred")), "{:?}", logs.lines);
    let runs: RunList = ok(&app, get("/api/runs")).await;
    assert_eq!(
        runs.runs.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        [&tuned.id, &run.id],
        "newest first"
    );

    // Everything is back after a restart.
    let system: SystemInfo = ok(&app, get("/api/system")).await;
    drop(sse);
    drop(app);
    s.shutdown();
    drop(s);
    let s = store(&home);
    let app = self::app(&s);
    let again: Run = ok(&app, get(&format!("/api/runs/{}", run.id))).await;
    assert_eq!(again, done);
    let m2: RunMetrics = ok(&app, get(&format!("/api/runs/{}/metrics", run.id))).await;
    assert_eq!(m2, m);
    let models2: ModelList = ok(&app, get("/api/models")).await;
    assert_eq!(models2.models.len(), models.models.len() + 2, "the fine-tuned run's two models too");
    let eval2: ModelEval = ok(&app, get(&format!("/api/models/{model_id}/eval"))).await;
    assert_eq!(eval2, eval);
    let p = json!({ "model_id": model_id, "dataset_id": ds.id, "image_id": "0", "conf": 0.001 });
    let r3: PredictResponse = ok(&app, post_json("/api/predict", p)).await;
    assert_eq!(r3.boxes, r.boxes, "same weights, same image, same boxes");
    assert_eq!(ok::<SystemInfo>(&app, get("/api/system")).await.gpus, system.gpus);

    // Deleting a finished run removes it, its folder and its models (their cached f16 weights too).
    let tuned_model = format!("m_{}", tuned.id.trim_start_matches("run_"));
    let (status, _) = send(&app, get(&format!("/api/models/{tuned_model}/weights?format=f16"))).await;
    assert_eq!(status, StatusCode::OK);
    let tuned_cache = || {
        std::fs::read_dir(home.join("cache/weights"))
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with(&format!("{tuned_model}.")))
            .count()
    };
    assert_eq!(tuned_cache(), 1);
    let (status, _) = send(&app, delete(&format!("/api/runs/{}", tuned.id))).await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(!home.join("runs").join(&tuned.id).exists());
    err(&app, get(&format!("/api/runs/{}", tuned.id)), StatusCode::NOT_FOUND, "run_not_found").await;
    let r = json!({ "resume_run_id": tuned.id });
    err(&app, post_json("/api/runs", r), StatusCode::NOT_FOUND, "run_not_found").await;
    assert_eq!(ok::<ModelList>(&app, get("/api/models")).await.models.len(), models.models.len());
    assert_eq!(tuned_cache(), 0);
    let uri = format!("/api/models/{tuned_model}/weights");
    err(&app, get(&uri), StatusCode::NOT_FOUND, "model_not_found").await;

    // A dataset whose validation split is empty now: `/eval` falls back to the training split, and says so.
    let yaml = home.join("datasets/tiny/data.yaml");
    let text = std::fs::read_to_string(&yaml).unwrap();
    std::fs::create_dir_all(home.join("datasets/tiny/images/none")).unwrap();
    std::fs::write(&yaml, text.replace("val: images/val", "val: images/none")).unwrap();
    let fell: ModelEval = ok(&app, get(&format!("/api/models/{model_id}/eval"))).await;
    assert_eq!((fell.split, fell.images), (Some(Split::Train), 5));
    let fb = fell.split_fallback.expect("a fallback is reported");
    assert_eq!((fb.requested, fb.reason.as_str()), (Split::Val, "empty_split"));
    assert!(fb.message.contains("'tiny' has no val images") && fb.message.contains("training-set"), "{fb:?}");
    std::fs::write(&yaml, &text).unwrap();
    let back: ModelEval = ok(&app, get(&format!("/api/models/{model_id}/eval"))).await;
    assert_eq!((back.split, back.split_fallback), (Some(Split::Val), None));

    // A dataset whose classes changed order since training is not evaluated against the old labels.
    std::fs::write(
        &yaml,
        text.replace("\"circle\"", "\"tmp\"")
            .replace("\"square\"", "\"circle\"")
            .replace("\"tmp\"", "\"square\""),
    )
    .unwrap();
    let e =
        err(&app, get(&format!("/api/models/{model_id}/eval")), StatusCode::BAD_REQUEST, "invalid_dataset")
            .await;
    assert!(e.message.contains("same order"), "{}", e.message);
}

/// A `multipart/form-data` predict request.
fn multipart(model_id: &str, file: Option<(&str, Vec<u8>)>, fields: &[(&str, &str)]) -> Request<Body> {
    let boundary = "folo-test-boundary";
    let mut body = Vec::new();
    let mut field = |name: &str, value: &str| {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                .as_bytes(),
        );
    };
    field("model_id", model_id);
    for (k, v) in fields {
        field(k, v);
    }
    if let Some((name, bytes)) = file {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: \
                 application/octet-stream\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(&bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    Request::post("/api/predict")
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={boundary}"))
        .body(Body::from(body))
        .unwrap()
}

/// `bytes` (a `.folo` file) with metadata `key` set to `value` (removed for `None`); the tensor data is kept.
fn with_metadata(bytes: &[u8], key: &str, value: Option<&str>) -> Vec<u8> {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let meta = header["__metadata__"].as_object_mut().unwrap();
    match value {
        Some(v) => meta.insert(key.into(), v.into()),
        None => meta.remove(key),
    };
    let header = serde_json::to_vec(&header).unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&bytes[8 + n..]);
    out
}
