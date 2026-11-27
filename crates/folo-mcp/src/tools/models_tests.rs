use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::Options;
use crate::proxy::HttpStore;
use crate::server::FoloMcp;
use folo_server::Store;

/// `list_models`, `list_runs`, `list_label_jobs`, `list_benchmarks`, `val`, and `predict_batch` validate
/// paging and argument ranges before hitting the store or proxy server, keeping embedded and `--server`
/// error codes identical even when the server is unreachable.
#[tokio::test]
async fn tools_validate_paging_and_args_before_proxy_access() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let hits = Arc::new(AtomicUsize::new(0));
    let hits_bg = Arc::clone(&hits);
    let accept = tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            hits_bg.fetch_add(1, Ordering::SeqCst);
            drop(stream);
        }
    });
    let http = Arc::new(HttpStore::new(&url, None));
    let home = folo_test_support::TestDir::new("mcp-upfront-validation");
    let opts = Options {
        data_dir: home.to_path_buf(),
        read_only: false,
        confine: false,
        device: "cpu".into(),
        published_weights: None,
    };
    let server = FoloMcp::new(http.clone(), Some(http), opts);

    let bad_page = || PageArgs { limit: Some(0), cursor: None };
    let bad_cursor = || PageArgs { limit: Some(10), cursor: Some("bogus".into()) };
    assert_eq!(server.inner.list_models(bad_page()).await.err().unwrap().body.code, "invalid_arguments");
    assert_eq!(server.inner.list_models(bad_cursor()).await.err().unwrap().body.code, "invalid_arguments");
    assert_eq!(
        server
            .inner
            .list_runs(ListRunsArgs { status: None, limit: Some(0), cursor: None })
            .await
            .err()
            .unwrap()
            .body
            .code,
        "invalid_arguments"
    );
    assert_eq!(server.inner.list_label_jobs(bad_page()).await.err().unwrap().body.code, "invalid_arguments");
    assert_eq!(server.inner.list_benchmarks(bad_page()).await.err().unwrap().body.code, "invalid_arguments");

    // `val` rejects bad `imgsz` before touching the proxy.
    let val_err = server
        .inner
        .val(
            ValArgs {
                model_id: "m_1".into(),
                dataset_id: "ds_1".into(),
                split: folo_api::Split::Val,
                imgsz: Some(33),
                conf: 0.001,
                iou: 0.7,
                diagnostics: false,
                worst: None,
                diag_conf: None,
                diag_tp_iou: None,
                diag_bg_iou: None,
            },
            &mut Progress::none(),
        )
        .await
        .err()
        .unwrap();
    assert_eq!(val_err.body.code, "invalid_arguments");

    // `export` rejects conflicting `dynamic` / `dynamic_batch` and non-ONNX axis flags before hitting proxy.
    for (format, dynamic, dynamic_batch) in [
        (folo_api::ExportFormat::Onnx, true, Some(false)),
        (folo_api::ExportFormat::FoloF16, false, Some(false)),
    ] {
        let exp_err = server
            .inner
            .export(ExportArgs {
                model_id: "m_1".into(),
                format,
                decode: false,
                dynamic,
                dynamic_batch,
                dynamic_hw: None,
                half: false,
            })
            .await
            .err()
            .unwrap();
        assert_eq!(exp_err.body.code, "invalid_arguments");
    }

    // `filter_dataset` rejects zero image dimension bounds upfront before hitting proxy.
    let filter_err = server
        .inner
        .filter_dataset(FilterDatasetArgs {
            dataset_id: "ds_1".into(),
            name: "filtered".into(),
            format: None,
            include_classes: None,
            exclude_classes: None,
            normalize: ClassNormalizeMode::Exact,
            compact_classes: false,
            splits: None,
            min_box_area: None,
            max_box_area: None,
            min_box_width_px: None,
            min_box_height_px: None,
            max_box_width_px: None,
            max_box_height_px: None,
            min_width: Some(0),
            min_height: None,
            max_width: None,
            max_height: None,
            drop_empty_images: false,
        })
        .await
        .err()
        .unwrap();
    assert_eq!(filter_err.body.code, "invalid_arguments");

    // `predict_batch` rejects empty paths, both paths+glob, or neither before looking up `model_id`.
    for (paths, glob) in
        [(Some(vec![]), None), (None, None), (Some(vec!["a.jpg".into()]), Some("*.jpg".into()))]
    {
        let err = server
            .inner
            .predict_batch(
                PredictBatchArgs {
                    model_id: "m_missing".into(),
                    paths,
                    glob,
                    conf: 0.25,
                    iou: 0.7,
                    tile: None,
                    tile_overlap: None,
                    tile_full_image: None,
                    tta: false,
                    tta_scales: None,
                    tta_hflip: None,
                    thresholds_path: None,
                    per_class_conf: None,
                },
                &mut Progress::none(),
            )
            .await
            .err()
            .unwrap();
        assert_eq!(err.body.code, "invalid_arguments");
    }

    // `predict` rejects blank `model_id`, empty `image_base64`, blank `dataset_id`/`image_id`, and invalid
    // `tile`/`tta` combinations identically on both embedded and proxied backends before hitting the store.
    let embedded = FoloMcp::new(
        Arc::new(folo_server::MockStore::new()),
        None,
        Options {
            data_dir: home.to_path_buf(),
            read_only: false,
            confine: false,
            device: "cpu".into(),
            published_weights: None,
        },
    );
    for backend in [&server, &embedded] {
        for args in [
            PredictArgs {
                model_id: "   ".into(),
                image_path: None,
                image_base64: Some("AAAA".into()),
                dataset_id: None,
                image_id: None,
                conf: 0.25,
                iou: 0.7,
                max_det: 10,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: false,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: None,
                per_class_conf: None,
            },
            PredictArgs {
                model_id: "m_1".into(),
                image_path: None,
                image_base64: Some("".into()),
                dataset_id: None,
                image_id: None,
                conf: 0.25,
                iou: 0.7,
                max_det: 10,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: false,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: None,
                per_class_conf: None,
            },
            PredictArgs {
                model_id: "m_1".into(),
                image_path: None,
                image_base64: None,
                dataset_id: Some("  ".into()),
                image_id: Some("img_1".into()),
                conf: 0.25,
                iou: 0.7,
                max_det: 10,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: false,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: None,
                per_class_conf: None,
            },
            PredictArgs {
                model_id: "m_1".into(),
                image_path: None,
                image_base64: Some("AAAA".into()),
                dataset_id: None,
                image_id: None,
                conf: 0.25,
                iou: 0.7,
                max_det: 10,
                tile: Some(640),
                tile_overlap: None,
                tile_full_image: None,
                tta: true,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: None,
                per_class_conf: None,
            },
            PredictArgs {
                model_id: "m_1".into(),
                image_path: None,
                image_base64: Some("AAAA".into()),
                dataset_id: None,
                image_id: None,
                conf: 0.25,
                iou: 0.7,
                max_det: 10,
                tile: None,
                tile_overlap: Some(0.2),
                tile_full_image: None,
                tta: false,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: None,
                per_class_conf: None,
            },
            PredictArgs {
                model_id: "m_1".into(),
                image_path: None,
                image_base64: Some("AAAA".into()),
                dataset_id: None,
                image_id: None,
                conf: 0.25,
                iou: 0.7,
                max_det: 10,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: false,
                tta_scales: Some(vec![1.0]),
                tta_hflip: None,
                thresholds_path: None,
                per_class_conf: None,
            },
        ] {
            let err = backend.inner.predict(args).await.err().unwrap();
            assert_eq!(err.body.code, "invalid_arguments");
        }
    }

    assert_eq!(hits.load(Ordering::SeqCst), 0, "invalid arguments must never hit the proxy server");
    accept.abort();
    let _ = accept.await;

    // `predict_batch` records the same `invalid_arguments` failure code for a zero-byte file across both
    // embedded and proxied backends (never hitting the server with an empty multipart `image`).
    let empty_img = home.join("empty.jpg");
    std::fs::write(&empty_img, b"").unwrap();
    let model_id = folo_server::MockStore::new().list_models().unwrap()[0].id.clone();
    let mock_server = folo_server::Server::new(Arc::new(folo_server::MockStore::new()), Default::default());
    let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_url = format!("http://{}", http_listener.local_addr().unwrap());
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let http_task = tokio::spawn(mock_server.serve(http_listener, async move {
        let _ = stop_rx.await;
    }));
    let proxied_live = FoloMcp::new(
        Arc::new(HttpStore::new(&http_url, None)),
        Some(Arc::new(HttpStore::new(&http_url, None))),
        Options {
            data_dir: home.to_path_buf(),
            read_only: false,
            confine: false,
            device: "cpu".into(),
            published_weights: None,
        },
    );
    for backend in [&embedded, &proxied_live] {
        let res = backend
            .inner
            .predict_batch(
                PredictBatchArgs {
                    model_id: model_id.clone(),
                    paths: Some(vec![empty_img.to_string_lossy().into_owned()]),
                    glob: None,
                    conf: 0.25,
                    iou: 0.7,
                    tile: None,
                    tile_overlap: None,
                    tile_full_image: None,
                    tta: false,
                    tta_scales: None,
                    tta_hflip: None,
                    thresholds_path: None,
                    per_class_conf: None,
                },
                &mut Progress::none(),
            )
            .await
            .unwrap();
        let failed = res.value["failed"].as_array().unwrap();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0]["code"], "invalid_arguments");
    }
    let _ = stop_tx.send(());
    let _ = http_task.await;
}

/// Blank or dot-only resource IDs (`""`, `"."`, `".."`) return the same `<resource>_not_found` error across
/// embedded and proxied MCP backends instead of hitting `404 route_not_found` or parent collection routes.
#[tokio::test]
async fn proxied_blank_and_dot_resource_ids_match_embedded_not_found() {
    let home = folo_test_support::TestDir::new("mcp-blank-id-parity");
    let mock_server = folo_server::Server::new(Arc::new(folo_server::MockStore::new()), Default::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(mock_server.serve(listener, async move {
        let _ = stop_rx.await;
    }));
    let embedded = FoloMcp::new(
        Arc::new(folo_server::MockStore::new()),
        None,
        Options {
            data_dir: home.to_path_buf(),
            read_only: false,
            confine: false,
            device: "cpu".into(),
            published_weights: None,
        },
    );
    let http = Arc::new(HttpStore::new(&url, None));
    let proxied = FoloMcp::new(
        http.clone(),
        Some(http),
        Options {
            data_dir: home.to_path_buf(),
            read_only: false,
            confine: false,
            device: "cpu".into(),
            published_weights: None,
        },
    );

    let real_store = Arc::new(folo_server::RealStore::open(home.join("real")).device("cpu"));
    let real_srv = folo_server::Server::new(real_store, Default::default());
    let rl = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let rurl = format!("http://{}", rl.local_addr().unwrap());
    let (rtx, rrx) = tokio::sync::oneshot::channel::<()>();
    let rtask = tokio::spawn(real_srv.serve(rl, async move {
        let _ = rrx.await;
    }));
    let rhttp = Arc::new(HttpStore::new(&rurl, None));
    let benches = crate::bench::Benchmarks::Http(rhttp.clone());
    let labels = crate::label::Labels::Http(rhttp);

    for bad in ["", ".", ".."] {
        for backend in [&embedded, &proxied] {
            assert_eq!(
                backend
                    .inner
                    .get_dataset(DatasetIdArgs { dataset_id: bad.into() })
                    .await
                    .err()
                    .unwrap()
                    .body
                    .code,
                "dataset_not_found",
                "get_dataset({bad:?})"
            );
            assert_eq!(
                backend
                    .inner
                    .run_status(RunStatusArgs { run_id: bad.into(), log_lines: 0 })
                    .await
                    .err()
                    .unwrap()
                    .body
                    .code,
                "run_not_found",
                "run_status({bad:?})"
            );
            assert_eq!(
                backend
                    .inner
                    .export(ExportArgs {
                        model_id: bad.into(),
                        format: folo_api::ExportFormat::Onnx,
                        decode: false,
                        dynamic: false,
                        dynamic_batch: None,
                        dynamic_hw: None,
                        half: false,
                    })
                    .await
                    .err()
                    .unwrap()
                    .body
                    .code,
                "model_not_found",
                "export({bad:?})"
            );
        }
        let b = benches.clone();
        let bad_s = bad.to_string();
        let b_err = tokio::task::spawn_blocking(move || b.get(&bad_s)).await.unwrap().unwrap_err();
        assert_eq!(b_err.body.code, "bench_not_found", "bench.get({bad:?})");
        let l = labels.clone();
        let bad_s = bad.to_string();
        let l_err = tokio::task::spawn_blocking(move || l.job(&bad_s)).await.unwrap().unwrap_err();
        assert_eq!(l_err.body.code, "label_job_not_found", "labels.job({bad:?})");
    }

    let _ = rtx.send(());
    let _ = rtask.await;
    let _ = stop_tx.send(());
    let _ = task.await;
}

/// Proxied `predict` refuses `tile`/`tta` against an older server without `predict_options` (`not_implemented`),
/// forwards both JSON (dataset source) and multipart (upload) options to a current server, and proves neither
/// transport drops options by confirming model-dependent rejections (`invalid_config`).
#[tokio::test]
async fn proxied_predict_gates_and_forwards_tile_and_tta_options() {
    let home = folo_test_support::TestDir::new("mcp-proxy-predict-opts");
    let mock_store: Arc<dyn Store> = Arc::new(folo_server::MockStore::new());
    let model_id = mock_store.list_models().unwrap()[0].id.clone();
    let ds_id = mock_store.list_datasets().unwrap()[0].id.clone();
    let image_id = mock_store
        .list_images(&ds_id, &folo_server::store::ImageQuery { limit: 1, ..Default::default() })
        .unwrap()
        .images[0]
        .id
        .clone();
    let mock_server = folo_server::Server::new(mock_store, Default::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(mock_server.serve(listener, async move {
        let _ = stop_rx.await;
    }));
    let opts = Options {
        data_dir: home.to_path_buf(),
        read_only: false,
        confine: false,
        device: "cpu".into(),
        published_weights: None,
    };

    // 1. Older server without `predict_options`: refused with `not_implemented`.
    let old_http = Arc::new(HttpStore::new(&url, None).assume_features(vec![]));
    let old_mcp = FoloMcp::new(old_http.clone(), Some(old_http), opts.clone());
    let err = old_mcp
        .inner
        .predict(PredictArgs {
            model_id: model_id.clone(),
            image_path: None,
            image_base64: None,
            dataset_id: Some(ds_id.clone()),
            image_id: Some(image_id.clone()),
            conf: 0.25,
            iou: 0.7,
            max_det: 10,
            tile: Some(320),
            tile_overlap: Some(0.2),
            tile_full_image: Some(true),
            tta: false,
            tta_scales: None,
            tta_hflip: None,
            thresholds_path: None,
            per_class_conf: None,
        })
        .await
        .err()
        .unwrap();
    assert_eq!(err.body.code, "not_implemented");

    // 2. Current server announcing `predict_options`: forwards `tile` (JSON) and `tta` (multipart).
    let live_http = Arc::new(HttpStore::new(&url, None));
    let live_mcp = FoloMcp::new(live_http.clone(), Some(live_http), opts);
    let tiled = live_mcp
        .inner
        .predict(PredictArgs {
            model_id: model_id.clone(),
            image_path: None,
            image_base64: None,
            dataset_id: Some(ds_id.clone()),
            image_id: Some(image_id.clone()),
            conf: 0.25,
            iou: 0.7,
            max_det: 10,
            tile: Some(320),
            tile_overlap: Some(0.2),
            tile_full_image: Some(false),
            tta: false,
            tta_scales: None,
            tta_hflip: None,
            thresholds_path: None,
            per_class_conf: None,
        })
        .await
        .unwrap();
    assert_eq!(tiled.value["model_id"], model_id);

    // JSON path reaches the server with `tta` and triggers model-dependent `invalid_config` (scale > 8192).
    let json_bad = live_mcp
        .inner
        .predict(PredictArgs {
            model_id: model_id.clone(),
            image_path: None,
            image_base64: None,
            dataset_id: Some(ds_id),
            image_id: Some(image_id),
            conf: 0.25,
            iou: 0.7,
            max_det: 10,
            tile: None,
            tile_overlap: None,
            tile_full_image: None,
            tta: true,
            tta_scales: Some(vec![200.0]),
            tta_hflip: Some(false),
            thresholds_path: None,
            per_class_conf: None,
        })
        .await
        .err()
        .unwrap();
    assert_eq!(json_bad.body.code, "invalid_config");

    let img = folo_data::Image::new(32, 32, [128, 128, 128]);
    let png = folo_data::encode(&img, folo_data::ImageFormat::Png).unwrap();
    let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
    let tta = live_mcp
        .inner
        .predict(PredictArgs {
            model_id: model_id.clone(),
            image_path: None,
            image_base64: Some(b64.clone()),
            dataset_id: None,
            image_id: None,
            conf: 0.25,
            iou: 0.7,
            max_det: 10,
            tile: None,
            tile_overlap: None,
            tile_full_image: None,
            tta: true,
            tta_scales: Some(vec![1.0]),
            tta_hflip: Some(true),
            thresholds_path: None,
            per_class_conf: None,
        })
        .await
        .unwrap();
    assert_eq!(tta.value["model_id"], model_id);

    // Multipart path forwards `tile` and rejects excessive tile grids (> MAX_PREDICT_TILES).
    let mp_bad = live_mcp
        .inner
        .predict(PredictArgs {
            model_id: model_id.clone(),
            image_path: None,
            image_base64: Some(b64),
            dataset_id: None,
            image_id: None,
            conf: 0.25,
            iou: 0.7,
            max_det: 10,
            tile: Some(1),
            tile_overlap: Some(0.0),
            tile_full_image: Some(false),
            tta: false,
            tta_scales: None,
            tta_hflip: None,
            thresholds_path: None,
            per_class_conf: None,
        })
        .await
        .err()
        .unwrap();
    assert_eq!(mp_bad.body.code, "invalid_config");

    // `predict_batch` rejects model-incompatible `tta_scales` upfront before reading images.
    let batch_bad = live_mcp
        .inner
        .predict_batch(
            PredictBatchArgs {
                model_id: model_id.clone(),
                paths: Some(vec!["does-not-exist.png".into()]),
                glob: None,
                conf: 0.25,
                iou: 0.7,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: true,
                tta_scales: Some(vec![200.0]),
                tta_hflip: None,
                thresholds_path: None,
                per_class_conf: None,
            },
            &mut Progress::none(),
        )
        .await
        .err()
        .unwrap();
    assert_eq!(batch_bad.body.code, "invalid_arguments");

    // Mixed `predict_batch`: one oversized image triggering per-image `invalid_config` (tile limit) is recorded
    // in `failed` without discarding valid images before or after it.
    let small_a = home.join("small_a.png");
    let huge = home.join("huge.png");
    let small_b = home.join("small_b.png");
    let small_png =
        folo_data::encode(&folo_data::Image::new(32, 32, [64, 64, 64]), folo_data::ImageFormat::Png).unwrap();
    let huge_png =
        folo_data::encode(&folo_data::Image::new(640, 640, [64, 64, 64]), folo_data::ImageFormat::Png)
            .unwrap();
    std::fs::write(&small_a, &small_png).unwrap();
    std::fs::write(&huge, &huge_png).unwrap();
    std::fs::write(&small_b, &small_png).unwrap();
    let mixed = live_mcp
        .inner
        .predict_batch(
            PredictBatchArgs {
                model_id,
                paths: Some(vec![
                    small_a.to_string_lossy().into_owned(),
                    huge.to_string_lossy().into_owned(),
                    small_b.to_string_lossy().into_owned(),
                ]),
                glob: None,
                conf: 0.25,
                iou: 0.7,
                tile: Some(32),
                tile_overlap: Some(0.0),
                tile_full_image: Some(false),
                tta: false,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: None,
                per_class_conf: None,
            },
            &mut Progress::none(),
        )
        .await
        .unwrap();
    assert_eq!(mixed.value["processed"], 2, "{}", mixed.value);
    assert_eq!(mixed.value["failed_total"], 1, "{}", mixed.value);
    assert_eq!(mixed.value["failed"][0]["code"], "invalid_config", "{}", mixed.value);
    let uri = mixed.value["details_uri"].as_str().unwrap();
    let batch_id = uri.strip_prefix("folo://predictions/").unwrap();
    let parsed = live_mcp.inner.batches.lock().unwrap().get(batch_id).unwrap();
    let imgs = parsed["images"].as_array().unwrap();
    assert_eq!(imgs.len(), 3);
    assert!(imgs[0]["error"].is_null() && imgs[2]["error"].is_null(), "{parsed}");
    assert_eq!(imgs[1]["error"]["code"], "invalid_config", "{parsed}");

    let _ = stop_tx.send(());
    let _ = task.await;
}

/// `val` returns calibrated per-class `thresholds` on both embedded and proxied servers; `predict` and
/// `predict_batch` apply `thresholds_path` (enforcing `--confine`) and `per_class_conf`, gate older servers on
/// `threshold_calibration`, and reject mutually exclusive or out-of-range arguments upfront.
#[tokio::test]
async fn val_and_predict_wire_calibrated_thresholds_embedded_and_proxied() {
    let home = folo_test_support::TestDir::new("mcp-threshold-calibration");
    let outside = folo_test_support::TestDir::new("mcp-threshold-outside");
    let mock_store: Arc<dyn Store> = Arc::new(folo_server::MockStore::new());
    let datasets = mock_store.list_datasets().unwrap();
    let models = mock_store.list_models().unwrap();
    let (model, ds_id) = models
        .iter()
        .find_map(|m| {
            datasets
                .iter()
                .find(|d| d.classes.iter().any(|c| m.classes.contains(c)))
                .map(|d| (m.clone(), d.id.clone()))
        })
        .expect("matching mock model and dataset");
    let image_id = mock_store
        .list_images(&ds_id, &folo_server::store::ImageQuery { limit: 1, ..Default::default() })
        .unwrap()
        .images[0]
        .id
        .clone();

    let mock_server = folo_server::Server::new(Arc::clone(&mock_store), Default::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(mock_server.serve(listener, async move {
        let _ = stop_rx.await;
    }));
    let opts = Options {
        data_dir: home.to_path_buf(),
        read_only: false,
        confine: true,
        device: "cpu".into(),
        published_weights: None,
    };
    let embedded = FoloMcp::new(Arc::clone(&mock_store), None, opts.clone());
    let live_http = Arc::new(HttpStore::new(&url, None));
    let proxied = FoloMcp::new(live_http.clone(), Some(live_http), opts.clone());
    let old_http = Arc::new(HttpStore::new(&url, None).assume_features(vec![]));
    let old_mcp = FoloMcp::new(old_http.clone(), Some(old_http), opts);

    // 1. `val` returns calibrated `thresholds` on both embedded and proxied servers.
    for backend in [&embedded, &proxied] {
        let v = backend
            .inner
            .val(
                ValArgs {
                    model_id: model.id.clone(),
                    dataset_id: ds_id.clone(),
                    split: folo_api::Split::Val,
                    imgsz: None,
                    conf: 0.001,
                    iou: 0.7,
                    diagnostics: false,
                    worst: None,
                    diag_conf: None,
                    diag_tp_iou: None,
                    diag_bg_iou: None,
                },
                &mut Progress::none(),
            )
            .await
            .unwrap();
        let thr = &v.value["thresholds"];
        assert!(thr.is_object(), "val must include thresholds: {}", v.value);
        assert!(thr["global"]["best_conf"].is_f64(), "{thr}");
        assert!(thr["macro_f1"].is_f64(), "{thr}");
        assert!(!thr["per_class"].as_array().unwrap().is_empty(), "{thr}");
    }

    // 2. Write a calibrated thresholds file inside `data_dir` that sets all classes to 1.0 (filters all boxes)
    //    vs 0.0 (keeps detections), and verify deterministic filtering via both `thresholds_path` and
    //    `per_class_conf` across embedded and proxied transports (JSON + multipart).
    let suppress_path = home.join("suppress.json");
    let keep_path = home.join("keep.json");
    let per_class_one: Vec<_> = (0..model.classes.len())
        .map(|c| serde_json::json!({"class": c, "best_conf": 1.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5, "truths": 4, "preds": 4}))
        .collect();
    let per_class_zero: Vec<_> = (0..model.classes.len())
        .map(|c| serde_json::json!({"class": c, "best_conf": 0.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5, "truths": 4, "preds": 4}))
        .collect();
    std::fs::write(
        &suppress_path,
        serde_json::to_vec(&serde_json::json!({
            "global": {"best_conf": 1.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5},
            "per_class": per_class_one,
            "macro_f1": 0.5,
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &keep_path,
        serde_json::to_vec(&serde_json::json!({
            "global": {"best_conf": 0.0, "best_f1": 0.5, "precision": 0.5, "recall": 0.5},
            "per_class": per_class_zero,
            "macro_f1": 0.5,
        }))
        .unwrap(),
    )
    .unwrap();

    for backend in [&embedded, &proxied] {
        let kept = backend
            .inner
            .predict(PredictArgs {
                model_id: model.id.clone(),
                image_path: None,
                image_base64: None,
                dataset_id: Some(ds_id.clone()),
                image_id: Some(image_id.clone()),
                conf: 0.25,
                iou: 0.7,
                max_det: 50,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: false,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: Some("keep.json".into()),
                per_class_conf: None,
            })
            .await
            .unwrap();
        let suppressed = backend
            .inner
            .predict(PredictArgs {
                model_id: model.id.clone(),
                image_path: None,
                image_base64: None,
                dataset_id: Some(ds_id.clone()),
                image_id: Some(image_id.clone()),
                conf: 0.25,
                iou: 0.7,
                max_det: 50,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: false,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: Some("suppress.json".into()),
                per_class_conf: None,
            })
            .await
            .unwrap();
        assert!(!kept.value["boxes"].as_array().unwrap().is_empty(), "keep.json should keep boxes");
        assert!(suppressed.value["boxes"].as_array().unwrap().is_empty(), "suppress.json should drop boxes");

        // Direct `per_class_conf` map over multipart (upload) also filters deterministically.
        let png =
            folo_data::encode(&folo_data::Image::new(32, 32, [100, 100, 100]), folo_data::ImageFormat::Png)
                .unwrap();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
        let all_one: BTreeMap<u32, f32> = (0..model.classes.len() as u32).map(|c| (c, 1.0)).collect();
        let mp = backend
            .inner
            .predict(PredictArgs {
                model_id: model.id.clone(),
                image_path: None,
                image_base64: Some(b64),
                dataset_id: None,
                image_id: None,
                conf: 0.0,
                iou: 0.7,
                max_det: 50,
                tile: None,
                tile_overlap: None,
                tile_full_image: None,
                tta: false,
                tta_scales: None,
                tta_hflip: None,
                thresholds_path: None,
                per_class_conf: Some(all_one),
            })
            .await
            .unwrap();
        assert!(mp.value["boxes"].as_array().unwrap().is_empty(), "per_class_conf=1.0 should drop boxes");
    }

    // 3. Mutual exclusion (`thresholds_path` + `per_class_conf`), `--confine` outside `data_dir`, out-of-range
    //    class id, and proxy feature gate on older servers all fail with the expected codes.
    let both = embedded
        .inner
        .predict(PredictArgs {
            model_id: model.id.clone(),
            image_path: None,
            image_base64: None,
            dataset_id: Some(ds_id.clone()),
            image_id: Some(image_id.clone()),
            conf: 0.25,
            iou: 0.7,
            max_det: 10,
            tile: None,
            tile_overlap: None,
            tile_full_image: None,
            tta: false,
            tta_scales: None,
            tta_hflip: None,
            thresholds_path: Some("keep.json".into()),
            per_class_conf: Some(BTreeMap::from([(0, 0.5)])),
        })
        .await
        .err()
        .unwrap();
    assert_eq!(both.body.code, "invalid_arguments");

    let escaped_file = outside.join("outside.json");
    std::fs::copy(&keep_path, &escaped_file).unwrap();
    let confined = embedded
        .inner
        .predict(PredictArgs {
            model_id: model.id.clone(),
            image_path: None,
            image_base64: None,
            dataset_id: Some(ds_id.clone()),
            image_id: Some(image_id.clone()),
            conf: 0.25,
            iou: 0.7,
            max_det: 10,
            tile: None,
            tile_overlap: None,
            tile_full_image: None,
            tta: false,
            tta_scales: None,
            tta_hflip: None,
            thresholds_path: Some(escaped_file.to_string_lossy().into_owned()),
            per_class_conf: None,
        })
        .await
        .err()
        .unwrap();
    assert_eq!(confined.body.code, "path_outside_data_dir");

    let old_err = old_mcp
        .inner
        .predict(PredictArgs {
            model_id: model.id.clone(),
            image_path: None,
            image_base64: None,
            dataset_id: Some(ds_id),
            image_id: Some(image_id),
            conf: 0.25,
            iou: 0.7,
            max_det: 10,
            tile: None,
            tile_overlap: None,
            tile_full_image: None,
            tta: false,
            tta_scales: None,
            tta_hflip: None,
            thresholds_path: None,
            per_class_conf: Some(BTreeMap::from([(0, 0.5)])),
        })
        .await
        .err()
        .unwrap();
    assert_eq!(old_err.body.code, "not_implemented");

    let _ = stop_tx.send(());
    let _ = task.await;
}
