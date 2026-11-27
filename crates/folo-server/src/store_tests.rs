use super::*;
use folo_api::{DatasetIssue, IssueSeverity};

#[test]
fn issue_query_page_counts_and_pages_in_one_pass() {
    let make = |kind: &str, severity: IssueSeverity, file: &str| DatasetIssue {
        kind: kind.into(),
        severity,
        file: file.into(),
        line: None,
        message: String::new(),
        fix: None,
    };
    let all = [
        make("broken_label", IssueSeverity::Error, "a.txt"),
        make("clipped_box", IssueSeverity::Warning, "b.txt"),
        make("clipped_box", IssueSeverity::Warning, "c.txt"),
        make("clipped_box", IssueSeverity::Warning, "d.txt"),
        make("empty_image", IssueSeverity::Info, "e.jpg"),
    ];
    // Wrap in a non-`Clone` iterator to prove single-pass consumption.
    let mut idx = 0usize;
    let iter = std::iter::from_fn(|| {
        let item = all.get(idx)?;
        idx += 1;
        Some(item)
    });
    let q = IssueQuery {
        kind: Some("clipped_box".into()),
        severity: Some(IssueSeverity::Warning),
        offset: 1,
        limit: 1,
    };
    let page = q.page(iter);
    assert_eq!(idx, all.len(), "consumed every issue once");
    assert_eq!(page.total, 3);
    assert_eq!(page.issues.len(), 1);
    assert_eq!(page.issues[0].file, "c.txt");
    assert_eq!((page.counts.error, page.counts.warning, page.counts.info), (1, 3, 1));
    assert_eq!(page.counts.by_kind.get("broken_label"), Some(&1));
    assert_eq!(page.counts.by_kind.get("clipped_box"), Some(&3));
    assert_eq!(page.counts.by_kind.get("empty_image"), Some(&1));
}

#[test]
fn check_predict_workload_caps_tile_count_and_crop_bytes() {
    // High tile count rejected.
    let many = PredictOptions {
        tile: Some(TileConfig {
            tile_width: 32,
            tile_height: 32,
            overlap_x: 0.0,
            overlap_y: 0.0,
            ..Default::default()
        }),
        tta: None,
    };
    assert_eq!(check_predict_workload(&many, 640, Some((640, 640))).unwrap_err().body.code, "invalid_config");

    // Large overlapping crops with high tile_batch exceed MAX_PREDICT_CROP_BYTES (256 MiB):
    // 8192x8192 with 4096x4096 tiles at 0.5 overlap -> 3x3 = 9 tiles; 9 * 4096 * 4096 * 3 = 432 MiB.
    let heavy = PredictOptions {
        tile: Some(TileConfig {
            tile_width: 4096,
            tile_height: 4096,
            overlap_x: 0.5,
            overlap_y: 0.5,
            tile_batch: 9,
            ..Default::default()
        }),
        tta: None,
    };
    let err = check_predict_workload(&heavy, 640, Some((8192, 8192))).unwrap_err();
    assert_eq!(err.body.code, "invalid_config");
    assert!(err.body.message.contains("MiB"), "{}", err.body.message);

    // Reducing tile_batch to 4 keeps peak chunk bytes at 192 MiB <= 256 MiB.
    let bounded = PredictOptions {
        tile: Some(TileConfig {
            tile_width: 4096,
            tile_height: 4096,
            overlap_x: 0.5,
            overlap_y: 0.5,
            tile_batch: 4,
            ..Default::default()
        }),
        tta: None,
    };
    check_predict_workload(&bounded, 640, Some((8192, 8192))).unwrap();
}

#[test]
fn check_val_diag_config_enforces_bounds() {
    check_val_diag_config(&ValDiagConfig::default()).unwrap();
    for bad in [
        ValDiagConfig { worst: 0, ..ValDiagConfig::default() },
        ValDiagConfig { worst: MAX_VAL_DIAG_WORST + 1, ..ValDiagConfig::default() },
        ValDiagConfig { conf: -0.1, ..ValDiagConfig::default() },
        ValDiagConfig { conf: f32::NAN, ..ValDiagConfig::default() },
        ValDiagConfig { tp_iou: 1.1, ..ValDiagConfig::default() },
        ValDiagConfig { bg_iou: 0.6, tp_iou: 0.5, ..ValDiagConfig::default() },
    ] {
        let e = check_val_diag_config(&bad).unwrap_err();
        assert_eq!((e.status, e.body.code.as_str()), (axum::http::StatusCode::BAD_REQUEST, "invalid_body"));
    }
}

#[test]
fn check_export_axes_rejects_dynamic_axes_for_folo_f16() {
    let f16 = ExportSpec { format: folo_api::ExportFormat::FoloF16, onnx: folo_api::OnnxSwitches::default() };
    check_export_axes(&f16, DynamicAxes::STATIC).unwrap();
    let e = check_export_axes(&f16, DynamicAxes::BATCH).unwrap_err();
    assert_eq!((e.status, e.body.code.as_str()), (axum::http::StatusCode::BAD_REQUEST, "invalid_body"));

    let onnx = ExportSpec { format: folo_api::ExportFormat::Onnx, onnx: folo_api::OnnxSwitches::default() };
    check_export_axes(&onnx, DynamicAxes::BATCH).unwrap();
    check_export_axes(&onnx, DynamicAxes::HW).unwrap();
}
