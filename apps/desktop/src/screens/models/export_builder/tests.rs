//! Unit and headless UI tests for the Model Detail Export & Inference CLI builder,
//! including selective-`f32` INT8 quantization presets (`docs/CLI.md`).

use super::command::{
    CliExportFormat, FEATURE_EXPORT_DYNAMIC_AXES, ModelCliMode, ModelCliState, PredictInt8Target,
    PredictPrecision, QuantF32MacBudget, QuantF32Policy, QuantMethod, ServerExportSupport, build_model_cli,
    check_server_export_support, derive_quant_output_path, is_int8_model_path,
};
use crate::api::{Backend, Head, Model, ModelExportOptions, ModelSource, Split, mock::MockBackend};

fn sample_model() -> Model {
    let api = MockBackend::new(4200, true);
    let mut m = api.models().unwrap().remove(0);
    m.path = Some("D:/folo/runs/shapes-n/weights/best.folo".into());
    m.head = Head::O2m;
    m.fused = Some(false);
    m
}

#[test]
fn onnx_export_combines_dynamic_axes_decode_half_and_server_spec() {
    let m = sample_model();
    let mut st = ModelCliState::default();

    let b = build_model_cli(&m, None, &st);
    assert_eq!(b.command, "folo export --model D:/folo/runs/shapes-n/weights/best.folo --format onnx");
    assert_eq!(b.server_export, Some(ModelExportOptions::plain("onnx")));

    st.onnx_dynamic_batch = true;
    let b = build_model_cli(&m, None, &st);
    assert!(b.command.contains("--dynamic-batch") && !b.command.contains("--dynamic "));
    assert_eq!(
        b.server_export,
        Some(ModelExportOptions {
            format: "onnx",
            decode: false,
            dynamic_batch: true,
            dynamic_hw: false,
            half: false,
        })
    );

    st.onnx_dynamic_hw = true;
    st.onnx_decode = true;
    st.onnx_half = true;
    let b = build_model_cli(&m, None, &st);
    assert_eq!(
        b.command,
        "folo export --model D:/folo/runs/shapes-n/weights/best.folo --format onnx --dynamic --decode --half"
    );
    assert_eq!(
        b.server_export,
        Some(ModelExportOptions {
            format: "onnx",
            decode: true,
            dynamic_batch: true,
            dynamic_hw: true,
            half: true,
        })
    );

    let mut o2o = m.clone();
    o2o.head = Head::O2o;
    st.onnx_head_o2m = true;
    let b = build_model_cli(&o2o, None, &st);
    assert!(b.command.ends_with("--head o2m"), "{}", b.command);
    assert!(b.server_export.is_none());
}

#[test]
fn server_export_support_gates_older_servers_and_remote_split_axes() {
    let plain = ModelExportOptions::plain("onnx");
    let switched = ModelExportOptions {
        format: "onnx",
        decode: true,
        dynamic_batch: false,
        dynamic_hw: false,
        half: false,
    };
    let split = ModelExportOptions {
        format: "onnx",
        decode: false,
        dynamic_batch: true,
        dynamic_hw: false,
        half: false,
    };
    let full_dyn = ModelExportOptions {
        format: "onnx",
        decode: true,
        dynamic_batch: true,
        dynamic_hw: true,
        half: true,
    };

    let no_features: Vec<String> = Vec::new();
    let switches_only = vec![folo_api::FEATURE_EXPORT_ONNX_SWITCHES.to_string()];
    let switches_and_axes =
        vec![folo_api::FEATURE_EXPORT_ONNX_SWITCHES.to_string(), FEATURE_EXPORT_DYNAMIC_AXES.to_string()];
    let all_features = vec![
        folo_api::FEATURE_EXPORT_ONNX_SWITCHES.to_string(),
        FEATURE_EXPORT_DYNAMIC_AXES.to_string(),
        folo_api::FEATURE_WEIGHTS_ONNX.to_string(),
    ];

    assert!(matches!(
        check_server_export_support(plain, &no_features, false),
        ServerExportSupport::Supported(_)
    ));
    assert!(matches!(
        check_server_export_support(switched, &no_features, false),
        ServerExportSupport::Unsupported(_)
    ));
    assert!(matches!(
        check_server_export_support(switched, &switches_only, false),
        ServerExportSupport::Supported(_)
    ));
    assert!(matches!(
        check_server_export_support(split, &switches_only, false),
        ServerExportSupport::Unsupported(_)
    ));
    assert!(matches!(
        check_server_export_support(split, &switches_and_axes, false),
        ServerExportSupport::Supported(_)
    ));
    assert!(matches!(
        check_server_export_support(switched, &switches_only, true),
        ServerExportSupport::Unsupported(_)
    ));
    assert!(matches!(
        check_server_export_support(split, &all_features, true),
        ServerExportSupport::Unsupported(_)
    ));
    assert!(matches!(
        check_server_export_support(full_dyn, &all_features, true),
        ServerExportSupport::Supported(_)
    ));
    assert_eq!(
        derive_quant_output_path(
            "D:/models/best.FOLO",
            QuantMethod::Percentile,
            QuantF32Policy::Auto,
            QuantF32MacBudget::Pct5,
        ),
        "D:/models/best-int8-percentile.folo"
    );
    assert!(is_int8_model_path("D:/models/best-INT8.FOLO"));
}

#[test]
fn quantize_selective_f32_presets_and_shell_unsafe_paths_stay_in_lockstep() {
    let mut m = sample_model();
    m.path = Some("D:/my models/best.folo".into());
    let mut st = ModelCliState {
        mode: ModelCliMode::Export,
        export_format: CliExportFormat::Int8Quantize,
        quant_method: QuantMethod::Mse,
        ..Default::default()
    };
    let b = build_model_cli(&m, Some("D:/my data/shapes"), &st);
    assert_eq!(
        b.command,
        "folo quantize --model \"D:/my models/best.folo\" --data \"D:/my data/shapes\" --method mse --output \"D:/my models/best-int8-mse.folo\""
    );
    assert!(b.quote_note.is_none());
    assert!(b.server_export.is_none());

    // Selective-f32 Auto with 10% MAC budget emits `--f32-policy auto --max-f32-macs 0.10` and a matching file tag.
    st.quant_f32_policy = QuantF32Policy::Auto;
    st.quant_f32_macs = QuantF32MacBudget::Pct10;
    let b_auto10 = build_model_cli(&m, Some("D:/my data/shapes"), &st);
    assert_eq!(
        b_auto10.command,
        "folo quantize --model \"D:/my models/best.folo\" --data \"D:/my data/shapes\" --method mse --f32-policy auto --max-f32-macs 0.10 --output \"D:/my models/best-int8-mse-f32-10.folo\""
    );

    // P5 tail policy never emits auto-only `--max-f32-macs`, matching `folo-cli` validation.
    st.quant_f32_policy = QuantF32Policy::P5Tail;
    let b_p5 = build_model_cli(&m, Some("D:/my data/shapes"), &st);
    assert_eq!(
        b_p5.command,
        "folo quantize --model \"D:/my models/best.folo\" --data \"D:/my data/shapes\" --method mse --f32-policy p5-tail --output \"D:/my models/best-int8-mse-p5-tail.folo\""
    );
    assert!(!b_p5.command.contains("--max-f32-macs"));

    // Full int8 (`--f32-policy none`) keeps Predict `--precision int8` in lockstep.
    st.quant_f32_policy = QuantF32Policy::None;
    st.quant_method = QuantMethod::MinMax;
    let b_none = build_model_cli(&m, Some("D:/my data/shapes"), &st);
    assert!(
        b_none.command.ends_with(
            "--method minmax --f32-policy none --output \"D:/my models/best-int8-minmax-full.folo\""
        ),
        "{}",
        b_none.command
    );
    st.mode = ModelCliMode::Predict;
    st.predict_precision = PredictPrecision::Int8;
    let b_p = build_model_cli(&m, Some("D:/my data/shapes"), &st);
    assert_eq!(
        b_p.command,
        "folo predict --model \"D:/my models/best-int8-minmax-full.folo\" <images> --precision int8"
    );

    // Re-quantizing an already policy-tagged file strips the old tag cleanly without doubling.
    assert!(is_int8_model_path("D:/models/best-int8-mse-p5-tail.folo"));
    assert!(is_int8_model_path("D:/models/best-int8-percentile-f32-10.folo"));
    assert_eq!(
        derive_quant_output_path(
            "D:/models/best-int8-mse-p5-tail.folo",
            QuantMethod::Percentile,
            QuantF32Policy::None,
            QuantF32MacBudget::Pct5,
        ),
        "D:/models/best-int8-percentile-full.folo"
    );

    // Shell-unsafe `$` path must never be interpolated into the command string.
    m.path = Some("D:/bad$path/best.folo".into());
    st.mode = ModelCliMode::Export;
    st.quant_method = QuantMethod::Percentile;
    st.quant_f32_policy = QuantF32Policy::Auto;
    st.quant_f32_macs = QuantF32MacBudget::Pct5;
    let b = build_model_cli(&m, Some("D:/ok/shapes"), &st);
    assert_eq!(b.command, "folo quantize --model <model> --data D:/ok/shapes --output <output-int8>");
    assert!(b.quote_note.as_deref().is_some_and(|n| n.contains("D:/bad$path/best.folo")));
    assert!(b.quote_note.as_deref().is_some_and(|n| n.contains("D:/bad$path/best-int8-percentile.folo")));

    // Published weights without local path resolve by published id.
    let mut pub_m = sample_model();
    pub_m.path = None;
    pub_m.source = Some(ModelSource::Published);
    pub_m.id = "folo-n-coco-v5".into();
    let b = build_model_cli(&pub_m, None, &ModelCliState::default());
    assert_eq!(b.command, "folo export --model folo-n-coco-v5 --format onnx");
    let b_pub_q = build_model_cli(&pub_m, Some("D:/ok/shapes"), &st);
    assert_eq!(
        b_pub_q.command,
        "folo quantize --model folo-n-coco-v5 --data D:/ok/shapes --output folo-n-coco-v5-int8-percentile.folo"
    );
}

#[test]
fn predict_flags_reflect_sahi_tta_track_int8_target_and_utf8_paths() {
    let m = sample_model();
    let mut st = ModelCliState {
        mode: ModelCliMode::Predict,
        predict_sahi: true,
        predict_track: true,
        predict_precision: PredictPrecision::F16,
        predict_conf: 0.40,
        predict_iou: 0.55,
        ..Default::default()
    };
    let b = build_model_cli(&m, None, &st);
    assert_eq!(
        b.command,
        "folo predict --model D:/folo/runs/shapes-n/weights/best.folo <images> --conf 0.40 --iou 0.55 --precision f16 --tile 640 --track"
    );

    let mut o2o = m.clone();
    o2o.head = Head::O2o;
    assert!(!build_model_cli(&o2o, None, &st).command.contains("--iou"));

    st.predict_sahi = false;
    st.predict_tta = true;
    st.predict_track = false;
    assert!(build_model_cli(&o2o, None, &st).command.contains("--iou 0.55"));

    st.predict_precision = PredictPrecision::Int8;
    st.quant_method = QuantMethod::Percentile;
    st.predict_conf = 0.25;
    st.predict_iou = 0.70;
    let b_int8 = build_model_cli(&m, None, &st);
    assert_eq!(
        b_int8.command,
        "folo predict --model D:/folo/runs/shapes-n/weights/best-int8-percentile.folo <images> --precision int8 --tta"
    );

    let def_p = QuantF32Policy::Auto;
    let def_m = QuantF32MacBudget::Pct5;
    assert!(is_int8_model_path("D:/models/best-int8-2.folo"));
    assert!(is_int8_model_path("D:/models/best-int8-percentile-recal.folo"));
    assert_eq!(
        derive_quant_output_path("D:/models/best-int8-2.folo", QuantMethod::MinMax, def_p, def_m),
        "D:/models/best-int8-minmax.folo"
    );
    assert_eq!(
        derive_quant_output_path(
            "D:/models/best-int8-percentile.folo",
            QuantMethod::Percentile,
            def_p,
            def_m
        ),
        "D:/models/best-int8-percentile-recal.folo"
    );
    assert_eq!(
        derive_quant_output_path("cafébest.folo", QuantMethod::Percentile, def_p, def_m),
        "cafébest-int8-percentile.folo"
    );
    assert_eq!(
        derive_quant_output_path("模型.folo", QuantMethod::MinMax, def_p, def_m),
        "模型-int8-minmax.folo"
    );
    assert_eq!(derive_quant_output_path("日本語", QuantMethod::Mse, def_p, def_m), "日本語-int8-mse.folo");
}

#[test]
fn model_detail_renders_selective_f32_quantize_controls() {
    use crate::app::Route;
    use crate::app::tests::{drive, open};

    let (mut app, window) = open();
    drive(&mut app, window, |this, w, cx| {
        let mid = this.store.models.ready().unwrap()[0].id.clone();
        this.model_cli.mode = ModelCliMode::Export;
        this.model_cli.export_format = CliExportFormat::Int8Quantize;
        this.model_cli.quant_f32_policy = QuantF32Policy::Auto;
        this.navigate(Route::Model(mid), w, cx);
    });
    app.frame(window.into());
    app.run_until_idle();
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        for expected in ["model-cli-quant-method", "model-cli-quant-f32-policy", "model-cli-quant-f32-macs"] {
            assert!(keys.iter().any(|k| k == expected), "missing {expected} in {keys:?}");
        }
    });
    // Switching to P5Tail hides the auto-only MAC budget control.
    drive(&mut app, window, |this, _, cx| {
        this.model_cli.quant_f32_policy = QuantF32Policy::P5Tail;
        this.focus.borrow_mut().clear();
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "model-cli-quant-f32-policy"));
        assert!(!keys.iter().any(|k| k == "model-cli-quant-f32-macs"));
    });

    // Predict --precision int8 renders Checkpoint target, Calibration, Selective f32, and F32 MAC budget.
    drive(&mut app, window, |this, _, cx| {
        this.model_cli.mode = ModelCliMode::Predict;
        this.model_cli.predict_precision = PredictPrecision::Int8;
        this.model_cli.quant_f32_policy = QuantF32Policy::Auto;
        this.model_cli.quant_f32_macs = QuantF32MacBudget::Pct15;
        this.focus.borrow_mut().clear();
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        for expected in [
            "model-cli-predict-int8-target",
            "model-cli-predict-quant-method",
            "model-cli-predict-quant-f32-policy",
            "model-cli-predict-quant-f32-macs",
        ] {
            assert!(keys.iter().any(|k| k == expected), "missing {expected} in {keys:?}");
        }
    });
    drive(&mut app, window, |this, _, cx| {
        this.model_cli.quant_f32_policy = QuantF32Policy::None;
        this.focus.borrow_mut().clear();
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, _, _| {
        let keys: Vec<String> = this.focus.borrow().keys().map(|k| k.to_string()).collect();
        assert!(keys.iter().any(|k| k == "model-cli-predict-quant-f32-policy"));
        assert!(!keys.iter().any(|k| k == "model-cli-predict-quant-f32-macs"));
    });

    // Navigating to another model resets explicit predict_int8_target so it does not leak across models.
    drive(&mut app, window, |this, w, cx| {
        let models = this.store.models.ready().unwrap();
        let next_mid = models.get(1).unwrap_or(&models[0]).id.clone();
        this.model_cli.predict_int8_target = Some(PredictInt8Target::QuantizedOutput);
        this.navigate(Route::Model(next_mid), w, cx);
        assert_eq!(this.model_cli.predict_int8_target, None);
    });
}

#[test]
fn predict_int8_recalibration_targets_current_or_quantized_output_explicitly() {
    let mut m = sample_model();
    m.path = Some("D:/models/best-int8-mse-p5-tail.folo".into());
    let mut st = ModelCliState {
        mode: ModelCliMode::Export,
        export_format: CliExportFormat::Int8Quantize,
        quant_method: QuantMethod::Mse,
        quant_f32_policy: QuantF32Policy::None,
        quant_f32_macs: QuantF32MacBudget::Pct5,
        predict_precision: PredictPrecision::Int8,
        predict_int8_target: None,
        ..Default::default()
    };
    let q = build_model_cli(&m, Some("D:/data/shapes"), &st);
    assert!(q.command.ends_with("--output D:/models/best-int8-mse-full.folo"), "{}", q.command);

    // Default (`None`) for an `-int8*.folo` file selects Current checkpoint verbatim and notes
    // that it requires an int8-calibrated file rather than asserting it unconditionally.
    st.mode = ModelCliMode::Predict;
    let p_cur = build_model_cli(&m, None, &st);
    assert_eq!(
        p_cur.command,
        "folo predict --model D:/models/best-int8-mse-p5-tail.folo <images> --precision int8"
    );
    assert!(p_cur.hint.as_deref().is_some_and(|h| h.contains("requires an int8-calibrated")));

    // QuantizedOutput matches the recalibrated `--output` file verbatim.
    st.predict_int8_target = Some(PredictInt8Target::QuantizedOutput);
    let p_out = build_model_cli(&m, None, &st);
    assert_eq!(
        p_out.command,
        "folo predict --model D:/models/best-int8-mse-full.folo <images> --precision int8"
    );

    // Auto + 10% MAC budget in Predict int8 stays in lockstep with quantize --output.
    st.quant_f32_policy = QuantF32Policy::Auto;
    st.quant_f32_macs = QuantF32MacBudget::Pct10;
    let p_macs = build_model_cli(&m, None, &st);
    assert_eq!(
        p_macs.command,
        "folo predict --model D:/models/best-int8-mse-f32-10.folo <images> --precision int8"
    );

    // Custom-named calibrated checkpoint without `-int8` in filename can explicitly target Current.
    let mut custom = sample_model();
    custom.path = Some("D:/models/custom-calibrated.folo".into());
    st.predict_int8_target = None;
    let p_custom_default = build_model_cli(&custom, None, &st);
    assert_eq!(
        p_custom_default.command,
        "folo predict --model D:/models/custom-calibrated-int8-mse-f32-10.folo <images> --precision int8"
    );
    st.predict_int8_target = Some(PredictInt8Target::Current);
    let p_custom_current = build_model_cli(&custom, None, &st);
    assert_eq!(
        p_custom_current.command,
        "folo predict --model D:/models/custom-calibrated.folo <images> --precision int8"
    );
}

#[test]
fn validate_save_thresholds_and_predict_thresholds_share_path_and_exclude_conf() {
    let mut m = sample_model();
    m.path = Some("D:/my models/best.folo".into());

    // Validate mode emits `folo val --model ... --data ... --save-thresholds <stem>-thresholds.json`.
    let mut st = ModelCliState {
        mode: ModelCliMode::Validate,
        val_split: Split::Val,
        val_save_thresholds: true,
        ..Default::default()
    };
    let v = build_model_cli(&m, Some("D:/my data/shapes"), &st);
    assert_eq!(
        v.command,
        "folo val --model \"D:/my models/best.folo\" --data \"D:/my data/shapes\" --save-thresholds \"D:/my models/best-thresholds.json\""
    );
    assert!(v.server_export.is_none());
    assert!(v.quote_note.is_none());

    // Non-default split & precision flags are included; disabling save_thresholds omits `--save-thresholds`.
    st.val_split = Split::Test;
    st.predict_precision = PredictPrecision::F16;
    st.val_save_thresholds = false;
    let v2 = build_model_cli(&m, Some("D:/my data/shapes"), &st);
    assert_eq!(
        v2.command,
        "folo val --model \"D:/my models/best.folo\" --data \"D:/my data/shapes\" --split test --precision f16"
    );

    // Predict mode with `--thresholds` uses the exact same derived thresholds path and omits `--conf`
    // even when `predict_conf` is non-default (`--thresholds` and `--conf` conflict in `folo-cli`).
    st.mode = ModelCliMode::Predict;
    st.predict_precision = PredictPrecision::F32;
    st.predict_conf = 0.55;
    st.predict_use_thresholds = true;
    let p = build_model_cli(&m, None, &st);
    assert_eq!(
        p.command,
        "folo predict --model \"D:/my models/best.folo\" <images> --thresholds \"D:/my models/best-thresholds.json\""
    );
    assert!(!p.command.contains("--conf"), "{}", p.command);

    // Shell-unsafe `$` path falls back to `<thresholds.json>` placeholder with a quote note.
    m.path = Some("D:/bad$dir/best.folo".into());
    st.mode = ModelCliMode::Validate;
    st.val_split = Split::Val;
    st.val_save_thresholds = true;
    let v_unsafe = build_model_cli(&m, Some("D:/ok/shapes"), &st);
    assert_eq!(
        v_unsafe.command,
        "folo val --model <model> --data D:/ok/shapes --save-thresholds <thresholds.json>"
    );
    assert!(v_unsafe.quote_note.as_deref().is_some_and(|n| n.contains("best-thresholds.json")));

    // Published model without local path uses `<id>-thresholds.json`.
    let mut pub_m = sample_model();
    pub_m.path = None;
    pub_m.source = Some(ModelSource::Published);
    pub_m.id = "folo-n-coco-v5".into();
    st.mode = ModelCliMode::Predict;
    st.predict_use_thresholds = true;
    let p_pub = build_model_cli(&pub_m, None, &st);
    assert_eq!(
        p_pub.command,
        "folo predict --model folo-n-coco-v5 <images> --thresholds folo-n-coco-v5-thresholds.json"
    );

    // Distinct precisions & INT8 quantization variants derive non-colliding threshold paths shared
    // identically between Validate and Predict.
    let mut vm = sample_model();
    vm.path = Some("D:/models/best.folo".into());
    st.predict_precision = PredictPrecision::F16;
    st.mode = ModelCliMode::Validate;
    st.val_save_thresholds = true;
    let v_f16 = build_model_cli(&vm, Some("D:/data/shapes"), &st);
    assert!(
        v_f16.command.ends_with("--precision f16 --save-thresholds D:/models/best-f16-thresholds.json"),
        "{}",
        v_f16.command
    );
    st.mode = ModelCliMode::Predict;
    let p_f16 = build_model_cli(&vm, None, &st);
    assert!(p_f16.command.contains("--thresholds D:/models/best-f16-thresholds.json"), "{}", p_f16.command);

    st.predict_precision = PredictPrecision::Int8;
    st.quant_method = QuantMethod::Mse;
    st.quant_f32_policy = QuantF32Policy::P5Tail;
    st.predict_int8_target = Some(PredictInt8Target::QuantizedOutput);
    st.mode = ModelCliMode::Validate;
    let v_int8 = build_model_cli(&vm, Some("D:/data/shapes"), &st);
    assert_eq!(
        v_int8.command,
        "folo val --model D:/models/best-int8-mse-p5-tail.folo --data D:/data/shapes --precision int8 --save-thresholds D:/models/best-int8-mse-p5-tail-thresholds.json"
    );
    st.mode = ModelCliMode::Predict;
    let p_int8 = build_model_cli(&vm, None, &st);
    assert_eq!(
        p_int8.command,
        "folo predict --model D:/models/best-int8-mse-p5-tail.folo <images> --thresholds D:/models/best-int8-mse-p5-tail-thresholds.json --precision int8"
    );

    // Custom-named INT8 checkpoint targeting Current gets `-int8-thresholds.json` so it never collides with f32.
    st.predict_int8_target = Some(PredictInt8Target::Current);
    let p_cur_int8 = build_model_cli(&vm, None, &st);
    assert_eq!(
        p_cur_int8.command,
        "folo predict --model D:/models/best.folo <images> --thresholds D:/models/best-int8-thresholds.json --precision int8"
    );
}
