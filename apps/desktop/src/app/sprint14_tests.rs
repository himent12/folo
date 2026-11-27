//! End-to-end UI interaction tests for Sprint 14 Desktop features:
//! 1. Model Detail Export & CLI Command Builder (ONNX dynamic/decode/half, quantize, predict
//!    SAHI/TTA/Track, clipboard copy + toast, and server export options).
//! 2. Dataset Class-Balance & Annotation Quality Strip (imbalance badge, rarest-class & empty-image
//!    one-click jumps from Statistics and header quick pills).
//! 3. Run Detail Per-Image Validation Error Explorer (All / FP / FN / TP filter pills, worst-error
//!    sorting, empty-filter reset, and confidence reactivity).

use folo_ui::{KeyDownEvent, KeyUpEvent, Keystroke, PlatformInput, WindowHandle};

use crate::api::{BBox, RunSamples, Sample, ScoredBox};
use crate::app::tests::{drive, open};
use crate::app::{DsTab, FoloApp, Route};
use crate::screens::models::export_builder::{CliExportFormat, ModelCliMode, PredictPrecision, QuantMethod};
use crate::screens::runs::{SampleErrorFilter, SampleSort};
use crate::store::Load;

fn send_key(app: &mut folo_ui::HeadlessApp, w: WindowHandle<FoloApp>, key: &str) {
    let ks = Keystroke::parse(key).unwrap();
    app.simulate_input(
        w.into(),
        PlatformInput::KeyDown(KeyDownEvent { keystroke: ks.clone(), is_held: false }),
    );
    app.run_until_idle();
    app.frame(w.into());
    app.simulate_input(w.into(), PlatformInput::KeyUp(KeyUpEvent { keystroke: ks }));
    app.run_until_idle();
    app.frame(w.into());
}

#[test]
fn model_export_cli_builder_toggles_copies_and_triggers_server_export() {
    let (mut app, window) = open();
    let mid = drive(&mut app, window, |this, w, cx| {
        let mut list = this.store.models.ready().unwrap().clone();
        list[0].path = Some("D:/folo/runs/demo/weights/best.folo".into());
        let id = list[0].id.clone();
        this.store.models = Load::Ready(list);
        this.navigate(Route::Model(id.clone()), w, cx);
        id
    });
    for _ in 0..3 {
        app.frame(window.into());
        app.run_until_idle();
    }

    // 1. Toggle ONNX dynamic batch + dynamic H×W + decode + half via keyboard activation.
    for key in ["model-cli-dyn-batch", "model-cli-dyn-hw", "model-cli-decode", "model-cli-half"] {
        drive(&mut app, window, |this, w, cx| this.focus_key(key, w, cx));
        app.frame(window.into());
        send_key(&mut app, window, "enter");
    }
    drive(&mut app, window, |this, _, _| {
        assert!(this.model_cli.onnx_dynamic_batch);
        assert!(this.model_cli.onnx_dynamic_hw);
        assert!(this.model_cli.onnx_decode);
        assert!(this.model_cli.onnx_half);
    });

    // 2. Copy command posts toast and writes exact shell-safe CLI command to clipboard.
    let toasts_before = drive(&mut app, window, |this, w, cx| {
        let n = this.toasts.len();
        this.focus_key("model-cli-copy", w, cx);
        n
    });
    app.frame(window.into());
    send_key(&mut app, window, "enter");
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.toasts.len(), toasts_before + 1);
        assert_eq!(this.toasts.last().unwrap().title.as_ref(), "Copied export command");
        let clip = cx.read_from_clipboard().and_then(|c| c.text()).unwrap_or_default();
        assert_eq!(
            clip,
            "folo export --model D:/folo/runs/demo/weights/best.folo --format onnx --dynamic --decode --half"
        );
    });

    // 3. Clicking Export ONNX now runs real backend export with options and posts completion toast.
    let toasts_before = drive(&mut app, window, |this, w, cx| {
        let n = this.toasts.len();
        this.focus_key("model-cli-server-export", w, cx);
        n
    });
    app.frame(window.into());
    send_key(&mut app, window, "enter");
    drive(&mut app, window, |this, _, _| {
        assert!(this.exporting.is_empty());
        assert_eq!(this.toasts.len(), toasts_before + 1);
        let t = this.toasts.last().unwrap();
        assert_eq!(t.title.as_ref(), "Export complete");
        assert!(t.body.as_deref().is_some_and(|b| b.contains("-decoded-f16.onnx")), "body={:?}", t.body);
    });

    // 4. Switch to Predict mode, verify SAHI and TTA are mutually exclusive, plus ByteTrack.
    drive(&mut app, window, |this, _, cx| {
        this.model_cli.mode = ModelCliMode::Predict;
        this.model_cli.predict_precision = PredictPrecision::F16;
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| this.focus_key("model-cli-sahi", w, cx));
    app.frame(window.into());
    send_key(&mut app, window, "enter");
    drive(&mut app, window, |this, w, cx| {
        assert!(this.model_cli.predict_sahi && !this.model_cli.predict_tta);
        this.focus_key("model-cli-tta", w, cx);
    });
    app.frame(window.into());
    send_key(&mut app, window, "enter");
    drive(&mut app, window, |this, w, cx| {
        assert!(this.model_cli.predict_tta && !this.model_cli.predict_sahi, "TTA clears SAHI");
        this.focus_key("model-cli-track", w, cx);
    });
    app.frame(window.into());
    send_key(&mut app, window, "enter");
    drive(&mut app, window, |this, w, cx| {
        assert!(this.model_cli.predict_track);
        this.focus_key("model-cli-copy", w, cx);
    });
    app.frame(window.into());
    send_key(&mut app, window, "enter");
    drive(&mut app, window, |_this, _, cx| {
        let clip = cx.read_from_clipboard().and_then(|c| c.text()).unwrap_or_default();
        assert_eq!(
            clip,
            "folo predict --model D:/folo/runs/demo/weights/best.folo <images> --precision f16 --tta --track"
        );
    });

    // 5. Switch to INT8 quantize format and verify clipboard command.
    drive(&mut app, window, |this, _, cx| {
        this.model_cli.mode = ModelCliMode::Export;
        this.model_cli.export_format = CliExportFormat::Int8Quantize;
        this.model_cli.quant_method = QuantMethod::MinMax;
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| this.focus_key("model-cli-copy", w, cx));
    app.frame(window.into());
    send_key(&mut app, window, "enter");
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.toasts.last().unwrap().title.as_ref(), "Copied quantize command");
        let clip = cx.read_from_clipboard().and_then(|c| c.text()).unwrap_or_default();
        assert!(clip.starts_with("folo quantize --model D:/folo/runs/demo/weights/best.folo --data "));
        assert!(clip.ends_with("--method minmax"), "clip={clip}");
        assert_eq!(this.route, Route::Model(mid));
    });
}

#[test]
fn dataset_balance_strip_jumps_to_rarest_class_and_empty_images() {
    let (mut app, window) = open();
    let ds_id = drive(&mut app, window, |this, w, cx| {
        let mut ds = this.store.datasets.ready().unwrap()[0].clone();
        ds.classes = vec!["circle".into(), "square".into(), "triangle".into()];
        let id = ds.id.clone();
        this.navigate(Route::Dataset(id.clone()), w, cx);
        id
    });
    for _ in 0..4 {
        app.frame(window.into());
        app.run_until_idle();
    }

    // Install skewed stats (class 2 = rarest with 15 boxes, 6 empty images).
    drive(&mut app, window, |this, _, cx| {
        let mut ds = this.store.dataset.get(&ds_id).unwrap().ready().unwrap().clone();
        ds.classes = vec!["circle".into(), "square".into(), "triangle".into()];
        let stats = ds.stats.as_mut().unwrap();
        stats.class_counts = vec![450, 200, 15];
        stats.boxes = 665;
        stats.empty_images = 6;
        this.store.dataset.insert(ds_id.clone(), Load::Ready(ds));
        this.ds_tab = DsTab::Statistics;
        cx.notify();
    });
    app.frame(window.into());

    // 1. Activate "Inspect rarest" in Statistics strip -> switches to Images filtered to class 2.
    drive(&mut app, window, |this, w, cx| this.focus_key("ds-balance-rarest", w, cx));
    app.frame(window.into());
    send_key(&mut app, window, "enter");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.ds_tab, DsTab::Images);
        assert_eq!(this.class_filter(&ds_id), Some(2));
        assert!(!this.empty_only(&ds_id));
    });

    // 2. Activate "Empty" quick pill on the tab row -> switches to empty-only filter and clears class.
    drive(&mut app, window, |this, w, cx| this.focus_key("ds-balance-empty-pill", w, cx));
    app.frame(window.into());
    send_key(&mut app, window, "enter");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.ds_tab, DsTab::Images);
        assert!(this.empty_only(&ds_id));
        assert_eq!(this.class_filter(&ds_id), None);
    });

    // 3. Switch back to Statistics and activate "Inspect empty" button inside the balance strip.
    drive(&mut app, window, |this, _, cx| {
        this.ds_empty_of = None;
        this.ds_tab = DsTab::Statistics;
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| this.focus_key("ds-balance-empty", w, cx));
    app.frame(window.into());
    send_key(&mut app, window, "enter");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.ds_tab, DsTab::Images);
        assert!(this.empty_only(&ds_id));
    });
}

#[test]
fn run_samples_error_explorer_filters_sorts_and_resets_on_navigation() {
    let (mut app, window) = open();
    let run_id = drive(&mut app, window, |this, w, cx| {
        let r = this.store.runs.ready().unwrap().iter().find(|r| r.epoch > 0).unwrap().clone();
        this.navigate(Route::Run(r.id.clone()), w, cx);
        r.id
    });
    for _ in 0..4 {
        app.frame(window.into());
        app.run_until_idle();
    }

    // Install deterministic samples covering clean TP, FP-only, FN-only, and high-error sample at index 9.
    let mk_gt = |class: u32, x1: f32, y1: f32, x2: f32, y2: f32| BBox { class, x1, y1, x2, y2 };
    let mk_pred = |class: u32, score: f32, x1: f32, y1: f32, x2: f32, y2: f32| ScoredBox {
        class,
        name: format!("c{class}"),
        score,
        x1,
        y1,
        x2,
        y2,
    };
    let samples = vec![
        // 0: Clean TP only (0 errors)
        Sample {
            image_id: "img_clean".into(),
            dataset_id: "ds_3b91".into(),
            width: 640,
            height: 480,
            truth: vec![mk_gt(0, 10., 10., 110., 110.)],
            pred: vec![mk_pred(0, 0.90, 10., 10., 110., 110.)],
        },
        // 1: FN only (2 missed GT)
        Sample {
            image_id: "img_fn".into(),
            dataset_id: "ds_3b91".into(),
            width: 640,
            height: 480,
            truth: vec![mk_gt(0, 10., 10., 80., 80.), mk_gt(1, 200., 200., 280., 280.)],
            pred: vec![],
        },
        // 2: FP only (3 false positives)
        Sample {
            image_id: "img_fp".into(),
            dataset_id: "ds_3b91".into(),
            width: 640,
            height: 480,
            truth: vec![],
            pred: vec![
                mk_pred(0, 0.85, 10., 10., 60., 60.),
                mk_pred(0, 0.70, 100., 10., 150., 60.),
                mk_pred(0, 0.40, 200., 10., 250., 60.),
            ],
        },
    ];
    drive(&mut app, window, |this, _, cx| {
        this.store.samples.insert(run_id.clone(), Load::Ready(RunSamples { epoch: 5, samples }));
        cx.notify();
    });
    app.frame(window.into());

    // 1. Activate FP filter pill -> only `img_fp` (1 tile: smp-0 present, smp-1 absent).
    drive(&mut app, window, |this, w, cx| this.focus_key("smp-filter-fp", w, cx));
    app.frame(window.into());
    send_key(&mut app, window, "enter");
    drive(&mut app, window, |this, _, _| {
        assert_eq!(this.sample_error_filter, SampleErrorFilter::Fp);
    });

    // 2. Raising confidence above all FP predictions yields empty-filter state with Reset button.
    drive(&mut app, window, |this, _, cx| {
        this.sample_conf = 0.95;
        cx.notify();
    });
    app.frame(window.into());
    drive(&mut app, window, |this, w, cx| this.focus_key("smp-filter-reset", w, cx));
    app.frame(window.into());
    send_key(&mut app, window, "enter");
    drive(&mut app, window, |this, _, cx| {
        assert_eq!(this.sample_error_filter, SampleErrorFilter::All);
        this.sample_conf = 0.25;
        this.sample_sort = SampleSort::WorstErrors;
        cx.notify();
    });
    app.frame(window.into());

    // 3. Navigating away resets error filter and sort to defaults.
    drive(&mut app, window, |this, w, cx| {
        this.sample_error_filter = SampleErrorFilter::Fn;
        this.sample_sort = SampleSort::MostFn;
        this.navigate(Route::Runs, w, cx);
        assert_eq!(this.sample_error_filter, SampleErrorFilter::All);
        assert_eq!(this.sample_sort, SampleSort::Default);
    });
}
