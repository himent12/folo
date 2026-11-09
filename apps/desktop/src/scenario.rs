//! Deterministic app states for screenshots: `--scenario <name>`.
//!
//! A scenario freezes the mock clock, picks a theme and a screen, and can open overlays or set
//! hover/focus so states that normally need a pointer are visible. Names compose with `+`:
//! `run-live+light` is the live run dashboard in the light theme.

use gpui::{Context, Window};

use crate::app::{FoloApp, Route, RunFilter, SampleView};
use crate::store::Load;
use crate::theme::Mode;

pub struct Spec {
    pub parts: Vec<String>,
    pub mode: Option<Mode>,
    pub live_step: u32,
    pub fail: bool,
    pub empty: bool,
}

/// Every scenario `tools/shoot-all.ps1` captures, in review order.
pub const ALL: &[&str] = &[
    "overview",
    "overview+light",
    "overview+loading",
    "overview+empty",
    "overview+error",
    "palette",
    "palette+light",
    "datasets",
    "dataset",
    "dataset+light",
    "dataset+filter",
    "inspector",
    "inspector+light",
    "inspector+sample",
    "runs",
    "run-live",
    "run-live+light",
    "run-live+hover",
    "run-live+charts-focus",
    "run-done",
    "run-done+bottom",
    "run-done+sample-focus",
    "run-failed",
    "new-run",
    "new-run+amp",
    "new-run+light",
    "stop-confirm",
    "stop-confirm+light",
    "models",
    "model",
    "model+light",
    "model+table-focus",
    "playground",
    "playground+nms",
    "playground+empty",
    "playground+light",
    "playground+canvas-focus",
    "settings",
    "settings+reference",
    "toast",
    "kb-focus",
];

impl Spec {
    pub fn parse(name: &str) -> Spec {
        let parts: Vec<String> = name.split('+').map(|s| s.trim().to_string()).collect();
        let has = |p: &str| parts.iter().any(|x| x == p);
        Spec {
            mode: if has("light") {
                Some(Mode::Light)
            } else if has("dark") {
                Some(Mode::Dark)
            } else {
                None
            },
            live_step: 4180,
            fail: has("error"),
            empty: has("empty"),
            parts,
        }
    }

    pub fn has(&self, p: &str) -> bool {
        self.parts.iter().any(|x| x == p)
    }
}

pub fn apply(app: &mut FoloApp, s: &Spec, window: &mut Window, cx: &mut Context<FoloApp>) {
    app.hold_loading = s.has("loading");
    if s.fail {
        // The failing backend answers at once, but through the async fetches a busy machine can
        // still show "Connecting…" at capture time: settle the failure before the first frame.
        app.store.health = app.store.api.health().into();
        app.store.runs = app.store.api.runs().into();
        app.store.models = app.store.api.models().into();
        app.store.datasets = app.store.api.datasets().into();
    }
    if !app.hold_loading && !s.fail {
        // Names in titles resolve from the dataset list; load it up front for stable captures.
        app.store.datasets = app.store.api.datasets().into();
    }
    let screen = s.parts.first().map(String::as_str).unwrap_or("overview");
    if !s.empty && !s.fail {
        // Two minutes of GPU utilisation history (one sample every 2 s).
        app.store.gpu_util = (0..60)
            .map(|i| {
                let i = i as f32;
                (0.93 + 0.025 * (i * 0.7).sin() + 0.012 * (i * 2.3).cos()
                    - if i > 38.0 && i < 41.0 { 0.18 } else { 0.0 })
                .clamp(0.0, 1.0)
            })
            .collect();
    }
    match screen {
        "overview" | "toast" | "kb-focus" | "palette" => app.route = Route::Overview,
        "datasets" => app.route = Route::Datasets,
        "dataset" => {
            app.route = Route::Dataset("ds_3b91".into());
            if s.has("filter") {
                app.ds_split = Some(crate::api::Split::Val);
                app.ds_class = Some(3);
            }
        }
        // Opened from a finished run's sample: its predictions over the dashed labels.
        "inspector" if s.has("sample") => {
            app.route = Route::Run("run_52e8".into());
            app.sample_view = SampleView::Both;
            if let Ok(smp) = app.store.api.samples("run_52e8", None)
                && let Some(x) = smp.samples.get(2).cloned()
            {
                app.store.samples.insert("run_52e8".into(), Load::Ready(smp));
                app.open_sample(x.dataset_id, x.image_id, "run_52e8".into(), window, cx);
            }
        }
        "inspector" => {
            app.route = Route::Dataset("ds_3b91".into());
            app.open_inspector("ds_3b91".into(), 6, window, cx);
            if let Some(i) = app.inspector.as_mut() {
                i.hover_box = Some(1);
            }
        }
        "runs" => app.route = Route::Runs,
        "run-live" | "stop-confirm" => {
            app.route = Route::Run("run_a19f".into());
            if s.has("hover") {
                app.chart_hover = Some(0.36);
            }
            if screen == "stop-confirm" {
                app.ask_stop("run_a19f".into(), window, cx);
            }
        }
        "run-done" => {
            app.route = Route::Run("run_52e8".into());
            app.sample_view = SampleView::Both;
            // The page's end: the log and the run's configuration (precision, head, the resolved
            // optimizer, the capped close-mosaic phase).
            if s.has("bottom") {
                app.scroll_end_pending = 60;
            }
        }
        "run-failed" => app.route = Route::Run("run_e4f0".into()),
        "new-run" => {
            app.route = Route::Runs;
            app.run_filter = RunFilter::All;
        }
        "models" => app.route = Route::Models,
        "model" => app.route = Route::Model("m_44c0".into()),
        "playground" => app.route = Route::Playground,
        "settings" => {
            app.route = Route::Settings;
            // The throughput table up front for a stable capture; `+reference` lists the reference
            // rows too, on the inference side.
            app.throughput = app.store.api.throughput().into();
            if s.has("reference") {
                app.tp_all = true;
                app.tp_view = 1;
            }
        }
        _ => {}
    }
    if screen == "palette" {
        app.open_palette(window, cx);
    }
    if screen == "new-run" {
        // The form needs the dataset list; load it synchronously from the frozen mock.
        app.store.datasets = app.store.api.datasets().into();
        app.open_new_run(Some("ds_3b91"), window, cx);
        if s.has("amp") {
            // Mixed precision on, scrolled to the dialog's end: the switch and the f16 estimate.
            if let Some(f) = app.new_run.as_mut() {
                f.amp = true;
                f.advanced = true;
            }
            app.dialog_scroll.set_offset(gpui::point(gpui::px(0.), gpui::px(-4000.)));
        }
    }
    if screen == "toast" {
        app.toast_with(
            crate::ui::Tone::Success,
            "Exported shapes-hard-s-best · 9.0 MiB",
            Some("…/models/shapes-hard-s-best.f16.folo".into()),
            Some(("Show in folder".into(), std::rc::Rc::new(|_, _, _| {}))),
            cx,
        );
        app.toast_with(
            crate::ui::Tone::Info,
            "Queued shapes-mini-s",
            Some("It starts when the current run finishes.".into()),
            // What the app shows for a queued run (its page opens anyway).
            Some(("View queue".into(), std::rc::Rc::new(|_, _, _| {}))),
            cx,
        );
    }
    if screen == "playground" && !s.has("empty") {
        app.store.models = app.store.api.models().into();
        // shapes-hard-s-best is end-to-end (no NMS row); `+nms` shows a model with the IoU slider.
        app.play_model = Some(if s.has("nms") { "m_31aa" } else { "m_44c0" }.into());
        app.play_conf = 0.35;
        app.play_hover = Some(0);
        app.play_dataset_image("ds_3b91", "img_03417", cx);
    }
    if screen == "kb-focus" {
        app.kb = true;
        app.focus_key("ov-new", window, cx);
    }
    // Keyboard focus on a control with its own keys, its hint showing (scrolled into view).
    let focus = [
        ("charts-focus", "run-charts"),
        ("sample-focus", "smp-2"),
        ("table-focus", "pc-row-1"),
        ("canvas-focus", "stage-canvas"),
    ]
    .into_iter()
    .find(|(part, _)| s.has(part));
    if let Some((_, key)) = focus {
        app.kb = true;
        app.focus_key(key, window, cx);
        // Enough frames for the async lists (samples, the eval) to land and lay out.
        app.reveal_pending = 90;
    }
    let _ = Load::<()>::Idle;
}
