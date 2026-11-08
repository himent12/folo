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
    "datasets",
    "dataset",
    "dataset+light",
    "dataset+filter",
    "inspector",
    "runs",
    "run-live",
    "run-live+light",
    "run-live+hover",
    "run-done",
    "run-failed",
    "new-run",
    "stop-confirm",
    "models",
    "model",
    "model+light",
    "playground",
    "playground+empty",
    "playground+light",
    "settings",
    "toast",
    "kb-focus",
];

impl Spec {
    pub fn parse(name: &str) -> Spec {
        let parts: Vec<String> = name.split('+').map(|s| s.trim().to_string()).collect();
        let has = |p: &str| parts.iter().any(|x| x == p);
        Spec {
            mode: if has("light") { Some(Mode::Light) } else if has("dark") { Some(Mode::Dark) } else { None },
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
    let screen = s.parts.first().map(String::as_str).unwrap_or("overview");
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
        "inspector" => {
            app.route = Route::Dataset("ds_3b91".into());
            app.inspector = Some(crate::app::Inspector { dataset: "ds_3b91".into(), index: 6, zoom: 1.0, pan: (0., 0.), drag_from: None, hover_box: Some(1) });
        }
        "runs" => app.route = Route::Runs,
        "run-live" | "stop-confirm" => {
            app.route = Route::Run("run_a19f".into());
            if s.has("hover") {
                app.chart_hover = Some(0.36);
            }
            if screen == "stop-confirm" {
                app.confirm_stop = Some("run_a19f".into());
            }
        }
        "run-done" => {
            app.route = Route::Run("run_52e8".into());
            app.sample_view = SampleView::Both;
        }
        "run-failed" => app.route = Route::Run("run_e4f0".into()),
        "new-run" => {
            app.route = Route::Runs;
            app.run_filter = RunFilter::All;
        }
        "models" => app.route = Route::Models,
        "model" => app.route = Route::Model("m_44c0".into()),
        "playground" => app.route = Route::Playground,
        "settings" => app.route = Route::Settings,
        _ => {}
    }
    if screen == "palette" {
        app.open_palette(window, cx);
    }
    if screen == "new-run" {
        // The form needs the dataset list; load it synchronously from the frozen mock.
        app.store.datasets = app.store.api.datasets().into();
        app.open_new_run(Some("ds_3b91"), window, cx);
    }
    if screen == "toast" {
        app.toast(crate::ui::Tone::Success, "Export complete", Some("D:/folo/models/shapes-hard-s-best.f16.folo · 9.4 MB".into()), cx);
        app.toast(crate::ui::Tone::Accent, "Training shapes-mini-s", Some("Live metrics stream into the dashboard.".into()), cx);
    }
    if screen == "playground" && !s.has("empty") {
        app.store.models = app.store.api.models().into();
        app.play_model = Some("m_44c0".into());
        app.play_conf = 0.35;
        app.play_hover = Some(0);
        app.play_dataset_image("ds_3b91", "img_03417", cx);
    }
    if screen == "kb-focus" {
        app.kb = true;
        app.focus_key("ov-new", window, cx);
    }
    let _ = Load::<()>::Idle;
}
