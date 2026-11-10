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
    /// The demo with cloud runs under way ([`crate::api::mock::MockBackend::with_cloud_runs`]):
    /// `+cloud` on any screen, Settings › Cloud's dialogs and the cloud run pages. Every other
    /// scenario keeps the default demo, whose one cloud run is long finished.
    pub cloud_runs: bool,
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
    "overview+cloud",
    "runs+cloud",
    "settings+cloud",
    "settings+cloud+light",
    "settings+cloud+empty",
    "settings+cloud+trouble",
    "settings+cloud+trouble+light",
    "cloud-connect",
    "cloud-connect+light",
    "cloud-connect+rejected",
    "cloud-connect+remote",
    "cloud-connect+remote+light",
    "cloud-disconnect",
    "cloud-disconnect+light",
    "cloud-terminate",
    "cloud-terminate+light",
    "new-run+cloud",
    "new-run+cloud+light",
    "run-cloud+syncing",
    "run-cloud+training",
    "run-cloud+training+light",
    "run-cloud+released",
    "toast",
    "kb-focus",
];

impl Spec {
    pub fn parse(name: &str) -> Spec {
        let parts: Vec<String> = name.split('+').map(|s| s.trim().to_string()).collect();
        let has = |p: &str| parts.iter().any(|x| x == p);
        let screen = parts.first().map_or("", String::as_str);
        Spec {
            cloud_runs: has("cloud") || screen == "run-cloud" || screen.starts_with("cloud-"),
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
        // Settings › Cloud's dialogs, over the section.
        "cloud-connect" | "cloud-disconnect" | "cloud-terminate" => {
            app.route = Route::Settings;
            app.throughput = app.store.api.throughput().into();
        }
        // A cloud run uploading its data, training on its instance, released with its final cost.
        "run-cloud" => {
            let id = if s.has("syncing") {
                "run_c1d0"
            } else if s.has("released") {
                "run_c9f1"
            } else {
                "run_c2e1"
            };
            app.route = Route::Run(id.into());
        }
        _ => {}
    }
    // The cloud states read the server's lists up front (the section, the run's provider, the
    // dialog's fields), as they are once loaded.
    let cloud = s.has("cloud") || screen.starts_with("cloud-") || screen == "run-cloud";
    if cloud && !s.fail {
        preload_cloud(app);
    }
    match screen {
        "cloud-connect" => {
            app.open_connect(window, cx);
            if s.has("remote") {
                // The second provider: a FOLO server of yours (a URL and an optional token).
                app.set_connect_provider(1, cx);
            }
            if s.has("rejected") {
                // A key RunPod turns down, sent as Connect sends it: the answer lands on its field.
                if let Some(f) = app.cloud.connect.as_ref() {
                    let key = f.fields[0].clone();
                    key.update(cx, |i, cx| i.set_text("rpa_bad3kQ9xV2", cx));
                }
                let runpod = app.cloud.providers.ready().and_then(|p| p.first()).cloned();
                let req = crate::api::ConnectCloud {
                    provider: "runpod".into(),
                    name: String::new(),
                    values: [("api_key".to_string(), "rpa_bad3kQ9xV2".to_string())].into(),
                };
                if let (Some(p), Err(e)) = (runpod, app.store.api.connect_cloud(&req))
                    && let Some(f) = app.cloud.connect.as_mut()
                {
                    f.error = Some(crate::cloud::connect_error(&p, e));
                }
            }
        }
        "cloud-disconnect" => {
            // Its runs still use it: the server's refusal shows in the dialog.
            app.ask_disconnect("cc_4d2a".into(), window, cx);
            let refused = app.store.api.disconnect_cloud("cc_4d2a").err();
            if let Some(c) = app.cloud.disconnect.as_mut() {
                c.error = refused;
            }
        }
        "cloud-terminate" => {
            let alive =
                app.cloud.instances.ready().and_then(|l| l.iter().find(|i| i.status.is_alive()).cloned());
            if let Some(i) = alive {
                app.ask_terminate(i.id, window, cx);
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
        if s.has("cloud") {
            // RunPod's RTX 4090 (Secure Cloud) chosen with its offers in; Stop after follows the
            // estimate.
            let offers = app.store.api.cloud_offers("cc_4d2a", false);
            app.cloud.offers.insert("cc_4d2a".into(), offers.into());
            if let Some(f) = app.new_run.as_mut() {
                f.run_on = Some("cc_4d2a".into());
                f.offer = Some("NVIDIA GeForce RTX 4090@secure".into());
            }
        }
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

/// The system answer (its `cloud` feature), the runs and the server's cloud lists, read at once.
fn preload_cloud(app: &mut FoloApp) {
    let api = app.store.api.clone();
    if let Ok(s) = api.system() {
        app.store.features = s.features;
        app.store.capabilities = s.capabilities;
        app.store.system = Load::Ready(s.system);
    }
    app.store.runs = api.runs().into();
    app.cloud.providers = api.cloud_providers().into();
    app.cloud.connections = api.cloud_connections().into();
    app.cloud.instances = api.cloud_instances().into();
}
