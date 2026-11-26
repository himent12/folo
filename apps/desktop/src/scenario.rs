//! Deterministic app states for screenshots: `--scenario <name>`.
//!
//! A scenario freezes the mock clock, picks a theme and a screen, and can open overlays or set
//! hover/focus so states that normally need a pointer are visible. Names compose with `+`:
//! `run-live+light` is the live run dashboard in the light theme.

use folo_ui::{Context, Window};

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
    "overview+lost",
    "palette",
    "palette+light",
    "datasets",
    "synth",
    "synth+invalid",
    "dataset",
    "dataset+light",
    "dataset+filter",
    "dataset+stats",
    "inspector",
    "inspector+light",
    "inspector+sample",
    "runs",
    "run-live",
    "run-live+light",
    "run-live+hover",
    "run-live+charts-focus",
    "run-live+log-focus",
    "run-done",
    "run-done+bottom",
    "run-done+sample-focus",
    "run-failed",
    "run-done+no-history",
    "new-run",
    "new-run+amp",
    "new-run+light",
    "stop-confirm",
    "stop-confirm+light",
    "delete-confirm",
    "delete-confirm+dataset",
    "models",
    "model",
    "model+light",
    "model+table-focus",
    "model+bottom",
    "playground",
    "playground+nms",
    "playground+empty",
    "playground+light",
    "playground+canvas-focus",
    "settings",
    "settings+reference",
    "settings+bottom",
    "settings+bottom+light",
    "benchmarks",
    "benchmarks+light",
    "benchmarks+empty",
    "bench-new",
    "bench",
    "bench+light",
    "bench-plan",
    "bench-design",
    "bench-design+proposal",
    "bench-loops",
    "bench-loop-new",
    "bench-entry",
    "bench-entry+light",
    "bench-entry+bottom",
    "labeling",
    "labeling+light",
    "label-job",
    "label-job+light",
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
    "run-cloud+silent",
    "run-cloud+trouble",
    "toast",
    "kb-focus",
];

/// Modifiers that apply on any screen.
const GLOBAL_MODIFIERS: &[&str] = &["light", "dark", "loading", "error", "empty", "cloud", "lost"];

/// Modifiers that alter the active cloud-runs fixture (`+cloud`, `run-cloud`, `cloud-*`).
const CLOUD_MODIFIERS: &[&str] = &["trouble", "silent"];

/// Modifiers supported for a specific scenario screen.
fn screen_modifiers(screen: &str) -> &'static [&'static str] {
    match screen {
        "synth" => &["invalid"],
        "dataset" => &["filter", "stats"],
        "inspector" => &["sample"],
        "run-live" | "stop-confirm" => &["hover", "charts-focus", "log-focus", "no-history"],
        "run-done" => &["bottom", "sample-focus", "no-history"],
        "run-failed" => &["no-history"],
        "delete-confirm" => &["dataset"],
        "new-run" => &["deleted", "amp"],
        "model" => &["bottom", "table-focus"],
        "playground" => &["nms", "canvas-focus"],
        "settings" => &["reference", "bottom"],
        "bench-design" => &["proposal"],
        "bench-entry" => &["bottom"],
        "cloud-connect" => &["remote", "rejected"],
        "run-cloud" => &["syncing", "training", "released", "no-history"],
        _ => &[],
    }
}

/// All screen-specific modifier names across every scenario (used to distinguish a typo from a
/// modifier that belongs to another screen).
const ALL_SCREEN_MODIFIERS: &[&str] = &[
    "invalid",
    "filter",
    "stats",
    "sample",
    "hover",
    "charts-focus",
    "log-focus",
    "no-history",
    "bottom",
    "sample-focus",
    "dataset",
    "deleted",
    "amp",
    "table-focus",
    "nms",
    "canvas-focus",
    "reference",
    "proposal",
    "remote",
    "rejected",
    "syncing",
    "training",
    "released",
];

/// Validates a `--scenario` string: checks both the base screen name and every `+`-separated
/// modifier token so typos like `overview+ligth` are caught instead of silently ignored.
pub fn validate(name: &str) -> Result<(), String> {
    let parts: Vec<&str> = name.split('+').map(str::trim).collect();
    let screen = parts.first().copied().unwrap_or("");
    if screen.is_empty() || !ALL.iter().any(|s| s.split('+').next() == Some(screen)) {
        return Err(format!("--scenario {name}: no such scenario (see --list-scenarios)"));
    }
    let allowed = screen_modifiers(screen);
    let has_cloud_fixture =
        screen == "run-cloud" || screen.starts_with("cloud-") || parts[1..].contains(&"cloud");
    let mut saw_light = false;
    let mut saw_dark = false;
    for &modifier in &parts[1..] {
        if modifier.is_empty() {
            return Err(format!("--scenario {name}: empty modifier (see --list-scenarios)"));
        }
        if modifier == "light" {
            saw_light = true;
        } else if modifier == "dark" {
            saw_dark = true;
        }
        if GLOBAL_MODIFIERS.contains(&modifier) || allowed.contains(&modifier) {
            continue;
        }
        if CLOUD_MODIFIERS.contains(&modifier) {
            if has_cloud_fixture {
                continue;
            }
            return Err(format!(
                "--scenario {name}: modifier `{modifier}` requires `+cloud` or a cloud scenario (see --list-scenarios)"
            ));
        }
        if ALL_SCREEN_MODIFIERS.contains(&modifier) {
            return Err(format!(
                "--scenario {name}: modifier `{modifier}` does not apply to `{screen}` (see --list-scenarios)"
            ));
        }
        return Err(format!("--scenario {name}: unknown modifier `{modifier}` (see --list-scenarios)"));
    }
    if saw_light && saw_dark {
        return Err(format!("--scenario {name}: conflicting modifiers `light` and `dark`"));
    }
    Ok(())
}

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
        // Generate synthetic; `+invalid` with an image count out of range (its message in place).
        "synth" => {
            app.route = Route::Datasets;
            // The demo can't generate (it says so instead): open the dialog as against folo serve.
            let mode = std::mem::replace(&mut app.server_mode, crate::app::ServerMode::Local);
            app.open_synth(window, cx);
            app.server_mode = mode;
            if s.has("invalid")
                && let Some(f) = app.synth.as_ref()
            {
                f.images.clone().update(cx, |i, cx| i.set_text("200000", cx));
            }
        }
        "dataset" => {
            app.route = Route::Dataset("ds_3b91".into());
            if s.has("filter") {
                app.ds_split = Some(crate::api::Split::Val);
                app.ds_class = Some(("ds_3b91".into(), 3));
            }
            // The Statistics tab: the health line over the KPI strip and the charts.
            if s.has("stats") {
                app.ds_tab = crate::app::DsTab::Statistics;
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
        // The delete confirmation, over a finished run's page (`+dataset`: over a dataset's).
        "delete-confirm" => {
            let target = if s.has("dataset") {
                app.route = Route::Dataset("ds_3b91".into());
                crate::app::DeleteTarget::Dataset("ds_3b91".into())
            } else {
                app.route = Route::Run("run_52e8".into());
                crate::app::DeleteTarget::Run("run_52e8".into())
            };
            app.store.models = app.store.api.models().into();
            app.ask_delete(target, window, cx);
        }
        "new-run" => {
            app.route = Route::Runs;
            app.run_filter = RunFilter::All;
        }
        "models" => app.route = Route::Models,
        // Vision benchmarks (the built-in demo's): the list, a leaderboard, its plan and loops, an entry's result.
        "benchmarks" | "bench-new" => app.route = Route::Benches,
        "bench" => app.route = Route::Bench("bn_3c90".into()),
        "bench-plan" => {
            app.route = Route::Bench("bn_3c90".into());
            app.bench.tab = crate::bench::BenchTab::Plan;
        }
        // A fresh benchmark, whose plan can still change.
        "bench-design" => {
            app.route = Route::Bench("bn_7a04".into());
            app.bench.tab = crate::bench::BenchTab::Plan;
        }
        "bench-loops" | "bench-loop-new" => {
            app.route = Route::Bench("bn_5e12".into());
            app.bench.tab = crate::bench::BenchTab::Loops;
        }
        "bench-entry" => {
            app.route = Route::BenchEntry("bn_3c90".into(), "be_3c01".into());
            if s.has("bottom") {
                app.scroll_end_pending = 60;
            }
        }
        "labeling" => app.route = Route::LabelJobs,
        "label-job" => app.route = Route::LabelJob("lb_4f10".into()),
        "model" => {
            app.route = Route::Model("m_44c0".into());
            // The page's end: the confusion matrix and the Details panel.
            if s.has("bottom") {
                app.scroll_end_pending = 60;
            }
        }
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
            // The page's end: Appearance (the theme rows, custom themes), Keyboard and About.
            if s.has("bottom") {
                app.scroll_end_pending = 60;
            }
        }
        // Settings › Cloud's dialogs, over the section.
        "cloud-connect" | "cloud-disconnect" | "cloud-terminate" => {
            app.route = Route::Settings;
            app.throughput = app.store.api.throughput().into();
        }
        // A cloud run uploading its data, training on its instance, released with its final cost;
        // `+silent` its worker not answering, `+trouble` released with its termination unconfirmed.
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
    if (screen.starts_with("bench") || screen == "benchmarks") && !s.fail && !app.hold_loading {
        preload_bench(app, &app.route.clone());
    }
    if (screen == "labeling" || screen == "label-job") && !s.fail && !app.hold_loading {
        preload_label(app, &app.route.clone());
    }
    match screen {
        "bench-new" => app.open_new_bench(window, cx),
        "bench-loop-new" => app.open_new_loop(cx),
        "bench-design" => {
            app.open_design("bn_7a04", window, cx);
            if s.has("proposal") {
                // The agent's answer, as Ask gets it.
                let req = crate::api::DesignBench {
                    agent: crate::api::LabelAgentId::Claude,
                    model: None,
                    effort: None,
                    note: String::new(),
                };
                let proposal = app.store.api.design_bench("bn_7a04", &req).ok();
                if let Some(f) = app.bench.design.as_mut() {
                    f.proposal = proposal;
                }
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
        if s.has("deleted")
            && let Some(list) = app.store.datasets.ready_mut()
        {
            // Its dataset deleted while the form is open: it asks for another.
            list.retain(|d| d.id != "ds_3b91");
        }
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
            app.dialog_scroll.set_offset(folo_ui::point(folo_ui::px(0.), folo_ui::px(-4000.)));
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
        // Row 5: the log (its level filter and counts) beside the config.
        ("log-focus", "log-scroll"),
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
    // `+no-history`: the run's history couldn't be read (the charts' error panel, Per class's).
    if s.has("no-history")
        && let Route::Run(id) = &app.route
    {
        let down =
            crate::api::ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7878");
        app.store.run_data.insert(id.clone(), Load::Failed(down));
    }
    // `+lost`: the server stopped answering mid-session (the banner over the page, Overview's strip).
    if s.has("lost") {
        let down =
            crate::api::ApiError::new("server_unreachable", "Could not reach folo serve at 127.0.0.1:7878");
        app.store.health = Load::Failed(down.clone());
        app.store.system = Load::Failed(down);
    }
    let _ = Load::<()>::Idle;
}

/// What the Benchmarks screens show, read at once: the system answer (its `vision_bench` feature), the
/// models, the benchmark list, the catalog, the agents and jobs, and the open benchmark's entries,
/// loops, first image and (on an entry page) the entry's result and images.
fn preload_bench(app: &mut FoloApp, route: &Route) {
    let api = app.store.api.clone();
    if let Ok(s) = api.system() {
        app.store.features = s.features;
        app.store.capabilities = s.capabilities;
        app.store.system = Load::Ready(s.system);
    }
    app.store.models = api
        .models()
        .map(|mut m| {
            m.sort_by(|a, b| b.created.cmp(&a.created));
            m
        })
        .into();
    app.store.runs = api.runs().into();
    app.bench.list = api.benches().into();
    app.bench.catalog = api.bench_catalog().into();
    app.bench.agents = api.label_agents().into();
    app.bench.jobs = api.label_jobs().into();
    let (bench, entry) = match route {
        Route::Bench(b) => (b.clone(), None),
        Route::BenchEntry(b, e) => (b.clone(), Some(e.clone())),
        _ => return,
    };
    app.bench.entries.insert(bench.clone(), api.bench_entries(&bench).into());
    app.bench.loops.insert(bench.clone(), api.bench_loops(&bench).into());
    app.bench.previews.insert(bench.clone(), api.bench_images(&bench, 0, 1, None).into());
    if let Some(e) = entry {
        let key = (bench.clone(), e.clone());
        app.bench.results.insert(key.clone(), api.bench_result(&bench, &e).into());
        let sort = app.bench.image_sort;
        let page = api.bench_entry_images(&bench, &e, sort.as_str(), 0, crate::bench::ENTRY_IMAGES_PAGE);
        let images =
            crate::bench::EntryImages { sort, page: page.into(), more: false, more_error: None, seq: 0 };
        app.bench.entry_images.insert(key, images);
    }
}

/// The system answer (its `agent_labeling` feature), agents, jobs, and open job's images/logs.
fn preload_label(app: &mut FoloApp, route: &Route) {
    let api = app.store.api.clone();
    if let Ok(s) = api.system() {
        app.store.features = s.features;
        app.store.capabilities = s.capabilities;
        app.store.system = Load::Ready(s.system);
    }
    app.label.jobs = api.label_jobs().into();
    app.label.agents = api.label_agents().into();
    if let Route::LabelJob(id) = route {
        app.label.images.insert(id.clone(), api.label_images(id, 0, 500).map(|p| p.images).into());
        app.label.logs.insert(id.clone(), api.label_logs(id).map(|r| r.lines).into());
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_all_review_scenarios_and_supported_compositions() {
        for scenario in ALL {
            assert!(validate(scenario).is_ok(), "{scenario}: {:?}", validate(scenario));
        }
        for valid in [
            "overview+dark",
            "new-run+deleted",
            "new-run+deleted+light",
            "run-live+no-history",
            "run-cloud+training+dark",
            "dataset+filter+light",
        ] {
            assert!(validate(valid).is_ok(), "{valid}: {:?}", validate(valid));
        }
    }

    #[test]
    fn rejects_unknown_empty_wrong_screen_and_conflicting_modifiers() {
        assert!(validate("").unwrap_err().contains("no such scenario"));
        assert!(validate("nope-not-real").unwrap_err().contains("no such scenario"));
        assert!(validate("overview+ligth").unwrap_err().contains("unknown modifier `ligth`"));
        assert!(validate("overview+").unwrap_err().contains("empty modifier"));
        assert!(validate("overview++light").unwrap_err().contains("empty modifier"));
        assert!(validate("overview+hover").unwrap_err().contains("does not apply to `overview`"));
        assert!(validate("overview+silent").unwrap_err().contains("requires `+cloud`"));
        assert!(validate("overview+trouble").unwrap_err().contains("requires `+cloud`"));
        assert!(validate("overview+cloud+silent").is_ok());
        assert!(validate("overview+light+dark").unwrap_err().contains("conflicting modifiers"));
    }
}
